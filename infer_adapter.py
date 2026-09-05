import os
import time
import json
import argparse
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm

from model import Net
from adapter import LLVCAdapterManager
from utils import glob_audio_files


DEFAULT_16K_CONFIG = {
    "data": {"sr": 16000, "wav_len": 65536},
    "model_params": {
        "label_len": 1, "L": 16, "enc_dim": 512, "num_enc_layers": 8, "dec_dim": 256,
        "num_dec_layers": 1, "dec_buf_len": 13, "dec_chunk_size": 13, "out_buf_len": 4,
        "use_pos_enc": True, "decoder_dropout": 0.1,
        "convnet_config": {
            "convnet_prenet": True,
            "out_channels": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "kernel_sizes": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "dilations": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

            "dropout": 0.5, "combine_residuals": None, "skip_connection": "add",
            "use_residual_blocks": True
        }
    }
}

def load_model_with_adapter(checkpoint_path, config_path=None, adapter_path=None, merge=False, device="cuda", dtype=torch.bfloat16):
    ckpt = None
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Check if checkpoint embeds config
    config = None
    if ckpt and isinstance(ckpt, dict) and "config" in ckpt and isinstance(ckpt["config"], dict):
        config = ckpt["config"]
    elif config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    elif os.path.exists("experiments/llvc/config.json"):
        with open("experiments/llvc/config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        # Robust built-in fallback configuration
        config = DEFAULT_16K_CONFIG

    sr = config['data']['sr']
    model = Net(**config['model_params'])

    
    # 1. Load base weights
    if ckpt is not None:
        state = ckpt.get('model', ckpt)
        model.load_state_dict(state, strict=False)

    
    # 2. Inject and load adapter with auto rank detection
    if adapter_path and os.path.exists(adapter_path):
        print(f"[Infer] Auto-injecting and loading LoRA adapter from {adapter_path}...")
        LLVCAdapterManager.inject_and_load(model, adapter_path, map_location="cpu")
        if merge:
            print(f"[Infer] Merging LoRA weights for zero overhead...")
            LLVCAdapterManager.merge_all(model)
            
    if device == "cuda" and dtype is not None:
        model.to(device=device, dtype=dtype)
    else:
        model.to(device)
    model.eval()
    return model, sr


def load_audio(audio_path, sample_rate):
    from dataset_adapter import load_any_audio
    return load_any_audio(audio_path, sample_rate)


def save_audio(audio, audio_path, sample_rate):
    if isinstance(audio, torch.Tensor):
        audio = audio.squeeze().detach().cpu().numpy()
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val * 0.99
    sf.write(audio_path, audio, sample_rate)


def infer_stream(model, audio, chunk_factor, sr, device="cuda", vocoder=None, vocoder_strength=0.8, key_shift=0.0):
    dtype = next(model.parameters()).dtype if isinstance(model, torch.nn.Module) else torch.bfloat16
    
    if abs(key_shift) >= 0.1:
        from pitch_shifter import ZeroLatencyPitchShifter
        ps = ZeroLatencyPitchShifter(sample_rate=sr)
        audio = ps.process_torch(audio, key_shift)

    L = model.L
    chunk_len = model.dec_chunk_size * L * chunk_factor
    original_len = len(audio)
    if len(audio) % chunk_len != 0:
        pad_len = chunk_len - (len(audio) % chunk_len)
        audio = torch.nn.functional.pad(audio, (0, pad_len))

    audio = audio.to(device=device, dtype=dtype)
    enc_buf, dec_buf, out_buf = model.init_buffers(1, torch.device(device), dtype=dtype)
    if hasattr(model, 'convnet_pre'):
        convnet_pre_ctx = model.convnet_pre.init_ctx_buf(1, torch.device(device), dtype=dtype)
    else:
        convnet_pre_ctx = None
    prev_front_ctx = torch.zeros(L * 2, dtype=dtype, device=device)

    out_chunks = []
    latencies = []

    with torch.inference_mode():
        for i in range(0, len(audio), chunk_len):
            t0 = time.perf_counter()
            chunk = audio[i:i + chunk_len]
            chunk_with_ctx = torch.cat([prev_front_ctx, chunk]).unsqueeze(0).unsqueeze(0)
            prev_front_ctx = chunk[-L * 2:]

            output, enc_buf, dec_buf, out_buf, convnet_pre_ctx = model(
                chunk_with_ctx, enc_buf, dec_buf, out_buf, convnet_pre_ctx,
                pad=(not model.lookahead)
            )

            # Optional Vocoder Enhancement
            if vocoder is not None:
                output = vocoder(output, strength=vocoder_strength)

            t_chunk = (time.perf_counter() - t0) * 1000.0
            latencies.append(t_chunk)
            out_chunks.append(output.squeeze(0).squeeze(0).float())

    full_output = torch.cat(out_chunks)[:original_len]
    avg_latency = np.mean(latencies) if latencies else 0.0
    rtf = (chunk_len / sr * 1000.0) / max(avg_latency, 0.001)

    return full_output, rtf, avg_latency


def main():
    parser = argparse.ArgumentParser(description="LLVC Adapter Inference (ROCm / CUDA)")
    parser.add_argument('--checkpoint_path', '-p', type=str,
                        default='llvc_models/models/checkpoints/llvc/G_500000.pth', help='Path to base LLVC checkpoint')
    parser.add_argument('--config_path', '-c', type=str,
                        default='experiments/llvc/config.json', help='Path to LLVC config')
    parser.add_argument('--adapter_path', '-a', type=str, default=None, help='Path to LoRA adapter (.pth)')
    parser.add_argument('--merge', action='store_true', help='Merge LoRA weights before inference')
    parser.add_argument('--fname', '-f', type=str, default='test_wavs', help='Input audio or directory')
    parser.add_argument('--out_dir', '-o', type=str, default='my_output', help='Output directory')
    parser.add_argument('--chunk_factor', '-n', type=int, default=1, help='Chunk factor for streaming')
    parser.add_argument('--stream', '-s', action='store_true', help='Use ultra-low latency streaming inference')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[Infer] Running on {args.device} ({torch.cuda.get_device_name(0) if args.device=='cuda' else 'CPU'})")
    model, sr = load_model_with_adapter(
        args.checkpoint_path, args.config_path, args.adapter_path,
        merge=args.merge, device=args.device
    )

    fnames = glob_audio_files(args.fname) if os.path.isdir(args.fname) else [args.fname]
    rtf_list = []
    latency_list = []

    for fname in tqdm(fnames, desc="Converting"):
        audio = load_audio(fname, sr)
        if args.stream:
            out, rtf_, latency_ = infer_stream(model, audio, args.chunk_factor, sr, device=args.device)
            rtf_list.append(rtf_)
            latency_list.append(latency_)
        else:
            with torch.inference_mode():
                out = model(audio.unsqueeze(0).unsqueeze(0).to(args.device, dtype=torch.float32)).squeeze(0)

        out_fname = os.path.join(args.out_dir, os.path.basename(fname))
        save_audio(out, out_fname, sr)

    print(f"\n[Infer] Converted {len(fnames)} files saved to {args.out_dir}/")
    if args.stream and rtf_list:
        print(f"Average RTF (Real-Time Factor): {np.mean(rtf_list):.2f}x (Higher is faster)")
        print(f"End-to-End Latency: {np.mean(latency_list):.2f} ms")


if __name__ == '__main__':
    main()
