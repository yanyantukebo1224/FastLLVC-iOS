import os
import sys
import glob
import time
import subprocess
import argparse
import soundfile as sf
import numpy as np
import shutil

RVC_ROOT = r"D:\music\RVC20260718Nvidia50x0\RVC20260718Nvidia50x0"

def get_rvc_models():
    weights_dir = os.path.join(RVC_ROOT, "assets", "weights")
    if not os.path.exists(weights_dir):
        return []
    models = glob.glob(os.path.join(weights_dir, "*.pth"))
    return sorted([os.path.basename(m) for m in models])

def generate_pairs_with_rvc(src_dir: str, rvc_model: str, key_shift: int = 0, f0_method: str = "rmvpe", out_dir: str = "dataset/train"):
    """
    Invokes RVC to convert audios from src_dir, and formats them into paired dataset in out_dir.
    Uses absolute paths for 100% reliable execution across working directories.
    """
    abs_out_dir = os.path.abspath(out_dir)
    os.makedirs(abs_out_dir, exist_ok=True)
    temp_conv_dir = os.path.abspath(os.path.join(out_dir, "temp_rvc_out"))
    os.makedirs(temp_conv_dir, exist_ok=True)

    src_files = glob.glob(os.path.join(src_dir, "**", "*.*"), recursive=True)
    src_files = [os.path.abspath(f) for f in src_files if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
    
    if not src_files:
        raise ValueError(f"No audio files found in '{src_dir}'")

    print(f"[AutoRVC] Found {len(src_files)} audio files. Starting RVC batch conversion with model: {rvc_model}...")

    # We create a lightweight batch infer script for RVC
    batch_script = os.path.join(RVC_ROOT, "_auto_batch_rvc.py")
    script_code = f"""
import os
import sys
import soundfile as sf
import torch

sys.path.insert(0, r"{RVC_ROOT}")
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:True"
os.environ.setdefault("weight_root", r"{os.path.join(RVC_ROOT, 'assets', 'weights')}")
os.environ.setdefault("rmvpe_root", r"{os.path.join(RVC_ROOT, 'assets', 'rmvpe')}")

from configs.config import Config
from infer.vc.modules import VC
from cmd_infer import process_smart_chunks

config = Config()
vc = VC(config)
vc.get_vc(r"{rvc_model}")

src_files = {repr(src_files)}
out_dir = r"{temp_conv_dir}"

for idx, f in enumerate(src_files):
    print(f"[RVC Convert {{idx+1}}/{{len(src_files)}}] {{os.path.basename(f)}}")
    try:
        out = process_smart_chunks(vc, f, {key_shift}, "{f0_method}")
        if out is not None:
            base_name = os.path.splitext(os.path.basename(f))[0]
            out_path = os.path.join(out_dir, f"{{base_name}}_converted.wav")
            if isinstance(out, tuple):
                sr, audio_data = out
                sf.write(out_path, audio_data, sr)
            elif isinstance(out, str) and os.path.exists(out):
                c_audio, out_sr = sf.read(out)
                sf.write(out_path, c_audio, out_sr)
    except Exception as e:
        print(f"Error on {{f}}: {{e}}")
"""
    with open(batch_script, "w", encoding="utf-8") as f:
        f.write(script_code)

    try:
        p = subprocess.Popen(["py", "-3.12", "_auto_batch_rvc.py"], cwd=RVC_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in iter(p.stdout.readline, ''):
            print(line, end="")
        p.wait()
    finally:
        if os.path.exists(batch_script):
            os.remove(batch_script)

    # Now organize into pair dataset
    from dataset_adapter import load_any_audio
    pair_count = 0
    for s_path in src_files:
        base_name = os.path.splitext(os.path.basename(s_path))[0]
        c_path = os.path.join(temp_conv_dir, f"{base_name}_converted.wav")
        if os.path.exists(c_path):
            dst_orig = os.path.join(abs_out_dir, f"{pair_count:04d}_{base_name}_original.wav")
            dst_conv = os.path.join(abs_out_dir, f"{pair_count:04d}_{base_name}_converted.wav")
            
            orig_wav = load_any_audio(s_path, 16000).numpy()
            conv_wav = load_any_audio(c_path, 16000).numpy()
            
            sf.write(dst_orig, orig_wav, 16000)
            sf.write(dst_conv, conv_wav, 16000)
            pair_count += 1

    shutil.rmtree(temp_conv_dir, ignore_errors=True)
    print(f"\n[AutoRVC] Successfully built {pair_count} aligned training pairs in '{out_dir}'!")
    return pair_count


def main():
    parser = argparse.ArgumentParser(description="Auto RVC Pair Generator")
    parser.add_argument("--src_dir", "-s", type=str, default="test_wavs", help="Source audios")
    parser.add_argument("--rvc_model", "-m", type=str, required=True, help="RVC model name")
    parser.add_argument("--key_shift", "-k", type=int, default=0, help="Key pitch shift")
    parser.add_argument("--f0_method", "-f", type=str, default="rmvpe", help="F0 method")
    parser.add_argument("--out_dir", "-o", type=str, default="dataset/train", help="Output pair dataset folder")
    args = parser.parse_args()

    generate_pairs_with_rvc(args.src_dir, args.rvc_model, args.key_shift, args.f0_method, args.out_dir)


if __name__ == "__main__":
    main()
