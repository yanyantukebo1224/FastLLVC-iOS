import os
import argparse
import torch
import torch.nn as nn
import numpy as np

try:
    import coremltools as ct
except ImportError:
    ct = None

from infer_adapter import load_model_with_adapter
from model import Net


class StreamingLLVCWrapper(nn.Module):
    """
    CoreML-friendly wrapper for LLVC streaming chunk inference.
    Takes (input_chunk, enc_buf, dec_buf, out_buf, convnet_pre_ctx, prev_front_ctx)
    and returns (output_chunk, next_enc_buf, next_dec_buf, next_out_buf, next_convnet_pre_ctx, next_prev_front_ctx)
    """
    def __init__(self, model: Net, chunk_len: int = 208, L: int = 8):
        super(StreamingLLVCWrapper, self).__init__()
        self.model = model
        self.chunk_len = chunk_len
        self.L = L

    def forward(
        self,
        input_chunk: torch.Tensor,       # [1, 1, 208]
        enc_buf: torch.Tensor,           # [1, num_layers, ctx_len, model_dim]
        dec_buf: torch.Tensor,           # [1, num_layers+1, ctx_len, model_dim]
        out_buf: torch.Tensor,           # [1, num_layers, ctx_len, model_dim]
        convnet_pre_ctx: torch.Tensor,   # [1, total_buf_len, channels]
        prev_front_ctx: torch.Tensor    # [1, 1, 16] (L * 2 = 16)
    ):
        # Concatenate lookahead context
        chunk_with_ctx = torch.cat([prev_front_ctx, input_chunk], dim=2)
        next_prev_front_ctx = input_chunk[:, :, -self.L * 2:]

        # LLVC Forward
        output, next_enc_buf, next_dec_buf, next_out_buf, next_convnet_pre_ctx = self.model(
            chunk_with_ctx, enc_buf, dec_buf, out_buf, convnet_pre_ctx,
            pad=(not self.model.lookahead)
        )

        return output, next_enc_buf, next_dec_buf, next_out_buf, next_convnet_pre_ctx, next_prev_front_ctx


def export_to_coreml(
    checkpoint_path: str,
    config_path: str,
    adapter_path: str = None,
    output_path: str = "FastLLVC.mlpackage",
    chunk_len: int = 208
):
    print("=" * 60)
    print("[*] Fast-LLVC -> Core ML (.mlpackage) Export Pipeline")
    print("=" * 60)
    
    # 1. Load Model with Adapter Merged
    print(f"[*] Loading PyTorch model: {checkpoint_path}")
    if adapter_path:
        print(f"[*] Merging adapter: {adapter_path}")
    
    model, sr = load_model_with_adapter(
        checkpoint_path, config_path, adapter_path, merge=True, device="cpu"
    )
    model.eval()

    L = 8
    wrapper = StreamingLLVCWrapper(model, chunk_len=chunk_len, L=L)
    wrapper.eval()

    # 2. Prepare Dummy Tensors for Tracing
    print("[*] Initializing state buffers...")
    enc_buf, dec_buf, out_buf = model.init_buffers(1, torch.device("cpu"))
    if hasattr(model, 'convnet_pre'):
        convnet_pre_ctx = model.convnet_pre.init_ctx_buf(1, torch.device("cpu"))
    else:
        convnet_pre_ctx = torch.zeros(1, 1, 1)
    
    dummy_input = torch.zeros(1, 1, chunk_len, dtype=torch.float32)
    dummy_prev_ctx = torch.zeros(1, 1, L * 2, dtype=torch.float32)

    example_inputs = (
        dummy_input,
        enc_buf,
        dec_buf,
        out_buf,
        convnet_pre_ctx,
        dummy_prev_ctx
    )

    print("[*] Tracing TorchScript model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapper, example_inputs)
        # Test trace execution
        _ = traced_model(*example_inputs)
    
    # Save TorchScript (.pt) for direct mobile use as well
    ts_path = output_path.replace(".mlpackage", ".torchscript.pt")
    traced_model.save(ts_path)
    print(f"[+] Saved TorchScript model: {ts_path}")

    # 3. Convert to Core ML
    if ct is None:
        print("\n[!] 'coremltools' is not installed in the current environment.")
        print("[!] Install it via: pip install coremltools")
        print("[!] The TorchScript model has been exported and can be converted on macOS.")
        return

    print("[*] Converting to Core ML (.mlpackage)...")
    inputs = [
        ct.TensorType(name="input_chunk", shape=dummy_input.shape, dtype=np.float32),
        ct.TensorType(name="enc_buf", shape=enc_buf.shape, dtype=np.float32),
        ct.TensorType(name="dec_buf", shape=dec_buf.shape, dtype=np.float32),
        ct.TensorType(name="out_buf", shape=out_buf.shape, dtype=np.float32),
        ct.TensorType(name="convnet_pre_ctx", shape=convnet_pre_ctx.shape, dtype=np.float32),
        ct.TensorType(name="prev_front_ctx", shape=dummy_prev_ctx.shape, dtype=np.float32)
    ]

    mlmodel = ct.convert(
        traced_model,
        inputs=inputs,
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT32
    )

    mlmodel.author = "Pop-chan & Antigravity"
    mlmodel.short_description = "Fast-LLVC Ultra-Low-Latency Real-Time Voice Conversion Model"
    mlmodel.save(output_path)
    print(f"[+] Successfully exported Core ML model to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export Fast-LLVC to CoreML")
    parser.add_argument('--checkpoint', type=str, default='llvc_models/models/checkpoints/llvc/G_500000.pth')
    parser.add_argument('--config', type=str, default='experiments/llvc/config.json')
    parser.add_argument('--adapter', type=str, default=None)
    parser.add_argument('--out', type=str, default='FastLLVC.mlpackage')
    parser.add_argument('--chunk', type=int, default=208)
    args = parser.parse_args()

    export_to_coreml(args.checkpoint, args.config, args.adapter, args.out, args.chunk)
