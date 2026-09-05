"""
Fast-LLVC: One-Click .pth to iOS CoreML Converter & Wi-Fi AirTransfer Server
Author: Pop-chan & Antigravity
"""

import os
import sys
import json
import socket
import argparse
import http.server
import socketserver
import threading
import torch
import torch.nn as nn
import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    tk = None

try:
    import coremltools as ct
except ImportError:
    ct = None

from infer_adapter import load_model_with_adapter
from model import Net


class StreamingLLVCWrapper(nn.Module):
    def __init__(self, model: Net, chunk_len: int = 208, L: int = 8):
        super(StreamingLLVCWrapper, self).__init__()
        self.model = model
        self.chunk_len = chunk_len
        self.L = L

    def forward(
        self,
        input_chunk: torch.Tensor,
        enc_buf: torch.Tensor,
        dec_buf: torch.Tensor,
        out_buf: torch.Tensor,
        convnet_pre_ctx: torch.Tensor,
        prev_front_ctx: torch.Tensor
    ):
        chunk_with_ctx = torch.cat([prev_front_ctx, input_chunk], dim=2)
        next_prev_front_ctx = input_chunk[:, :, -self.L * 2:]

        output, next_enc_buf, next_dec_buf, next_out_buf, next_convnet_pre_ctx = self.model(
            chunk_with_ctx, enc_buf, dec_buf, out_buf, convnet_pre_ctx,
            pad=(not self.model.lookahead)
        )
        return output, next_enc_buf, next_dec_buf, next_out_buf, next_convnet_pre_ctx, next_prev_front_ctx


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def convert_pth_to_ios(
    checkpoint_path: str,
    config_path: str = None,
    adapter_path: str = None,
    output_dir: str = "ios_models",
    chunk_len: int = 208,
    progress_callback = None
):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    
    if progress_callback:
        progress_callback(10, f"Loading model weights: {base_name}...")
    print(f"[*] Loading model: {checkpoint_path}")

    # Auto find config if not specified
    if not config_path or not os.path.exists(config_path):
        candidate_configs = [
            "experiments/llvc/config.json",
            "experiments/llvc_48k/config.json",
            os.path.join(os.path.dirname(checkpoint_path), "config.json")
        ]
        for c in candidate_configs:
            if os.path.exists(c):
                config_path = c
                break

    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found. Please provide config.json.")

    if progress_callback:
        progress_callback(30, "Merging adapters and preparing neural layers...")

    model, sr = load_model_with_adapter(
        checkpoint_path, config_path, adapter_path, merge=True, device="cpu"
    )
    model.eval()

    L = 8
    wrapper = StreamingLLVCWrapper(model, chunk_len=chunk_len, L=L)
    wrapper.eval()

    if progress_callback:
        progress_callback(50, "Tracing neural graph for Apple Neural Engine...")

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

    with torch.no_grad():
        traced_model = torch.jit.trace(wrapper, example_inputs)
        _ = traced_model(*example_inputs)

    # Save TorchScript (.torchscript.pt)
    ts_path = os.path.join(output_dir, f"{base_name}.torchscript.pt")
    traced_model.save(ts_path)
    print(f"[+] Saved TorchScript model to: {ts_path}")

    # CoreML Export if coremltools available
    mlpackage_path = os.path.join(output_dir, f"{base_name}.mlpackage")
    if ct is not None:
        if progress_callback:
            progress_callback(75, "Compiling Core ML (.mlpackage)...")
        
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
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT32
        )
        mlmodel.author = "Pop-chan & Antigravity"
        mlmodel.short_description = f"Fast-LLVC model: {base_name}"
        mlmodel.save(mlpackage_path)
        print(f"[+] Saved Core ML model to: {mlpackage_path}")
        result_file = mlpackage_path
    else:
        result_file = ts_path

    # Save JSON metadata for easy mobile importing
    meta_path = os.path.join(output_dir, f"{base_name}.json")
    meta_info = {
        "model_name": base_name,
        "sample_rate": sr,
        "chunk_length": chunk_len,
        "latency_ms": (chunk_len / sr) * 1000.0,
        "has_coreml": (ct is not None),
        "files": {
            "torchscript": os.path.basename(ts_path),
            "coreml": os.path.basename(mlpackage_path) if ct is not None else None
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    if progress_callback:
        progress_callback(100, f"Done! Saved to {output_dir}")

    return result_file, meta_path


def start_airtransfer_server(serve_dir: str, port: int = 8080):
    os.chdir(serve_dir)
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)
    local_ip = get_local_ip()
    print(f"\n" + "=" * 60)
    print(f"🚀 Fast-LLVC Wi-Fi AirTransfer Server Running!")
    print(f"📲 On your iPhone / iPad, open Safari or FastLLVC app and access:")
    print(f"👉 http://{local_ip}:{port}")
    print("=" * 60 + "\n")
    httpd.serve_forever()


class ConverterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fast-LLVC ⚡ .pth to iOS CoreML Converter")
        self.root.geometry("620x460")
        self.root.resizable(False, False)

        style = ttk.Style()
        style.theme_use('clam')

        # Header
        header_frame = tk.Frame(root, bg="#1E222D", height=65)
        header_frame.pack(fill=tk.X)
        
        lbl_title = tk.Label(header_frame, text="Fast-LLVC: .pth → iOS CoreML Converter", font=("Arial", 14, "bold"), fg="#FFFFFF", bg="#1E222D")
        lbl_title.pack(pady=8)
        lbl_sub = tk.Label(header_frame, text="Convert PyTorch checkpoints & send wirelessly to iPhone/iPad", font=("Arial", 9), fg="#9AA0A6", bg="#1E222D")
        lbl_sub.pack()

        main_frame = tk.Frame(root, padx=20, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. Model Selection
        tk.Label(main_frame, text="1. Select .pth Model Checkpoint:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(5, 2))
        self.pth_var = tk.StringVar(value="llvc_models/models/checkpoints/llvc/G_500000.pth")
        entry_pth = ttk.Entry(main_frame, textvariable=self.pth_var, width=50)
        entry_pth.grid(row=1, column=0, padx=(0, 10), pady=2)
        btn_browse_pth = ttk.Button(main_frame, text="Browse...", command=self.browse_pth)
        btn_browse_pth.grid(row=1, column=1)

        # 2. Config Selection
        tk.Label(main_frame, text="2. Select config.json (Optional / Auto-detect):", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, pady=(10, 2))
        self.cfg_var = tk.StringVar(value="experiments/llvc/config.json")
        entry_cfg = ttk.Entry(main_frame, textvariable=self.cfg_var, width=50)
        entry_cfg.grid(row=3, column=0, padx=(0, 10), pady=2)
        btn_browse_cfg = ttk.Button(main_frame, text="Browse...", command=self.browse_cfg)
        btn_browse_cfg.grid(row=3, column=1)

        # 3. Optional Adapter Selection
        tk.Label(main_frame, text="3. Target Voice Adapter .pth (Optional):", font=("Arial", 10, "bold")).grid(row=4, column=0, sticky=tk.W, pady=(10, 2))
        self.adapter_var = tk.StringVar(value="")
        entry_adapter = ttk.Entry(main_frame, textvariable=self.adapter_var, width=50)
        entry_adapter.grid(row=5, column=0, padx=(0, 10), pady=2)
        btn_browse_adapter = ttk.Button(main_frame, text="Browse...", command=self.browse_adapter)
        btn_browse_adapter.grid(row=5, column=1)

        # Progress
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=(15, 5))

        self.status_var = tk.StringVar(value="Ready. Select a .pth file and click Convert.")
        lbl_status = tk.Label(main_frame, textvariable=self.status_var, font=("Arial", 9), fg="#333333")
        lbl_status.grid(row=7, column=0, columnspan=2, sticky=tk.W)

        # Action Buttons
        btn_frame = tk.Frame(main_frame, pady=10)
        btn_frame.grid(row=8, column=0, columnspan=2, sticky=tk.EW)

        self.btn_convert = tk.Button(
            btn_frame,
            text="⚡ Convert to iOS CoreML Model",
            font=("Arial", 11, "bold"),
            bg="#007AFF",
            fg="white",
            padx=15,
            pady=8,
            relief=tk.FLAT,
            command=self.run_conversion
        )
        self.btn_convert.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_airtransfer = tk.Button(
            btn_frame,
            text="📡 Start Wi-Fi Transfer to iPhone",
            font=("Arial", 11, "bold"),
            bg="#34C759",
            fg="white",
            padx=15,
            pady=8,
            relief=tk.FLAT,
            command=self.run_airtransfer
        )
        self.btn_airtransfer.pack(side=tk.LEFT)

    def browse_pth(self):
        file = filedialog.askopenfilename(filetypes=[("PyTorch Weights", "*.pth;*.pt")])
        if file:
            self.pth_var.set(file)

    def browse_cfg(self):
        file = filedialog.askopenfilename(filetypes=[("JSON Config", "*.json")])
        if file:
            self.cfg_var.set(file)

    def browse_adapter(self):
        file = filedialog.askopenfilename(filetypes=[("Adapter Weights", "*.pth;*.pt")])
        if file:
            self.adapter_var.set(file)

    def update_progress(self, percent, msg):
        self.progress_var.set(percent)
        self.status_var.set(msg)
        self.root.update_idletasks()

    def run_conversion(self):
        pth = self.pth_var.get()
        if not os.path.exists(pth):
            messagebox.showerror("Error", f"File does not exist: {pth}")
            return

        self.btn_convert.config(state=tk.DISABLED)
        threading.Thread(target=self._convert_thread, daemon=True).start()

    def _convert_thread(self):
        try:
            out_file, meta = convert_pth_to_ios(
                self.pth_var.get(),
                self.cfg_var.get(),
                self.adapter_var.get() or None,
                output_dir="ios_models",
                progress_callback=self.update_progress
            )
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Model successfully converted!\nOutput: {out_file}"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Conversion Failed", str(e)))
        finally:
            self.root.after(0, lambda: self.btn_convert.config(state=tk.NORMAL))

    def run_airtransfer(self):
        os.makedirs("ios_models", exist_ok=True)
        local_ip = get_local_ip()
        url = f"http://{local_ip}:8080"
        threading.Thread(target=lambda: start_airtransfer_server("ios_models", 8080), daemon=True).start()
        messagebox.showinfo(
            "Wi-Fi AirTransfer Server Active",
            f"Server is running!\n\nOpen Safari or FastLLVC app on your iPhone/iPad and visit:\n{url}\n\nYou can download and import models instantly!"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert .pth to iOS CoreML")
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--adapter', type=str, default=None)
    parser.add_argument('--serve', action='store_true', help="Start Wi-Fi AirTransfer Server")
    parser.add_argument('--gui', action='store_true', help="Launch GUI converter")
    args = parser.parse_args()

    if args.serve:
        start_airtransfer_server("ios_models", 8080)
    elif args.checkpoint:
        convert_pth_to_ios(args.checkpoint, args.config, args.adapter, output_dir="ios_models")
    else:
        if tk is not None:
            root = tk.Tk()
            app = ConverterGUI(root)
            root.mainloop()
        else:
            print("[!] Tkinter not available, running CLI mode.")
            convert_pth_to_ios("llvc_models/models/checkpoints/llvc/G_500000.pth", output_dir="ios_models")
