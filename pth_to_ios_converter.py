"""
Fast-LLVC: One-Click .pth to iOS CoreML Converter & Wi-Fi AirTransfer Server
Author: Pop-chan & Antigravity
"""

import os
import re
import sys
import json
import socket
import urllib.parse
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

from infer_adapter import load_model_with_adapter
from model import Net


def sanitize_filename(filename: str) -> str:
    # Replace parentheses, spaces and special chars with underscores
    clean = re.sub(r'[\(\)\s\[\]\{\}\+\,\;\:]', '_', filename)
    clean = re.sub(r'_+', '_', clean).strip('_')
    return clean


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
    raw_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    base_name = sanitize_filename(raw_name)
    
    if progress_callback:
        progress_callback(10, f"Loading weights: {base_name}...")
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

    if progress_callback:
        progress_callback(30, "Merging layers and building network...")

    model, sr = load_model_with_adapter(
        checkpoint_path, config_path, adapter_path, merge=True, device="cpu"
    )
    model.eval()

    L = 8
    wrapper = StreamingLLVCWrapper(model, chunk_len=chunk_len, L=L)
    wrapper.eval()

    if progress_callback:
        progress_callback(50, "Tracing TorchScript graph...")

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

    # Save TorchScript (.torchscript.pt) with URL-safe filename
    ts_path = os.path.join(output_dir, f"{base_name}.torchscript.pt")
    traced_model.save(ts_path)
    print(f"[+] Saved TorchScript model to: {ts_path}")

    # Also save mobile .pt
    mobile_pth_path = os.path.join(output_dir, f"{base_name}.pt")
    torch.save({
        'model': model.state_dict(),
        'sr': sr,
        'chunk_len': chunk_len,
        'config': config_path
    }, mobile_pth_path)

    # CoreML Export Attempt
    mlpackage_path = os.path.join(output_dir, f"{base_name}.mlpackage")
    coreml_success = False
    try:
        import coremltools as ct
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
        coreml_success = True
    except Exception as ex:
        print(f"[*] CoreML conversion skipped on Windows: {ex}")

    # Save metadata JSON
    meta_path = os.path.join(output_dir, f"{base_name}.json")
    meta_info = {
        "model_name": base_name,
        "sample_rate": sr,
        "chunk_length": chunk_len,
        "latency_ms": (chunk_len / sr) * 1000.0,
        "has_coreml": coreml_success,
        "files": {
            "torchscript": os.path.basename(ts_path),
            "model_pt": os.path.basename(mobile_pth_path),
            "coreml": os.path.basename(mlpackage_path) if coreml_success else None
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    # Regenerate Mobile-friendly AirTransfer Web Page
    local_ip = get_local_ip()
    html_path = os.path.join(output_dir, "index.html")
    all_files = [f for f in os.listdir(output_dir) if f.endswith(('.pt', '.pth', '.json', '.mlpackage'))]
    
    file_links_html = ""
    for f in all_files:
        encoded_name = urllib.parse.quote(f)
        size_mb = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024) if os.path.isfile(os.path.join(output_dir, f)) else 0
        file_links_html += f"""
        <div class="file-card">
            <div class="file-info">
                <span class="file-name">{f}</span>
                <span class="file-size">{size_mb:.1f} MB</span>
            </div>
            <a class="btn" href="/download/{encoded_name}" download="{f}">📥 Download to iPhone / iPad</a>
        </div>
        """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Fast-LLVC AirTransfer</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0D1117; color: #E6EDF3; padding: 20px; line-height: 1.5; }}
        .header {{ text-align: center; margin-bottom: 24px; padding-top: 10px; }}
        h1 {{ font-size: 22px; color: #58A6FF; margin-bottom: 6px; }}
        p.sub {{ font-size: 13px; color: #8B949E; }}
        .container {{ max-width: 500px; margin: 0 auto; }}
        .file-card {{ background: #161B22; border: 1px solid #30363D; border-radius: 14px; padding: 18px; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
        .file-info {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
        .file-name {{ font-weight: 700; font-size: 16px; color: #F0F6FC; word-break: break-all; }}
        .file-size {{ font-size: 12px; background: #21262D; padding: 4px 8px; border-radius: 6px; color: #8B949E; }}
        a.btn {{ display: block; background: #238636; color: white; text-align: center; text-decoration: none; padding: 14px; border-radius: 10px; font-size: 15px; font-weight: 600; transition: background 0.2s; }}
        a.btn:active {{ background: #2EA043; transform: scale(0.98); }}
        .tips {{ background: #1F242C; border-left: 4px solid #58A6FF; border-radius: 8px; padding: 14px; margin-top: 24px; font-size: 13px; color: #C9D1D9; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ Fast-LLVC AirTransfer</h1>
            <p class="sub">Tap Download below, then open FastLLVC app to import!</p>
        </div>
        {file_links_html}
        <div class="tips">
            💡 <strong>How to import:</strong><br>
            1. Tap <strong>Download</strong> on the model file above.<br>
            2. When Safari asks, tap <strong>Download</strong> to save it to your Files app.<br>
            3. Open <strong>Fast-LLVC</strong> app &gt; <strong>Models</strong> &gt; <strong>Import from Files</strong>!
        </div>
    </div>
</body>
</html>""")

    if progress_callback:
        progress_callback(100, f"Done! Model: {base_name}")

    return ts_path, meta_path


class AirTransferHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, directory="ios_models", **kwargs):
        self.serve_dir = os.path.abspath(directory)
        super().__init__(*args, directory=self.serve_dir, **kwargs)

    def do_GET(self):
        # Handle custom /download/ URL with forced attachment headers
        decoded_path = urllib.parse.unquote(self.path)
        if decoded_path.startswith("/download/"):
            filename = decoded_path.replace("/download/", "")
            file_path = os.path.join(self.serve_dir, filename)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Content-Length", str(os.path.getsize(file_path)))
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, f"File not found: {filename}")
                return
        
        super().do_GET()


def start_airtransfer_server(serve_dir: str = "ios_models", port: int = 8080):
    os.makedirs(serve_dir, exist_ok=True)
    local_ip = get_local_ip()
    
    # Pre-generate index.html if not present
    index_file = os.path.join(serve_dir, "index.html")
    if not os.path.exists(index_file):
        with open(index_file, "w", encoding="utf-8") as f:
            f.write("<h1>Fast-LLVC AirTransfer Ready</h1><p>Convert a model on PC to view it here.</p>")

    try:
        server = socketserver.ThreadingTCPServer(("0.0.0.0", port), lambda *args: AirTransferHTTPHandler(*args, directory=serve_dir))
        server.allow_reuse_address = True
        print("\n" + "=" * 60)
        print("🚀 Fast-LLVC Wi-Fi AirTransfer Server Running on 0.0.0.0:8080!")
        print("📲 On your iPhone / iPad Safari, visit:")
        print(f"👉 http://{local_ip}:{port}")
        print("=" * 60 + "\n")
        server.serve_forever()
    except Exception as e:
        print(f"[!] Server error: {e}")


class ConverterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fast-LLVC ⚡ .pth to iOS CoreML Converter")
        self.root.geometry("640x480")
        self.root.resizable(False, False)

        style = ttk.Style()
        style.theme_use('clam')

        # Header
        header_frame = tk.Frame(root, bg="#1E222D", height=65)
        header_frame.pack(fill=tk.X)
        
        lbl_title = tk.Label(header_frame, text="Fast-LLVC: .pth → iOS Mobile Converter", font=("Arial", 14, "bold"), fg="#FFFFFF", bg="#1E222D")
        lbl_title.pack(pady=8)
        lbl_sub = tk.Label(header_frame, text="Convert PyTorch checkpoints and transfer wirelessly to iPhone / iPad", font=("Arial", 9), fg="#9AA0A6", bg="#1E222D")
        lbl_sub.pack()

        main_frame = tk.Frame(root, padx=20, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. Model Selection
        tk.Label(main_frame, text="1. Select .pth Model Checkpoint:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(5, 2))
        self.pth_var = tk.StringVar(value="my_adapter/zundamon(sin).pth")
        entry_pth = ttk.Entry(main_frame, textvariable=self.pth_var, width=52)
        entry_pth.grid(row=1, column=0, padx=(0, 10), pady=2)
        btn_browse_pth = ttk.Button(main_frame, text="Browse...", command=self.browse_pth)
        btn_browse_pth.grid(row=1, column=1)

        # 2. Config Selection
        tk.Label(main_frame, text="2. Select config.json (Auto-detected if blank):", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, pady=(10, 2))
        self.cfg_var = tk.StringVar(value="experiments/llvc/config.json")
        entry_cfg = ttk.Entry(main_frame, textvariable=self.cfg_var, width=52)
        entry_cfg.grid(row=3, column=0, padx=(0, 10), pady=2)
        btn_browse_cfg = ttk.Button(main_frame, text="Browse...", command=self.browse_cfg)
        btn_browse_cfg.grid(row=3, column=1)

        # 3. Optional Adapter Selection
        tk.Label(main_frame, text="3. Target Voice Adapter .pth (Optional):", font=("Arial", 10, "bold")).grid(row=4, column=0, sticky=tk.W, pady=(10, 2))
        self.adapter_var = tk.StringVar(value="")
        entry_adapter = ttk.Entry(main_frame, textvariable=self.adapter_var, width=52)
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
            text="⚡ Convert for iOS",
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
        err_msg = None
        out_file = None
        try:
            out_file, meta = convert_pth_to_ios(
                self.pth_var.get(),
                self.cfg_var.get(),
                self.adapter_var.get() or None,
                output_dir="ios_models",
                progress_callback=self.update_progress
            )
        except Exception as e:
            err_msg = str(e)
            print(f"[!] Conversion error: {e}")

        def done_callback(out=out_file, err=err_msg):
            self.btn_convert.config(state=tk.NORMAL)
            if err is not None:
                messagebox.showerror("Conversion Notice", f"Conversion notice:\n{err}")
            else:
                messagebox.showinfo(
                    "Success!",
                    f"Model converted successfully!\n\nOutput: {out}\n\nClick 'Start Wi-Fi Transfer to iPhone' and open Safari on your phone to download!"
                )

        self.root.after(0, done_callback)

    def run_airtransfer(self):
        os.makedirs("ios_models", exist_ok=True)
        local_ip = get_local_ip()
        url = f"http://{local_ip}:8080"
        threading.Thread(target=lambda: start_airtransfer_server("ios_models", 8080), daemon=True).start()
        messagebox.showinfo(
            "Wi-Fi AirTransfer Server Active",
            f"Server is running on your Wi-Fi!\n\nOpen Safari on your iPhone / iPad and access:\n\n👉 {url}\n\nTap Download next to the model to download it into Files app!"
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
            convert_pth_to_ios("my_adapter/zundamon(sin).pth", output_dir="ios_models")
