import os
import sys
import time
import socket
import asyncio
import threading
import subprocess
import webbrowser
import tempfile
import io
from typing import Optional, Dict, Any, List

# ---------------------------------------------------------
# PyInstaller Bundle & Robust DLL Path Resolution
# ---------------------------------------------------------
if getattr(sys, 'frozen', False):
    BUNDLE_DIR = sys._MEIPASS
    EXE_DIR = os.path.dirname(sys.executable)
else:
    BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))
    EXE_DIR = BUNDLE_DIR

sys.path.insert(0, BUNDLE_DIR)
sys.path.insert(0, os.path.join(BUNDLE_DIR, "FastLLVC_MultiStudio"))

# Register DLL directories to prevent c10.dll / torch_cpu.dll / VC++ runtime dependency errors
torch_lib_dir = os.path.join(BUNDLE_DIR, "torch", "lib")
dll_paths = [BUNDLE_DIR, torch_lib_dir, EXE_DIR]

existing_path = os.environ.get('PATH', '')
os.environ['PATH'] = os.pathsep.join(dll_paths) + os.pathsep + existing_path

for p in dll_paths:
    if os.path.exists(p):
        try:
            os.add_dll_directory(p)
        except Exception:
            pass

# Setup safe logger to fastllvc_studio.log in EXE directory
log_file = os.path.join(EXE_DIR, "fastllvc_studio.log")
class SafeStreamLogger(io.TextIOBase):
    def __init__(self, filename, original_stream):
        self.filename = filename
        self.original_stream = original_stream
    def isatty(self):
        return False
    def write(self, s):
        if self.original_stream:
            try: self.original_stream.write(s)
            except: pass
        try:
            with open(self.filename, 'a', encoding='utf-8', errors='ignore') as f:
                f.write(str(s))
        except: pass
        return len(s) if s else 0
    def flush(self):
        if self.original_stream:
            try: self.original_stream.flush()
            except: pass

sys.stdout = SafeStreamLogger(log_file, sys.stdout)
sys.stderr = SafeStreamLogger(log_file, sys.stderr)

# Check USB/external models directory
MODELS_DIR = os.environ.get('LLVC_MODELS_DIR') or os.path.join(EXE_DIR, 'models')
if not os.path.exists(MODELS_DIR):
    try: os.makedirs(MODELS_DIR, exist_ok=True)
    except: pass

os.environ['LLVC_MODELS_DIR'] = MODELS_DIR
os.environ['PORTABLE_EXECUTABLE_DIR'] = EXE_DIR

# Import application components
import sounddevice as sd
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

# Import FastLLVC multi-engine
from server_multi import create_app

app = create_app()

def find_available_port(start_port: int = 7860) -> int:
    port = start_port
    while port < start_port + 50:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
        port += 1
    return start_port

def run_uvicorn_server(server_instance):
    try:
        server_instance.run()
    except Exception as e:
        print(f"[FastLLVC Studio Server] Server exception: {e}")

def find_browser_app_executable():
    """Find Microsoft Edge or Google Chrome executable for App Mode."""
    candidates = [
        # Edge (Standard Windows 10/11 path)
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
        # Chrome
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        # Brave / Other Chromium
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None

def create_launch_redirect_file(target_url: str) -> str:
    """Create local HTML file that redirects to avoid web search intercept."""
    temp_dir = tempfile.gettempdir()
    launch_html_path = os.path.join(temp_dir, "fastllvc_studio_launch.html")
    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Fast-LLVC Multi-Studio</title>
<meta http-equiv="refresh" content="0; url={target_url}">
<script>
window.location.replace("{target_url}");
</script>
<style>
body {{
  background-color: #0f172a;
  color: #f8fafc;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100vh;
  margin: 0;
}}
.loader {{
  text-align: center;
  padding: 30px;
  background: #1e293b;
  border-radius: 12px;
  border: 1px solid #334155;
  box-shadow: 0 10px 25px rgba(0,0,0,0.5);
}}
a {{ color: #38bdf8; text-decoration: none; font-weight: bold; }}
a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="loader">
  <h2 style="margin-top:0; color:#38bdf8;">Fast-LLVC Multi-Studio</h2>
  <p>スタジオ画面を読み込み中...</p>
  <p style="font-size: 0.9em; opacity: 0.8;">自動的に画面が切り替わらない場合は <a href="{target_url}">こちらをクリック</a></p>
</div>
</body>
</html>"""
    try:
        with open(launch_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return launch_html_path
    except Exception as e:
        print(f"[FastLLVC Studio] Launch HTML error: {e}")
        return ""

def launch_native_app_window(url: str):
    """Launch dedicated standalone app window without URL/search bar."""
    browser_exe = find_browser_app_executable()
    redirect_file = create_launch_redirect_file(url)
    
    if browser_exe:
        # Use direct URL or redirect file
        target = f"file:///{redirect_file.replace(os.sep, '/')}" if redirect_file else url
        args = [
            browser_exe,
            f"--app={target}",
            "--window-size=1280,820"
        ]
        print(f"[FastLLVC Studio] Launching Native App Window using: {os.path.basename(browser_exe)}")
        try:
            proc = subprocess.Popen(args)
            return proc
        except Exception as e:
            print(f"[FastLLVC Studio] Browser app mode launch error: {e}")

    # Fallback to system default browser
    print(f"[FastLLVC Studio] Opening in default web browser...")
    try:
        if redirect_file:
            webbrowser.open(f"file:///{redirect_file.replace(os.sep, '/')}")
        else:
            webbrowser.open(url)
    except Exception as e:
        print(f"[FastLLVC Studio] Browser open error: {e}")
    return "browser_opened"

if __name__ == '__main__':
    port = find_available_port(7860)
    print("================================================================")
    print(f"  Fast-LLVC Studio (Ultra Slim CPU Single EXE - Native Window)")
    print(f"  Models Directory: {MODELS_DIR}")
    print(f"  Starting on: http://127.0.0.1:{port}/")
    print("================================================================")

    # Configure Uvicorn Server instance
    config = uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning')
    server = uvicorn.Server(config)

    # Start server in background daemon thread
    server_thread = threading.Thread(target=run_uvicorn_server, args=(server,), daemon=True)
    server_thread.start()

    # Wait for server to become responsive
    url = f"http://127.0.0.1:{port}/"
    for _ in range(40):
        time.sleep(0.1)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) == 0:
                break

    # Launch native desktop application window
    start_time = time.time()
    app_proc = launch_native_app_window(url)

    # Keep application alive and responsive
    try:
        if isinstance(app_proc, subprocess.Popen):
            print("[FastLLVC Studio] Running with native window process. Waiting for window close...")
            # Wait for user to close window
            app_proc.wait()
            elapsed = time.time() - start_time
            # If closed too quickly (< 2 seconds), Edge likely delegated to an existing instance. Keep server alive!
            if elapsed < 2.0:
                print("[FastLLVC Studio] Window delegated to background browser session. Keeping server alive...")
                while server_thread.is_alive():
                    time.sleep(1)
            else:
                print("[FastLLVC Studio] Native window closed by user.")
        else:
            print("[FastLLVC Studio] Server running. Press Ctrl+C in terminal or close window to exit.")
            while server_thread.is_alive():
                time.sleep(1)
    except KeyboardInterrupt:
        print("[FastLLVC Studio] Termination requested...")

    print("[FastLLVC Studio] Exiting application cleanly...")
    server.should_exit = True
    time.sleep(0.5)
    sys.exit(0)
