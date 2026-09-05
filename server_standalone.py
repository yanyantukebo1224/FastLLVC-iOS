import os
import sys
import time
import glob
import json
import asyncio
import threading
import subprocess
import webbrowser
from typing import Optional, Dict, Any, List
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

# Standalone Path Resolver (Handles PyInstaller _MEIPASS)
def get_bundle_dir():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

def get_app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BUNDLE_DIR = get_bundle_dir()
APP_DIR = get_app_dir()
MODELS_DIR = os.path.join(APP_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Copy default models to models/ if empty
default_ckpt = os.path.join(APP_DIR, "llvc_models/models/checkpoints/llvc/G_500000.pth")
if os.path.exists(default_ckpt) and not glob.glob(os.path.join(MODELS_DIR, "*.pth")):
    import shutil
    try:
        shutil.copy(default_ckpt, os.path.join(MODELS_DIR, "G_500000.pth"))
    except Exception:
        pass

from realtime_vc_jp import RealtimeVCEngineJP, get_audio_devices_jp
from infer_adapter import load_model_with_adapter, load_audio, save_audio, infer_stream

app = FastAPI(title="Fast-LLVC Studio Standalone Engine")

# Static assets mount
static_dir = os.path.join(BUNDLE_DIR, "web")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global Engine & State
engine: Optional[RealtimeVCEngineJP] = None
engine_lock = threading.Lock()
connected_websockets: List[WebSocket] = []

live_stats = {
    "is_running": False,
    "in_rms": 0.0,
    "out_rms": 0.0,
    "infer_ms": 0.0,
    "latency_ms": 0.0,
    "rtf": 0.0
}

distill_process = None
distill_logs = []
distill_completed_model = None


def on_engine_latency(infer_ms: float, rtf: float):
    global live_stats, engine
    if engine:
        live_stats["infer_ms"] = infer_ms
        live_stats["rtf"] = rtf
        live_stats["latency_ms"] = (engine.chunk_len / engine.sr * 1000.0) + infer_ms


def on_engine_volume(in_rms: float, out_rms: float):
    global live_stats
    live_stats["in_rms"] = in_rms
    live_stats["out_rms"] = out_rms


def get_available_models() -> List[str]:
    models = []
    # Search in models/
    for p in glob.glob(os.path.join(MODELS_DIR, "*.pth")):
        models.append(os.path.normpath(p))
    # Also check my_adapter/
    for p in glob.glob(os.path.join(APP_DIR, "my_adapter", "*.pth")):
        models.append(os.path.normpath(p))
    # Fallback default
    base_def = os.path.join(APP_DIR, "llvc_models/models/checkpoints/llvc/G_500000.pth")
    if os.path.exists(base_def) and base_def not in models:
        models.append(os.path.normpath(base_def))
    return sorted(list(set(models)))


def get_rvc_model_list() -> List[str]:
    w_dir = os.path.join(APP_DIR, "assets", "weights")
    os.makedirs(w_dir, exist_ok=True)
    return [os.path.basename(p) for p in glob.glob(os.path.join(w_dir, "*.pth"))]


# ----------------- HTTP Routes -----------------
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(BUNDLE_DIR, "web", "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Fast-LLVC WebUI not found.</h1>")


@app.get("/api/init")
async def get_init_data():
    in_devs, out_devs = get_audio_devices_jp()
    models = get_available_models()
    rvc_models = get_rvc_model_list()
    
    if torch.cuda.is_available():
        hw = f"🚀 GPU: {torch.cuda.get_device_name(0)} (BF16最速)"
    else:
        hw = f"⚡ CPU: {os.cpu_count() or 4}スレッド並列処理 (~13ms)"

    return {
        "hardware": hw,
        "in_devices": in_devs,
        "out_devices": out_devs,
        "models": models,
        "rvc_models": rvc_models,
        "force_cpu": not torch.cuda.is_available()
    }


@app.get("/api/models")
async def get_models_api():
    return {"models": get_available_models()}


class VCStartRequest(BaseModel):
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    model_path: str
    chunk_factor: int = 1
    input_gain: float = 1.0
    output_gain: float = 1.0
    threshold_db: float = -45.0
    key_shift: float = 0.0
    force_cpu: bool = False


@app.post("/api/vc/start")
async def start_vc_api(req: VCStartRequest):
    global engine, live_stats
    with engine_lock:
        if engine is not None:
            engine.stop()
            engine = None

        cfg_path = os.path.join(APP_DIR, "experiments/llvc/config.json")
        try:
            engine = RealtimeVCEngineJP(
                checkpoint_path=req.model_path,
                config_path=cfg_path,
                chunk_factor=req.chunk_factor,
                input_device=req.input_device,
                output_device=req.output_device,
                input_gain=req.input_gain,
                output_gain=req.output_gain,
                threshold_db=req.threshold_db,
                key_shift=req.key_shift,
                enable_vocoder=False,
                enable_low_cut=True,
                force_cpu=req.force_cpu,
                on_latency_update=on_engine_latency,
                on_volume_update=on_engine_volume
            )
            engine.start()
            live_stats["is_running"] = True
            return {"status": "ok", "device_name": engine.device_name}
        except Exception as e:
            live_stats["is_running"] = False
            return {"status": "error", "message": str(e)}


@app.post("/api/vc/stop")
async def stop_vc_api():
    global engine, live_stats
    with engine_lock:
        if engine is not None:
            engine.stop()
            engine = None
        live_stats["is_running"] = False
        live_stats["in_rms"] = 0.0
        live_stats["out_rms"] = 0.0
        return {"status": "ok"}


class VCParamsRequest(BaseModel):
    input_gain: Optional[float] = None
    output_gain: Optional[float] = None
    threshold_db: Optional[float] = None
    key_shift: Optional[float] = None


@app.post("/api/vc/params")
async def update_params_api(req: VCParamsRequest):
    global engine
    with engine_lock:
        if engine is not None and engine.is_running:
            engine.update_params(
                in_gain=req.input_gain,
                out_gain=req.output_gain,
                gate_db=req.threshold_db,
                key_shift=req.key_shift,
                enable_vocoder=False
            )
    return {"status": "ok"}


# ----------------- File Convert API -----------------
@app.post("/api/convert/file")
async def convert_file_api(
    file: UploadFile = File(...),
    model_path: str = Form(...),
    key_shift: float = Form(0.0)
):
    temp_in = os.path.join(APP_DIR, "converted_out", f"upload_{file.filename}")
    os.makedirs(os.path.join(APP_DIR, "converted_out"), exist_ok=True)
    with open(temp_in, "wb") as f:
        f.write(await file.read())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    cfg_path = os.path.join(APP_DIR, "experiments/llvc/config.json")

    try:
        model, sr = load_model_with_adapter(model_path, cfg_path, None, merge=False, device=device, dtype=dtype)
        audio = load_audio(temp_in, sr)

        if abs(key_shift) >= 0.1:
            from pitch_shifter import ZeroLatencyPitchShifter
            ps = ZeroLatencyPitchShifter(sample_rate=sr)
            audio = ps.process_torch(audio, key_shift)

        t0 = time.time()
        out, rtf, latency = infer_stream(model, audio, 1, sr, device=device, key_shift=0)
        elapsed = time.time() - t0
        speed_factor = (len(audio) / sr) / max(elapsed, 0.001)

        out_name = f"converted_{os.path.splitext(file.filename)[0]}.wav"
        out_path = os.path.join(APP_DIR, "converted_out", out_name)
        save_audio(out, out_path, sr)

        return {
            "status": "ok",
            "audio_url": f"/converted/{out_name}",
            "message": f"変換完了！ (所要時間: {elapsed*1000:.1f}ms | 速度: {speed_factor:.1f}倍速 | 遅延: {latency:.2f}ms)"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/converted/{filename}")
async def get_converted_file(filename: str):
    p = os.path.join(APP_DIR, "converted_out", filename)
    if os.path.exists(p):
        return FileResponse(p, media_type="audio/wav")
    return JSONResponse(status_code=404, content={"message": "File not found"})


# ----------------- Distill API -----------------
class DistillRequest(BaseModel):
    rvc_model: str
    src_dir: str = "test_wavs"
    out_name: str = "my_custom_voice"


@app.post("/api/distill/start")
async def start_distill_api(req: DistillRequest):
    global distill_process, distill_logs, distill_completed_model
    distill_logs = []
    distill_completed_model = None

    def _worker():
        global distill_process, distill_logs, distill_completed_model
        clean_name = os.path.splitext(req.out_name)[0]
        pair_dir = os.path.join(APP_DIR, "dataset", "train")
        os.makedirs(pair_dir, exist_ok=True)

        distill_logs.append(f"[Step 1] RVC音声変換で学習ペアを生成中...\n")
        cmd1 = ["py", "-3.12", "dataset_builder.py", "-s", req.src_dir, "-m", req.rvc_model, "-o", pair_dir]
        distill_process = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=APP_DIR)
        for line in iter(distill_process.stdout.readline, ''):
            if not line: break
            distill_logs.append(line)
        distill_process.wait()

        if distill_process.returncode != 0:
            distill_logs.append("[エラー] ペア生成が失敗しました。\n")
            distill_process = None
            return

        distill_logs.append(f"[Step 2] ノイズ遮断保護付き LLVC 高速蒸留開始...\n")
        out_save_dir = MODELS_DIR  # Save directly to models/
        cmd2 = ["py", "-3.12", "train_distill.py", "-d", pair_dir, "-o", out_save_dir, "-n", clean_name, "-e", "30", "-b", "8"]
        distill_process = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=APP_DIR)
        for line in iter(distill_process.stdout.readline, ''):
            if not line: break
            distill_logs.append(line)
        distill_process.wait()

        if distill_process.returncode == 0:
            saved_m = os.path.normpath(os.path.join(MODELS_DIR, f"{clean_name}.pth"))
            distill_logs.append(f"\n🎉 全自動蒸留が完了しました！ 保存先: {saved_m}\n")
            distill_completed_model = saved_m
        else:
            distill_logs.append("[エラー] 蒸留処理が失敗しました。\n")
        distill_process = None

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started"}


@app.post("/api/distill/stop")
async def stop_distill_api():
    global distill_process
    if distill_process:
        distill_process.terminate()
        distill_process = None
    return {"status": "stopped"}


@app.get("/api/distill/status")
async def get_distill_status():
    global distill_process, distill_logs, distill_completed_model
    return {
        "is_running": distill_process is not None,
        "logs": "".join(distill_logs[-100:]),
        "completed_model": distill_completed_model
    }


# ----------------- WebSocket Live Monitor -----------------
@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    try:
        while True:
            await websocket.send_json(live_stats)
            await asyncio.sleep(0.06)  # ~16 FPS live update
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)
    except Exception:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)


def main():
    port = 7860
    url = f"http://127.0.0.1:{port}"
    print(f"========================================================")
    print(f" Fast-LLVC Studio (Standalone Engine)")
    print(f" Server URL: {url}")
    print(f" Models Directory: {MODELS_DIR}")
    print(f" Hardware: {'GPU Acceleration' if torch.cuda.is_available() else 'CPU Multi-Threading'}")
    print(f"========================================================")

    # Auto open browser in background
    def _open_browser():
        time.sleep(1.2)
        webbrowser.open(url)

    threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
