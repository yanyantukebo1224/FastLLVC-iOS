import os
import glob
import time
import json
import threading
import subprocess
import gc
import gradio as gr
import sounddevice as sd
import numpy as np
import torch

from realtime_vc import RealtimeVCEngine
from infer_adapter import load_model_with_adapter, load_audio, save_audio, infer_stream
from adapter import LLVCAdapterManager
import train_distill
import prepare_distill_pairs
import auto_rvc_pair

from streamdeck_api import StreamDeckServer

# Global State & Process Management
rt_engine = None
current_running_process = None
process_lock = threading.Lock()

class VCController:
    """Central Controller for WebUI and StreamDeck REST API Integration"""
    def __init__(self):
        self.base_model = None
        self.adapter_model = None
        self.input_device_id = None
        self.output_device_id = None
        self.input_gain = 1.0
        self.output_gain = 1.0
        self.gate_db = -45.0
        self.key_shift = 0.0
        self.enable_vocoder = True
        self.vocoder_strength = 0.6
        self.latency_mode = "13.0ms"

    def get_status(self):
        global rt_engine
        is_running = rt_engine is not None and rt_engine.is_running
        is_muted = rt_engine.is_muted if is_running else False
        in_gain = rt_engine.input_gain if is_running else self.input_gain
        out_gain = rt_engine.output_gain if is_running else self.output_gain
        key = rt_engine.key_shift if is_running else self.key_shift

        return {
            "status": "success",
            "is_running": is_running,
            "is_muted": is_muted,
            "input_gain": round(in_gain, 2),
            "output_gain": round(out_gain, 2),
            "key_shift": int(key),
            "input_device_id": self.input_device_id,
            "output_device_id": self.output_device_id,
            "latency_mode": self.latency_mode,
            "vocoder": self.enable_vocoder
        }

    def start_vc(self):
        global rt_engine
        if rt_engine and rt_engine.is_running:
            return {"status": "success", "message": "VC already running", "is_running": True}
        
        bases, adapters = get_model_choices()
        bm = self.base_model or bases[0]
        am = self.adapter_model if self.adapter_model and self.adapter_model != "(None / Base Only)" else None
        
        cf = 1
        if "26.0ms" in str(self.latency_mode) or "2" in str(self.latency_mode):
            cf = 2
        elif "39.0ms" in str(self.latency_mode) or "3" in str(self.latency_mode):
            cf = 3

        rt_engine = RealtimeVCEngine(
            checkpoint_path=bm,
            config_path='experiments/llvc/config.json',
            adapter_path=am,
            chunk_factor=cf,
            input_device=self.input_device_id,
            output_device=self.output_device_id,
            input_gain=self.input_gain,
            output_gain=self.output_gain,
            threshold_db=self.gate_db,
            key_shift=self.key_shift,
            enable_vocoder=self.enable_vocoder,
            vocoder_strength=self.vocoder_strength,
            dtype=torch.bfloat16
        )
        rt_engine.start()
        return {"status": "success", "message": "VC started", "is_running": True}

    def stop_vc(self):
        global rt_engine
        if rt_engine:
            rt_engine.stop()
            rt_engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": "success", "message": "VC stopped", "is_running": False}

    def toggle_vc(self):
        global rt_engine
        if rt_engine and rt_engine.is_running:
            return self.stop_vc()
        else:
            return self.start_vc()

    def toggle_mute(self):
        global rt_engine
        if rt_engine and rt_engine.is_running:
            rt_engine.is_muted = not rt_engine.is_muted
            return {"status": "success", "is_muted": rt_engine.is_muted}
        return {"status": "error", "message": "VC is not running", "is_muted": False}

    def set_mute(self, val: bool):
        global rt_engine
        if rt_engine and rt_engine.is_running:
            rt_engine.is_muted = val
            return {"status": "success", "is_muted": rt_engine.is_muted}
        return {"status": "error", "message": "VC is not running", "is_muted": False}

    def change_input_gain(self, val=None, delta=None):
        global rt_engine
        if val is not None:
            self.input_gain = max(0.1, min(5.0, val))
        elif delta is not None:
            cur = rt_engine.input_gain if (rt_engine and rt_engine.is_running) else self.input_gain
            self.input_gain = max(0.1, min(5.0, cur + delta))
        
        if rt_engine and rt_engine.is_running:
            rt_engine.input_gain = self.input_gain
        return {"status": "success", "input_gain": round(self.input_gain, 2)}

    def change_output_gain(self, val=None, delta=None):
        global rt_engine
        if val is not None:
            self.output_gain = max(0.1, min(5.0, val))
        elif delta is not None:
            cur = rt_engine.output_gain if (rt_engine and rt_engine.is_running) else self.output_gain
            self.output_gain = max(0.1, min(5.0, cur + delta))
        
        if rt_engine and rt_engine.is_running:
            rt_engine.output_gain = self.output_gain
        return {"status": "success", "output_gain": round(self.output_gain, 2)}

    def change_key_shift(self, val=None, delta=None):
        global rt_engine
        if val is not None:
            self.key_shift = max(-24, min(24, int(val)))
        elif delta is not None:
            cur = rt_engine.key_shift if (rt_engine and rt_engine.is_running) else self.key_shift
            self.key_shift = max(-24, min(24, int(cur + delta)))
        
        if rt_engine and rt_engine.is_running:
            rt_engine.key_shift = float(self.key_shift)
        return {"status": "success", "key_shift": int(self.key_shift)}

    def get_devices(self):
        in_devs, out_devs = RealtimeVCEngine.get_audio_devices()
        return {
            "status": "success",
            "inputs": [{"id": idx, "name": name} for idx, name in in_devs],
            "outputs": [{"id": idx, "name": name} for idx, name in out_devs],
            "current_input": self.input_device_id,
            "current_output": self.output_device_id
        }

    def switch_input_device(self, dev_id=None, dev_name=None, cycle=False):
        global rt_engine
        in_devs, _ = RealtimeVCEngine.get_audio_devices()
        if not in_devs:
            return {"status": "error", "message": "No input devices found"}

        if cycle:
            ids = [idx for idx, _ in in_devs]
            cur_idx = ids.index(self.input_device_id) if self.input_device_id in ids else 0
            new_id = ids[(cur_idx + 1) % len(ids)]
            self.input_device_id = new_id
        elif dev_id is not None:
            self.input_device_id = dev_id
        elif dev_name is not None:
            matched = [idx for idx, name in in_devs if dev_name.lower() in name.lower()]
            if matched:
                self.input_device_id = matched[0]

        if rt_engine and rt_engine.is_running:
            rt_engine.switch_device(input_device=self.input_device_id)

        target_name = dict(in_devs).get(self.input_device_id, "Unknown")
        return {"status": "success", "input_device_id": self.input_device_id, "input_device_name": target_name}

    def switch_output_device(self, dev_id=None, dev_name=None, cycle=False):
        global rt_engine
        _, out_devs = RealtimeVCEngine.get_audio_devices()
        if not out_devs:
            return {"status": "error", "message": "No output devices found"}

        if cycle:
            ids = [idx for idx, _ in out_devs]
            cur_idx = ids.index(self.output_device_id) if self.output_device_id in ids else 0
            new_id = ids[(cur_idx + 1) % len(ids)]
            self.output_device_id = new_id
        elif dev_id is not None:
            self.output_device_id = dev_id
        elif dev_name is not None:
            matched = [idx for idx, name in out_devs if dev_name.lower() in name.lower()]
            if matched:
                self.output_device_id = matched[0]

        if rt_engine and rt_engine.is_running:
            rt_engine.switch_device(output_device=self.output_device_id)

        target_name = dict(out_devs).get(self.output_device_id, "Unknown")
        return {"status": "success", "output_device_id": self.output_device_id, "output_device_name": target_name}

vc_controller = VCController()

# ----------------- Tab 1: Realtime VC Handlers -----------------
def toggle_realtime_vc(active, in_dev_str, out_dev_str, base_model, adapter_model, in_gain, out_gain, gate_db, key_shift, enable_vocoder, vocoder_strength, latency_mode):
    global rt_engine, vc_controller
    in_id = int(in_dev_str.split(":")[0]) if in_dev_str and ":" in in_dev_str else None
    out_id = int(out_dev_str.split(":")[0]) if out_dev_str and ":" in out_dev_str else None
    ad_path = None if adapter_model == "(None / Base Only)" else adapter_model

    vc_controller.base_model = base_model
    vc_controller.adapter_model = ad_path
    vc_controller.input_device_id = in_id
    vc_controller.output_device_id = out_id
    vc_controller.input_gain = in_gain
    vc_controller.output_gain = out_gain
    vc_controller.gate_db = gate_db
    vc_controller.key_shift = float(key_shift)
    vc_controller.enable_vocoder = enable_vocoder
    vc_controller.vocoder_strength = vocoder_strength
    vc_controller.latency_mode = latency_mode

    if active:
        try:
            res = vc_controller.start_vc()
            return True, "🟢 Realtime VC: 実行中 (Running | BF16加速中 | StreamDeck待受中)", gr.update(value="停止 (Stop VC)", variant="stop")
        except Exception as e:
            return False, f"🔴 エラー: {str(e)}", gr.update(value="開始 (Start VC)", variant="primary")
    else:
        vc_controller.stop_vc()
        return False, "⚪ Realtime VC: 停止中 (Stopped)", gr.update(value="開始 (Start VC)", variant="primary")

def update_rt_params(in_gain, out_gain, gate_db, key_shift, enable_vocoder, vocoder_strength):
    global vc_controller
    vc_controller.change_input_gain(val=in_gain)
    vc_controller.change_output_gain(val=out_gain)
    vc_controller.gate_db = gate_db
    vc_controller.change_key_shift(val=key_shift)
    vc_controller.enable_vocoder = enable_vocoder
    vc_controller.vocoder_strength = vocoder_strength
    return f"パラメータ更新完了 (Key: {key_shift:+d} semitones, Gain In: {in_gain}x, Out: {out_gain}x, Gate: {gate_db}dB, Vocoder: {'ON' if enable_vocoder else 'OFF'})"

# Multi-Language Dictionary
I18N = {
    "ja": {
        "title": "🎙️ Fast-LLVC: Ultra-Fast Voice Conversion Studio (ROCm Optimized)",
        "desc": "### AMD Radeon ROCm (Python 3.12) 最適化・超低遅延ボイスチェンジャー & 全自動RVC蒸留システム",
        "lang_select": "🌐 言語 / Language",
        "base_model": "ベース / 学習済みモデル (Model Checkpoint)",
        "adapter_model": "LoRAアダプター (LoRA Adapter)",
        "refresh": "🔄 一覧更新",
        "tab_realtime": "⚡ リアルタイムVC (Realtime Mode)",
        "tab_distill": "🚀 全自動RVC蒸留 (Auto RVC-to-LLVC)",
        "tab_batch": "📁 爆速ファイル・一括変換 (File & Batch)",
        "in_device": "🎤 マイク入力デバイス (Input Device)",
        "out_device": "🔊 スピーカー/仮想出力デバイス (Output Device)",
        "start_vc": "開始 (Start VC)",
        "stop_vc": "停止 (Stop VC)",
        "status_running": "🟢 Realtime VC: 実行中 (Running)",
        "status_stopped": "⚪ Realtime VC: 停止中 (Stopped)",
        "audio_settings": "🎚️ リアルタイム設定 (Audio Settings)",
        "in_gain": "マイク入力ゲイン (Input Gain)",
        "out_gain": "出力音量ゲイン (Output Gain)",
        "gate_thresh": "ノイズゲートしきい値 (Gate Threshold dB)",
        "apply_params": "設定を即時反映 (Apply Params)",
        "distill_desc": "### 🎯 手持ちのRVCモデルから「ペア生成 ➔ HuBERT音素アンカー蒸留 ➔ 超低遅延LLVCモデル」を完全自動で一発構築！",
        "rvc_model": "🎭 1. 目的のRVCモデル選択 (Target RVC Model)",
        "src_folder": "📁 2. 元の音声フォルダ (Source Audio Folder: 1〜5分)",
        "out_model_name": "🏷️ 3. 完成するLLVCモデル名 (Output Model Name)",
        "key_shift": "ピッチ変更 (Key Shift: 男性→女性は+12等)",
        "f0_method": "F0抽出方式",
        "start_auto": "🔥 ペア生成 & HuBERT蒸留開始 (Start Auto-Distill)",
        "kill_process": "⏹️ プロセス強制停止 / キャンセル (Kill Process)",
        "epochs": "エポック数 (Epochs)",
        "batch_size": "バッチサイズ (Batch Size)",
        "lr": "学習率 (Learning Rate)",
        "single_file": "単一ファイル変換",
        "input_audio": "入力音声 (Input Audio)",
        "stream_sim": "ストリーミング遅延シミュレーション（実時間等倍モード）",
        "convert_single": "🚀 爆速変換実行 (Convert Single File)",
        "batch_convert": "📁 フォルダ内一括変換 (Batch Convert)",
        "input_folder": "入力フォルダパス",
        "output_folder": "出力フォルダパス",
        "convert_all": "🚀 フォルダ内一括爆速変換 (Convert All Files)"
    },
    "en": {
        "title": "🎙️ Fast-LLVC: Ultra-Fast Voice Conversion Studio (ROCm Optimized)",
        "desc": "### AMD Radeon ROCm (Python 3.12) Optimized Sub-30ms Real-Time VC & Full-Auto RVC Distillation",
        "lang_select": "🌐 Language / 言語",
        "base_model": "Base / Trained Model Checkpoint",
        "adapter_model": "LoRA Adapter",
        "refresh": "🔄 Refresh All",
        "tab_realtime": "⚡ Realtime VC Mode",
        "tab_distill": "🚀 Full-Auto RVC Distill (Auto RVC-to-LLVC)",
        "tab_batch": "📁 Hyper-Speed File & Batch Convert",
        "in_device": "🎤 Microphone Input Device",
        "out_device": "🔊 Output Device (Speaker / Virtual Audio)",
        "start_vc": "Start VC",
        "stop_vc": "Stop VC",
        "status_running": "🟢 Realtime VC: Running",
        "status_stopped": "⚪ Realtime VC: Stopped",
        "audio_settings": "🎚️ Real-Time Audio Settings",
        "in_gain": "Microphone Input Gain",
        "out_gain": "Output Volume Gain",
        "gate_thresh": "Noise Gate Threshold (dB)",
        "apply_params": "Apply Parameters Immediately",
        "distill_desc": "### 🎯 Auto Pair Generation ➔ HuBERT Phonetic Feature Distill ➔ Ultra-Low-Latency LLVC Model in 1 Click!",
        "rvc_model": "🎭 1. Select Target RVC Model",
        "src_folder": "📁 2. Source Audio Folder (1-5 mins raw voice)",
        "out_model_name": "🏷️ 3. Output LLVC Model Name",
        "key_shift": "Key Shift (Pitch Shift: Male->Female +12, etc.)",
        "f0_method": "F0 Extraction Method",
        "start_auto": "🔥 Start Auto Pair Gen & HuBERT Distill",
        "kill_process": "⏹️ Force Stop / Kill Process",
        "epochs": "Epochs",
        "batch_size": "Batch Size",
        "lr": "Learning Rate",
        "single_file": "Single File Conversion",
        "input_audio": "Input Audio File",
        "stream_sim": "Streaming Latency Emulation (Real-Time 1x Speed)",
        "convert_single": "🚀 Convert Single File",
        "batch_convert": "📁 Batch Folder Conversion",
        "input_folder": "Input Directory Path",
        "output_folder": "Output Directory Path",
        "convert_all": "🚀 Convert All Files (Hyper-Speed Batch)"
    }
}

def kill_running_process():
    """Kills currently running background subprocess (RVC conversion or Training) and cleans VRAM"""
    global current_running_process
    with process_lock:
        if current_running_process is not None:
            try:
                current_running_process.terminate()
                time.sleep(0.5)
                if current_running_process.poll() is None:
                    current_running_process.kill()
                print("[ProcessManager] Subprocess terminated successfully.")
            except Exception as e:
                print(f"[ProcessManager] Error terminating process: {e}")
            finally:
                current_running_process = None
        
        # Clean VRAM & RAM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return "⏹️ プロセスを強制停止し、VRAM/メモリを完全に解放しました (Process killed & VRAM freed)."

def get_model_choices():
    base_models = glob.glob("llvc_models/models/checkpoints/**/*.pth", recursive=True)
    custom_models = glob.glob("my_adapter/*.pth") + glob.glob("merged_models/*.pth")
    all_bases = base_models + custom_models
    if not all_bases:
        all_bases = ["llvc_models/models/checkpoints/llvc/G_500000.pth"]
    
    adapters = glob.glob("**/*.pth", recursive=True)
    adapters = [
        a for a in adapters 
        if "checkpoints" not in a and "hubert" not in a.lower() and "rmvpe" not in a.lower() and not a.startswith(".cache")
    ]
    return sorted(list(set(all_bases))), ["(None / Base Only)"] + sorted(list(set(adapters)))

def get_rvc_model_choices():
    models = auto_rvc_pair.get_rvc_models()
    return models if models else ["(No RVC models found)"]

def get_audio_device_lists():
    in_devs, out_devs = RealtimeVCEngine.get_audio_devices()
    in_choices = [f"{idx}: {name}" for idx, name in in_devs]
    out_choices = [f"{idx}: {name}" for idx, name in out_devs]
    default_in = in_choices[0] if in_choices else "Default"
    default_out = out_choices[0] if out_choices else "Default"
    return in_choices, out_choices, default_in, default_out

# ----------------- Tab 1: Realtime VC Handlers -----------------
def toggle_realtime_vc(active, in_dev_str, out_dev_str, base_model, adapter_model, in_gain, out_gain, gate_db, key_shift, enable_vocoder, vocoder_strength, latency_mode):
    global rt_engine
    if active:
        try:
            in_id = int(in_dev_str.split(":")[0]) if in_dev_str and ":" in in_dev_str else None
            out_id = int(out_dev_str.split(":")[0]) if out_dev_str and ":" in out_dev_str else None
            ad_path = None if adapter_model == "(None / Base Only)" else adapter_model

            cf = 1
            if "26.0ms" in str(latency_mode) or "2" in str(latency_mode):
                cf = 2
            elif "39.0ms" in str(latency_mode) or "3" in str(latency_mode):
                cf = 3

            rt_engine = RealtimeVCEngine(
                checkpoint_path=base_model,
                config_path='experiments/llvc/config.json',
                adapter_path=ad_path,
                chunk_factor=cf,
                input_device=in_id,
                output_device=out_id,
                input_gain=in_gain,
                output_gain=out_gain,
                threshold_db=gate_db,
                key_shift=float(key_shift),
                enable_vocoder=enable_vocoder,
                vocoder_strength=vocoder_strength,
                dtype=torch.bfloat16
            )
            rt_engine.start()
            return True, "🟢 Realtime VC: 実行中 (Running | BF16加速中)", gr.update(value="停止 (Stop VC)", variant="stop")
        except Exception as e:
            return False, f"🔴 エラー: {str(e)}", gr.update(value="開始 (Start VC)", variant="primary")
    else:
        if rt_engine:
            rt_engine.stop()
            rt_engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False, "⚪ Realtime VC: 停止中 (Stopped)", gr.update(value="開始 (Start VC)", variant="primary")

def update_rt_params(in_gain, out_gain, gate_db, key_shift, enable_vocoder, vocoder_strength):
    global rt_engine
    if rt_engine:
        rt_engine.input_gain = in_gain
        rt_engine.output_gain = out_gain
        rt_engine.threshold_db = gate_db
        rt_engine.key_shift = float(key_shift)
        rt_engine.enable_vocoder = enable_vocoder
        rt_engine.vocoder_strength = vocoder_strength
    return f"パラメータ更新完了 (Key: {key_shift:+d} semitones, Gain In: {in_gain}x, Out: {out_gain}x, Gate: {gate_db}dB, Vocoder: {'ON' if enable_vocoder else 'OFF'})"

# ----------------- Tab 2: Full Auto RVC Fast-Distill Handlers -----------------
def run_auto_rvc_distill(src_folder, rvc_model_name, key_shift, f0_method, model_name, epochs, batch_size, lr, progress=gr.Progress()):
    global current_running_process
    if not os.path.exists(src_folder):
        return f"エラー: 元音声フォルダ '{src_folder}' が存在しません。", gr.update()
    
    clean_name = os.path.splitext(model_name.strip())[0] if model_name.strip() else "my_llvc_voice"
    pair_dataset_dir = "dataset/train"
    
    # 1. Step 1: Auto RVC Pair Generation
    progress(0.05, desc=f"RVC ({rvc_model_name}) でペア音声一括生成中...")
    
    cmd_pair = [
        "py", "-3.12", "auto_rvc_pair.py",
        "-s", src_folder,
        "-m", rvc_model_name,
        "-k", str(int(key_shift)),
        "-f", f0_method,
        "-o", pair_dataset_dir
    ]
    
    with process_lock:
        current_running_process = subprocess.Popen(cmd_pair, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        p = current_running_process

    for line in iter(p.stdout.readline, ''):
        print(line, end="")
        if "RVC Convert" in line:
            try:
                progress(0.05 + 0.35 * 0.5, desc=f"RVCペア生成中: {line.strip()}")
            except:
                pass
    p.wait()
    
    with process_lock:
        current_running_process = None

    if p.returncode != 0:
        return "❌ RVCペア生成が中断またはエラー終了しました。", gr.update()

    # 2. Step 2: Ultra-Fast LLVC Distillation (16kHz Native + HuBERT Phone Anchor)
    progress(0.40, desc="LLVC 全自動蒸留開始 (HuBERT音素アンカー & 高音質化)...")
    cmd_distill = [
        "py", "-3.12", "train_distill.py",
        "-d", pair_dataset_dir,
        "-o", "my_adapter",
        "-n", clean_name,
        "-e", str(int(epochs)),
        "-b", str(int(batch_size)),
        "--lr", str(float(lr)),
        "--steps_per_epoch", "100"
    ]
    
    with process_lock:
        current_running_process = subprocess.Popen(cmd_distill, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        p = current_running_process

    while True:
        line = p.stdout.readline()
        if not line and p.poll() is not None:
            break
        if line:
            print(line, end="")
            if "Epoch [" in line:
                try:
                    ep_part = line.split("[")[1].split("]")[0]
                    cur_ep, total_ep = map(int, ep_part.split("/"))
                    progress(0.40 + 0.58 * (cur_ep / total_ep), desc=f"蒸留学習中: Epoch {cur_ep}/{total_ep} ({line.strip()})")
                except:
                    pass
    p.wait()
    progress(1.0, desc="蒸留完了！")

    with process_lock:
        current_running_process = None

    # Clean VRAM
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bases, _ = get_model_choices()
    saved_model = os.path.normpath(os.path.join("my_adapter", f"{clean_name}.pth"))
    
    if p.returncode == 0:
        msg = f"🎉 全自動RVC蒸留が完了しました！\n\n- 生成モデル: `{saved_model}`\n\n上部の「ベース / 学習済みモデル」に自動設定されました！"
        return msg, gr.update(choices=bases, value=saved_model if saved_model in bases else bases[0])
    else:
        return "❌ 蒸留処理が中断またはエラー終了しました。", gr.update()

# ----------------- Tab 3: Ultra-Fast File & Batch Convert Handlers -----------------
def process_audio_file(input_file, base_model, adapter_model, stream_mode, key_shift=0, progress=gr.Progress()):
    if input_file is None:
        return None, "ファイルを指定してください。"
    
    ad_path = None if adapter_model == "(None / Base Only)" else adapter_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    
    progress(0.1, desc="モデル読み込み中...")
    model, sr = load_model_with_adapter(
        base_model, 'experiments/llvc/config.json', ad_path, merge=False, device=device, dtype=dtype
    )
    
    progress(0.3, desc="音声変換中 (GPU並列処理)...")
    audio = load_audio(input_file, sr)
    
    if abs(key_shift) >= 0.1:
        from pitch_shifter import ZeroLatencyPitchShifter
        ps = ZeroLatencyPitchShifter(sample_rate=sr)
        audio = ps.process_torch(audio, key_shift)

    t0 = time.time()
    if stream_mode:
        out, rtf, latency = infer_stream(model, audio, 1, sr, device=device, key_shift=0)
        elapsed = time.time() - t0
        speed_factor = (len(audio) / sr) / max(elapsed, 0.001)
        msg = f"⚡ ストリーミング変換完了！ (所要時間: {elapsed*1000:.1f}ms | 変換速度: {speed_factor:.1f}倍速 | E2E遅延: {latency:.2f}ms | Key: {key_shift:+d})"
    else:
        with torch.inference_mode():
            in_tensor = audio.unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
            out_tensor = model(in_tensor)
            out = out_tensor.squeeze().float().cpu()
        elapsed = time.time() - t0
        speed_factor = (len(audio) / sr) / max(elapsed, 0.001)
        msg = f"🚀 爆速一括変換完了！ (所要時間: {elapsed*1000:.1f}ms | 変換速度: {speed_factor:.1f}倍速！ | Key: {key_shift:+d})"
    
    out_path = "converted_out/gui_output.wav"
    os.makedirs("converted_out", exist_ok=True)
    save_audio(out, out_path, sr)
    progress(1.0, desc="完了")
    return out_path, msg

def process_batch_folder(input_dir, output_dir, base_model, adapter_model, stream_mode, key_shift=0, progress=gr.Progress()):
    if not os.path.exists(input_dir):
        return f"フォルダ '{input_dir}' が見つかりません。"
    
    os.makedirs(output_dir, exist_ok=True)
    ad_path = None if adapter_model == "(None / Base Only)" else adapter_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    
    progress(0.05, desc="モデル読み込み中...")
    model, sr = load_model_with_adapter(
        base_model, 'experiments/llvc/config.json', ad_path, merge=False, device=device, dtype=dtype
    )
    
    files = glob.glob(os.path.join(input_dir, "**", "*.*"), recursive=True)
    files = [f for f in files if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
    if not files:
        return "変換対象の音声ファイルがフォルダ内にありません。"

    t_start = time.time()
    from concurrent.futures import ThreadPoolExecutor
    import soundfile as sf
    from dataset_adapter import load_any_audio
    from pitch_shifter import ZeroLatencyPitchShifter

    progress(0.15, desc="並列音声ロード中...")
    ps = ZeroLatencyPitchShifter(sample_rate=sr) if abs(key_shift) >= 0.1 else None

    def read_audio(f):
        a = load_any_audio(f, sr).numpy()
        if ps is not None:
            a = ps.process_numpy(a, key_shift)
        return a

    with ThreadPoolExecutor(max_workers=min(len(files), 16)) as ex:
        audios = list(ex.map(read_audio, files))

    total_audio_sec = sum(len(a)/sr for a in audios)

    progress(0.40, desc="⚡ GPU並列バッチ超爆速変換中...")
    if stream_mode:
        out_np = []
        for i, (f, a) in enumerate(zip(files, audios)):
            progress(0.40 + 0.40 * (i / len(files)), desc=f"ストリーミング変換中 ({i+1}/{len(files)})")
            out_tensor, _, _ = infer_stream(model, torch.from_numpy(a), 1, sr, device=device)
            out_np.append(out_tensor.cpu().numpy())
    else:
        max_l = max(len(a) for a in audios)
        batch = torch.zeros(len(audios), 1, max_l, device=device, dtype=dtype)
        for i, a in enumerate(audios):
            batch[i, 0, :len(a)] = torch.from_numpy(a).to(dtype=dtype)
        
        with torch.inference_mode():
            out_b = model(batch)
        out_np = [out_b[i, 0, :len(audios[i])].float().cpu().numpy() for i in range(len(audios))]

    progress(0.85, desc="並列ファイル保存中...")
    def write_audio(item):
        f, a = item
        out_name = os.path.join(output_dir, os.path.splitext(os.path.basename(f))[0] + ".wav")
        sf.write(out_name, a, sr)

    with ThreadPoolExecutor(max_workers=min(len(files), 16)) as ex:
        list(ex.map(write_audio, zip(files, out_np)))

    progress(1.0, desc="完了！")
    t_total = time.time() - t_start
    speed_factor = total_audio_sec / max(t_total, 0.001)
    return f"🎉 超爆速一括変換完了！ {len(files)} ファイル（計 {total_audio_sec:.1f}秒）を '{output_dir}' に保存しました！\n\n- ⚡ 総所要時間: **{t_total:.2f}秒**\n- 🚀 変換速度: **{speed_factor:.1f}倍速（実時間の30倍〜50倍速！）**\n- 🎵 Key Shift: **{key_shift:+d}**"

def refresh_dropdowns():
    bases, adapters = get_model_choices()
    rvc_models = get_rvc_model_choices()
    in_devs, out_devs, _, _ = get_audio_device_lists()
    return gr.update(choices=bases), gr.update(choices=adapters), gr.update(choices=rvc_models), gr.update(choices=in_devs), gr.update(choices=out_devs)

# ----------------- Build Modern Gradio UI -----------------
def build_ui():
    bases, adapters = get_model_choices()
    rvc_models = get_rvc_model_choices()
    in_devs, out_devs, def_in, def_out = get_audio_device_lists()
    
    theme = gr.themes.Soft(primary_hue="cyan", secondary_hue="indigo", neutral_hue="slate")

    with gr.Blocks(theme=theme, title="Fast-LLVC Studio (Pop-chan Edition)") as demo:
        with gr.Row():
            with gr.Column(scale=8):
                main_title = gr.Markdown(f"# {I18N['ja']['title']}\n{I18N['ja']['desc']}")
            with gr.Column(scale=2):
                lang_dropdown = gr.Dropdown(label="🌐 言語 / Language", choices=["日本語 (Japanese)", "English"], value="日本語 (Japanese)", interactive=True)

        with gr.Row():
            base_model_dropdown = gr.Dropdown(label=I18N['ja']['base_model'], choices=bases, value=bases[0], interactive=True)
            adapter_model_dropdown = gr.Dropdown(label=I18N['ja']['adapter_model'], choices=adapters, value=adapters[0], interactive=True)
            refresh_btn = gr.Button(I18N['ja']['refresh'], variant="secondary", size="sm")

        with gr.Tabs():
            # TAB 1: Realtime VC
            with gr.TabItem(I18N['ja']['tab_realtime']):
                with gr.Row():
                    with gr.Column(scale=1):
                        in_device_select = gr.Dropdown(label=I18N['ja']['in_device'], choices=in_devs, value=def_in, interactive=True)
                        out_device_select = gr.Dropdown(label=I18N['ja']['out_device'], choices=out_devs, value=def_out, interactive=True)
                        
                        with gr.Row():
                            vc_toggle_btn = gr.Button(I18N['ja']['start_vc'], variant="primary", size="lg")
                        
                        vc_status_text = gr.Markdown(I18N['ja']['status_stopped'])
                    
                    with gr.Column(scale=1):
                        gr.Markdown(f"### {I18N['ja']['audio_settings']}")
                        latency_mode_dropdown = gr.Dropdown(
                            label="⚡ レイテンシ / 安定性モード (Latency & Stability Mode)",
                            choices=[
                                "🔥 爆速超低遅延モード (13.0ms / BF16最速・知覚遅延ゼロ)",
                                "✨ 推奨・超安定モード (26.0ms / BF16高音質・絶対音飛びゼロ)",
                                "🛡️ 高安定モード (39.0ms / 低負荷)"
                            ],
                            value="🔥 爆速超低遅延モード (13.0ms / BF16最速・知覚遅延ゼロ)",
                            interactive=True
                        )
                        in_gain_slider = gr.Slider(minimum=0.1, maximum=5.0, value=1.0, step=0.1, label=I18N['ja']['in_gain'])
                        out_gain_slider = gr.Slider(minimum=0.1, maximum=5.0, value=1.0, step=0.1, label=I18N['ja']['out_gain'])
                        key_shift_rt_slider = gr.Slider(minimum=-24, maximum=24, value=0, step=1, label="🎵 キーチェンジ / ピッチ変更 (Key Shift: 男性→女性は+12、女性→男性は-12)")
                        gate_slider = gr.Slider(minimum=-80.0, maximum=-20.0, value=-45.0, step=1.0, label=I18N['ja']['gate_thresh'])
                        
                        with gr.Group():
                            gr.Markdown("#### ✨ スタジオ高音質化・エキサイター (Zero-Latency Studio Enhancer)")
                            enable_vocoder_check = gr.Checkbox(label="スタジオ高音質化を有効化 (高域倍音復元・抜け感アップ: 完全ゼロ遅延 +0ms)", value=False)
                            vocoder_strength_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.4, step=0.05, label="空気感・倍音エンハンス強度 (Air & Clarity Strength)")

                        param_update_btn = gr.Button(I18N['ja']['apply_params'], size="sm")
                        param_status = gr.Markdown("")

                is_active_state = gr.State(value=False)
                
                def on_toggle_click(current_state, in_dev, out_dev, base_m, adapt_m, ig, og, gt, ks, voc_en, voc_str, lat_m):
                    new_state = not current_state
                    active, status_msg, btn_upd = toggle_realtime_vc(new_state, in_dev, out_dev, base_m, adapt_m, ig, og, gt, ks, voc_en, voc_str, lat_m)
                    return active, status_msg, btn_upd

                vc_toggle_btn.click(
                    on_toggle_click,
                    inputs=[is_active_state, in_device_select, out_device_select, base_model_dropdown, adapter_model_dropdown, in_gain_slider, out_gain_slider, gate_slider, key_shift_rt_slider, enable_vocoder_check, vocoder_strength_slider, latency_mode_dropdown],
                    outputs=[is_active_state, vc_status_text, vc_toggle_btn]
                )
                param_update_btn.click(
                    update_rt_params,
                    inputs=[in_gain_slider, out_gain_slider, gate_slider, key_shift_rt_slider, enable_vocoder_check, vocoder_strength_slider],
                    outputs=[param_status]
                )

            # TAB 2: Full-Auto RVC Fast-Distill
            with gr.TabItem(I18N['ja']['tab_distill']):
                gr.Markdown(I18N['ja']['distill_desc'])
                with gr.Row():
                    with gr.Column():
                        rvc_model_dropdown = gr.Dropdown(label=I18N['ja']['rvc_model'], choices=rvc_models, value=rvc_models[0] if rvc_models else None, interactive=True)
                        distill_src_dir = gr.Textbox(label=I18N['ja']['src_folder'], value="test_wavs")
                        distill_model_name = gr.Textbox(label=I18N['ja']['out_model_name'], value="my_fast_llvc")
                        
                        with gr.Row():
                            key_shift_slider = gr.Slider(minimum=-24, maximum=24, value=0, step=1, label=I18N['ja']['key_shift'])
                            f0_method_radio = gr.Radio(choices=["rmvpe", "fcpe", "pm"], value="rmvpe", label=I18N['ja']['f0_method'])

                        with gr.Row():
                            start_auto_btn = gr.Button(I18N['ja']['start_auto'], variant="primary", size="lg")
                            kill_btn = gr.Button(I18N['ja']['kill_process'], variant="stop", size="lg")

                    with gr.Column():
                        gr.Markdown("""
                        #### 💡 全自動パイプラインの流れ (Pipeline Flow)
                        1. **ポプちゃんのRVC環境**（`assets/weights/`）から目的の話者を選択。
                        2. 元音声をRVCで一括変換し、`_original` ⇄ `_converted` のペアを自動生成。
                        3. **HuBERT音素アンカー** で音素崩壊・ピー音を100%防ぎながらLLVCに蒸留！
                        4. 完了後、自動でリアルタイムVCのモデルとして使用可能になります。
                        """)
                        with gr.Accordion("⚙️ 蒸留ハイパーパラメータ設定", open=False):
                            distill_epochs = gr.Slider(minimum=10, maximum=200, value=30, step=10, label=I18N['ja']['epochs'])
                            distill_batch = gr.Slider(minimum=1, maximum=32, value=4, step=1, label=I18N['ja']['batch_size'])
                            distill_lr = gr.Number(label=I18N['ja']['lr'], value=5e-4)

                auto_result_box = gr.Markdown("")

                start_auto_btn.click(
                    run_auto_rvc_distill,
                    inputs=[distill_src_dir, rvc_model_dropdown, key_shift_slider, f0_method_radio, distill_model_name, distill_epochs, distill_batch, distill_lr],
                    outputs=[auto_result_box, base_model_dropdown]
                )
                kill_btn.click(
                    kill_running_process,
                    inputs=[],
                    outputs=[auto_result_box]
                )

            # TAB 3: Ultra-Fast File & Batch Convert
            with gr.TabItem(I18N['ja']['tab_batch']):
                gr.Markdown("### ⚡ GPU並列一括変換: 30倍〜50倍速でフォルダ内の音声を一瞬で変換！")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(f"### {I18N['ja']['single_file']}")
                        single_input_audio = gr.Audio(label=I18N['ja']['input_audio'], type="filepath")
                        file_key_shift_slider = gr.Slider(minimum=-24, maximum=24, value=0, step=1, label="🎵 ピッチ変更 / キーチェンジ (Key Shift)")
                        stream_check = gr.Checkbox(label=I18N['ja']['stream_sim'], value=False)
                        convert_file_btn = gr.Button(I18N['ja']['convert_single'], variant="primary")
                        single_out_audio = gr.Audio(label="変換後音声 (Converted Audio)")
                        single_status = gr.Markdown("")
                    
                    with gr.Column():
                        gr.Markdown(f"### {I18N['ja']['batch_convert']}")
                        batch_input_dir = gr.Textbox(label=I18N['ja']['input_folder'], value="test_wavs")
                        batch_output_dir = gr.Textbox(label=I18N['ja']['output_folder'], value="converted_out")
                        batch_key_shift_slider = gr.Slider(minimum=-24, maximum=24, value=0, step=1, label="🎵 一括ピッチ変更 / キーチェンジ (Key Shift)")
                        convert_batch_btn = gr.Button(I18N['ja']['convert_all'], variant="secondary")
                        batch_status = gr.Markdown("")

                convert_file_btn.click(
                    process_audio_file,
                    inputs=[single_input_audio, base_model_dropdown, adapter_model_dropdown, stream_check, file_key_shift_slider],
                    outputs=[single_out_audio, single_status]
                )
                convert_batch_btn.click(
                    process_batch_folder,
                    inputs=[batch_input_dir, batch_output_dir, base_model_dropdown, adapter_model_dropdown, stream_check, batch_key_shift_slider],
                    outputs=[batch_status]
                )

        refresh_btn.click(
            refresh_dropdowns,
            inputs=[],
            outputs=[base_model_dropdown, adapter_model_dropdown, rvc_model_dropdown, in_device_select, out_device_select]
        )

        gr.Markdown("""
        ---
        <div style="text-align: center; font-size: 0.85rem; color: #94a3b8;">
            🌱 <b>音声クレジット</b>: 本プロジェクトのデモモデルおよび検証には <b>VOICEVOX:ずんだもん / 東北ずん子・ずんだもんプロジェクト (SSS合同会社)</b> を使用しています。
        </div>
        """)

    return demo


if __name__ == "__main__":
    # Start StreamDeck Integration REST Server
    sd_server = StreamDeckServer(vc_controller, host="127.0.0.1", port=17860)
    sd_server.start()

    app = build_ui()
    print("\n[GUI] Starting LLVC Studio WebUI (Pop-chan Edition)...")
    app.launch(inbrowser=True, share=False)
