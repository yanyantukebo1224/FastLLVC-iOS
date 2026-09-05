import os
import sys
import time
import math
import queue
import threading
from typing import Optional, Callable, Tuple, List
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import torch.nn.functional as F

# CPU Multi-threading Optimization
if not torch.cuda.is_available():
    cpu_cores = os.cpu_count() or 4
    try: torch.set_num_threads(max(1, min(4, cpu_cores - 1)))
    except: pass
    try: torch.set_num_interop_threads(1)
    except: pass


from model import Net
from infer_adapter import load_model_with_adapter
from pitch_shifter import ZeroLatencyPitchShifter
from audio_enhancer import ZeroLatencyStudioEnhancer


class CleanVocalHighPassFilter:
    """Zero-latency high pass filter (cuts <100Hz chest resonance)"""
    def __init__(self, sample_rate: int = 16000, cutoff_hz: float = 100.0):
        self.sr = sample_rate
        self.cutoff = cutoff_hz
        self.prev_in = 0.0
        self.prev_out = 0.0
        self._calc_alpha()

    def _calc_alpha(self):
        rc = 1.0 / (2.0 * math.pi * self.cutoff)
        dt = 1.0 / self.sr
        self.alpha = rc / (rc + dt)

    def reset(self):
        self.prev_in = 0.0
        self.prev_out = 0.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        out = np.zeros_like(audio)
        for i in range(len(audio)):
            out[i] = self.alpha * (self.prev_out + audio[i] - self.prev_in)
            self.prev_in = audio[i]
            self.prev_out = out[i]
        return out


class RealtimeVCEngineJP:
    """
    Fast-LLVC 完全日本語・CPU/GPU自動対応 リアルタイム音声変換エンジン
    - GPU搭載PC: NVIDIA CUDA / AMD ROCm BF16 で爆速推論 (~3.5ms)
    - CPU搭載PC (ノートPCなど): PyTorch マルチスレッド最適化で低遅延動作 (~13ms)
    - サンプル精度Lookahead先読み + タイピング音・環境雑音カットフィルター内蔵
    """
    def __init__(
        self,
        checkpoint_path: str = "llvc_models/models/checkpoints/llvc/G_500000.pth",
        config_path: str = "experiments/llvc/config.json",
        adapter_path: Optional[str] = None,
        chunk_factor: int = 1,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        input_gain: float = 1.0,
        output_gain: float = 1.0,
        threshold_db: float = -45.0,
        key_shift: float = 0.0,
        enable_vocoder: bool = False,
        vocoder_strength: float = 0.4,
        enable_low_cut: bool = True,
        force_cpu: bool = False,
        on_latency_update: Optional[Callable[[float, float], None]] = None,
        on_volume_update: Optional[Callable[[float, float], None]] = None
    ):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.adapter_path = adapter_path
        self.chunk_factor = max(1, chunk_factor)
        self.input_device = input_device
        self.output_device = output_device
        self.input_gain = input_gain
        self.output_gain = output_gain
        self.threshold_db = threshold_db
        self.key_shift = float(key_shift)
        self.is_muted = False
        self.enable_vocoder = enable_vocoder
        self.vocoder_strength = vocoder_strength
        self.enable_low_cut = enable_low_cut
        self.on_latency_update = on_latency_update
        self.on_volume_update = on_volume_update

        # Device Auto-detection (GPU vs CPU)
        if force_cpu or not torch.cuda.is_available():
            self.device = "cpu"
            self.dtype = torch.float32
            # Set optimal CPU thread count (2 threads prevents thread contention overhead)
            try: torch.set_num_threads(2)
            except: pass
            try: torch.set_num_interop_threads(1)
            except: pass

            # For CPU, ensure chunk_factor >= 2 (26.0ms) to guarantee plenty of inference headroom
            if chunk_factor == 1:
                chunk_factor = 2
            self.chunk_factor = chunk_factor
            self.device_name = f"Intel/AMD CPU (超最適化 2スレッド並列 / 26ms安定モード)"
        else:
            self.device = "cuda"
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
            gpu_name = torch.cuda.get_device_name(0)
            self.device_name = f"GPU ({gpu_name}) ハードウェアアクセラレーション"

        self.model = None
        self.sr = 16000
        self.L = 16
        self.dec_chunk_size = 13
        self.chunk_len = self.dec_chunk_size * self.L * self.chunk_factor  # 416 samples = 26.0ms on CPU
        self.ctx_len = self.L * 2  # 32 samples
        self.total_chunk_len = self.ctx_len + self.chunk_len

        self.enhancer = None
        self.pitch_shifter = None
        self.low_cut_filter = None
        
        self.is_running = False
        self.stream = None
        self.worker_thread = None
        
        # Audio queues with pre-buffer capacity
        self.in_queue = queue.Queue(maxsize=16)
        self.out_queue = queue.Queue(maxsize=16)
        
        # Filter States
        self.dc_prev_in = 0.0
        self.dc_prev_out = 0.0
        self.current_gate_gain = 1.0
        self.last_out_sample = 0.0
        
        # Buffers
        self.enc_buf = None
        self.dec_buf = None
        self.out_buf = None
        self.convnet_pre_ctx = None
        self.prev_front_ctx = None
        self.prev_lookahead_carry = None
        self.chunk_with_ctx_buf = None

    def _auto_detect_config(self, cp: str) -> str:
        if self.config_path and os.path.exists(self.config_path):
            return self.config_path
        return "experiments/llvc/config.json"

    def load_model(self, checkpoint_path=None, adapter_path=None):
        cp = checkpoint_path or self.checkpoint_path
        ap = adapter_path if adapter_path is not None else self.adapter_path
        cfg_path = self._auto_detect_config(cp)
        self.config_path = cfg_path

        self.model, self.sr = load_model_with_adapter(
            cp, cfg_path, ap, merge=False, device=self.device, dtype=self.dtype
        )
        self.model.to(device=self.device, dtype=self.dtype).eval()
        
        self.L = getattr(self.model, 'L', 16)
        self.dec_chunk_size = getattr(self.model, 'dec_chunk_size', 13)
        self.chunk_len = self.dec_chunk_size * self.L * self.chunk_factor
        self.ctx_len = self.L * 2
        self.total_chunk_len = self.ctx_len + self.chunk_len

        self.enhancer = ZeroLatencyStudioEnhancer(sample_rate=self.sr)
        self.pitch_shifter = ZeroLatencyPitchShifter(sample_rate=self.sr)
        self.low_cut_filter = CleanVocalHighPassFilter(sample_rate=self.sr, cutoff_hz=100.0)
        print(f"[EngineJP] モデル読み込み完了: @{self.sr}Hz (L={self.L}, 1フレーム={self.chunk_len/self.sr*1000:.1f}ms | 稼働環境: {self.device_name})")

    def _init_gpu_buffers(self):
        dev = torch.device(self.device)
        self.enc_buf, self.dec_buf, self.out_buf = self.model.init_buffers(1, dev)
        self.enc_buf = self.enc_buf.to(dtype=self.dtype)
        self.dec_buf = self.dec_buf.to(dtype=self.dtype)
        self.out_buf = self.out_buf.to(dtype=self.dtype)
        
        if hasattr(self.model, 'convnet_pre'):
            self.convnet_pre_ctx = self.model.convnet_pre.init_ctx_buf(1, dev).to(dtype=self.dtype)
        else:
            self.convnet_pre_ctx = None
            
        self.prev_front_ctx = torch.zeros(self.ctx_len, dtype=self.dtype, device=self.device)
        self.prev_lookahead_carry = torch.zeros(self.L, dtype=self.dtype, device=self.device)
        self.chunk_with_ctx_buf = torch.zeros(1, 1, self.total_chunk_len, dtype=self.dtype, device=self.device)

    def reset_buffers(self):
        if self.model is not None:
            self._init_gpu_buffers()
        if self.pitch_shifter is not None:
            self.pitch_shifter.reset()
        if self.low_cut_filter is not None:
            self.low_cut_filter.reset()
        if self.enhancer is not None:
            self.enhancer.reset()
        self.dc_prev_in = 0.0
        self.dc_prev_out = 0.0
        self.current_gate_gain = 1.0
        self.last_out_sample = 0.0
        
        while not self.in_queue.empty():
            try: self.in_queue.get_nowait()
            except: break
        while not self.out_queue.empty():
            try: self.out_queue.get_nowait()
            except: break

    def _dc_blocker(self, audio: np.ndarray, r: float = 0.995) -> np.ndarray:
        out = np.zeros_like(audio)
        for i in range(len(audio)):
            out[i] = audio[i] - self.dc_prev_in + r * self.dc_prev_out
            self.dc_prev_in = audio[i]
            self.dc_prev_out = out[i]
        return out

    def warmup(self):
        if self.model is None:
            self.load_model()
        self._init_gpu_buffers()
        
        with torch.inference_mode():
            dummy_in = torch.zeros(1, 1, self.total_chunk_len, dtype=self.dtype, device=self.device)
            for _ in range(5):
                out, self.enc_buf, self.dec_buf, self.out_buf, self.convnet_pre_ctx = self.model(
                    dummy_in, self.enc_buf, self.dec_buf, self.out_buf, self.convnet_pre_ctx,
                    pad=False
                )
            if self.device == "cuda":
                torch.cuda.synchronize()
        self._init_gpu_buffers()

    def _inference_worker(self):
        while self.is_running:
            try:
                in_audio, in_rms = self.in_queue.get(timeout=0.03)
            except queue.Empty:
                continue

            in_db = 20 * np.log10(in_rms + 1e-9)
            target_gain = 1.0 if in_db >= self.threshold_db else 0.0

            # 1. Clean Vocal Low-Cut (Cuts chest resonance)
            if self.enable_low_cut and self.low_cut_filter is not None:
                in_proc = self.low_cut_filter.process(in_audio)
            else:
                in_proc = in_audio

            # 2. Key Shift
            if abs(self.key_shift) >= 0.1 and self.pitch_shifter is not None:
                in_proc = self.pitch_shifter.process_numpy(in_proc, self.key_shift)

            with torch.inference_mode():
                t0 = time.perf_counter()
                
                in_tensor = torch.from_numpy(in_proc).to(device=self.device, dtype=self.dtype)
                
                # Sample-Accurate Lookahead Alignment (Zero Comb Filter Artifacts)
                curr_stream_chunk = torch.cat([self.prev_lookahead_carry, in_tensor[:-self.L]])
                self.prev_lookahead_carry.copy_(in_tensor[-self.L:])

                self.chunk_with_ctx_buf[0, 0, :self.ctx_len] = self.prev_front_ctx
                self.chunk_with_ctx_buf[0, 0, self.ctx_len:] = curr_stream_chunk
                self.prev_front_ctx.copy_(curr_stream_chunk[-self.ctx_len:])

                # Neural Forward Pass (BF16 on GPU / Parallel FP32 on CPU)
                output, self.enc_buf, self.dec_buf, self.out_buf, self.convnet_pre_ctx = self.model(
                    self.chunk_with_ctx_buf, self.enc_buf, self.dec_buf, self.out_buf, self.convnet_pre_ctx,
                    pad=False
                )

                t_infer = (time.perf_counter() - t0) * 1000.0
                rtf = (self.chunk_len / self.sr * 1000.0) / max(t_infer, 0.001)

                if self.on_latency_update:
                    self.on_latency_update(t_infer, rtf)

                out_audio = output.squeeze().float().cpu().numpy()

                # Smooth Noise Gate Linear Ramp
                gain_start = self.current_gate_gain
                self.current_gate_gain = 0.80 * self.current_gate_gain + 0.20 * target_gain
                gain_end = self.current_gate_gain
                
                gain_ramp = np.linspace(gain_start, gain_end, len(out_audio), dtype=np.float32)
                out_audio = out_audio * (self.output_gain * gain_ramp)

                # Studio Audio Enhancer (Optional)
                if self.enable_vocoder and self.enhancer is not None and self.current_gate_gain > 0.02:
                    out_audio = self.enhancer.process_numpy(
                        out_audio,
                        clarity=0.3,
                        air=float(self.vocoder_strength),
                        warmth=0.2
                    )

                # Soft Limiter
                out_audio = np.nan_to_num(out_audio, nan=0.0, posinf=0.95, neginf=-0.95)
                out_audio = np.tanh(out_audio * 0.95)
                out_rms = float(np.sqrt(np.mean(out_audio**2) + 1e-9))

            if self.on_volume_update:
                self.on_volume_update(in_rms, out_rms)

            if self.out_queue.full():
                try: self.out_queue.get_nowait()
                except: pass
            self.out_queue.put(out_audio)

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        in_audio = np.mean(indata, axis=1) if indata.ndim > 1 else indata[:, 0]
        
        if self.is_muted:
            in_audio = np.zeros_like(in_audio)
            in_rms = 0.0
        else:
            in_audio = in_audio * self.input_gain
            in_audio = self._dc_blocker(in_audio)
            in_rms = float(np.sqrt(np.mean(in_audio**2) + 1e-9))

        if not self.in_queue.full():
            self.in_queue.put((in_audio, in_rms))

        try:
            out_audio = self.out_queue.get_nowait()
            self.last_out_sample = out_audio[-1]
        except queue.Empty:
            if abs(self.last_out_sample) > 1e-4:
                out_audio = np.linspace(self.last_out_sample, 0.0, frames, dtype=np.float32)
                self.last_out_sample = 0.0
            else:
                out_audio = np.zeros(frames, dtype=np.float32)

        if len(out_audio) < frames:
            out_audio = np.pad(out_audio, (0, frames - len(out_audio)), mode='constant')
        elif len(out_audio) > frames:
            out_audio = out_audio[:frames]

        if outdata.shape[1] == 1:
            outdata[:] = out_audio[:, np.newaxis]
        else:
            outdata[:, 0] = out_audio
            outdata[:, 1] = out_audio

    def start(self):
        if self.is_running:
            return
        
        if self.model is None:
            self.load_model()
            
        self.warmup()
        self.reset_buffers()
        self.is_running = True

        self.worker_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.worker_thread.start()

        # Pre-buffer 2 silent blocks to completely eliminate buffer underruns and audio stuttering
        silent_block = np.zeros(self.chunk_len, dtype=np.float32)
        self.out_queue.put(silent_block)
        if self.device == "cpu":
            self.out_queue.put(silent_block)

        self.stream = sd.Stream(
            samplerate=self.sr,
            blocksize=self.chunk_len,
            device=(self.input_device, self.output_device),
            channels=1,
            dtype='float32',
            latency='low',
            callback=self._audio_callback
        )
        self.stream.start()
        print(f"[EngineJP] 音声ストリーム開始: @{self.sr}Hz, バッファ={self.chunk_len}サンプル ({self.chunk_len/self.sr*1000:.1f}ms遅延)")

    def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except: pass
            self.stream = None
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=0.5)
            self.worker_thread = None
        self.reset_buffers()
        print("[EngineJP] 音声ストリーム停止完了")

    def update_params(self, in_gain=None, out_gain=None, gate_db=None, key_shift=None, enable_vocoder=None, vocoder_strength=None):
        if in_gain is not None: self.input_gain = float(in_gain)
        if out_gain is not None: self.output_gain = float(out_gain)
        if gate_db is not None: self.threshold_db = float(gate_db)
        if key_shift is not None: self.key_shift = float(key_shift)
        if enable_vocoder is not None: self.enable_vocoder = bool(enable_vocoder)
        if vocoder_strength is not None: self.vocoder_strength = float(vocoder_strength)


def get_audio_devices_jp():
    """Get friendly Japanese audio device list"""
    devices = sd.query_devices()
    in_devs = []
    out_devs = []
    for idx, d in enumerate(devices):
        name = f"[{idx}] {d['name']}"
        if d['max_input_channels'] > 0:
            in_devs.append((name, idx))
        if d['max_output_channels'] > 0:
            out_devs.append((name, idx))
    return in_devs, out_devs
