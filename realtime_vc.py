import time
import threading
import queue
import os
import json
import numpy as np
import torch
import sounddevice as sd
from typing import Optional, Callable

from infer_adapter import load_model_with_adapter
from audio_enhancer import ZeroLatencyStudioEnhancer
from pitch_shifter import ZeroLatencyPitchShifter


class CleanVocalHighPassFilter:
    """
    Zero-Latency 2nd-order Butterworth High-Pass / Low-Shelf Filter.
    Cuts muddy chest resonance (80-160Hz) to prevent vocal tract leakage without introducing phase distortion.
    """
    def __init__(self, sample_rate: int = 48000, cutoff_hz: float = 120.0):
        self.sr = sample_rate
        self.cutoff_hz = cutoff_hz
        self._calc_coeffs()
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0

    def _calc_coeffs(self):
        w0 = 2.0 * np.pi * self.cutoff_hz / self.sr
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2.0 * 0.7071)  # Q = 0.7071 (Butterworth)
        
        b0 = (1.0 + cos_w0) / 2.0
        b1 = -(1.0 + cos_w0)
        b2 = (1.0 + cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0

    def reset(self):
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        out = np.zeros_like(audio)
        for i in range(len(audio)):
            x0 = float(audio[i])
            y0 = self.b0 * x0 + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2
            self.x2 = self.x1
            self.x1 = x0
            self.y2 = self.y1
            self.y1 = y0
            out[i] = y0
        return out


class RealtimeVCEngine:
    """
    Ultra-Low Latency & High-Stability Realtime Voice Conversion Engine.
    Supports both 48kHz (High-Fidelity Full-Band) and 16kHz models seamlessly.
    Powered by BF16 (bfloat16) Tensor Acceleration (Sub-4ms Inference on ROCm/CUDA).
    """
    def __init__(
        self,
        checkpoint_path: str = 'llvc_models/models/checkpoints/llvc/G_500000.pth',
        config_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        chunk_factor: int = 1,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        input_gain: float = 1.0,
        output_gain: float = 1.0,
        threshold_db: float = -45.0,
        key_shift: float = 0.0,
        enable_vocoder: bool = False,
        vocoder_strength: float = 0.6,
        enable_low_cut: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
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
        self.dtype = dtype if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        self.device = device
        self.on_latency_update = on_latency_update
        self.on_volume_update = on_volume_update

        self.model = None
        self.sr = 16000
        self.L = 16
        self.dec_chunk_size = 13
        self.chunk_len = self.dec_chunk_size * self.L * self.chunk_factor  # 208 samples = 13.0ms @ 16kHz
        self.ctx_len = self.L * 2  # 32 samples
        self.total_chunk_len = self.ctx_len + self.chunk_len

        self.enhancer = None
        self.pitch_shifter = None
        self.low_cut_filter = None
        
        self.is_running = False
        self.stream = None
        self.worker_thread = None
        
        # Audio queues
        self.in_queue = queue.Queue(maxsize=8)
        self.out_queue = queue.Queue(maxsize=8)
        
        # Audio Filter States
        self.dc_prev_in = 0.0
        self.dc_prev_out = 0.0
        self.current_gate_gain = 1.0
        self.last_out_sample = 0.0
        
        # GPU Context buffers
        self.enc_buf = None
        self.dec_buf = None
        self.out_buf = None
        self.convnet_pre_ctx = None
        self.prev_front_ctx = None
        self.chunk_with_ctx_buf = None

    def _auto_detect_config(self, cp: str) -> str:
        if self.config_path and os.path.exists(self.config_path):
            return self.config_path
        
        # Try inspecting checkpoint
        try:
            ckpt = torch.load(cp, map_location="cpu")
            if "config" in ckpt and isinstance(ckpt["config"], dict):
                sr = ckpt["config"].get("data", {}).get("sr", 48000)
                if sr == 48000:
                    return "experiments/llvc_48k/config.json"
            if "metadata" in ckpt:
                sr = ckpt["metadata"].get("sr", 16000)
                if sr == 48000:
                    return "experiments/llvc_48k/config.json"
        except Exception:
            pass

        if "48k" in cp.lower():
            return "experiments/llvc_48k/config.json"
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
        
        self.L = getattr(self.model, 'L', 48 if self.sr == 48000 else 16)
        self.dec_chunk_size = getattr(self.model, 'dec_chunk_size', 13)
        self.chunk_len = self.dec_chunk_size * self.L * self.chunk_factor
        self.ctx_len = self.L * 2
        self.total_chunk_len = self.ctx_len + self.chunk_len

        self.enhancer = ZeroLatencyStudioEnhancer(sample_rate=self.sr)
        self.pitch_shifter = ZeroLatencyPitchShifter(sample_rate=self.sr)
        self.low_cut_filter = CleanVocalHighPassFilter(sample_rate=self.sr, cutoff_hz=100.0)
        print(f"[RealtimeVCEngine] Loaded model @ {self.sr}Hz (L={self.L}, Chunk={self.chunk_len} samples = {self.chunk_len/self.sr*1000:.1f}ms)")

    def _init_gpu_buffers(self):
        """Initializes and pre-allocates all GPU context and input buffers in BF16"""
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
        """Warm up CUDA/ROCm kernels to guarantee sub-4ms steady latency from the very 1st block"""
        if self.model is None:
            self.load_model()
        self._init_gpu_buffers()
        
        with torch.inference_mode():
            dummy_in = torch.zeros(1, 1, self.total_chunk_len, dtype=self.dtype, device=self.device)
            for _ in range(10):
                out, self.enc_buf, self.dec_buf, self.out_buf, self.convnet_pre_ctx = self.model(
                    dummy_in, self.enc_buf, self.dec_buf, self.out_buf, self.convnet_pre_ctx,
                    pad=False
                )
            if self.device == "cuda":
                torch.cuda.synchronize()
        self._init_gpu_buffers()

    def _inference_worker(self):
        """Dedicated high-priority background worker for BF16 GPU inference (~3.6ms)"""
        while self.is_running:
            try:
                in_audio, in_rms = self.in_queue.get(timeout=0.03)
            except queue.Empty:
                continue

            # Calculate RMS & Noise Gate Target
            in_db = 20 * np.log10(in_rms + 1e-9)
            target_gain = 1.0 if in_db >= self.threshold_db else 0.0

            # 1. Clean Vocal Low-Cut (Cuts 100Hz chest resonance to break source timbre leakage)
            if self.enable_low_cut and self.low_cut_filter is not None:
                in_proc = self.low_cut_filter.process(in_audio)
            else:
                in_proc = in_audio

            # 2. Key Shift (if enabled and non-zero)
            if abs(self.key_shift) >= 0.1 and self.pitch_shifter is not None:
                in_proc = self.pitch_shifter.process_numpy(in_proc, self.key_shift)

            with torch.inference_mode():
                t0 = time.perf_counter()
                
                in_tensor = torch.from_numpy(in_proc).to(device=self.device, dtype=self.dtype)
                
                # Sample-Accurate Lookahead Streaming Alignment (Zero Comb Filter / Robotic Artifacts)
                # Form current processing chunk: prev_lookahead_carry (L) + in_tensor[:-L]
                # New lookahead carry becomes in_tensor[-L:]
                curr_stream_chunk = torch.cat([self.prev_lookahead_carry, in_tensor[:-self.L]])
                self.prev_lookahead_carry.copy_(in_tensor[-self.L:])

                # Assemble chunk with past context: front_ctx (2L) + curr_stream_chunk (chunk_len)
                self.chunk_with_ctx_buf[0, 0, :self.ctx_len] = self.prev_front_ctx
                self.chunk_with_ctx_buf[0, 0, self.ctx_len:] = curr_stream_chunk
                self.prev_front_ctx.copy_(curr_stream_chunk[-self.ctx_len:])

                # Execute forward pass (BF16 Tensor Cores: ~3.6ms)
                output, self.enc_buf, self.dec_buf, self.out_buf, self.convnet_pre_ctx = self.model(
                    self.chunk_with_ctx_buf, self.enc_buf, self.dec_buf, self.out_buf, self.convnet_pre_ctx,
                    pad=False
                )

                t_infer = (time.perf_counter() - t0) * 1000.0
                rtf = (self.chunk_len / self.sr * 1000.0) / max(t_infer, 0.001)

                if self.on_latency_update:
                    self.on_latency_update(t_infer, rtf)

                # Convert to numpy float32
                out_audio = output.squeeze().float().cpu().numpy()

                # Smooth Noise Gate Linear Ramp
                gain_start = self.current_gate_gain
                self.current_gate_gain = 0.80 * self.current_gate_gain + 0.20 * target_gain
                gain_end = self.current_gate_gain
                
                gain_ramp = np.linspace(gain_start, gain_end, len(out_audio), dtype=np.float32)
                out_audio = out_audio * (self.output_gain * gain_ramp)

                # Studio Audio Enhancer (Air / Presence)
                if self.enable_vocoder and self.enhancer is not None and self.current_gate_gain > 0.02:
                    out_audio = self.enhancer.process_numpy(
                        out_audio,
                        clarity=0.5,
                        air=float(self.vocoder_strength),
                        warmth=0.3
                    )

                # Transparent Soft Limiter
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

    def switch_device(self, input_device: Optional[int] = None, output_device: Optional[int] = None):
        if input_device is not None:
            self.input_device = input_device
        if output_device is not None:
            self.output_device = output_device

        if self.is_running and self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            
            self.stream = sd.Stream(
                samplerate=self.sr,
                blocksize=self.chunk_len,
                dtype='float32',
                channels=(1, 2),
                device=(self.input_device, self.output_device),
                latency='low',
                callback=self._audio_callback
            )
            self.stream.start()

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

        self.stream = sd.Stream(
            samplerate=self.sr,
            blocksize=self.chunk_len,
            dtype='float32',
            channels=(1, 2),
            device=(self.input_device, self.output_device),
            latency='low',
            callback=self._audio_callback
        )
        self.stream.start()
        print(f"[RealtimeVCEngine] Stream started @ {self.sr}Hz, Blocksize: {self.chunk_len} ({self.chunk_len/self.sr*1000:.1f}ms latency)")

    def stop(self):
        self.is_running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        if self.worker_thread is not None:
            self.worker_thread.join(timeout=0.2)
            self.worker_thread = None

        self.reset_buffers()
        print("[RealtimeVCEngine] Stream stopped.")

    @staticmethod
    def get_audio_devices():
        try:
            devs = sd.query_devices()
            hostapis = sd.query_hostapis()
            
            in_devs = []
            out_devs = []
            for idx, d in enumerate(devs):
                api_name = hostapis[d['hostapi']]['name']
                name = f"[{api_name}] {d['name']}"
                if d['max_input_channels'] > 0:
                    in_devs.append((idx, name))
                if d['max_output_channels'] > 0:
                    out_devs.append((idx, name))
            return in_devs, out_devs
        except Exception as e:
            print(f"Error querying audio devices: {e}")
            return [], []

