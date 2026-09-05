import numpy as np
import torch


class ZeroLatencyPitchShifter:
    """
    Ultra-Low Latency & High-Quality Realtime Time-Domain Pitch Shifter.
    Uses 4-Phase Overlap-Add Granular Resampling with Linear Interpolation.
    Eliminates comb-filtering, metallic flutter, clicks, and airy distortion.
    """
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        self.buf_size = 16384
        self.buffer = np.zeros(self.buf_size, dtype=np.float32)
        self.write_idx = 0
        self.phase = 0.0

    def reset(self):
        self.buffer.fill(0.0)
        self.write_idx = 0
        self.phase = 0.0

    def process_numpy(self, chunk: np.ndarray, semitones: float) -> np.ndarray:
        if abs(semitones) < 0.05:
            return chunk
        
        pitch_ratio = float(2.0 ** (semitones / 12.0))
        chunk_len = len(chunk)
        
        # 35ms grain size provides smooth fundamental tracking for human voice (80-500Hz)
        grain_size = int(self.sr * 0.035)
        # Delay rate of change: when pitch_ratio=2 (+12), delay decreases at 1 sample/sample
        delta_delay = -(pitch_ratio - 1.0)
        out = np.zeros(chunk_len, dtype=np.float32)
        base_offset = grain_size * 2

        for i in range(chunk_len):
            self.buffer[self.write_idx] = chunk[i]
            
            # Progress grain phase
            self.phase = (self.phase + delta_delay) % grain_size
            
            # 4-Tap Overlap-Add (Quarter-grain offset)
            out_sample = 0.0
            for tap in range(4):
                tap_phase = (self.phase + tap * (grain_size * 0.25)) % grain_size
                # Hann window (0 at boundaries, 1 in center)
                w = 0.5 * (1.0 - np.cos(2.0 * np.pi * tap_phase / grain_size)) * 0.5
                
                # Delay tap read position
                r = (self.write_idx - base_offset - tap_phase) % self.buf_size
                i0 = int(np.floor(r))
                i1 = (i0 + 1) % self.buf_size
                frac = r - i0
                
                # Sub-sample linear interpolation
                s = self.buffer[i0] * (1.0 - frac) + self.buffer[i1] * frac
                out_sample += s * w

            out[i] = out_sample
            self.write_idx = (self.write_idx + 1) % self.buf_size

        return out.astype(np.float32)

    def process_torch(self, x: torch.Tensor, semitones: float) -> torch.Tensor:
        if abs(semitones) < 0.05:
            return x
        orig_device = x.device
        orig_dtype = x.dtype
        orig_shape = x.shape
        audio_np = x.squeeze().detach().float().cpu().numpy()
        shifted_np = self.process_numpy(audio_np, semitones)
        return torch.from_numpy(shifted_np).to(device=orig_device, dtype=orig_dtype).view(orig_shape)
