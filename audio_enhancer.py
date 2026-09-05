import numpy as np
import torch
import torch.nn as nn

class ZeroLatencyStudioEnhancer:
    """
    Zero-Latency Studio Audio Enhancer & Harmonic Exciter (Sample-Rate Adaptive).
    Operates at 0.0ms added latency (pure sample-by-sample DSP).
    
    Features:
    1. Sample-Rate Adaptive RC Filter: Consistent 3.5kHz crossover @ 16kHz, 44.1kHz, 48kHz.
    2. Harmonic Sheen: Soft odd/even harmonic restoration without metallic aliasing.
    3. Dynamic Sibilance Limiter: Tames harsh highs and prevents squeaking/whistling artifacts.
    4. Analog Body Warmth: Smooth low-order warmth saturation.
    """
    def __init__(self, sample_rate: int = 48000):
        self.sr = sample_rate
        self.reset()

    def reset(self):
        self.hp_prev_in = 0.0
        self.hp_prev_out = 0.0
        self.lp_prev = 0.0
        self.de_prev = 0.0

    def process_numpy(self, audio: np.ndarray, clarity: float = 0.3, air: float = 0.3, warmth: float = 0.2) -> np.ndarray:
        """
        Process audio with zero-latency harmonic excitation and clarity enhancement.
        
        clarity: 0.0 to 1.0 (Presence & formant crispness)
        air: 0.0 to 1.0 (Air band sheen)
        warmth: 0.0 to 1.0 (Body warmth)
        """
        if len(audio) == 0:
            return audio

        out = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.95, neginf=-0.95)

        # 1. Sample-Rate Adaptive High-Pass Filter (Cutoff fc = 3500 Hz)
        fc = 3500.0
        rc = 1.0 / (2.0 * np.pi * fc)
        dt = 1.0 / self.sr
        alpha = rc / (rc + dt)

        hp_filtered = np.zeros_like(out)
        for i in range(len(out)):
            hp_filtered[i] = alpha * (self.hp_prev_out + out[i] - self.hp_prev_in)
            self.hp_prev_in = out[i]
            self.hp_prev_out = hp_filtered[i]

        # 2. Smooth High-Frequency Soft Saturation (No harsh clipping)
        harmonics = np.tanh(hp_filtered * 1.2) - 0.2 * hp_filtered
        
        # 3. Dynamic De-Esser (Suppresses whistle/squeak resonances)
        sibilance = np.abs(harmonics)
        de_gain = 1.0 / (1.0 + 4.0 * sibilance)
        smooth_harmonics = harmonics * de_gain

        # 4. Controlled Presence & Air Mix
        air_injection = smooth_harmonics * (air * 0.25)
        presence_injection = hp_filtered * (clarity * 0.20)

        # 5. Analog Warmth Body
        warm_signal = out + (warmth * 0.15) * (np.tanh(out * 1.1) - out)

        # 6. Composite Output
        enhanced = warm_signal + air_injection + presence_injection
        enhanced = np.tanh(enhanced * 0.96)
        
        return enhanced.astype(np.float32)

    def process_torch(self, x: torch.Tensor, clarity: float = 0.3, air: float = 0.3, warmth: float = 0.2) -> torch.Tensor:
        orig_device = x.device
        orig_shape = x.shape
        audio_np = x.squeeze().detach().cpu().numpy()
        enhanced_np = self.process_numpy(audio_np, clarity=clarity, air=air, warmth=warmth)
        return torch.from_numpy(enhanced_np).to(orig_device).view(orig_shape)
