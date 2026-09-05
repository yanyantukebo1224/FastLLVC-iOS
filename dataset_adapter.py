import os
import glob
import random
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma')


class AudioAugmentationPipeline:
    """
    Robust acoustic perturbation to transform single-speaker target audio (One)
    into diverse pseudo-source speaker inputs (Any) for Any-to-One voice conversion.
    """
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def perturb(self, wav: torch.Tensor) -> torch.Tensor:
        orig_len = wav.shape[-1]
        x = wav.clone()

        # 1. Significant Pitch & Formant Shift (Simulates different vocal tract & fundamental pitch)
        # Shift pitch by +/- 3 to 8 semitones
        pitch_factor = random.choice([
            random.uniform(0.70, 0.88),  # Lower pitch (male/deeper voice)
            random.uniform(1.12, 1.35)   # Higher pitch (female/higher voice)
        ])
        
        resampled_len = int(orig_len / pitch_factor)
        x_interp = F.interpolate(x.unsqueeze(0), size=resampled_len, mode='linear', align_corners=False).squeeze(0)
        
        if x_interp.shape[-1] < orig_len:
            x = F.pad(x_interp, (0, orig_len - x_interp.shape[-1]))
        else:
            x = x_interp[:, :orig_len]

        # 2. Parametric Formant & EQ Shaping (Alters spectral envelope)
        spec = torch.fft.rfft(x, dim=-1)
        freq_bins = spec.shape[-1]
        
        filter_mode = random.choice(['tilt_low', 'tilt_high', 'notch', 'bandpass'])
        if filter_mode == 'tilt_low':
            # Boost bass, cut treble
            slope = torch.linspace(1.5, 0.4, freq_bins, device=x.device)
            spec = spec * slope
        elif filter_mode == 'tilt_high':
            # Boost treble, cut bass
            slope = torch.linspace(0.4, 1.5, freq_bins, device=x.device)
            spec = spec * slope
        elif filter_mode == 'notch':
            center = random.randint(int(freq_bins * 0.1), int(freq_bins * 0.8))
            bw = random.randint(int(freq_bins * 0.05), int(freq_bins * 0.2))
            notch = 1.0 - 0.7 * torch.exp(-0.5 * ((torch.arange(freq_bins, device=x.device) - center) / max(bw, 1)) ** 2)
            spec = spec * notch
        elif filter_mode == 'bandpass':
            center = random.randint(int(freq_bins * 0.2), int(freq_bins * 0.6))
            bw = random.randint(int(freq_bins * 0.15), int(freq_bins * 0.35))
            bp = 0.3 + 0.7 * torch.exp(-0.5 * ((torch.arange(freq_bins, device=x.device) - center) / max(bw, 1)) ** 2)
            spec = spec * bp

        x = torch.fft.irfft(spec, n=orig_len, dim=-1)

        # 3. Dynamic Compression / Gain Variation
        gain = random.uniform(0.6, 1.3)
        x = x * gain
        
        # 4. Subtle Background Noise
        if random.random() < 0.4:
            noise = torch.randn_like(x) * random.uniform(0.001, 0.005)
            x = x + noise

        return torch.clamp(x, -1.0, 1.0)


def load_any_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    try:
        audio, sr = sf.read(path, dtype='float32')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        wav = torch.from_numpy(audio).float()
    except Exception:
        try:
            import librosa
            audio, sr = librosa.load(path, sr=target_sr, mono=True)
            return torch.from_numpy(audio).float()
        except Exception:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(path).set_channels(1).set_frame_rate(target_sr)
            samples = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768.0
            return torch.from_numpy(samples).float()

    if sr != target_sr:
        import torchaudio.transforms as T
        wav = T.Resample(sr, target_sr)(wav)
    return wav


class FastAudioDataset(Dataset):
    """
    High-density in-memory cached audio dataset for fast Any-to-One voice adaptation.
    Ensures thousands of diverse speech segments are sampled every epoch regardless of file count.
    """
    def __init__(self, data_dir: str, segment_size: int = 16384, sample_rate: int = 16000,
                 cache_in_ram: bool = True, unpaired_mode: bool = False, steps_per_epoch: int = 200, is_eval: bool = False):
        super().__init__()
        self.segment_size = segment_size
        self.sample_rate = sample_rate
        self.cache_in_ram = cache_in_ram
        self.unpaired_mode = unpaired_mode
        self.steps_per_epoch = steps_per_epoch
        self.is_eval = is_eval
        self.augmentor = AudioAugmentationPipeline(sample_rate=sample_rate)

        self.pairs: List[Tuple[str, Optional[str]]] = []
        
        all_audio_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(AUDIO_EXTENSIONS):
                    all_audio_files.append(os.path.join(root, file))

        orig_files = [f for f in all_audio_files if "_original." in f]
        if orig_files and not unpaired_mode:
            print("[FastAudioDataset] Found paired dataset (*_original / *_converted).")
            for orig in orig_files:
                for ext in AUDIO_EXTENSIONS:
                    conv = orig.replace("_original.", "_converted.")
                    if os.path.exists(conv):
                        self.pairs.append((orig, conv))
                        break
        else:
            print(f"[FastAudioDataset] Unpaired Any-to-One Mode: Found {len(all_audio_files)} audio files in '{data_dir}'.")
            for w in all_audio_files:
                self.pairs.append((w, w))

        if len(self.pairs) == 0:
            raise ValueError(f"No audio files found in '{data_dir}'. Supported formats: {AUDIO_EXTENSIONS}")

        self.cached_data: List[Tuple[torch.Tensor, torch.Tensor]] = []
        total_seconds = 0.0
        if self.cache_in_ram:
            print(f"[FastAudioDataset] Pre-loading and caching {len(self.pairs)} audio files into RAM...")
            for src_p, tgt_p in self.pairs:
                src_wav = load_any_audio(src_p, self.sample_rate)
                tgt_wav = load_any_audio(tgt_p, self.sample_rate) if (tgt_p and tgt_p != src_p) else src_wav
                self.cached_data.append((src_wav, tgt_wav))
                total_seconds += len(tgt_wav) / self.sample_rate
            print(f"[FastAudioDataset] Cache completed: {len(self.cached_data)} files ({total_seconds:.1f} seconds of audio cached).")

    def __len__(self):
        # In unpaired/training mode, return virtual epoch size (e.g. 200 batches) for deep learning convergence
        if self.is_eval:
            return len(self.pairs)
        return max(self.steps_per_epoch, len(self.pairs))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Randomly choose an audio file from cache
        file_idx = random.randint(0, len(self.cached_data) - 1) if not self.is_eval else idx % len(self.cached_data)
        src_wav, tgt_wav = self.cached_data[file_idx]

        min_len = min(len(src_wav), len(tgt_wav))

        if min_len < self.segment_size:
            src_wav = torch.nn.functional.pad(src_wav, (0, self.segment_size - len(src_wav)))
            tgt_wav = torch.nn.functional.pad(tgt_wav, (0, self.segment_size - len(tgt_wav)))
            src_seg = src_wav.unsqueeze(0)
            tgt_seg = tgt_wav.unsqueeze(0)
        else:
            if self.is_eval:
                start = 0
            else:
                start = random.randint(0, min_len - self.segment_size)
            src_seg = src_wav[start : start + self.segment_size].unsqueeze(0)
            tgt_seg = tgt_wav[start : start + self.segment_size].unsqueeze(0)

        # Unpaired Any-to-One mode: synthesize pseudo-source speaker
        if self.unpaired_mode and not self.is_eval:
            src_seg = self.augmentor.perturb(tgt_seg)

        return src_seg, tgt_seg
