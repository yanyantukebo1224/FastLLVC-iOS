import os
import glob
import random
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
import torchaudio
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma')


class RobustAudioAugmentor48k:
    """
    48kHz Advanced Acoustic Perturbation Engine.
    Simulates diverse speaker pitch ranges, formants, microphone curves, and dynamic ranges
    to prevent the student model from leaking source speaker timbre.
    """
    def __init__(self, sample_rate: int = 48000):
        self.sr = sample_rate

    def augment_source(self, wav: torch.Tensor, pitch_shift_prob: float = 0.8) -> torch.Tensor:
        """
        Applies pitch/formant shifting and EQ perturbation to the source (input) speech.
        Target speech remains 100% pristine.
        """
        orig_len = wav.shape[-1]
        x = wav.clone()

        # 1. Pitch & Formant Shift (Simulate Male <-> Female / Child / Deep voice)
        if random.random() < pitch_shift_prob:
            # Shift from -10 semitones (deeper) to +12 semitones (higher)
            pitch_ratio = random.choice([
                random.uniform(0.65, 0.88),  # Deep male voice
                random.uniform(1.12, 1.65),  # High female / anime voice
                random.uniform(0.90, 1.10)   # Subtle pitch jitter
            ])
            resampled_len = max(int(orig_len / pitch_ratio), 128)
            x_interp = F.interpolate(x.unsqueeze(0), size=resampled_len, mode='linear', align_corners=False).squeeze(0)
            if x_interp.shape[-1] < orig_len:
                x = F.pad(x_interp, (0, orig_len - x_interp.shape[-1]))
            else:
                x = x_interp[:, :orig_len]

        # 2. Spectral Tilt & Formant Reshaping (Alters throat resonance & mic EQ)
        spec = torch.fft.rfft(x, dim=-1)
        freq_bins = spec.shape[-1]
        
        mode = random.choice(['tilt_low', 'tilt_high', 'notch', 'vocal_cut', 'none'])
        if mode == 'tilt_low':
            slope = torch.linspace(1.6, 0.4, freq_bins, device=x.device)
            spec = spec * slope
        elif mode == 'tilt_high':
            slope = torch.linspace(0.4, 1.6, freq_bins, device=x.device)
            spec = spec * slope
        elif mode == 'notch':
            center = random.randint(int(freq_bins * 0.05), int(freq_bins * 0.7))
            bw = random.randint(int(freq_bins * 0.03), int(freq_bins * 0.15))
            notch = 1.0 - 0.75 * torch.exp(-0.5 * ((torch.arange(freq_bins, device=x.device) - center) / max(bw, 1)) ** 2)
            spec = spec * notch
        elif mode == 'vocal_cut':
            # Cut typical 200-500Hz chest resonance
            center = int(freq_bins * (300.0 / (self.sr / 2)))
            bw = int(freq_bins * (150.0 / (self.sr / 2)))
            notch = 1.0 - 0.8 * torch.exp(-0.5 * ((torch.arange(freq_bins, device=x.device) - center) / max(bw, 1)) ** 2)
            spec = spec * notch

        x = torch.fft.irfft(spec, n=orig_len, dim=-1)

        # 3. Dynamic Gain & Low-level noise
        gain = random.uniform(0.7, 1.3)
        x = x * gain
        if random.random() < 0.3:
            x = x + torch.randn_like(x) * random.uniform(0.0005, 0.003)

        return torch.clamp(x, -1.0, 1.0)


def load_any_audio_48k(path: str, target_sr: int = 48000) -> torch.Tensor:
    try:
        audio, sr = torchaudio.load(path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != target_sr:
            audio = torchaudio.transforms.Resample(sr, target_sr)(audio)
        return audio.squeeze(0).float()
    except Exception:
        try:
            data, sr = sf.read(path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            t = torch.from_numpy(data).float().unsqueeze(0)
            if sr != target_sr:
                t = torchaudio.transforms.Resample(sr, target_sr)(t)
            return t.squeeze(0).float()
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(target_sr * 2, dtype=torch.float32)


class FastAudioDataset48k(Dataset):
    """
    High-Performance 48kHz Dataset for LLVC Distillation.
    Pre-caches audio in RAM to ensure GPU compute is 100% saturated without disk I/O bottlenecks.
    """
    def __init__(
        self,
        data_dir: str,
        segment_size: int = 24576,  # ~512ms @ 48kHz
        sample_rate: int = 48000,
        cache_in_ram: bool = True,
        unpaired_mode: bool = False,
        pitch_augmentation: bool = True,
        steps_per_epoch: int = 200,
        is_eval: bool = False
    ):
        super().__init__()
        self.segment_size = segment_size
        self.sample_rate = sample_rate
        self.unpaired_mode = unpaired_mode
        self.pitch_augmentation = pitch_augmentation and (not is_eval)
        self.steps_per_epoch = steps_per_epoch
        self.is_eval = is_eval
        self.augmentor = RobustAudioAugmentor48k(sample_rate=sample_rate)

        # 1. Discover audio pairs
        all_files = []
        for ext in AUDIO_EXTENSIONS:
            all_files.extend(glob.glob(os.path.join(data_dir, f"**/*{ext}"), recursive=True))

        self.pairs = []
        if not unpaired_mode:
            # Paired mode: *_original vs *_converted
            orig_files = [f for f in all_files if "_original" in os.path.basename(f)]
            for orig in orig_files:
                base = orig.replace("_original", "_converted")
                if os.path.exists(base):
                    self.pairs.append((orig, base))
            
            # If no suffix found, check paired folders (src/ vs tgt/)
            if len(self.pairs) == 0:
                src_files = sorted(glob.glob(os.path.join(data_dir, "src", "*.*")))
                tgt_files = sorted(glob.glob(os.path.join(data_dir, "tgt", "*.*")))
                for s, t in zip(src_files, tgt_files):
                    self.pairs.append((s, t))

        if len(self.pairs) == 0:
            print(f"[Dataset48k] Paired files not found in {data_dir}. Falling back to Unpaired Self-Supervised Augmentation Mode.")
            self.unpaired_mode = True
            self.single_files = [f for f in all_files if os.path.isfile(f)]
        else:
            print(f"[Dataset48k] Found {len(self.pairs)} paired audio files @ 48kHz.")

        # 2. RAM Pre-caching
        self.cached_pairs = []
        self.cached_singles = []
        if cache_in_ram:
            if not self.unpaired_mode:
                for src_p, tgt_p in self.pairs:
                    s_wav = load_any_audio_48k(src_p, self.sample_rate)
                    t_wav = load_any_audio_48k(tgt_p, self.sample_rate)
                    min_len = min(len(s_wav), len(t_wav))
                    if min_len >= self.segment_size:
                        self.cached_pairs.append((s_wav[:min_len], t_wav[:min_len]))
                print(f"[Dataset48k] Cached {len(self.cached_pairs)} valid audio pairs in RAM.")
            else:
                for f in self.single_files:
                    wav = load_any_audio_48k(f, self.sample_rate)
                    if len(wav) >= self.segment_size:
                        self.cached_singles.append(wav)
                print(f"[Dataset48k] Cached {len(self.cached_singles)} single target audio files in RAM.")

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.unpaired_mode and len(self.cached_pairs) > 0:
            src_full, tgt_full = random.choice(self.cached_pairs)
            max_start = len(src_full) - self.segment_size
            start = random.randint(0, max_start) if max_start > 0 else 0
            
            src_seg = src_full[start:start + self.segment_size].clone().unsqueeze(0)
            tgt_seg = tgt_full[start:start + self.segment_size].clone().unsqueeze(0)

            # On-the-fly pitch & formant augmentation on source to break voice leakage
            if self.pitch_augmentation and random.random() < 0.6:
                src_seg = self.augmentor.augment_source(src_seg)

            return src_seg, tgt_seg
        else:
            tgt_full = random.choice(self.cached_singles)
            max_start = len(tgt_full) - self.segment_size
            start = random.randint(0, max_start) if max_start > 0 else 0
            tgt_seg = tgt_full[start:start + self.segment_size].clone().unsqueeze(0)
            
            # Synthesize pseudo-source with full acoustic perturbation
            src_seg = self.augmentor.augment_source(tgt_seg, pitch_shift_prob=1.0)
            return src_seg, tgt_seg
