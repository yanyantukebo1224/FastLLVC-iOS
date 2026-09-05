import os
import sys
import time
import json
import argparse
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import auraloss
from tqdm import tqdm

from model import Net
from dataset_adapter_48k import FastAudioDataset48k
from hfg_disc import MultiPeriodDiscriminator, feature_loss, generator_loss, discriminator_loss


def transfer_pretrained_weights(model_48k: nn.Module, base_ckpt_path: str):
    """
    Transfers pretrained knowledge from 16kHz base model (G_500000.pth) to 48kHz architecture.
    Performs 1:1 transfer for all attention/causal layers and smooth linear interpolation for I/O convs.
    Eliminates cold-start white noise (zar-zar noise) and enables ultra-fast convergence.
    """
    if not os.path.exists(base_ckpt_path):
        print(f"[Init] Base checkpoint {base_ckpt_path} not found. Starting from scratch.", flush=True)
        return

    print(f"[Init] Transferring pretrained weights from {base_ckpt_path} -> 48kHz Model...", flush=True)
    ckpt = torch.load(base_ckpt_path, map_location="cpu")
    state_16k = ckpt.get("model", ckpt)
    state_48k = model_48k.state_dict()

    transferred = 0
    interpolated = 0

    for k, v in state_16k.items():
        if k in state_48k:
            if state_48k[k].shape == v.shape:
                state_48k[k].copy_(v)
                transferred += 1
            elif state_48k[k].ndim == v.ndim and state_48k[k].ndim == 3:
                # 1D conv kernel expansion (e.g., L=16 -> L=48)
                v_interp = F.interpolate(v, size=state_48k[k].shape[-1], mode='linear', align_corners=False)
                state_48k[k].copy_(v_interp)
                interpolated += 1

    model_48k.load_state_dict(state_48k)
    print(f"[Init] Warm-Start Ready: {transferred} exact layers transferred, {interpolated} boundary convs adapted.", flush=True)


def mel_spectrogram_48k(y, n_fft=2048, num_mels=128, sampling_rate=48000, hop_size=512, win_size=2048, fmin=0, fmax=24000):
    hann_window = torch.hann_window(win_size).to(y.device)
    spec = torch.stft(
        y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
        center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True
    )
    spec = torch.abs(spec) + 1e-9
    
    mel_basis = torchaudio.functional.melscale_fbanks(
        n_freqs=(n_fft // 2) + 1,
        sample_rate=sampling_rate,
        f_min=fmin,
        f_max=fmax if fmax is not None else sampling_rate // 2,
        n_mels=num_mels,
        norm='slaney'
    ).to(y.device)
    
    mel = torch.matmul(mel_basis.T, spec)
    return torch.log(torch.clamp(mel, min=1e-5))


class PureTorchHuBERTFeatureLoss48k(nn.Module):
    """
    Pure PyTorch 48k -> 16k Semantic Phone Anchor Loss.
    Uses pure tensor-based differentiable downsampling (bypassing problematic C++ torchaudio DLLs).
    Guarantees 100% gradient backpropagation into the 48kHz generator.
    """
    def __init__(self, device: str = "cuda"):
        super().__init__()
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.model = bundle.get_model().to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, pred_wav_48k: torch.Tensor, tgt_wav_48k: torch.Tensor) -> torch.Tensor:
        # Differentiable 3:1 Downsampling (48kHz -> 16kHz)
        # pred_wav_48k: [B, 1, T_48k]
        pred_w = pred_wav_48k if pred_wav_48k.ndim == 3 else pred_wav_48k.unsqueeze(1)
        tgt_w = tgt_wav_48k if tgt_wav_48k.ndim == 3 else tgt_wav_48k.unsqueeze(1)

        t_16k = pred_w.shape[-1] // 3
        pred_16k = F.interpolate(pred_w.float(), size=t_16k, mode='linear', align_corners=False).squeeze(1)
        tgt_16k = F.interpolate(tgt_w.float(), size=t_16k, mode='linear', align_corners=False).squeeze(1)
        
        with torch.no_grad():
            tgt_feats, _ = self.model.extract_features(tgt_16k)
            tgt_feat = tgt_feats[-1].detach()
            
        pred_feats, _ = self.model.extract_features(pred_16k)
        pred_feat = pred_feats[-1]
        
        return F.mse_loss(pred_feat, tgt_feat)


def main():
    parser = argparse.ArgumentParser(description="Hyper-Upgrade 48kHz LLVC Fast-Distill Training Engine")
    parser.add_argument("--data_dir", "-d", type=str, default="dataset/train", help="Dataset folder containing audio")
    parser.add_argument("--base_checkpoint", "-p", type=str, default="llvc_models/models/checkpoints/llvc/G_500000.pth", help="Pretrained 16k base weights")
    parser.add_argument("--base_config", "-c", type=str, default="experiments/llvc_48k/config.json", help="48k config")
    parser.add_argument("--out_dir", "-o", type=str, default="my_adapter", help="Output directory")
    parser.add_argument("--model_name", "-n", type=str, default="hyper_48k_voice", help="Target model name")
    
    # Mode & Architecture
    parser.add_argument("--unpaired", "-u", action="store_true", default=False, help="Unpaired mode")
    parser.add_argument("--use_hubert", action="store_true", default=True, help="Enable HuBERT semantic anchor")
    parser.add_argument("--use_gan", action="store_true", default=True, help="Enable Multi-Period Discriminator GAN")
    parser.add_argument("--no_pitch_aug", action="store_true", default=False, help="Disable on-the-fly pitch augmentation")
    
    # Hyperparameters
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Epochs (default: 30)")
    parser.add_argument("--batch_size", "-b", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate")
    parser.add_argument("--steps_per_epoch", type=int, default=150, help="Steps per epoch")
    parser.add_argument("--segment_size", type=int, default=24576, help="Segment length (samples @ 48kHz = 512ms)")
    
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    clean_name = os.path.splitext(args.model_name)[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"==================================================")
    print(f" [Hyper-Upgrade 48kHz Fast-LLVC Distill Engine]")
    print(f" Target Model: {clean_name}.pth")
    print(f" Sample Rate: 48,000 Hz (Full-Band Studio Quality)")
    print(f" Dataset: {args.data_dir} (Paired: {not args.unpaired})")
    print(f" Pitch Augmentation: {not args.no_pitch_aug}")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Epochs: {args.epochs} | LR: {args.lr} | GAN: {args.use_gan} | HuBERT: {args.use_hubert}")
    print(f"==================================================", flush=True)

    # 1. Load 48k Model Config & Warm Transfer Pretrained Base Weights
    with open(args.base_config) as f:
        cfg = json.load(f)
    sr = cfg.get("data", {}).get("sr", 48000)
    model = Net(**cfg["model_params"])
    
    transfer_pretrained_weights(model, args.base_checkpoint)
    
    for p in model.parameters():
        p.requires_grad = True
    model.to(device)

    # 2. Discriminator (Multi-Period GAN)
    if args.use_gan:
        disc = MultiPeriodDiscriminator().to(device)
        opt_disc = torch.optim.AdamW(disc.parameters(), lr=args.lr * 0.4, betas=(0.8, 0.99))
    else:
        disc = None
        opt_disc = None

    # 3. Pure Torch HuBERT Feature Loss (Zero DLL errors, 100% Autograd)
    hubert_loss_fn = PureTorchHuBERTFeatureLoss48k(device=str(device)) if args.use_hubert else None

    # 4. Fast 48k Dataset with Pitch Augmentation
    train_dataset = FastAudioDataset48k(
        data_dir=args.data_dir,
        segment_size=args.segment_size,
        sample_rate=sr,
        cache_in_ram=True,
        unpaired_mode=args.unpaired,
        pitch_augmentation=(not args.no_pitch_aug),
        steps_per_epoch=args.steps_per_epoch,
        is_eval=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )

    # 5. Multi-Resolution STFT & Mel Losses for 48kHz
    mrstft_loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[2048, 1024, 512, 256],
        hop_sizes=[512, 256, 128, 64],
        win_lengths=[2048, 1024, 512, 256]
    ).to(device)
    l1_loss_fn = nn.L1Loss().to(device)

    opt_gen = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=1e-4)
    sched_gen = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gen, T_max=args.epochs, eta_min=4e-5)

    # 6. Training Loop with Progressive GAN Warmup
    print(f"\n[Training] Starting {args.epochs} epochs of 48kHz Warm-Start Distillation...", flush=True)
    start_time = time.time()
    best_loss = float("inf")
    best_model_path = os.path.join(args.out_dir, f"{clean_name}.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        if disc: disc.train()
        
        # Progressive GAN Weight Warmup: Let STFT & Mel establish solid harmonic foundation first
        if epoch <= 3:
            adv_weight = 0.0
        elif epoch <= 8:
            adv_weight = 0.1 * ((epoch - 3) / 5.0)
        else:
            adv_weight = 0.2

        epoch_loss = 0.0
        epoch_stft = 0.0
        epoch_mel = 0.0
        epoch_hubert = 0.0
        valid_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for src_audio, tgt_audio in pbar:
            src_audio = src_audio.to(device, non_blocking=True).float()
            tgt_audio = tgt_audio.to(device, non_blocking=True).float()

            # Train Generator
            opt_gen.zero_grad(set_to_none=True)

            pred_audio = model(src_audio)
            pred_f32 = torch.clamp(pred_audio.float(), min=-1.0, max=1.0)
            tgt_f32 = tgt_audio

            # Spectral & Waveform Losses
            loss_stft = mrstft_loss_fn(pred_f32, tgt_f32)
            loss_l1 = l1_loss_fn(pred_f32, tgt_f32)
            
            # 48kHz Full-band Mel Loss
            pred_mel = mel_spectrogram_48k(pred_f32.squeeze(1))
            tgt_mel = mel_spectrogram_48k(tgt_f32.squeeze(1))
            loss_mel = F.l1_loss(pred_mel, tgt_mel)

            # HuBERT Semantic Phone Anchor (Differentiable 48k -> 16k)
            if hubert_loss_fn is not None:
                loss_hubert = hubert_loss_fn(pred_f32, tgt_f32) * 2.0
            else:
                loss_hubert = torch.tensor(0.0, device=device)

            # GAN Adversarial Loss with Warmup
            if disc is not None and adv_weight > 0.0:
                y_d_rs, y_d_gs, fmap_rs, fmap_gs = disc(tgt_f32, pred_f32)
                loss_gen, _ = generator_loss(y_d_gs)
                loss_feat = feature_loss(fmap_rs, fmap_gs)
                loss_adv = (loss_gen + 2.0 * loss_feat) * adv_weight
            else:
                loss_adv = torch.tensor(0.0, device=device)

            total_gen_loss = loss_stft * 1.0 + loss_mel * 2.5 + loss_l1 * 1.5 + loss_hubert + loss_adv

            if torch.isnan(total_gen_loss) or torch.isinf(total_gen_loss):
                continue

            total_gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_gen.step()

            # Train Discriminator only when active
            if disc is not None and adv_weight > 0.0:
                opt_disc.zero_grad(set_to_none=True)
                y_d_rs, y_d_gs, _, _ = disc(tgt_f32, pred_f32.detach())
                loss_d, _, _ = discriminator_loss(y_d_rs, y_d_gs)
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
                opt_disc.step()

            epoch_loss += total_gen_loss.item()
            epoch_stft += loss_stft.item()
            epoch_mel += loss_mel.item()
            epoch_hubert += loss_hubert.item()
            valid_steps += 1

            pbar.set_postfix({"loss": f"{total_gen_loss.item():.3f}", "mel": f"{loss_mel.item():.2f}", "stft": f"{loss_stft.item():.2f}"})

        sched_gen.step()
        num_batches = max(valid_steps, 1)
        avg_loss = epoch_loss / num_batches
        avg_stft = epoch_stft / num_batches
        avg_mel = epoch_mel / num_batches
        avg_hubert = epoch_hubert / num_batches

        if epoch % 2 == 0 or epoch == args.epochs or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] Loss: {avg_loss:.3f} (STFT: {avg_stft:.2f}, Mel: {avg_mel:.2f}, HuBERT: {avg_hubert:.3f}) | Adv: {adv_weight:.2f} | Elapsed: {elapsed:.1f}s", flush=True)

        # Save Best Clean 48k Model
        if avg_loss < best_loss and not math.isnan(avg_loss) and avg_loss > 0:
            has_nan = any(torch.isnan(p).any().item() for p in model.parameters())
            if not has_nan:
                best_loss = avg_loss
                torch.save({
                    "model": model.state_dict(),
                    "config": cfg,
                    "metadata": {"sr": 48000, "L": 48, "loss": best_loss, "epoch": epoch, "type": "hyper_48k_distill"}
                }, best_model_path)

    total_time = time.time() - start_time
    print(f"\n[Done] 48kHz Hyper-Distill completed in {total_time:.2f}s ({total_time/args.epochs*1000:.1f}ms/epoch)!", flush=True)
    print(f"Saved 48kHz Studio Model to {best_model_path} (Best Loss: {best_loss:.3f})", flush=True)


if __name__ == "__main__":
    main()
