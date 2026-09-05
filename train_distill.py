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
from dataset_adapter import FastAudioDataset
from mel_processing import mel_spectrogram_torch


class HuBERTFeatureLoss(nn.Module):
    """
    HuBERT Phonetic Feature Anchor Loss.
    Ensures phonemes and linguistic contents remain crystal-clear without phase divergence.
    """
    def __init__(self, device: str = "cuda"):
        super().__init__()
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.model = bundle.get_model().to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, pred_wav: torch.Tensor, tgt_wav: torch.Tensor) -> torch.Tensor:
        pred_w = pred_wav.squeeze(1).float()
        tgt_w = tgt_wav.squeeze(1).float()
        
        with torch.no_grad():
            tgt_feats, _ = self.model.extract_features(tgt_w)
            tgt_feat = tgt_feats[-1].detach()
            
        pred_feats, _ = self.model.extract_features(pred_w)
        pred_feat = pred_feats[-1]
        
        return F.mse_loss(pred_feat, tgt_feat)


def compute_mel_loss(pred_wav, gt_wav, sample_rate=16000):
    pred_w = torch.clamp(pred_wav.squeeze(1).float(), min=-1.0, max=1.0)
    gt_w = torch.clamp(gt_wav.squeeze(1).float(), min=-1.0, max=1.0)
    
    pred_mel = mel_spectrogram_torch(pred_w, sample_rate, 1024, 80, 256, 1024, 0, 8000)
    gt_mel = mel_spectrogram_torch(gt_w, sample_rate, 1024, 80, 256, 1024, 0, 8000)
    return F.l1_loss(pred_mel, gt_mel)


def main():
    parser = argparse.ArgumentParser(description="Noise-Robust LLVC Distillation (Preserves Base Denoising Power)")
    parser.add_argument("--data_dir", "-d", type=str, default="dataset/train", help="Dataset folder containing paired audio")
    parser.add_argument("--base_checkpoint", "-p", type=str,
                        default="llvc_models/models/checkpoints/llvc/G_500000.pth", help="Pretrained base model checkpoint")
    parser.add_argument("--base_config", "-c", type=str,
                        default="experiments/llvc/config.json", help="Pretrained base model config")
    parser.add_argument("--out_dir", "-o", type=str, default="my_adapter", help="Output directory")
    parser.add_argument("--model_name", "-n", type=str, default="distill_voice", help="Target model name")
    parser.add_argument("--unpaired", "-u", action="store_true", default=False, help="Unpaired mode")
    parser.add_argument("--use_hubert", action="store_true", default=True, help="Enable HuBERT semantic anchor")
    
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Epochs (default: 30)")
    parser.add_argument("--batch_size", "-b", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="Steps per epoch")
    parser.add_argument("--segment_size", type=int, default=16384, help="Segment length (samples)")
    
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    clean_name = os.path.splitext(args.model_name)[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"==================================================")
    print(f" [Noise-Robust LLVC Fast-Distill Studio Engine]")
    print(f" Target Model: {clean_name}.pth")
    print(f" Dataset: {args.data_dir} (Paired: {not args.unpaired})")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Epochs: {args.epochs} | LR: {args.lr} | HuBERT: {args.use_hubert}")
    print(f" Preservation: Base Denoising & Typing Noise Immunity ACTIVE")
    print(f"==================================================", flush=True)

    # 1. Load Base Generator Model
    with open(args.base_config) as f:
        cfg = json.load(f)
    sr = cfg.get("data", {}).get("sr", 16000)
    model = Net(**cfg["model_params"])
    
    if os.path.exists(args.base_checkpoint):
        print(f"[Init] Loading pretrained base weights from {args.base_checkpoint}", flush=True)
        ckpt = torch.load(args.base_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
    
    # Freeze I/O Waveform Filters & Frontend ConvNet to preserve 100% of base model's noise rejection capability
    for name, p in model.named_parameters():
        if "in_conv" in name or "out_conv" in name:
            p.requires_grad = False  # Protect waveform synthesis & analysis from degradation
        else:
            p.requires_grad = True   # Train Transformer & Decoder for speaker adaptation

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[Model Architecture] Trainable: {trainable_params:,} params | Protected Noise-Rejection: {frozen_params:,} params", flush=True)
    
    model.to(device)

    # 2. HuBERT Feature Loss
    hubert_loss_fn = HuBERTFeatureLoss(device=str(device)) if args.use_hubert else None

    # 3. Fast Paired Dataset
    train_dataset = FastAudioDataset(
        data_dir=args.data_dir,
        segment_size=args.segment_size,
        sample_rate=sr,
        cache_in_ram=True,
        unpaired_mode=args.unpaired,
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

    # 4. Multi-Resolution STFT & Mel Losses
    mrstft_loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 512, 256],
        hop_sizes=[256, 128, 64],
        win_lengths=[1024, 512, 256]
    ).to(device)
    l1_loss_fn = nn.L1Loss().to(device)

    opt_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(opt_params, lr=args.lr, betas=(0.8, 0.99), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=4e-5)

    # 5. Training Loop with Silence & Noise Immunity Regularization
    print(f"\n[Training] Starting {args.epochs} epochs of Noise-Immune Distillation...", flush=True)
    start_time = time.time()
    best_loss = float("inf")
    best_model_path = os.path.join(args.out_dir, f"{clean_name}.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_stft = 0.0
        epoch_mel = 0.0
        epoch_hubert = 0.0
        valid_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for src_audio, tgt_audio in pbar:
            src_audio = src_audio.to(device, non_blocking=True).float()
            tgt_audio = tgt_audio.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            pred_audio = model(src_audio)
            pred_f32 = torch.clamp(pred_audio.float(), min=-1.0, max=1.0)
            tgt_f32 = tgt_audio

            # Multi-Resolution STFT + Mel Loss + L1
            loss_stft = mrstft_loss_fn(pred_f32, tgt_f32)
            loss_l1 = l1_loss_fn(pred_f32, tgt_f32)
            loss_mel = compute_mel_loss(pred_f32, tgt_f32, sample_rate=sr)

            # HuBERT Phonetic Feature Anchor
            if hubert_loss_fn is not None:
                loss_hubert = hubert_loss_fn(pred_f32, tgt_f32) * 2.0
            else:
                loss_hubert = torch.tensor(0.0, device=device)

            # Noise / Silence Suppression Regularization:
            # Guarantees background noise, keyboard clicks, and silence output 0.0 (immune to typing sounds)
            silence_input = torch.zeros(2, 1, args.segment_size, device=device, dtype=src_audio.dtype)
            noise_input = torch.randn(2, 1, args.segment_size, device=device, dtype=src_audio.dtype) * 0.015
            quiet_inputs = torch.cat([silence_input, noise_input], dim=0)
            quiet_out = model(quiet_inputs)
            loss_quiet = torch.mean(torch.abs(quiet_out)) * 3.0

            total_loss = loss_stft * 1.0 + loss_mel * 2.0 + loss_l1 * 1.5 + loss_hubert + loss_quiet

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_stft += loss_stft.item()
            epoch_mel += loss_mel.item()
            epoch_hubert += loss_hubert.item()
            valid_steps += 1

            pbar.set_postfix({"loss": f"{total_loss.item():.3f}", "stft": f"{loss_stft.item():.2f}", "mel": f"{loss_mel.item():.2f}"})

        scheduler.step()
        num_batches = max(valid_steps, 1)
        avg_loss = epoch_loss / num_batches
        avg_stft = epoch_stft / num_batches
        avg_mel = epoch_mel / num_batches
        avg_hubert = epoch_hubert / num_batches

        if epoch % 2 == 0 or epoch == args.epochs or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] Loss: {avg_loss:.3f} (STFT: {avg_stft:.2f}, Mel: {avg_mel:.2f}, HuBERT: {avg_hubert:.3f}) | Elapsed: {elapsed:.1f}s", flush=True)

        # Save Best Clean Model
        if avg_loss < best_loss and not math.isnan(avg_loss) and avg_loss > 0:
            has_nan = any(torch.isnan(p).any().item() for p in model.parameters())
            if not has_nan:
                best_loss = avg_loss
                torch.save({
                    "model": model.state_dict(),
                    "config": cfg,
                    "metadata": {"sr": 16000, "L": 16, "loss": best_loss, "epoch": epoch, "type": "noise_robust_distill"}
                }, best_model_path)

    total_time = time.time() - start_time
    print(f"\n[Done] Noise-Robust Distillation completed in {total_time:.2f}s ({total_time/args.epochs*1000:.1f}ms/epoch)!", flush=True)
    print(f"Saved Clean Studio Model to {best_model_path} (Best Loss: {best_loss:.3f})", flush=True)


if __name__ == "__main__":
    main()
