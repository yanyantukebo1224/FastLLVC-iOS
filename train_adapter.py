import os
import time
import json
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import auraloss
from tqdm import tqdm

from model import Net
from adapter import LLVCAdapterManager
from dataset_adapter import FastAudioDataset
from mel_processing import mel_spectrogram_torch


def compute_mel_loss(pred_wav, gt_wav, sample_rate=16000):
    pred_w = torch.clamp(pred_wav.squeeze(1).float(), min=-1.0, max=1.0)
    gt_w = torch.clamp(gt_wav.squeeze(1).float(), min=-1.0, max=1.0)
    
    pred_mel = mel_spectrogram_torch(
        pred_w, sample_rate, 1024, 80, 256, 1024, 0, 8000
    )
    gt_mel = mel_spectrogram_torch(
        gt_w, sample_rate, 1024, 80, 256, 1024, 0, 8000
    )
    return F.l1_loss(pred_mel, gt_mel)


def main():
    parser = argparse.ArgumentParser(description="Ultra-Clean High-Fidelity LLVC Training (ROCm)")
    parser.add_argument("--data_dir", "-d", type=str, default="test_wavs", help="Path to training dataset folder")
    parser.add_argument("--val_dir", type=str, default=None, help="Path to validation dataset folder (optional)")
    parser.add_argument("--base_checkpoint", "-p", type=str,
                        default="llvc_models/models/checkpoints/llvc/G_500000.pth", help="Pretrained base model checkpoint")
    parser.add_argument("--base_config", "-c", type=str,
                        default="experiments/llvc/config.json", help="Pretrained base model config")
    parser.add_argument("--out_dir", "-o", type=str, default="my_adapter", help="Output directory for models & logs")
    parser.add_argument("--adapter_name", "-n", type=str, default="my_voice",
                        help="Custom name for the output model (e.g. yonedu, zundamon)")
    
    # Adaptation Mode: Full Fine-Tuning vs LoRA Adapter
    parser.add_argument("--full_finetune", action="store_true", default=True,
                        help="Full Fine-Tuning mode: trains all 3.2M params for maximum voice transformation quality (Recommended!)")
    parser.add_argument("--lora_mode", action="store_true", default=False,
                        help="Use lightweight LoRA adapter mode instead of full fine-tuning")
    
    # Unpaired Any-to-One Mode
    parser.add_argument("--unpaired", "-u", action="store_true", default=True,
                        help="Enable Unpaired Any-to-One mode (default: True)")
    parser.add_argument("--steps_per_epoch", type=int, default=100,
                        help="Number of random segment batches sampled per epoch (default: 100)")
    
    # Training parameters
    parser.add_argument("--lora_rank", "-r", type=int, default=16, help="LoRA Rank (if lora_mode)")
    parser.add_argument("--lora_alpha", "-a", type=float, default=64.0, help="LoRA Alpha (if lora_mode)")
    parser.add_argument("--epochs", "-e", type=int, default=40, help="Number of training epochs (default: 40)")
    parser.add_argument("--batch_size", "-b", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate (default: 1.5e-4)")
    parser.add_argument("--segment_size", type=int, default=16384, help="Segment length in samples")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable AMP FP16 on ROCm GPU")
    
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    clean_name = os.path.splitext(args.adapter_name)[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    is_full = not args.lora_mode
    mode_desc = "[Full Model Fine-Tuning] (Clean High-Fidelity Shift)" if is_full else f"[LoRA Adapter] (Rank={args.lora_rank})"
    
    print(f"==================================================")
    print(f" LLVC Ultra-Clean Voice Conversion Studio (ROCm)")
    print(f" Target Model: {clean_name}.pth")
    print(f" Mode: {mode_desc}")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Epochs: {args.epochs} | Steps/Epoch: {args.steps_per_epoch} | FP16: {args.fp16}")
    print(f"==================================================")

    # 1. Load Base Generator Model
    with open(args.base_config) as f:
        cfg = json.load(f)
    sr = cfg.get("data", {}).get("sr", 16000)
    model = Net(**cfg["model_params"])
    
    if os.path.exists(args.base_checkpoint):
        print(f"[Init] Loading base weights from {args.base_checkpoint}")
        ckpt = torch.load(args.base_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
    else:
        print(f"[Warning] Checkpoint {args.base_checkpoint} not found!")

    # 2. Setup Mode
    if is_full:
        print(f"[Model] Enabling Clean Full Model Tuning (3.25M params)...")
        for param in model.parameters():
            param.requires_grad = True
        train_lr = args.lr
    else:
        print(f"[Model] Injecting LoRA layers (rank={args.lora_rank}, alpha={args.lora_alpha})...")
        model = LLVCAdapterManager.apply_lora(model, r=args.lora_rank, lora_alpha=args.lora_alpha)
        train_lr = args.lr if args.lr != 1.5e-4 else 1.0e-3

    model.to(device)

    # 3. High-Density Dataset & DataLoader
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

    # 4. Multi-Resolution STFT & Mel Losses (Clean and Stable)
    mrstft_loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 512, 256],
        hop_sizes=[256, 128, 64],
        win_lengths=[1024, 512, 256]
    ).to(device)
    l1_loss_fn = nn.L1Loss().to(device)

    opt_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(opt_params, lr=train_lr, betas=(0.8, 0.99), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    scaler = torch.amp.GradScaler('cuda', enabled=(args.fp16 and torch.cuda.is_available()))
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "logs"))

    # 5. Training Loop
    print(f"\n[Training] Starting {args.epochs} epochs of Clean High-Fidelity Voice Adaptation...")
    start_time = time.time()
    best_loss = float("inf")

    best_model_path = os.path.join(args.out_dir, f"{clean_name}.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_stft_loss = 0.0
        epoch_mel_loss = 0.0
        valid_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for src_audio, tgt_audio in pbar:
            src_audio = src_audio.to(device, non_blocking=True)
            tgt_audio = tgt_audio.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(args.fp16 and torch.cuda.is_available()), dtype=torch.float16):
                pred_audio = model(src_audio)
                pred_f32 = torch.nan_to_num(pred_audio.float(), nan=0.0, posinf=1.0, neginf=-1.0)
                tgt_f32 = tgt_audio.float()

                # Multi-Resolution STFT + Mel Loss + L1
                loss_stft = mrstft_loss_fn(pred_f32, tgt_f32)
                loss_l1 = l1_loss_fn(pred_f32, tgt_f32)
                loss_mel = compute_mel_loss(pred_f32, tgt_f32, sample_rate=sr)

                total_loss = loss_stft * 1.5 + loss_mel * 2.5 + loss_l1 * 1.0

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(opt_params, max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
            epoch_stft_loss += loss_stft.item()
            epoch_mel_loss += loss_mel.item()
            valid_steps += 1

            pbar.set_postfix({"loss": f"{total_loss.item():.2f}", "mel": f"{loss_mel.item():.2f}"})

        scheduler.step()
        num_batches = max(valid_steps, 1)
        avg_loss = epoch_loss / num_batches
        avg_stft = epoch_stft_loss / num_batches
        avg_mel = epoch_mel_loss / num_batches

        if epoch % 5 == 0 or epoch == args.epochs:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] TotalLoss: {avg_loss:.3f} | STFT: {avg_stft:.3f} | Mel: {avg_mel:.3f} | Elapsed: {elapsed:.1f}s")

        # Save Best Model (strictly check non-nan)
        if avg_loss < best_loss and not math.isnan(avg_loss) and avg_loss > 0:
            has_nan = any(torch.isnan(p).any().item() for p in model.parameters())
            if not has_nan:
                best_loss = avg_loss
                if is_full:
                    torch.save({"model": model.state_dict(), "metadata": {"loss": best_loss, "epoch": epoch, "type": "full"}}, best_model_path)
                else:
                    LLVCAdapterManager.save_adapter(model, best_model_path, metadata={"epoch": epoch, "loss": best_loss, "type": "lora"})

    total_time = time.time() - start_time
    print(f"\n[Done] Training completed in {total_time:.2f}s ({total_time/args.epochs*1000:.1f}ms/epoch)!")
    print(f"Best clean model saved to {best_model_path} (Loss: {best_loss:.3f})")


if __name__ == "__main__":
    main()
