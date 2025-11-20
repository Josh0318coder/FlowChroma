"""
Fusion Training Script

Online training mode: runs MemFlow and SwinTExCo on-the-fly for each batch.

Usage:
    python fusion/train.py \
        --memflow_path ../MemFlow \
        --swintexco_path ../SwinSingle \
        --memflow_ckpt ../MemFlow/checkpoints/memflow_best.pth \
        --swintexco_ckpt ../SwinSingle/checkpoints/best/ \
        --data_root /path/to/train_videos \
        --batch_size 2 \
        --epochs 50
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, '.')

from train.fusion_system import FusionSystem
from FusionNet.fusion_unet import SimpleFusionNet
from train.fusion_loss import FusionLoss
from train.fusion_dataset import FusionDataset


def train_epoch(system, dataloader, criterion, optimizer, scaler, epoch, args):
    """Train for one epoch"""

    system.train()
    epoch_losses = {
        'total': 0.0,
        'l1': 0.0,
        'perceptual': 0.0,
        'contextual': 0.0,
        'temporal': 0.0
    }

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (frame_t, frame_t1, reference_pil, target_pil, gt_ab) in enumerate(pbar):
        # Move to device
        frame_t = frame_t.to(args.device)
        frame_t1 = frame_t1.to(args.device)
        gt_ab = gt_ab.to(args.device)

        # Handle PIL images (batch processing)
        # Note: SwinTExCo processes one image at a time
        # So we need to loop for batch > 1
        batch_size = frame_t.shape[0]

        # Reset memory for each video sequence
        if batch_idx % args.video_length == 0:
            system.reset_memory()

        # Forward pass with mixed precision
        with autocast(enabled=args.use_amp):
            # For now, process batch_size=1
            # TODO: Add batch processing for SwinTExCo
            if batch_size > 1:
                raise NotImplementedError("Batch size > 1 not yet supported. Use --batch_size 1")

            fused_ab = system(
                frame_t,
                frame_t1,
                reference_pil[0],  # First item in batch
                target_pil[0]
            )

            # Compute loss
            loss, loss_dict = criterion(fused_ab, gt_ab)

            # Scale loss for gradient accumulation
            loss = loss / args.accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % args.accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(system.parameters(), args.max_grad_norm)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Accumulate losses
        for key in epoch_losses.keys():
            if key in loss_dict:
                epoch_losses[key] += loss_dict[key]

        # Update progress bar
        pbar.set_postfix({
            'loss': loss_dict['total'],
            'l1': loss_dict['l1'],
            'ctx': loss_dict.get('contextual', 0)
        })

    # Average losses
    num_batches = len(dataloader)
    for key in epoch_losses.keys():
        epoch_losses[key] /= num_batches

    return epoch_losses


def validate(system, dataloader, criterion, args):
    """Validation"""

    system.eval()
    val_losses = {
        'total': 0.0,
        'l1': 0.0,
        'perceptual': 0.0,
        'contextual': 0.0
    }

    with torch.no_grad():
        for batch_idx, (frame_t, frame_t1, reference_pil, target_pil, gt_ab) in enumerate(dataloader):
            frame_t = frame_t.to(args.device)
            frame_t1 = frame_t1.to(args.device)
            gt_ab = gt_ab.to(args.device)

            # Reset memory
            if batch_idx % args.video_length == 0:
                system.reset_memory()

            # Forward
            fused_ab = system(frame_t, frame_t1, reference_pil[0], target_pil[0])

            # Loss
            loss, loss_dict = criterion(fused_ab, gt_ab)

            # Accumulate
            for key in val_losses.keys():
                if key in loss_dict:
                    val_losses[key] += loss_dict[key]

    # Average
    num_batches = len(dataloader)
    for key in val_losses.keys():
        val_losses[key] /= num_batches

    return val_losses


def main():
    parser = argparse.ArgumentParser(description='Train Fusion UNet')

    # Paths
    parser.add_argument('--memflow_path', type=str, required=True,
                        help='Path to MemFlow repository')
    parser.add_argument('--swintexco_path', type=str, required=True,
                        help='Path to SwinSingle repository')
    parser.add_argument('--memflow_ckpt', type=str, required=True,
                        help='Path to MemFlow checkpoint')
    parser.add_argument('--swintexco_ckpt', type=str, required=True,
                        help='Path to SwinTExCo checkpoint directory')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of training videos')

    # Training
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (only 1 supported for now)')
    parser.add_argument('--accumulation_steps', type=int, default=8,
                        help='Gradient accumulation steps (effective batch size = batch_size * accumulation_steps)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--video_length', type=int, default=100,
                        help='Assumed video length for memory reset')

    # Optimization
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')

    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='fusion/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*80)
    print(" Fusion Training".center(80))
    print("="*80)

    # Initialize system
    print("\nInitializing Fusion System...")
    system = FusionSystem(
        memflow_path=args.memflow_path,
        swintexco_path=args.swintexco_path,
        memflow_ckpt=args.memflow_ckpt,
        swintexco_ckpt=args.swintexco_ckpt,
        fusion_net=SimpleFusionNet(),  # Use real UNet
        device=args.device
    )

    # Loss
    criterion = FusionLoss(
        lambda_l1=1.0,
        lambda_perceptual=0.05,
        lambda_contextual=0.1,
        lambda_temporal=0.5,
        use_temporal=True,
        device=args.device
    )

    # Optimizer (only Fusion UNet parameters)
    optimizer = torch.optim.AdamW(
        system.parameters(),  # Only returns Fusion UNet params
        lr=args.lr,
        weight_decay=1e-4
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # Dataset
    print("\nLoading dataset...")
    train_dataset = FusionDataset(
        data_root=args.data_root,
        target_size=(224, 224)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Mixed precision scaler
    scaler = GradScaler(enabled=args.use_amp)

    # Training loop
    print("\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Accumulation steps: {args.accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    print()

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Train
        train_losses = train_epoch(
            system, train_loader, criterion, optimizer, scaler, epoch, args
        )

        # Print
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_losses['total']:.4f}")
        print(f"    - L1: {train_losses['l1']:.4f}")
        print(f"    - Perceptual: {train_losses['perceptual']:.4f}")
        print(f"    - Contextual: {train_losses['contextual']:.4f}")
        if train_losses['temporal'] > 0:
            print(f"    - Temporal: {train_losses['temporal']:.4f}")

        # Learning rate
        scheduler.step()
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        if epoch % args.save_freq == 0 or train_losses['total'] < best_loss:
            is_best = train_losses['total'] < best_loss
            best_loss = min(best_loss, train_losses['total'])

            checkpoint = {
                'epoch': epoch,
                'fusion_unet': system.fusion_unet.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_loss': best_loss,
                'train_losses': train_losses
            }

            save_path = os.path.join(args.save_dir, f'fusion_epoch_{epoch}.pth')
            torch.save(checkpoint, save_path)
            print(f"  ✅ Checkpoint saved: {save_path}")

            if is_best:
                best_path = os.path.join(args.save_dir, 'fusion_best.pth')
                torch.save(checkpoint, best_path)
                print(f"  ⭐ Best model saved: {best_path}")

    print("\n" + "="*80)
    print(" Training completed!".center(80))
    print("="*80)


if __name__ == '__main__':
    main()
