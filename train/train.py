"""
Fusion Training Script

Sequential training mode: processes 4-frame sequences with MemFlow memory accumulation.

Usage:
    # Train from scratch
    python train/train.py \
        --memflow_path MemFlow \
        --swintexco_path SwinSingle \
        --memflow_ckpt MemFlow/ckpt/memflow_colorization.pth \
        --swintexco_ckpt SwinSingle/ckpt/epoch_1 \
        --dataset /path/to/dataset1,/path/to/dataset2 \
        --imagenet /path/to/imagenet \
        --batch_size 2 \
        --epochs 50

    # Resume training from checkpoint
    python train/train.py \
        --memflow_path MemFlow \
        --swintexco_path SwinSingle \
        --memflow_ckpt MemFlow/ckpt/memflow_colorization.pth \
        --swintexco_ckpt SwinSingle/ckpt/epoch_1 \
        --dataset /path/to/dataset1,/path/to/dataset2 \
        --imagenet /path/to/imagenet \
        --batch_size 2 \
        --epochs 50 \
        --resume fusion/checkpoints/fusion_epoch_10.pth
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
from FusionNet.fusion_unet import FusionNetV1
from train.fusion_loss import FusionLoss
from train.fusion_dataset import FusionSequenceDataset, fusion_sequence_collate_fn


def train_epoch(system, dataloader, criterion, optimizer, scaler, epoch, args):
    """Train for one epoch with 4-frame sequences"""

    system.train()
    epoch_losses = {
        'total': 0.0,
        'l1': 0.0,
        'perceptual': 0.0,
        'contextual': 0.0,
        'temporal': 0.0
    }

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    total_sequences = 0

    for batch_idx, batch_data in enumerate(pbar):
        # Extract batch data from fusion_sequence_collate_fn
        frames_lab_batch = batch_data['frames_lab']  # List[List[Tensor]], batch_size x seq_len
        frames_pil_batch = batch_data['frames_pil']  # List[List[PIL.Image]]
        references_pil_batch = batch_data['references_pil']  # List[List[PIL.Image]]
        video_names = batch_data['video_names']  # List[str]

        batch_size = len(frames_lab_batch)
        sequence_losses = []

        # Process each sequence in the batch
        for seq_idx in range(batch_size):
            frames_lab = frames_lab_batch[seq_idx]  # List of 4 LAB tensors
            frames_pil = frames_pil_batch[seq_idx]  # List of 4 PIL images
            references_pil = references_pil_batch[seq_idx]  # List of 4 PIL references

            # Move LAB tensors to device
            frames_lab = [f.to(args.device) for f in frames_lab]

            # Process sequence (4 frames) with forward_sequence
            with autocast(enabled=args.use_amp):
                # Forward sequence: returns (outputs, memflow_outputs, memflow_confs) if return_memflow=True
                if criterion.use_temporal and criterion.use_adaptive_temporal:
                    outputs, memflow_outputs, memflow_confs = system.forward_sequence(
                        frames_lab, frames_pil, references_pil, return_memflow=True
                    )
                else:
                    outputs = system.forward_sequence(frames_lab, frames_pil, references_pil)
                    memflow_outputs = None
                    memflow_confs = None

                # Compute loss for each frame in the sequence
                frame_losses = []
                frame_0_loss_dict = None  # Save frame 0's loss_dict for contextual loss display
                for i, (output_lab, gt_lab) in enumerate(zip(outputs, frames_lab)):
                    # Extract AB channels for loss computation
                    output_ab = output_lab[1:3, :, :].unsqueeze(0)  # [1, 2, H, W]
                    gt_ab = gt_lab[1:3, :, :].unsqueeze(0)  # [1, 2, H, W]

                    # Prepare for Swin Contextual Loss (only needed for frame 0)
                    if i == 0:
                        output_lab_batch = output_lab.unsqueeze(0)  # [1, 3, H, W]
                        reference_lab_batch = system.swintexco.processor(references_pil[i]).unsqueeze(0).to(args.device)
                        embed_net = system.swintexco.embed_net
                    else:
                        output_lab_batch = None
                        reference_lab_batch = None
                        embed_net = None

                    # Prepare for Adaptive Temporal Loss (from frame 1 onwards)
                    if i >= 1 and memflow_outputs is not None:
                        # Previous frame outputs
                        prev_pred_ab = outputs[i-1][1:3, :, :].unsqueeze(0)  # [1, 2, H, W]
                        prev_memflow_ab = memflow_outputs[i-1][1:3, :, :].unsqueeze(0)  # [1, 2, H, W]
                        prev_memflow_conf = memflow_confs[i-1].unsqueeze(0)  # [1, 1, H, W]

                        # Current frame MemFlow outputs
                        memflow_ab = memflow_outputs[i][1:3, :, :].unsqueeze(0)  # [1, 2, H, W]
                        memflow_conf = memflow_confs[i].unsqueeze(0)  # [1, 1, H, W]
                    else:
                        prev_pred_ab = None
                        prev_memflow_ab = None
                        prev_memflow_conf = None
                        memflow_ab = None
                        memflow_conf = None

                    # Compute loss with all components
                    loss, loss_dict = criterion(
                        output_ab, gt_ab,
                        frame_idx=i,
                        pred_lab=output_lab_batch,
                        reference_lab=reference_lab_batch,
                        embed_net=embed_net,
                        # Adaptive temporal loss parameters
                        prev_pred_ab=prev_pred_ab,
                        memflow_ab=memflow_ab,
                        memflow_conf=memflow_conf,
                        prev_memflow_ab=prev_memflow_ab,
                        prev_memflow_conf=prev_memflow_conf
                    )

                    # Save frame 0's loss_dict (contains contextual loss)
                    if i == 0:
                        frame_0_loss_dict = loss_dict

                    # 🔥 NaN Detection: Check if loss is valid
                    if not torch.isfinite(loss):
                        print(f"\n⚠️  NaN/Inf detected in frame {i}!")
                        print(f"  Loss value: {loss.item()}")
                        print(f"  Loss dict: {loss_dict}")
                        print(f"  Output AB range: [{output_ab.min().item():.3f}, {output_ab.max().item():.3f}]")
                        print(f"  GT AB range: [{gt_ab.min().item():.3f}, {gt_ab.max().item():.3f}]")
                        # Skip this batch to prevent crash
                        print(f"  Skipping this batch to prevent crash...")
                        continue

                    frame_losses.append(loss)

                # Average loss over 4 frames
                seq_loss = sum(frame_losses) / len(frame_losses)
                sequence_losses.append(seq_loss)

        # Average loss over batch
        batch_loss = sum(sequence_losses) / len(sequence_losses)

        # Scale for gradient accumulation
        scaled_loss = batch_loss / args.accumulation_steps

        # Backward pass
        scaler.scale(scaled_loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % args.accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(system.parameters(), args.max_grad_norm)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Accumulate losses (use frame 0's loss_dict for contextual loss)
        epoch_losses['total'] += batch_loss.item()
        if frame_0_loss_dict:
            for key in ['l1', 'perceptual', 'contextual', 'temporal']:
                if key in frame_0_loss_dict:
                    epoch_losses[key] += frame_0_loss_dict[key]

        total_sequences += batch_size

        # Update progress bar with detailed losses
        postfix_dict = {
            'loss': f"{batch_loss.item():.4f}",
            'seqs': total_sequences
        }
        # Add individual losses to progress bar (from frame 0)
        if frame_0_loss_dict:
            if 'l1' in frame_0_loss_dict:
                postfix_dict['l1'] = f"{frame_0_loss_dict['l1']:.4f}"
            if 'perceptual' in frame_0_loss_dict:
                postfix_dict['per'] = f"{frame_0_loss_dict['perceptual']:.4f}"
            if 'contextual' in frame_0_loss_dict:
                postfix_dict['ctx'] = f"{frame_0_loss_dict['contextual']:.4f}"

        pbar.set_postfix(postfix_dict)

    # Average losses
    num_batches = len(dataloader)
    for key in epoch_losses.keys():
        epoch_losses[key] /= num_batches

    return epoch_losses


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
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset path(s): single or comma-separated (e.g., /path1,/path2,/path3)')
    parser.add_argument('--imagenet', type=str, required=True,
                        help='ImageNet path(s): single or comma-separated (e.g., /path1,/path2,/path3)')

    # Training
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (only 1 supported for now)')
    parser.add_argument('--accumulation_steps', type=int, default=8,
                        help='Gradient accumulation steps (effective batch size = batch_size * accumulation_steps)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr_swintexco', type=float, default=1e-5,
                        help='Learning rate for SwinTExCo (fine-tuning)')
    parser.add_argument('--lr_fusion', type=float, default=1e-4,
                        help='Learning rate for FusionNet (training from scratch)')
    parser.add_argument('--sequence_length', type=int, default=3,
                        help='Length of frame sequences (default: 4)')

    # Optimization
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')

    # Memory Optimization
    parser.add_argument('--target_size', type=int, default=224,
                        help='Target frame size (default: 224 for 224x224). Use 128 or 160 to save memory')
    parser.add_argument('--freeze_swintexco', action='store_true',
                        help='Freeze SwinTExCo (only train FusionNet) to save memory')
    parser.add_argument('--contextual_chunk_size', type=int, default=256,
                        help='Chunk size for Contextual Loss (default: 256). Use 64 or 128 for less memory')

    # Loss Weights
    parser.add_argument('--lambda_temporal', type=float, default=0.5,
                        help='Weight for temporal loss (default: 0.5)')
    parser.add_argument('--lambda_smooth', type=float, default=0.3,
                        help='Weight for smooth component in adaptive temporal loss (default: 0.3)')
    parser.add_argument('--use_adaptive_temporal', action='store_true', default=True,
                        help='Use adaptive temporal loss (no optical flow required)')

    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='fusion/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')

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
        fusion_net=FusionNetV1(),  # Real FusionNet UNet
        device=args.device
    )

    # Loss
    criterion = FusionLoss(
        lambda_l1=1.0,
        lambda_perceptual=0.05,
        lambda_contextual=0.015,  # 🔥 CRITICAL FIX: Reduced from 0.1 to 0.015 (SwinTExCo paper value)
        lambda_temporal=args.lambda_temporal,
        lambda_smooth=args.lambda_smooth,
        use_temporal=True,
        use_adaptive_temporal=args.use_adaptive_temporal,
        contextual_chunk_size=args.contextual_chunk_size,
        device=args.device
    )

    # Optimizer with layered learning rates
    # SwinTExCo: fine-tuning with lower LR (1e-5)
    # FusionNet: training from scratch with higher LR (1e-4)
    param_groups = system.get_parameter_groups()
    optimizer = torch.optim.AdamW([
        {'params': param_groups[0]['params'], 'lr': args.lr_swintexco, 'name': 'swintexco'},
        {'params': param_groups[1]['params'], 'lr': args.lr_fusion, 'name': 'fusion'}
    ], weight_decay=1e-4)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # Dataset
    print("\nLoading dataset...")
    train_dataset = FusionSequenceDataset(
        davis_root=args.dataset,
        imagenet_root=args.imagenet,
        sequence_length=args.sequence_length,
        target_size=(args.target_size, args.target_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=fusion_sequence_collate_fn
    )

    # Mixed precision scaler
    scaler = GradScaler(enabled=args.use_amp)

    # Resume training from checkpoint
    start_epoch = 1
    best_loss = float('inf')

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\n📂 Loading checkpoint from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=args.device)

            # Restore model states
            system.fusion_unet.load_state_dict(checkpoint['fusion_unet'])
            system.swintexco.embed_net.load_state_dict(checkpoint['swintexco_embed'])
            system.swintexco.nonlocal_net.load_state_dict(checkpoint['swintexco_nonlocal'])
            system.swintexco.colornet.load_state_dict(checkpoint['swintexco_colornet'])

            # Restore optimizer and scheduler
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

            # Restore training state
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))

            print(f"✅ Checkpoint loaded successfully!")
            print(f"   Resuming from epoch {start_epoch}")
            print(f"   Best loss so far: {best_loss:.4f}")
        else:
            print(f"⚠️  Checkpoint not found at {args.resume}")
            print(f"   Starting training from scratch...")

    # Training loop
    print("\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Start epoch: {start_epoch}")
    print(f"  Sequences: {len(train_dataset)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Accumulation steps: {args.accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"  Sequence length: {args.sequence_length}")
    print(f"  Learning rate (SwinTExCo): {args.lr_swintexco}")
    print(f"  Learning rate (FusionNet): {args.lr_fusion}")
    print(f"  Device: {args.device}")
    print()

    for epoch in range(start_epoch, args.epochs + 1):
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
                'swintexco_embed': system.swintexco.embed_net.state_dict(),
                'swintexco_nonlocal': system.swintexco.nonlocal_net.state_dict(),
                'swintexco_colornet': system.swintexco.colornet.state_dict(),
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
