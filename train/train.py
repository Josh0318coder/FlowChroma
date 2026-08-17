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
from train.fusion_loss import (
    FusionLoss, discriminator_loss_fn, generator_loss_fn,
    discriminator_loss_fn_loggan, generator_loss_fn_loggan,
)
from train.fusion_dataset import FusionSequenceDataset, fusion_sequence_collate_fn
from train.discriminator import Discriminator


def train_epoch(system, dataloader, criterion, optimizer, scaler, epoch, args, discriminator=None, optimizer_d=None, scheduler_d=None, discriminator_temp=None, optimizer_d_temp=None, discriminator_diff=None, optimizer_d_diff=None):
    """Train for one epoch with 4-frame sequences"""

    system.train()
    if discriminator is not None:
        discriminator.train()
    if discriminator_temp is not None:
        discriminator_temp.train()
    if discriminator_diff is not None:
        discriminator_diff.train()

    epoch_losses = {
        'total': 0.0,
        'l1_ref': 0.0,
        'l1_gt': 0.0,
        'perceptual': 0.0,
        'contextual': 0.0,
        'smooth': 0.0,
        'temporal': 0.0,
        'cdc': 0.0,
        'discriminator': 0.0,
        'generator': 0.0,
        'discriminator_temp': 0.0,
        'generator_temp': 0.0,
        'discriminator_diff': 0.0,
        'generator_diff': 0.0,
    }

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    total_sequences = 0

    step_count = 0
    step_losses = {k: 0.0 for k in epoch_losses}
    step_batch_count = 0
    latest_d_loss = 1.0       # Track current dis loss for D skip
    latest_d_temp_loss = 1.0  # Track current temporal dis loss for D skip
    latest_d_diff_loss = 1.0  # Track current diff dis loss for D skip

    for batch_idx, batch_data in enumerate(pbar):
        # Extract batch data from fusion_sequence_collate_fn
        frames_lab_batch = batch_data['frames_lab']  # List[List[Tensor]], batch_size x seq_len
        frames_swintexco_lab_batch = batch_data['frames_swintexco_lab']  # List[List[Tensor]]
        reference_swintexco_lab_batch = batch_data['reference_swintexco_lab']  # List[Tensor]
        video_names = batch_data['video_names']  # List[str]

        batch_size = len(frames_lab_batch)
        seq_len = len(frames_lab_batch[0])

        # Stack all sequences into true batches: [B, 3, H, W] per frame
        frames_lab_stacked = [
            torch.stack([frames_lab_batch[b][i] for b in range(batch_size)]).to(args.device, non_blocking=True)
            for i in range(seq_len)
        ]
        frames_swintexco_lab_stacked = [
            torch.stack([frames_swintexco_lab_batch[b][i] for b in range(batch_size)])
            for i in range(seq_len)
        ]
        reference_swintexco_lab_stacked = torch.stack(
            [reference_swintexco_lab_batch[b] for b in range(batch_size)]
        )

        # Process all sequences in the batch simultaneously through MemFlow/SwinTExCo/FusionNet
        with autocast(enabled=args.use_amp, dtype=torch.bfloat16):
            outputs, memflow_outputs, memflow_confs, swintexco_outputs, swintexco_confs, memflow_outputs_small, memflow_confs_small = system.forward_sequence(
                frames_lab_stacked, frames_swintexco_lab_stacked, reference_swintexco_lab_stacked,
                return_memflow=True
            )

            # Compute loss for each frame (outputs are [B, 3, H, W] tensors)
            frame_losses = []
            frame_0_loss_dict = None  # Save frame 0's loss_dict for contextual loss display
            frame_1_loss_dict = None  # Save frame 1's loss_dict for GAN losses
            latest_loss_dict = None  # Save latest frame's loss_dict for temporal loss display

            for i, (output_lab, gt_lab) in enumerate(zip(outputs, frames_lab_stacked)):
                # output_lab: [B, 3, H, W], gt_lab: [B, 3, H, W]
                output_ab = output_lab[:, 1:3, :, :]  # [B, 2, H, W]
                gt_ab = gt_lab[:, 1:3, :, :]  # [B, 2, H, W]

                # Prepare for Swin Contextual Loss (only needed for frame 0)
                if i == 0:
                    reference_lab_batch = reference_swintexco_lab_stacked.to(args.device)  # [B, 3, H, W]
                    embed_net = system.swintexco.embed_net  # contextual + perceptual loss
                else:
                    reference_lab_batch = None
                    embed_net = None

                # Prepare for Adaptive Temporal Loss (from frame 1 onwards)
                if i >= 1 and memflow_outputs is not None:
                    prev_pred_ab = outputs[i-1][:, 1:3, :, :]  # [B, 2, H, W]
                    prev_memflow_ab = memflow_outputs_small[i-1][:, 1:3, :, :]  # [B, 2, 56, 56]
                    prev_memflow_conf = memflow_confs_small[i-1]  # [B, 1, 56, 56]
                    memflow_ab = memflow_outputs_small[i][:, 1:3, :, :]  # [B, 2, 56, 56]
                    memflow_conf = memflow_confs_small[i]  # [B, 1, 56, 56]
                else:
                    prev_pred_ab = None
                    prev_memflow_ab = None
                    prev_memflow_conf = None
                    memflow_ab = None
                    memflow_conf = None

                # Prepare SwinTExCo outputs for weighted L1 loss (upsample to full resolution)
                H, W = output_ab.shape[2], output_ab.shape[3]
                swintexco_ab_small = swintexco_outputs[i]  # [B, 2, ...]
                swintexco_conf_small = swintexco_confs[i]  # [B, 1, ...]
                swintexco_ab = nn.functional.interpolate(
                    swintexco_ab_small, size=(H, W), mode='bilinear', align_corners=True
                )
                swintexco_conf = nn.functional.interpolate(
                    swintexco_conf_small, size=(H, W), mode='bilinear', align_corners=True
                )

                # Compute loss with all components (losses average over batch B internally)
                loss, loss_dict = criterion(
                    output_ab, gt_ab,
                    frame_idx=i,
                    pred_lab=output_lab,
                    gt_lab=gt_lab,
                    reference_lab=reference_lab_batch,
                    embed_net=embed_net,
                    prev_pred_ab=prev_pred_ab,
                    memflow_ab=memflow_ab,
                    memflow_conf=memflow_conf,
                    prev_memflow_ab=prev_memflow_ab,
                    prev_memflow_conf=prev_memflow_conf,
                    swintexco_ab=swintexco_ab,
                    swintexco_conf=swintexco_conf,
                    prev_pred_lab_full=outputs[i-1] if i >= 1 else None
                )

                # GAN Loss — single-frame 3ch (computed on frame 1)
                discriminator_loss = torch.tensor(0.0, device=args.device)
                generator_loss = torch.tensor(0.0, device=args.device)

                if i == 1 and discriminator is not None and args.weight_gan > 0:
                    fake_data_lab = torch.cat((output_lab[:, 0:1, :, :], output_ab), dim=1)  # [B, 3, H, W]
                    real_data_lab = torch.cat((gt_lab[:, 0:1, :, :], gt_ab), dim=1)  # [B, 3, H, W]
                    with autocast(enabled=False):
                        fake_3ch = fake_data_lab.float()
                        real_3ch = real_data_lab.float()

                        discriminator_loss = discriminator_loss_fn(real_3ch, fake_3ch, discriminator)
                        (discriminator_loss / args.accumulation_steps).backward()

                        if epoch > args.epoch_train_discriminator:
                            for p in discriminator.parameters():
                                p.requires_grad_(False)
                            generator_loss = generator_loss_fn(
                                real_3ch, fake_3ch, discriminator, args.weight_gan, args.device
                            )
                            scaler.scale(generator_loss / args.accumulation_steps).backward(retain_graph=True)
                            for p in discriminator.parameters():
                                p.requires_grad_(True)

                # Temporal GAN Loss — frame pairs (method A: 6ch concatenated frames)
                discriminator_temp_loss = torch.tensor(0.0, device=args.device)
                generator_temp_loss = torch.tensor(0.0, device=args.device)

                if i >= 1 and discriminator_temp is not None and args.weight_gan_temp > 0:
                    fake_6ch = torch.cat([outputs[i-1], output_lab], dim=1).float()  # [B, 6, H, W]
                    real_6ch = torch.cat([frames_lab_stacked[i-1], gt_lab], dim=1).float()  # [B, 6, H, W]

                    with autocast(enabled=False):
                        discriminator_temp_loss = discriminator_loss_fn(real_6ch, fake_6ch, discriminator_temp)
                        (discriminator_temp_loss / args.accumulation_steps).backward()

                        if epoch > args.epoch_train_discriminator:
                            for p in discriminator_temp.parameters():
                                p.requires_grad_(False)
                            generator_temp_loss = generator_loss_fn(
                                real_6ch, fake_6ch, discriminator_temp, args.weight_gan_temp, args.device
                            )
                            scaler.scale(generator_temp_loss / args.accumulation_steps).backward(retain_graph=True)
                            for p in discriminator_temp.parameters():
                                p.requires_grad_(True)

                # Temporal GAN Loss — method D: diff-based (3ch: AB_diff + L_diff)
                discriminator_diff_loss = torch.tensor(0.0, device=args.device)
                generator_diff_loss = torch.tensor(0.0, device=args.device)

                if i >= 1 and discriminator_diff is not None and args.weight_gan_diff > 0:
                    l_diff = frames_lab_stacked[i-1][:, 0:1] - gt_lab[:, 0:1]  # [B, 1, H, W]
                    fake_ab_diff = outputs[i-1][:, 1:3] - output_lab[:, 1:3]  # [B, 2, H, W]
                    fake_3ch_diff = torch.cat([fake_ab_diff, l_diff], dim=1).float()  # [B, 3, H, W]
                    real_ab_diff = frames_lab_stacked[i-1][:, 1:3] - gt_lab[:, 1:3]  # [B, 2, H, W]
                    real_3ch_diff = torch.cat([real_ab_diff, l_diff], dim=1).float()  # [B, 3, H, W]

                    # Select adversarial loss form: RaLSGAN (default) or log-GAN (thesis Eq. 3.36/3.37)
                    d_loss_fn = discriminator_loss_fn_loggan if args.gan_type == 'loggan' else discriminator_loss_fn
                    g_loss_fn = generator_loss_fn_loggan if args.gan_type == 'loggan' else generator_loss_fn

                    with autocast(enabled=False):
                        discriminator_diff_loss = d_loss_fn(
                            real_3ch_diff, fake_3ch_diff, discriminator_diff
                        )
                        (discriminator_diff_loss / args.accumulation_steps).backward()

                        if epoch > args.epoch_train_discriminator:
                            for p in discriminator_diff.parameters():
                                p.requires_grad_(False)
                            generator_diff_loss = g_loss_fn(
                                real_3ch_diff, fake_3ch_diff,
                                discriminator_diff, args.weight_gan_diff, args.device
                            )
                            scaler.scale(generator_diff_loss / args.accumulation_steps).backward(retain_graph=True)
                            for p in discriminator_diff.parameters():
                                p.requires_grad_(True)

                # Store loss_dicts per frame; GAN losses live at frame 1 (single-frame D)
                loss_dict['discriminator'] = discriminator_loss.item()
                loss_dict['generator'] = generator_loss.item()
                loss_dict['discriminator_temp'] = discriminator_temp_loss.item()
                loss_dict['generator_temp'] = generator_temp_loss.item()
                loss_dict['discriminator_diff'] = discriminator_diff_loss.item()
                loss_dict['generator_diff'] = generator_diff_loss.item()
                if i == 0:
                    frame_0_loss_dict = loss_dict
                elif i == 1:
                    frame_1_loss_dict = loss_dict

                latest_loss_dict = loss_dict

                #  NaN Detection: Check if loss is valid
                if not torch.isfinite(loss):
                    print(f"\n⚠️  NaN/Inf detected in frame {i}!")
                    print(f"  Loss value: {loss.item()}")
                    print(f"  Loss dict: {loss_dict}")
                    print(f"  Output AB range: [{output_ab.min().item():.3f}, {output_ab.max().item():.3f}]")
                    print(f"  GT AB range: [{gt_ab.min().item():.3f}, {gt_ab.max().item():.3f}]")
                    print(f"  Skipping this batch to prevent crash...")
                    continue

                frame_losses.append(loss)

        # Update latest_d_loss for D skip decision
        if frame_1_loss_dict and 'discriminator' in frame_1_loss_dict:
            latest_d_loss = frame_1_loss_dict['discriminator']
        if frame_1_loss_dict and 'discriminator_temp' in frame_1_loss_dict:
            latest_d_temp_loss = frame_1_loss_dict['discriminator_temp']
        if frame_1_loss_dict and 'discriminator_diff' in frame_1_loss_dict:
            latest_d_diff_loss = frame_1_loss_dict['discriminator_diff']

        # Sum loss over frames (losses already averaged over batch B by criterion internals)
        batch_loss = sum(frame_losses)

        # Scale for gradient accumulation
        scaled_loss = batch_loss / args.accumulation_steps

        # Backward pass
        scaler.scale(scaled_loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % args.accumulation_steps == 0:
            # Gradient clipping (generator)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(system.parameters(), args.max_grad_norm)

            # Optimizer step (generator)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Discriminator step — skip if D is already too strong (D skip)
            # Use rolling average from step_losses for stable decision; fallback to latest batch if window just reset
            if optimizer_d is not None and args.weight_gan > 0:
                d_ref = (step_losses['discriminator'] / step_batch_count) if step_batch_count > 0 else latest_d_loss
                if d_ref > args.d_skip_threshold:
                    optimizer_d.step()
                optimizer_d.zero_grad()

            # Temporal discriminator step — skip if D is already too strong
            if optimizer_d_temp is not None and args.weight_gan_temp > 0:
                d_temp_ref = (step_losses['discriminator_temp'] / step_batch_count) if step_batch_count > 0 else latest_d_temp_loss
                if d_temp_ref > args.d_skip_threshold:
                    optimizer_d_temp.step()
                optimizer_d_temp.zero_grad()

            # Diff discriminator step — skip if D is already too strong
            if optimizer_d_diff is not None and args.weight_gan_diff > 0:
                d_diff_ref = (step_losses['discriminator_diff'] / step_batch_count) if step_batch_count > 0 else latest_d_diff_loss
                if d_diff_ref > args.d_skip_threshold:
                    optimizer_d_diff.step()
                optimizer_d_diff.zero_grad()

            # Step-level checkpoint print (every checkpoint_step optimizer steps)
            step_count += 1
            if step_count % args.checkpoint_step == 0 and step_batch_count > 0:
                print(f"\n=== Step {step_count} (Epoch {epoch}) ===")
                print(f"  total:         {step_losses['total'] / step_batch_count:.6f}")
                print(f"  l1_ref:        {step_losses['l1_ref'] / step_batch_count:.6f}")
                print(f"  l1_gt:         {step_losses['l1_gt'] / step_batch_count:.6f}")
                print(f"  perceptual:    {step_losses['perceptual'] / step_batch_count:.6f}")
                print(f"  contextual:    {step_losses['contextual'] / step_batch_count:.6f}")
                print(f"  smooth:        {step_losses['smooth'] / step_batch_count:.6f}")
                print(f"  temporal:      {step_losses['temporal'] / step_batch_count:.6f}")
                print(f"  discriminator: {step_losses['discriminator'] / step_batch_count:.6f}")
                print(f"  generator:     {step_losses['generator'] / step_batch_count:.6f}")
                if step_losses['discriminator_temp'] > 0:
                    print(f"  dis_temp:      {step_losses['discriminator_temp'] / step_batch_count:.6f}")
                if step_losses['generator_temp'] > 0:
                    print(f"  gen_temp:      {step_losses['generator_temp'] / step_batch_count:.6f}")
                if step_losses['discriminator_diff'] > 0:
                    print(f"  dis_diff:      {step_losses['discriminator_diff'] / step_batch_count:.6f}")
                if step_losses['generator_diff'] > 0:
                    print(f"  gen_diff:      {step_losses['generator_diff'] / step_batch_count:.6f}")
                if step_losses['cdc'] > 0:
                    print(f"  cdc:           {step_losses['cdc'] / step_batch_count:.6f}")
                print("=" * 40)
                step_losses = {k: 0.0 for k in step_losses}
                step_batch_count = 0

        # Accumulate losses
        epoch_losses['total'] += batch_loss.item()
        step_losses['total'] += batch_loss.item()
        if frame_0_loss_dict:
            for key in ['l1_ref', 'l1_gt', 'perceptual', 'contextual', 'smooth']:
                if key in frame_0_loss_dict:
                    epoch_losses[key] += frame_0_loss_dict[key]
                    step_losses[key] += frame_0_loss_dict[key]
        if frame_1_loss_dict:
            for key in ['l1_ref', 'l1_gt', 'temporal', 'smooth',
                        'cdc',
                        'discriminator', 'generator',
                        'discriminator_temp', 'generator_temp',
                        'discriminator_diff', 'generator_diff']:
                if key in frame_1_loss_dict:
                    epoch_losses[key] += frame_1_loss_dict[key]
                    step_losses[key] += frame_1_loss_dict[key]
        step_batch_count += 1

        total_sequences += batch_size

        # Update progress bar with detailed losses
        postfix_dict = {}

        # Add temporal loss components (align and smooth) from latest frame
        if latest_loss_dict:
            if 'align' in latest_loss_dict:
                postfix_dict['alg'] = f"{latest_loss_dict['align']:.4f}"
            if 'smooth' in latest_loss_dict:
                postfix_dict['smo'] = f"{latest_loss_dict['smooth']:.4f}"

        # Add other losses (from frame 0)
        if frame_0_loss_dict:
            if 'l1_ref' in frame_0_loss_dict:
                postfix_dict['l1r'] = f"{frame_0_loss_dict['l1_ref']:.4f}"
            if 'l1_gt' in frame_0_loss_dict:
                postfix_dict['l1g'] = f"{frame_0_loss_dict['l1_gt']:.4f}"
            if 'contextual' in frame_0_loss_dict:
                postfix_dict['ctx'] = f"{frame_0_loss_dict['contextual']:.4f}"
            if 'perceptual' in frame_0_loss_dict:
                postfix_dict['per'] = f"{frame_0_loss_dict['perceptual']:.4f}"
        # GAN losses from frame 1 (single-frame discriminator)
        if frame_1_loss_dict:
            if frame_1_loss_dict.get('discriminator', 0) > 0:
                postfix_dict['dis'] = f"{frame_1_loss_dict['discriminator']:.4f}"
            if frame_1_loss_dict.get('generator', 0) > 0:
                postfix_dict['gan'] = f"{frame_1_loss_dict['generator']:.4f}"
        # Temporal GAN losses from frame 1 (frame-pair discriminator)
        if frame_1_loss_dict:
            if frame_1_loss_dict.get('discriminator_temp', 0) > 0:
                postfix_dict['dis_t'] = f"{frame_1_loss_dict['discriminator_temp']:.4f}"
            if frame_1_loss_dict.get('generator_temp', 0) > 0:
                postfix_dict['gan_t'] = f"{frame_1_loss_dict['generator_temp']:.4f}"
            if frame_1_loss_dict.get('discriminator_diff', 0) > 0:
                postfix_dict['dis_d'] = f"{frame_1_loss_dict['discriminator_diff']:.4f}"
            if frame_1_loss_dict.get('generator_diff', 0) > 0:
                postfix_dict['gan_d'] = f"{frame_1_loss_dict['generator_diff']:.4f}"
        if latest_loss_dict and latest_loss_dict.get('cdc', 0) > 0:
            postfix_dict['cdc'] = f"{latest_loss_dict['cdc']:.4f}"

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
    parser.add_argument('--swintexco_ckpt', type=str, default=None,
                        help='Path to SwinTExCo checkpoint directory. '
                             'Not needed when --pretrain_newsingle is set (leave unset or omit)')
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
    parser.add_argument('--sequence_length', type=int, default=2,
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
    parser.add_argument('--no_memflow', action='store_true',
                        help='Ablation: replace MemFlow with downsampled previous prediction (conf=0)')
    parser.add_argument('--contextual_chunk_size', type=int, default=256,
                        help='Chunk size for Contextual Loss (default: 256). Use 64 or 128 for less memory')

    # Loss Weights
    parser.add_argument('--lambda_l1_gt', type=float, default=1.0,
                        help='Weight for GT L1 loss (default: 0.1)')
    parser.add_argument('--lambda_temporal', type=float, default=0.0,
                        help='Weight for temporal loss (default: 0.5)')
    parser.add_argument('--lambda_align', type=float, default=0.0,
                        help='Weight for align component in adaptive temporal loss (default: 1.0)')
    parser.add_argument('--lambda_smooth', type=float, default=150.0,
                        help='Weight for spatial smoothness loss (default: 0.3)')
    parser.add_argument('--lambda_cdc', type=float, default=9.0,#9.0
                        help='Weight for color consistency loss (Wasserstein-1, targets CDC metric, default: 0)')
    parser.add_argument('--use_adaptive_temporal', action='store_true', default=True,
                        help='Use adaptive temporal loss (no optical flow required)')

    # Logging
    parser.add_argument('--checkpoint_step', type=int, default=10000,
                        help='Print average loss every N optimizer steps (default: 500, following newsingle)')

    # GAN Loss (from SwinSingle)
    parser.add_argument('--weight_gan', type=float, default=0.0,
                        help='Weight for GAN loss (default: 0.015, set 0 to disable)')
    parser.add_argument('--weight_gan_temp', type=float, default=0.0,
                        help='Weight for temporal GAN loss — frame-pair discriminator (default: 0, set e.g. 0.005)')
    parser.add_argument('--weight_gan_diff', type=float, default=0.015,#0.015
                        help='Weight for diff GAN loss — method D: AB_diff+L_diff 3ch discriminator (default: 0, set e.g. 0.005)')
    parser.add_argument('--epoch_train_discriminator', type=int, default=0,
                        help='Start generator GAN loss after N epochs (default: 0)')
    parser.add_argument('--lr_discriminator', type=float, default=1e-4,
                        help='Learning rate for discriminator (default: 1e-4)')
    parser.add_argument('--reset_discriminator', action='store_true',
                        help='Reset discriminator to random init even when resuming (for testing D collapse)')
    parser.add_argument('--reset_scheduler', action='store_true',
                        help='Reset LR scheduler when resuming (use when extending training beyond original epochs)')
    parser.add_argument('--d_skip_threshold', type=float, default=0.0,
                        help='Skip D update when dis loss is below this threshold (default: 0.4)')
    parser.add_argument('--gan_type', type=str, default='ralsgan',
                        choices=['ralsgan', 'loggan'],
                        help='Adversarial loss form for the diff discriminator: '
                             'ralsgan (default, current code) or loggan (thesis Eq. 3.36/3.37)')

    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='fusion/checkpoints/test1',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--pretrain_newsingle', type=str, default=None,
                        help='Path to newsingle-8ch checkpoint dir (e.g. checkpoints/epoch_5). '
                             'Fully replaces --swintexco_ckpt as the weight source: loads '
                             'embed_net, nonlocal_net, and colornet → fusion_unet. '
                             'Do NOT pass --swintexco_ckpt when using this flag. '
                             'Overridden by --resume if both are set.')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    # Memory Occupier
    parser.add_argument('--occupy_vram', type=float, default=0.0,
                        help='Extra GPU VRAM to occupy in GB via mem_occupy.py (default: 0)')
    parser.add_argument('--occupy_ram', type=float, default=0.0,
                        help='Extra CPU RAM to occupy in GB via mem_occupy.py (default: 0)')

    # DataLoader
    parser.add_argument('--num_workers', type=int, default=12,
                        help='Number of DataLoader workers (default: 8). Use 4-8 for 16-core CPU, 12-16 for 32+ core CPU')

    args = parser.parse_args()

    # Occupy memory in-process (same PID as training)
    _occupy_holders = []
    if args.occupy_vram > 0:
        n_elem = int(args.occupy_vram * 1024 ** 3 / 4)
        _occupy_holders.append(torch.empty(n_elem, dtype=torch.float32, device=args.device))
        print(f"VRAM occupied: {args.occupy_vram:.2f} GB on {args.device}")
    if args.occupy_ram > 0:
        n_elem = int(args.occupy_ram * 1024 ** 3 / 4)
        _occupy_holders.append(torch.empty(n_elem, dtype=torch.float32, device='cpu').pin_memory())
        print(f"CPU RAM occupied: {args.occupy_ram:.2f} GB")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*80)
    print(" Fusion Training".center(80))
    print("="*80)

    # Initialize system
    # When --pretrain_newsingle is set, swintexco_ckpt is irrelevant:
    # FusionSystem will use random init, then pretrain_newsingle overwrites all weights.
    if args.pretrain_newsingle:
        args.swintexco_ckpt = None

    print("\nInitializing Fusion System...")
    system = FusionSystem(
        memflow_path=args.memflow_path,
        swintexco_path=args.swintexco_path,
        memflow_ckpt=args.memflow_ckpt,
        swintexco_ckpt=args.swintexco_ckpt,
        fusion_net=FusionNetV1(),  # Real FusionNet UNet
        device=args.device,
        freeze_swintexco=args.freeze_swintexco,
        no_memflow=args.no_memflow
    )

    # Loss
    criterion = FusionLoss(
        lambda_l1=0.0,
        lambda_l1_gt=args.lambda_l1_gt,
        lambda_perceptual=0.15,#0.15
        lambda_contextual=0.0,
        lambda_temporal=args.lambda_temporal,
        lambda_align=args.lambda_align,
        lambda_smooth=args.lambda_smooth,
        lambda_cdc=args.lambda_cdc,
        use_temporal=True,
        use_adaptive_temporal=args.use_adaptive_temporal,
        contextual_chunk_size=args.contextual_chunk_size,
        device=args.device
    )

    # Optimizer with layered learning rates
    # SwinTExCo: fine-tuning with lower LR (1e-5) (if not frozen)
    # FusionNet: training from scratch with higher LR (1e-4)
    param_groups = system.get_parameter_groups()

    if args.freeze_swintexco:
        # Only FusionNet parameters (single group)
        optimizer = torch.optim.AdamW([
            {'params': param_groups[0]['params'], 'lr': args.lr_fusion, 'name': 'fusion'}
        ], weight_decay=1e-4)
    else:
        # Both SwinTExCo and FusionNet parameters (two groups)
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

    # Discriminator (optional, only if weight_gan > 0)
    discriminator = None
    optimizer_d = None
    scheduler_d = None
    if args.weight_gan > 0:
        print("\nInitializing Discriminator (GAN training enabled)...")
        discriminator = Discriminator(in_channels=3, ndf=64).to(args.device)
        optimizer_d = torch.optim.AdamW(
            discriminator.parameters(),
            lr=args.lr_discriminator,
            betas=(0.5, 0.999),
            amsgrad=True
        )
        scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_d,
            T_max=args.epochs,
            eta_min=1e-7
        )
        print(f"  Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Temporal Discriminator — frame-pair (6ch), method A
    discriminator_temp = None
    optimizer_d_temp = None
    if args.weight_gan_temp > 0:
        print("\nInitializing Temporal Discriminator (frame-pair GAN enabled)...")
        discriminator_temp = Discriminator(in_channels=6, ndf=64).to(args.device)
        optimizer_d_temp = torch.optim.AdamW(
            discriminator_temp.parameters(),
            lr=args.lr_discriminator,
            betas=(0.5, 0.999),
            amsgrad=True
        )
        print(f"  Temporal Discriminator parameters: {sum(p.numel() for p in discriminator_temp.parameters()):,}")

    # Diff Discriminator — AB_diff + L_diff (3ch), method D
    discriminator_diff = None
    optimizer_d_diff = None
    if args.weight_gan_diff > 0:
        print("\nInitializing Diff Discriminator (method D: diff-based GAN enabled)...")
        discriminator_diff = Discriminator(in_channels=3, ndf=64).to(args.device)
        optimizer_d_diff = torch.optim.AdamW(
            discriminator_diff.parameters(),
            lr=args.lr_discriminator,
            betas=(0.5, 0.999),
            amsgrad=True
        )
        print(f"  Diff Discriminator parameters: {sum(p.numel() for p in discriminator_diff.parameters()):,}")
        print(f"  GAN weight: {args.weight_gan}")
        print(f"  Will start generator GAN loss after epoch {args.epoch_train_discriminator}")
        print(f"  D skip threshold: {args.d_skip_threshold}")

    # Dataset
    print("\nLoading dataset...")
    train_dataset = FusionSequenceDataset(
        davis_root=args.dataset,
        imagenet_root=args.imagenet,
        sequence_length=args.sequence_length,
        target_size=(args.target_size, args.target_size),
        swintexco_processor=system.swintexco.processor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=fusion_sequence_collate_fn
    )

    # Mixed precision scaler
    scaler = GradScaler(enabled=args.use_amp)

    # Initialize from pretrain checkpoint (replaces swintexco_ckpt as weight source)
    # Supports two formats:
    #   1. Single .pth file (our checkpoint dict) — pass path to the .pth file
    #   2. Directory with separate .pth files (newsingle format) — pass path to the directory
    # No optimizer/scheduler state loaded — this is fine-tuning initialization, not resume
    if args.pretrain_newsingle:
        pretrain_path = args.pretrain_newsingle
        print(f"\nInitializing from pretrain: {pretrain_path}")

        if os.path.isfile(pretrain_path):
            # Format 1: our checkpoint dict (fusion_epoch_N.pth)
            ckpt = torch.load(pretrain_path, map_location=args.device)
            system.swintexco.embed_net.load_state_dict(ckpt['swintexco_embed'])
            print(f"  ✅ swintexco.embed_net")
            system.swintexco.nonlocal_net.load_state_dict(ckpt['swintexco_nonlocal'])
            print(f"  ✅ swintexco.nonlocal_net")
            system.fusion_unet.load_state_dict(ckpt['fusion_unet'])
            print(f"  ✅ fusion_unet")
        else:
            # Format 2: newsingle directory (embed_net.pth, nonlocal_net.pth, colornet.pth)
            to_load = [
                ("embed_net.pth",    system.swintexco.embed_net,    "swintexco.embed_net"),
                ("nonlocal_net.pth", system.swintexco.nonlocal_net, "swintexco.nonlocal_net"),
                ("colornet.pth",     system.fusion_unet,             "fusion_unet (← colornet)"),
            ]
            for filename, module, label in to_load:
                path = os.path.join(pretrain_path, filename)
                if os.path.exists(path):
                    module.load_state_dict(torch.load(path, map_location=args.device))
                    print(f"  ✅ {label}")
                else:
                    print(f"  ⚠️  {label}: file not found ({path})")

    # Resume training from checkpoint
    start_epoch = 1
    best_loss = float('inf')

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\n Loading checkpoint from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=args.device)

            # Restore model states
            system.fusion_unet.load_state_dict(checkpoint['fusion_unet'])
            system.swintexco.embed_net.load_state_dict(checkpoint['swintexco_embed'])
            system.swintexco.nonlocal_net.load_state_dict(checkpoint['swintexco_nonlocal'])
            system.swintexco.colornet.load_state_dict(checkpoint['swintexco_colornet'])

            # Restore optimizer and scheduler
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.reset_scheduler:
                # Rebuild scheduler for remaining epochs (use when extending beyond original T_max)
                remaining = args.epochs - start_epoch + 1
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=remaining, eta_min=1e-6
                )
                print(f"   ⚠️  Scheduler reset: new CosineAnnealingLR T_max={remaining} (epochs {start_epoch}→{args.epochs})")
            else:
                scheduler.load_state_dict(checkpoint['scheduler'])

            # Restore discriminator if using GAN
            if discriminator is not None and 'discriminator' in checkpoint:
                if args.reset_discriminator:
                    print(f"   ⚠️  Discriminator reset to random init (--reset_discriminator)")
                else:
                    discriminator.load_state_dict(checkpoint['discriminator'])
                    if optimizer_d is not None and 'optimizer_d' in checkpoint:
                        optimizer_d.load_state_dict(checkpoint['optimizer_d'])
                    if scheduler_d is not None and 'scheduler_d' in checkpoint:
                        scheduler_d.load_state_dict(checkpoint['scheduler_d'])
                    print(f"   ✅ Discriminator state loaded")

            # Restore temporal discriminator if using temporal GAN
            if discriminator_temp is not None and 'discriminator_temp' in checkpoint:
                if args.reset_discriminator:
                    print(f"   ⚠️  Temporal Discriminator reset to random init (--reset_discriminator)")
                else:
                    discriminator_temp.load_state_dict(checkpoint['discriminator_temp'])
                    if optimizer_d_temp is not None and 'optimizer_d_temp' in checkpoint:
                        optimizer_d_temp.load_state_dict(checkpoint['optimizer_d_temp'])
                    print(f"   ✅ Temporal Discriminator state loaded")

            # Restore diff discriminator if using diff GAN
            if discriminator_diff is not None and 'discriminator_diff' in checkpoint:
                if args.reset_discriminator:
                    print(f"   ⚠️  Diff Discriminator reset to random init (--reset_discriminator)")
                else:
                    discriminator_diff.load_state_dict(checkpoint['discriminator_diff'])
                    if optimizer_d_diff is not None and 'optimizer_d_diff' in checkpoint:
                        optimizer_d_diff.load_state_dict(checkpoint['optimizer_d_diff'])
                    print(f"   ✅ Diff Discriminator state loaded")

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
    print(f"  DataLoader workers: {args.num_workers}")
    if args.no_memflow:
        print(f"  MemFlow: DISABLED (ablation - using downsampled prev prediction, conf=0)")
    if args.freeze_swintexco:
        print(f"  SwinTExCo: FROZEN (only training FusionNet)")
        print(f"  Learning rate (FusionNet): {args.lr_fusion}")
    else:
        print(f"  SwinTExCo: nonlocal_net trainable")
        print(f"  Learning rate (SwinTExCo): {args.lr_swintexco}")
        print(f"  Learning rate (FusionNet): {args.lr_fusion}")
    print(f"  Device: {args.device}")
    print()

    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_losses = train_epoch(
            system, train_loader, criterion, optimizer, scaler, epoch, args,
            discriminator=discriminator, optimizer_d=optimizer_d, scheduler_d=scheduler_d,
            discriminator_temp=discriminator_temp, optimizer_d_temp=optimizer_d_temp,
            discriminator_diff=discriminator_diff, optimizer_d_diff=optimizer_d_diff
        )

        # Print
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_losses['total']:.4f}")
        print(f"    - L1 Ref (×λ_l1):    {train_losses['l1_ref']:.4f}")
        print(f"    - L1 GT  (×λ_l1_gt): {train_losses['l1_gt']:.4f}")
        print(f"    - Perceptual: {train_losses['perceptual']:.4f}")
        print(f"    - Contextual: {train_losses['contextual']:.4f}")
        print(f"    - Smooth:     {train_losses['smooth']:.4f}")
        if train_losses['temporal'] > 0:
            print(f"    - Temporal: {train_losses['temporal']:.4f}")
        if train_losses['discriminator'] > 0:
            print(f"    - Discriminator: {train_losses['discriminator']:.4f}")
        if train_losses['generator'] > 0:
            print(f"    - Generator: {train_losses['generator']:.4f}")
        if train_losses['discriminator_temp'] > 0:
            print(f"    - Discriminator (temporal): {train_losses['discriminator_temp']:.4f}")
        if train_losses['generator_temp'] > 0:
            print(f"    - Generator (temporal):     {train_losses['generator_temp']:.4f}")
        if train_losses['discriminator_diff'] > 0:
            print(f"    - Discriminator (diff):     {train_losses['discriminator_diff']:.4f}")
        if train_losses['generator_diff'] > 0:
            print(f"    - Generator (diff):         {train_losses['generator_diff']:.4f}")

        # Learning rate
        scheduler.step()
        if scheduler_d is not None:
            scheduler_d.step()
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

            # Save discriminator if using GAN
            if discriminator is not None:
                checkpoint['discriminator'] = discriminator.state_dict()
                if optimizer_d is not None:
                    checkpoint['optimizer_d'] = optimizer_d.state_dict()
                if scheduler_d is not None:
                    checkpoint['scheduler_d'] = scheduler_d.state_dict()

            # Save temporal discriminator if using temporal GAN
            if discriminator_temp is not None:
                checkpoint['discriminator_temp'] = discriminator_temp.state_dict()
                if optimizer_d_temp is not None:
                    checkpoint['optimizer_d_temp'] = optimizer_d_temp.state_dict()

            # Save diff discriminator if using diff GAN
            if discriminator_diff is not None:
                checkpoint['discriminator_diff'] = discriminator_diff.state_dict()
                if optimizer_d_diff is not None:
                    checkpoint['optimizer_d_diff'] = optimizer_d_diff.state_dict()

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
