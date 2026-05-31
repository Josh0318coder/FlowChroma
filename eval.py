#!/usr/bin/env python3
"""
eval_all_checkpoints.py

One-command evaluation of all trained FlowChroma checkpoints in a directory.

For each .pth file found:
  1. Swap weights into the already-initialized FusionSystem  (MemFlow/backbone loaded ONCE)
  2. Run inference on --input_dirs  →  output_root/<ckpt_stem>/
  3. Compute SSIM / PSNR / LPIPS / CDC / FID  against --gt_dir
  4. Collect results and print a ranked summary table

Results are also saved to  <output_root>/eval_summary.csv

Usage:
    python eval_all_checkpoints.py \
        --ckpt_dir  fusion/checkpoints/run_A \
        --memflow_path   MemFlow \
        --swintexco_path SwinSingle \
        --memflow_ckpt   MemFlow/ckpt/memflow_colorization.pth \
        --input_dirs /data/test_videos \
        --gt_dir     /data/test_gt \
        --output_root /tmp/flowchroma_eval \
        [--target_size 224 224] \
        [--pattern "fusion_epoch_*.pth"] \
        [--device cuda] \
        [--no_cleanup] \
        [--skip_existing] \
        [--lpips_batch_size 4] \
        [--num_workers 8]
"""

import argparse
import csv
import glob
import os
import re
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '.')

from train.fusion_system import FusionSystem
from FusionNet.fusion_unet import FusionNetV1


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _epoch_sort_key(path: str):
    """Sort by the first integer in the filename (epoch), then lexicographically."""
    m = re.search(r'(\d+)', os.path.basename(path))
    return (int(m.group(1)), os.path.basename(path)) if m else (float('inf'), os.path.basename(path))


def _load_ckpt_weights(system: FusionSystem, ckpt_path: str, device: str):
    """
    Swap only the trainable weights into an already-running FusionSystem.
    MemFlow stays untouched (it is always frozen).

    Mirrors inference_test.load_checkpoint exactly:
      - fusion_unet loaded independently
      - SwinTExCo: all three sub-modules loaded together under 'swintexco_embed' guard

    Returns (epoch, train_loss) metadata from the checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if 'fusion_unet' in ckpt:
        system.fusion_unet.load_state_dict(ckpt['fusion_unet'])

    # Mirror original: load all three SwinTExCo components together
    if 'swintexco_embed' in ckpt:
        system.swintexco.embed_net.load_state_dict(ckpt['swintexco_embed'])
        system.swintexco.nonlocal_net.load_state_dict(ckpt['swintexco_nonlocal'])
        system.swintexco.colornet.load_state_dict(ckpt['swintexco_colornet'])

    # Restore eval mode (load_state_dict does not change training flags)
    system.eval()

    epoch      = ckpt.get('epoch', '?')
    train_loss = ckpt.get('best_loss', float('nan'))
    return epoch, train_loss


def _aggregate_scenes(scene_results: dict) -> dict:
    """Average per-scene metrics into a single dict."""
    def _mean(key):
        return float(np.mean([r[key] for r in scene_results.values()]))
    return {
        'num_scenes': len(scene_results),
        'ssim':  _mean('ssim_mean'),
        'psnr':  _mean('psnr_mean'),
        'lpips': _mean('lpips_mean'),
        'cdc':   _mean('cdc'),
        'fid':   _mean('fid'),
    }


def _print_summary(all_results: list):
    W = 100
    print("\n" + "=" * W)
    print(" Summary Table".center(W))
    print("=" * W)
    hdr = (f"{'Checkpoint':<38} {'Ep':>4}  "
           f"{'SSIM':>8} {'PSNR(dB)':>9} {'LPIPS':>8} {'CDC':>11} {'FID':>8}")
    print(hdr)
    print("-" * W)
    for r in all_results:
        ep_str = str(r['epoch']) if r['epoch'] != '?' else '?'
        print(f"{r['ckpt']:<38} {ep_str:>4}  "
              f"{r['ssim']:>8.4f} {r['psnr']:>9.2f} {r['lpips']:>8.4f} "
              f"{r['cdc']:>11.6f} {r['fid']:>8.2f}")

    print("\n" + "=" * W)
    print(" Best per Metric".center(W))
    print("=" * W)
    for metric, label, higher_better in [
        ('ssim',  'SSIM',  True),
        ('psnr',  'PSNR',  True),
        ('lpips', 'LPIPS', False),
        ('cdc',   'CDC',   False),
        ('fid',   'FID',   False),
    ]:
        best = (max if higher_better else min)(all_results, key=lambda r: r[metric])
        arrow = '↑' if higher_better else '↓'
        print(f"  Best {label} {arrow}:  {best['ckpt']}   ({best[metric]:.4f})")
    print("=" * W)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate all FlowChroma checkpoints in one command',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Checkpoints ──
    parser.add_argument('--ckpt_dir', required=True,
                        help='Directory containing .pth checkpoint files')
    parser.add_argument('--pattern', default='*.pth',
                        help='Glob pattern to select checkpoints')

    # ── Model paths ──
    parser.add_argument('--memflow_path',   required=True, help='Path to MemFlow repo')
    parser.add_argument('--swintexco_path', required=True, help='Path to SwinSingle repo')
    parser.add_argument('--memflow_ckpt',   required=True, help='MemFlow checkpoint path')
    # swintexco_ckpt is intentionally omitted: every fusion checkpoint already embeds
    # swintexco_embed / nonlocal / colornet weights, which overwrite whatever _load_swintexco
    # would load.  Passing swintexco_ckpt=None lets FusionSystem use the pretrained SwinV2
    # backbone for embed_net (still needed), while nonlocal/colornet start random and are
    # immediately replaced by the fusion checkpoint weights.

    # ── Data ──
    parser.add_argument('--input_dirs', required=True,
                        help='Comma-separated inference input dirs (video scene folders)')
    parser.add_argument('--gt_dir', required=True,
                        help='Ground-truth root dir (same scene structure as input_dirs)')
    parser.add_argument('--output_root', required=True,
                        help='Root dir for per-checkpoint inference outputs')

    # ── Processing ──
    parser.add_argument('--target_size', type=int, nargs=2, default=[224, 224],
                        metavar=('H', 'W'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Threads for SSIM/PSNR and image loading')
    parser.add_argument('--lpips_batch_size', type=int, default=4,
                        help='Batch size for LPIPS GPU inference (reduce if OOM)')

    # ── Behaviour ──
    parser.add_argument('--no_cleanup', action='store_true',
                        help='Keep inference images on disk after evaluation')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip a checkpoint if its output dir already exists')
    parser.add_argument('--results_dir', default=None,
                        help='Where to save eval_summary.csv (default: --output_root)')

    args = parser.parse_args()

    # ── Find checkpoints ──
    ckpt_paths = sorted(
        glob.glob(os.path.join(args.ckpt_dir, args.pattern)),
        key=_epoch_sort_key,
    )
    if not ckpt_paths:
        print(f"❌  No checkpoints found in  {args.ckpt_dir}  matching  '{args.pattern}'")
        sys.exit(1)

    print("=" * 70)
    print(f"  FlowChroma Batch Evaluation  —  {len(ckpt_paths)} checkpoint(s)")
    print("=" * 70)
    for i, p in enumerate(ckpt_paths, 1):
        print(f"  [{i:2d}]  {os.path.basename(p)}")
    print()

    # ── Init model ONCE ──
    print("Initializing FusionSystem (MemFlow + SwinTExCo backbone)...")
    system = FusionSystem(
        memflow_path=args.memflow_path,
        swintexco_path=args.swintexco_path,
        memflow_ckpt=args.memflow_ckpt,
        swintexco_ckpt=None,   # always overwritten by fusion checkpoint weights
        fusion_net=FusionNetV1(),
        device=args.device,
    )
    system.eval()

    # ── Init eval calculators ONCE ──
    from evaluate_test import LPIPS_Calculator, FID_Calculator, evaluate_scene

    print("\nInitializing LPIPS calculator (VGG)...")
    lpips_calc = LPIPS_Calculator(device=args.device)
    print("Initializing FID  calculator (InceptionV3)...")
    fid_calc = FID_Calculator(device=args.device)

    # ── Import inference function ──
    from inference_test import process_datasets

    input_dirs  = [d.strip() for d in args.input_dirs.split(',')]
    target_size = tuple(args.target_size)

    results_dir = args.results_dir or args.output_root
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    # ──────────────────────────────────────────
    # Main loop: one checkpoint at a time
    # ──────────────────────────────────────────
    for ckpt_idx, ckpt_path in enumerate(ckpt_paths, 1):
        ckpt_stem  = Path(ckpt_path).stem
        output_dir = os.path.join(args.output_root, ckpt_stem)

        print("\n" + "=" * 70)
        print(f"  [{ckpt_idx}/{len(ckpt_paths)}]  {ckpt_stem}")
        print("=" * 70)

        # ── Optional skip ──
        if args.skip_existing and os.path.exists(output_dir):
            print(f"  ⏭  Skipping — output already exists: {output_dir}")
            continue

        try:
            # 1. Swap weights (MemFlow untouched)
            epoch, train_loss = _load_ckpt_weights(system, ckpt_path, args.device)
            loss_str = f"{train_loss:.4f}" if not np.isnan(train_loss) else "n/a"
            print(f"  ✅  Weights loaded  |  epoch={epoch}  train_loss={loss_str}")

            # 2. Inference
            print(f"  🎬  Running inference  →  {output_dir}")
            process_datasets(
                system=system,
                input_dirs=input_dirs,
                output_dir=output_dir,
                target_size=target_size,
            )

            # 3. Evaluate per scene
            print(f"  📊  Computing metrics...")
            pred_scenes = sorted([
                d for d in os.listdir(output_dir)
                if os.path.isdir(os.path.join(output_dir, d))
            ])

            if not pred_scenes:
                print(f"  ❌  No scene sub-dirs found in {output_dir}")
                continue

            scene_results = {}
            for scene_name in pred_scenes:
                pred_scene_dir = os.path.join(output_dir, scene_name)
                gt_scene_dir   = os.path.join(args.gt_dir, scene_name)

                if not os.path.exists(gt_scene_dir):
                    print(f"    ⚠  GT not found for scene: {scene_name}  (skipped)")
                    continue

                r = evaluate_scene(
                    pred_scene_dir, gt_scene_dir,
                    lpips_calc, fid_calc,
                    num_workers=args.num_workers,
                    lpips_batch_size=args.lpips_batch_size,
                )
                if r is not None:
                    scene_results[scene_name] = r
                    print(f"    {scene_name:<20}  "
                          f"SSIM={r['ssim_mean']:.4f}  PSNR={r['psnr_mean']:.2f}  "
                          f"LPIPS={r['lpips_mean']:.4f}  CDC={r['cdc']:.6f}  FID={r['fid']:.2f}")

            if not scene_results:
                print(f"  ❌  No valid scenes evaluated — skipping this checkpoint")
                continue

            # 4. Aggregate
            agg = _aggregate_scenes(scene_results)
            row = {
                'ckpt':       ckpt_stem,
                'epoch':      epoch,
                'train_loss': train_loss if not np.isnan(train_loss) else '',
                **agg,
            }
            all_results.append(row)

            print(f"\n  ► Aggregate:  "
                  f"SSIM={agg['ssim']:.4f}  PSNR={agg['psnr']:.2f} dB  "
                  f"LPIPS={agg['lpips']:.4f}  CDC={agg['cdc']:.6f}  FID={agg['fid']:.2f}  "
                  f"({agg['num_scenes']} scenes)")

            # 5. Cleanup (optional)
            if not args.no_cleanup:
                shutil.rmtree(output_dir)
                print(f"  🗑  Inference outputs removed: {output_dir}")

        except Exception as e:
            print(f"  ❌  Error processing {ckpt_stem}:\n      {e}")
            traceback.print_exc()
            continue

    # ── Final report ──
    if not all_results:
        print("\n❌  No results collected.")
        sys.exit(1)

    _print_summary(all_results)

    # ── Save CSV ──
    csv_path = os.path.join(results_dir, 'eval_summary.csv')
    fieldnames = ['ckpt', 'epoch', 'train_loss', 'num_scenes',
                  'ssim', 'psnr', 'lpips', 'cdc', 'fid']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, '') for k in fieldnames})

    print(f"\n📄  Summary CSV saved to: {csv_path}")


if __name__ == '__main__':
    main()
