"""
Analyze SwinTExCo Confidence Distribution

統計 SwinTExCo 輸出的置信度（similarity）分布情況
"""

import os
import sys
import numpy as np
import glob
from collections import defaultdict
import argparse


def analyze_scene(npy_dir):
    """分析單個場景的置信度分布"""
    npy_files = sorted(glob.glob(os.path.join(npy_dir, '*.npy')))

    if not npy_files:
        return None

    all_values = []
    frame_stats = []

    for npy_path in npy_files:
        data = np.load(npy_path)
        values = data.flatten()
        all_values.extend(values)

        # 每幀統計
        frame_stats.append({
            'file': os.path.basename(npy_path),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
        })

    all_values = np.array(all_values)

    # 場景整體統計
    scene_stats = {
        'num_frames': len(npy_files),
        'total_pixels': len(all_values),
        # 基本統計
        'mean': float(np.mean(all_values)),
        'std': float(np.std(all_values)),
        'min': float(np.min(all_values)),
        'max': float(np.max(all_values)),
        'median': float(np.median(all_values)),
        # 分位數
        'p5': float(np.percentile(all_values, 5)),
        'p10': float(np.percentile(all_values, 10)),
        'p25': float(np.percentile(all_values, 25)),
        'p75': float(np.percentile(all_values, 75)),
        'p90': float(np.percentile(all_values, 90)),
        'p95': float(np.percentile(all_values, 95)),
        # 區間占比
        'ratio_0.0_0.2': float(np.mean((all_values >= 0.0) & (all_values < 0.2))),
        'ratio_0.2_0.4': float(np.mean((all_values >= 0.2) & (all_values < 0.4))),
        'ratio_0.4_0.6': float(np.mean((all_values >= 0.4) & (all_values < 0.6))),
        'ratio_0.6_0.8': float(np.mean((all_values >= 0.6) & (all_values < 0.8))),
        'ratio_0.8_1.0': float(np.mean((all_values >= 0.8) & (all_values <= 1.0))),
        # 幀統計
        'frame_stats': frame_stats,
    }

    return scene_stats


def print_scene_stats(scene_name, stats):
    """打印場景統計"""
    print(f"\n{'='*60}")
    print(f"場景: {scene_name}")
    print(f"{'='*60}")
    print(f"幀數: {stats['num_frames']}, 總像素: {stats['total_pixels']:,}")
    print()
    print(f"基本統計:")
    print(f"  Mean:   {stats['mean']:.4f}")
    print(f"  Std:    {stats['std']:.4f}")
    print(f"  Min:    {stats['min']:.4f}")
    print(f"  Max:    {stats['max']:.4f}")
    print(f"  Median: {stats['median']:.4f}")
    print()
    print(f"分位數:")
    print(f"  P5:  {stats['p5']:.4f}  |  P95: {stats['p95']:.4f}")
    print(f"  P10: {stats['p10']:.4f}  |  P90: {stats['p90']:.4f}")
    print(f"  P25: {stats['p25']:.4f}  |  P75: {stats['p75']:.4f}")
    print()
    print(f"區間占比:")
    print(f"  [0.0, 0.2): {stats['ratio_0.0_0.2']*100:5.1f}%")
    print(f"  [0.2, 0.4): {stats['ratio_0.2_0.4']*100:5.1f}%")
    print(f"  [0.4, 0.6): {stats['ratio_0.4_0.6']*100:5.1f}%")
    print(f"  [0.6, 0.8): {stats['ratio_0.6_0.8']*100:5.1f}%")
    print(f"  [0.8, 1.0]: {stats['ratio_0.8_1.0']*100:5.1f}%")


def print_histogram(all_values, bins=20):
    """打印文字直方圖"""
    hist, bin_edges = np.histogram(all_values, bins=bins, range=(0, 1))
    max_count = max(hist)
    bar_width = 40

    print(f"\n直方圖 (總計 {len(all_values):,} 像素):")
    print("-" * 60)

    for i in range(len(hist)):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        count = hist[i]
        bar_len = int(count / max_count * bar_width) if max_count > 0 else 0
        bar = '█' * bar_len
        pct = count / len(all_values) * 100
        print(f"[{left:.2f}-{right:.2f}] {bar:<{bar_width}} {pct:5.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Analyze confidence distribution')
    parser.add_argument('--npy_dir', type=str, required=True,
                        help='Path to directory containing .npy files (or parent directory with scene subdirs)')
    parser.add_argument('--type', type=str, default='similarity',
                        choices=['confidence', 'similarity'],
                        help='Type of data to analyze (default: similarity)')
    parser.add_argument('--histogram', action='store_true',
                        help='Show histogram')
    parser.add_argument('--bins', type=int, default=20,
                        help='Number of histogram bins (default: 20)')

    args = parser.parse_args()

    print("="*60)
    print(f" SwinTExCo {args.type.capitalize()} 分布分析")
    print("="*60)

    # 檢查是單個場景還是多個場景
    npy_subdir = f"{args.type}_npy"

    # 查找所有場景
    scenes = []

    # 情況1: 直接是 npy 目錄
    if glob.glob(os.path.join(args.npy_dir, '*.npy')):
        scenes.append(('直接目錄', args.npy_dir))

    # 情況2: 包含場景子目錄
    for item in sorted(os.listdir(args.npy_dir)):
        item_path = os.path.join(args.npy_dir, item)
        if os.path.isdir(item_path):
            # 檢查是否有 similarity_npy 或 confidence_npy 子目錄
            npy_path = os.path.join(item_path, npy_subdir)
            if os.path.isdir(npy_path):
                scenes.append((item, npy_path))
            # 或者直接包含 npy 文件
            elif glob.glob(os.path.join(item_path, '*.npy')):
                scenes.append((item, item_path))

    if not scenes:
        print(f"❌ 找不到 .npy 文件在 {args.npy_dir}")
        return

    print(f"\n找到 {len(scenes)} 個場景")

    # 收集所有場景的數據
    all_scene_stats = {}
    all_values_global = []

    for scene_name, npy_path in scenes:
        stats = analyze_scene(npy_path)
        if stats:
            all_scene_stats[scene_name] = stats
            print_scene_stats(scene_name, stats)

            # 收集全局數據
            for npy_file in sorted(glob.glob(os.path.join(npy_path, '*.npy'))):
                data = np.load(npy_file)
                all_values_global.extend(data.flatten())

    # 全局統計
    if len(all_scene_stats) > 1:
        all_values_global = np.array(all_values_global)
        print(f"\n{'='*60}")
        print(f"全局統計 (所有場景)")
        print(f"{'='*60}")
        print(f"總場景: {len(all_scene_stats)}")
        print(f"總像素: {len(all_values_global):,}")
        print(f"Mean:   {np.mean(all_values_global):.4f}")
        print(f"Std:    {np.std(all_values_global):.4f}")
        print(f"Min:    {np.min(all_values_global):.4f}")
        print(f"Max:    {np.max(all_values_global):.4f}")
        print(f"P5-P95: [{np.percentile(all_values_global, 5):.4f}, {np.percentile(all_values_global, 95):.4f}]")

        if args.histogram:
            print_histogram(all_values_global, bins=args.bins)

    elif len(all_scene_stats) == 1 and args.histogram:
        scene_name = list(all_scene_stats.keys())[0]
        npy_path = scenes[0][1]
        all_values = []
        for npy_file in sorted(glob.glob(os.path.join(npy_path, '*.npy'))):
            data = np.load(npy_file)
            all_values.extend(data.flatten())
        print_histogram(np.array(all_values), bins=args.bins)


if __name__ == '__main__':
    main()
