#!/usr/bin/env python3
"""
視頻上色綜合評估腳本
支持計算: SSIM, PSNR, LPIPS, CDC, FID
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from scipy import stats
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import torch
import lpips
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from concurrent.futures import ThreadPoolExecutor

# ============ CDC 相關函數 ============

def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * stats.entropy(p, M) + 0.5 * stats.entropy(q, M)


def compute_JS_bgr_from_arrays(img_bgr_arrays, dilation=1):
    if len(img_bgr_arrays) == 0:
        return [], [], []

    hist_b_list, hist_g_list, hist_r_list = [], [], []

    for img_bgr in img_bgr_arrays:
        H, W = img_bgr.shape[:2]
        hist_b_list.append(cv2.calcHist([img_bgr], [0], None, [256], [0, 256]) / (H * W))
        hist_g_list.append(cv2.calcHist([img_bgr], [1], None, [256], [0, 256]) / (H * W))
        hist_r_list.append(cv2.calcHist([img_bgr], [2], None, [256], [0, 256]) / (H * W))

    JS_b_list, JS_g_list, JS_r_list = [], [], []

    for i in range(len(hist_b_list)):
        if i + dilation > len(hist_b_list) - 1:
            break
        JS_b_list.append(JS_divergence(hist_b_list[i], hist_b_list[i + dilation]))
        JS_g_list.append(JS_divergence(hist_g_list[i], hist_g_list[i + dilation]))
        JS_r_list.append(JS_divergence(hist_r_list[i], hist_r_list[i + dilation]))

    return JS_b_list, JS_g_list, JS_r_list


def calc_cdc_from_arrays(img_bgr_arrays, dilation=[1, 2, 4], weight=[1/3, 1/3, 1/3]):
    mean_b, mean_g, mean_r = 0, 0, 0

    for d, w in zip(dilation, weight):
        JS_b_list, JS_g_list, JS_r_list = compute_JS_bgr_from_arrays(img_bgr_arrays, d)
        if len(JS_b_list) == 0:
            continue
        mean_b += w * np.mean(JS_b_list)
        mean_g += w * np.mean(JS_g_list)
        mean_r += w * np.mean(JS_r_list)

    return np.mean([mean_b, mean_g, mean_r])


# ============ 對齊尺寸工具 ============

def align_to_gt(pred_np, gt_np):
    if pred_np.shape[:2] != gt_np.shape[:2]:
        h, w = gt_np.shape[:2]
        pred_np = cv2.resize(pred_np, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return pred_np


# ============ 逐幀質量指標 ============

def calc_ssim(pred_image, gt_image):
    pred_np = np.array(pred_image.convert('RGB')).astype(np.float32)
    gt_np   = np.array(gt_image.convert('RGB')).astype(np.float32)
    pred_np = align_to_gt(pred_np, gt_np)
    return structural_similarity(pred_np, gt_np, channel_axis=2, data_range=255.)


def calc_psnr(pred_image, gt_image):
    pred_np = np.array(pred_image.convert('RGB')).astype(np.float32)
    gt_np   = np.array(gt_image.convert('RGB')).astype(np.float32)
    pred_np = align_to_gt(pred_np, gt_np)
    return peak_signal_noise_ratio(gt_np, pred_np, data_range=255.)


class LPIPS_Calculator:
    def __init__(self, device='cuda'):
        self.loss_fn = lpips.LPIPS(net='vgg', spatial=True).to(device)
        self.device = device

    def compare_batch(self, imgs_pred, imgs_gt, batch_size=4):
        """批次計算 LPIPS；spatial=True 顯存佔用高，batch_size 預設 4"""
        pred_arrays, gt_arrays = [], []
        for pred_img, gt_img in zip(imgs_pred, imgs_gt):
            pred_np = np.array(pred_img)
            gt_np   = np.array(gt_img)
            pred_np = align_to_gt(pred_np, gt_np)
            pred_arrays.append(pred_np)
            gt_arrays.append(gt_np)

        uniform_size = len(set(a.shape for a in gt_arrays)) == 1

        results = []
        for start in range(0, len(pred_arrays), batch_size):
            p_batch = pred_arrays[start:start + batch_size]
            g_batch = gt_arrays[start:start + batch_size]

            if uniform_size:
                bp = torch.stack([
                    torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)
                    for a in p_batch]).to(self.device)
                bg = torch.stack([
                    torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)
                    for a in g_batch]).to(self.device)
                with torch.no_grad():
                    dist = self.loss_fn.forward(bp, bg)  # [N, 1, H, W]
                results.extend(dist.mean(dim=[1, 2, 3]).cpu().tolist())
            else:
                for p_np, g_np in zip(p_batch, g_batch):
                    pt = torch.from_numpy(p_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    gt = torch.from_numpy(g_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        dist = self.loss_fn.forward(pt, gt)
                    results.append(dist.mean().item())

            torch.cuda.empty_cache()

        return results


class FID_Calculator:
    def __init__(self, device='cuda'):
        self.device = device
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx]).to(device)
        self.model.eval()

    def get_activations_batch(self, images, batch_size=32):
        tensors = [
            torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)
            for img in images
        ]
        uniform_size = len(set(t.shape for t in tensors)) == 1

        act_list = []
        for start in range(0, len(tensors), batch_size):
            chunk = tensors[start:start + batch_size]
            if uniform_size:
                batch = torch.stack(chunk).to(self.device)
                with torch.no_grad():
                    pred = self.model(batch)[0]
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = torch.nn.functional.adaptive_avg_pool2d(pred, (1, 1))
                act_list.append(pred.squeeze(-1).squeeze(-1).cpu().numpy())
            else:
                for t in chunk:
                    single = t.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        pred = self.model(single)[0]
                    if pred.size(2) != 1 or pred.size(3) != 1:
                        pred = torch.nn.functional.adaptive_avg_pool2d(pred, (1, 1))
                    act_list.append(pred.squeeze(-1).squeeze(-1).cpu().numpy())
            torch.cuda.empty_cache()

        return np.concatenate(act_list, axis=0) if uniform_size else np.array(act_list)

    def calculate_activation_statistics(self, images):
        activations = self.get_activations_batch(images)
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def calculate_fid(self, images_pred, images_gt):
        images_pred_aligned = []
        for pred_img, gt_img in zip(images_pred, images_gt):
            pred_np = np.array(pred_img)
            gt_np   = np.array(gt_img)
            pred_np = align_to_gt(pred_np, gt_np)
            images_pred_aligned.append(Image.fromarray(pred_np))

        mu_pred, sigma_pred = self.calculate_activation_statistics(images_pred_aligned)
        mu_gt,   sigma_gt   = self.calculate_activation_statistics(images_gt)
        return calculate_frechet_distance(mu_pred, sigma_pred, mu_gt, sigma_gt)


# ============ 主評估函數 ============

def load_image_pair(args):
    img_name, pred_dir, gt_dir = args
    pred_img = Image.open(os.path.join(pred_dir, img_name)).convert('RGB')
    gt_img   = Image.open(os.path.join(gt_dir,   img_name)).convert('RGB')
    return pred_img, gt_img


def evaluate_scene(pred_dir, gt_dir, lpips_calc, fid_calc, num_workers=8, lpips_batch_size=4):
    pred_images = sorted([f for f in os.listdir(pred_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    gt_images   = sorted([f for f in os.listdir(gt_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    if len(pred_images) == 0 or len(gt_images) == 0:
        return None

    common_images = sorted(set(pred_images) & set(gt_images))
    if len(common_images) == 0:
        print(f"  ⚠️ 警告: 找不到對應的圖片")
        return None

    # 多線程並行載入圖片
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        loaded = list(ex.map(load_image_pair,
                             [(name, pred_dir, gt_dir) for name in common_images]))

    pred_images_pil = [pair[0] for pair in loaded]
    gt_images_pil   = [pair[1] for pair in loaded]

    # SSIM / PSNR 並行（CPU bound）
    def compute_ssim_psnr(pair):
        return calc_ssim(pair[0], pair[1]), calc_psnr(pair[0], pair[1])

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        sp_results = list(ex.map(compute_ssim_psnr, zip(pred_images_pil, gt_images_pil)))

    ssim_list = [r[0] for r in sp_results]
    psnr_list = [r[1] for r in sp_results]

    # LPIPS 批次 GPU 推理
    lpips_list = lpips_calc.compare_batch(pred_images_pil, gt_images_pil,
                                          batch_size=lpips_batch_size)

    # CDC：複用已載入圖片，PIL RGB → BGR（與原始 cv2.imread 一致）
    pred_bgr_arrays = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in pred_images_pil]
    cdc_value = calc_cdc_from_arrays(pred_bgr_arrays)

    # FID 批次 GPU 推理
    fid_value = fid_calc.calculate_fid(pred_images_pil, gt_images_pil)

    return {
        'num_frames': len(common_images),
        'ssim_mean':  np.mean(ssim_list),
        'ssim_std':   np.std(ssim_list),
        'psnr_mean':  np.mean(psnr_list),
        'psnr_std':   np.std(psnr_list),
        'lpips_mean': np.mean(lpips_list),
        'lpips_std':  np.std(lpips_list),
        'cdc':        cdc_value,
        'fid':        fid_value,
    }


def main():
    parser = argparse.ArgumentParser(description='視頻上色綜合評估')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='預測結果根目錄（包含多個場景子資料夾）')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='GT根目錄（包含多個場景子資料夾）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='計算設備 (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='圖片載入與 CPU 計算的線程數')
    parser.add_argument('--lpips_batch_size', type=int, default=4,
                        help='LPIPS 每批幀數，顯存不足時調小，預設 4')
    args = parser.parse_args()

    if not os.path.exists(args.pred_dir):
        print(f"錯誤: 預測目錄 {args.pred_dir} 不存在"); return
    if not os.path.exists(args.gt_dir):
        print(f"錯誤: GT目錄 {args.gt_dir} 不存在"); return

    print("=" * 70)
    print(" 視頻上色綜合評估")
    print("=" * 70)
    print(f"預測目錄:       {args.pred_dir}")
    print(f"GT目錄:         {args.gt_dir}")
    print(f"計算設備:       {args.device}")
    print(f"線程數:         {args.num_workers}")
    print(f"LPIPS批次大小:  {args.lpips_batch_size}")
    print("=" * 70)

    print("\n初始化 LPIPS 計算器...")
    lpips_calc = LPIPS_Calculator(device=args.device)
    print("初始化 FID 計算器...")
    fid_calc = FID_Calculator(device=args.device)

    pred_scenes = sorted([d for d in os.listdir(args.pred_dir)
                          if os.path.isdir(os.path.join(args.pred_dir, d))])

    if len(pred_scenes) == 0:
        print("錯誤: 預測目錄中找不到場景子資料夾"); return

    print(f"\n找到 {len(pred_scenes)} 個場景\n")

    all_results = {}
    for scene_name in tqdm(pred_scenes, desc="評估場景"):
        pred_scene_dir = os.path.join(args.pred_dir, scene_name)
        gt_scene_dir   = os.path.join(args.gt_dir,   scene_name)

        if not os.path.exists(gt_scene_dir):
            print(f"\n⚠️ 警告: GT中找不到場景 {scene_name}，跳過")
            continue

        results = evaluate_scene(pred_scene_dir, gt_scene_dir, lpips_calc, fid_calc,
                                 num_workers=args.num_workers,
                                 lpips_batch_size=args.lpips_batch_size)
        if results is not None:
            all_results[scene_name] = results

    if len(all_results) == 0:
        print("\n錯誤: 沒有成功評估任何場景"); return

    all_ssim  = [r['ssim_mean']  for r in all_results.values()]
    all_psnr  = [r['psnr_mean']  for r in all_results.values()]
    all_lpips = [r['lpips_mean'] for r in all_results.values()]
    all_cdc   = [r['cdc']        for r in all_results.values()]
    all_fid   = [r['fid']        for r in all_results.values()]

    print("\n" + "=" * 70)
    print(" 評估結果")
    print("=" * 70)
    print(f"\n{'場景名稱':<20} {'SSIM':<12} {'PSNR':<12} {'LPIPS':<12} {'CDC':<12} {'FID':<12}")
    print("-" * 80)
    for scene_name, r in all_results.items():
        print(f"{scene_name:<20} "
              f"{r['ssim_mean']:<12.4f} "
              f"{r['psnr_mean']:<12.2f} "
              f"{r['lpips_mean']:<12.4f} "
              f"{r['cdc']:<12.6f} "
              f"{r['fid']:<12.2f}")

    print("\n" + "=" * 80)
    print(" 總體統計")
    print("=" * 80)
    print(f"總場景數: {len(all_results)}")
    print(f"\nSSIM:  均值={np.mean(all_ssim):.4f}, 標準差={np.std(all_ssim):.4f}")
    print(f"PSNR:  均值={np.mean(all_psnr):.2f} dB, 標準差={np.std(all_psnr):.2f} dB")
    print(f"LPIPS: 均值={np.mean(all_lpips):.4f}, 標準差={np.std(all_lpips):.4f}")
    print(f"CDC:   均值={np.mean(all_cdc):.6f}, 標準差={np.std(all_cdc):.6f}")
    print(f"FID:   均值={np.mean(all_fid):.2f}, 標準差={np.std(all_fid):.2f}")

    print("\n指標說明:")
    print("  - SSIM:  越接近 1 越好 (結構相似度)")
    print("  - PSNR:  越高越好 (峰值信噪比)")
    print("  - LPIPS: 越小越好 (感知相似度)")
    print("  - CDC:   越小越好 (顏色一致性)")
    print("  - FID:   越小越好 (生成質量)")
    print("=" * 80)


if __name__ == '__main__':
    main()

