#!/usr/bin/env python3
"""
視頻上色綜合評估腳本
支持計算: SSIM, PSNR, LPIPS, CDC
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

# ============ CDC 相關函數 ============

def JS_divergence(p, q):
    """計算 Jensen-Shannon 散度"""
    M = (p + q) / 2
    return 0.5 * stats.entropy(p, M) + 0.5 * stats.entropy(q, M)


def compute_JS_bgr(input_dir, dilation=1):
    """計算指定間隔的幀之間的JS散度"""
    input_img_list = sorted(os.listdir(input_dir))
    input_img_list = [f for f in input_img_list 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if len(input_img_list) == 0:
        return [], [], []
    
    hist_b_list, hist_g_list, hist_r_list = [], [], []
    
    for img_name in input_img_list:
        img_path = os.path.join(input_dir, img_name)
        img_in = cv2.imread(img_path)
        if img_in is None:
            continue
        H, W, C = img_in.shape
        
        hist_b = cv2.calcHist([img_in], [0], None, [256], [0, 256]) / (H * W)
        hist_g = cv2.calcHist([img_in], [1], None, [256], [0, 256]) / (H * W)
        hist_r = cv2.calcHist([img_in], [2], None, [256], [0, 256]) / (H * W)
        
        hist_b_list.append(hist_b)
        hist_g_list.append(hist_g)
        hist_r_list.append(hist_r)
    
    JS_b_list, JS_g_list, JS_r_list = [], [], []
    
    for i in range(len(hist_b_list)):
        if i + dilation > len(hist_b_list) - 1:
            break
        JS_b_list.append(JS_divergence(hist_b_list[i], hist_b_list[i + dilation]))
        JS_g_list.append(JS_divergence(hist_g_list[i], hist_g_list[i + dilation]))
        JS_r_list.append(JS_divergence(hist_r_list[i], hist_r_list[i + dilation]))
    
    return JS_b_list, JS_g_list, JS_r_list


def calc_cdc(vid_folder, dilation=[1, 2, 4], weight=[1/3, 1/3, 1/3]):
    """計算視頻資料夾的 CDC 指標"""
    mean_b, mean_g, mean_r = 0, 0, 0
    
    for d, w in zip(dilation, weight):
        JS_b_list, JS_g_list, JS_r_list = compute_JS_bgr(vid_folder, d)
        if len(JS_b_list) == 0:
            continue
        mean_b += w * np.mean(JS_b_list)
        mean_g += w * np.mean(JS_g_list)
        mean_r += w * np.mean(JS_r_list)
    
    return np.mean([mean_b, mean_g, mean_r])


# ============ 逐幀質量指標 ============

def calc_ssim(pred_image, gt_image):
    """計算 SSIM"""
    pred_image = np.array(pred_image.convert('RGB')).astype(np.float32)
    gt_image = np.array(gt_image.convert('RGB')).astype(np.float32)
    
    # Resize到224x224
    pred_image = cv2.resize(pred_image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    gt_image = cv2.resize(gt_image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    
    ssim = structural_similarity(pred_image, gt_image, channel_axis=2, data_range=255.)
    return ssim


def calc_psnr(pred_image, gt_image):
    """計算 PSNR"""
    pred_image = np.array(pred_image.convert('RGB')).astype(np.float32)
    gt_image = np.array(gt_image.convert('RGB')).astype(np.float32)
    
    # Resize到224x224
    pred_image = cv2.resize(pred_image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    gt_image = cv2.resize(gt_image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    
    psnr = peak_signal_noise_ratio(gt_image, pred_image, data_range=255.)
    return psnr


class LPIPS_Calculator:
    """LPIPS 計算器 (與原始metrics.py一致)"""
    def __init__(self, device='cuda'):
        self.loss_fn = lpips.LPIPS(net='vgg', spatial=True)  # 加上spatial=True
        self.loss_fn = self.loss_fn.to(device)
        self.device = device
    
    def compare(self, img_pred, img_gt):
        """計算 LPIPS (與原始compare_lpips一致)"""
        # 先轉為numpy array
        img_pred_np = np.array(img_pred)
        img_gt_np = np.array(img_gt)
        
        # Resize到224x224
        img_pred_np = cv2.resize(img_pred_np, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        img_gt_np = cv2.resize(img_gt_np, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        
        img_pred = torch.from_numpy(img_pred_np.astype(np.float32) / 255.0)
        img_gt = torch.from_numpy(img_gt_np.astype(np.float32) / 255.0)
        
        if img_pred.ndim == 3:
            img_pred = img_pred.permute(2, 0, 1).unsqueeze(0)
            img_gt = img_gt.permute(2, 0, 1).unsqueeze(0)
        
        img_pred = img_pred.to(self.device)
        img_gt = img_gt.to(self.device)
        
        dist = self.loss_fn.forward(img_pred, img_gt)
        return dist.mean().item()


class FID_Calculator:
    """FID 計算器 (與原始metrics.py一致)"""
    def __init__(self, device='cuda'):
        self.device = device
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx]).to(device)
        self.model.eval()
    
    def get_activations(self, batch):
        """提取Inception特徵"""
        with torch.no_grad():
            pred = self.model(batch)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
        return pred
    
    def calculate_activation_statistics(self, images):
        """計算特徵的均值和協方差
        images: list of PIL Images
        """
        # 收集所有特徵
        act_list = []
        
        for img in images:
            # 轉換為tensor並resize到224x224
            img_np = np.array(img)
            img_np = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            img_tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0)
            
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            
            img_tensor = img_tensor.to(self.device)
            
            # 提取特徵
            activations = self.get_activations(img_tensor)
            activations = activations.cpu().numpy().squeeze()
            act_list.append(activations)
        
        # 計算統計
        activations = np.array(act_list)
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        
        return mu, sigma
    
    def calculate_fid(self, images_pred, images_gt):
        """計算FID分數
        images_pred: list of PIL Images (預測)
        images_gt: list of PIL Images (GT)
        """
        mu_pred, sigma_pred = self.calculate_activation_statistics(images_pred)
        mu_gt, sigma_gt = self.calculate_activation_statistics(images_gt)
        
        fid_score = calculate_frechet_distance(mu_pred, sigma_pred, mu_gt, sigma_gt)
        return fid_score


# ============ 主評估函數 ============

def evaluate_scene(pred_dir, gt_dir, lpips_calc, fid_calc):
    """評估單個場景"""
    # 獲取圖片列表
    pred_images = sorted([f for f in os.listdir(pred_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    gt_images = sorted([f for f in os.listdir(gt_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    if len(pred_images) == 0 or len(gt_images) == 0:
        return None
    
    # 確保對應
    common_images = set(pred_images) & set(gt_images)
    if len(common_images) == 0:
        print(f"  ⚠️ 警告: 找不到對應的圖片")
        return None
    
    image_list = sorted(list(common_images))
    
    ssim_list = []
    psnr_list = []
    lpips_list = []
    
    # 收集所有圖片用於FID計算
    pred_images_pil = []
    gt_images_pil = []
    
    # 逐幀計算
    for img_name in image_list:
        pred_path = os.path.join(pred_dir, img_name)
        gt_path = os.path.join(gt_dir, img_name)
        
        pred_img = Image.open(pred_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        
        ssim_list.append(calc_ssim(pred_img, gt_img))
        psnr_list.append(calc_psnr(pred_img, gt_img))
        lpips_list.append(lpips_calc.compare(pred_img, gt_img))
        
        # 收集圖片用於FID
        pred_images_pil.append(pred_img)
        gt_images_pil.append(gt_img)
    
    # 計算 CDC
    cdc_value = calc_cdc(pred_dir)
    
    # 計算 FID
    fid_value = fid_calc.calculate_fid(pred_images_pil, gt_images_pil)
    
    results = {
        'num_frames': len(image_list),
        'ssim_mean': np.mean(ssim_list),
        'ssim_std': np.std(ssim_list),
        'psnr_mean': np.mean(psnr_list),
        'psnr_std': np.std(psnr_list),
        'lpips_mean': np.mean(lpips_list),
        'lpips_std': np.std(lpips_list),
        'cdc': cdc_value,
        'fid': fid_value
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='視頻上色綜合評估')
    parser.add_argument('--pred_dir', type=str, required=True,
                       help='預測結果根目錄（包含多個場景子資料夾）')
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='GT根目錄（包含多個場景子資料夾）')
    parser.add_argument('--device', type=str, default='cuda',
                       help='計算設備 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 檢查路徑
    if not os.path.exists(args.pred_dir):
        print(f"錯誤: 預測目錄 {args.pred_dir} 不存在")
        return
    if not os.path.exists(args.gt_dir):
        print(f"錯誤: GT目錄 {args.gt_dir} 不存在")
        return
    
    print("=" * 70)
    print("🎯 視頻上色綜合評估")
    print("=" * 70)
    print(f"預測目錄: {args.pred_dir}")
    print(f"GT目錄: {args.gt_dir}")
    print(f"計算設備: {args.device}")
    print("=" * 70)
    
    # 初始化 LPIPS
    print("\n初始化 LPIPS 計算器...")
    lpips_calc = LPIPS_Calculator(device=args.device)
    
    # 初始化 FID
    print("初始化 FID 計算器...")
    fid_calc = FID_Calculator(device=args.device)
    
    # 查找所有場景
    pred_scenes = sorted([d for d in os.listdir(args.pred_dir) 
                         if os.path.isdir(os.path.join(args.pred_dir, d))])
    
    if len(pred_scenes) == 0:
        print("錯誤: 預測目錄中找不到場景子資料夾")
        return
    
    print(f"\n找到 {len(pred_scenes)} 個場景\n")
    
    # 評估每個場景
    all_results = {}
    
    for scene_name in tqdm(pred_scenes, desc="評估場景"):
        pred_scene_dir = os.path.join(args.pred_dir, scene_name)
        gt_scene_dir = os.path.join(args.gt_dir, scene_name)
        
        if not os.path.exists(gt_scene_dir):
            print(f"\n⚠️ 警告: GT中找不到場景 {scene_name}，跳過")
            continue
        
        results = evaluate_scene(pred_scene_dir, gt_scene_dir, lpips_calc, fid_calc)
        
        if results is not None:
            all_results[scene_name] = results
    
    if len(all_results) == 0:
        print("\n錯誤: 沒有成功評估任何場景")
        return
    
    # 計算總體統計
    all_ssim = [r['ssim_mean'] for r in all_results.values()]
    all_psnr = [r['psnr_mean'] for r in all_results.values()]
    all_lpips = [r['lpips_mean'] for r in all_results.values()]
    all_cdc = [r['cdc'] for r in all_results.values()]
    all_fid = [r['fid'] for r in all_results.values()]
    
    # 打印結果
    print("\n" + "=" * 70)
    print("📊 評估結果")
    print("=" * 70)
    
    print("\n各場景詳細結果:")
    print(f"{'場景名稱':<20} {'SSIM':<12} {'PSNR':<12} {'LPIPS':<12} {'CDC':<12} {'FID':<12}")
    print("-" * 80)
    for scene_name, results in all_results.items():
        print(f"{scene_name:<20} "
              f"{results['ssim_mean']:<12.4f} "
              f"{results['psnr_mean']:<12.2f} "
              f"{results['lpips_mean']:<12.4f} "
              f"{results['cdc']:<12.6f} "
              f"{results['fid']:<12.2f}")
    
    print("\n" + "=" * 80)
    print("📈 總體統計")
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



