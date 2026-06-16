from __future__ import print_function
import argparse
import os
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform_lib
from PIL import Image
from tqdm import tqdm
import time

import lib.TestTransforms as transforms
from models.ColorVidNet import ColorVidNet
from models.FrameColor import frame_colorization
from models.NonlocalNet import VGG19_pytorch, WarpNet
from utils.util import (batch_lab2rgb_transpose_mc, folder2vid, mkdir_if_not,
                        save_frames, tensor_lab2rgb, uncenter_l)
from utils.util_distortion import CenterPad, Normalize, RGB2Lab, ToTensor

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_float32_matmul_precision('high')
device = torch.device("cuda")

def colorize_video(opt, input_path, reference_file, output_path, nonlocal_net, colornet, vggnet):
    wls_filter_on = True
    lambda_value = 500
    sigma_color = 4
    mkdir_if_not(output_path)

    filenames = sorted([f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                       key=lambda f: int("".join(filter(str.isdigit, f) or -1)))

    if not filenames: return

    print(f"\n--- [正在處理影片資料夾: {os.path.basename(input_path)}] ---")

    input_size = tuple(opt.image_size)
    transform = transforms.Compose(
        [CenterPad(input_size), transform_lib.CenterCrop(input_size), RGB2Lab(), ToTensor(), Normalize()]
    )

    frame_ref = Image.open(reference_file).convert('RGB')
    IB_lab_large_cpu = transform(frame_ref).unsqueeze(0).cpu()
    IB_lab_cpu = torch.nn.functional.interpolate(IB_lab_large_cpu, scale_factor=0.5, mode="bilinear", align_corners=False)

    I_reference_lab = IB_lab_cpu.to(device)
    I_reference_l_cpu = IB_lab_cpu[:, 0:1, :, :]
    I_reference_ab_cpu = IB_lab_cpu[:, 1:3, :, :]

    with torch.no_grad():
        ref_lab_combined = torch.cat((uncenter_l(I_reference_l_cpu), I_reference_ab_cpu), dim=1)
        I_reference_rgb_cpu = tensor_lab2rgb(ref_lab_combined)
        features_B = vggnet(I_reference_rgb_cpu.to(device), ["r12", "r22", "r32", "r42", "r52"], preprocess=True)

    I_last_lab_predict = None

    # ── 計時用變數 ──────────────────────────────
    frame_times = []
    WARMUP = 5  # 前 5 幀不計入（GPU 暖機）
    # ────────────────────────────────────────────

    for index, frame_name in enumerate(tqdm(filenames, desc="5090 運算中")):
        img_raw = Image.open(os.path.join(input_path, frame_name))
        frame1 = img_raw.convert('L').convert('RGB')

        IA_lab_large_cpu = transform(frame1).unsqueeze(0).cpu()
        IA_lab_cpu = torch.nn.functional.interpolate(IA_lab_large_cpu, scale_factor=0.5, mode="bilinear", align_corners=False)
        IA_lab = IA_lab_cpu.to(device)
        IA_l = IA_lab[:, 0:1, :, :]

        if I_last_lab_predict is None:
            I_last_lab_predict = torch.zeros_like(IA_lab).to(device)

        with torch.no_grad():
            # ── 核心推論計時開始 ──────────────────
            torch.cuda.synchronize()
            t_start = time.perf_counter()

            I_current_ab_predict, _, _ = frame_colorization(
                IA_lab, I_reference_lab, I_last_lab_predict,
                features_B, vggnet, nonlocal_net, colornet,
                feature_noise=0, temperature=1e-10,
            )

            torch.cuda.synchronize()
            t_end = time.perf_counter()
            # ── 核心推論計時結束 ──────────────────

            if index >= WARMUP:
                frame_times.append((t_end - t_start) * 1000)  # 毫秒

            if index % 50 == 0:
                ab_mean = I_current_ab_predict.abs().mean().item()
                print(f" [偵測] Frame {index} | 顏色強度: {ab_mean:.4f}")

            I_last_lab_predict = torch.cat((IA_l, I_current_ab_predict), dim=1)

        curr_bs_l_cpu = IA_lab_large_cpu[:, 0:1, :, :]
        curr_predict_cpu = torch.nn.functional.interpolate(I_current_ab_predict.detach().cpu(),
                                                           scale_factor=2, mode="bilinear", align_corners=False)
        curr_predict_cpu = torch.clamp(curr_predict_cpu, -128, 128)

        if wls_filter_on:
            l_for_wls = uncenter_l(curr_bs_l_cpu).clamp(0, 100).numpy()
            guide_image = (l_for_wls[0, 0, :, :] * 2.55).astype(np.uint8)
            wls_filter = cv2.ximgproc.createFastGlobalSmootherFilter(guide_image, lambda_value, sigma_color)
            curr_predict_a = wls_filter.filter(curr_predict_cpu[0, 0, :, :].numpy())
            curr_predict_b = wls_filter.filter(curr_predict_cpu[0, 1, :, :].numpy())
            curr_predict_filter = torch.cat((torch.from_numpy(curr_predict_a).unsqueeze(0).unsqueeze(0),
                                            torch.from_numpy(curr_predict_b).unsqueeze(0).unsqueeze(0)), dim=1)
            IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l_cpu, curr_predict_filter)
        else:
            IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l_cpu, curr_predict_cpu)

        save_frames(IA_predict_rgb, output_path, index)

    folder2vid(image_folder=output_path, output_dir=output_path, filename=f"{os.path.basename(input_path)}_res.avi")

    # ── 輸出計時結果 ─────────────────────────────
    if frame_times:
        avg_ms = sum(frame_times) / len(frame_times)
        print(f"\n{'='*40}")
        print(f"  Deep Exemplar 推論速度統計")
        print(f"  計時幀數:   {len(frame_times)} 幀（跳過前 {WARMUP} 幀暖機）")
        print(f"  平均時間:   {avg_ms:.2f} ms / 幀")
        print(f"  FPS:        {1000/avg_ms:.2f}")
        print(f"  最快:       {min(frame_times):.2f} ms")
        print(f"  最慢:       {max(frame_times):.2f} ms")
        print(f"{'='*40}\n")
    # ─────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, nargs=2, default=[480, 832])
    parser.add_argument("--clip_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./results")
    opt = parser.parse_args()

    print("📦 正在 CPU 載入權重...")
    nonlocal_net = WarpNet(1).cpu()
    colornet = ColorVidNet(7).cpu()
    vggnet = VGG19_pytorch().cpu()

    vggnet.load_state_dict(torch.load("data/vgg19_conv.pth", map_location="cpu", weights_only=True))
    nonlocal_net.load_state_dict(torch.load("checkpoints/video_moredata_l1/nonlocal_net_iter_76000.pth", map_location="cpu", weights_only=True))
    colornet.load_state_dict(torch.load("checkpoints/video_moredata_l1/colornet_iter_76000.pth", map_location="cpu", weights_only=True))

    print("🚀 啟動 5090 穩定運行模式...")
    vggnet = vggnet.to(device).eval()
    nonlocal_net = nonlocal_net.to(device).eval()
    colornet = colornet.to(device).eval()
    print("✅ 5090 模式已就緒")

    subfolders = sorted([f.path for f in os.scandir(opt.clip_path) if f.is_dir()])
    for subfolder in subfolders:
        imgs = sorted([f for f in os.listdir(subfolder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if imgs:
            colorize_video(opt, subfolder, os.path.join(subfolder, imgs[0]),
                           os.path.join(opt.output_path, os.path.basename(subfolder)),
                           nonlocal_net, colornet, vggnet)
