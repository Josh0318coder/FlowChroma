# core/datasets.py (修改後的色彩化dataloader - 支援4幀序列)

import os
import glob
import random
import numpy as np
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class VideoColorizationDataset(Dataset):
    """
    視頻上色數據集 - 4幀序列版本
    適配MemFlowNet的輸入格式
    """
    def __init__(self, 
                 video_data_root_list,
                 image_size=[384, 512],
                 min_frames=4,  # ← 改成4
                 augment=True):
        """
        Args:
            video_data_root_list: 視頻數據根目錄列表
            image_size: [H, W]
            min_frames: 最少幀數 (必須≥4)
            augment: 是否數據增強
        """
        self.video_data_root_list = video_data_root_list if isinstance(video_data_root_list, list) else [video_data_root_list]
        self.image_size = image_size
        self.min_frames = max(min_frames, 4)  # 確保至少4幀
        self.augment = augment
        
        # ImageNet標準化
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.to_tensor = transforms.ToTensor()
        
        print(f"📂 Loading dataset from {len(self.video_data_root_list)} paths...")
        for i, path in enumerate(self.video_data_root_list):
            print(f"   Path {i+1}: {path}")
            
        self.frame_sequences = self._load_frame_sequences()  # ← 改名
        print(f"📊 Found {len(self.frame_sequences)} 4-frame sequences")
        print(f"🎬 Scenes: {len(set(s['scene_id'] for s in self.frame_sequences))}")

    def _load_frame_sequences(self):
        """載入所有連續4幀序列"""
        all_sequences = []
        
        for root_idx, video_data_root in enumerate(self.video_data_root_list):
            if not os.path.exists(video_data_root):
                print(f"⚠️ Path not found: {video_data_root}")
                continue
                
            seq_count = 0
            for item in os.listdir(video_data_root):
                item_path = os.path.join(video_data_root, item)
                if os.path.isdir(item_path):
                    # 收集所有圖像
                    frames = []
                    for ext in ['*.jpg', '*.jpeg', '*.png']:
                        frames.extend(glob.glob(os.path.join(item_path, ext)))
                    
                    frames = sorted(frames)
                    if len(frames) >= self.min_frames:
                        unique_scene_id = f"path{root_idx}_{item}"
                        
                        # ← 關鍵修改: 生成連續4幀序列
                        for i in range(len(frames) - 3):  # -3 因為需要4幀
                            all_sequences.append({
                                'scene_id': unique_scene_id,
                                'frame_paths': [
                                    frames[i],
                                    frames[i + 1],
                                    frames[i + 2],
                                    frames[i + 3]
                                ],
                                'start_idx': i,
                                'total_frames': len(frames)
                            })
                            seq_count += 1
                            
            print(f"   ✅ Path {root_idx+1}: {seq_count} sequences")

        random.shuffle(all_sequences)
        return all_sequences

    def _load_and_process_image(self, path):
        """
        載入並處理單張圖像
        
        Returns:
            rgb_gray: [3, H, W] - ImageNet標準化的灰階RGB（給SwinV2）
            lab: [3, H, W] - LAB格式 (L:[0,100], ab:[-128,127])
        """
        try:
            # 載入並調整大小
            image = Image.open(path).convert('RGB')
            image = image.resize((self.image_size[1], self.image_size[0]), Image.LANCZOS)
            
            # ===== 1. 處理 RGB 灰階（給 SwinV2） =====
            image_gray = image.convert('L')
            gray_tensor = self.to_tensor(image_gray)  # [1, H, W]
            rgb_gray = gray_tensor.repeat(3, 1, 1)    # [3, H, W]
            rgb_gray = self.normalize(rgb_gray)       # ImageNet標準化
            
            # ===== 2. 處理 LAB =====
            image_np = np.array(image, dtype=np.uint8)
            lab_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB).astype(np.float32)
            
            # LAB 轉換到標準範圍
            lab_np[:, :, 0] = lab_np[:, :, 0] * 100.0 / 255.0  # L: [0,100]
            lab_np[:, :, 1] = lab_np[:, :, 1] - 128.0          # a: [-128,127]
            lab_np[:, :, 2] = lab_np[:, :, 2] - 128.0          # b: [-128,127]
            
            lab = torch.from_numpy(lab_np).permute(2, 0, 1)  # [3, H, W]
            
            return rgb_gray, lab
            
        except Exception as e:
            print(f"❌ Error loading {path}: {e}")
            # 返回默認值
            rgb_gray = torch.zeros(3, self.image_size[0], self.image_size[1])
            lab = torch.zeros(3, self.image_size[0], self.image_size[1])
            lab[0] = 50.0
            return rgb_gray, lab

    def _apply_augmentation(self, rgb_list, lab_list):
        """
        同步數據增強 - 應用到所有4幀
        
        Args:
            rgb_list: list of [3, H, W]
            lab_list: list of [3, H, W]
        """
        if not self.augment:
            return rgb_list, lab_list
        
        # 水平翻轉（同步所有幀）
        if random.random() > 0.5:
            rgb_list = [torch.flip(rgb, [-1]) for rgb in rgb_list]
            lab_list = [torch.flip(lab, [-1]) for lab in lab_list]
        
        # 亮度調整（只對LAB的L通道,所有幀同步）
        if random.random() > 0.7:
            factor = random.uniform(0.8, 1.2)
            for lab in lab_list:
                lab[0] = torch.clamp(lab[0] * factor, 0, 100)
        
        # 飽和度調整（只對LAB的ab通道,所有幀同步）
        if random.random() > 0.7:
            factor = random.uniform(0.8, 1.2)
            for lab in lab_list:
                lab[1:3] = torch.clamp(lab[1:3] * factor, -128, 127)
        
        return rgb_list, lab_list

    def __len__(self):
        return len(self.frame_sequences)

    def __getitem__(self, idx):
        seq_info = self.frame_sequences[idx]
        
        # ← 關鍵修改: 載入4幀
        rgb_list = []
        lab_list = []
        for frame_path in seq_info['frame_paths']:
            rgb, lab = self._load_and_process_image(frame_path)
            rgb_list.append(rgb)
            lab_list.append(lab)
        
        # 數據增強（同步所有幀）
        rgb_list, lab_list = self._apply_augmentation(rgb_list, lab_list)
        
        # ← 關鍵修改: Stack成序列格式
        rgb_seq = torch.stack(rgb_list, dim=0)  # [4, 3, H, W]
        lab_seq = torch.stack(lab_list, dim=0)  # [4, 3, H, W]
        
        # ===== 準備輸出 (MemFlowNet格式) =====
        return {
            'images': lab_seq,  # [4, 3, H, W] - LAB序列
            'rgb_inputs': rgb_seq,  # [4, 3, H, W] - 灰階RGB序列
            'scene_id': seq_info['scene_id']
        }


def fetch_dataloader(args):
    """
    創建DataLoader
    
    Args:
        args.data_path: 數據路徑,逗號分隔多個路徑
        args.batch_size: batch大小
        args.image_size: [H, W]
    """
    # 解析數據路徑
    data_paths = args.data_path.split(',')
    data_paths = [p.strip() for p in data_paths]
    
    # 創建數據集
    train_dataset = VideoColorizationDataset(
        video_data_root_list=data_paths,
        image_size=args.image_size,
        min_frames=4,
        augment=True
    )
    
    # 創建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    
    print(f'✅ Training with {len(train_dataset)} 4-frame sequences')
    return train_loader


