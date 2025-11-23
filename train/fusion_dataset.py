"""
Fusion Dataset

Dataset loader for online training mode.
Loads video frames on-the-fly without pre-computing MemFlow/SwinTExCo outputs.
"""

import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2


class FusionDataset(Dataset):
    """
    Online Fusion Training Dataset

    Loads video sequences and provides:
        - Consecutive frames (for MemFlow)
        - Reference frame (for SwinTExCo)
        - Ground truth LAB

    Directory structure expected:
        data_root/
            video1/
                frame_000001.jpg
                frame_000002.jpg
                ...
            video2/
                ...
    """

    def __init__(self, data_root, target_size=(224, 224), max_frames_per_video=None):
        """
        Args:
            data_root: Root directory or comma-separated directories containing video folders
                       Example: '/path1' or '/path1,/path2,/path3'
            target_size: (H, W) tuple for resizing
            max_frames_per_video: Maximum frames to use per video (None = all)
        """
        # Support comma-separated multiple paths
        if isinstance(data_root, str) and ',' in data_root:
            self.data_roots = [path.strip() for path in data_root.split(',')]
        elif isinstance(data_root, str):
            self.data_roots = [data_root]
        else:
            self.data_roots = data_root if isinstance(data_root, list) else [data_root]

        self.target_size = target_size
        self.max_frames_per_video = max_frames_per_video

        # Scan all videos
        self.videos = self._scan_videos()
        self.samples = self._build_samples()

        print(f"FusionDataset initialized:")
        print(f"  - {len(self.data_roots)} data root(s)")
        for i, root in enumerate(self.data_roots, 1):
            print(f"    [{i}] {root}")
        print(f"  - {len(self.videos)} videos")
        print(f"  - {len(self.samples)} frame pairs")

    def _scan_videos(self):
        """Scan all video directories from all data roots"""
        videos = []
        for data_root in self.data_roots:
            if not os.path.exists(data_root):
                print(f"⚠️  Warning: data_root does not exist: {data_root}")
                continue

            for item in sorted(os.listdir(data_root)):
                item_path = os.path.join(data_root, item)
                if os.path.isdir(item_path):
                    # Get all image files
                    frames = self._get_frames(item_path)
                    if len(frames) >= 2:  # Need at least 2 frames
                        videos.append({
                            'name': item,
                            'path': item_path,
                            'frames': frames
                        })
        return videos

    def _get_frames(self, video_path):
        """Get all frame paths in a video directory"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        frames = []
        for ext in extensions:
            frames.extend(glob.glob(os.path.join(video_path, ext)))
            frames.extend(glob.glob(os.path.join(video_path, ext.upper())))
        return sorted(frames)

    def _build_samples(self):
        """Build list of (video_idx, frame_idx) tuples"""
        samples = []
        for video_idx, video in enumerate(self.videos):
            num_frames = len(video['frames'])
            if self.max_frames_per_video is not None:
                num_frames = min(num_frames, self.max_frames_per_video)

            # Each sample is (current_frame, next_frame)
            for frame_idx in range(num_frames - 1):
                samples.append((video_idx, frame_idx))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            frame_t_lab: [3, H, W] LAB tensor (normalized to [-1, 1])
            frame_t1_lab: [3, H, W] LAB tensor (normalized to [-1, 1])
            reference_pil: PIL Image (RGB) - first frame of video
            target_pil: PIL Image (RGB) - frame_t1
            gt_ab: [2, H, W] ground truth AB channels
        """
        video_idx, frame_idx = self.samples[idx]
        video = self.videos[video_idx]

        # Load frames
        frame_t_path = video['frames'][frame_idx]
        frame_t1_path = video['frames'][frame_idx + 1]
        reference_path = video['frames'][0]  # First frame as reference

        # Load as PIL
        frame_t_pil = Image.open(frame_t_path).convert('RGB')
        frame_t1_pil = Image.open(frame_t1_path).convert('RGB')
        reference_pil = Image.open(reference_path).convert('RGB')

        # Resize
        frame_t_pil = frame_t_pil.resize(self.target_size[::-1], Image.LANCZOS)
        frame_t1_pil = frame_t1_pil.resize(self.target_size[::-1], Image.LANCZOS)
        reference_pil = reference_pil.resize(self.target_size[::-1], Image.LANCZOS)

        # Convert to LAB
        frame_t_lab = self._rgb_to_lab_tensor(frame_t_pil)
        frame_t1_lab = self._rgb_to_lab_tensor(frame_t1_pil)

        # Ground truth AB
        gt_ab = frame_t1_lab[1:3, :, :]

        # Keep PIL for SwinTExCo
        target_pil = frame_t1_pil

        return frame_t_lab, frame_t1_lab, reference_pil, target_pil, gt_ab

    def _rgb_to_lab_tensor(self, pil_image):
        """
        Convert PIL RGB image to LAB tensor

        Returns:
            lab_tensor: [3, H, W] normalized to [-1, 1]
        """
        # Convert to numpy
        rgb_np = np.array(pil_image, dtype=np.uint8)

        # RGB to LAB
        lab_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Normalize
        lab_np[:, :, 0] = lab_np[:, :, 0] * 100.0 / 255.0  # L: [0, 100]
        lab_np[:, :, 1] = lab_np[:, :, 1] - 128.0          # a: [-128, 127]
        lab_np[:, :, 2] = lab_np[:, :, 2] - 128.0          # b: [-128, 127]

        # Further normalize to [-1, 1]
        lab_np[:, :, 0] = (lab_np[:, :, 0] / 50.0) - 1.0   # L: [-1, 1]
        lab_np[:, :, 1] = lab_np[:, :, 1] / 127.0          # a: [-1, 1]
        lab_np[:, :, 2] = lab_np[:, :, 2] / 127.0          # b: [-1, 1]

        # To tensor
        lab_tensor = torch.from_numpy(lab_np).permute(2, 0, 1).float()

        return lab_tensor


class VideoSequenceDataset(Dataset):
    """
    Alternative dataset that returns full video sequences

    Useful for proper temporal training with memory.
    """

    def __init__(self, data_root, target_size=(224, 224), sequence_length=8):
        self.data_root = data_root
        self.target_size = target_size
        self.sequence_length = sequence_length

        self.videos = self._scan_videos()

    def _scan_videos(self):
        """Same as FusionDataset"""
        videos = []
        for item in sorted(os.listdir(self.data_root)):
            item_path = os.path.join(self.data_root, item)
            if os.path.isdir(item_path):
                frames = self._get_frames(item_path)
                if len(frames) >= self.sequence_length:
                    videos.append({
                        'name': item,
                        'frames': frames
                    })
        return videos

    def _get_frames(self, video_path):
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        frames = []
        for ext in extensions:
            frames.extend(glob.glob(os.path.join(video_path, ext)))
        return sorted(frames)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        """
        Returns a sequence of frames from one video
        """
        video = self.videos[idx]

        # Sample sequence_length consecutive frames
        total_frames = len(video['frames'])
        start_idx = np.random.randint(0, total_frames - self.sequence_length + 1)

        sequence = []
        for i in range(start_idx, start_idx + self.sequence_length):
            frame_pil = Image.open(video['frames'][i]).convert('RGB')
            frame_pil = frame_pil.resize(self.target_size[::-1], Image.LANCZOS)
            frame_lab = self._rgb_to_lab_tensor(frame_pil)
            sequence.append(frame_lab)

        # Stack to [T, 3, H, W]
        sequence_tensor = torch.stack(sequence, dim=0)

        # Reference is first frame
        reference_path = video['frames'][0]
        reference_pil = Image.open(reference_path).convert('RGB')
        reference_pil = reference_pil.resize(self.target_size[::-1], Image.LANCZOS)

        return sequence_tensor, reference_pil

    def _rgb_to_lab_tensor(self, pil_image):
        """Same as FusionDataset"""
        rgb_np = np.array(pil_image, dtype=np.uint8)
        lab_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_np[:, :, 0] = lab_np[:, :, 0] * 100.0 / 255.0
        lab_np[:, :, 1] = lab_np[:, :, 1] - 128.0
        lab_np[:, :, 2] = lab_np[:, :, 2] - 128.0
        lab_np[:, :, 0] = (lab_np[:, :, 0] / 50.0) - 1.0
        lab_np[:, :, 1] = lab_np[:, :, 1] / 127.0
        lab_np[:, :, 2] = lab_np[:, :, 2] / 127.0
        lab_tensor = torch.from_numpy(lab_np).permute(2, 0, 1).float()
        return lab_tensor


class FusionSequenceDataset(Dataset):
    """
    Sequential dataset for FlowChroma fusion training with ImageNet references

    Loads sequences of frames (default 4 frames) for sequential training,
    enabling MemFlow's memory mechanism and using ImageNet semantic references.

    Args:
        davis_root: Path to DAVIS video frames (e.g., '/data/DAVIS/')
        imagenet_root: Path to ImageNet images (e.g., '/data/ImageNet/')
        annot_csv: Path to davis_annot.csv
        sequence_length: Number of frames per sequence (default 4)
        real_reference_probability: Probability of using ImageNet reference (default 1.0)
        target_size: (H, W) tuple for resizing
    """

    def __init__(
        self,
        davis_root,
        imagenet_root,
        annot_csv,
        sequence_length=4,
        real_reference_probability=1.0,
        target_size=(224, 224)
    ):
        import pandas as pd

        self.davis_root = davis_root
        self.imagenet_root = imagenet_root
        self.sequence_length = sequence_length
        self.real_reference_probability = real_reference_probability
        self.target_size = target_size

        # Load annotations
        self.annotations = pd.read_csv(annot_csv)
        print(f"Loaded {len(self.annotations)} frame pairs from {annot_csv}")

        # Group by video to create sequences
        self.sequences = self._build_sequences()
        print(f"Created {len(self.sequences)} sequences of length {sequence_length}")

    def _build_sequences(self):
        """
        Build sequences from frame pairs

        Returns:
            List of sequences, where each sequence is a list of annotation indices
        """
        sequences = []

        # Group annotations by video
        video_groups = self.annotations.groupby('video_name')

        for video_name, group in video_groups:
            # Sort by frame number (assuming frame names are sequential)
            group_sorted = group.sort_values('current_frame')
            indices = group_sorted.index.tolist()

            # Create sliding window sequences
            for i in range(len(indices) - self.sequence_length + 1):
                sequence_indices = indices[i:i + self.sequence_length]
                sequences.append({
                    'video_name': video_name,
                    'indices': sequence_indices,
                    'start_frame': i
                })

        return sequences

    def __len__(self):
        return len(self.sequences)

    def _load_frame(self, video_name, frame_name):
        """Load a single frame as PIL Image"""
        frame_path = os.path.join(self.davis_root, video_name, frame_name)
        return Image.open(frame_path).convert('RGB')

    def _select_reference(self, ref_list):
        """
        Select one reference from the list of 5 candidates

        Args:
            ref_list: List of 5 ImageNet reference paths

        Returns:
            PIL Image of selected reference
        """
        import random

        # Randomly select one from the 5 candidates
        ref_path = random.choice(ref_list)
        ref_full_path = os.path.join(self.imagenet_root, ref_path)

        try:
            ref_image = Image.open(ref_full_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load reference {ref_full_path}: {e}")
            # Fallback: return a blank reference
            ref_image = Image.new('RGB', (256, 256), color='gray')

        return ref_image

    def _rgb_to_lab_tensor(self, pil_image):
        """Convert PIL RGB image to LAB tensor (same as FusionDataset)"""
        rgb_np = np.array(pil_image, dtype=np.uint8)
        lab_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_np[:, :, 0] = lab_np[:, :, 0] * 100.0 / 255.0
        lab_np[:, :, 1] = lab_np[:, :, 1] - 128.0
        lab_np[:, :, 2] = lab_np[:, :, 2] - 128.0
        lab_np[:, :, 0] = (lab_np[:, :, 0] / 50.0) - 1.0
        lab_np[:, :, 1] = lab_np[:, :, 1] / 127.0
        lab_np[:, :, 2] = lab_np[:, :, 2] / 127.0
        lab_tensor = torch.from_numpy(lab_np).permute(2, 0, 1).float()
        return lab_tensor

    def __getitem__(self, idx):
        """
        Get a sequence of frames with references

        Returns:
            Dictionary containing:
                - frames_lab: List of [3, H, W] LAB tensors (sequence_length frames)
                - frames_pil: List of PIL Images (for SwinTExCo, sequence_length frames)
                - references_pil: List of PIL Images (sequence_length references)
                - video_name: str
        """
        sequence_info = self.sequences[idx]
        video_name = sequence_info['video_name']
        indices = sequence_info['indices']

        frames_lab = []
        frames_pil = []
        references_pil = []

        for idx in indices:
            row = self.annotations.iloc[idx]

            # Load current frame
            frame_pil = self._load_frame(video_name, row['current_frame'])
            frame_pil_resized = frame_pil.resize(self.target_size[::-1], Image.LANCZOS)

            # Convert to LAB
            frame_lab = self._rgb_to_lab_tensor(frame_pil_resized)

            frames_pil.append(frame_pil_resized)
            frames_lab.append(frame_lab)

            # Select reference image
            ref_list = [row['ref1'], row['ref2'], row['ref3'], row['ref4'], row['ref5']]

            if np.random.random() < self.real_reference_probability:
                # Use ImageNet external reference
                reference = self._select_reference(ref_list)
            else:
                # Use video first frame
                first_frame_name = self.annotations[
                    self.annotations['video_name'] == video_name
                ].iloc[0]['current_frame']
                reference = self._load_frame(video_name, first_frame_name)

            reference_resized = reference.resize(self.target_size[::-1], Image.LANCZOS)
            references_pil.append(reference_resized)

        return {
            'frames_lab': frames_lab,  # List of LAB tensors
            'frames_pil': frames_pil,  # List of PIL Images
            'references_pil': references_pil,  # List of PIL Images
            'video_name': video_name
        }


def fusion_sequence_collate_fn(batch):
    """
    Custom collate function for FusionSequenceDataset

    Keeps the batch structure organized for easy sequence iteration.

    Args:
        batch: List of dictionaries from __getitem__

    Returns:
        Dictionary containing:
            - frames_lab: List[List[Tensor]] - batch_size x sequence_length
            - frames_pil: List[List[PIL.Image]] - batch_size x sequence_length
            - references_pil: List[List[PIL.Image]] - batch_size x sequence_length
            - video_names: List[str]
    """
    return {
        'frames_lab': [item['frames_lab'] for item in batch],
        'frames_pil': [item['frames_pil'] for item in batch],
        'references_pil': [item['references_pil'] for item in batch],
        'video_names': [item['video_name'] for item in batch]
    }
