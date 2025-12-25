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
        davis_root: Path(s) to DAVIS video frames
                   - Single path: '/data/DAVIS/'
                   - Multiple paths (comma-separated): '/data/DAVIS1,/data/DAVIS2'
                   - CSV file will be auto-detected in each path (first .csv file found)
        imagenet_root: Path(s) to ImageNet images
                      - Single path: '/data/ImageNet/'
                      - Multiple paths (comma-separated): '/data/ImageNet1,/data/ImageNet2'
        sequence_length: Number of frames per sequence (default 4)
        real_reference_probability: Probability of using ImageNet reference (default 1.0)
        target_size: (H, W) tuple for resizing
    """

    def __init__(
        self,
        davis_root,
        imagenet_root,
        sequence_length=4,
        real_reference_probability=1.0,
        target_size=(224, 224)
    ):
        import pandas as pd

        # Parse multiple paths
        self.davis_roots = self._parse_paths(davis_root)
        self.imagenet_roots = self._parse_paths(imagenet_root)
        self.sequence_length = sequence_length
        self.real_reference_probability = real_reference_probability
        self.target_size = target_size

        print(f"Dataset paths: {self.davis_roots}")
        print(f"ImageNet paths: {self.imagenet_roots}")

        # Load annotations from all paths (auto-detect CSV files)
        self.annotations = self._load_annotations()
        print(f"Loaded {len(self.annotations)} total frame pairs")

        # Group by video to create sequences
        self.sequences = self._build_sequences()
        print(f"Created {len(self.sequences)} sequences of length {sequence_length}")

    def _parse_paths(self, path_str):
        """
        Parse path string into list of paths

        Args:
            path_str: Single path or comma-separated paths (no spaces)

        Returns:
            List of paths
        """
        if isinstance(path_str, list):
            return path_str
        elif ',' in path_str:
            return [p.strip() for p in path_str.split(',')]
        else:
            return [path_str]

    def _load_annotations(self):
        """
        Load annotations from all dataset paths (auto-detect CSV files)

        Searches for the first .csv file in each path and loads it.

        Returns:
            Combined DataFrame
        """
        import pandas as pd
        import glob

        all_annotations = []

        for davis_path in self.davis_roots:
            # Find all CSV files in the path
            csv_files = glob.glob(os.path.join(davis_path, '*.csv'))

            if csv_files:
                # Use the first CSV file found
                csv_path = csv_files[0]
                df = pd.read_csv(csv_path)
                # Add source path column to help locate files later
                df['_source_path'] = davis_path
                all_annotations.append(df)
                print(f"  Loaded {len(df)} pairs from {csv_path}")
            else:
                print(f"  Warning: No CSV file found in {davis_path}")

        if not all_annotations:
            raise FileNotFoundError(f"No CSV files found in any dataset path: {self.davis_roots}")

        # Combine all annotations
        return pd.concat(all_annotations, ignore_index=True)

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

    def _load_frame(self, video_name, frame_name, source_path=None):
        """
        Load a single frame as PIL Image (flexible path search)
        Based on SwinTExCo's approach with flexible video folder matching

        Args:
            video_name: Video folder name (from CSV)
            frame_name: Frame filename (from CSV)
            source_path: Preferred source path (from annotation)

        Returns:
            PIL Image
        """
        import glob

        # Prepare search paths
        search_paths = [source_path] if source_path else []
        search_paths.extend(self.davis_roots)

        for base_path in search_paths:
            if not base_path:
                continue

            # Try multiple path patterns (based on SwinTExCo's approach)
            patterns = [
                # Standard: video_data_root/video_name/frame_name
                os.path.join(base_path, video_name, frame_name),

                # Flexible video folder matching (e.g., CSV has "1" but folder is "001")
                os.path.join(base_path, f"*{video_name}", frame_name),
                os.path.join(base_path, f"*{video_name}*", frame_name),
            ]

            # Add zero-padded versions if video_name is numeric
            if video_name.isdigit():
                num = int(video_name)
                patterns.extend([
                    os.path.join(base_path, f"{num:03d}", frame_name),  # 001, 002, etc.
                    os.path.join(base_path, f"{num:04d}", frame_name),  # 0001, 0002, etc.
                ])

            # Also try flat structure
            patterns.append(os.path.join(base_path, frame_name))

            for pattern in patterns:
                # Direct path check
                if '*' not in pattern and os.path.exists(pattern):
                    return Image.open(pattern).convert('RGB')

                # Glob pattern matching
                if '*' in pattern:
                    matches = glob.glob(pattern)
                    if matches:
                        # Sort to get consistent results
                        matches.sort()
                        return Image.open(matches[0]).convert('RGB')

        # If not found, raise error with helpful message
        raise FileNotFoundError(
            f"Frame not found: {video_name}/{frame_name}\n"
            f"Searched in paths: {self.davis_roots}\n"
            f"Hint: Check if video folders exist in the dataset path"
        )

    def _select_reference(self, ref_list):
        """
        Select one reference from the list of 5 candidates (flexible path search)

        Args:
            ref_list: List of 5 ImageNet reference paths

        Returns:
            PIL Image of selected reference
        """
        import random
        import glob

        # Randomly select one from the 5 candidates
        ref_path = random.choice(ref_list)

        # Search in all ImageNet paths with flexible matching
        for imagenet_path in self.imagenet_roots:
            patterns = [
                os.path.join(imagenet_path, ref_path),              # Direct path
                os.path.join(imagenet_path, f"*{ref_path}*"),       # Flexible matching
                os.path.join(imagenet_path, os.path.basename(ref_path)),  # Just filename
            ]

            for pattern in patterns:
                # Direct path check
                if '*' not in pattern and os.path.exists(pattern):
                    try:
                        return Image.open(pattern).convert('RGB')
                    except Exception as e:
                        print(f"Warning: Failed to load reference {pattern}: {e}")
                        continue

                # Glob pattern matching
                if '*' in pattern:
                    matches = glob.glob(pattern)
                    if matches:
                        try:
                            return Image.open(matches[0]).convert('RGB')
                        except Exception as e:
                            print(f"Warning: Failed to load reference {matches[0]}: {e}")
                            continue

        # Fallback: return a blank reference
        print(f"Warning: Reference not found in any path: {ref_path}")
        return Image.new('RGB', (256, 256), color='gray')

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
        video_name = str(sequence_info['video_name'])
        indices = sequence_info['indices']

        frames_lab = []
        frames_pil = []
        references_pil = []

        for idx in indices:
            row = self.annotations.iloc[idx]
            source_path = row.get('_source_path', None)

            # Load current frame (ensure frame_name is string)
            frame_pil = self._load_frame(video_name, str(row['current_frame']), source_path)
            frame_pil_resized = frame_pil.resize(self.target_size[::-1], Image.LANCZOS)

            # Convert to LAB
            frame_lab = self._rgb_to_lab_tensor(frame_pil_resized)

            frames_pil.append(frame_pil_resized)
            frames_lab.append(frame_lab)

            # Select reference image
            ref_list = [str(row['ref1']), str(row['ref2']), str(row['ref3']), str(row['ref4']), str(row['ref5'])]

            if np.random.random() < self.real_reference_probability:
                # Use ImageNet external reference
                reference = self._select_reference(ref_list)
            else:
                # Use video first frame
                try:
                    video_frames = self.annotations[
                        self.annotations['video_name'] == video_name
                    ]
                    if len(video_frames) > 0:
                        first_frame_name = video_frames.iloc[0]['current_frame']
                        reference = self._load_frame(video_name, str(first_frame_name), source_path)
                    else:
                        # Fallback: use current frame as reference
                        reference = frame_pil
                except (IndexError, KeyError) as e:
                    # Fallback: use current frame as reference
                    reference = frame_pil

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
