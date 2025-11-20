"""
Fusion System

Integrates MemFlow, SwinTExCo, and Fusion UNet into a complete training/inference system.

Usage:
    from fusion.fusion_system import FusionSystem
    from FusionNet.fusion_unet import SimpleFusionNet

    system = FusionSystem(
        memflow_path='../MemFlow',
        swintexco_path='../SwinSingle',
        memflow_ckpt='checkpoints/memflow_best.pth',
        swintexco_ckpt='checkpoints/best/',
        fusion_net=SimpleFusionNet()
    )

    output = system(frame_t, frame_t1, reference_frame, target_frame)
"""

import sys
import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast


class FusionSystem(nn.Module):
    """
    Complete Fusion System

    Integrates three modules:
        1. MemFlow (frozen) - Temporal consistency via optical flow
        2. SwinTExCo (frozen) - Semantic matching via exemplar
        3. Fusion UNet (trainable) - Intelligent fusion

    During training, only Fusion UNet parameters are updated.
    """

    def __init__(self,
                 memflow_path,
                 swintexco_path,
                 memflow_ckpt,
                 swintexco_ckpt,
                 fusion_net=None,
                 device='cuda'):
        """
        Args:
            memflow_path: Path to MemFlow repository
            swintexco_path: Path to SwinSingle repository
            memflow_ckpt: Path to MemFlow checkpoint
            swintexco_ckpt: Path to SwinTExCo checkpoint directory
            fusion_net: Fusion network instance (default: PlaceholderFusion)
            device: cuda or cpu
        """
        super().__init__()

        self.device = device

        # Convert to absolute paths
        self.memflow_path = os.path.abspath(memflow_path)
        self.swintexco_path = os.path.abspath(swintexco_path)

        # Add paths for imports
        if self.memflow_path not in sys.path:
            sys.path.insert(0, self.memflow_path)
        if self.swintexco_path not in sys.path:
            sys.path.insert(0, self.swintexco_path)

        # ============ Load MemFlow ============
        print("="*60)
        print("Loading MemFlow...")
        print("="*60)
        self.memflow = self._load_memflow(memflow_ckpt)
        print(f"✅ MemFlow loaded from {memflow_ckpt}\n")

        # ============ Load SwinTExCo ============
        print("="*60)
        print("Loading SwinTExCo...")
        print("="*60)
        self.swintexco = self._load_swintexco(swintexco_ckpt)
        print(f"✅ SwinTExCo loaded from {swintexco_ckpt}\n")

        # ============ Create Fusion UNet ============
        print("="*60)
        print("Creating Fusion UNet...")
        print("="*60)

        if fusion_net is None:
            from FusionNet.fusion_unet import PlaceholderFusion
            self.fusion_unet = PlaceholderFusion().to(device)
            print("Using PlaceholderFusion (testing mode)")
        else:
            self.fusion_unet = fusion_net.to(device)
            print(f"Using {fusion_net.__class__.__name__}")

        print("✅ Fusion UNet created\n")

        # MemFlow memory management
        self.reset_memory()

    def _load_memflow(self, checkpoint_path):
        """Load MemFlow model"""
        try:
            from core.Networks import build_network
            from configs.colorization_memflownet import get_cfg

            cfg = get_cfg()
            cfg.restore_ckpt = checkpoint_path

            model = build_network(cfg).to(self.device)

            # Load checkpoint
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            if 'model' in ckpt:
                model.load_state_dict(ckpt['model'], strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)

            # Freeze and eval
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load MemFlow: {e}\n"
                             f"Make sure memflow_path points to MemFlow repository")

    def _load_swintexco(self, checkpoint_path):
        """Load SwinTExCo model"""
        try:
            from inference import SwinTExCo

            model = SwinTExCo(
                weights_path=checkpoint_path,
                device=self.device
            )

            # Already frozen in SwinTExCo.__init__
            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load SwinTExCo: {e}\n"
                             f"Make sure swintexco_path points to SwinSingle repository\n"
                             f"and similarity_map modifications are applied")

    def reset_memory(self):
        """Reset MemFlow memory (call at start of each video)"""
        self.memflow_memory = None
        self.curr_ti = -1

    def memflow_inference(self, frame_t, frame_t1):
        """
        MemFlow inference

        Args:
            frame_t: [B, 3, H, W] LAB tensor (normalized to [-1, 1])
            frame_t1: [B, 3, H, W] LAB tensor (normalized to [-1, 1])

        Returns:
            memflow_ab: [B, 2, H, W]
            memflow_conf: [B, 1, H, W]
        """
        self.curr_ti += 1

        # Prepare input
        images_norm = torch.stack([frame_t, frame_t1], dim=1)  # [B, 2, 3, H, W]

        with torch.no_grad():
            # Encode context
            query, key, net, inp = self.memflow.encode_context(images_norm[:, 0, ...])

            # Encode features
            coords0, coords1, fmaps = self.memflow.encode_features(images_norm)

            # Memory management
            if self.curr_ti == 0:
                ref_values = None
                ref_keys = key.unsqueeze(2)
            else:
                ref_values = self.memflow_memory
                ref_keys = torch.cat([self.memflow_memory, key.unsqueeze(2)], dim=2)

            # Predict flow
            flow_predictions, current_value, confidence_map = self.memflow.predict_flow(
                net, inp, coords0, coords1, fmaps,
                query.unsqueeze(2), ref_keys, ref_values
            )

            # Update memory
            if self.memflow_memory is None:
                self.memflow_memory = current_value
            else:
                self.memflow_memory = torch.cat([self.memflow_memory, current_value], dim=2)

            # Get final flow
            flow_final = flow_predictions[-1]

            # Upsample flow
            H, W = frame_t1.shape[2:]
            flow_up = nn.functional.interpolate(
                flow_final * 8,
                size=(H, W),
                mode='bilinear',
                align_corners=True
            )

            # Warp color
            from core.loss_new import warp_color_by_flow
            prev_ab = frame_t[:, 1:3, :, :]
            memflow_ab = warp_color_by_flow(prev_ab, flow_up)

        return memflow_ab, confidence_map

    def swintexco_inference(self, reference_pil, target_pil):
        """
        SwinTExCo inference

        Args:
            reference_pil: PIL Image (RGB)
            target_pil: PIL Image (RGB)

        Returns:
            swintexco_ab: [B, 2, H, W] (normalized to [-1, 1])
            similarity_map: [B, 1, H, W]

        Note:
            This requires SwinTExCo to be modified to return similarity_map.
            See fusion/README.md for required modifications.
        """
        with torch.no_grad():
            # Process reference
            ref_lab = self.swintexco.processor(reference_pil).unsqueeze(0).to(self.device)

            # Get reference features
            from src.utils import uncenter_l, tensor_lab2rgb
            ref_l = ref_lab[:, 0:1, :, :]
            ref_ab = ref_lab[:, 1:3, :, :]
            ref_rgb = tensor_lab2rgb(torch.cat([uncenter_l(ref_l), ref_ab], dim=1))
            features_B = self.swintexco.embed_net(ref_rgb)

            # Process target and get prediction with similarity_map
            # Note: This calls the modified __proccess_sample that returns similarity_map
            target_lab = self.swintexco.processor(target_pil).unsqueeze(0).to(self.device)
            target_l = target_lab[:, 0:1, :, :]

            # Call frame_colorization (modified to return 3 values)
            from src.models.CNN.FrameColor import frame_colorization
            ab_predict, _, similarity_map = frame_colorization(
                target_l,
                ref_lab,
                features_B,
                self.swintexco.embed_net,
                self.swintexco.nonlocal_net,
                self.swintexco.colornet,
                luminance_noise=0,
                temperature=1e-10,
                joint_training=False
            )

        return ab_predict, similarity_map

    def forward(self, frame_t, frame_t1, reference_pil, target_pil):
        """
        Complete forward pass

        Args:
            frame_t: [B, 3, H, W] LAB tensor (normalized)
            frame_t1: [B, 3, H, W] LAB tensor (normalized)
            reference_pil: PIL Image (RGB)
            target_pil: PIL Image (RGB)

        Returns:
            fused_ab: [B, 2, H, W]
        """
        # 1. MemFlow inference (frozen)
        memflow_ab, memflow_conf = self.memflow_inference(frame_t, frame_t1)

        # 2. SwinTExCo inference (frozen)
        swintexco_ab, swintexco_sim = self.swintexco_inference(reference_pil, target_pil)

        # 3. Extract L channel
        L_channel = frame_t1[:, 0:1, :, :]

        # 4. Fusion UNet inference (trainable)
        fused_ab = self.fusion_unet(
            memflow_ab,
            swintexco_ab,
            memflow_conf,
            swintexco_sim,
            L_channel
        )

        return fused_ab

    def train(self, mode=True):
        """Override train to only affect Fusion UNet"""
        if mode:
            self.fusion_unet.train()
        else:
            self.fusion_unet.eval()
        # MemFlow and SwinTExCo always stay in eval mode
        return self

    def parameters(self, recurse=True):
        """Override to only return Fusion UNet parameters"""
        return self.fusion_unet.parameters(recurse=recurse)
