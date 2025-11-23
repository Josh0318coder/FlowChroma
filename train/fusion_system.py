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
        2. SwinTExCo (trainable) - Semantic matching via exemplar
        3. Fusion UNet (trainable) - Intelligent fusion

    During training, SwinTExCo and Fusion UNet are jointly optimized,
    while MemFlow remains frozen.
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

            # Keep model in fp32, will use autocast for mixed precision
            # This matches the original training setup

            # Freeze and eval
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load MemFlow: {e}\n"
                             f"Make sure memflow_path points to MemFlow repository")

    def _load_swintexco(self, checkpoint_path):
        """Load SwinTExCo model (trainable for joint training)"""
        try:
            from inference import SwinTExCo

            model = SwinTExCo(
                weights_path=checkpoint_path,
                device=self.device
            )

            # Unfreeze for joint training
            for param in model.embed_net.parameters():
                param.requires_grad = True
            for param in model.nonlocal_net.parameters():
                param.requires_grad = True
            for param in model.colornet.parameters():
                param.requires_grad = True

            # Set to train mode
            model.embed_net.train()
            model.nonlocal_net.train()
            model.colornet.train()

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
            memflow_lab: [B, 3, H, W] - Complete LAB prediction
            memflow_conf: [B, 1, H, W] - Confidence map
        """
        self.curr_ti += 1

        # Prepare input
        images_norm = torch.stack([frame_t, frame_t1], dim=1)  # [B, 2, 3, H, W]

        with torch.no_grad(), autocast(enabled=True):
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

            # Predict flow with autocast (FlashAttention will get fp16 automatically)
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

            # Combine with current frame L channel to form complete LAB
            current_L = frame_t1[:, 0:1, :, :]
            memflow_lab = torch.cat([current_L, memflow_ab], dim=1)

        return memflow_lab, confidence_map

    def swintexco_inference(self, reference_pil, target_pil):
        """
        SwinTExCo inference

        Args:
            reference_pil: PIL Image (RGB)
            target_pil: PIL Image (RGB)

        Returns:
            swintexco_lab: [B, 3, H, W] - Complete LAB (L from target + predicted AB)
            similarity_map: [B, 1, H, W]

        Note:
            This requires SwinTExCo to be modified to return similarity_map.
            SwinTExCo only predicts AB channels, we combine with target L channel.
        """
        # Disable autocast for SwinTExCo (not compatible with mixed precision)
        with torch.no_grad(), autocast(enabled=False):
            # Process reference
            ref_lab = self.swintexco.processor(reference_pil).unsqueeze(0).to(self.device)

            # Get reference features
            from src.utils import uncenter_l, tensor_lab2rgb
            ref_l = ref_lab[:, 0:1, :, :]
            ref_ab = ref_lab[:, 1:3, :, :]
            ref_rgb = tensor_lab2rgb(torch.cat([uncenter_l(ref_l), ref_ab], dim=1))
            features_B = self.swintexco.embed_net(ref_rgb)

            # Process target and get prediction with similarity_map
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

            # Combine target L with predicted AB to form complete LAB
            swintexco_lab = torch.cat([target_l, ab_predict], dim=1)

        return swintexco_lab, similarity_map

    def forward_sequence(self, frames_lab, frames_pil, references_pil):
        """
        Process a 4-frame sequence

        Args:
            frames_lab: List of 4 LAB tensors [3, H, W] (normalized to [-1, 1])
            frames_pil: List of 4 PIL Images (RGB) for target frames
            references_pil: List of 4 PIL Images (RGB) for reference frames

        Returns:
            outputs: List of 4 fused LAB tensors [3, H, W]
        """
        # Reset memory for new sequence
        self.reset_memory()

        outputs = []

        # Process each frame pair sequentially
        for i in range(len(frames_lab)):
            # Add batch dimension
            frame_lab = frames_lab[i].unsqueeze(0)  # [1, 3, H, W]
            frame_pil = frames_pil[i]
            reference_pil = references_pil[i]

            if i == 0:
                # First frame: MemFlow uses zero placeholder
                output = self.forward_single_frame(
                    None,  # No previous frame
                    frame_lab,
                    reference_pil,
                    frame_pil,
                    is_first=True
                )
            else:
                # Subsequent frames: use previous frame
                prev_frame_lab = frames_lab[i-1].unsqueeze(0)
                output = self.forward_single_frame(
                    prev_frame_lab,
                    frame_lab,
                    reference_pil,
                    frame_pil,
                    is_first=False
                )

            # Remove batch dimension and store
            outputs.append(output.squeeze(0))

        return outputs

    def forward_single_frame(self, frame_t, frame_t1, reference_pil, target_pil, is_first=False):
        """
        Process a single frame (used within sequence processing)

        Args:
            frame_t: [B, 3, H, W] LAB tensor (previous frame, None if first)
            frame_t1: [B, 3, H, W] LAB tensor (current frame)
            reference_pil: PIL Image (RGB)
            target_pil: PIL Image (RGB)
            is_first: bool, whether this is the first frame

        Returns:
            fused_lab: [B, 3, H, W] - Complete LAB prediction
        """
        B, _, H, W = frame_t1.shape
        L_channel = frame_t1[:, 0:1, :, :]

        # 1. MemFlow inference (frozen)
        if is_first:
            # First frame: use zero placeholder
            memflow_lab = torch.zeros(B, 3, H, W, device=self.device)
            memflow_conf = torch.zeros(B, 1, H, W, device=self.device)
        else:
            # Subsequent frames: normal inference
            memflow_lab, memflow_conf = self.memflow_inference(frame_t, frame_t1)

        # 2. SwinTExCo inference (always valid)
        swintexco_lab, swintexco_sim = self.swintexco_inference(reference_pil, target_pil)

        # 3. Fusion UNet inference (trainable)
        fused_lab = self.fusion_unet(
            memflow_lab,
            memflow_conf,
            swintexco_lab,
            swintexco_sim,
            L_channel
        )

        return fused_lab

    def forward(self, frame_t, frame_t1, reference_pil, target_pil):
        """
        Complete forward pass (legacy interface for 2-frame processing)

        Args:
            frame_t: [B, 3, H, W] LAB tensor (normalized)
            frame_t1: [B, 3, H, W] LAB tensor (normalized)
            reference_pil: PIL Image (RGB)
            target_pil: PIL Image (RGB)

        Returns:
            fused_lab: [B, 3, H, W] - Complete LAB prediction
        """
        return self.forward_single_frame(frame_t, frame_t1, reference_pil, target_pil, is_first=(self.curr_ti == -1))

    def train(self, mode=True):
        """Override train to affect SwinTExCo and Fusion UNet"""
        if mode:
            # Train mode: SwinTExCo and FusionNet
            self.swintexco.embed_net.train()
            self.swintexco.nonlocal_net.train()
            self.swintexco.colornet.train()
            self.fusion_unet.train()
        else:
            # Eval mode
            self.swintexco.embed_net.eval()
            self.swintexco.nonlocal_net.eval()
            self.swintexco.colornet.eval()
            self.fusion_unet.eval()
        # MemFlow always stays in eval mode
        return self

    def parameters(self, recurse=True):
        """Override to return SwinTExCo and Fusion UNet parameters"""
        import itertools
        return itertools.chain(
            self.swintexco.embed_net.parameters(recurse=recurse),
            self.swintexco.nonlocal_net.parameters(recurse=recurse),
            self.swintexco.colornet.parameters(recurse=recurse),
            self.fusion_unet.parameters(recurse=recurse)
        )

    def get_parameter_groups(self):
        """
        Get parameter groups for different learning rates

        Returns:
            list: [{'params': swintexco_params, 'name': 'swintexco'},
                   {'params': fusion_params, 'name': 'fusion'}]
        """
        import itertools
        swintexco_params = list(itertools.chain(
            self.swintexco.embed_net.parameters(),
            self.swintexco.nonlocal_net.parameters(),
            self.swintexco.colornet.parameters()
        ))
        fusion_params = list(self.fusion_unet.parameters())

        return [
            {'params': swintexco_params, 'name': 'swintexco'},
            {'params': fusion_params, 'name': 'fusion'}
        ]
