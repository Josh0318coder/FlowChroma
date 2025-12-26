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
                 swintexco_ckpt=None,
                 fusion_net=None,
                 device='cuda'):
        """
        Args:
            memflow_path: Path to MemFlow repository
            swintexco_path: Path to SwinSingle repository
            memflow_ckpt: Path to MemFlow checkpoint
            swintexco_ckpt: Path to SwinTExCo checkpoint directory (None for random init)
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
        if swintexco_ckpt is not None:
            print(f"✅ SwinTExCo loaded from {swintexco_ckpt}\n")
        else:
            print(f"✅ SwinTExCo created with random NonLocalNet initialization\n")

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

    def _load_swintexco(self, checkpoint_path=None):
        """
        Load SwinTExCo model (trainable for joint training)

        Args:
            checkpoint_path: Path to pretrained weights (None for random initialization)

        Strategy:
            - If checkpoint_path is provided: Load fully pretrained model
            - If checkpoint_path is None: Use pretrained Swin backbone + random NonLocalNet
        """
        try:
            if checkpoint_path is not None:
                # Option 1: Load fully pretrained model
                from inference import SwinTExCo

                model = SwinTExCo(
                    weights_path=checkpoint_path,
                    device=self.device
                )
                print("  ✓ Loaded pretrained embed_net, nonlocal_net, colornet")
            else:
                # Option 2: Pretrained Swin backbone + Random NonLocalNet
                from src.models.vit.embed import SwinModel
                from src.models.CNN.NonlocalNet import WarpNet
                from src.models.CNN.ColorVidNet import ColorVidNet
                from src.utils import RGB2Lab, ToTensor, Normalize
                import torchvision.transforms as T

                # Create a pseudo-SwinTExCo object
                class SwinTExCoWrapper:
                    def __init__(self, embed_net, nonlocal_net, colornet, processor):
                        self.embed_net = embed_net
                        self.nonlocal_net = nonlocal_net
                        self.colornet = colornet
                        self.processor = processor
                        self.device = embed_net.device

                # Load pretrained Swin Transformer backbone
                embed_net = SwinModel(
                    pretrained_model='swinv2-cr-t-224',
                    device=self.device
                ).to(self.device)
                print("  ✓ Loaded pretrained Swin backbone (swinv2-cr-t-224)")

                # Random initialize NonLocalNet
                nonlocal_net = WarpNet(feature_channel=128).to(self.device)
                print("  ⚠️  NonLocalNet randomly initialized (will be trained)")

                # Random initialize ColorVidNet (will be frozen anyway)
                colornet = ColorVidNet(4).to(self.device)
                print("  ✓ ColorVidNet randomly initialized (frozen)")

                # Create processor
                processor = T.Compose([
                    RGB2Lab(),
                    ToTensor(),
                    Normalize()
                ])

                # Wrap into pseudo-SwinTExCo object
                model = SwinTExCoWrapper(embed_net, nonlocal_net, colornet, processor)

            # Set trainable/frozen status
            for param in model.embed_net.parameters():
                param.requires_grad = False
            for param in model.nonlocal_net.parameters():
                param.requires_grad = True
            for param in model.colornet.parameters():
                param.requires_grad = False

            # Set train/eval modes
            model.embed_net.eval()
            model.nonlocal_net.train()
            model.colornet.eval()

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load SwinTExCo: {e}\n"
                             f"Make sure swintexco_path points to SwinSingle repository\n"
                             f"and similarity_map modifications are applied")

    def reset_memory(self):
        """Reset MemFlow memory (call at start of each video)"""
        self.memflow_memory_keys = None
        self.memflow_memory_values = None
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
            if self.memflow_memory_keys is None:
                # First frame: no history
                ref_values = None
                ref_keys = key.unsqueeze(2)
            else:
                # Use historical keys and values
                ref_values = self.memflow_memory_values
                ref_keys = torch.cat([self.memflow_memory_keys, key.unsqueeze(2)], dim=2)

            # Predict flow with autocast (FlashAttention will get fp16 automatically)
            flow_predictions, current_value, confidence_map = self.memflow.predict_flow(
                net, inp, coords0, coords1, fmaps,
                query.unsqueeze(2), ref_keys, ref_values
            )

            # Update memory: store both keys and values separately
            if self.memflow_memory_keys is None:
                self.memflow_memory_keys = key.unsqueeze(2)
                self.memflow_memory_values = current_value
            else:
                self.memflow_memory_keys = torch.cat([self.memflow_memory_keys, key.unsqueeze(2)], dim=2)
                self.memflow_memory_values = torch.cat([self.memflow_memory_values, current_value], dim=2)

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
        SwinTExCo inference - Using WarpNet (NonLocalNet) output directly

        Skips ColorVidNet and uses WarpNet's output directly.
        FusionNet will handle the refinement (similar to ColorVidNet's role).

        Args:
            reference_pil: PIL Image (RGB)
            target_pil: PIL Image (RGB)

        Returns:
            warpnet_ab: [B, 2, H, W] - NonLocalNet AB output (normalized to [-1, 1])
            similarity_map: [B, 1, H, W] - Feature similarity map

        Note:
            This bypasses ColorVidNet. FusionNet now takes WarpNet's raw output
            and performs the final refinement, combining it with MemFlow's temporal info.
        """
        # Disable autocast for SwinTExCo (not compatible with mixed precision)
        # Note: Do NOT use torch.no_grad() here during training, as SwinTExCo is trainable
        with autocast(enabled=False):
            # Process reference
            ref_lab = self.swintexco.processor(reference_pil).unsqueeze(0).to(self.device)

            # Get reference features
            from src.utils import uncenter_l, uncenter_ab, tensor_lab2rgb
            ref_l = ref_lab[:, 0:1, :, :]
            ref_ab = ref_lab[:, 1:3, :, :]
            ref_rgb = tensor_lab2rgb(torch.cat([uncenter_l(ref_l), uncenter_ab(ref_ab)], dim=1))
            features_B = self.swintexco.embed_net(ref_rgb)

            # Process target
            target_lab = self.swintexco.processor(target_pil).unsqueeze(0).to(self.device)
            target_l = target_lab[:, 0:1, :, :]

            # Call warp_color (NonLocalNet only, skip ColorVidNet)
            from src.models.CNN.FrameColor import warp_color
            nonlocal_BA_lab, similarity_map = warp_color(
                target_l,
                ref_lab,
                features_B,
                self.swintexco.embed_net,
                self.swintexco.nonlocal_net,
                temperature=1e-10,
            )

            # Extract AB channels from WarpNet output
            warpnet_ab = nonlocal_BA_lab[:, 1:3, :, :]

        return warpnet_ab, similarity_map

    def forward_sequence(self, frames_lab, frames_pil, references_pil, return_memflow=False):
        """
        Process a sequence of frames

        Args:
            frames_lab: list of [3, H, W] LAB tensors (normalized to [-1, 1])
            frames_pil: list of PIL Images (RGB, target frames)
            references_pil: list of PIL Images (RGB, reference frames, typically all the same)
            return_memflow: bool, whether to return MemFlow outputs (for temporal loss in training)

        Returns:
            If return_memflow=False:
                results: list of [3, H, W] LAB tensors (colorized results)
            If return_memflow=True:
                (results, memflow_outputs, memflow_confs): tuple of lists
        """
        # Reset memory at the start of sequence
        self.reset_memory()

        results = []
        memflow_outputs = [] if return_memflow else None
        memflow_confs = [] if return_memflow else None

        for i in range(len(frames_lab)):
            # Convert to batch [1, 3, H, W]
            frame_t1_batch = frames_lab[i].unsqueeze(0).to(self.device)

            if i == 0:
                # First frame: manually construct output without calling forward()
                # to avoid state management issues
                B, _, H, W = frame_t1_batch.shape
                L_channel = frame_t1_batch[:, 0:1, :, :]

                # MemFlow: output zero (no temporal info for first frame)
                memflow_lab = torch.zeros(B, 3, H, W, device=self.device)
                memflow_conf = torch.zeros(B, 1, H, W, device=self.device)

                # SwinTExCo: process reference-based colorization
                swintexco_ab, swintexco_sim = self.swintexco_inference(
                    references_pil[i],
                    frames_pil[i]
                )

                # Construct complete SwinTExCo LAB prediction (for symmetry with MemFlow)
                swintexco_lab = torch.cat([L_channel, swintexco_ab], dim=1)

                # FusionNet: fuse results (returns AB channels only)
                fused_ab = self.fusion_unet(
                    memflow_lab,
                    memflow_conf,
                    swintexco_lab,
                    swintexco_sim
                )

                # Construct complete LAB output
                output_lab = torch.cat([L_channel, fused_ab], dim=1)

                # After first frame, increment curr_ti to 0 (so next frame is not treated as first)
                self.curr_ti += 1
            else:
                # Subsequent frames: use PREVIOUS PREDICTION (not GT)
                # This enables error accumulation training (like real inference)

                # Use previous frame's prediction as input
                prev_output = results[-1]  # Get previous frame's output

                # Detach to prevent gradient backprop through entire sequence
                # (saves memory while still training with accumulated errors)
                frame_t_batch = prev_output.detach().unsqueeze(0)

                # Alternative: Full BPTT (expensive but more accurate)
                # frame_t_batch = prev_output.unsqueeze(0)  # No detach - gradients flow through

                # Forward pass (curr_ti will be managed automatically)
                # If return_memflow is True, we need to capture MemFlow outputs
                if return_memflow:
                    output_lab, memflow_lab, memflow_conf = self.forward_with_memflow(
                        frame_t_batch,
                        frame_t1_batch,
                        references_pil[i],
                        frames_pil[i]
                    )
                else:
                    output_lab = self.forward(
                        frame_t_batch,
                        frame_t1_batch,
                        references_pil[i],
                        frames_pil[i]
                    )
                    # Set dummy values (won't be used)
                    memflow_lab = None
                    memflow_conf = None

            # Store MemFlow outputs if requested
            if return_memflow:
                # Remove batch dimension and store
                memflow_outputs.append(memflow_lab.squeeze(0))
                memflow_confs.append(memflow_conf.squeeze(0))

            # Remove batch dimension
            # Keep on device for training (gradient computation)
            # Move to CPU only during inference (when torch.no_grad() is active)
            if torch.is_grad_enabled():
                results.append(output_lab.squeeze(0))
            else:
                results.append(output_lab.squeeze(0).cpu())

        if return_memflow:
            return results, memflow_outputs, memflow_confs
        else:
            return results

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
        swintexco_ab, swintexco_sim = self.swintexco_inference(reference_pil, target_pil)

        # Construct complete SwinTExCo LAB prediction (for symmetry with MemFlow)
        swintexco_lab = torch.cat([L_channel, swintexco_ab], dim=1)

        # 3. Fusion UNet inference (trainable)
        # Note: fusion_unet returns AB channels only (2 channels)
        fused_ab = self.fusion_unet(
            memflow_lab,
            memflow_conf,
            swintexco_lab,
            swintexco_sim
        )

        # Construct complete LAB output by concatenating L channel
        fused_lab = torch.cat([L_channel, fused_ab], dim=1)

        return fused_lab

    def forward_with_memflow(self, frame_t, frame_t1, reference_pil, target_pil):
        """
        Complete forward pass that also returns MemFlow outputs (for temporal loss)

        Args:
            frame_t: [B, 3, H, W] LAB tensor (normalized)
            frame_t1: [B, 3, H, W] LAB tensor (normalized)
            reference_pil: PIL Image (RGB)
            target_pil: PIL Image (RGB)

        Returns:
            fused_lab: [B, 3, H, W] - Complete LAB prediction
            memflow_lab: [B, 3, H, W] - MemFlow LAB output
            memflow_conf: [B, 1, H, W] - MemFlow confidence
        """
        B, _, H, W = frame_t1.shape
        L_channel = frame_t1[:, 0:1, :, :]

        # 1. MemFlow inference (frozen)
        is_first = (self.curr_ti == -1)
        if is_first:
            # First frame: use zero placeholder
            memflow_lab = torch.zeros(B, 3, H, W, device=self.device)
            memflow_conf = torch.zeros(B, 1, H, W, device=self.device)
        else:
            # Subsequent frames: normal inference
            memflow_lab, memflow_conf = self.memflow_inference(frame_t, frame_t1)

        # 2. SwinTExCo inference (always valid)
        swintexco_ab, swintexco_sim = self.swintexco_inference(reference_pil, target_pil)

        # Construct complete SwinTExCo LAB prediction (for symmetry with MemFlow)
        swintexco_lab = torch.cat([L_channel, swintexco_ab], dim=1)

        # 3. Fusion UNet inference (trainable)
        # Note: fusion_unet returns AB channels only (2 channels)
        fused_ab = self.fusion_unet(
            memflow_lab,
            memflow_conf,
            swintexco_lab,
            swintexco_sim
        )

        # Construct complete LAB output by concatenating L channel
        fused_lab = torch.cat([L_channel, fused_ab], dim=1)

        return fused_lab, memflow_lab, memflow_conf

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
            self.swintexco.embed_net.eval()
            self.swintexco.nonlocal_net.train()
            self.swintexco.colornet.eval()
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
            #self.swintexco.embed_net.parameters(recurse=recurse),
            self.swintexco.nonlocal_net.parameters(recurse=recurse),
            #self.swintexco.colornet.parameters(recurse=recurse),
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
            #self.swintexco.embed_net.parameters(),
            self.swintexco.nonlocal_net.parameters(),
            #self.swintexco.colornet.parameters()
        ))
        fusion_params = list(self.fusion_unet.parameters())

        return [
            {'params': swintexco_params, 'name': 'swintexco'},
            {'params': fusion_params, 'name': 'fusion'}
        ]
