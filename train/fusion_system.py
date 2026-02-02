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
                 device='cuda',
                 freeze_swintexco=False):
        """
        Args:
            memflow_path: Path to MemFlow repository
            swintexco_path: Path to SwinSingle repository
            memflow_ckpt: Path to MemFlow checkpoint
            swintexco_ckpt: Path to SwinTExCo checkpoint directory
            fusion_net: Fusion network instance (default: PlaceholderFusion)
            device: cuda or cpu
            freeze_swintexco: bool, if True, freeze all SwinTExCo parameters (only train FusionNet)
        """
        super().__init__()

        self.device = device
        self.freeze_swintexco = freeze_swintexco

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
        """Load SwinTExCo model (trainable or frozen based on freeze_swintexco flag)"""
        try:
            if checkpoint_path is None:
                # Random initialization (no pretrained weights)
                print("  ⚠️  No SwinTExCo checkpoint provided - using random initialization")

                from src.models.vit.embed import SwinModel
                from src.models.CNN.NonlocalNet import WarpNet
                from src.models.CNN.ColorVidNet import ColorVidNet
                import torchvision.transforms as T
                from src.utils import RGB2Lab, ToTensor, Normalize

                # Create a simple wrapper class for consistency
                class SwinTExCoWrapper:
                    def __init__(self, device):
                        self.device = device
                        self.embed_net = SwinModel(pretrained_model='swinv2-cr-t-224', device=device).to(device)
                        self.nonlocal_net = WarpNet(feature_channel=128).to(device)
                        self.colornet = ColorVidNet(4).to(device)

                        # All start in eval mode (will be set to train later if needed)
                        self.embed_net.eval()
                        self.nonlocal_net.eval()
                        self.colornet.eval()

                        self.processor = T.Compose([
                            T.Resize((224,224)),
                            RGB2Lab(),
                            ToTensor(),
                            Normalize()
                        ])

                model = SwinTExCoWrapper(self.device)
                print("  ✅ SwinTExCo initialized with random weights (embed_net uses pretrained Swin backbone)")
            else:
                # Load from checkpoint
                from inference import SwinTExCo
                model = SwinTExCo(
                    weights_path=checkpoint_path,
                    device=self.device
                )
                print(f"  ✅ SwinTExCo loaded from checkpoint: {checkpoint_path}")

            # Set freeze/train state
            if self.freeze_swintexco:
                # Freeze all SwinTExCo components (only train FusionNet)
                for param in model.embed_net.parameters():
                    param.requires_grad = False
                for param in model.nonlocal_net.parameters():
                    param.requires_grad = False
                for param in model.colornet.parameters():
                    param.requires_grad = False

                # Set all to eval mode
                model.embed_net.eval()
                model.nonlocal_net.eval()
                model.colornet.eval()

                print("  SwinTExCo: ALL components frozen (embed_net, nonlocal_net, colornet)")
            else:
                # Partial unfreezing for joint training (default behavior)
                for param in model.embed_net.parameters():
                    param.requires_grad = False
                for param in model.nonlocal_net.parameters():
                    param.requires_grad = True
                for param in model.colornet.parameters():
                    param.requires_grad = False

                # Set to appropriate modes
                model.embed_net.eval()
                model.nonlocal_net.train()
                model.colornet.eval()

                print("  SwinTExCo: nonlocal_net trainable (embed_net, colornet frozen)")

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
        MemFlow inference (H/4 resolution output for 56×56 fusion)

        Args:
            frame_t: [B, 3, H, W] LAB tensor (normalized to [-1, 1])
            frame_t1: [B, 3, H, W] LAB tensor (normalized to [-1, 1])

        Returns:
            memflow_lab: [B, 3, H/4, W/4] - Complete LAB prediction at quarter resolution
            memflow_conf: [B, 1, H/4, W/4] - Confidence map at quarter resolution
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
            if self.memflow_memory is None:
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

            # Get final flow (already at H/4 resolution after 2x convex upsample)
            flow_final = flow_predictions[-1]  # [B, 2, H/4, W/4]

            # Get flow resolution (H/4, W/4)
            flow_h, flow_w = flow_final.shape[2:]

            # Downsample prev_ab to flow resolution (H/4) for warping
            H, W = frame_t1.shape[2:]
            prev_ab = frame_t[:, 1:3, :, :]  # [B, 2, H, W]
            prev_ab_down = nn.functional.interpolate(
                prev_ab,
                size=(flow_h, flow_w),
                mode='bilinear',
                align_corners=True
            )

            # Warp color at H/4 resolution (no additional flow scaling needed)
            from core.loss_new import warp_color_by_flow
            memflow_ab = warp_color_by_flow(prev_ab_down, flow_final)

            # Downsample current L channel to H/4
            current_L = frame_t1[:, 0:1, :, :]
            current_L_down = nn.functional.interpolate(
                current_L,
                size=(flow_h, flow_w),
                mode='bilinear',
                align_corners=True
            )

            # Combine to form complete LAB at H/4 resolution
            memflow_lab = torch.cat([current_L_down, memflow_ab], dim=1)

            # Downsample confidence map to match (should already be at correct resolution)
            if confidence_map.shape[2:] != (flow_h, flow_w):
                confidence_map = nn.functional.interpolate(
                    confidence_map,
                    size=(flow_h, flow_w),
                    mode='bilinear',
                    align_corners=True
                )

        return memflow_lab, confidence_map

    def swintexco_inference(self, reference_pil, target_pil, cached_ref_features=None, return_full_lab=False):
        """
        SwinTExCo inference - Using WarpNet (NonLocalNet) output directly (H/4 resolution)

        Skips ColorVidNet and uses WarpNet's output directly at 56×56 resolution.
        FusionNet will handle the refinement, combining it with MemFlow's temporal info.

        Args:
            reference_pil: PIL Image (RGB)
            target_pil: PIL Image (RGB)
            cached_ref_features: Optional tuple (ref_lab, features_B) for feature caching
                                If provided, skips reference feature extraction (25% faster)
            return_full_lab: If True, return complete LAB (for debugging/visualization)

        Returns:
            If return_full_lab=False (default, training):
                warpnet_ab: [B, 2, H/4, W/4] - NonLocalNet AB output at 56×56 (normalized to [-1, 1])
                similarity_map: [B, 1, H/4, W/4] - Feature similarity map at 56×56
            If return_full_lab=True (inference with intermediates):
                swintexco_lab: [B, 3, H/4, W/4] - Complete LAB prediction at 56×56
                similarity_map: [B, 1, H/4, W/4] - Feature similarity map at 56×56

        Note:
            Output is at H/4 (56×56) resolution to match MemFlow downsample output.
            FusionNet processes at 56×56, then upsamples to full resolution if needed.
        """
        # Disable autocast for SwinTExCo (not compatible with mixed precision)
        # Note: Do NOT use torch.no_grad() here during training, as SwinTExCo is trainable
        with autocast(enabled=False):
            # Use cached reference features if provided (optimization for sequences)
            if cached_ref_features is not None:
                ref_lab, features_B = cached_ref_features
            else:
                # Process reference (only if not cached)
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

            # Return format depends on use case
            if return_full_lab:
                # Return complete LAB prediction (for inference/debugging)
                swintexco_lab = nonlocal_BA_lab  # Already includes L channel
                return swintexco_lab, similarity_map
            else:
                # Extract AB channels from WarpNet output (for training)
                warpnet_ab = nonlocal_BA_lab[:, 1:3, :, :]
                return warpnet_ab, similarity_map

    def forward_sequence(self, frames_lab, frames_pil, reference_pil, return_memflow=False):
        """
        Process a sequence of frames with shared reference

        Args:
            frames_lab: list of [3, H, W] LAB tensors (normalized to [-1, 1])
            frames_pil: list of PIL Images (RGB, target frames)
            reference_pil: Single PIL Image (RGB, shared reference for entire sequence)
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

        # Cache reference features once for the entire sequence (computed on first frame)
        cached_ref_features = None

        for i in range(len(frames_lab)):
            # Convert to batch [1, 3, H, W]
            frame_t1_batch = frames_lab[i].unsqueeze(0).to(self.device)

            if i == 0:
                # First frame: manually construct output without calling forward()
                # to avoid state management issues
                B, _, H, W = frame_t1_batch.shape
                H_small, W_small = H // 4, W // 4  # 56×56 for 224 input

                L_channel = frame_t1_batch[:, 0:1, :, :]  # Full resolution for final output

                # Downsample L channel for FusionNet input (56×56)
                L_channel_small = nn.functional.interpolate(
                    L_channel, size=(H_small, W_small), mode='bilinear', align_corners=True
                )

                # MemFlow: output zero at 56×56 (no temporal info for first frame)
                memflow_lab = torch.zeros(B, 3, H_small, W_small, device=self.device)
                memflow_conf = torch.zeros(B, 1, H_small, W_small, device=self.device)

                # Compute and cache reference features (only once per sequence)
                with autocast(enabled=False):
                    from src.utils import uncenter_l, uncenter_ab, tensor_lab2rgb
                    ref_lab = self.swintexco.processor(reference_pil).unsqueeze(0).to(self.device)
                    ref_l = ref_lab[:, 0:1, :, :]
                    ref_ab = ref_lab[:, 1:3, :, :]
                    ref_rgb = tensor_lab2rgb(torch.cat([uncenter_l(ref_l), uncenter_ab(ref_ab)], dim=1))
                    features_B = self.swintexco.embed_net(ref_rgb)
                    cached_ref_features = (ref_lab, features_B)

                # SwinTExCo: process reference-based colorization (outputs 56×56)
                swintexco_ab, swintexco_sim = self.swintexco_inference(
                    reference_pil,
                    frames_pil[i],
                    cached_ref_features=cached_ref_features
                )

                # Construct complete SwinTExCo LAB prediction at 56×56
                swintexco_lab = torch.cat([L_channel_small, swintexco_ab], dim=1)

                # FusionNet: fuse results at 56×56 (returns AB channels only)
                fused_ab = self.fusion_unet(
                    memflow_lab,
                    memflow_conf,
                    swintexco_lab,
                    swintexco_sim
                )

                # Upsample fused_ab to full resolution (224×224)
                fused_ab_full = nn.functional.interpolate(
                    fused_ab, size=(H, W), mode='bilinear', align_corners=True
                )

                # Construct complete LAB output at full resolution
                output_lab = torch.cat([L_channel, fused_ab_full], dim=1)

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
                        reference_pil,
                        frames_pil[i],
                        cached_ref_features=cached_ref_features
                    )
                else:
                    output_lab = self.forward(
                        frame_t_batch,
                        frame_t1_batch,
                        reference_pil,
                        frames_pil[i],
                        cached_ref_features=cached_ref_features
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

    def forward_single_frame(self, frame_t, frame_t1, reference_pil, target_pil, is_first=False,
                            cached_ref_features=None, return_intermediates=False):
        """
        Process a single frame (used within sequence processing)

        Args:
            frame_t: [B, 3, H, W] LAB tensor (previous frame, None if first)
            frame_t1: [B, 3, H, W] LAB tensor (current frame)
            reference_pil: PIL Image (RGB)
            target_pil: PIL Image (RGB)
            is_first: bool, whether this is the first frame
            cached_ref_features: Optional tuple (ref_lab, features_B) for SwinTExCo optimization
            return_intermediates: bool, if True, return dict with intermediate results

        Returns:
            If return_intermediates=False (default):
                fused_lab: [B, 3, H, W] - Complete LAB prediction at full resolution
            If return_intermediates=True:
                dict with keys:
                    'fused': [B, 3, H, W] - Final fused result at full resolution
                    'memflow': [B, 3, H/4, W/4] - MemFlow prediction at 56×56
                    'swintexco': [B, 3, H/4, W/4] - SwinTExCo prediction at 56×56
                    'memflow_conf': [B, 1, H/4, W/4] - MemFlow confidence at 56×56
                    'swintexco_sim': [B, 1, H/4, W/4] - SwinTExCo similarity at 56×56
        """
        B, _, H, W = frame_t1.shape
        H_small, W_small = H // 4, W // 4  # 56×56 for 224 input

        L_channel = frame_t1[:, 0:1, :, :]  # Full resolution for final output

        # Downsample L channel for FusionNet input (56×56)
        L_channel_small = nn.functional.interpolate(
            L_channel, size=(H_small, W_small), mode='bilinear', align_corners=True
        )

        # 1. MemFlow inference (frozen) - outputs at 56×56
        if is_first:
            # First frame: use zero placeholder at 56×56
            memflow_lab = torch.zeros(B, 3, H_small, W_small, device=self.device)
            memflow_conf = torch.zeros(B, 1, H_small, W_small, device=self.device)
        else:
            # Subsequent frames: normal inference (already outputs 56×56)
            memflow_lab, memflow_conf = self.memflow_inference(frame_t, frame_t1)

        # 2. SwinTExCo inference (always valid) - outputs at 56×56
        # Use cached features if provided (optimization for batch inference)
        result = self.swintexco_inference(
            reference_pil,
            target_pil,
            cached_ref_features=cached_ref_features,
            return_full_lab=return_intermediates  # Return full LAB if we need intermediates
        )

        if return_intermediates:
            swintexco_lab, swintexco_sim = result
        else:
            swintexco_ab, swintexco_sim = result
            # Construct complete SwinTExCo LAB prediction at 56×56
            swintexco_lab = torch.cat([L_channel_small, swintexco_ab], dim=1)

        # 3. Fusion UNet inference (trainable) at 56×56
        # Note: fusion_unet returns AB channels only (2 channels)
        fused_ab = self.fusion_unet(
            memflow_lab,
            memflow_conf,
            swintexco_lab,
            swintexco_sim
        )

        # Upsample fused_ab to full resolution (224×224)
        fused_ab_full = nn.functional.interpolate(
            fused_ab, size=(H, W), mode='bilinear', align_corners=True
        )

        # Construct complete LAB output at full resolution
        fused_lab = torch.cat([L_channel, fused_ab_full], dim=1)

        # Return format based on return_intermediates flag
        if return_intermediates:
            return {
                'fused': fused_lab,
                'memflow': memflow_lab,
                'swintexco': swintexco_lab,
                'memflow_conf': memflow_conf,
                'swintexco_sim': swintexco_sim
            }
        else:
            return fused_lab

    def forward_with_memflow(self, frame_t, frame_t1, reference_pil, target_pil, cached_ref_features=None):
        """
        Complete forward pass that also returns MemFlow outputs (for temporal loss)

        Args:
            frame_t: [B, 3, H, W] LAB tensor (normalized)
            frame_t1: [B, 3, H, W] LAB tensor (normalized)
            reference_pil: PIL Image (RGB)
            target_pil: PIL Image (RGB)
            cached_ref_features: Optional tuple (ref_lab, features_B) for feature caching

        Returns:
            fused_lab: [B, 3, H, W] - Complete LAB prediction at full resolution
            memflow_lab: [B, 3, H/4, W/4] - MemFlow LAB output at 56×56
            memflow_conf: [B, 1, H/4, W/4] - MemFlow confidence at 56×56
        """
        B, _, H, W = frame_t1.shape
        H_small, W_small = H // 4, W // 4  # 56×56 for 224 input

        L_channel = frame_t1[:, 0:1, :, :]  # Full resolution for final output

        # Downsample L channel for FusionNet input (56×56)
        L_channel_small = nn.functional.interpolate(
            L_channel, size=(H_small, W_small), mode='bilinear', align_corners=True
        )

        # 1. MemFlow inference (frozen) - outputs at 56×56
        is_first = (self.curr_ti == -1)
        if is_first:
            # First frame: use zero placeholder at 56×56
            memflow_lab = torch.zeros(B, 3, H_small, W_small, device=self.device)
            memflow_conf = torch.zeros(B, 1, H_small, W_small, device=self.device)
        else:
            # Subsequent frames: normal inference (already outputs 56×56)
            memflow_lab, memflow_conf = self.memflow_inference(frame_t, frame_t1)

        # 2. SwinTExCo inference (always valid) - outputs at 56×56
        swintexco_ab, swintexco_sim = self.swintexco_inference(
            reference_pil, target_pil, cached_ref_features=cached_ref_features
        )

        # Construct complete SwinTExCo LAB prediction at 56×56
        swintexco_lab = torch.cat([L_channel_small, swintexco_ab], dim=1)

        # 3. Fusion UNet inference (trainable) at 56×56
        # Note: fusion_unet returns AB channels only (2 channels)
        fused_ab = self.fusion_unet(
            memflow_lab,
            memflow_conf,
            swintexco_lab,
            swintexco_sim
        )

        # Upsample fused_ab to full resolution (224×224)
        fused_ab_full = nn.functional.interpolate(
            fused_ab, size=(H, W), mode='bilinear', align_corners=True
        )

        # Construct complete LAB output at full resolution
        fused_lab = torch.cat([L_channel, fused_ab_full], dim=1)

        return fused_lab, memflow_lab, memflow_conf

    def forward(self, frame_t, frame_t1, reference_pil, target_pil, cached_ref_features=None):
        """
        Complete forward pass (legacy interface for 2-frame processing)

        Args:
            frame_t: [B, 3, H, W] LAB tensor (normalized)
            frame_t1: [B, 3, H, W] LAB tensor (normalized)
            reference_pil: PIL Image (RGB)
            target_pil: PIL Image (RGB)
            cached_ref_features: Optional tuple (ref_lab, features_B) for feature caching

        Returns:
            fused_lab: [B, 3, H, W] - Complete LAB prediction
        """
        return self.forward_single_frame(
            frame_t, frame_t1, reference_pil, target_pil,
            is_first=(self.curr_ti == -1),
            cached_ref_features=cached_ref_features
        )

    def train(self, mode=True):
        """Override train to affect SwinTExCo and Fusion UNet"""
        if mode:
            # Train mode
            if self.freeze_swintexco:
                # SwinTExCo fully frozen - only train FusionNet
                self.swintexco.embed_net.eval()
                self.swintexco.nonlocal_net.eval()
                self.swintexco.colornet.eval()
            else:
                # Default: partial SwinTExCo training (nonlocal_net only)
                self.swintexco.embed_net.eval()
                self.swintexco.nonlocal_net.train()
                self.swintexco.colornet.eval()
            self.fusion_unet.train()
        else:
            # Eval mode - freeze everything
            self.swintexco.embed_net.eval()
            self.swintexco.nonlocal_net.eval()
            self.swintexco.colornet.eval()
            self.fusion_unet.eval()
        # MemFlow always stays in eval mode
        return self

    def parameters(self, recurse=True):
        """Override to return trainable parameters based on freeze_swintexco flag"""
        import itertools
        if self.freeze_swintexco:
            # Only FusionNet parameters (SwinTExCo fully frozen)
            return self.fusion_unet.parameters(recurse=recurse)
        else:
            # Default: SwinTExCo nonlocal_net + FusionNet parameters
            return itertools.chain(
                self.swintexco.nonlocal_net.parameters(recurse=recurse),
                self.fusion_unet.parameters(recurse=recurse)
            )

    def get_parameter_groups(self):
        """
        Get parameter groups for different learning rates

        Returns:
            If freeze_swintexco=False:
                list: [{'params': swintexco_params, 'name': 'swintexco'},
                       {'params': fusion_params, 'name': 'fusion'}]
            If freeze_swintexco=True:
                list: [{'params': fusion_params, 'name': 'fusion'}]
        """
        fusion_params = list(self.fusion_unet.parameters())

        if self.freeze_swintexco:
            # Only FusionNet parameters
            return [
                {'params': fusion_params, 'name': 'fusion'}
            ]
        else:
            # Default: SwinTExCo nonlocal_net + FusionNet parameters
            import itertools
            swintexco_params = list(self.swintexco.nonlocal_net.parameters())
            return [
                {'params': swintexco_params, 'name': 'swintexco'},
                {'params': fusion_params, 'name': 'fusion'}
            ]
