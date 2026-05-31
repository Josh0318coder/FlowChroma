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
                 freeze_swintexco=False,
                 no_memflow=False):
        """
        Args:
            memflow_path: Path to MemFlow repository
            swintexco_path: Path to SwinSingle repository
            memflow_ckpt: Path to MemFlow checkpoint
            swintexco_ckpt: Path to SwinTExCo checkpoint directory
            fusion_net: Fusion network instance (default: PlaceholderFusion)
            device: cuda or cpu
            freeze_swintexco: bool, if True, freeze all SwinTExCo parameters (only train FusionNet)
            no_memflow: bool, ablation flag — replace MemFlow with downsampled prev prediction (conf=0)
        """
        super().__init__()

        self.device = device
        self.freeze_swintexco = freeze_swintexco
        self.no_memflow = no_memflow

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
                from inference import SwinTExCoWithVisualization as SwinTExCo
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
        self.memflow_keys = None
        self.curr_ti = -1
        # fnet feature cache: fnet_cache_t = previous frame's raw SwinV2 stage1 (for MemFlow frame_t)
        #                      fnet_cache_t1 = current frame's raw SwinV2 stage1 (set by swintexco_inference)
        self.fnet_cache_t = None
        self.fnet_cache_t1 = None

    def _lab_to_gray_rgb(self, frame_lab_norm):
        """
        LAB normalized [-1,1] → [0,1] grayscale RGB (3 channels)
        swinv2_tiny applies ImageNet normalization internally.
        """
        x = frame_lab_norm.float()
        L_01 = (x[:, 0:1] + 1.0) * 0.5   # L: [-1, 1] → [0, 1]
        return L_01.expand(-1, 3, -1, -1).contiguous()

    def _lab_to_rgb(self, frame_lab_norm):
        """
        LAB normalized [-1,1] → [0,1] colorized RGB
        swinv2_tiny applies ImageNet normalization internally.
        """
        x = frame_lab_norm.float()

        L = (x[:, 0:1] + 1.0) * 50.0
        a = x[:, 1:2] * 127.0
        b = x[:, 2:3] * 127.0

        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0

        t = 0.20689655
        X = torch.where(fx > t, fx ** 3, (fx - 16.0 / 116.0) / 7.787) * 0.95047
        Y = torch.where(L > 7.9996, ((L + 16.0) / 116.0) ** 3, L / 903.3)
        Z = torch.where(fz > t, fz ** 3, (fz - 16.0 / 116.0) / 7.787) * 1.08883

        R  =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
        G  = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
        B_ =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z

        def _gamma(t_val):
            return torch.where(
                t_val <= 0.0031308,
                12.92 * t_val,
                1.055 * t_val.clamp(min=1e-10) ** (1.0 / 2.4) - 0.055
            )

        rgb = torch.cat([_gamma(R), _gamma(G), _gamma(B_)], dim=1).clamp(0.0, 1.0)
        return rgb

    def memflow_inference(self, frame_t, frame_t1, precomputed_fmaps=None):
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

        # Build precomputed fmaps if both frames' SwinV2 features are cached
        # (fnet_cache_t = previous frame's raw stage1, fnet_cache_t1 = current frame from swintexco_inference)
        if precomputed_fmaps is None and self.fnet_cache_t is not None and self.fnet_cache_t1 is not None:
            with torch.no_grad():
                fmap_t  = self.memflow.channel_convertor(self.fnet_cache_t.float())
                fmap_t1 = self.memflow.channel_convertor(self.fnet_cache_t1.float())
            precomputed_fmaps = torch.stack([fmap_t, fmap_t1], dim=1)  # [B, 2, 256, H/8, W/8]
        self.fnet_cache_t = self.fnet_cache_t1  # advance cache: current t1 becomes next t

        with torch.no_grad(), autocast(enabled=True, dtype=torch.bfloat16):
            # Convert LAB [-1,1] to the format expected by pre-trained MemFlow encoders
            gray_t  = self._lab_to_gray_rgb(frame_t)   # [B, 3, H, W]
            gray_t1 = self._lab_to_gray_rgb(frame_t1)  # [B, 3, H, W]
            rgb_inputs = torch.stack([gray_t, gray_t1], dim=1)   # [B, 2, 3, H, W] (used for coords only when cache hits)

            # cnet: [0,1] colorized RGB of the source frame
            cnet_input = self._lab_to_rgb(frame_t)  # [B, 3, H, W]

            # Encode context (source frame colorized RGB → cnet)
            query, key, net, inp = self.memflow.encode_context(cnet_input)

            # Encode features: skip fnet if precomputed_fmaps available (SwinV2 already ran in swintexco_inference)
            coords0, coords1, fmaps = self.memflow.encode_features(rgb_inputs, precomputed_fmaps=precomputed_fmaps)

            # Memory management: keys and values stored separately
            if self.memflow_keys is None:
                ref_values = None
                ref_keys   = key.unsqueeze(2)
            else:
                ref_values = self.memflow_memory
                ref_keys   = torch.cat([self.memflow_keys, key.unsqueeze(2)], dim=2)

            # Predict flow
            flow_predictions, current_value, confidence_map = self.memflow.predict_flow(
                net, inp, coords0, coords1, fmaps,
                query.unsqueeze(2), ref_keys, ref_values
            )

            # Update memory (keys and values stored separately)
            if self.memflow_keys is None:
                self.memflow_keys   = key.unsqueeze(2)
                self.memflow_memory = current_value
            else:
                self.memflow_keys   = key.unsqueeze(2)
                self.memflow_memory = current_value

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

    def swintexco_inference(self, ref_lab_tensor, target_lab_tensor, cached_ref_features=None,
                            cached_B_warp_features=None, return_full_lab=False):
        """
        SwinTExCo inference - Using WarpNet (NonLocalNet) output directly (full resolution)

        Skips ColorVidNet and uses WarpNet's output directly at 224×224 resolution.
        FusionNet processes at full resolution together with upsampled MemFlow output.

        Args:
            ref_lab_tensor: [B, 3, H, W] tensor (pre-processed by swintexco.processor)
            target_lab_tensor: [B, 3, H, W] tensor (pre-processed by swintexco.processor)
            cached_ref_features: Optional tuple (ref_lab, features_B) for feature caching
                                If provided, skips reference feature extraction (25% faster)
            return_full_lab: If True, return complete LAB (for debugging/visualization)

        Returns:
            If return_full_lab=False (default, training):
                warpnet_ab: [B, 2, H, W] - NonLocalNet AB output at 224×224 (normalized to [-1, 1])
                similarity_map: [B, 1, H, W] - Feature similarity map at 224×224
            If return_full_lab=True (inference with intermediates):
                swintexco_lab: [B, 3, H, W] - Complete LAB prediction at 224×224
                similarity_map: [B, 1, H, W] - Feature similarity map at 224×224
        """
        # Disable autocast for SwinTExCo (not compatible with mixed precision)
        # Note: Do NOT use torch.no_grad() here during training, as SwinTExCo is trainable
        with autocast(enabled=False):
            # Use cached reference features if provided (optimization for sequences)
            if cached_ref_features is not None:
                ref_lab, features_B = cached_ref_features
            else:
                # Process reference (only if not cached)
                # Support both [3, H, W] (single, inference) and [B, 3, H, W] (batched, training)
                ref_lab = ref_lab_tensor.to(self.device)
                if ref_lab.dim() == 3:
                    ref_lab = ref_lab.unsqueeze(0)

                # Get reference features
                from src.utils import uncenter_l, uncenter_ab, tensor_lab2rgb
                ref_l = ref_lab[:, 0:1, :, :]
                ref_ab = ref_lab[:, 1:3, :, :]
                # NOTE: uncenter_ab matches newsingle pre-training (train_fixseed.py)
                ref_rgb = tensor_lab2rgb(torch.cat([uncenter_l(ref_l), uncenter_ab(ref_ab)], dim=1))
                features_B = self.swintexco.embed_net(ref_rgb)

            # Process target
            # Support both [3, H, W] (single, inference) and [B, 3, H, W] (batched, training)
            target_lab = target_lab_tensor.to(self.device)
            if target_lab.dim() == 3:
                target_lab = target_lab.unsqueeze(0)
            target_l = target_lab[:, 0:1, :, :]

            # Compute A-side (target) embed_net explicitly to capture raw_stage1 for MemFlow fnet sharing
            from src.utils import gray2rgb_batch
            target_gray = gray2rgb_batch(target_l)
            with torch.no_grad():
                features_A, raw_stage1_A = self.swintexco.embed_net(target_gray, return_raw_stage1=True)
            self.fnet_cache_t1 = raw_stage1_A  # will be used as frame_t in next frame's MemFlow call

            # Call warp_color with precomputed A-side features (skip embed_net call inside)
            from src.models.CNN.FrameColor import warp_color
            nonlocal_BA_lab, similarity_map = warp_color(
                target_l,
                ref_lab,
                features_B,
                self.swintexco.embed_net,
                self.swintexco.nonlocal_net,
                temperature=1e-10, #1e-10
                cached_B_features=cached_B_warp_features,
                features_A=features_A,
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

    def forward_sequence(self, frames_lab, frames_swintexco_lab, reference_swintexco_lab, return_memflow=False):
        """
        Process a sequence of frames with shared reference (true batching: all sequences in parallel)

        Args:
            frames_lab: list of [B, 3, H, W] LAB tensors on device (normalized to [-1, 1])
            frames_swintexco_lab: list of [B, 3, H, W] tensors (pre-processed by swintexco.processor)
            reference_swintexco_lab: [B, 3, H, W] tensor (pre-processed by swintexco.processor)
            return_memflow: bool, whether to return MemFlow/SwinTExCo outputs (for loss in training)

        Returns:
            If return_memflow=False:
                results: list of [B, 3, H, W] LAB tensors (colorized results)
            If return_memflow=True:
                (results, memflow_outputs, memflow_confs, swintexco_outputs, swintexco_confs,
                 memflow_outputs_small, memflow_confs_small): tuple of lists of [B, ...] tensors
        """
        # Reset memory at the start of sequence
        self.reset_memory()

        results = []
        memflow_outputs = [] if return_memflow else None
        memflow_confs = [] if return_memflow else None
        memflow_outputs_small = [] if return_memflow else None
        memflow_confs_small = [] if return_memflow else None
        swintexco_outputs = [] if return_memflow else None
        swintexco_confs = [] if return_memflow else None

        # Cache reference features once for the entire sequence (computed on first frame)
        cached_ref_features = None
        cached_B_warp_features = None

        for i in range(len(frames_lab)):
            # frames_lab[i] is already [B, 3, H, W] on device (pre-stacked and transferred)
            frame_t1_batch = frames_lab[i]

            if i == 0:
                # First frame: manually construct output without calling forward()
                # to avoid state management issues
                B, _, H, W = frame_t1_batch.shape

                L_channel = frame_t1_batch[:, 0:1, :, :]  # Full resolution

                # Compute and cache reference features once for the entire sequence
                with autocast(enabled=False):
                    from src.utils import uncenter_l, uncenter_ab, tensor_lab2rgb, feature_normalize
                    ref_lab = reference_swintexco_lab.to(self.device)  # [B, 3, H, W]
                    ref_l = ref_lab[:, 0:1, :, :]
                    ref_ab = ref_lab[:, 1:3, :, :]
                    # NOTE: uncenter_ab matches newsingle pre-training (train_fixseed.py)
                    ref_rgb = tensor_lab2rgb(torch.cat([uncenter_l(ref_l), uncenter_ab(ref_ab)], dim=1))
                    features_B = self.swintexco.embed_net(ref_rgb)
                    cached_ref_features = (ref_lab, features_B)

                    # Pre-compute WarpNet B-side features (phi and B_lab are constant for fixed reference)
                    B_f0, B_f1, B_f2, B_f3 = features_B
                    cached_B_warp_features = self.swintexco.nonlocal_net.precompute_B_features(
                        ref_lab,
                        feature_normalize(B_f0),
                        feature_normalize(B_f1),
                        feature_normalize(B_f2),
                        feature_normalize(B_f3),
                    )

                # SwinTExCo: process reference-based colorization (outputs 224×224)
                swintexco_ab, swintexco_sim = self.swintexco_inference(
                    reference_swintexco_lab,
                    frames_swintexco_lab[i],
                    cached_ref_features=cached_ref_features,
                    cached_B_warp_features=cached_B_warp_features,
                )

                # Construct complete SwinTExCo LAB prediction at 224×224
                swintexco_lab = torch.cat([L_channel, swintexco_ab], dim=1)

                # Advance fnet cache: frame 0's t1 features become frame 1's t features
                self.fnet_cache_t = self.fnet_cache_t1

                # MemFlow: use SwinTExCo output as substitute (detached) for first frame
                # conf=0 signals FusionNet this is not real MemFlow
                memflow_lab = swintexco_lab.detach()
                memflow_conf = torch.zeros(B, 1, H, W, device=self.device)
                if return_memflow:
                    memflow_lab_small = nn.functional.interpolate(
                        swintexco_lab.detach(), size=(H // 4, W // 4), mode='bilinear', align_corners=True
                    )
                    memflow_conf_small = torch.zeros(B, 1, H // 4, W // 4, device=self.device)

                # FusionNet: predict delta AB correction on top of SwinTExCo
                delta_ab = self.fusion_unet(
                    memflow_lab,
                    memflow_conf,
                    swintexco_lab,
                    swintexco_sim
                )
                fused_ab = torch.tanh(swintexco_ab + delta_ab)

                # Construct complete LAB output at full resolution
                output_lab = torch.cat([L_channel, fused_ab], dim=1)

                # After first frame, increment curr_ti to 0 (so next frame is not treated as first)
                self.curr_ti += 1
            else:
                # Subsequent frames: use PREVIOUS PREDICTION (not GT)
                # This enables error accumulation training (like real inference)

                # Use previous frame's prediction as input
                prev_output = results[-1]  # [B, 3, H, W] from previous iteration

                # Detach to prevent gradient backprop through entire sequence
                frame_t_batch = prev_output.detach()

                if self.no_memflow:
                    # Ablation: skip real MemFlow inference
                    # Replace MemFlow with downsampled previous prediction at 56×56, conf=0
                    # FusionNet sees identical input structure — only MemFlow's temporal
                    # inference is removed; everything else (SwinTExCo, FusionNet) is unchanged.
                    B, _, H, W = frame_t1_batch.shape
                    L_channel = frame_t1_batch[:, 0:1, :, :]

                    swintexco_ab, swintexco_sim = self.swintexco_inference(
                        reference_swintexco_lab,
                        frames_swintexco_lab[i],
                        cached_ref_features=cached_ref_features,
                        cached_B_warp_features=cached_B_warp_features,
                    )
                    swintexco_lab = torch.cat([L_channel, swintexco_ab], dim=1)

                    # Downsample prev prediction to MemFlow native resolution (H/4, W/4)
                    memflow_lab_small = nn.functional.interpolate(
                        frame_t_batch, size=(H // 4, W // 4), mode='bilinear', align_corners=True
                    )
                    memflow_conf_small = torch.zeros(B, 1, H // 4, W // 4, device=self.device)
                    memflow_lab = nn.functional.interpolate(
                        memflow_lab_small, size=(H, W), mode='bilinear', align_corners=True
                    )
                    memflow_conf = torch.zeros(B, 1, H, W, device=self.device)

                    delta_ab = self.fusion_unet(memflow_lab, memflow_conf, swintexco_lab, swintexco_sim)
                    fused_ab = torch.tanh(swintexco_ab + delta_ab)
                    output_lab = torch.cat([L_channel, fused_ab], dim=1)

                elif return_memflow:
                    # Normal path with MemFlow, return intermediates for loss
                    output_lab, memflow_lab, memflow_conf, swintexco_ab, swintexco_sim, memflow_lab_small, memflow_conf_small = self.forward_with_memflow(
                        frame_t_batch,
                        frame_t1_batch,
                        reference_swintexco_lab,
                        frames_swintexco_lab[i],
                        cached_ref_features=cached_ref_features,
                        cached_B_warp_features=cached_B_warp_features,
                    )
                else:
                    # Normal path with MemFlow, no intermediates needed
                    output_lab = self.forward(
                        frame_t_batch,
                        frame_t1_batch,
                        reference_swintexco_lab,
                        frames_swintexco_lab[i],
                        cached_ref_features=cached_ref_features,
                        cached_B_warp_features=cached_B_warp_features,
                    )
                    memflow_lab = None
                    memflow_conf = None
                    swintexco_ab = None
                    swintexco_sim = None

            # Store MemFlow and SwinTExCo outputs if requested
            if return_memflow:
                # Store [B, ...] tensors directly (batch dimension preserved)
                memflow_outputs.append(memflow_lab)
                memflow_confs.append(memflow_conf)
                memflow_outputs_small.append(memflow_lab_small)
                memflow_confs_small.append(memflow_conf_small)
                swintexco_outputs.append(swintexco_ab)
                swintexco_confs.append(swintexco_sim)

            # Keep on device for training (gradient computation)
            # Move to CPU only during inference (when torch.no_grad() is active)
            if torch.is_grad_enabled():
                results.append(output_lab)
            else:
                results.append(output_lab.cpu())

        if return_memflow:
            return results, memflow_outputs, memflow_confs, swintexco_outputs, swintexco_confs, memflow_outputs_small, memflow_confs_small
        else:
            return results

    def forward_single_frame(self, frame_t, frame_t1, reference_pil, target_pil, is_first=False,
                            cached_ref_features=None, cached_B_warp_features=None,
                            return_intermediates=False):
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

        L_channel = frame_t1[:, 0:1, :, :]  # Full resolution

        # 1. SwinTExCo inference (always valid) - outputs at 224×224
        # Use cached features if provided (optimization for batch inference)
        # Convert PIL images to LAB tensors for swintexco_inference
        target_swintexco_lab = self.swintexco.processor(target_pil)
        ref_swintexco_lab = None if cached_ref_features is not None else self.swintexco.processor(reference_pil)

        result = self.swintexco_inference(
            ref_swintexco_lab,
            target_swintexco_lab,
            cached_ref_features=cached_ref_features,
            cached_B_warp_features=cached_B_warp_features,
            return_full_lab=return_intermediates  # Return full LAB if we need intermediates
        )

        if return_intermediates:
            swintexco_lab, swintexco_sim = result
            swintexco_ab = swintexco_lab[:, 1:3, :, :]  # extract AB from full LAB
        else:
            swintexco_ab, swintexco_sim = result
            # Construct complete SwinTExCo LAB prediction at 224×224
            swintexco_lab = torch.cat([L_channel, swintexco_ab], dim=1)

        # 2. MemFlow inference (frozen) - outputs at 56×56, upsample to 224×224
        if is_first:
            # First frame: use SwinTExCo output as substitute (detached)
            # conf=0 signals FusionNet this is not real MemFlow
            memflow_lab = swintexco_lab.detach()
            memflow_conf = torch.zeros(B, 1, H, W, device=self.device)
        else:
            # Subsequent frames: inference at 56×56, then upsample to 224×224
            memflow_lab_small, memflow_conf_small = self.memflow_inference(frame_t, frame_t1)
            memflow_lab = nn.functional.interpolate(
                memflow_lab_small, size=(H, W), mode='bilinear', align_corners=True
            )
            memflow_conf = nn.functional.interpolate(
                memflow_conf_small, size=(H, W), mode='bilinear', align_corners=True
            )

        # 3. Fusion UNet inference (trainable) at 224×224
        # fusion_unet returns delta AB correction on top of SwinTExCo
        delta_ab = self.fusion_unet(
            memflow_lab,
            memflow_conf,
            swintexco_lab,
            swintexco_sim
        )
        fused_ab = torch.tanh(swintexco_ab + delta_ab)

        # Construct complete LAB output at full resolution
        fused_lab = torch.cat([L_channel, fused_ab], dim=1)

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

    def forward_with_memflow(self, frame_t, frame_t1, ref_lab_tensor, target_lab_tensor,
                             cached_ref_features=None, cached_B_warp_features=None):
        """
        Complete forward pass that also returns MemFlow and SwinTExCo outputs (for loss computation)

        Args:
            frame_t: [B, 3, H, W] LAB tensor (normalized)
            frame_t1: [B, 3, H, W] LAB tensor (normalized)
            ref_lab_tensor: [3, H, W] CPU tensor (pre-processed by swintexco.processor)
            target_lab_tensor: [3, H, W] CPU tensor (pre-processed by swintexco.processor)
            cached_ref_features: Optional tuple (ref_lab, features_B) for feature caching
            cached_B_warp_features: Optional tuple (phi, B_lab) pre-computed by WarpNet.precompute_B_features

        Returns:
            fused_lab: [B, 3, H, W] - Complete LAB prediction at full resolution
            memflow_lab: [B, 3, H/4, W/4] - MemFlow LAB output at 56×56
            memflow_conf: [B, 1, H/4, W/4] - MemFlow confidence at 56×56
            swintexco_ab: [B, 2, H/4, W/4] - SwinTExCo AB output at 56×56
            swintexco_conf: [B, 1, H/4, W/4] - SwinTExCo confidence at 56×56
        """
        B, _, H, W = frame_t1.shape

        L_channel = frame_t1[:, 0:1, :, :]  # Full resolution

        # 1. SwinTExCo inference (always valid) - outputs at 224×224
        swintexco_ab, swintexco_sim = self.swintexco_inference(
            ref_lab_tensor, target_lab_tensor,
            cached_ref_features=cached_ref_features,
            cached_B_warp_features=cached_B_warp_features,
        )

        # Construct complete SwinTExCo LAB prediction at 224×224
        swintexco_lab = torch.cat([L_channel, swintexco_ab], dim=1)

        # 2. MemFlow inference (frozen) - outputs at 56×56, upsample to 224×224
        is_first = (self.curr_ti == -1)
        if is_first:
            # First frame: use SwinTExCo output as substitute (detached)
            # conf=0 signals FusionNet this is not real MemFlow
            memflow_lab = swintexco_lab.detach()
            memflow_conf = torch.zeros(B, 1, H, W, device=self.device)
            # Small versions for loss at native resolution (no upsampling artifacts)
            memflow_lab_small = nn.functional.interpolate(
                swintexco_lab.detach(), size=(H // 4, W // 4), mode='bilinear', align_corners=True
            )
            memflow_conf_small = torch.zeros(B, 1, H // 4, W // 4, device=self.device)
        else:
            # Subsequent frames: inference at 56×56, then upsample to 224×224
            memflow_lab_small, memflow_conf_small = self.memflow_inference(frame_t, frame_t1)
            memflow_lab = nn.functional.interpolate(
                memflow_lab_small, size=(H, W), mode='bilinear', align_corners=True
            )
            memflow_conf = nn.functional.interpolate(
                memflow_conf_small, size=(H, W), mode='bilinear', align_corners=True
            )

        # 3. Fusion UNet inference (trainable) at 224×224
        # fusion_unet returns delta AB correction on top of SwinTExCo
        delta_ab = self.fusion_unet(
            memflow_lab,
            memflow_conf,
            swintexco_lab,
            swintexco_sim
        )
        fused_ab = torch.tanh(swintexco_ab + delta_ab)

        # Construct complete LAB output at full resolution
        fused_lab = torch.cat([L_channel, fused_ab], dim=1)

        return fused_lab, memflow_lab, memflow_conf, swintexco_ab, swintexco_sim, memflow_lab_small, memflow_conf_small

    def forward(self, frame_t, frame_t1, reference_pil, target_pil,
                cached_ref_features=None, cached_B_warp_features=None):
        """
        Complete forward pass (legacy interface for 2-frame processing)

        Args:
            frame_t: [B, 3, H, W] LAB tensor (normalized)
            frame_t1: [B, 3, H, W] LAB tensor (normalized)
            reference_pil: PIL Image (RGB)
            target_pil: PIL Image (RGB)
            cached_ref_features: Optional tuple (ref_lab, features_B) for feature caching
            cached_B_warp_features: Optional tuple (phi, B_lab) pre-computed by WarpNet.precompute_B_features

        Returns:
            fused_lab: [B, 3, H, W] - Complete LAB prediction
        """
        return self.forward_single_frame(
            frame_t, frame_t1, reference_pil, target_pil,
            is_first=(self.curr_ti == -1),
            cached_ref_features=cached_ref_features,
            cached_B_warp_features=cached_B_warp_features,
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
