"""
Fusion Loss Functions

Combines multiple loss components for training the Fusion UNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using Swin Transformer feat_3 (SwinTExCo paper implementation)

    Replaces VGG with the shared embed_net (SwinV2) already present in the system.
    Only the deepest feature layer (feat_3, 512-ch) is used, matching the paper.
    Optionally applies InstanceNorm2d(512) for domain-invariant mode.

    Reference: SwinTExCo paper - losses.py Perceptual_loss()
    """
    def __init__(self, domain_invariant=False):
        super().__init__()
        self.domain_invariant = domain_invariant
        if domain_invariant:
            # Normalize per-instance to remove domain-specific statistics
            self.instancenorm = nn.InstanceNorm2d(512, affine=False)

    def forward(self, pred_lab, gt_lab, embed_net):
        """
        Args:
            pred_lab:  [B, 3, H, W] predicted LAB (normalized to [-1, 1])
            gt_lab:    [B, 3, H, W] ground truth LAB (normalized to [-1, 1])
            embed_net: Swin Transformer model for feature extraction (frozen)

        Returns:
            loss: scalar
        """
        from src.utils import uncenter_l, uncenter_ab, tensor_lab2rgb

        # Convert LAB → RGB for Swin feature extraction
        pred_l  = uncenter_l(pred_lab[:, 0:1, :, :])
        pred_ab = uncenter_ab(pred_lab[:, 1:3, :, :])
        gt_l    = uncenter_l(gt_lab[:, 0:1, :, :])
        gt_ab_u = uncenter_ab(gt_lab[:, 1:3, :, :])

        with torch.cuda.amp.autocast(enabled=False):
            pred_rgb = tensor_lab2rgb(torch.cat([pred_l, pred_ab], dim=1).float())
            gt_rgb   = tensor_lab2rgb(torch.cat([gt_l, gt_ab_u], dim=1).float())

            # Extract feat_3 inside autocast(enabled=False) so Swin runs in float32.
            # Under outer AMP the backward through ~100 Swin layers uses float16,
            # causing gradient underflow that zeroes the signal reaching FusionNet.
            _, _, _, pred_feat_3 = embed_net(pred_rgb)

            with torch.no_grad():
                _, _, _, gt_feat_3 = embed_net(gt_rgb)

        # MSE on feat_3, with optional InstanceNorm (domain_invariant mode)
        if self.domain_invariant:
            loss = F.mse_loss(
                self.instancenorm(pred_feat_3),
                self.instancenorm(gt_feat_3.detach())
            ) * 1e5 * 0.2
        else:
            loss = F.mse_loss(pred_feat_3, gt_feat_3.detach())

        return loss


class ContextualLoss(nn.Module):
    """
    Contextual Loss (Standard Implementation from SwinTExCo)

    Measures feature distribution similarity using normalized cosine distance
    and exponential affinity kernel.

    Reference: https://arxiv.org/abs/1803.02077
    """
    def __init__(self, layers=['relu3_3'], h=0.1, device='cuda'):
        super().__init__()

        self.h = h  # bandwidth parameter (default: 0.1, same as SwinTExCo)

    def _feature_normalize(self, feature_in):
        """
        L2 normalization (same as SwinTExCo's feature_normalize)

        Args:
            feature_in: [B, C, H, W] or [B, C, N]

        Returns:
            Normalized features
        """
        feature_in_norm = torch.norm(feature_in, 2, 1, keepdim=True) + 1e-10
        feature_in_norm = torch.div(feature_in, feature_in_norm)
        return feature_in_norm

    def forward(self, pred, target, feature_centering=True):
        """
        Standard Contextual Loss (NUMERICALLY STABLE VERSION)

        Args:
            pred: [B, 2, H, W] predicted AB channels
            target: [B, 2, H, W] ground truth AB channels
            feature_centering: Whether to subtract mean (default: True)

        Returns:
            loss: scalar (averaged over batch)
        """
        # 🔥 CRITICAL: Disable autocast to prevent FP16 overflow in contextual loss
        with torch.cuda.amp.autocast(enabled=False):
            # Ensure FP32 computation
            pred = pred.float()
            target = target.float()

            batch_size = pred.shape[0]
            feature_depth = pred.shape[1]

            # Convert to feature vectors
            X_features = pred  # [B, 2, H, W]
            Y_features = target  # [B, 2, H, W]

            # Feature centering (subtract mean from target features)
            if feature_centering:
                Y_mean = Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
                X_features = X_features - Y_mean
                Y_features = Y_features - Y_mean

            # Normalize features (L2 normalization)
            X_features = self._feature_normalize(X_features).view(batch_size, feature_depth, -1)  # [B, 2, H*W]
            Y_features = self._feature_normalize(Y_features).view(batch_size, feature_depth, -1)  # [B, 2, H*W]

            # Cosine distance = 1 - similarity
            X_features_permute = X_features.permute(0, 2, 1)  # [B, H*W, 2]
            d = 1 - torch.matmul(X_features_permute, Y_features)  # [B, H*W, H*W]

            # Clamp distance to prevent extreme values
            d = torch.clamp(d, min=0.0, max=2.0)  # Cosine distance is in [0, 2]

            # Normalized distance (with larger epsilon for stability)
            d_min = torch.min(d, dim=-1, keepdim=True)[0]
            d_norm = d / (d_min + 1e-3)  # Increased epsilon from 1e-5 to 1e-3

            # Clamp d_norm to prevent extreme exp() inputs
            d_norm = torch.clamp(d_norm, min=0.0, max=1.0 + 50.0 * self.h)  # For h=0.1, max d_norm=6.0

            # Pairwise affinity (numerically stable)
            exp_input = (1 - d_norm) / self.h
            exp_input = torch.clamp(exp_input, min=-20.0, max=20.0)
            w = torch.exp(exp_input)

            # Normalize to get affinity matrix (add epsilon to prevent division by zero)
            A_ij = w / (torch.sum(w, dim=-1, keepdim=True) + 1e-8)

            # Contextual similarity per sample
            # For each position in pred, find best match in target
            CX = torch.mean(torch.max(A_ij, dim=1)[0], dim=-1)  # [B]

            # Clamp CX to prevent log(0)
            CX = torch.clamp(CX, min=1e-6, max=1.0)

            # Contextual loss (negative log)
            loss = -torch.log(CX)  # [B]

            # Average over batch
            return loss.mean()


class SwinContextualLoss(nn.Module):
    """
    Swin-based Contextual Loss (SwinTExCo paper implementation)

    Computes Contextual Loss in Swin Transformer feature space instead of pixel space.
    Uses multi-scale features from 4 Swin layers with weighted aggregation.

    Reference: SwinTExCo paper - https://github.com/Josh0318coder/SwinTExCo.git
    """
    def __init__(self, h=0.1, device='cuda'):
        super().__init__()

        self.h = h  # bandwidth parameter (default: 0.1, same as SwinTExCo)
        self.device = device

        # Base contextual loss function (works on features)
        self.contextual_loss = ContextualLoss(h=h, device=device)

    def _feature_normalize(self, feature_in):
        """L2 normalization (same as SwinTExCo)"""
        feature_in_norm = torch.norm(feature_in, 2, 1, keepdim=True) + 1e-10
        feature_in_norm = torch.div(feature_in, feature_in_norm)
        return feature_in_norm

    def _compute_contextual_on_features(self, pred_feat, target_feat, feature_centering=True):
        """
        Compute contextual loss on Swin features (NUMERICALLY STABLE VERSION)

        Args:
            pred_feat: [B, C, H, W] Swin feature map
            target_feat: [B, C, H, W] Swin feature map
            feature_centering: Whether to subtract mean

        Returns:
            loss: scalar
        """
        # 🔥 CRITICAL: Disable autocast to prevent FP16 overflow
        with torch.cuda.amp.autocast(enabled=False):
            # Ensure FP32 computation
            pred_feat = pred_feat.float()
            target_feat = target_feat.float()

            batch_size = pred_feat.shape[0]
            feature_depth = pred_feat.shape[1]

            X_features = pred_feat
            Y_features = target_feat

            # Feature centering
            if feature_centering:
                Y_mean = Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
                X_features = X_features - Y_mean
                Y_features = Y_features - Y_mean

            # Normalize features (L2 normalization)
            X_features = self._feature_normalize(X_features).view(batch_size, feature_depth, -1)  # [B, C, H*W]
            Y_features = self._feature_normalize(Y_features).view(batch_size, feature_depth, -1)  # [B, C, H*W]

            # Cosine distance = 1 - similarity
            X_features_permute = X_features.permute(0, 2, 1)  # [B, H*W, C]
            d = 1 - torch.matmul(X_features_permute, Y_features)  # [B, H*W, H*W]

            # Clamp distance to prevent extreme values
            d = torch.clamp(d, min=0.0, max=2.0)  # Cosine distance is in [0, 2]

            # Normalized distance (with larger epsilon for stability)
            d_min = torch.min(d, dim=-1, keepdim=True)[0]
            d_norm = d / (d_min + 1e-3)  # Increased epsilon from 1e-5 to 1e-3

            # Clamp d_norm to prevent extreme exp() inputs
            # exp(x) overflows when x > ~88, so we limit (1 - d_norm) / h
            d_norm = torch.clamp(d_norm, min=0.0, max=1.0 + 50.0 * self.h)  # For h=0.1, max d_norm=6.0

            # Pairwise affinity (numerically stable)
            # Clamp the exp input to prevent overflow (max ~20 for safety)
            exp_input = (1 - d_norm) / self.h
            exp_input = torch.clamp(exp_input, min=-20.0, max=20.0)
            w = torch.exp(exp_input)

            # Normalize to get affinity matrix (add epsilon to prevent division by zero)
            A_ij = w / (torch.sum(w, dim=-1, keepdim=True) + 1e-8)

            # Contextual similarity
            CX = torch.mean(torch.max(A_ij, dim=-1)[0], dim=1)  # [B] - forward matching

            # Clamp CX to prevent log(0)
            CX = torch.clamp(CX, min=1e-6, max=1.0)

            # Contextual loss (negative log-likelihood)
            loss = -torch.log(CX)  # [B]

            return loss.mean()

    def forward(self, pred_lab, reference_lab, embed_net):
        """
        Compute Swin Contextual Loss (style matching following SwinTExCo paper)

        Matches predicted frame to reference image in Swin feature space.
        This is STYLE MATCHING, not reconstruction.

        Args:
            pred_lab: [B, 3, H, W] predicted LAB (normalized to [-1, 1])
            reference_lab: [B, 3, H, W] reference image LAB (normalized to [-1, 1])
            embed_net: Swin Transformer model for feature extraction (frozen)

        Returns:
            loss: scalar (weighted sum of 4-layer contextual losses)
        """
        # Convert LAB to RGB for Swin feature extraction
        from src.utils import uncenter_l, uncenter_ab, tensor_lab2rgb

        # Uncenter L channel (from [-1, 1] to [0, 100]) and AB channels (from [-1, 1] to [-127, 127])
        pred_l = uncenter_l(pred_lab[:, 0:1, :, :])
        pred_ab = uncenter_ab(pred_lab[:, 1:3, :, :])

        ref_l = uncenter_l(reference_lab[:, 0:1, :, :])
        ref_ab = uncenter_ab(reference_lab[:, 1:3, :, :])

        # Disable autocast for tensor_lab2rgb to prevent FP16/FP32 dtype mismatch
        # SwinTExCo paper doesn't use AMP, so tensor_lab2rgb needs FP32
        with torch.cuda.amp.autocast(enabled=False):
            pred_rgb = tensor_lab2rgb(torch.cat([pred_l, pred_ab], dim=1).float())
            ref_rgb = tensor_lab2rgb(torch.cat([ref_l, ref_ab], dim=1).float())

        # Extract Swin features
        # Note: embed_net is frozen (requires_grad=False, eval mode), but we DON'T use no_grad()
        # to allow gradients to flow back to pred_rgb (following SwinTExCo paper)
        pred_features = embed_net(pred_rgb)  # [feat_0, feat_1, feat_2, feat_3]

        # Reference features can use no_grad since we don't need gradients for reference
        with torch.no_grad():
            ref_features = embed_net(ref_rgb)      # [feat_0, feat_1, feat_2, feat_3]

        # Multi-scale contextual loss (following SwinTExCo paper)
        # Weights: 1x, 2x, 4x, 8x for layers 0, 1, 2, 3
        # Compare pred vs reference (style matching)
        loss_feat_0 = self._compute_contextual_on_features(pred_features[0], ref_features[0]) * 1
        loss_feat_1 = self._compute_contextual_on_features(pred_features[1], ref_features[1]) * 2
        loss_feat_2 = self._compute_contextual_on_features(pred_features[2], ref_features[2]) * 4
        loss_feat_3 = self._compute_contextual_on_features(pred_features[3], ref_features[3]) * 8

        # Total contextual loss (sum of weighted losses)
        total_loss = loss_feat_0 + loss_feat_1 + loss_feat_2 + loss_feat_3

        return total_loss


class TemporalLoss(nn.Module):
    """
    Temporal Consistency Loss

    Ensures smooth color transitions across frames using optical flow.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_t, pred_t1, flow, mask=None):
        """
        Args:
            pred_t: [B, 2, H, W] prediction at frame t
            pred_t1: [B, 2, H, W] prediction at frame t+1
            flow: [B, 2, H, W] optical flow from t to t+1
            mask: [B, 1, H, W] valid region mask (optional)

        Returns:
            loss: scalar
        """
        from core.loss_new import warp_color_by_flow

        # Warp t+1 to t using flow
        warped_t1 = warp_color_by_flow(pred_t1, flow)

        # Compute difference
        diff = torch.abs(pred_t - warped_t1)

        # Apply mask if provided
        if mask is not None:
            diff = diff * mask

        loss = diff.mean()
        return loss


class AdaptiveTemporalLoss(nn.Module):
    """
    Adaptive Temporal Consistency Loss

    Align Loss: Force high-confidence regions to follow MemFlow's flow-warped prediction.
    Computed at MemFlow native resolution (56×56) to avoid conf upsampling block artifacts.
    Spatial smoothness is handled separately by SpatialSmoothnessLoss.
    """
    def __init__(self, lambda_align=1.0):
        super().__init__()
        self.lambda_align = lambda_align

    def forward(self, fusion_t, fusion_t1, memflow_t, memflow_t1,
                memflow_conf_t, memflow_conf_t1):
        # Align loss at MemFlow native resolution (56×56)
        native_size = memflow_conf_t.shape[2:]
        fusion_t_small  = F.interpolate(fusion_t,  size=native_size, mode='bilinear', align_corners=True)
        fusion_t1_small = F.interpolate(fusion_t1, size=native_size, mode='bilinear', align_corners=True)

        diff_t  = torch.abs(fusion_t_small  - memflow_t)
        diff_t1 = torch.abs(fusion_t1_small - memflow_t1)

        align_loss_t  = (diff_t  * memflow_conf_t).mean()
        align_loss_t1 = (diff_t1 * memflow_conf_t1).mean()
        align_loss = (align_loss_t + align_loss_t1) / 2

        weighted_align = self.lambda_align * align_loss
        return weighted_align, weighted_align


class SpatialSmoothnessLoss(nn.Module):
    """
    Spatial Smoothness Loss (from SwinTExCo/newsingle)

    Pixels with similar GT colors in a 3×3 neighborhood should have similar
    predicted colors. Prevents isolated color blobs and noise within a single frame.

    Reference: newsingle/train_fixseed.py smoothness_loss_fn + WeightedAverage_color
    """
    def __init__(self, patch_size=3, alpha=10):
        super().__init__()
        self.patch_size = patch_size
        self.alpha = alpha
        self._weighted_layer = None

    def _get_weighted_layer(self, device):
        if self._weighted_layer is None:
            from src.models.CNN.NonlocalNet import WeightedAverage_color
            self._weighted_layer = WeightedAverage_color().to(device)
        return self._weighted_layer

    def forward(self, pred_ab, gt_lab, pred_lab):
        """
        Args:
            pred_ab:  [B, 2, H, W] predicted AB channels
            gt_lab:   [B, 3, H, W] ground truth LAB (normalized [-1, 1])
            pred_lab: [B, 3, H, W] predicted LAB (normalized [-1, 1])
        Returns:
            loss: scalar
        """
        weighted_layer = self._get_weighted_layer(pred_ab.device)
        with torch.cuda.amp.autocast(enabled=False):
            ab_weighted = weighted_layer(
                gt_lab.float(),
                pred_lab.float(),
                patch_size=self.patch_size,
                alpha=self.alpha,
                scale_factor=1,
            )
        return F.mse_loss(pred_ab.float(), ab_weighted)


class ColorConsistencyLoss(nn.Module):
    """
    Color Consistency Loss (Wasserstein-1 on AB channel distributions)

    Wasserstein-1 on 1D distributions = sort and take L1 difference.
    AB space gives clean direct gradients since the model outputs AB channels.
    """
    def forward(self, pred_lab_t, pred_lab_t1):
        """
        Args:
            pred_lab_t:  [B, 3, H, W] full LAB of previous frame (normalized [-1, 1])
            pred_lab_t1: [B, 3, H, W] full LAB of current frame (normalized [-1, 1])

        Returns:
            loss: scalar
        """
        # Extract AB channels only — model directly controls these
        ab_t  = pred_lab_t[:, 1:3, :, :]   # [B, 2, H, W]
        ab_t1 = pred_lab_t1[:, 1:3, :, :]  # [B, 2, H, W]

        B, C, H, W = ab_t.shape
        x = ab_t.reshape(B, C, -1)
        y = ab_t1.reshape(B, C, -1)
        x_sorted = x.sort(dim=-1)[0]
        y_sorted = y.sort(dim=-1)[0]
        return F.l1_loss(x_sorted, y_sorted)


class FusionLoss(nn.Module):
    """
    Complete Fusion Loss

    Combines:
        - L1 Loss (pixel-wise accuracy)
        - Perceptual Loss (feature similarity)
        - Contextual Loss (distribution similarity)
        - Temporal Loss (temporal consistency, optional)
        - GAN Loss (optional)

    Weights:
        - L1: 1.0 (baseline)
        - Perceptual: 0.05
        - Contextual (Swin): 0.015 (SwinTExCo paper, only on frame 0)
        - Temporal: 0.5 (if used)
    """
    def __init__(self,
                 lambda_l1=1.0,
                 lambda_l1_gt=0.1,
                 lambda_perceptual=0.05,
                 lambda_contextual=0.015,  # SwinTExCo paper uses 0.015, not 0.5!
                 lambda_temporal=0.5,
                 lambda_align=1.0,   # Weight for align component in adaptive temporal loss
                 lambda_smooth=0.3,  # Weight for spatial smoothness loss (replaces temporal smooth)
                 lambda_cdc=0.0,     # Weight for color consistency loss (Wasserstein-1 on AB distributions)
                 use_temporal=True,
                 use_swin_contextual=True,  # Use Swin-based contextual loss
                 use_adaptive_temporal=True,  # Use adaptive temporal loss (no optical flow)
                 contextual_chunk_size=256,
                 domain_invariant=False,  # InstanceNorm2d for perceptual loss (SwinTExCo paper flag)
                 device='cuda'):
        super().__init__()

        self.lambda_l1 = lambda_l1
        self.lambda_l1_gt = lambda_l1_gt
        self.lambda_perceptual = lambda_perceptual
        self.lambda_contextual = lambda_contextual
        self.lambda_temporal = lambda_temporal
        self.lambda_smooth = lambda_smooth
        self.lambda_cdc = lambda_cdc
        self.contextual_chunk_size = contextual_chunk_size
        self.use_temporal = use_temporal
        self.use_swin_contextual = use_swin_contextual
        self.use_adaptive_temporal = use_adaptive_temporal

        # Loss components
        self.l1_loss = nn.L1Loss()
        # Swin-based perceptual loss (feat_3, following SwinTExCo paper)
        self.perceptual_loss = PerceptualLoss(domain_invariant=domain_invariant)

        # Contextual loss: always initialize both for fallback support
        self.swin_contextual_loss = SwinContextualLoss(device=device)
        self.contextual_loss = ContextualLoss(device=device)  # Fallback for AB-based

        # Spatial smoothness loss (replaces blunt temporal smooth)
        self.spatial_smooth_loss = SpatialSmoothnessLoss(patch_size=3, alpha=10)

        # Color consistency loss (Wasserstein-1 on AB distributions, targets CDC metric)
        self.color_consistency_loss = ColorConsistencyLoss()

        if use_temporal:
            if use_adaptive_temporal:
                # Use new adaptive temporal loss (align only, no smooth)
                self.temporal_loss = AdaptiveTemporalLoss(lambda_align=lambda_align)
            else:
                # Use old optical flow-based temporal loss
                self.temporal_loss = TemporalLoss()

    def forward(self, pred_ab, gt_ab, flow=None, mask=None, prev_pred_ab=None,
                frame_idx=None, pred_lab=None, gt_lab=None, reference_lab=None, embed_net=None,
                memflow_ab=None, memflow_conf=None, prev_memflow_ab=None, prev_memflow_conf=None,
                swintexco_ab=None, swintexco_conf=None, prev_pred_lab_full=None):
        """
        Compute total loss

        Args:
            pred_ab: [B, 2, H, W] predicted AB channels
            gt_ab: [B, 2, H, W] ground truth AB channels
            flow: [B, 2, H, W] optical flow (for old temporal loss)
            mask: [B, 1, H, W] valid mask (for old temporal loss)
            prev_pred_ab: [B, 2, H, W] previous frame prediction (for old temporal loss)
            frame_idx: int, frame index in sequence (for Swin contextual loss)
            pred_lab: [B, 3, H, W] predicted LAB (for Swin contextual loss and perceptual loss)
            gt_lab: [B, 3, H, W] ground truth LAB (for perceptual loss)
            reference_lab: [B, 3, H, W] reference image LAB (for Swin contextual loss)
            embed_net: Swin model for feature extraction (for Swin contextual loss)
            memflow_ab: [B, 2, H, W] MemFlow AB output (for adaptive temporal loss)
            memflow_conf: [B, 1, H, W] MemFlow confidence (for adaptive temporal loss)
            prev_memflow_ab: [B, 2, H, W] previous MemFlow AB (for adaptive temporal loss)
            prev_memflow_conf: [B, 1, H, W] previous MemFlow confidence (for adaptive temporal loss)
            swintexco_ab: [B, 2, H, W] SwinTExCo AB output (for weighted L1 loss)
            swintexco_conf: [B, 1, H, W] SwinTExCo confidence (for weighted L1 loss)

        Returns:
            total_loss: scalar
            loss_dict: dictionary of individual losses
        """
        # Ref L1 Loss (confidence-weighted vs SwinTExCo output)
        if self.lambda_l1 > 0 and swintexco_ab is not None and swintexco_conf is not None:
            l1_pixel = torch.abs(pred_ab - swintexco_ab)  # [B, 2, H, W]
            loss_l1_ref = (swintexco_conf * l1_pixel).mean()
        else:
            loss_l1_ref = torch.tensor(0.0, device=pred_ab.device)

        # GT L1 Loss (plain L1 vs ground truth, always computed)
        loss_l1_gt = self.l1_loss(pred_ab, gt_ab)

        # Perceptual Loss (Swin feat_3, only on frame 0)
        if self.lambda_perceptual > 0 and pred_lab is not None and gt_lab is not None and embed_net is not None and frame_idx == 0:
            loss_perceptual = self.perceptual_loss(pred_lab, gt_lab, embed_net)
        else:
            loss_perceptual = torch.tensor(0.0, device=pred_ab.device)

        # Total loss
        total_loss = (
            self.lambda_l1 * loss_l1_ref +
            self.lambda_l1_gt * loss_l1_gt +
            self.lambda_perceptual * loss_perceptual
        )

        loss_dict = {
            'l1_ref': (self.lambda_l1 * loss_l1_ref).item(),
            'l1_gt': (self.lambda_l1_gt * loss_l1_gt).item(),
            'perceptual': (self.lambda_perceptual * loss_perceptual).item(),
        }

        # Contextual Loss (only compute on frame 0 to save memory)
        # Compares predicted frame to reference image (style matching)
        if self.lambda_contextual > 0 and frame_idx is not None:
            if frame_idx == 0:  # Only compute on first frame
                if self.use_swin_contextual and pred_lab is not None and reference_lab is not None and embed_net is not None:
                    # Swin-based contextual loss (multi-scale, following SwinTExCo paper)
                    # Style matching: pred vs reference
                    loss_contextual = self.swin_contextual_loss(pred_lab, reference_lab, embed_net)
                    total_loss += self.lambda_contextual * loss_contextual
                    loss_dict['contextual'] = (self.lambda_contextual * loss_contextual).item()
                else:
                    # Fallback to AB-based contextual loss
                    loss_contextual = self.contextual_loss(pred_ab, gt_ab)
                    total_loss += self.lambda_contextual * loss_contextual
                    loss_dict['contextual'] = (self.lambda_contextual * loss_contextual).item()
            else:
                # Skip contextual loss for frame 1, 2, 3
                loss_dict['contextual'] = 0.0
        else:
            loss_dict['contextual'] = 0.0

        # Spatial Smoothness Loss (per-frame, replaces blunt temporal smooth)
        # GT-color-guided: pixels similar in GT should have similar predicted colors
        if pred_lab is not None and gt_lab is not None and self.lambda_smooth > 0:
            loss_spatial_smooth = self.spatial_smooth_loss(pred_ab, gt_lab, pred_lab)
            total_loss += self.lambda_smooth * loss_spatial_smooth
            loss_dict['smooth'] = (self.lambda_smooth * loss_spatial_smooth).item()
        else:
            loss_dict['smooth'] = 0.0

        # Color Consistency Loss (Wasserstein-1 on RGB distributions, targets CDC metric)
        if self.lambda_cdc > 0 and prev_pred_lab_full is not None and pred_lab is not None:
            loss_cdc = self.color_consistency_loss(prev_pred_lab_full, pred_lab)
            total_loss += self.lambda_cdc * loss_cdc
            loss_dict['cdc'] = (self.lambda_cdc * loss_cdc).item()
        else:
            loss_dict['cdc'] = 0.0

        # Temporal Align Loss (optional, requires MemFlow outputs)
        if self.use_temporal and self.lambda_temporal > 0:
            if self.use_adaptive_temporal:
                if (memflow_ab is not None and memflow_conf is not None and
                    prev_memflow_ab is not None and prev_memflow_conf is not None and
                    prev_pred_ab is not None):
                    loss_temporal, align_loss = self.temporal_loss(
                        prev_pred_ab, pred_ab,
                        prev_memflow_ab, memflow_ab,
                        prev_memflow_conf, memflow_conf
                    )
                    total_loss += self.lambda_temporal * loss_temporal
                    loss_dict['temporal'] = (self.lambda_temporal * loss_temporal).item()
                    loss_dict['align'] = align_loss.item()
                else:
                    loss_dict['temporal'] = 0.0
                    loss_dict['align'] = 0.0
            else:
                if flow is not None and prev_pred_ab is not None:
                    loss_temporal = self.temporal_loss(prev_pred_ab, pred_ab, flow, mask)
                    total_loss += self.lambda_temporal * loss_temporal
                    loss_dict['temporal'] = (self.lambda_temporal * loss_temporal).item()
                    loss_dict['align'] = 0.0
                else:
                    loss_dict['temporal'] = 0.0
                    loss_dict['align'] = 0.0
        else:
            loss_dict['temporal'] = 0.0
            loss_dict['align'] = 0.0

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


# ===========================
# GAN Loss Functions (from SwinSingle)
# ===========================

def generator_loss_fn(real_data_lab, fake_data_lab, discriminator, weight_gan, device):
    """
    Relativistic GAN Generator Loss (from SwinSingle)

    Args:
        real_data_lab: [B, 3, H, W] Ground truth LAB image
        fake_data_lab: [B, 3, H, W] Generated LAB image
        discriminator: Discriminator network
        weight_gan: Weight for GAN loss
        device: torch device

    Returns:
        generator_loss: scalar
    """
    if weight_gan > 0:
        # CRITICAL: Order matters for SpectralNorm!
        # First do no_grad forward (real), then do forward that needs gradients (fake)
        # This ensures the weight version is correct when backward() is called

        # Use no_grad for real to prevent SpectralNorm from affecting gradient graph
        with torch.no_grad():
            y_pred_real, _ = discriminator(real_data_lab)
        y_pred_real = y_pred_real.detach()

        # For generator loss, we need gradients for fake (to train generator)
        # This must be LAST so backward() sees the correct weight version
        y_pred_fake, _ = discriminator(fake_data_lab)

        y = torch.ones_like(y_pred_real)
        generator_loss = (
            (
                torch.mean((y_pred_real - torch.mean(y_pred_fake) + y) ** 2)
                + torch.mean((y_pred_fake - torch.mean(y_pred_real) - y) ** 2)
            )
            / 2
            * weight_gan
        )
        return generator_loss

    return torch.tensor([0.0], device=device)


def discriminator_loss_fn(real_data_lab, fake_data_lab, discriminator):
    """
    Relativistic GAN Discriminator Loss (from SwinSingle)

    Args:
        real_data_lab: [B, 3, H, W] Ground truth LAB image
        fake_data_lab: [B, 3, H, W] Generated LAB image (detached)
        discriminator: Discriminator network

    Returns:
        discriminator_loss: scalar
    """
    y_pred_fake, _ = discriminator(fake_data_lab.detach())
    y_pred_real, _ = discriminator(real_data_lab.detach())

    y = torch.ones_like(y_pred_real)
    discriminator_loss = (
        torch.mean((y_pred_real - torch.mean(y_pred_fake) - y) ** 2)
        + torch.mean((y_pred_fake - torch.mean(y_pred_real) + y) ** 2)
    ) / 2
    return discriminator_loss


# ===========================
# Standard (log) GAN Loss — matches thesis Eq. (3.36) / (3.37)
# Kept alongside the RaLSGAN variant above so the two can be swapped for comparison.
# NOTE: Discriminator_x64_224 outputs raw logits (no sigmoid), so we use
# binary_cross_entropy_with_logits, which applies the sigmoid internally.
# Do NOT add an explicit sigmoid, or it would be applied twice.
# ===========================

def generator_loss_fn_loggan(real_data_lab, fake_data_lab, discriminator, weight_gan, device):
    """
    Standard (log) GAN Generator Loss — thesis Eq. (3.37), non-saturating form.

        L_G = -E[log D(fake)]        with D = sigmoid(discriminator logits)

    Non-saturating is the standard practical form (avoids the vanishing
    gradients of the original minimax +E[log(1 - D(fake))]).

    Args:
        real_data_lab: [B, C, H, W] real input (unused here; kept for a signature
                       matching generator_loss_fn so call sites can swap freely)
        fake_data_lab: [B, C, H, W] generated input
        discriminator: Discriminator network (returns raw logits)
        weight_gan: Weight for GAN loss
        device: torch device

    Returns:
        generator_loss: scalar
    """
    if weight_gan > 0:
        y_pred_fake, _ = discriminator(fake_data_lab)
        target_real = torch.ones_like(y_pred_fake)  # generator wants fake judged as real
        generator_loss = F.binary_cross_entropy_with_logits(y_pred_fake, target_real) * weight_gan
        return generator_loss

    return torch.tensor([0.0], device=device)


def discriminator_loss_fn_loggan(real_data_lab, fake_data_lab, discriminator):
    """
    Standard (log) GAN Discriminator Loss — thesis Eq. (3.36).

        L_D = -E[log D(real)] - E[log(1 - D(fake))]
              with D = sigmoid(discriminator logits)

    Args:
        real_data_lab: [B, C, H, W] real input
        fake_data_lab: [B, C, H, W] generated input (detached)
        discriminator: Discriminator network (returns raw logits)

    Returns:
        discriminator_loss: scalar
    """
    y_pred_fake, _ = discriminator(fake_data_lab.detach())
    y_pred_real, _ = discriminator(real_data_lab.detach())

    target_real = torch.ones_like(y_pred_real)   # real -> 1
    target_fake = torch.zeros_like(y_pred_fake)  # fake -> 0
    discriminator_loss = (
        F.binary_cross_entropy_with_logits(y_pred_real, target_real)
        + F.binary_cross_entropy_with_logits(y_pred_fake, target_fake)
    )
    return discriminator_loss
