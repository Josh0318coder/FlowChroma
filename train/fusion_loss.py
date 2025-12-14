"""
Fusion Loss Functions

Combines multiple loss components for training the Fusion UNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG16

    Computes feature similarity in VGG feature space.
    """
    def __init__(self, layers=['relu3_3'], device='cuda'):
        super().__init__()

        vgg = models.vgg16(pretrained=True).features
        self.vgg_layers = nn.ModuleDict()

        layer_mapping = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22
        }

        for name in layers:
            if name in layer_mapping:
                self.vgg_layers[name] = nn.Sequential(*list(vgg.children())[:layer_mapping[name]+1])

        # Freeze VGG
        for param in self.parameters():
            param.requires_grad = False

        self.to(device)
        self.eval()

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 2, H, W] AB channels
            target: [B, 2, H, W] AB channels

        Returns:
            loss: scalar
        """
        # Convert AB to RGB (approximate for VGG)
        pred_rgb = self._ab_to_rgb_approx(pred)
        target_rgb = self._ab_to_rgb_approx(target)

        loss = 0.0
        for layer in self.vgg_layers.values():
            pred_feat = layer(pred_rgb)
            target_feat = layer(target_rgb)
            loss += F.mse_loss(pred_feat, target_feat)

        return loss / len(self.vgg_layers)

    def _ab_to_rgb_approx(self, ab):
        """Approximate AB to RGB conversion for VGG (from local-2)"""
        # Simple approximation: use L=0 and normalize to [0, 1]
        # This is not true LAB->RGB conversion, but works for VGG perceptual loss
        b, _, h, w = ab.shape
        L = torch.zeros(b, 1, h, w, device=ab.device, dtype=ab.dtype)
        lab = torch.cat([L, ab], dim=1)

        # Normalize to [0, 1] range for VGG
        rgb = (lab + 1.0) / 2.0
        return rgb


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
        self.perceptual = PerceptualLoss(layers, device)

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
        from src.utils import uncenter_l, tensor_lab2rgb

        # Uncenter L channel (from [-1, 1] to [0, 1])
        pred_l = uncenter_l(pred_lab[:, 0:1, :, :])
        pred_ab = pred_lab[:, 1:3, :, :]

        ref_l = uncenter_l(reference_lab[:, 0:1, :, :])
        ref_ab = reference_lab[:, 1:3, :, :]

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
    Adaptive Temporal Consistency Loss (New version - no optical flow required)

    Combines two strategies:
    1. Align Loss: High-confidence regions should follow MemFlow (confidence-weighted)
    2. Smooth Loss: All regions should maintain temporal smoothness (global constraint)

    This design addresses the limitation that video colorization lacks pre-computed optical flow,
    while ensuring both spatial alignment with MemFlow and temporal coherence across frames.
    """
    def __init__(self, lambda_smooth=0.3):
        """
        Args:
            lambda_smooth: Weight for the global smoothness loss (default: 0.3)
        """
        super().__init__()
        self.lambda_smooth = lambda_smooth

    def forward(self, fusion_t, fusion_t1, memflow_t, memflow_t1,
                memflow_conf_t, memflow_conf_t1):
        """
        Compute adaptive temporal loss

        Args:
            fusion_t: [B, 2, H, W] Fusion output AB channels at frame t
            fusion_t1: [B, 2, H, W] Fusion output AB channels at frame t+1
            memflow_t: [B, 2, H, W] MemFlow output AB channels at frame t
            memflow_t1: [B, 2, H, W] MemFlow output AB channels at frame t+1
            memflow_conf_t: [B, 1, H, W] MemFlow confidence at frame t
            memflow_conf_t1: [B, 1, H, W] MemFlow confidence at frame t+1

        Returns:
            loss: scalar
        """
        # 1. Align Loss: Force high-confidence regions to follow MemFlow
        # This ensures temporal coherence by leveraging MemFlow's temporal consistency
        diff_t = torch.abs(fusion_t - memflow_t)
        diff_t1 = torch.abs(fusion_t1 - memflow_t1)

        align_loss_t = (diff_t * memflow_conf_t).mean()
        align_loss_t1 = (diff_t1 * memflow_conf_t1).mean()
        align_loss = (align_loss_t + align_loss_t1) / 2

        # 2. Smooth Loss: Encourage smooth transitions across all regions
        # This prevents flickering even in low-confidence areas where SwinTExCo dominates
        smooth_loss = torch.abs(fusion_t1 - fusion_t).mean()

        # Combine both losses
        total_loss = align_loss + self.lambda_smooth * smooth_loss

        return total_loss, align_loss, smooth_loss


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
                 lambda_perceptual=0.05,
                 lambda_contextual=0.015,  # SwinTExCo paper uses 0.015, not 0.5!
                 lambda_temporal=0.5,
                 lambda_smooth=0.3,  # Weight for smooth component in adaptive temporal loss
                 use_temporal=True,
                 use_swin_contextual=True,  # Use Swin-based contextual loss
                 use_adaptive_temporal=True,  # Use adaptive temporal loss (no optical flow)
                 contextual_chunk_size=256,
                 device='cuda'):
        super().__init__()

        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_contextual = lambda_contextual
        self.lambda_temporal = lambda_temporal
        self.contextual_chunk_size = contextual_chunk_size
        self.use_temporal = use_temporal
        self.use_swin_contextual = use_swin_contextual
        self.use_adaptive_temporal = use_adaptive_temporal

        # Loss components
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss(device=device)

        # Contextual loss: always initialize both for fallback support
        self.swin_contextual_loss = SwinContextualLoss(device=device)
        self.contextual_loss = ContextualLoss(device=device)  # Fallback for AB-based

        if use_temporal:
            if use_adaptive_temporal:
                # Use new adaptive temporal loss (no optical flow required)
                self.temporal_loss = AdaptiveTemporalLoss(lambda_smooth=lambda_smooth)
            else:
                # Use old optical flow-based temporal loss
                self.temporal_loss = TemporalLoss()

    def forward(self, pred_ab, gt_ab, flow=None, mask=None, prev_pred_ab=None,
                frame_idx=None, pred_lab=None, reference_lab=None, embed_net=None,
                memflow_ab=None, memflow_conf=None, prev_memflow_ab=None, prev_memflow_conf=None):
        """
        Compute total loss

        Args:
            pred_ab: [B, 2, H, W] predicted AB channels
            gt_ab: [B, 2, H, W] ground truth AB channels
            flow: [B, 2, H, W] optical flow (for old temporal loss)
            mask: [B, 1, H, W] valid mask (for old temporal loss)
            prev_pred_ab: [B, 2, H, W] previous frame prediction (for old temporal loss)
            frame_idx: int, frame index in sequence (for Swin contextual loss)
            pred_lab: [B, 3, H, W] predicted LAB (for Swin contextual loss)
            reference_lab: [B, 3, H, W] reference image LAB (for Swin contextual loss)
            embed_net: Swin model for feature extraction (for Swin contextual loss)
            memflow_ab: [B, 2, H, W] MemFlow AB output (for adaptive temporal loss)
            memflow_conf: [B, 1, H, W] MemFlow confidence (for adaptive temporal loss)
            prev_memflow_ab: [B, 2, H, W] previous MemFlow AB (for adaptive temporal loss)
            prev_memflow_conf: [B, 1, H, W] previous MemFlow confidence (for adaptive temporal loss)

        Returns:
            total_loss: scalar
            loss_dict: dictionary of individual losses
        """
        # L1 Loss
        loss_l1 = self.l1_loss(pred_ab, gt_ab)

        # Perceptual Loss
        loss_perceptual = self.perceptual_loss(pred_ab, gt_ab)

        # Total loss
        total_loss = (
            self.lambda_l1 * loss_l1 +
            self.lambda_perceptual * loss_perceptual
        )

        loss_dict = {
            'l1': (self.lambda_l1 * loss_l1).item(),
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

        # Temporal Loss (optional)
        if self.use_temporal:
            if self.use_adaptive_temporal:
                # Adaptive temporal loss (requires MemFlow outputs)
                if (memflow_ab is not None and memflow_conf is not None and
                    prev_memflow_ab is not None and prev_memflow_conf is not None and
                    prev_pred_ab is not None):
                    loss_temporal, align_loss, smooth_loss = self.temporal_loss(
                        prev_pred_ab, pred_ab,  # fusion outputs
                        prev_memflow_ab, memflow_ab,  # memflow outputs
                        prev_memflow_conf, memflow_conf  # confidences
                    )
                    total_loss += self.lambda_temporal * loss_temporal
                    loss_dict['temporal'] = (self.lambda_temporal * loss_temporal).item()
                    loss_dict['align'] = align_loss.item()
                    loss_dict['smooth'] = smooth_loss.item()
                else:
                    loss_dict['temporal'] = 0.0
                    loss_dict['align'] = 0.0
                    loss_dict['smooth'] = 0.0
            else:
                # Old optical flow-based temporal loss
                if flow is not None and prev_pred_ab is not None:
                    loss_temporal = self.temporal_loss(prev_pred_ab, pred_ab, flow, mask)
                    total_loss += self.lambda_temporal * loss_temporal
                    loss_dict['temporal'] = (self.lambda_temporal * loss_temporal).item()
                    loss_dict['align'] = 0.0
                    loss_dict['smooth'] = 0.0
                else:
                    loss_dict['temporal'] = 0.0
                    loss_dict['align'] = 0.0
                    loss_dict['smooth'] = 0.0
        else:
            loss_dict['temporal'] = 0.0
            loss_dict['align'] = 0.0
            loss_dict['smooth'] = 0.0

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict
