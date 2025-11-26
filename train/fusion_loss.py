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
        Standard Contextual Loss (SwinTExCo implementation without chunking)

        Args:
            pred: [B, 2, H, W] predicted AB channels
            target: [B, 2, H, W] ground truth AB channels
            feature_centering: Whether to subtract mean (default: True)

        Returns:
            loss: scalar (averaged over batch)
        """
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

        # Normalized distance: d_ij / min(d_i)
        # This emphasizes relative matching quality
        d_norm = d / (torch.min(d, dim=-1, keepdim=True)[0] + 1e-5)  # [B, H*W, H*W]

        # Pairwise affinity using exponential kernel
        w = torch.exp((1 - d_norm) / self.h)  # [B, H*W, H*W]
        A_ij = w / torch.sum(w, dim=-1, keepdim=True)  # Softmax normalization

        # Contextual similarity per sample
        # For each position in pred, find best match in target
        CX = torch.mean(torch.max(A_ij, dim=1)[0], dim=-1)  # [B]

        # Contextual loss (negative log)
        loss = -torch.log(CX + 1e-5)  # [B]

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
        Compute contextual loss on Swin features

        Args:
            pred_feat: [B, C, H, W] Swin feature map
            target_feat: [B, C, H, W] Swin feature map
            feature_centering: Whether to subtract mean

        Returns:
            loss: scalar
        """
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

        # Normalized distance
        d_norm = d / (torch.min(d, dim=-1, keepdim=True)[0] + 1e-5)

        # Pairwise affinity
        w = torch.exp((1 - d_norm) / self.h)
        A_ij = w / torch.sum(w, dim=-1, keepdim=True)

        # Contextual similarity
        CX = torch.mean(torch.max(A_ij, dim=-1)[0], dim=1)  # [B] - forward matching

        # Contextual loss
        loss = -torch.log(CX + 1e-5)  # [B]

        return loss.mean()

    def forward(self, pred_lab, gt_lab, embed_net):
        """
        Compute Swin Contextual Loss (multi-scale version following SwinTExCo paper)

        Args:
            pred_lab: [B, 3, H, W] predicted LAB (normalized to [-1, 1])
            gt_lab: [B, 3, H, W] ground truth LAB (normalized to [-1, 1])
            embed_net: Swin Transformer model for feature extraction (frozen)

        Returns:
            loss: scalar (weighted sum of 4-layer contextual losses)
        """
        # Convert LAB to RGB for Swin feature extraction
        from src.utils import uncenter_l, tensor_lab2rgb

        # Force float32 for tensor_lab2rgb (not compatible with AMP FP16)
        pred_lab_fp32 = pred_lab.float()
        gt_lab_fp32 = gt_lab.float()

        # Uncenter L channel (from [-1, 1] to [0, 1])
        pred_l = uncenter_l(pred_lab_fp32[:, 0:1, :, :])
        pred_ab = pred_lab_fp32[:, 1:3, :, :]
        pred_rgb = tensor_lab2rgb(torch.cat([pred_l, pred_ab], dim=1))

        gt_l = uncenter_l(gt_lab_fp32[:, 0:1, :, :])
        gt_ab = gt_lab_fp32[:, 1:3, :, :]
        gt_rgb = tensor_lab2rgb(torch.cat([gt_l, gt_ab], dim=1))

        # Extract Swin features (embed_net is frozen, so use no_grad)
        with torch.no_grad():
            pred_features = embed_net(pred_rgb)  # [feat_0, feat_1, feat_2, feat_3]
            gt_features = embed_net(gt_rgb)      # [feat_0, feat_1, feat_2, feat_3]

        # Multi-scale contextual loss (following SwinTExCo paper)
        # Weights: 1x, 2x, 4x, 8x for layers 0, 1, 2, 3
        loss_feat_0 = self._compute_contextual_on_features(pred_features[0], gt_features[0]) * 1
        loss_feat_1 = self._compute_contextual_on_features(pred_features[1], gt_features[1]) * 2
        loss_feat_2 = self._compute_contextual_on_features(pred_features[2], gt_features[2]) * 4
        loss_feat_3 = self._compute_contextual_on_features(pred_features[3], gt_features[3]) * 8

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
        - Contextual (Swin): 0.5 (SwinTExCo paper, only on frame 0)
        - Temporal: 0.5 (if used)
    """
    def __init__(self,
                 lambda_l1=1.0,
                 lambda_perceptual=0.05,
                 lambda_contextual=0.5,  # Changed to 0.5 following SwinTExCo paper
                 lambda_temporal=0.5,
                 use_temporal=True,
                 use_swin_contextual=True,  # Use Swin-based contextual loss
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

        # Loss components
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss(device=device)

        # Contextual loss: use Swin-based or AB-based
        if use_swin_contextual:
            self.swin_contextual_loss = SwinContextualLoss(device=device)
        else:
            self.contextual_loss = ContextualLoss(device=device)

        if use_temporal:
            self.temporal_loss = TemporalLoss()

    def forward(self, pred_ab, gt_ab, flow=None, mask=None, prev_pred_ab=None,
                frame_idx=None, pred_lab=None, gt_lab=None, embed_net=None):
        """
        Compute total loss

        Args:
            pred_ab: [B, 2, H, W] predicted AB channels
            gt_ab: [B, 2, H, W] ground truth AB channels
            flow: [B, 2, H, W] optical flow (for temporal loss)
            mask: [B, 1, H, W] valid mask (for temporal loss)
            prev_pred_ab: [B, 2, H, W] previous frame prediction (for temporal loss)
            frame_idx: int, frame index in sequence (for Swin contextual loss)
            pred_lab: [B, 3, H, W] predicted LAB (for Swin contextual loss)
            gt_lab: [B, 3, H, W] ground truth LAB (for Swin contextual loss)
            embed_net: Swin model for feature extraction (for Swin contextual loss)

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
            'l1': loss_l1.item(),
            'perceptual': loss_perceptual.item(),
        }

        # Contextual Loss (only compute on frame 0 to save memory)
        if self.lambda_contextual > 0 and frame_idx is not None:
            if frame_idx == 0:  # Only compute on first frame
                if self.use_swin_contextual and pred_lab is not None and gt_lab is not None and embed_net is not None:
                    # Swin-based contextual loss (multi-scale, following SwinTExCo paper)
                    loss_contextual = self.swin_contextual_loss(pred_lab, gt_lab, embed_net)
                    total_loss += self.lambda_contextual * loss_contextual
                    loss_dict['contextual'] = loss_contextual.item()
                else:
                    # Fallback to AB-based contextual loss
                    loss_contextual = self.contextual_loss(pred_ab, gt_ab)
                    total_loss += self.lambda_contextual * loss_contextual
                    loss_dict['contextual'] = loss_contextual.item()
            else:
                # Skip contextual loss for frame 1, 2, 3
                loss_dict['contextual'] = 0.0
        else:
            loss_dict['contextual'] = 0.0

        # Temporal Loss (optional)
        if self.use_temporal and flow is not None and prev_pred_ab is not None:
            loss_temporal = self.temporal_loss(prev_pred_ab, pred_ab, flow, mask)
            total_loss += self.lambda_temporal * loss_temporal
            loss_dict['temporal'] = loss_temporal.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict
