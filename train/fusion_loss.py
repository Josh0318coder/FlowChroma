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
        """Approximate AB to RGB conversion for VGG"""
        # Simple approximation: replicate AB to 3 channels
        # In practice, you might want to use proper LAB->RGB conversion
        b, _, h, w = ab.shape
        L = torch.zeros(b, 1, h, w, device=ab.device)
        lab = torch.cat([L, ab], dim=1)

        # Normalize to [0, 1] range for VGG
        rgb = (lab + 1.0) / 2.0
        return rgb.repeat(1, 1, 1, 1) if rgb.shape[1] == 1 else rgb[:, :3, :, :]


class ContextualLoss(nn.Module):
    """
    Contextual Loss

    Measures feature distribution similarity (useful for exemplar-based methods).

    Reference: https://arxiv.org/abs/1803.02077
    """
    def __init__(self, layers=['relu3_3'], h=0.5, device='cuda'):
        super().__init__()

        self.h = h  # bandwidth parameter
        self.perceptual = PerceptualLoss(layers, device)

    def forward(self, pred, target):
        """
        Simplified contextual loss using cosine similarity

        Args:
            pred: [B, 2, H, W]
            target: [B, 2, H, W]

        Returns:
            loss: scalar
        """
        # Flatten spatial dimensions
        pred_flat = pred.flatten(2)  # [B, 2, H*W]
        target_flat = target.flatten(2)

        # Normalize
        pred_norm = F.normalize(pred_flat, dim=1)
        target_norm = F.normalize(target_flat, dim=1)

        # Cosine similarity
        similarity = torch.bmm(pred_norm.transpose(1, 2), target_norm)  # [B, H*W, H*W]

        # Contextual similarity (max over target features)
        cx = torch.max(similarity, dim=2)[0]  # [B, H*W]

        # Contextual loss (negative log)
        loss = -torch.log(cx + 1e-5).mean()

        return loss


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
        - Contextual: 0.1
        - Temporal: 0.5 (if used)
    """
    def __init__(self,
                 lambda_l1=1.0,
                 lambda_perceptual=0.05,
                 lambda_contextual=0.1,
                 lambda_temporal=0.5,
                 use_temporal=True,
                 device='cuda'):
        super().__init__()

        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_contextual = lambda_contextual
        self.lambda_temporal = lambda_temporal
        self.use_temporal = use_temporal

        # Loss components
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss(device=device)
        self.contextual_loss = ContextualLoss(device=device)

        if use_temporal:
            self.temporal_loss = TemporalLoss()

    def forward(self, pred_ab, gt_ab, flow=None, mask=None, prev_pred_ab=None):
        """
        Compute total loss

        Args:
            pred_ab: [B, 2, H, W] predicted AB channels
            gt_ab: [B, 2, H, W] ground truth AB channels
            flow: [B, 2, H, W] optical flow (for temporal loss)
            mask: [B, 1, H, W] valid mask (for temporal loss)
            prev_pred_ab: [B, 2, H, W] previous frame prediction (for temporal loss)

        Returns:
            total_loss: scalar
            loss_dict: dictionary of individual losses
        """
        # L1 Loss
        loss_l1 = self.l1_loss(pred_ab, gt_ab)

        # Perceptual Loss
        loss_perceptual = self.perceptual_loss(pred_ab, gt_ab)

        # Contextual Loss
        loss_contextual = self.contextual_loss(pred_ab, gt_ab)

        # Total loss
        total_loss = (
            self.lambda_l1 * loss_l1 +
            self.lambda_perceptual * loss_perceptual +
            self.lambda_contextual * loss_contextual
        )

        loss_dict = {
            'l1': loss_l1.item(),
            'perceptual': loss_perceptual.item(),
            'contextual': loss_contextual.item(),
        }

        # Temporal Loss (optional)
        if self.use_temporal and flow is not None and prev_pred_ab is not None:
            loss_temporal = self.temporal_loss(prev_pred_ab, pred_ab, flow, mask)
            total_loss += self.lambda_temporal * loss_temporal
            loss_dict['temporal'] = loss_temporal.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict
