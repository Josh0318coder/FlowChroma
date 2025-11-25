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

    def forward(self, pred, target, feature_centering=False, chunk_size=256):
        """
        Memory-efficient Contextual Loss (Simplified stable version)

        Args:
            pred: [B, 2, H, W] predicted AB channels
            target: [B, 2, H, W] ground truth AB channels
            feature_centering: Whether to subtract mean (default: False for stability)
            chunk_size: Process features in chunks to save memory (default: 256)

        Returns:
            loss: scalar (averaged over batch)
        """
        batch_size = pred.shape[0]
        feature_depth = pred.shape[1]

        # Flatten spatial dimensions
        pred_flat = pred.view(batch_size, feature_depth, -1)  # [B, 2, H*W]
        target_flat = target.view(batch_size, feature_depth, -1)

        # L2 normalization (more stable than feature centering + normalize)
        pred_norm = F.normalize(pred_flat, dim=1, eps=1e-8)
        target_norm = F.normalize(target_flat, dim=1, eps=1e-8)

        # Process in chunks to save memory
        CX_list = []

        for b in range(batch_size):
            pred_b = pred_norm[b]  # [2, H*W]
            target_b = target_norm[b]  # [2, H*W]

            N = pred_b.shape[1]  # H*W
            cx_values = []

            # Split into chunks
            for i in range(0, N, chunk_size):
                end_i = min(i + chunk_size, N)
                pred_chunk = pred_b[:, i:end_i]  # [2, chunk]

                # Cosine similarity (not distance)
                similarity = torch.matmul(pred_chunk.t(), target_b)  # [chunk, H*W]

                # Clamp to valid range [-1, 1] for numerical stability
                similarity = torch.clamp(similarity, -1.0, 1.0)

                # Max similarity for this chunk
                cx_chunk = torch.max(similarity, dim=1)[0]  # [chunk]

                # Clamp to avoid log(0)
                cx_chunk = torch.clamp(cx_chunk, min=1e-6, max=1.0)
                cx_values.append(cx_chunk)

            # Concatenate all chunks
            CX_b = torch.cat(cx_values, dim=0)  # [H*W]
            CX_list.append(torch.mean(CX_b))

        # Stack batch results
        CX = torch.stack(CX_list)  # [B]

        # Contextual loss (negative log)
        # CX is in [1e-6, 1], so -log(CX) is in [0, ~14]
        loss = -torch.log(CX)  # [B]

        # Average over batch
        return loss.mean()


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
                 contextual_chunk_size=256,
                 device='cuda'):
        super().__init__()

        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_contextual = lambda_contextual
        self.lambda_temporal = lambda_temporal
        self.contextual_chunk_size = contextual_chunk_size
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

        # Total loss
        total_loss = (
            self.lambda_l1 * loss_l1 +
            self.lambda_perceptual * loss_perceptual
        )

        loss_dict = {
            'l1': loss_l1.item(),
            'perceptual': loss_perceptual.item(),
        }

        # Contextual Loss (only compute if weight > 0 to save memory)
        if self.lambda_contextual > 0:
            loss_contextual = self.contextual_loss(
                pred_ab, gt_ab,
                chunk_size=self.contextual_chunk_size
            )
            total_loss += self.lambda_contextual * loss_contextual
            loss_dict['contextual'] = loss_contextual.item()
        else:
            loss_dict['contextual'] = 0.0

        # Temporal Loss (optional)
        if self.use_temporal and flow is not None and prev_pred_ab is not None:
            loss_temporal = self.temporal_loss(prev_pred_ab, pred_ab, flow, mask)
            total_loss += self.lambda_temporal * loss_temporal
            loss_dict['temporal'] = loss_temporal.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict
