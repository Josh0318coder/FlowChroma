"""
Fusion UNet Models

Defines network architectures for fusing MemFlow and SwinTExCo predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlaceholderFusion(nn.Module):
    """
    Placeholder Fusion Network (for testing data flow)

    This is a simple weighted average fusion used to verify that
    the entire system works before implementing the real UNet.

    Input:
        - memflow_ab: [B, 2, H, W] MemFlow AB prediction
        - swintexco_ab: [B, 2, H, W] SwinTExCo AB prediction
        - memflow_conf: [B, 1, H, W] MemFlow confidence
        - swintexco_sim: [B, 1, H, W] SwinTExCo similarity
        - L_channel: [B, 1, H, W] Luminance channel

    Output:
        - fused_ab: [B, 2, H, W] Fused AB channels
    """
    def __init__(self):
        super().__init__()

    def forward(self, memflow_ab, swintexco_ab, memflow_conf, swintexco_sim, L_channel):
        """
        Simple confidence-based weighted average
        """
        # Normalize confidences to sum to 1
        total_conf = memflow_conf + swintexco_sim + 1e-6
        weight_memflow = memflow_conf / total_conf
        weight_swin = swintexco_sim / total_conf

        # Weighted fusion
        fused_ab = weight_memflow * memflow_ab + weight_swin * swintexco_ab

        return fused_ab


class SimpleFusionNet(nn.Module):
    """
    Simple Fusion Network

    A lightweight network for intelligent fusion based on dual confidence signals.

    Input channels: 7
        - memflow_ab: 2 channels
        - swintexco_ab: 2 channels
        - memflow_conf: 1 channel
        - swintexco_sim: 1 channel
        - L_channel: 1 channel

    Output channels: 2 (AB channels)
    """
    def __init__(self, in_channels=7, out_channels=2):
        super().__init__()

        self.net = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 2
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 3
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Output layer
            nn.Conv2d(16, out_channels, kernel_size=1)
        )

    def forward(self, memflow_ab, swintexco_ab, memflow_conf, swintexco_sim, L_channel):
        """
        Forward pass with residual connection

        Args:
            memflow_ab: [B, 2, H, W]
            swintexco_ab: [B, 2, H, W]
            memflow_conf: [B, 1, H, W]
            swintexco_sim: [B, 1, H, W]
            L_channel: [B, 1, H, W]

        Returns:
            fused_ab: [B, 2, H, W]
        """
        # Concatenate all inputs
        x = torch.cat([
            memflow_ab,      # 2
            swintexco_ab,    # 2
            memflow_conf,    # 1
            swintexco_sim,   # 1
            L_channel        # 1
        ], dim=1)  # [B, 7, H, W]

        # Forward through network
        residual = self.net(x)  # [B, 2, H, W]

        # Residual connection: MemFlow as backbone
        fused_ab = memflow_ab + residual

        return fused_ab


class UNetFusionNet(nn.Module):
    """
    UNet-based Fusion Network (optional, for complex scenarios)

    A more complex architecture with encoder-decoder structure and skip connections.
    Use this if SimpleFusionNet doesn't provide enough capacity.
    """
    def __init__(self, in_channels=7, out_channels=2):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)

        # Decoder
        self.dec3 = self._upconv_block(512, 256)
        self.dec2 = self._upconv_block(256, 128)
        self.dec1 = self._upconv_block(128, 64)

        # Output
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # Pooling
        self.pool = nn.MaxPool2d(2)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, memflow_ab, swintexco_ab, memflow_conf, swintexco_sim, L_channel):
        # Concatenate inputs
        x = torch.cat([memflow_ab, swintexco_ab, memflow_conf, swintexco_sim, L_channel], dim=1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder with skip connections
        d3 = self.dec3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self._conv_block(d3.shape[1], 256)(d3)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self._conv_block(d2.shape[1], 128)(d2)

        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self._conv_block(d1.shape[1], 64)(d1)

        # Output
        residual = self.out_conv(d1)
        fused_ab = memflow_ab + residual

        return fused_ab


class FusionNetV1(nn.Module):
    """
    FusionNet V1 - Based on VidNet UNet Architecture

    Designed for joint training with SwinTExCo, inspired by ColorVidNet.
    Uses 8-channel input for symmetric fusion of MemFlow and SwinTExCo predictions.

    Input channels: 8
        - memflow_lab: 3 channels (L + ab)
        - memflow_conf: 1 channel
        - swintexco_lab: 3 channels (L + ab)
        - swintexco_sim: 1 channel

    Output channels: 2 (ab channels)

    Architecture:
        - Encoder: 64 → 128 → 256 → 512 (3x downsampling)
        - Bottleneck: 512 → 512 (2 layers dilated conv)
        - Decoder: 512 → 256 → 128 → 128 (3x upsampling)
        - Skip connections from encoder to decoder
        - Residual output based on MemFlow ab
    """

    def __init__(self, ic=8):
        super(FusionNetV1, self).__init__()

        # Encoder
        self.conv1_1 = nn.Sequential(nn.Conv2d(ic, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv1_2norm = nn.InstanceNorm2d(64)
        self.conv1_2norm_ss = nn.Conv2d(64, 64, 1, 2, bias=False, groups=64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv2_2norm = nn.InstanceNorm2d(128)
        self.conv2_2norm_ss = nn.Conv2d(128, 128, 1, 2, bias=False, groups=128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3norm = nn.InstanceNorm2d(256)
        self.conv3_3norm_ss = nn.Conv2d(256, 256, 1, 2, bias=False, groups=256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3norm = nn.InstanceNorm2d(512)

        # Bottleneck (dilated convolutions)
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_3norm = nn.InstanceNorm2d(512)

        self.conv6_1 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_2 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_3 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_3norm = nn.InstanceNorm2d(512)

        self.conv7_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_3norm = nn.InstanceNorm2d(512)

        # Decoder (upsampling)
        self.conv8_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(512, 256, 3, 1, 1))
        self.conv3_3_short = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_3norm = nn.InstanceNorm2d(256)

        self.conv9_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(256, 128, 3, 1, 1))
        self.conv2_2_short = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv9_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv9_2norm = nn.InstanceNorm2d(128)

        self.conv10_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(128, 128, 3, 1, 1))
        self.conv1_2_short = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv10_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv10_ab = nn.Conv2d(128, 2, 1, 1)

        # Activation functions
        self.relu1_1 = nn.PReLU()
        self.relu1_2 = nn.PReLU()
        self.relu2_1 = nn.PReLU()
        self.relu2_2 = nn.PReLU()
        self.relu3_1 = nn.PReLU()
        self.relu3_2 = nn.PReLU()
        self.relu3_3 = nn.PReLU()
        self.relu4_1 = nn.PReLU()
        self.relu4_2 = nn.PReLU()
        self.relu4_3 = nn.PReLU()
        self.relu5_1 = nn.PReLU()
        self.relu5_2 = nn.PReLU()
        self.relu5_3 = nn.PReLU()
        self.relu6_1 = nn.PReLU()
        self.relu6_2 = nn.PReLU()
        self.relu6_3 = nn.PReLU()
        self.relu7_1 = nn.PReLU()
        self.relu7_2 = nn.PReLU()
        self.relu7_3 = nn.PReLU()
        self.relu8_1_comb = nn.PReLU()
        self.relu8_2 = nn.PReLU()
        self.relu8_3 = nn.PReLU()
        self.relu9_1_comb = nn.PReLU()
        self.relu9_2 = nn.PReLU()
        self.relu10_1_comb = nn.PReLU()
        self.relu10_2 = nn.LeakyReLU(0.2, True)

    def forward(self, memflow_lab, memflow_conf, swintexco_ab, swintexco_sim, L_channel):
        """
        Forward pass with 8-channel input

        Args:
            memflow_lab: [B, 3, H, W] - MemFlow LAB prediction
            memflow_conf: [B, 1, H, W] - MemFlow confidence
            swintexco_ab: [B, 2, H, W] - SwinTExCo ab prediction
            swintexco_sim: [B, 1, H, W] - SwinTExCo similarity
            L_channel: [B, 1, H, W] - Input luminance

        Returns:
            fused_lab: [B, 3, H, W] - Fused LAB prediction
        """
        # Construct SwinTExCo LAB
        swintexco_lab = torch.cat([L_channel, swintexco_ab], dim=1)

        # Concatenate all inputs (8 channels)
        x = torch.cat([
            memflow_lab,      # 3
            memflow_conf,     # 1
            swintexco_lab,    # 3
            swintexco_sim,    # 1
        ], dim=1)  # [B, 8, H, W]

        # Encoder
        conv1_1 = self.relu1_1(self.conv1_1(x))
        conv1_2 = self.relu1_2(self.conv1_2(conv1_1))
        conv1_2norm = self.conv1_2norm(conv1_2)
        conv1_2norm_ss = self.conv1_2norm_ss(conv1_2norm)

        conv2_1 = self.relu2_1(self.conv2_1(conv1_2norm_ss))
        conv2_2 = self.relu2_2(self.conv2_2(conv2_1))
        conv2_2norm = self.conv2_2norm(conv2_2)
        conv2_2norm_ss = self.conv2_2norm_ss(conv2_2norm)

        conv3_1 = self.relu3_1(self.conv3_1(conv2_2norm_ss))
        conv3_2 = self.relu3_2(self.conv3_2(conv3_1))
        conv3_3 = self.relu3_3(self.conv3_3(conv3_2))
        conv3_3norm = self.conv3_3norm(conv3_3)
        conv3_3norm_ss = self.conv3_3norm_ss(conv3_3norm)

        conv4_1 = self.relu4_1(self.conv4_1(conv3_3norm_ss))
        conv4_2 = self.relu4_2(self.conv4_2(conv4_1))
        conv4_3 = self.relu4_3(self.conv4_3(conv4_2))
        conv4_3norm = self.conv4_3norm(conv4_3)

        # Bottleneck
        conv5_1 = self.relu5_1(self.conv5_1(conv4_3norm))
        conv5_2 = self.relu5_2(self.conv5_2(conv5_1))
        conv5_3 = self.relu5_3(self.conv5_3(conv5_2))
        conv5_3norm = self.conv5_3norm(conv5_3)

        conv6_1 = self.relu6_1(self.conv6_1(conv5_3norm))
        conv6_2 = self.relu6_2(self.conv6_2(conv6_1))
        conv6_3 = self.relu6_3(self.conv6_3(conv6_2))
        conv6_3norm = self.conv6_3norm(conv6_3)

        conv7_1 = self.relu7_1(self.conv7_1(conv6_3norm))
        conv7_2 = self.relu7_2(self.conv7_2(conv7_1))
        conv7_3 = self.relu7_3(self.conv7_3(conv7_2))
        conv7_3norm = self.conv7_3norm(conv7_3)

        # Decoder with skip connections
        conv8_1 = self.conv8_1(conv7_3norm)
        conv3_3_short = self.conv3_3_short(conv3_3norm)
        conv8_1_comb = self.relu8_1_comb(conv8_1 + conv3_3_short)
        conv8_2 = self.relu8_2(self.conv8_2(conv8_1_comb))
        conv8_3 = self.relu8_3(self.conv8_3(conv8_2))
        conv8_3norm = self.conv8_3norm(conv8_3)

        conv9_1 = self.conv9_1(conv8_3norm)
        conv2_2_short = self.conv2_2_short(conv2_2norm)
        conv9_1_comb = self.relu9_1_comb(conv9_1 + conv2_2_short)
        conv9_2 = self.relu9_2(self.conv9_2(conv9_1_comb))
        conv9_2norm = self.conv9_2norm(conv9_2)

        conv10_1 = self.conv10_1(conv9_2norm)
        conv1_2_short = self.conv1_2_short(conv1_2norm)
        conv10_1_comb = self.relu10_1_comb(conv10_1 + conv1_2_short)
        conv10_2 = self.relu10_2(self.conv10_2(conv10_1_comb))

        # Output residual ab
        residual_ab = torch.tanh(self.conv10_ab(conv10_2))

        # Residual connection based on MemFlow ab
        memflow_ab = memflow_lab[:, 1:3, :, :]
        fused_ab = memflow_ab + residual_ab

        # Construct final LAB
        fused_lab = torch.cat([L_channel, fused_ab], dim=1)

        return fused_lab
