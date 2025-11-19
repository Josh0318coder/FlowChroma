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
