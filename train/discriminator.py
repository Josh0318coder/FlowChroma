"""
Discriminator Network for GAN Training

Based on SwinTExCo's Discriminator_x64_224 with Spectral Normalization
and Self-Attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


def l2normalize(v, eps=1e-12):
    """L2 normalization for spectral normalization"""
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """Spectral Normalization wrapper"""
    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class SelfAttention(nn.Module):
    """Self-Attention Layer with Spectral Normalization"""

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 1, kernel_size=1))
        self.key_conv = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 1, kernel_size=1))
        self.value_conv = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class Discriminator(nn.Module):
    """
    Discriminator Network for 224x224 images

    Based on SwinTExCo's Discriminator_x64_224.
    Uses Spectral Normalization and Self-Attention.

    Input: [B, 6, 224, 224] - concatenated predicted LAB and ground truth LAB
    Output:
        - discriminator score: [B, 1]
        - feature map: [B, 256, 14, 14]
    """

    def __init__(self, in_channels=6, ndf=64):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.ndf = ndf

        self.layer1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.in_channels, self.ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.ndf, self.ndf, 4, 2, 1)),
            nn.InstanceNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.attention = SelfAttention(self.ndf)

        self.layer3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1)),
            nn.InstanceNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1)),
            nn.InstanceNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer5 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1)),
            nn.InstanceNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer6 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.ndf * 8, self.ndf * 16, 4, 2, 1)),
            nn.InstanceNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.last = SpectralNorm(nn.Conv2d(self.ndf * 16, 1, [3, 3], 1, 0))

    def forward(self, input):
        """
        Args:
            input: [B, 6, H, W] - concatenated LAB images

        Returns:
            output: [B, 1] - discriminator score
            feature4: [B, 256, 14, 14] - intermediate features
        """
        feature1 = self.layer1(input)
        feature2 = self.layer2(feature1)
        feature_attention = self.attention(feature2)
        feature3 = self.layer3(feature_attention)
        feature4 = self.layer4(feature3)
        feature5 = self.layer5(feature4)
        feature6 = self.layer6(feature5)
        output = self.last(feature6)
        output = F.avg_pool2d(output, output.size()[2:]).view(output.size()[0], -1)

        return output, feature4
