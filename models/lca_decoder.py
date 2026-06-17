import torch
import torch.nn as nn


class ChannelAttention3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attn(x)


class LCABlock(nn.Module):
    """Depthwise 3x3x3 + pointwise 1x1x1 convolution + channel attention."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.GELU()
        self.attn = ChannelAttention3D(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.act(self.norm(x))
        return self.attn(x)


class LCADecoder(nn.Module):
    def __init__(self, encoder_channels, out_channels: int = 4):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = LCABlock(c3 + c3, c3)
        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = LCABlock(c2 + c2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = LCABlock(c1 + c1, c1)
        self.out = nn.Conv3d(c1, out_channels, kernel_size=1)

    def forward(self, features):
        f1, f2, f3, f4 = features
        x = self.dec3(torch.cat([self.up3(f4), f3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), f2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), f1], dim=1))
        return self.out(x)
