import torch.nn as nn


class ConvPatchEmbed3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        return self.norm(self.proj(x))


class SwinBlockPlaceholder(nn.Module):
    """
    Placeholder for a 3D Swin-Transformer block.

    Replace with full shifted-window self-attention for production use.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(channels),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class SwinTransformer3DEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, base_channels: int = 48):
        super().__init__()
        c = base_channels
        self.stem = ConvPatchEmbed3D(in_channels, c)
        self.stage1 = nn.Sequential(SwinBlockPlaceholder(c), SwinBlockPlaceholder(c))
        self.down1 = nn.Conv3d(c, c * 2, kernel_size=2, stride=2)
        self.stage2 = nn.Sequential(SwinBlockPlaceholder(c * 2), SwinBlockPlaceholder(c * 2))
        self.down2 = nn.Conv3d(c * 2, c * 4, kernel_size=2, stride=2)
        self.stage3 = nn.Sequential(SwinBlockPlaceholder(c * 4), SwinBlockPlaceholder(c * 4))
        self.down3 = nn.Conv3d(c * 4, c * 8, kernel_size=2, stride=2)
        self.stage4 = nn.Sequential(SwinBlockPlaceholder(c * 8), SwinBlockPlaceholder(c * 8))

    def forward(self, x):
        f1 = self.stage1(self.stem(x))
        f2 = self.stage2(self.down1(f1))
        f3 = self.stage3(self.down2(f2))
        f4 = self.stage4(self.down3(f3))
        return [f1, f2, f3, f4]
