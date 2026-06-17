import torch
import torch.nn as nn

from .swin_encoder import SwinTransformer3DEncoder
from .eem import EdgeEnhancementModule
from .lca_decoder import LCADecoder


class EESUNet(nn.Module):
    """EESU-Net for 3D brain-tumor segmentation."""

    def __init__(self, in_channels: int = 4, out_channels: int = 4, base_channels: int = 48):
        super().__init__()
        self.encoder = SwinTransformer3DEncoder(in_channels=in_channels, base_channels=base_channels)
        self.eem = EdgeEnhancementModule(base_channels * 8)
        self.decoder = LCADecoder(
            encoder_channels=[base_channels, base_channels * 2, base_channels * 4, base_channels * 8],
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        features[-1] = self.eem(features[-1])
        return self.decoder(features)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
