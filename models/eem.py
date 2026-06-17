import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeEnhancementModule(nn.Module):
    """Sobel/Laplacian-inspired edge enhancement with learnable attention fusion."""

    def __init__(self, channels: int):
        super().__init__()
        self.edge_fusion = nn.Sequential(
            nn.Conv3d(channels * 3, channels, kernel_size=1),
            nn.InstanceNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def _sobel_3d(self, x):
        dx = F.pad(x[:, :, :, :, 1:] - x[:, :, :, :, :-1], (0, 1, 0, 0, 0, 0))
        dy = F.pad(x[:, :, :, 1:, :] - x[:, :, :, :-1, :], (0, 0, 0, 1, 0, 0))
        dz = F.pad(x[:, :, 1:, :, :] - x[:, :, :-1, :, :], (0, 0, 0, 0, 0, 1))
        return torch.sqrt(dx * dx + dy * dy + dz * dz + 1e-6)

    def _laplacian_3d(self, x):
        lap = -6 * x
        lap += F.pad(x[:, :, :, :, 1:], (0, 1, 0, 0, 0, 0))
        lap += F.pad(x[:, :, :, :, :-1], (1, 0, 0, 0, 0, 0))
        lap += F.pad(x[:, :, :, 1:, :], (0, 0, 0, 1, 0, 0))
        lap += F.pad(x[:, :, :, :-1, :], (0, 0, 1, 0, 0, 0))
        lap += F.pad(x[:, :, 1:, :, :], (0, 0, 0, 0, 0, 1))
        lap += F.pad(x[:, :, :-1, :, :], (0, 0, 0, 0, 1, 0))
        return torch.abs(lap)

    def forward(self, x):
        sobel = self._sobel_3d(x)
        lap = self._laplacian_3d(x)
        gate = self.edge_fusion(torch.cat([x, sobel, lap], dim=1))
        return x * (1.0 + gate)
