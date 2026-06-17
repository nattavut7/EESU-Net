import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_1h = F.one_hot(targets.long(), num_classes=logits.shape[1]).permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        inter = torch.sum(probs * targets_1h, dims)
        denom = torch.sum(probs + targets_1h, dims)
        return 1.0 - ((2.0 * inter + self.smooth) / (denom + self.smooth)).mean()


class BoundaryLoss(nn.Module):
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_1h = F.one_hot(targets.long(), num_classes=logits.shape[1]).permute(0, 4, 1, 2, 3).float()

        def grad(x):
            dx = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).mean()
            dy = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).mean()
            dz = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).mean()
            return dx + dy + dz

        return torch.abs(grad(probs) - grad(targets_1h))


class DiceBoundaryLoss(nn.Module):
    def __init__(self, lambda_dice: float = 1.0, lambda_boundary: float = 0.1):
        super().__init__()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
        self.lambda_dice = lambda_dice
        self.lambda_boundary = lambda_boundary

    def forward(self, logits, targets):
        return self.lambda_dice * self.dice(logits, targets) + self.lambda_boundary * self.boundary(logits, targets)
