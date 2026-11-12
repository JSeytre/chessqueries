"""Auxiliary viewing direction from encoder tokens."""

import torch
from torch import nn


def unit_direction(vectors: torch.Tensor) -> torch.Tensor:
    """Normalize in float32; a zero predictor output uses up as its fallback."""
    values = vectors.float()
    lengths = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    fallback = torch.zeros_like(values)
    fallback[..., 0] = 1
    return torch.where(lengths > 1e-6, values / lengths.clamp_min(1e-6), fallback)


class OrientationPredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2)
        )
        nn.init.constant_(self.net[-1].bias, 0.0)
        with torch.no_grad():
            self.net[-1].bias[0] = 1.0

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        return unit_direction(self.net(memory.mean(dim=1)))
