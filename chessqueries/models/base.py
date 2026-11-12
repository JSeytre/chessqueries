"""The `BoardRecognizer` ABC: the contract every recognizer implements.

A recognizer maps a batch of images to per-square class ids (our canonical
label space).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

from chessqueries.core import NUM_PIECES


@dataclass(frozen=True)
class Recognition:
    labels: torch.Tensor
    directions: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.labels.ndim != 2 or self.labels.shape[1] != 64:
            raise ValueError("expected labels shaped [batch,64]")
        if self.labels.dtype != torch.int64 or ((self.labels < 0) | (self.labels >= NUM_PIECES)).any():
            raise ValueError("invalid piece labels")
        if self.directions is not None:
            if (self.directions.shape != (len(self.labels), 2)
                    or self.directions.dtype != torch.float32
                    or not torch.isfinite(self.directions).all()
                    or (self.directions.abs() > 1).any()):
                raise ValueError("expected finite directions shaped [batch,2]")
            lengths = self.directions.float().norm(dim=-1)
            if not torch.allclose(lengths, torch.ones_like(lengths), atol=1e-4, rtol=0):
                raise ValueError("directions must be unit vectors")


class BoardRecognizer(nn.Module, ABC):
    def recognize(self, x: torch.Tensor) -> Recognition:
        return Recognition(self.predict_labels(x))

    @abstractmethod
    def predict_labels(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) -> (B,64) class ids in our canonical label space."""
        ...


@dataclass(frozen=True)
class EvalPredictions:
    """An inference pass over a dataset: predicted boards aligned with their ground
    truth, both as per-board 64-label lists — what `metrics.aggregate` consumes."""

    preds: list[list[int]]
    gts: list[list[int]]

    def __post_init__(self) -> None:
        if len(self.preds) != len(self.gts):
            raise ValueError(f"{len(self.preds)} preds vs {len(self.gts)} gts")

    def __len__(self) -> int:
        return len(self.preds)


@torch.no_grad()
def predict_all(
    model: BoardRecognizer,
    dataset,
    *,
    device: str = "cpu",
    batch_size: int = 32,
    workers: int = 8,
    desc: str = "eval",
) -> EvalPredictions:
    """Run a recognizer over a `BoardImageDataset`."""
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)
    preds: list[list[int]] = []
    gts: list[list[int]] = []
    for imgs, labels in tqdm(loader, desc=desc):
        out = model.predict_labels(imgs.to(device)).cpu()
        preds.extend(out.tolist())
        gts.extend(labels.tolist())
    return EvalPredictions(preds=preds, gts=gts)
