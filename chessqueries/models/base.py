"""The `BoardRecognizer` ABC: the contract every recognizer implements.

A recognizer maps a batch of images to per-square class ids (our canonical
label space).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


class BoardRecognizer(nn.Module, ABC):
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
