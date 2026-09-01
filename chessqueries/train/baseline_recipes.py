"""Training recipes for the ResNeXt baseline on the joint CR+CC+SLCC set.

Two named recipes answer two questions about V2's headroom:

- ``FAITHFUL`` reproduces Masouris & van Gemert (VISAPP 2024) exactly — BCE on
  one-hot targets, Adam 1e-3 with a x0.1 step at epoch 100, 1024px + their
  normalization, no augmentation — and swaps *only* the training data to the
  joint set ("their model, more data").
- ``V2`` transplants V2's training recipe onto the same ResNeXt — per-square
  softmax CE, AdamW + cosine, 644px ImageNet-norm, geometric_mild — isolating
  the model (backbone + pretraining + head) as the one variable vs V2.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from chessqueries.data.transforms import Augment, Normalization


class BaselineRecipe(str, Enum):
    FAITHFUL = "faithful"
    V2 = "v2"


class ResNeXtLoss(str, Enum):
    BCE = "bce"                # ChessReD: binary_cross_entropy_with_logits on one-hot 64x13
    SOFTMAX_CE = "softmax_ce"  # V2: per-square 13-way softmax cross-entropy


class Optimizer(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"


class Schedule(str, Enum):
    NONE = "none"
    STEP = "step"      # StepLR(step_size, gamma) — ChessReD's x0.1 @ epoch 100
    COSINE = "cosine"  # CosineAnnealingLR(T_max=epochs) — V2


@dataclass(frozen=True)
class RecipeSpec:
    """Everything that differs between the two baseline runs, validated up front."""

    loss: ResNeXtLoss
    optimizer: Optimizer
    lr: float
    weight_decay: float
    schedule: Schedule
    step_size: int      # only used when schedule is STEP
    gamma: float        # only used when schedule is STEP
    epochs: int
    batch_size: int
    resolution: int
    normalization: Normalization
    augment: Augment
    empty_weight: float  # only used by SOFTMAX_CE
    grad_clip: float    # 0.0 disables
    seed: int
    early_stop_patience: int  # epochs of no val_board_acc gain before stopping; 0 disables

    def __post_init__(self) -> None:
        if not isinstance(self.augment, Augment):
            object.__setattr__(self, "augment", Augment(self.augment))
        if self.loss is ResNeXtLoss.BCE and self.empty_weight != 1.0:
            raise ValueError("empty_weight only applies to SOFTMAX_CE")


RECIPES: dict[BaselineRecipe, RecipeSpec] = {
    BaselineRecipe.FAITHFUL: RecipeSpec(
        loss=ResNeXtLoss.BCE,
        optimizer=Optimizer.ADAM,
        lr=1e-3,
        weight_decay=0.0,
        schedule=Schedule.STEP,
        step_size=100,
        gamma=0.1,
        epochs=200,
        batch_size=8,
        resolution=1024,
        normalization=Normalization.CHESSRED,
        augment=Augment.NONE,
        empty_weight=1.0,
        grad_clip=0.0,
        seed=42,
        early_stop_patience=20,  # paper: "200 epochs with early stopping"
    ),
    BaselineRecipe.V2: RecipeSpec(
        loss=ResNeXtLoss.SOFTMAX_CE,
        optimizer=Optimizer.ADAMW,
        lr=1.4e-4,
        weight_decay=0.05,
        schedule=Schedule.COSINE,
        step_size=0,
        gamma=0.0,
        epochs=45,
        batch_size=8,
        resolution=644,
        normalization=Normalization.IMAGENET,
        augment=Augment.GEOMETRIC_MILD,
        empty_weight=1.0,
        grad_clip=1.0,
        seed=42,
        early_stop_patience=0,  # V2 runs the full 45 epochs (no early stopping)
    ),
}
