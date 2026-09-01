"""ChessReD baseline: ResNeXt-101 with a 64x13 linear head — a clean
reimplementation of Masouris & van Gemert (VISAPP 2024) that loads their
released Lightning checkpoint.

The checkpoint's 13 logits are in ChessReD's category-id order, not our `Piece`
order; :meth:`set_class_order` remaps them so predictions land in our space.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms

from chessqueries.core import NUM_PIECES
from chessqueries.data.chessred import CATEGORY_TO_PIECE
from chessqueries.data.transforms import CHESSRED_MEAN as _MEAN
from chessqueries.data.transforms import CHESSRED_STD as _STD
from chessqueries.models.base import BoardRecognizer
from chessqueries.train.baseline_recipes import Optimizer, ResNeXtLoss, Schedule

NUM_CLASSES = NUM_PIECES

# Project checkpoints contain these plain-data enums in Lightning's saved
# hyperparameters. Older released runs used the package's former ``chesswizard``
# name, so allow those three historical paths as well as the current ones.
_PROJECT_CHECKPOINT_SAFE_GLOBALS = [
    Optimizer,
    ResNeXtLoss,
    Schedule,
    (Optimizer, "chesswizard.train.baseline_recipes.Optimizer"),
    (ResNeXtLoss, "chesswizard.train.baseline_recipes.ResNeXtLoss"),
    (Schedule, "chesswizard.train.baseline_recipes.Schedule"),
]

# No resize: ChessReD ships the authors' 1024x1024 images (``ChessReD``), the
# exact inputs the checkpoint was trained on; `Resize(1024)` targets the raw
# images, a distribution the checkpoint never saw (4.84% vs the paper's 15.26%).
# The ``ToPILImage`` round-trip on the ``read_image().float()`` [0,255] tensor is
# load-bearing AND destructive: ToPILImage assumes floats are [0,1] and does
# ``mul(255).byte()``, so pixels wrap modulo 256 (e.g. 200.0 -> 56). That wrapped
# input is exactly what the weights were trained on — dropping the round-trip
# costs ~17 squares/board. See tmasouris/end-to-end-chess-recognition#5.
INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ]
)


class ChessReDResNeXt(BoardRecognizer):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        # ``pretrained`` loads torchvision's ImageNet weights — the init ChessReD
        # train from (``weights="DEFAULT"`` in their repo). Leave False when about
        # to ``load_state_dict`` a trained checkpoint (the default load paths).
        weights = "DEFAULT" if pretrained else None
        backbone = models.resnext101_32x8d(weights=weights)
        num_filters = backbone.fc.in_features
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(num_filters, 64 * NUM_CLASSES)
        # Permutation: model-class index -> our class id. Identity until set.
        self.register_buffer("_class_perm", torch.arange(NUM_CLASSES), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x).flatten(1)
        return self.classifier(x)  # (B, 64*13)

    # -- class-order alignment ------------------------------------------------ #
    def set_class_order(self, category_names: list[str]) -> None:
        """Tell the model what each of its 13 output channels means.

        ``category_names[i]`` is the ChessReD category name (e.g. "white-pawn",
        "empty") for the model's i-th class channel. Read it from the dataset's
        ``annotations.json`` ``categories`` table (sorted by id).
        """
        if len(category_names) != NUM_CLASSES:
            raise ValueError(f"Expected {NUM_CLASSES} category names, got {len(category_names)}")
        perm = torch.empty(NUM_CLASSES, dtype=torch.long)
        for model_idx, name in enumerate(category_names):
            perm[model_idx] = int(CATEGORY_TO_PIECE[name])
        self._class_perm = perm.to(self._class_perm.device)

    @torch.no_grad()
    def predict_labels(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) -> (B,64) class ids in our canonical label space."""
        logits = self.forward(x).reshape(-1, 64, NUM_CLASSES)
        model_classes = logits.argmax(dim=2)  # (B,64) in ChessReD class order
        return self._class_perm.to(model_classes.device)[model_classes]

    # -- loading -------------------------------------------------------------- #
    @classmethod
    def _from_state(cls, state: dict) -> "ChessReDResNeXt":
        """Build an eval-mode model from a state dict. Missing keys raise (the model
        would silently evaluate on randomly-initialized weights); extra keys warn."""
        model = cls()
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise ValueError(f"checkpoint is missing {len(missing)} keys: {sorted(missing)[:5]}...")
        if unexpected:
            warnings.warn(f"checkpoint has {len(unexpected)} unused keys: {sorted(unexpected)[:5]}...")
        model.eval()
        return model

    @classmethod
    def from_checkpoint(cls, ckpt_path: str | Path, map_location="cpu") -> "ChessReDResNeXt":
        ckpt = torch.load(str(ckpt_path), map_location=map_location, weights_only=True)
        state = ckpt.get("state_dict", ckpt)
        # Lightning saves params under the LightningModule's attribute names,
        # which match ours (feature_extractor.*, classifier.*).
        return cls._from_state(state)

    @classmethod
    def from_lightning_checkpoint(cls, ckpt_path: str | Path, map_location="cpu") -> "ChessReDResNeXt":
        """Load a checkpoint produced by our own ``LitChessReDResNeXt`` training.

        Unlike :meth:`from_checkpoint` (the authors' weights, in ChessReD's
        category order), a retrained head already predicts in our canonical
        ``Piece`` order, so the class permutation stays identity — do NOT call
        :meth:`set_class_order`. Keys are nested under the LightningModule's
        ``model.`` attribute; we strip that prefix.
        """
        with torch.serialization.safe_globals(_PROJECT_CHECKPOINT_SAFE_GLOBALS):
            ckpt = torch.load(str(ckpt_path), map_location=map_location, weights_only=True)
        state = ckpt.get("state_dict", ckpt)
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
        return cls._from_state(state)
