"""Load a LoRA-adapted recognizer and emit per-square distributions for the
cross-check. Wraps the ViT-L base + SLCC adapter saved by ``scripts/lora_fewshot.py``
(``{adapter, base_checkpoint, targets, rank, alpha, resolution}``) and feeds crops
through the exact eval transform the model trained on, so a board read off a video
frame matches a board read off a stored dataset crop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch



@dataclass(frozen=True)
class Prediction:
    """The recognizer's per-square output for one crop."""

    log_probs: np.ndarray  # (64, 13) log-softmax over the piece classes, FEN order


class Recognizer:
    """A loaded (base + LoRA) ``ChessQueriesModel`` that scores crops. Heavy to build
    (loads the ViT-L checkpoint), so construct once and reuse across a video."""

    def __init__(self, model, transform, device: str) -> None:
        self.model = model
        self.transform = transform
        self.device = device

    @classmethod
    def from_adapter(cls, adapter_path: Path, device: str | None = None) -> "Recognizer":
        from chessqueries.data.transforms import build_transform
        from chessqueries.models.lora import inject_lora
        from chessqueries.train.lit import LitChessQueriesModel

        ad = torch.load(adapter_path, map_location="cpu")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        lit = LitChessQueriesModel.load_from_checkpoint(ad["base_checkpoint"], map_location=device)
        model = lit.model.to(device).eval()
        inject_lora(
            model.encoder,
            targets=tuple(ad["targets"]),
            r=ad["rank"],
            alpha=ad["alpha"],
            dropout=ad.get("dropout", 0.0),
        )
        model.to(device)
        _, unexpected = model.load_state_dict(ad["adapter"], strict=False)
        if unexpected:
            raise ValueError(f"adapter has unexpected keys: {unexpected[:5]}...")
        model.eval()
        return cls(model, build_transform(ad["resolution"], train=False), device)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: Path, resolution: int, device: str | None = None
    ) -> "Recognizer":
        """Load a plain (non-LoRA) ``LitChessQueriesModel`` checkpoint — e.g. the joint
        V2 model, which is a full fine-tune, not an adapter. ``resolution`` must be the
        checkpoint's eval resolution (it lives on the DataModule, not in the saved
        hparams): ViT-L V2 trained at 644. Mirrors the eval transform exactly (see
        ``scripts/eval_cross_dataset.py``): ``train=False``, default IMAGENET norm."""
        from chessqueries.data.transforms import build_transform
        from chessqueries.train.lit import LitChessQueriesModel

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        lit = LitChessQueriesModel.load_from_checkpoint(checkpoint_path, map_location=device)
        model = lit.model.to(device).eval()
        return cls(model, build_transform(resolution, train=False), device)

    @torch.no_grad()
    def predict_crop(self, crop_bgr: np.ndarray) -> Prediction:
        """Score a BGR crop (as read by OpenCV). Matched to ``read_image``: RGB,
        channels-first, float in [0,255] — the transform divides by 255 itself."""
        rgb = np.ascontiguousarray(crop_bgr[:, :, ::-1])
        t = torch.from_numpy(rgb).permute(2, 0, 1).float()
        x = self.transform(t).unsqueeze(0).to(self.device)
        logits = self.model(x)[0]
        return Prediction(log_probs=torch.log_softmax(logits, dim=-1).cpu().numpy())
