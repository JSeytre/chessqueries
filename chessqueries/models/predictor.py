"""Single-image inference: checkpoint + image path(s) -> `Board` per image.

The eval path (`predict_all`) runs a recognizer over a `BoardImageDataset`; this
is the arbitrary-image path behind the demo, the CLI, and any downstream caller
that has photos rather than a dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision.io import ImageReadMode, read_image

from chessqueries.core import Board
from chessqueries.data.download import AnonymizedArtifactError, download_release_checkpoint
from chessqueries.data.transforms import build_transform
from chessqueries.models.base import BoardRecognizer

# The released/paper checkpoints are ViT-L trained at 644px. Resolution is NOT a
# LitChessQueriesModel hyperparameter (it belongs to the data module), so it cannot be
# recovered from a checkpoint file — every entrypoint has to state it explicitly.
PAPER_RESOLUTION = 644


def resolve_checkpoint(checkpoint: Path | str | None) -> Path:
    """Resolve an override or download the hash-verified release checkpoint.

    A review export may replace the public URL with an independently pinned
    anonymous release. If an older export instead contains an unavailable
    placeholder, surface its explanation as a plain message rather than a traceback.
    """
    if checkpoint is not None:
        return Path(checkpoint)
    # Downloader status belongs on stderr: stdout is the CLI's TSV/JSON API.
    with redirect_stdout(sys.stderr):
        try:
            return download_release_checkpoint()
        except AnonymizedArtifactError as exc:
            raise SystemExit(str(exc)) from exc


@dataclass(frozen=True)
class Prediction:
    """One image's predicted board."""

    image_path: Path
    board: Board

    @property
    def fen(self) -> str:
        return self.board.to_fen()

    @property
    def lichess_url(self) -> str:
        return self.board.lichess_url()

    def to_record(self) -> dict[str, str]:
        """The serialized (JSON/API) shape."""
        return {
            "image": str(self.image_path),
            "fen": self.fen,
            "lichess_url": self.lichess_url,
        }


class Predictor:
    """A recognizer plus its eval transform: load the checkpoint once, predict many.

    `resolution` must match the checkpoint's training resolution (see
    `PAPER_RESOLUTION`); it is not stored in the checkpoint.
    """

    def __init__(self, model: BoardRecognizer, *, resolution: int, device: str = "cpu") -> None:
        self.model = model
        self.resolution = resolution
        self.device = device
        self._transform = build_transform(resolution, train=False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path | str,
        *,
        resolution: int = PAPER_RESOLUTION,
        device: str | None = None,
    ) -> "Predictor":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = Path(checkpoint)
        if checkpoint.suffix == ".safetensors":
            from chessqueries.models.checkpoint import load_safetensors_model

            model = load_safetensors_model(checkpoint, device=device)
        else:
            from chessqueries.train.lit import LitChessQueriesModel

            lit = LitChessQueriesModel.load_from_checkpoint(str(checkpoint), map_location=device)
            model = lit.model.eval().to(device)
        return cls(model, resolution=resolution, device=device)

    @torch.no_grad()
    def predict(self, paths: Sequence[Path | str], *, batch_size: int = 8) -> list[Prediction]:
        """Predict a board per image, in the order the paths were given."""
        paths = [Path(p) for p in paths]
        if not paths:
            return []
        out: list[Prediction] = []
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            # Transform per image before stacking: arbitrary photos differ in size,
            # and the resize to a square `resolution` is what makes them stackable.
            batch = torch.stack([self._load(p) for p in chunk]).to(self.device)
            labels = self.model.predict_labels(batch).cpu()
            out.extend(
                Prediction(image_path=p, board=Board.from_tensor(labels[i]))
                for i, p in enumerate(chunk)
            )
        return out

    def _load(self, path: Path) -> torch.Tensor:
        # RGB mode drops alpha and expands grayscale: uploads are not dataset images.
        return self._transform(read_image(str(path), mode=ImageReadMode.RGB).float())


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="chessqueries-predict",
        description="Read chessboards off photos: image(s) -> FEN.",
    )
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="custom checkpoint (default: download/cache the released weights)",
    )
    p.add_argument("--resolution", type=int, default=PAPER_RESOLUTION,
                   help="must match the checkpoint's training resolution")
    p.add_argument("--device", default=None, help="default: cuda when available")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--json", action="store_true", help="emit JSON records instead of TSV")
    p.add_argument("--viz", type=Path, default=None,
                   help="also write an input-vs-predicted-board figure to this path")
    args = p.parse_args(argv)

    predictor = Predictor.from_checkpoint(
        resolve_checkpoint(args.checkpoint),
        resolution=args.resolution,
        device=args.device,
    )
    preds = predictor.predict(args.images, batch_size=args.batch_size)

    if args.json:
        print(json.dumps([pred.to_record() for pred in preds], indent=2))
    else:
        for pred in preds:
            print(f"{pred.image_path}\t{pred.fen}")

    if args.viz:
        _write_figure(preds, args.viz)


def _write_figure(preds: Sequence[Prediction], out: Path) -> None:
    """Save the input-photo/predicted-board figure (viz extras only)."""
    try:
        from chessqueries.viz.render import BoardPanel, side_by_side_figure
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise SystemExit(
            f"--viz needs the viz extras ({exc}); install them: poetry install --with viz"
        ) from exc

    panels = [
        BoardPanel(image_path=pred.image_path, board=pred.board, title=pred.image_path.name)
        for pred in preds
    ]
    fig = side_by_side_figure(panels)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
