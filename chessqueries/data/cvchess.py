"""CVChess dataset: images plus FENs recovered from the vendored labels."""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from chessqueries.config import get_config
from chessqueries.core import Board, Split
from chessqueries.data.base import ChessDataset, DatasetName, BoardSample
from chessqueries.data.inventory import FROZEN_SAMPLE_COUNTS

VENDORED_LABELS = Path(__file__).parent / "resources" / "cvchess_labels.json"
VENDORED_VIEWPOINTS = Path(__file__).parent / "resources" / "cvchess_viewpoints.json"


class Viewpoint(str, Enum):
    """Where the WHITE player sits in the photo -- a single cell of a 3x3 spatial
    grid, so board corners are valid answers. ``BOTTOM`` is the standard
    white-player view; the others see the board obliquely or from Black's side."""

    TOP_LEFT = "top_left"
    TOP = "top"
    TOP_RIGHT = "top_right"
    LEFT = "left"
    RIGHT = "right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM = "bottom"
    BOTTOM_RIGHT = "bottom_right"


def _labels_path(cvchess_root: Path) -> Path:
    runtime = cvchess_root / "annotations.json"
    return runtime if runtime.is_file() else VENDORED_LABELS


def _load_labels(path: Path = VENDORED_LABELS) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _load_viewpoints(path: Path = VENDORED_VIEWPOINTS) -> dict[str, Viewpoint | None]:
    """sample_id -> camera viewpoint (where the white player sits), if labelled."""
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text()).get("white_side", {})
    return {sid: Viewpoint(v) if v else None for sid, v in raw.items()}


class CVChess(ChessDataset):
    name = DatasetName.CVCHESS
    splits = ()  # no official split
    expected_samples = FROZEN_SAMPLE_COUNTS[name]

    def __init__(self, images_dir: Path | None = None) -> None:
        self.cvchess_root = get_config().cvchess_root
        self.images_dir = Path(images_dir) if images_dir else self.cvchess_root / "images"

    def _load_samples(self, split: Split | None) -> list[BoardSample]:
        labels = _load_labels(_labels_path(self.cvchess_root))
        viewpoints = _load_viewpoints()
        samples: list[BoardSample] = []
        for entry in labels:
            img_path = self.images_dir / entry["image"]
            sample_id = Path(entry["image"]).stem
            samples.append(
                BoardSample(
                    image_path=img_path,
                    board=Board.from_fen(entry["gt_fen"]),
                    dataset=DatasetName.CVCHESS,
                    sample_id=sample_id,
                    meta={"gt_fen": entry["gt_fen"], "white_side": viewpoints.get(sample_id)},
                )
            )
        return samples
