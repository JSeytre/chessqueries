"""ChessCog dataset: synthetic rendered boards with per-image FEN annotations."""
from __future__ import annotations

import json
from pathlib import Path

from chessqueries.config import get_config
from chessqueries.core import Board, Split
from chessqueries.data.base import ChessDataset, DatasetName, BoardSample
from chessqueries.data.inventory import FROZEN_SAMPLE_COUNTS


class ChessCog(ChessDataset):
    name = DatasetName.CHESSCOG
    splits = (Split.TRAIN, Split.VAL, Split.TEST)
    expected_samples = FROZEN_SAMPLE_COUNTS[name]

    def __init__(self, dataroot: Path | None = None) -> None:
        self.dataroot = Path(dataroot) if dataroot else get_config().chesscog_root

    def _load_samples(self, split: Split | None) -> list[BoardSample]:
        split_dir = self.dataroot / "render" / split.value
        samples: list[BoardSample] = []
        for json_path in sorted(split_dir.glob("*.json")):
            with open(json_path) as f:
                ann = json.load(f)
            img_path = json_path.with_suffix(".png")
            samples.append(
                BoardSample(
                    image_path=img_path,
                    board=Board.from_fen(ann["fen"]),
                    dataset=DatasetName.CHESSCOG,
                    sample_id=json_path.stem,
                    split=split,
                )
            )
        return samples
