"""ChessReD dataset (4TU.ResearchData): real smartphone photos."""
from __future__ import annotations

import json
from pathlib import Path

from chessqueries.config import get_config
from chessqueries.core import Board, Piece, Split, Square
from chessqueries.data.base import ChessDataset, DatasetName, BoardSample
from chessqueries.data.inventory import FROZEN_SAMPLE_COUNTS

ANNOTATIONS_FILE = "annotations.json"
# We ship one set of images: the authors' offline-resized 1024x1024 set — the
# exact inputs the released checkpoint was trained on, and what reproduces the
# paper (the raw 3072x3072 images + a runtime `Resize` do not; see
# `models/chessred_resnext.py` and upstream issue
# tmasouris/end-to-end-chess-recognition#5).
IMAGE_SIZE = 1024

# ChessReD category names -> Piece.
CATEGORY_TO_PIECE: dict[str, Piece] = {
    "empty": Piece.EMPTY,
    "white-pawn": Piece.WHITE_PAWN, "white-knight": Piece.WHITE_KNIGHT,
    "white-bishop": Piece.WHITE_BISHOP, "white-rook": Piece.WHITE_ROOK,
    "white-queen": Piece.WHITE_QUEEN, "white-king": Piece.WHITE_KING,
    "black-pawn": Piece.BLACK_PAWN, "black-knight": Piece.BLACK_KNIGHT,
    "black-bishop": Piece.BLACK_BISHOP, "black-rook": Piece.BLACK_ROOK,
    "black-queen": Piece.BLACK_QUEEN, "black-king": Piece.BLACK_KING,
}


def load_annotations(dataroot: Path) -> dict:
    path = dataroot / ANNOTATIONS_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Download ChessReD first:\n"
            f"    poetry run python scripts/download_data.py chessred"
        )
    with open(path) as f:
        return json.load(f)


def category_names_in_id_order(dataroot: Path) -> list[str]:
    """Category names sorted by id — i.e. the order of the released model's 13
    output channels (needed to remap them to our `Piece` order)."""
    cats = load_annotations(dataroot)["categories"]
    return [c["name"] for c in sorted(cats, key=lambda c: c["id"])]


class ChessReD(ChessDataset):
    name = DatasetName.CHESSRED
    splits = (Split.TRAIN, Split.VAL, Split.TEST)
    expected_samples = FROZEN_SAMPLE_COUNTS[name]

    def __init__(self, dataroot: Path | None = None) -> None:
        self.dataroot = Path(dataroot) if dataroot else get_config().chessred_root

    def _load_samples(self, split: Split | None) -> list[BoardSample]:
        ann = load_annotations(self.dataroot)
        split_ids = set(ann["splits"][split.value]["image_ids"])
        cat_name = {c["id"]: c["name"] for c in ann["categories"]}
        img_path = {im["id"]: im["path"] for im in ann["images"]}

        by_image: dict[int, list[tuple[int, Piece]]] = {}
        for piece in ann["annotations"]["pieces"]:
            img_id = piece["image_id"]
            if img_id not in split_ids:
                continue
            p = CATEGORY_TO_PIECE[cat_name[piece["category_id"]]]
            if p.is_empty:
                continue
            idx = Square.from_name(piece["chessboard_position"]).index
            by_image.setdefault(img_id, []).append((idx, p))

        samples: list[BoardSample] = []
        for img_id in ann["splits"][split.value]["image_ids"]:
            board_pieces = [Piece.EMPTY] * 64
            for idx, p in by_image.get(img_id, []):
                board_pieces[idx] = p
            samples.append(
                BoardSample(
                    image_path=self.dataroot / img_path[img_id],
                    board=Board(tuple(board_pieces)),
                    dataset=DatasetName.CHESSRED,
                    sample_id=str(img_id),
                    split=split,
                )
            )
        return samples
