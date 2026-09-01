"""Self-contained data loading for the paper reproduction.

This module reads raw annotation files from the repository's ``data/`` tree;
it deliberately imports no ``chessqueries`` code. Labels are 64 integers in
FEN square order (a8=0, ..., h1=63), with class ids in ``.PNBRQKpnbrqk`` order.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(
    os.environ.get("CHESSQUERIES_DATA_ROOT", REPOSITORY_ROOT / "data")
).expanduser()

# Immutable identities of the annotation files used by the paper experiments.
SLCC_MANIFEST_SHA256 = "057f247ae92b134ca2b172317335919df01b22cfaa7472ddaf53393c2515ab75"
CHESSRED_ANNOTATIONS_SHA256 = (
    "16e99d7e8535c0fc56507caa2aa5f7594d7ca076ebab3f5a432cfd4aa10668cc"
)
CVCHESS_ANNOTATIONS_SHA256 = (
    "e1ef7404d1a742828f6970e963a1fc929f34a46eff80ef47a8c1afc903b90e64"
)

PIECE_SYMBOLS = ".PNBRQKpnbrqk"  # index == class id
NUM_CLASSES = 13

CHESSRED_CATEGORY_TO_CLASS = {
    "white-pawn": 1,
    "white-knight": 2,
    "white-bishop": 3,
    "white-rook": 4,
    "white-queen": 5,
    "white-king": 6,
    "black-pawn": 7,
    "black-knight": 8,
    "black-bishop": 9,
    "black-rook": 10,
    "black-queen": 11,
    "black-king": 12,
    "empty": 0,
}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    labels: tuple[int, ...]
    sample_id: str


def square_index(name: str) -> int:
    """Convert a square such as ``e4`` to its FEN-order index."""
    file = ord(name[0]) - ord("a")
    rank = int(name[1])
    assert 0 <= file < 8 and 1 <= rank <= 8, name
    return (8 - rank) * 8 + file


def fen_to_labels(fen: str) -> tuple[int, ...]:
    """Convert the placement field of a FEN to 64 validated class ids."""
    ranks = fen.split()[0].split("/")
    assert len(ranks) == 8, fen
    out: list[int] = []
    for rank in ranks:
        row: list[int] = []
        for char in rank:
            if char.isdigit():
                row += [0] * int(char)
            else:
                row.append(PIECE_SYMBOLS.index(char))
        assert len(row) == 8, fen
        out += row
    return tuple(out)


def load_chessred(split: str) -> list[Sample]:
    """Read an official ChessReD split directly from ``annotations.json``."""
    annotation_path = DATA_ROOT / "chessred" / "annotations.json"
    annotations = json.loads(annotation_path.read_text())
    split_ids = list(annotations["splits"][split]["image_ids"])
    image_paths = {image["id"]: image["path"] for image in annotations["images"]}
    category_names = {
        category["id"]: category["name"] for category in annotations["categories"]
    }
    boards: dict[int, list[int]] = {image_id: [0] * 64 for image_id in split_ids}
    in_split = set(split_ids)
    for piece in annotations["annotations"]["pieces"]:
        if piece["image_id"] not in in_split:
            continue
        class_id = CHESSRED_CATEGORY_TO_CLASS[category_names[piece["category_id"]]]
        if class_id:
            boards[piece["image_id"]][square_index(piece["chessboard_position"])] = class_id
    return [
        Sample(
            DATA_ROOT / "chessred" / image_paths[image_id],
            tuple(boards[image_id]),
            str(image_id),
        )
        for image_id in split_ids
    ]


def load_chesscog(split: str) -> list[Sample]:
    """Read ChessCog renders; its directory layout defines its splits."""
    split_dir = DATA_ROOT / "chesscog" / "render" / split
    samples = []
    for annotation_path in sorted(split_dir.glob("*.json")):
        fen = json.loads(annotation_path.read_text())["fen"]
        samples.append(
            Sample(
                annotation_path.with_suffix(".png"),
                fen_to_labels(fen),
                annotation_path.stem,
            )
        )
    return samples


def load_slcc(split: str) -> list[Sample]:
    """Read the frozen SLCC v1 manifest, whose records carry game-level splits."""
    root = DATA_ROOT / "slcc" / "dataset"
    manifest_path = root / "annotations.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"SLCC dataset manifest not found: {manifest_path}. A strict reconstruction "
            "only publishes this file after every required source succeeds. Run the "
            "--check-sources and reconstruction commands in "
            "minimal_reproduction/README.md, resolve any reported YouTube access error, "
            "and rerun reconstruction before training. Do not use --allow-partial for "
            "the paper reproduction."
        )
    manifest = json.loads(manifest_path.read_text())
    return [
        Sample(
            root / record["image"],
            fen_to_labels(record["gt_fen"]),
            Path(record["image"]).stem,
        )
        for record in manifest["samples"]
        if record["split"] == split
    ]


def load_cvchess() -> list[Sample]:
    """Read all 352 corrected CVChess labels; CVChess is evaluation-only."""
    entries = json.loads((DATA_ROOT / "cvchess" / "annotations.json").read_text())
    return [
        Sample(
            DATA_ROOT / "cvchess" / "images" / entry["image"],
            fen_to_labels(entry["gt_fen"]),
            Path(entry["image"]).stem,
        )
        for entry in entries
    ]


# This closed tuple is the complete training-source boundary. CVChess is absent.
TRAIN_SOURCES = (load_chessred, load_chesscog, load_slcc)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def eval_transform(resolution: int):
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image / 255.0),
            transforms.Resize((resolution, resolution), antialias=True),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def train_transform(resolution: int):
    """The paper's ``geometric_mild`` pipeline, before ImageNet normalization."""
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image / 255.0),
            transforms.Resize((resolution, resolution), antialias=True),
            transforms.RandomApply([transforms.RandomRotation(45, fill=0)], p=0.9),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.7, fill=0),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.85, 1.1),
                fill=0,
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class BoardDataset(Dataset):
    """Pairs of ``image[C,H,W]`` and ``labels[64]``, read with torchvision."""

    def __init__(self, samples: list[Sample], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self.transform(read_image(str(sample.image_path)).float())
        return image, torch.tensor(sample.labels, dtype=torch.long)
