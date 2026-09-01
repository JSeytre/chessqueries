"""Dataset abstractions: the `BoardSample` schema, the `ChessDataset` ABC, and
the dataset registry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import ClassVar

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from chessqueries.core import Board, Split


class DatasetName(str, Enum):
    CHESSRED = "chessred"
    CHESSCOG = "chesscog"
    CVCHESS = "cvchess"
    SLCC = "slcc"


@dataclass
class BoardSample:
    image_path: Path
    board: Board
    dataset: DatasetName
    sample_id: str
    split: Split | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.image_path = Path(self.image_path)

    def __repr__(self) -> str:
        return (
            f"BoardSample(dataset={self.dataset.value!r}, id={self.sample_id!r}, "
            f"split={self.split}, fen={self.board.placement!r})"
        )


class DatasetIncompleteError(ValueError):
    """A labelled dataset does not match the paper evaluation inventory."""


@dataclass(frozen=True)
class DatasetCompleteness:
    """Expected, labelled, and locally available records for one dataset split."""

    dataset: DatasetName
    split: Split | None
    expected_samples: int
    labelled_samples: int
    available_samples: int
    missing_sample_ids: tuple[str, ...]
    allow_partial: bool
    structural_issues: tuple[str, ...] = ()

    @property
    def partial(self) -> bool:
        return (
            self.labelled_samples != self.expected_samples
            or self.available_samples != self.expected_samples
            or bool(self.structural_issues)
        )

    def as_dict(
        self,
        *,
        evaluated_samples: int | None = None,
        expected_evaluated_samples: int | None = None,
    ) -> dict:
        actual = self.available_samples if evaluated_samples is None else evaluated_samples
        expected_evaluation = (
            self.expected_samples
            if expected_evaluated_samples is None
            else expected_evaluated_samples
        )
        return {
            "dataset": self.dataset.value,
            "split": self.split.value if self.split is not None else None,
            "mode": "allow_partial" if self.allow_partial else "strict",
            "scope": (
                "full_split"
                if expected_evaluation == self.expected_samples
                else "explicit_subset"
            ),
            "data_complete": not self.partial,
            "expected_samples": self.expected_samples,
            "expected_evaluated_samples": expected_evaluation,
            "labelled_samples": self.labelled_samples,
            "available_samples": self.available_samples,
            "actual_samples": actual,
            "missing_images": len(self.missing_sample_ids),
            "structural_issues": list(self.structural_issues),
        }


@dataclass(frozen=True)
class DatasetLoad:
    """Validated samples together with the inventory used to admit them."""

    samples: list[BoardSample]
    completeness: DatasetCompleteness


class BoardImageDataset(Dataset):
    """torch Dataset over `BoardSample`s. Returns ``(image[C,H,W] float,
    labels[64] long)``; the raw `BoardSample` is at ``ds.samples[i]``.

    Every dataset reuses it via `ChessDataset.torch_dataset`.
    """

    def __init__(
        self,
        samples: list[BoardSample],
        transform=None,
        completeness: DatasetCompleteness | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.completeness = completeness

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        img = read_image(str(sample.image_path)).float()
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(sample.board.labels, dtype=torch.long)


DATASET_REGISTRY: dict[DatasetName, type["ChessDataset"]] = {}


class ChessDataset(ABC):
    name: ClassVar[DatasetName]
    splits: ClassVar[tuple[Split, ...]] = ()  # () = no official split
    expected_samples: ClassVar[dict[Split | None, int]]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", None) is not None:
            DATASET_REGISTRY[cls.name] = cls

    @abstractmethod
    def _load_samples(self, split: Split | None) -> list[BoardSample]:
        ...

    def _validate_split(self, split: Split | None) -> None:
        if self.splits and split not in self.splits:
            raise ValueError(f"{self.name.value}: split must be one of {[s.value for s in self.splits]}, got {split!r}")
        if not self.splits and split is not None:
            raise ValueError(f"{self.name.value} has no splits; pass split=None")

    def _structural_issues(self) -> tuple[str, ...]:
        return ()

    def load_with_report(
        self, split: Split | None = None, *, allow_partial: bool = False
    ) -> DatasetLoad:
        """Load a canonical split, refusing missing or unexpected records by default."""
        self._validate_split(split)
        labelled = self._load_samples(split)
        expected = self.expected_samples[split]

        seen: set[str] = set()
        duplicates: set[str] = set()
        for sample in labelled:
            if sample.sample_id in seen:
                duplicates.add(sample.sample_id)
            seen.add(sample.sample_id)
        if duplicates:
            raise ValueError(
                f"{self.name.value}: duplicate sample IDs: {sorted(duplicates)[:5]}"
            )

        available = [sample for sample in labelled if sample.image_path.is_file()]
        missing = tuple(
            sample.sample_id for sample in labelled if not sample.image_path.is_file()
        )
        completeness = DatasetCompleteness(
            dataset=self.name,
            split=split,
            expected_samples=expected,
            labelled_samples=len(labelled),
            available_samples=len(available),
            missing_sample_ids=missing,
            allow_partial=allow_partial,
            structural_issues=self._structural_issues(),
        )
        split_name = split.value if split else "all"
        detail = (
            f"expected {expected}, found {len(labelled)} labelled and "
            f"{len(available)} available sample(s)"
        )
        if not available:
            raise DatasetIncompleteError(
                f"{self.name.value}[{split_name}] has no available images ({detail})"
            )
        if completeness.partial and not allow_partial:
            examples = f"; missing IDs include {list(missing[:5])}" if missing else ""
            issues = (
                f"; structural issues: {list(completeness.structural_issues)}"
                if completeness.structural_issues
                else ""
            )
            raise DatasetIncompleteError(
                f"{self.name.value}[{split_name}] is incomplete: "
                f"{detail}{examples}{issues}. "
                "Pass allow_partial=True only for a diagnostic subset."
            )
        if completeness.partial:
            print(
                f"WARNING [allow_partial] {self.name.value}[{split_name}]: {detail}; "
                "paper-comparable evaluation is disabled"
            )
        return DatasetLoad(available, completeness)

    def load_samples(
        self, split: Split | None = None, *, allow_partial: bool = False
    ) -> list[BoardSample]:
        return self.load_with_report(split, allow_partial=allow_partial).samples

    def torch_dataset(
        self,
        split: Split | None = None,
        transform=None,
        *,
        allow_partial: bool = False,
    ) -> BoardImageDataset:
        loaded = self.load_with_report(split, allow_partial=allow_partial)
        return BoardImageDataset(loaded.samples, transform, loaded.completeness)


def get_dataset(name: DatasetName) -> ChessDataset:
    return DATASET_REGISTRY[name]()
