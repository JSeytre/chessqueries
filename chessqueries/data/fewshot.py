"""Support-set selection for few-shot domain adaptation."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from chessqueries.core import Split
from chessqueries.data.base import (
    DATASET_REGISTRY,
    BoardSample,
    DatasetCompleteness,
    DatasetName,
    get_dataset,
)

FULL_SPLIT = 0  # --k sentinel: adapt on the whole train split


def adaptation_targets() -> tuple[DatasetName, ...]:
    """Datasets we can adapt *to*: one needs a TRAIN split to draw the support set
    from and a VAL split to select the checkpoint on. CVChess has neither — it is a
    single game, so it stays an eval-only domain."""
    return tuple(name for name, cls in DATASET_REGISTRY.items()
                 if {Split.TRAIN, Split.VAL} <= set(cls.splits))


@dataclass(frozen=True)
class FewShotSplit:
    """A k-image support set plus the held-out evaluation splits."""

    train: list[BoardSample]
    val: list[BoardSample]
    test: list[BoardSample]
    completeness: dict[Split, DatasetCompleteness] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.train:
            raise ValueError("support set is empty")
        if not self.test:
            raise ValueError("evaluation set is empty")
        ids = {"support": {s.sample_id for s in self.train},
               "val": {s.sample_id for s in self.val},
               "test": {s.sample_id for s in self.test}}
        for a, b in (("support", "test"), ("support", "val"), ("val", "test")):
            overlap = ids[a] & ids[b]
            if overlap:
                raise ValueError(f"{len(overlap)} sample(s) in both {a} and {b} sets")

    @property
    def train_ids(self) -> list[str]:
        return [s.sample_id for s in self.train]


def _subsample(samples: list[BoardSample], k: int, seed: int) -> list[BoardSample]:
    if k == FULL_SPLIT or k >= len(samples):
        return samples
    idx = sorted(random.Random(seed).sample(range(len(samples)), k))
    return [samples[i] for i in idx]


def load_support(
    name: DatasetName, k: int, seed: int, *, allow_partial: bool = False
) -> FewShotSplit:
    """Load `name` and reduce its TRAIN split to k images chosen at random
    (`FULL_SPLIT` = keep all); VAL and TEST are the official splits, untouched."""
    if k < 0:
        raise ValueError(f"k must be >= 0 ({FULL_SPLIT} = full split), got {k}")
    if name not in adaptation_targets():
        raise ValueError(
            f"{name.value} has no train/val split, so it cannot be an adaptation "
            f"target; adaptable datasets are "
            f"{[n.value for n in adaptation_targets()]}"
        )
    dataset = get_dataset(name)
    loaded = {
        split: dataset.load_with_report(split, allow_partial=allow_partial)
        for split in (Split.TRAIN, Split.VAL, Split.TEST)
    }
    return FewShotSplit(
        train=_subsample(loaded[Split.TRAIN].samples, k, seed),
        val=loaded[Split.VAL].samples,
        test=loaded[Split.TEST].samples,
        completeness={split: result.completeness for split, result in loaded.items()},
    )
