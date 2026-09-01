"""SLCC broadcast dataset: human-reviewed frames reconstructed from relay videos.

Manifest-driven and append-only. The on-disk manifest
(``data/slcc/dataset/annotations.json``) holds one record per frame, each
carrying its own ``split``. Splits are assigned by *group* (a game, by default)
so frames sharing a game — near-duplicate positions, the same set/lighting —
never straddle train and test (see :func:`plan_splits`). Growth is additive:
re-running the builder with more reviewed videos keeps every existing group's
split fixed and partitions only the *new* groups by a :class:`SplitRatio`
(e.g. ``0/0/100`` to add a video straight to TEST), so existing splits never
silently reshuffle.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from chessqueries.config import get_config
from chessqueries.core import Board, Split
from chessqueries.data.base import BoardSample, ChessDataset, DatasetName
from chessqueries.data.inventory import FROZEN_SAMPLE_COUNTS

MANIFEST_NAME = "annotations.json"

# Sample fields copied straight onto BoardSample.meta (provenance for review/eval).
_META_KEYS = (
    "video_id",
    "frame_index",
    "timestamp_s",
    "game_id",
    "round_id",
    "ply",
    "side_to_move",
    "players",
    "template_id",
    "crop_bbox",
    "confidence",
    "source",
    "requires_review",
    "verified_by_human",
)


@dataclass(frozen=True)
class SplitRatio:
    """Percentage-point split of *new* frames into train/val/test (must sum to 100)."""

    train: int
    val: int
    test: int

    def __post_init__(self) -> None:
        if any(p < 0 for p in (self.train, self.val, self.test)):
            raise ValueError(f"split percentages must be non-negative: {self}")
        if self.train + self.val + self.test != 100:
            raise ValueError(
                f"split percentages must sum to 100, got {self} = "
                f"{self.train + self.val + self.test}"
            )

    @classmethod
    def from_str(cls, spec: str) -> "SplitRatio":
        """Parse ``"60/30/10"`` -> SplitRatio(60, 30, 10)."""
        parts = spec.split("/")
        if len(parts) != 3 or not all(p.strip().lstrip("-").isdigit() for p in parts):
            raise ValueError(f"split ratio must be 'train/val/test' integers, got {spec!r}")
        return cls(*(int(p) for p in parts))


def plan_splits(
    groups: dict[str, str],
    *,
    ratio: SplitRatio,
    seed: int,
    existing: dict[str, str] | None = None,
) -> dict[str, str]:
    """Assign each sample to a split, keeping every *group* whole (no leakage).

    ``groups`` maps each sample id to its group key (e.g. ``game_id``): all frames
    sharing a key land in the same split, so near-duplicate / same-game frames
    never straddle train and test. Only groups with no ``existing`` assignment (the
    *new* frames) are partitioned; a group already represented in ``existing`` keeps
    that split for its new frames too. Keeping prior groups fixed is what makes
    dataset growth append-only — e.g. ``0/0/100`` sends a freshly-added video
    entirely to TEST without touching what's already there.

    Fresh groups are allocated by *size-stratified* apportionment: sorted by size
    and walked small->large, each handed to whichever split is furthest below its
    running target share. Walking in size order spreads small/medium/large groups
    evenly across splits instead of letting one split hoard the big games — e.g. 15
    games each of 3/8/50 frames at 34/33/33 put 5 games of every size in each split.
    ``ratio`` is the share of *groups* (≈ the frame share when groups are similarly
    sized; with few or wildly-uneven groups the frame split drifts — the reconstruct
    summary prints the realized counts). The ``seed`` shuffles equal-sized groups, so
    the same data + seed always yields the same split.
    """
    out = dict(existing or {})
    members: dict[str, list[str]] = {}
    for sid, key in groups.items():
        members.setdefault(key, []).append(sid)

    # A group is pinned if any of its samples already has a split; its new samples
    # inherit that split (majority wins if an existing manifest has a group straddling splits).
    pinned: dict[str, str] = {}
    for key, sids in members.items():
        prior = [out[s] for s in sids if s in out]
        if prior:
            pinned[key] = Counter(prior).most_common(1)[0][0]
    for key, split in pinned.items():
        for sid in members[key]:
            out.setdefault(sid, split)

    # Sort small->large (seed breaks ties among equal sizes) and apportion each group
    # to the split furthest below its running target share. Walking in size order is
    # what stratifies: a split's picks land evenly across the size range, so no split
    # ends up with all the big (or all the small) games.
    fresh = [k for k in members if k not in pinned]
    jitter = random.Random(seed)
    fresh.sort(key=lambda k: (len(members[k]), jitter.random()))
    pct = {Split.TRAIN.value: ratio.train, Split.VAL.value: ratio.val, Split.TEST.value: ratio.test}
    order = [Split.TRAIN.value, Split.VAL.value, Split.TEST.value]
    have = {s: 0 for s in order}
    for n, key in enumerate(fresh, start=1):
        split = max(
            order, key=lambda s: pct[s] * n / 100 - have[s]
        )  # ratio-deficit; ties->earliest
        have[split] += 1
        for sid in members[key]:
            out[sid] = split
    return out


class SLCC(ChessDataset):
    name = DatasetName.SLCC
    splits = (Split.TRAIN, Split.VAL, Split.TEST)
    expected_samples = FROZEN_SAMPLE_COUNTS[name]

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else get_config().DATA_ROOT / "slcc" / "dataset"
        self.manifest_path = self.root / MANIFEST_NAME

    def _records(self) -> list[dict]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"SLCC manifest not found at {self.manifest_path}. "
                "Build it with `python -m chessqueries.annotate.reconstruct`."
            )
        return json.loads(self.manifest_path.read_text())["samples"]

    def _structural_issues(self) -> tuple[str, ...]:
        payload = json.loads(self.manifest_path.read_text())
        records = payload["samples"]
        issues: list[str] = []
        if payload.get("provenance", {}).get("partial") is True:
            issues.append("manifest provenance marks this reconstruction partial")

        sample_ids: list[str] = []
        image_paths: list[str] = []
        game_splits: dict[str, set[str]] = defaultdict(set)
        valid_splits = {split.value for split in self.splits}
        for index, record in enumerate(records):
            image = record.get("image")
            explicit_id = record.get("sample_id")
            sample_id = (
                explicit_id
                if isinstance(explicit_id, str) and explicit_id
                else Path(image).stem if isinstance(image, str) and image else ""
            )
            if not sample_id:
                issues.append(f"record {index} has no sample identity")
            else:
                sample_ids.append(sample_id)

            if not isinstance(image, str) or not image:
                issues.append(f"record {index} has no image path")
            else:
                image_paths.append(image)
                expected_image = f"images/{sample_id}.jpg"
                if not sample_id or image != expected_image:
                    issues.append(
                        f"record {index} image path is {image!r}, expected {expected_image!r}"
                    )

            split = record.get("split")
            if split not in valid_splits:
                issues.append(f"record {index} has invalid split {split!r}")
            game_id = record.get("game_id")
            if not isinstance(game_id, str) or not game_id:
                issues.append(f"record {index} has no game_id")
            elif split in valid_splits:
                game_splits[game_id].add(split)

        duplicate_ids = sorted(
            sample_id for sample_id, count in Counter(sample_ids).items() if count > 1
        )
        if duplicate_ids:
            issues.append(f"duplicate sample IDs: {duplicate_ids[:5]}")
        duplicate_images = sorted(
            image for image, count in Counter(image_paths).items() if count > 1
        )
        if duplicate_images:
            issues.append(f"duplicate image paths: {duplicate_images[:5]}")
        straddling_games = sorted(
            game_id for game_id, splits in game_splits.items() if len(splits) > 1
        )
        if straddling_games:
            issues.append(f"game IDs cross splits: {straddling_games[:5]}")

        actual = Counter(record.get("split") for record in records)
        expected = {split.value: count for split, count in self.expected_samples.items()}
        if dict(actual) != expected:
            issues.append(f"full split counts are {dict(actual)}, expected {expected}")
        missing = sum(
            not (self.root / image).is_file()
            for image in image_paths
            if image == f"images/{Path(image).stem}.jpg"
        )
        if missing:
            issues.append(f"{missing} crop image(s) are missing across the full reconstruction")
        return tuple(issues)

    def _load_samples(self, split: Split | None) -> list[BoardSample]:
        samples: list[BoardSample] = []
        for rec in self._records():
            if split is not None and rec["split"] != split.value:
                continue
            img_path = self.root / rec["image"]
            sample_id = rec.get("sample_id") or Path(rec["image"]).stem
            samples.append(
                BoardSample(
                    image_path=img_path,
                    board=Board.from_fen(rec["gt_fen"]),
                    dataset=DatasetName.SLCC,
                    sample_id=sample_id,
                    split=Split(rec["split"]),
                    meta={"gt_fen": rec["gt_fen"], **{k: rec[k] for k in _META_KEYS if k in rec}},
                )
            )
        return samples
