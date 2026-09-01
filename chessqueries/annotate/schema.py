"""The released annotation format: video id + timestamp + crop + FEN, plus a
provenance manifest. No pixels are shipped — `reconstruct` rebuilds frames from
these records (Kinetics-style).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from chessqueries.core import Board


class Source(str, Enum):
    """How a label's FEN was derived (provenance, and a filter for eval splits)."""

    CLOCK_OCR = "clock_ocr"  # nameplate + clock OCR matched to the relay (pipeline)
    MODEL_RETRIEVAL = "model_retrieval"  # board-only: recognizer prediction snapped to relay
    MANUAL = "manual"  # a human entered/located it directly


class Stage(str, Enum):
    """Where a file sits in the lifecycle."""

    CANDIDATES = "candidates"  # auto-produced, not yet human-verified (WIP)
    CROSSCHECKED = "crosschecked"  # model-vs-clock consensus run, triaged into buckets
    REVIEWED = "reviewed"  # human-confirmed (rejects removed, accepts/corrections kept)


class Bucket(str, Enum):
    """Crosscheck triage: how the independent visual model agreed with the clock match."""

    ACCEPT = "accept"  # fit + margin both pass -> two signals agree (lightweight skim only)
    REVIEW = "review"  # exactly one gate failed -> standard human review
    QUARANTINE = "quarantine"  # both gates failed -> bad frame (occlusion/wrong game); review last


@dataclass(frozen=True)
class CrossCheck:
    """The visual model's verdict on a clock-OCR candidate: which ply it picked from
    the candidate window, how far its prediction sat from that ply (``fit_diff``
    squares), and how decisively it beat the best *different* position (``margin``,
    log-prob; ``None`` when the only rivals are repetitions of the same placement)."""

    bucket: Bucket
    chosen_ply: int  # ply the model picked (may differ from the clock ply -> desync fix)
    clock_ply: int  # the nominal clock-OCR ply, before the model
    fit_diff: int  # squares the model's argmax board differs from the chosen position
    margin: float | None  # log-prob gap to the best different-placement candidate
    window: list[int]  # candidate plies the model scored
    repetition: bool = False  # a same-placement rival sat within the window (harmless tie)
    duplicate: bool = False  # an earlier accepted frame in this game already holds this FEN

    def __post_init__(self) -> None:
        if not isinstance(self.bucket, Bucket):
            object.__setattr__(self, "bucket", Bucket(self.bucket))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bucket"] = self.bucket.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CrossCheck":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


@dataclass(frozen=True)
class Annotation:
    """One labeled frame. ``crop_bbox`` is the template's board region in full-frame
    pixels; with the pinned format it reproduces the exact crop. ``source`` records
    how the FEN was derived; ``verified_by_human`` marks it confirmed."""

    video_id: str
    frame_index: int
    timestamp_s: float
    template_id: str
    crop_bbox: list[int]  # [x, y, w, h]
    game_id: str
    round_id: str  # the relay round this frame was attributed to (chronology key)
    ply: int
    fen: str  # full FEN
    placement: str  # FEN placement field
    side_to_move: str  # "w" or "b"
    white: str
    black: str
    white_clk_s: int | None
    black_clk_s: int | None
    confidence: float
    source: Source = Source.CLOCK_OCR
    requires_review: bool = False  # HARD-angle / chronology-flagged -> manual curation
    verified_by_human: bool = False
    crosscheck: CrossCheck | None = None  # set by the visual cross-check pass

    def __post_init__(self) -> None:
        if not isinstance(self.source, Source):
            object.__setattr__(self, "source", Source(self.source))
        if self.crosscheck is not None and not isinstance(self.crosscheck, CrossCheck):
            object.__setattr__(self, "crosscheck", CrossCheck.from_dict(self.crosscheck))
        if len(self.crop_bbox) != 4:
            raise ValueError(f"crop_bbox must be [x,y,w,h], got {self.crop_bbox}")
        Board.from_fen(self.placement)  # full structural validation (8 ranks x 8 squares)
        if self.side_to_move not in ("w", "b"):
            raise ValueError(f"side_to_move must be 'w'/'b', got {self.side_to_move!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source"] = self.source.value
        d["crosscheck"] = self.crosscheck.to_dict() if self.crosscheck else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        fields = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in d.items() if k in fields}
        if isinstance(kw.get("crosscheck"), dict):
            kw["crosscheck"] = CrossCheck.from_dict(kw["crosscheck"])
        return cls(**kw)


@dataclass
class AnnotationFile:
    """The full release artifact: a provenance manifest + the annotation records."""

    provenance: dict
    annotations: list[Annotation] = field(default_factory=list)

    def save(self, path: Path) -> None:
        # Atomic write: serialize to a temp sibling, then rename over the target. A crash
        # mid-write leaves the previous file intact instead of a truncated one — matters
        # for the review app, which autosaves after every decision.
        path = Path(path)
        payload = json.dumps(
            {
                "provenance": self.provenance,
                "annotations": [a.to_dict() for a in self.annotations],
            },
            indent=2,
        )
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(payload)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "AnnotationFile":
        d = json.loads(Path(path).read_text())
        return cls(
            provenance=d["provenance"],
            annotations=[Annotation.from_dict(a) for a in d["annotations"]],
        )

    def by_game(self) -> dict[str, list[Annotation]]:
        out: dict[str, list[Annotation]] = {}
        for a in self.annotations:
            out.setdefault(a.game_id, []).append(a)
        return out
