"""Reconcile per-frame identifications within a shot into trustworthy labels.

A single frame's OCR can misread a digit or a name. Within one shot the camera
stays on one board, so the fix is consensus: sample several frames, identify each,
then keep only labels that agree on a single game and whose ply advances
monotonically in time. Lone disagreements are dropped, not silently trusted.

`reconcile_shot` is pure (testable); `label_shot` is the thin layer that pulls
frames, OCRs the template's regions, and identifies.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from chessqueries.annotate.identify import (
    Identification,
    Nameplates,
    OcrReader,
    identify_position,
)
from chessqueries.annotate.relay import GameTimeline
from chessqueries.annotate.templates import Layout, Shot
from chessqueries.annotate.video import FrameReader, VideoFile


@dataclass(frozen=True)
class FrameObservation:
    frame_index: int
    timestamp_s: float
    identification: Identification | None


@dataclass(frozen=True)
class FrameLabel:
    frame_index: int
    timestamp_s: float
    game_index: int
    ply: int
    fen: str
    placement: str
    white: str
    black: str
    confidence: float


def reconcile_shot(
    observations: list[FrameObservation],
    *,
    min_game_samples: int = 3,
    max_ply_regression: int = 2,
) -> list[FrameLabel]:
    """Emit one label per distinct position, for *every* game shown in the shot.

    The director can swing the featured board mid-shot (no scene cut), so a shot may
    contain more than one game — we keep them all rather than forcing a single
    dominant game. A game is kept only if at least ``min_game_samples`` samples
    identified it (so isolated 1-2 frame OCR misreads are dropped). Within each game
    we drop backward ply jumps and keep the highest-confidence frame per distinct ply.
    """
    ided = [o for o in observations if o.identification is not None]
    if not ided:
        return []

    by_game: dict[int, list[FrameObservation]] = {}
    for o in ided:
        by_game.setdefault(o.identification.game_index, []).append(o)

    labels: list[FrameLabel] = []
    for game, obs_list in by_game.items():
        if len(obs_list) < min_game_samples:
            continue  # too little support -> likely a misread, not a real board switch
        best_per_ply: dict[int, FrameObservation] = {}
        max_ply = -1
        for o in sorted(obs_list, key=lambda o: o.timestamp_s):
            ply = o.identification.ply
            if ply < max_ply - max_ply_regression:
                continue
            max_ply = max(max_ply, ply)
            cur = best_per_ply.get(ply)
            if cur is None or o.identification.confidence > cur.identification.confidence:
                best_per_ply[ply] = o
        labels += [
            FrameLabel(
                frame_index=o.frame_index,
                timestamp_s=o.timestamp_s,
                game_index=o.identification.game_index,
                ply=o.identification.ply,
                fen=o.identification.fen,
                placement=o.identification.placement,
                white=o.identification.white,
                black=o.identification.black,
                confidence=o.identification.confidence,
            )
            for o in best_per_ply.values()
        ]
    return sorted(labels, key=lambda lab: (lab.timestamp_s, lab.game_index))


def _sample_indices(shot: Shot, fps: float, *, interval_s: float, min_samples: int) -> list[int]:
    """Evenly-spaced frame indices across the shot: one every ``interval_s`` seconds,
    but never fewer than ``min_samples`` (short shots just sample closer together)."""
    span = shot.end_frame - shot.start_frame
    n = max(min_samples, ceil((span / fps) / interval_s))
    n = min(n, span)  # can't sample more frames than the shot has
    if n <= 1:
        return [shot.keyframe_index]
    step = span / n
    return [shot.start_frame + int(step * (k + 0.5)) for k in range(n)]


def _read_clocks(ocr: OcrReader, frame, layout: Layout) -> list[int]:
    """Both clock values from the broadcast digital overlay, trusted only when reliable.

    The overlay can desync or not-yet-load and show 0 for a player (``0:0``, both at
    zero, is the clearest tell). Such a reading would match a bogus ply, so we never
    trust it: we return *no* clock, leaving the frame unidentified for the review/model
    pass rather than fabricating a match from a stale overlay.
    """
    digital = (
        ocr.clocks_in(layout.digital_clock_rect.crop(frame)) if layout.digital_clock_rect else []
    )
    if len(digital) >= 2 and 0 not in digital[:2]:
        return digital
    return []  # no usable overlay clock -> don't match this frame


def _read_names(ocr: OcrReader, frame, layout: Layout) -> Nameplates:
    """Text from the separate white/black nameplate regions, kept split by region so
    White/Black orientation is preserved (which disambiguates a reversed-colour rematch)."""
    white = ocr.texts(layout.white_name_rect.crop(frame)) if layout.white_name_rect else []
    black = ocr.texts(layout.black_name_rect.crop(frame)) if layout.black_name_rect else []
    return Nameplates(tuple(white), tuple(black))


def label_shot(
    shot: Shot,
    layout: Layout,
    reader: FrameReader,
    ocr: OcrReader,
    timelines: list[GameTimeline],
    *,
    sample_interval_s: float = 0.5,
    min_samples: int = 5,
    min_game_samples: int = 3,
) -> list[FrameLabel]:
    """Sample a processable shot (every ``sample_interval_s`` s, but at least
    ``min_samples`` frames even for short shots), identify each frame, and reconcile
    into one label per distinct position — for every game the shot shows."""
    if not layout.processable or layout.board_rect is None:
        return []
    video: VideoFile = reader.video
    observations: list[FrameObservation] = []
    for index in _sample_indices(
        shot, video.fps, interval_s=sample_interval_s, min_samples=min_samples
    ):
        frame = reader.frame_at_index(index)
        clocks = _read_clocks(ocr, frame, layout)
        ident = None
        if len(clocks) >= 2:  # only OCR names once we have a usable clock pair
            names = _read_names(ocr, frame, layout)
            ident = identify_position(timelines, (clocks[0], clocks[1]), names or None)
        observations.append(FrameObservation(index, index / video.fps, ident))
    return reconcile_shot(observations, min_game_samples=min_game_samples)
