"""Orchestrate the producer: video + relay timelines + labeled templates -> the
released `AnnotationFile`. Reports counts at every gate (shots -> useful -> labeled
frames) so nothing is silently dropped.
"""

from __future__ import annotations

import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chessqueries.annotate.align import FrameLabel, label_shot
from chessqueries.annotate.embedding import DESCRIPTOR_DIM, shot_descriptors
from chessqueries.annotate.identify import OcrReader, surname
from chessqueries.annotate.relay import LICHESS_ROUND_PGN_URL, GameTimeline
from chessqueries.annotate.schema import Annotation, AnnotationFile, Stage
from chessqueries.annotate.templates import (
    MIN_SIMILARITY,
    Layout,
    LayoutRegistry,
    Shot,
    load_shots,
    save_shots,
    segment_shots,
)
from chessqueries.annotate.validate import chronology_outliers
from chessqueries.annotate.video import FrameReader, VideoFile

# The shot-descriptor cache is representation-specific: the extension names the
# descriptor space, so a cache written in a different space lives under a different
# name and is simply ignored (survey sees the video as needing re-ingest) —
# incompatible descriptors are never silently mixed.
DESCRIPTORS_EXT = ".descriptors.dino256.npy"

# Shot labeling is decode + OCR, both single-core (PaddleOCR measured at a 1.0 CPU/wall
# ratio), so `produce` fans shots out over worker processes. 12 gives ~10x on a full
# broadcast while leaving most of a workstation free; tune with `produce --workers`.
DEFAULT_PRODUCE_WORKERS = 12


def game_id(round_id: str, timeline: GameTimeline) -> str:
    return f"{round_id}-{surname(timeline.white)}-{surname(timeline.black)}"


def timeline_hash(timelines: list[GameTimeline]) -> str:
    """Integrity digest of the relay (placement sequence of every game)."""
    h = hashlib.sha256()
    for t in timelines:
        for p in t.positions:
            h.update(p.placement.encode())
    return h.hexdigest()[:16]


def label_to_annotation(
    label: FrameLabel,
    video: VideoFile,
    crop_bbox: list[int],
    template_id: str,
    timeline: GameTimeline,
    round_id: str,
    requires_review: bool = False,
) -> Annotation:
    pos = timeline.position_at(label.ply)
    return Annotation(
        video_id=video.video_id,
        frame_index=label.frame_index,
        timestamp_s=label.timestamp_s,
        template_id=template_id,
        crop_bbox=list(crop_bbox),
        game_id=game_id(round_id, timeline),
        round_id=round_id,
        ply=label.ply,
        fen=label.fen,
        placement=label.placement,
        side_to_move=pos.turn.fen,
        white=timeline.white,
        black=timeline.black,
        white_clk_s=pos.white_clk_s,
        black_clk_s=pos.black_clk_s,
        confidence=label.confidence,
        requires_review=requires_review,
    )


@dataclass(frozen=True)
class ShotFingerprints:
    """A video's shot boundaries with the keyframe descriptor of each, row-aligned:
    ``descriptors[i]`` fingerprints ``shots[i]``."""

    shots: list[Shot]
    descriptors: np.ndarray

    def __post_init__(self) -> None:
        if len(self.shots) != len(self.descriptors):
            raise ValueError(f"{len(self.shots)} shots vs {len(self.descriptors)} descriptors")

    def pairs(self):
        """``(shot, descriptor)`` pairs — the way consumers walk a video. Deliberately
        not ``__iter__``: an unpack has to name the field it wants."""
        return zip(self.shots, self.descriptors)

    def __len__(self) -> int:
        return len(self.shots)


def segment_and_fingerprint(
    video: VideoFile,
    *,
    shots_cache: Path | None = None,
    descriptors_cache: Path | None = None,
    threshold: float = 27.0,
    frame_skip: int = 0,
    max_frames: int | None = None,
    log=print,
) -> ShotFingerprints:
    """Phase A: split the video into shots and fingerprint each keyframe, caching both.

    The full-video decode (the slow step) runs once; ``produce`` and ``status`` reuse
    the caches — segmentation is independent of the relay and the layout registry.
    """
    have_shots = bool(shots_cache and Path(shots_cache).exists())
    have_descriptors = bool(descriptors_cache and Path(descriptors_cache).exists())

    if have_shots and have_descriptors:
        descriptors = np.load(descriptors_cache)
        if descriptors.ndim == 2 and descriptors.shape[1] == DESCRIPTOR_DIM:
            shots = load_shots(shots_cache)
            log(f"shots: {len(shots)} (cached)")
            return ShotFingerprints(shots=shots, descriptors=descriptors)
        log(
            f"descriptor cache dim {descriptors.shape}; expected [*, {DESCRIPTOR_DIM}] — recomputing"
        )
        have_descriptors = False

    if have_shots:  # shots cached but descriptors missing -> fingerprint only (no re-segment)
        shots = load_shots(shots_cache)
        log(f"shots: {len(shots)} (cached); fingerprinting keyframes...")
    else:
        log("segmenting shots (full-video scan; cached for next time)...")
        shots = segment_shots(
            video,
            threshold=threshold,
            frame_skip=frame_skip,
            max_frames=max_frames,
            show_progress=True,
        )
        if shots_cache:
            save_shots(shots, shots_cache)

    descriptors = shot_descriptors(video, shots, show_progress=True)
    if descriptors_cache:
        np.save(descriptors_cache, descriptors)
    log(f"shots: {len(shots)}")
    return ShotFingerprints(shots=shots, descriptors=descriptors)


@dataclass(frozen=True)
class TriagedShot:
    """A shot that survived template triage and is headed for clock-OCR labeling."""

    index: int
    shot: Shot
    template_id: str
    layout: Layout


@dataclass(frozen=True)
class ShotTriage:
    """Outcome of classifying every shot against the registry: the shots to label,
    plus the counts for the gates that dropped the rest."""

    tasks: list[TriagedShot]
    n_unknown: int
    n_board_only: int

    @property
    def n_processable(self) -> int:
        return len(self.tasks)

    @property
    def n_hard(self) -> int:
        return sum(t.layout.requires_curation for t in self.tasks)


def triage_shots(
    registry: LayoutRegistry,
    shots: list[Shot],
    descriptors: np.ndarray,
    *,
    min_similarity: float = MIN_SIMILARITY,
) -> ShotTriage:
    """Classify each shot to its template and keep the clock-OCR-able ones.
    **Board-only** templates carry no clock/name to read and are left for the
    model-retrieval pass — counted as gaps, not labeled."""
    tasks: list[TriagedShot] = []
    n_unknown = n_board_only = 0
    for i, shot in enumerate(shots):
        if not registry.centroids:
            n_unknown += 1
            continue
        match = registry.classify(descriptors[i])
        if match.similarity < min_similarity:
            n_unknown += 1
            continue
        layout = registry.layouts[match.template_id]
        if not layout.processable:
            continue
        if layout.board_only:
            n_board_only += 1  # gap: needs the model-retrieval pass
            continue
        tasks.append(TriagedShot(index=i, shot=shot, template_id=match.template_id, layout=layout))
    return ShotTriage(tasks=tasks, n_unknown=n_unknown, n_board_only=n_board_only)


# Per-process state for the produce worker pool. A worker owns its own decoder and OCR
# engine (neither survives pickling, and paddle/FFmpeg thread state is not fork-safe,
# hence spawn + initializer) and reuses them for every shot it labels.
_shot_worker: dict = {}


def _init_shot_worker(
    video: VideoFile,
    timelines: list[GameTimeline],
    sample_interval_s: float,
    min_samples: int,
    min_game_samples: int,
) -> None:
    _shot_worker.update(
        reader=FrameReader(video),
        ocr=OcrReader(),
        timelines=timelines,
        sample_interval_s=sample_interval_s,
        min_samples=min_samples,
        min_game_samples=min_game_samples,
    )


def _label_shot_task(task: TriagedShot) -> list[FrameLabel]:
    w = _shot_worker
    return label_shot(
        task.shot,
        task.layout,
        w["reader"],
        w["ocr"],
        w["timelines"],
        sample_interval_s=w["sample_interval_s"],
        min_samples=w["min_samples"],
        min_game_samples=w["min_game_samples"],
    )


def produce_one(
    video: VideoFile,
    timelines: list[GameTimeline],
    ocr: OcrReader | None,
    registry: LayoutRegistry,
    shots: list[Shot],
    descriptors: np.ndarray,
    *,
    video_url: str,
    video_title: str = "",
    min_similarity: float = MIN_SIMILARITY,
    sample_interval_s: float = 0.5,
    min_samples: int = 5,
    min_game_samples: int = 3,
    workers: int = DEFAULT_PRODUCE_WORKERS,
    log=print,
) -> AnnotationFile:
    """Phase C: classify the (cached) shots against the registry and clock-OCR align
    the processable ones into an ``AnnotationFile`` (``stage=candidates``).

    ``timelines`` may span several rounds (rapid + blitz); each must carry its
    ``source_round_id``, so repeat pairings across rounds are separated by the clock,
    not the names.

    Shots are labeled by ``workers`` processes (decode + OCR are both single-core);
    results are reassembled in shot order, so the output is identical to a
    sequential run. With ``workers=1`` everything runs in-process using ``ocr``
    (created here if None)."""
    from tqdm import tqdm

    log(f"labeling from: {video_title or '(untitled)'} [{video.video_id}]")
    triage = triage_shots(registry, shots, descriptors, min_similarity=min_similarity)
    n_unknown, n_board_only = triage.n_unknown, triage.n_board_only
    n_processable, n_hard = triage.n_processable, triage.n_hard

    workers = max(1, min(workers, n_processable))
    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_shot_worker,
            initargs=(video, timelines, sample_interval_s, min_samples, min_game_samples),
        ) as pool:  # pool.map preserves task order -> deterministic reassembly below
            label_lists = list(
                tqdm(
                    pool.map(_label_shot_task, triage.tasks),
                    total=n_processable,
                    desc="shots",
                    unit="shot",
                )
            )
    else:
        if ocr is None:
            ocr = OcrReader()
        with FrameReader(video) as reader:
            label_lists = [
                label_shot(
                    task.shot,
                    task.layout,
                    reader,
                    ocr,
                    timelines,
                    sample_interval_s=sample_interval_s,
                    min_samples=min_samples,
                    min_game_samples=min_game_samples,
                )
                for task in tqdm(triage.tasks, desc="shots", unit="shot")
            ]

    annotations: list[Annotation] = []
    for task, labels in zip(triage.tasks, label_lists):
        for label in labels:
            tl = timelines[label.game_index]
            annotations.append(
                label_to_annotation(
                    label,
                    video,
                    task.layout.board_rect.as_list(),
                    task.template_id,
                    tl,
                    tl.source_round_id or "?",
                    requires_review=task.layout.requires_curation,
                )
            )

    # Chronology guard: rounds are sequential in the broadcast, so flag any frame
    # whose round disagrees with the round active at its timestamp (sends to review).
    outliers = set(chronology_outliers(annotations))
    if outliers:
        annotations = [
            (
                a
                if i not in outliers
                else Annotation.from_dict({**a.to_dict(), "requires_review": True})
            )
            for i, a in enumerate(annotations)
        ]
        log(f"chronology: {len(outliers)} frame(s) flagged for review (round/time mismatch)")

    n_review = sum(a.requires_review for a in annotations)
    n_games = len({a.game_id for a in annotations})
    round_ids = sorted({t.source_round_id for t in timelines if t.source_round_id})
    log(
        f"SUMMARY: {len(shots)} shots -> {n_processable} processable -> "
        f"{len(annotations)} positions across {n_games} game(s) / {len(round_ids)} round(s); "
        f"{n_review} flagged for review, {len(annotations) - n_review} auto-confident; "
        f"{n_board_only} board-only shot(s) await the model pass; {n_unknown} unknown-template"
    )
    provenance = {
        "stage": Stage.CANDIDATES.value,
        "video_id": video.video_id,
        "video_title": video_title,
        "video_url": video_url,
        "format_id": video.format_id,
        "width": video.width,
        "height": video.height,
        "fps": video.fps,
        "round_ids": round_ids,
        "pgn_sources": [LICHESS_ROUND_PGN_URL.format(round_id=r) for r in round_ids],
        "fen_timeline_hash": timeline_hash(timelines),
        "n_shots": len(shots),
        "n_processable_shots": n_processable,
        "n_hard_shots": n_hard,
        "n_board_only_shots": n_board_only,
        "n_unknown_template_shots": n_unknown,
        "n_frames": len(annotations),
        "n_frames_need_review": n_review,
    }
    return AnnotationFile(provenance=provenance, annotations=annotations)


def salvage_key(a: Annotation) -> tuple[str, str, int]:
    """Cross-run frame identity: ``(video_id, game_id, ply)``. Stable across the
    descriptor swap (``ply`` comes from clock-OCR, not the descriptor) and robust to
    frame-index drift, so a shot captured in both a prior and the current run is recognized
    as the same frame — not mistaken for a fresh salvage. ``frame_index`` stays the
    within-run review key; this is the *between*-run one."""
    return (a.video_id, a.game_id, a.ply)


def preseed_reviewed(candidates: AnnotationFile, verified_frames: set[int]) -> AnnotationFile:
    """A reviewed file mirroring ``candidates``, with the prior-verified frames marked
    confirmed and everything else left pending. Written next to the candidates so
    ``annotate review`` resumes straight onto the pending (newly-salvaged) frames:
    ``resume_decisions`` reads verified->accept (skipped) and pending->undecided (surfaced)."""
    anns = [
        Annotation.from_dict({**a.to_dict(), "verified_by_human": a.frame_index in verified_frames})
        for a in candidates.annotations
    ]
    prov = {**candidates.provenance, "stage": Stage.REVIEWED.value}
    return AnnotationFile(provenance=prov, annotations=anns)


@dataclass(frozen=True)
class SalvageMerge:
    """Both halves of a salvage re-produce, written together: the merged candidates
    substrate and the reviewed file pre-seeded with the verdicts carried over."""

    candidates: AnnotationFile
    reviewed: AnnotationFile


def merge_salvage(
    old_candidates: AnnotationFile,
    prior_reviewed: AnnotationFile,
    fresh: AnnotationFile,
    *,
    log=print,
) -> SalvageMerge:
    """Fold a re-produced candidate set into prior human decisions, surfacing only the
    frames the better matching newly rescued.

    The merged candidates file keeps every prior *verified* frame verbatim (original
    crop, human-corrected ply) plus genuinely-new frames. A fresh candidate is *new*
    iff its :func:`salvage_key`
    was absent from ``old_candidates`` (so a previously-captured shot — kept or rejected —
    never re-surfaces, and a frame-index drift doesn't fake a salvage). Verified frames are
    normalized to ``verified_by_human=False`` in the candidates substrate; the reviewed file
    carries the verdicts."""
    old_keys = {salvage_key(a) for a in old_candidates.annotations}
    verified = [a for a in prior_reviewed.annotations if a.verified_by_human]
    verified_frames = {a.frame_index for a in verified}

    merged = [Annotation.from_dict({**a.to_dict(), "verified_by_human": False}) for a in verified]
    claimed = set(verified_frames)
    n_new = 0
    for a in fresh.annotations:
        if salvage_key(a) in old_keys:
            continue  # a shot already captured (same game+ply) — not a new salvage
        if a.frame_index in claimed:
            # frame-index reuse against a kept frame (rare drift/collision): keep the
            # decided one, drop the ambiguous newcomer rather than shadow a verdict.
            log(f"salvage: frame {a.frame_index} collides with a kept frame — skipped")
            continue
        claimed.add(a.frame_index)
        merged.append(Annotation.from_dict({**a.to_dict(), "verified_by_human": False}))
        n_new += 1

    prov = {
        **fresh.provenance,
        "stage": Stage.CANDIDATES.value,
        "salvage": True,
        "n_verified_kept": len(verified),
        "n_salvaged_new": n_new,
    }
    candidates = AnnotationFile(provenance=prov, annotations=merged)
    return SalvageMerge(candidates=candidates,
                        reviewed=preseed_reviewed(candidates, verified_frames))
