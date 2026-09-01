"""Broadcast layout templates: segment the video into shots, cluster them by
composition, and keep only shots whose template a human marked *useful* (entire
board visible) — cropping the board region labeled once per template.

The director cycles a small fixed set of compositions, so classifying each shot to
a template turns the heavy discard into a cheap lookup and gives a stable crop rect
(and clock/name regions) per composition. This module is pure logic; the Gradio
labeling UI lives in `labeler`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from chessqueries.annotate.video import VideoFile

# Templates are shared across SLCC broadcasts (the production reuses compositions),
# so the registry is vendored package metadata, independent of any single video.
# New videos reuse it and only append genuinely new templates.
DEFAULT_REGISTRY_PATH = Path(__file__).parent / "resources" / "slcc_layouts.json"

# Matching thresholds, in the DINOv2 PCA-256 descriptor's cosine space (see
# `annotate.embedding`). A shot at or above MIN_SIMILARITY to its nearest template is
# that composition; below, it is a candidate new template. Derived on 20 videos:
# board-bearing shots match their template at cosine >= 0.87 (the hand-picked board
# anchors floor at 0.899), while the gate sits below that to keep them and above the
# genuinely-new tail.
MIN_SIMILARITY = 0.85
# Greedy cluster radius (1 - cosine) for pooling genuinely-new shots into candidate
# templates. Same-composition shots sit within ~0.08; = 1 - MIN_SIMILARITY keeps them
# together while separating distinct new compositions.
CLUSTER_MAX_DISTANCE = 0.15

# An already-labeled composition re-surfaces for labeling forever when its shots fall in
# the gap between the clustering radius and the strict match threshold: they cluster +
# get labeled, but no single shot ever re-matches the saved centroid. We suppress this
# re-appearance at the cheapest *safe* level (see `partition_templates`), and the safe
# level depends only on what a wrong suppression would cost:
#
#   * JUNK / no-crop templates -> per shot, loosely. Misattributing a shot among junk
#     compositions is harmless (it's discarded either way), so we can absorb any shot
#     whose nearest template is junk at this looser bar.
#   * USEFUL / HARD / cropped templates -> per cluster, strictly. A loose per-shot match
#     could swallow a genuinely-new lookalike composition we should have labeled, so we
#     only suppress when the *denoised cluster mean* lands on a template — confident
#     enough that it is the same composition, not a new one.
JUNK_MATCH_SIMILARITY = 0.80
KNOWN_COMPOSITION_SIMILARITY = 0.95


@dataclass(frozen=True)
class Rect:
    """An axis-aligned pixel rectangle in full-frame coordinates."""

    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"rect needs positive size, got {self.w}x{self.h}")
        if self.x < 0 or self.y < 0:
            raise ValueError(f"rect origin must be non-negative, got ({self.x},{self.y})")

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y : self.y + self.h, self.x : self.x + self.w]

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]

    @classmethod
    def from_list(cls, v: list[int]) -> "Rect":
        return cls(*(int(c) for c in v))


class Quality(str, Enum):
    """How usable a composition is for the dataset."""

    USEFUL = "useful"  # clean board -> auto-process
    HARD = "hard"  # difficult angle -> process but flag for manual curation
    JUNK = "junk"  # no usable board -> stored only so future videos auto-discard it


# Region kinds a Layout may carry, in label order: board crop, the broadcast
# digital clock overlay, separate white/black nameplates. Each maps to a Layout
# ``<kind>_rect`` field; the labeler draws them by kind.
REGION_KINDS = ("board", "digital_clock", "white_name", "black_name")
RECT_FIELDS = tuple(f"{kind}_rect" for kind in REGION_KINDS)


@dataclass(frozen=True)
class Layout:
    """One broadcast composition: a quality grade plus the regions `identify` reads.

    ``USEFUL``/``HARD`` shots must carry a board crop rect; clock/name rects are
    optional (filled only where those overlays appear in this composition)."""

    id: str
    quality: Quality
    board_rect: Rect | None = None
    digital_clock_rect: Rect | None = None
    white_name_rect: Rect | None = None
    black_name_rect: Rect | None = None

    def __post_init__(self) -> None:
        if self.processable and self.board_rect is None:
            raise ValueError(f"{self.quality.value} layout {self.id!r} must have a board_rect")

    @property
    def processable(self) -> bool:
        """Produces samples (USEFUL or HARD); JUNK is discarded."""
        return self.quality in (Quality.USEFUL, Quality.HARD)

    @property
    def requires_curation(self) -> bool:
        """HARD angles always go through manual review."""
        return self.quality == Quality.HARD

    @property
    def board_only(self) -> bool:
        """Processable but with no clock/name overlay to OCR — the board is the only
        signal, so these shots are a *gap* for the clock-OCR pass and must be labeled
        by the model-retrieval pass once a recognizer exists."""
        return self.processable and not any(
            getattr(self, f) for f in RECT_FIELDS if f != "board_rect"
        )

    def to_dict(self) -> dict:
        d: dict = {"id": self.id, "quality": self.quality.value}
        for f in RECT_FIELDS:
            r = getattr(self, f)
            d[f] = r.as_list() if r else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Layout":
        rects = {f: (Rect.from_list(d[f]) if d.get(f) else None) for f in RECT_FIELDS}
        return cls(id=str(d["id"]), quality=Quality(d["quality"]), **rects)


@dataclass(frozen=True)
class TemplateMatch:
    """The nearest template to one descriptor, with the cosine similarity that gates it."""

    template_id: str
    similarity: float


@dataclass(frozen=True)
class TemplateAssignment:
    """A batch of descriptors sorted against the registry: the template each one matched
    (``None`` where nothing cleared the gate) and the positions of those unmatched
    descriptors — the candidates for a *new* template. Blank frames are neither."""

    template_ids: list[str | None]
    unmatched: list[int]

    def __post_init__(self) -> None:
        if any(self.template_ids[i] is not None for i in self.unmatched):
            raise ValueError("unmatched index points at an assigned descriptor")


@dataclass
class LayoutRegistry:
    """Named templates plus their cluster centroids, so future shots/frames can be
    classified by nearest centroid. Persisted as JSON alongside the dataset."""

    layouts: dict[str, Layout]
    centroids: dict[str, list[float]]  # layout id -> normalized descriptor
    # Cached (ids, stacked-centroid-matrix) so classify/assign don't rebuild it per
    # call; invalidated (set to None) on every add/upsert. Not part of identity.
    _matrix: "tuple[list[str], np.ndarray] | None" = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        missing = set(self.layouts) ^ set(self.centroids)
        if missing:
            raise ValueError(f"layouts and centroids must share ids; differ on {missing}")

    def _stacked(self) -> "tuple[list[str], np.ndarray]":
        """(ids, [n_templates, dim] centroid matrix), built once and cached."""
        if self._matrix is None:
            ids = list(self.centroids)
            mat = np.asarray([self.centroids[i] for i in ids], dtype=np.float32)
            self._matrix = (ids, mat)
        return self._matrix

    @classmethod
    def empty(cls) -> "LayoutRegistry":
        return cls(layouts={}, centroids={})

    def upsert(self, layout: Layout, centroid: np.ndarray) -> None:
        """Add a template, or replace an existing one with the same id.

        Idempotent: re-committing a layout (e.g. clicking "Save all" twice, or
        saving a session whose earlier templates already landed in the registry)
        overwrites in place instead of raising."""
        self.layouts[layout.id] = layout
        self.centroids[layout.id] = [float(x) for x in centroid]
        self._matrix = None

    def classify(self, descriptor: np.ndarray) -> TemplateMatch:
        """Nearest template and its cosine similarity for a frame descriptor."""
        if not self.centroids:
            raise ValueError("registry has no templates to classify against")
        ids, mat = self._stacked()
        sims = mat @ descriptor
        best = int(np.argmax(sims))
        return TemplateMatch(template_id=ids[best], similarity=float(sims[best]))

    def assign(
        self,
        descriptors: np.ndarray,
        *,
        min_similarity: float,
        junk_similarity: float | None = None,
    ) -> TemplateAssignment:
        """Classify each descriptor against known templates; below ``min_similarity``
        it is left unmatched (a candidate *new* template).

        One batched matmul over all descriptors against the cached centroid matrix —
        O(n_desc · n_templates) in BLAS, not a Python loop of per-shot matrix rebuilds.

        When ``junk_similarity`` is given, a shot still unmatched by the strict rule is
        absorbed into its nearest template if that template is *junk* and the similarity
        clears ``junk_similarity`` — recognizing recurring known-junk whose shots fall in
        the cluster/match gap, so it stops re-surfacing for labeling (see
        :data:`JUNK_MATCH_SIMILARITY`). Restricting it to the *nearest* template being junk
        keeps shots that are closer to a real composition out of the junk bucket.

        A *blank* frame (uniform/black) zero-means to the zero vector, so it has cosine 0
        to everything — it can never match (not even its own saved centroid) and would
        re-surface as a singleton forever. Blanks carry no board, so they are skipped
        entirely: assigned no template and left out of ``unmatched`` (never labeled).
        """
        d = np.asarray(descriptors, dtype=np.float32)
        if d.ndim == 1:
            d = d[None, :]
        blank = np.linalg.norm(d, axis=1) < 1e-6  # uniform frame -> degenerate descriptor
        if not self.centroids:
            return TemplateAssignment(
                template_ids=[None] * len(d),
                unmatched=[i for i in range(len(d)) if not blank[i]],
            )
        ids, mat = self._stacked()
        sims = d @ mat.T  # [n_desc, n_templates]
        best = sims.argmax(axis=1)
        best_sim = sims[np.arange(len(d)), best]
        absorb_junk = junk_similarity is not None
        is_junk = (
            [lyt.quality is Quality.JUNK for lyt in (self.layouts[i] for i in ids)]
            if absorb_junk
            else None
        )
        assignments: list[str | None] = []
        unmatched: list[int] = []
        for i, (b, s) in enumerate(zip(best, best_sim)):
            if blank[i]:
                assignments.append(None)  # no content -> skip, never surface for labeling
            elif s >= min_similarity:
                assignments.append(ids[b])
            elif absorb_junk and is_junk[b] and s >= junk_similarity:
                assignments.append(ids[b])  # recurring known-junk in the cluster/match gap
            else:
                assignments.append(None)
                unmatched.append(i)
        return TemplateAssignment(template_ids=assignments, unmatched=unmatched)

    def save(self, path: Path) -> None:
        """Persist the registry atomically, keeping one prior generation as ``.bak``.

        The whole payload is serialized, written to a sibling temp file, fsync'd, then
        moved into place with ``os.replace`` (an atomic rename on the same filesystem),
        so a crash/kill mid-write cannot truncate the live registry. The previous file
        is rotated to ``<path>.bak`` first, a cheap one-generation fallback."""
        path = Path(path)
        payload = json.dumps(
            {
                "layouts": [self.layouts[i].to_dict() for i in self.layouts],
                "centroids": self.centroids,
            },
            indent=2,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            os.replace(path, path.with_name(path.name + ".bak"))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "LayoutRegistry":
        """Load the registry, tolerating a missing or empty file (a fresh setup, or a
        save that never completed) by returning an empty registry instead of crashing.
        A ``<path>.bak`` sibling is never loaded automatically — recovery is deliberate."""
        path = Path(path)
        text = path.read_text() if path.exists() else ""
        if not text.strip():
            return cls.empty()
        d = json.loads(text)
        layouts = {item["id"]: Layout.from_dict(item) for item in d["layouts"]}
        return cls(layouts=layouts, centroids=d["centroids"])


@dataclass(frozen=True)
class Shot:
    """A contiguous run of frames (one camera composition)."""

    index: int
    start_frame: int
    end_frame: int  # exclusive

    def __post_init__(self) -> None:
        if self.end_frame <= self.start_frame:
            raise ValueError(f"shot {self.index}: end {self.end_frame} <= start {self.start_frame}")

    @property
    def keyframe_index(self) -> int:
        return (self.start_frame + self.end_frame) // 2


def segment_shots(
    video: VideoFile,
    *,
    threshold: float = 27.0,
    max_frames: int | None = None,
    frame_skip: int = 0,
    show_progress: bool = False,
) -> list[Shot]:
    """Split the video into shots via PySceneDetect content detection.

    This decodes every frame, so it dominates runtime on a full broadcast (~133
    fps on 1080p here, i.e. several minutes for a 1.5h video). Pass
    ``show_progress=True`` for PySceneDetect's tqdm bar (frames done + ETA), and
    ``frame_skip>0`` to process every ``frame_skip+1``-th frame for a roughly
    proportional speedup at the cost of coarser cut boundaries.
    """
    from scenedetect import ContentDetector, SceneManager, open_video

    stream = open_video(str(video.path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
    end_time = None if max_frames is None else min(max_frames, video.frame_count)
    manager.detect_scenes(
        stream, end_time=end_time, frame_skip=frame_skip, show_progress=show_progress
    )
    scenes = manager.get_scene_list()
    if not scenes:  # no cuts detected -> whole span is one shot
        end = max_frames or video.frame_count
        return [Shot(index=0, start_frame=0, end_frame=end)]
    return [
        Shot(index=i, start_frame=s.get_frames(), end_frame=e.get_frames())
        for i, (s, e) in enumerate(scenes)
    ]


def save_shots(shots: list[Shot], path: Path) -> None:
    """Cache shot boundaries so the slow full-video segmentation runs only once."""
    Path(path).write_text(json.dumps([[s.start_frame, s.end_frame] for s in shots]))


def load_shots(path: Path) -> list[Shot]:
    rows = json.loads(Path(path).read_text())
    return [Shot(index=i, start_frame=s, end_frame=e) for i, (s, e) in enumerate(rows)]


@dataclass(frozen=True)
class Clustering:
    """Descriptors grouped by similarity: the cluster index each descriptor landed in,
    and the ``[n_clusters, dim]`` matrix of their unit-norm mean centroids."""

    labels: list[int]
    centroids: np.ndarray

    def __post_init__(self) -> None:
        if self.labels and max(self.labels) >= len(self.centroids):
            raise ValueError(f"label {max(self.labels)} has no centroid "
                             f"({len(self.centroids)} clusters)")

    def __len__(self) -> int:
        return len(self.centroids)


def cluster_descriptors(
    descriptors: np.ndarray, *, max_distance: float = CLUSTER_MAX_DISTANCE
) -> Clustering:
    """Greedy single-pass clustering by cosine distance (no sklearn dependency).

    Assigns each descriptor to the nearest existing cluster within ``max_distance``
    (``1 - cosine``), else starts a new one; centroids are running unit-norm means.
    """
    sums: list[np.ndarray] = []
    counts: list[int] = []
    centroids: list[np.ndarray] = []
    labels: list[int] = []
    for d in descriptors:
        if centroids:
            sims = np.asarray([float(d @ c) for c in centroids])
            best = int(np.argmax(sims))
            if (1.0 - sims[best]) <= max_distance:
                labels.append(best)
                sums[best] += d
                counts[best] += 1
                centroids[best] = _unit(sums[best])
                continue
        labels.append(len(centroids))
        sums.append(d.copy())
        counts.append(1)
        centroids.append(_unit(d))
    return Clustering(labels=labels, centroids=np.asarray(centroids))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


@dataclass(frozen=True)
class TemplateCluster:
    """A candidate new template pooled across videos: its centroid and the
    ``(video_index, shot_index)`` members that fed it."""

    centroid: np.ndarray
    members: list[tuple[int, int]]


@dataclass(frozen=True)
class TemplatePartition:
    """Result of sorting every shot against the registry in one pass.

    ``new_clusters`` are genuinely-new compositions to hand to the labeler.
    ``unmatched_per_video`` / ``board_only_per_video`` are per-video tallies (aligned to
    ``descriptors_per_video``) for the dashboard and the produce gate. Shots belonging to
    an already-labeled composition — junk absorbed per-shot in the gap, or a cluster whose
    denoised mean lands on a template — are *suppressed* from both the queue and the
    unmatched tally, so they never re-surface and never block produce."""

    new_clusters: list[TemplateCluster]
    unmatched_per_video: list[int]
    board_only_per_video: list[int]


def partition_templates(
    descriptors_per_video: list[np.ndarray],
    registry: "LayoutRegistry",
    *,
    min_similarity: float = MIN_SIMILARITY,
    max_distance: float = CLUSTER_MAX_DISTANCE,
) -> TemplatePartition:
    """One 'known-composition suppression' pass over every shot, jointly across videos.

    Each shot is either matched (>= ``min_similarity``), suppressed as an already-labeled
    composition (see :data:`JUNK_MATCH_SIMILARITY` / :data:`KNOWN_COMPOSITION_SIMILARITY`),
    or genuinely new. Genuinely-new shots cluster into ``new_clusters`` to label; the rest
    drop out of ``unmatched_per_video`` so they stop re-surfacing and stop gating produce.

    Pure (no IO): the survey, the produce gate, and the labeler all share this one pass.
    """
    n = len(descriptors_per_video)
    unmatched_per_video = [0] * n
    board_only_per_video = [0] * n
    refs: list[tuple[int, int]] = []
    vecs: list[np.ndarray] = []
    for vi, desc in enumerate(descriptors_per_video):
        if len(desc) == 0:
            continue
        # Level 1 — per-shot suppression: assign(junk_similarity=...) absorbs gap shots
        # whose nearest template is junk (no crop to mislabel), so they leave `unmatched`.
        assigned = registry.assign(
            desc, min_similarity=min_similarity, junk_similarity=JUNK_MATCH_SIMILARITY
        )
        board_only_per_video[vi] = sum(
            1 for a in assigned.template_ids if a is not None and registry.layouts[a].board_only
        )
        unmatched_per_video[vi] = len(assigned.unmatched)
        for si in assigned.unmatched:
            refs.append((vi, si))
            vecs.append(desc[si])
    if not vecs:
        return TemplatePartition([], unmatched_per_video, board_only_per_video)
    clustering = cluster_descriptors(np.asarray(vecs), max_distance=max_distance)
    new_clusters: list[TemplateCluster] = []
    for c, centroid in enumerate(clustering.centroids):
        members = [refs[i] for i, lab in enumerate(clustering.labels) if lab == c]
        # Level 2 — per-cluster suppression: a cluster whose denoised mean lands on an
        # existing template is that composition leaking through the gap (works for cropped
        # useful/hard templates, where a per-shot loose match would be unsafe). Drop it
        # from the queue AND from the unmatched tally — these shots are not new work.
        if (registry.centroids
                and registry.classify(centroid).similarity >= KNOWN_COMPOSITION_SIMILARITY):
            for vi, _si in members:
                unmatched_per_video[vi] -= 1
            continue
        new_clusters.append(TemplateCluster(centroid=centroid, members=members))
    return TemplatePartition(new_clusters, unmatched_per_video, board_only_per_video)
