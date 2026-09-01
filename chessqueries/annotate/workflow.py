"""The front door: survey what's on disk against the manifest and report what's
pending — *videos to download, templates to label, candidates to review, board-only
gaps awaiting the model* — each line naming the command that acts on it. Plus the
batch orchestrators (`ingest`, `produce`, `build_label_queue`) those commands call.

State is derived from disk markers (+ cheap classification of cached descriptors), so
`status` needs no network and re-reflects progress the instant a phase completes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from chessqueries.data.slcc import SplitRatio

from chessqueries.annotate import relay
from chessqueries.core import Split
from chessqueries.annotate import video as video_mod
from chessqueries.annotate.manifest import Manifest, VideoEntry, rounds_for_video
from chessqueries.annotate.pipeline import (
    DEFAULT_PRODUCE_WORKERS,
    DESCRIPTORS_EXT,
    merge_salvage,
    produce_one,
    segment_and_fingerprint,
)
from chessqueries.annotate.schema import AnnotationFile
from chessqueries.annotate.reconstruction import (
    MANIFEST_NAME,
    ReconstructionError,
    ReconstructionRecord,
    ReconstructionTransaction,
    loader_sample_id,
    stable_sample_id,
)
from chessqueries.annotate.templates import (
    DEFAULT_REGISTRY_PATH,
    MIN_SIMILARITY,
    LayoutRegistry,
    Rect,
    TemplateCluster,
    load_shots,
    partition_templates,
)

class SplitBy(str, Enum):
    """Leakage-grouping unit for dataset export: frames sharing a group never
    straddle splits. GAME keeps a game's positions together; VIDEO is stricter
    (whole broadcast); FRAME groups nothing — frames shuffle independently,
    permitting same-game leakage."""

    GAME = "game"
    VIDEO = "video"
    FRAME = "frame"


DEFAULT_DATA_DIR = Path("data/slcc")
DEFAULT_DATASET_DIR = DEFAULT_DATA_DIR / "dataset"  # the assembled, split-tagged export
RECOGNIZER_PREF_NAME = "recognizer.json"  # remembered crosscheck recognizer (local, gitignored)


class RecognizerKind(str, Enum):
    CHECKPOINT = "checkpoint"  # a full LitChessQueriesModel fine-tune (e.g. joint V2)
    ADAPTER = "adapter"  # a LoRA adapter (.pt), carries its own eval resolution


@dataclass(frozen=True)
class RecognizerRef:
    """The recognizer `crosscheck` should use, remembered across runs so the dashboard
    suggests a copy-pasteable command and a bare `annotate crosscheck` just works.

    A LoRA adapter embeds its eval resolution; a full checkpoint does not (it lives on
    the DataModule, not the hparams), so ``resolution`` is required for a checkpoint
    (ViT-L V2 = 644) and unused for an adapter."""

    kind: RecognizerKind
    path: Path
    resolution: int | None = None

    def __post_init__(self) -> None:
        if self.kind is RecognizerKind.CHECKPOINT and self.resolution is None:
            raise ValueError("a checkpoint recognizer needs an eval resolution (ViT-L V2 = 644)")

    def crosscheck_command(self) -> str:
        if self.kind is RecognizerKind.ADAPTER:
            return f"annotate crosscheck --adapter {self.path}"
        return f"annotate crosscheck --checkpoint {self.path} --resolution {self.resolution}"

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "path": str(self.path), "resolution": self.resolution}

    @classmethod
    def from_dict(cls, d: dict) -> "RecognizerRef":
        return cls(RecognizerKind(d["kind"]), Path(d["path"]), d.get("resolution"))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "RecognizerRef":
        return cls.from_dict(json.loads(Path(path).read_text()))


def recognizer_pref_path(data_dir: Path) -> Path:
    return Path(data_dir) / RECOGNIZER_PREF_NAME


def load_recognizer(data_dir: Path) -> RecognizerRef | None:
    """The remembered recognizer, or None if unset/unreadable (tolerant like the registry)."""
    p = recognizer_pref_path(data_dir)
    if not p.is_file():
        return None
    try:
        return RecognizerRef.load(p)
    except (ValueError, KeyError, OSError, json.JSONDecodeError):
        return None


def save_recognizer(ref: RecognizerRef, data_dir: Path) -> Path:
    p = recognizer_pref_path(data_dir)
    ref.save(p)
    return p


# --- on-disk artifact paths (one convention, shared by status + the orchestrators) ---
def video_files(data_dir: Path, vid: str, fmt: str) -> list[Path]:
    return [
        p
        for p in Path(data_dir).glob(f"{vid}.{fmt}.*")
        if p.suffix.lower() not in (".json", ".npy")
    ]


def shots_path(data_dir: Path, vid: str, fmt: str) -> Path:
    return Path(data_dir) / f"{vid}.{fmt}.shots.json"


def descriptors_path(data_dir: Path, vid: str, fmt: str) -> Path:
    return Path(data_dir) / f"{vid}.{fmt}{DESCRIPTORS_EXT}"


def candidates_path(data_dir: Path, vid: str) -> Path:
    return Path(data_dir) / f"{vid}.candidates.json"


def crosschecked_path(data_dir: Path, vid: str) -> Path:
    return Path(data_dir) / f"{vid}.crosschecked.json"


def reviewed_path(data_dir: Path, vid: str) -> Path:
    return Path(data_dir) / f"{vid}.reviewed.json"


def dataset_manifest_path(data_dir: Path) -> Path:
    from chessqueries.data.slcc import MANIFEST_NAME

    return Path(data_dir) / "dataset" / MANIFEST_NAME


@dataclass
class VideoState:
    """Where one video sits in the pipeline, from disk + cached-descriptor classify."""

    entry: VideoEntry
    downloaded: bool = False
    segmented: bool = False
    n_shots: int = 0
    n_unmatched: int = 0  # shots matching no template -> need labeling
    n_board_only: int = 0  # shots on a board-only template -> model-pass gap
    produced: bool = False
    crosschecked: bool = False  # the visual cross-check pass has triaged the candidates
    reviewed: bool = False  # every candidate decided (accepted/corrected or rejected)
    review_started: bool = False  # a reviewed file exists but some candidates are still pending
    n_candidates: int = 0
    n_verified: int = 0  # accepted/corrected by a human
    n_pending_review: int = 0  # kept by review but not yet verified
    n_rejected: int = 0  # candidates a reviewer dropped
    verified_stems: frozenset[str] = field(default_factory=frozenset)  # "<vid>_<frame>" shippable

    @property
    def video_id(self) -> str:
        return self.entry.video_id

    @property
    def templates_ok(self) -> bool:
        return self.segmented and self.n_unmatched == 0

    @property
    def stage(self) -> str:
        if self.reviewed:
            return "reviewed"
        if self.produced:
            return "produced"
        if not self.downloaded:
            return "registered"
        if not self.segmented:
            return "downloaded"
        if not self.templates_ok:
            return "needs-templates"
        if not self.entry.has_relay_mapping:
            return "needs-relay-map"
        return "ready-to-produce"


@dataclass
class Survey:
    states: list[VideoState]
    new_templates: list[TemplateCluster] = field(default_factory=list)
    n_templates: int = 0  # labeled templates in the shared registry
    # frame stems already in the assembled dataset export, grouped by video id.
    shipped_by_vid: dict[str, frozenset[str]] = field(default_factory=dict)
    # Verified frames intentionally omitted by the sharpest-frame deduplication pass.
    deduplicated_by_vid: dict[str, frozenset[str]] = field(default_factory=dict)
    recognizer: RecognizerRef | None = None  # remembered crosscheck recognizer, if any


def survey(
    manifest: Manifest,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    format_id: str = video_mod.DEFAULT_FORMAT_ID,
    min_similarity: float = MIN_SIMILARITY,
    show_progress: bool = False,
) -> Survey:
    """Per-video state + the joint set of unlabeled templates pooled across videos.

    Surveying loads cached descriptors and candidate/reviewed files per video, so it
    can take a moment over a large manifest; pass ``show_progress`` for a tqdm bar.
    """
    registry = LayoutRegistry.load(registry_path)
    states: list[VideoState] = []
    descriptors_per_video: list[np.ndarray] = []
    segmented_states: list[VideoState] = []  # parallel to descriptors_per_video
    entries = manifest
    if show_progress:
        from tqdm import tqdm

        entries = tqdm(list(manifest), desc="surveying", unit="video")
    for entry in entries:
        vid = entry.video_id
        st = VideoState(entry=entry)
        st.downloaded = bool(video_files(data_dir, vid, format_id))
        sp, dp = shots_path(data_dir, vid, format_id), descriptors_path(data_dir, vid, format_id)
        if sp.exists() and dp.exists():
            st.segmented = True
            st.n_shots = len(load_shots(sp))
            descriptors_per_video.append(np.load(dp))
            segmented_states.append(st)
        cp, rp = candidates_path(data_dir, vid), reviewed_path(data_dir, vid)
        st.produced = cp.exists()
        st.crosschecked = crosschecked_path(data_dir, vid).exists()
        if st.produced:
            st.n_candidates = len(AnnotationFile.load(cp).annotations)
            if rp.exists():
                # A reviewed file keeps accepted/corrected (verified) + untouched (not yet
                # verified) and drops rejected. So "done" means no kept-but-unverified remain.
                st.review_started = True
                rev = AnnotationFile.load(rp).annotations
                st.n_verified = sum(a.verified_by_human for a in rev)
                st.n_pending_review = sum(not a.verified_by_human for a in rev)
                st.n_rejected = st.n_candidates - len(rev)  # candidates dropped on review
                st.reviewed = st.n_pending_review == 0
                st.verified_stems = frozenset(
                    f"{a.video_id}_{a.frame_index}" for a in rev if a.verified_by_human
                )
            else:
                st.n_pending_review = st.n_candidates
        states.append(st)

    # One shared pass: genuinely-new clusters to label + per-video unmatched/board-only
    # tallies (already-labeled compositions suppressed from both — see partition_templates).
    part = partition_templates(descriptors_per_video, registry, min_similarity=min_similarity)
    for st, n_unmatched, n_board_only in zip(
        segmented_states, part.unmatched_per_video, part.board_only_per_video
    ):
        st.n_unmatched = n_unmatched
        st.n_board_only = n_board_only
    export_state = _dataset_export_state(data_dir)
    return Survey(
        states=states,
        new_templates=part.new_clusters,
        n_templates=len(registry.layouts),
        shipped_by_vid=export_state.shipped_by_vid,
        deduplicated_by_vid=export_state.deduplicated_by_vid,
        recognizer=load_recognizer(data_dir),
    )


@dataclass(frozen=True)
class DatasetExportState:
    shipped_by_vid: dict[str, frozenset[str]]
    deduplicated_by_vid: dict[str, frozenset[str]]


def _dataset_export_state(data_dir: Path) -> DatasetExportState:
    """Shipped and intentionally deduplicated frame identities, grouped by video.

    Lets the dashboard report what `reconstruct` would *add* versus what it already
    holds (and what a re-review would now drop). Tolerant of a missing/partial file.
    """
    manifest = dataset_manifest_path(data_dir)
    if not manifest.is_file():
        return DatasetExportState({}, {})
    try:
        payload = json.loads(manifest.read_text())
        records = payload.get("samples", [])
        deduplicated = payload.get("provenance", {}).get("deduplicated_by_video", {})
    except (json.JSONDecodeError, OSError):
        return DatasetExportState({}, {})
    grouped: dict[str, set[str]] = {}
    for r in records:
        if "image" not in r:
            continue
        stem = Path(r["image"]).stem
        grouped.setdefault(r.get("video_id") or stem, set()).add(stem)
    if not isinstance(deduplicated, dict):
        deduplicated = {}
    return DatasetExportState(
        shipped_by_vid={vid: frozenset(stems) for vid, stems in grouped.items()},
        deduplicated_by_vid={
            str(video_id): frozenset(str(sample_id) for sample_id in sample_ids)
            for video_id, sample_ids in deduplicated.items()
            if isinstance(sample_ids, list)
        },
    )


@dataclass(frozen=True)
class Action:
    icon: str
    title: str
    detail: str
    command: str


def pending_actions(survey: Survey) -> list[Action]:
    """The actionable lines for the dashboard (only non-empty ones)."""
    s = survey.states
    out: list[Action] = []

    need_ingest = [v for v in s if not (v.downloaded and v.segmented)]
    if need_ingest:
        n_dl = sum(not v.downloaded for v in need_ingest)
        out.append(
            Action(
                "⬇",
                f"{len(need_ingest)} video(s) to ingest",
                f"{n_dl} not downloaded; the rest downloaded but not segmented",
                "annotate ingest",
            )
        )

    if survey.new_templates:
        n_videos = len({m[0] for c in survey.new_templates for m in c.members})
        out.append(
            Action(
                "🏷",
                f"{len(survey.new_templates)} new template(s) to label",
                f"pooled across {n_videos} video(s)",
                "annotate label",
            )
        )

    ready = [v for v in s if v.templates_ok and v.entry.has_relay_mapping and not v.produced]
    if ready:
        out.append(
            Action(
                "⚙",
                f"{len(ready)} video(s) ready to auto-label",
                ", ".join(v.video_id for v in ready),
                "annotate produce",
            )
        )

    no_map = [v for v in s if v.templates_ok and not v.entry.has_relay_mapping and not v.produced]
    if no_map:
        out.append(
            Action(
                "🔗",
                f"{len(no_map)} video(s) need a relay mapping",
                "set `tournaments` (or `round_ids`) in the manifest: "
                + ", ".join(v.video_id for v in no_map),
                "edit resources/slcc_videos.json",
            )
        )

    to_xcheck = [v for v in s if v.produced and not v.crosschecked and not v.reviewed]
    if to_xcheck:
        ids = ", ".join(v.video_id for v in to_xcheck)
        if survey.recognizer is not None:
            # A recognizer was used before -> suggest the exact, copy-pasteable command.
            command = survey.recognizer.crosscheck_command()
            detail = f"visual model vs. clock consensus ({survey.recognizer.path.name}): {ids}"
        else:
            command = "annotate crosscheck --checkpoint <v2.ckpt> --resolution 644"
            detail = (
                "optional — visual model vs. clock consensus; pass a recognizer once "
                "(--checkpoint V2 @ 644 or --adapter <lora.pt>) and it's remembered: " + ids
            )
        out.append(Action("🔬", f"{len(to_xcheck)} video(s) to cross-check", detail, command))

    to_review = [v for v in s if v.produced and not v.reviewed]
    if to_review:
        n_pending = sum(v.n_pending_review for v in to_review)
        detail = f"across {len(to_review)} video(s)"
        resumed = [v for v in to_review if v.review_started]
        if resumed:
            detail += f"; {len(resumed)} partially reviewed — resume to finish"
        out.append(
            Action(
                "✅",
                f"{n_pending} candidate(s) to review",
                detail,
                "annotate review",
            )
        )

    n_gaps = sum(v.n_board_only for v in s)
    if n_gaps:
        out.append(
            Action(
                "🧩",
                f"~{n_gaps} board-only shot(s) await the model",
                "no clock/nameplate to OCR — fill once a v1 recognizer exists",
                "annotate model-pass  (deferred)",
            )
        )

    # `reconstruct` ships verified frames from fully-reviewed videos (a video with
    # candidates still pending waits — same gate as export_dataset's default).
    releasable = [v for v in s if v.reviewed]
    if releasable:
        changed = [
            v
            for v in releasable
            if v.verified_stems
            != survey.shipped_by_vid.get(v.video_id, frozenset())
            | survey.deduplicated_by_vid.get(v.video_id, frozenset())
        ]
        if not changed:
            return out
        verified = frozenset().union(*(v.verified_stems for v in releasable))
        shipped = frozenset().union(
            *(survey.shipped_by_vid.get(v.video_id, frozenset()) for v in releasable)
        )
        deduplicated = frozenset().union(
            *(survey.deduplicated_by_vid.get(v.video_id, frozenset()) for v in releasable)
        )
        new = verified - shipped - deduplicated
        already = verified & shipped
        dropped = shipped - verified  # shipped, but a re-review no longer verifies them
        detail = f"{len(verified)} verified across {len(releasable)} reviewed video(s)"
        if already:
            detail += f"; {len(already)} already in the dataset"
        if deduplicated & verified:
            detail += f"; {len(deduplicated & verified)} intentionally deduplicated"
        if dropped:
            detail += f"; ⚠ {len(dropped)} stale shipped frame(s) would be dropped"
        out.append(
            Action(
                "📦",
                f"{len(new)} new frame(s) to ship",
                detail,
                "annotate reconstruct --video "
                + " ".join(video.video_id for video in changed),
            )
        )
    return out


def render_status(manifest: Manifest, survey: Survey) -> str:
    """The dashboard string `status` prints."""
    n_tours = len({t for e in manifest for t in e.tournaments})
    head = f"SLCC broadcast dataset — {len(manifest)} video(s) · {n_tours} tournament(s)"
    actions = pending_actions(survey)
    if not actions:
        body = "  ✓  all caught up — nothing pending"
    else:
        body = "\n".join(
            f"  {a.icon}  {a.title:<34} → {a.command}\n       {a.detail}" for a in actions
        )
    return f"{head}\n\n{_render_totals(survey)}\n\n{body}\n"


def _render_totals(survey: Survey) -> str:
    """Registry size, the auto-labeled-sample tally, and labeling coverage toward
    `produce` (a video unlocks `produce` only once every shot matches a template)."""
    s = survey.states
    lines = [f"  registry: {survey.n_templates} labeled template(s)"]

    n_produced = sum(v.produced for v in s)
    n_candidates = sum(v.n_candidates for v in s)
    accepted = sum(v.n_verified for v in s)
    rejected = sum(v.n_rejected for v in s)
    pending = sum(v.n_pending_review for v in s if v.produced)
    if n_candidates == 0:
        lines.append("  samples: none auto-labeled yet (run `annotate produce`)")
    else:
        lines.append(
            f"  samples: {n_candidates} auto-labeled across {n_produced} video(s) — "
            f"{accepted} accepted · {pending} pending · {rejected} rejected"
        )

    # Convergence toward produce: surface how close the still-unlabeled videos are,
    # since `produce` stays hidden until a video hits 0 unmatched shots.
    needs = [v for v in s if v.segmented and not v.produced and not v.templates_ok]
    if needs:
        unmatched = sum(v.n_unmatched for v in needs)
        shots = sum(v.n_shots for v in needs)
        pct = round(100 * (1 - unmatched / shots)) if shots else 0
        note = (
            f"  coverage: {len(needs)} video(s) not fully labeled · {unmatched} shot(s) "
            f"unmatched ({pct}% matched) — keep running `annotate label` until 0 to unlock produce"
        )
        no_map = sum(not v.entry.has_relay_mapping for v in needs)
        if no_map:
            note += f"; {no_map} will then need a relay mapping"
        lines.append(note)
    return "\n".join(lines)


def pending_reviews(survey: Survey) -> list[VideoState]:
    """Produced videos still awaiting review, in the order `annotate review` walks them:
    partially-reviewed (resumable) first, then by video id — so a half-done video is
    finished before a fresh one is opened."""
    pend = [s for s in survey.states if s.produced and not s.reviewed]
    return sorted(pend, key=lambda s: (not s.review_started, s.video_id))


def review_queue_lines(queue: list[VideoState], *, position: int) -> str:
    """The batch-review header: which video of how many, how many remain after it, and
    the upcoming ids — so the reviewer sees the whole queue, not just the open video."""
    total = len(queue)
    cur = queue[position - 1]
    resume = ", resuming" if cur.review_started else ""
    upcoming = [s.video_id for s in queue[position:]]
    return (
        f"reviewing {position}/{total} · {total - position} video(s) left after this\n"
        f"  now:  {cur.video_id}  ({cur.n_pending_review} of {cur.n_candidates} candidate(s){resume})\n"
        f"  next: {', '.join(upcoming) if upcoming else '—'}"
    )


# --------------------------- batch orchestrators ---------------------------
def _targets(manifest: Manifest, video_ids: list[str] | None) -> list[VideoEntry]:
    if video_ids:
        return [manifest[v] for v in video_ids]
    return list(manifest)


def ingest(
    manifest: Manifest,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    video_ids: list[str] | None = None,
    format_id: str = video_mod.DEFAULT_FORMAT_ID,
    max_frames: int | None = None,
    log=print,
) -> None:
    """Phase A over a batch: download (skip if present) + segment + fingerprint, caching
    shots and descriptors. Needs no relay/registry — runs the moment videos are added."""
    targets = _targets(manifest, video_ids)
    failed: list[str] = []
    done: list[str] = []  # already fully ingested -> nothing to do
    for entry in targets:
        vid = entry.video_id
        sp = shots_path(data_dir, vid, format_id)
        dp = descriptors_path(data_dir, vid, format_id)
        if video_files(data_dir, vid, format_id) and sp.exists() and dp.exists():
            done.append(vid)
            log(f"=== {vid} ({entry.title or '—'}) — already ingested, skipping ===")
            continue
        log(f"\n=== ingest {vid} ({entry.title or '—'}) ===")
        try:
            vf = video_mod.download(vid, data_dir, format_id=format_id)
            segment_and_fingerprint(
                vf,
                shots_cache=sp,
                descriptors_cache=dp,
                max_frames=max_frames,
                log=log,
            )
        except Exception as e:  # one bad download/decode shouldn't abort an overnight batch
            failed.append(vid)
            log(f"!! {vid} failed: {e}\n   skipped — rerun `annotate ingest --video {vid}` to retry")
    n_new = len(targets) - len(failed) - len(done)
    log(f"\ningested {n_new} new, {len(done)} already done, {len(failed)} failed (of {len(targets)})")
    if failed:
        log(f"failed: {', '.join(failed)}")


def build_label_queue(
    manifest: Manifest,
    queue_dir: Path,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    format_id: str = video_mod.DEFAULT_FORMAT_ID,
    log=print,
) -> int:
    """Pool the unlabeled templates across every segmented video into one queue the
    Gradio labeler serves. Returns the number of new templates found."""
    from tqdm import tqdm

    from chessqueries.annotate.labeler import prepare_labeling_batch

    segmented = [
        e
        for e in manifest
        if shots_path(data_dir, e.video_id, format_id).exists()
        and descriptors_path(data_dir, e.video_id, format_id).exists()
    ]
    if not segmented:
        log("nothing segmented yet — run `annotate ingest` first")
        return 0
    log(f"loading {len(segmented)} segmented video(s)...")
    videos = []
    for entry in tqdm(segmented, desc="loading videos", unit="video"):
        vid = entry.video_id
        vf = video_mod.probe(video_files(data_dir, vid, format_id)[0], vid, format_id)
        videos.append((vf, load_shots(shots_path(data_dir, vid, format_id)), np.load(descriptors_path(data_dir, vid, format_id))))
    new = prepare_labeling_batch(videos, queue_dir, registry_path=registry_path)
    return len(new)


def produce(
    manifest: Manifest,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    video_ids: list[str] | None = None,
    format_id: str = video_mod.DEFAULT_FORMAT_ID,
    min_similarity: float = MIN_SIMILARITY,
    force: bool = False,
    salvage: bool = False,
    workers: int = DEFAULT_PRODUCE_WORKERS,
    log=print,
) -> list[Path]:
    """Phase C over a batch: clock-OCR align every video that is segmented, has its
    templates labeled, and carries a relay mapping. Skips (with a note) those not ready.
    Shots are labeled by ``workers`` processes (each owning a decoder + OCR engine);
    ``workers=1`` runs in-process, reusing one OCR reader across videos. Returns the
    candidate files written.

    ``salvage`` instead revisits already-produced, *fully-reviewed* videos with the current
    registry, preserving every prior human decision and surfacing for review ONLY the frames
    the better matching newly rescued (see :func:`~chessqueries.annotate.pipeline.merge_salvage`).
    It skips videos that aren't produced or whose review is unfinished."""
    from chessqueries.annotate.identify import OcrReader

    registry = LayoutRegistry.load(registry_path)
    # "Fully labeled?" must be decided with the SAME cross-video suppression the dashboard
    # uses: pooling rim shots across all videos is what lets an already-labeled composition
    # be recognized, so a per-video count would disagree with the survey's "ready" verdict.
    seg_ids: list[str] = []
    seg_descs: list[np.ndarray] = []
    for e in manifest:
        se, de = shots_path(data_dir, e.video_id, format_id), descriptors_path(data_dir, e.video_id, format_id)
        if se.exists() and de.exists():
            seg_ids.append(e.video_id)
            seg_descs.append(np.load(de))
    unmatched_by_vid = dict(
        zip(seg_ids, partition_templates(seg_descs, registry, min_similarity=min_similarity).unmatched_per_video)
    )
    ocr: OcrReader | None = None
    written: list[Path] = []
    for entry in _targets(manifest, video_ids):
        vid = entry.video_id
        sp, dp = shots_path(data_dir, vid, format_id), descriptors_path(data_dir, vid, format_id)
        if not (sp.exists() and dp.exists()):
            log(f"skip {vid}: not segmented (run ingest)")
            continue
        cp = candidates_path(data_dir, vid)
        if salvage:
            rp = reviewed_path(data_dir, vid)
            if not cp.exists():
                log(f"skip {vid}: not produced — salvage revisits produced videos (run produce)")
                continue
            if not rp.exists():
                log(f"skip {vid}: not reviewed — finish `annotate review` before salvaging")
                continue
            if not entry.has_relay_mapping:
                log(f"skip {vid}: no relay mapping (set tournaments/round_ids in the manifest)")
                continue
            prior_reviewed = AnnotationFile.load(rp)
            n_pending = sum(not a.verified_by_human for a in prior_reviewed.annotations)
            if n_pending:
                log(f"skip {vid}: review unfinished ({n_pending} pending) — finish `annotate review`")
                continue
            old_candidates = AnnotationFile.load(cp)
            descriptors = np.load(dp)
            vf = video_mod.probe(video_files(data_dir, vid, format_id)[0], vid, format_id)
            round_ids = rounds_for_video(vid)
            timelines = [tl for rid in round_ids for tl in relay.load_round(rid)]
            log(f"\n=== salvage {vid} — {entry.title or '(untitled)'} ({len(round_ids)} round(s)) ===")
            if workers == 1 and ocr is None:
                ocr = OcrReader()
            fresh = produce_one(
                vf, timelines, ocr, registry, load_shots(sp), descriptors,
                video_url=video_mod.YOUTUBE_WATCH_URL.format(video_id=vid),
                video_title=entry.title, min_similarity=min_similarity,
                workers=workers, log=log,
            )
            salvaged = merge_salvage(old_candidates, prior_reviewed, fresh, log=log)
            merged = salvaged.candidates
            n_new = merged.provenance["n_salvaged_new"]
            if n_new == 0:
                log(f"{vid}: no new frames salvaged — candidates/reviewed left untouched")
                continue
            merged.save(cp)
            salvaged.reviewed.save(rp)
            xp = crosschecked_path(data_dir, vid)
            if xp.exists():
                xp.unlink()  # stale: it triaged the OLD candidates; review falls back to candidates
                log(f"{vid}: removed stale {xp.name} (was triaged against the old candidates)")
            written.append(cp)
            log(f"{vid}: {n_new} new frame(s) salvaged, {merged.provenance['n_verified_kept']} "
                f"verified kept — `annotate review` will surface only the {n_new} new")
            continue
        if cp.exists() and not force:
            log(f"skip {vid}: already produced ({cp.name}); pass force to redo")
            continue
        if not entry.has_relay_mapping:
            log(f"skip {vid}: no relay mapping (set tournaments/round_ids in the manifest)")
            continue
        n_unmatched = unmatched_by_vid.get(vid, 0)
        if n_unmatched:
            log(f"skip {vid}: {n_unmatched} shot(s) unlabeled (run `annotate label` first)")
            continue

        descriptors = np.load(dp)
        vf = video_mod.probe(video_files(data_dir, vid, format_id)[0], vid, format_id)
        round_ids = rounds_for_video(vid)
        timelines = [tl for rid in round_ids for tl in relay.load_round(rid)]
        log(f"\n=== produce {vid} — {entry.title or '(untitled)'} ({len(round_ids)} round(s)) ===")
        if workers == 1 and ocr is None:
            ocr = OcrReader()
        ann = produce_one(
            vf,
            timelines,
            ocr,
            registry,
            load_shots(sp),
            descriptors,
            video_url=video_mod.YOUTUBE_WATCH_URL.format(video_id=vid),
            video_title=entry.title,
            min_similarity=min_similarity,
            workers=workers,
            log=log,
        )
        ann.save(cp)
        written.append(cp)
        log(f"wrote {len(ann.annotations)} candidate(s) -> {cp}")
    return written


def crosscheck(
    manifest: Manifest,
    *,
    adapter: Path | None = None,
    checkpoint: Path | None = None,
    resolution: int | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    video_ids: list[str] | None = None,
    format_id: str = video_mod.DEFAULT_FORMAT_ID,
    desync: int | None = None,
    tau_fit: int | None = None,
    tau_margin: float | None = None,
    force: bool = False,
    log=print,
) -> list[Path]:
    """Visual cross-check over a batch: for every produced video, run the recognizer
    against each candidate's ply window and triage into accept/review/quarantine.
    Loads the recognizer once and reuses it. Returns the files written.

    The recognizer is either a LoRA ``adapter`` or a full ``checkpoint`` (e.g. the joint
    V2 model, a full fine-tune rather than an adapter); pass exactly one. ``checkpoint``
    needs its eval ``resolution`` (ViT-L V2 = 644), which isn't stored in the ckpt."""
    from chessqueries.annotate import crosscheck as cc
    from chessqueries.annotate.recognize import Recognizer
    from chessqueries.annotate.video import FrameReader, probe

    if (adapter is None) == (checkpoint is None):
        raise ValueError("pass exactly one of adapter= or checkpoint=")
    if checkpoint is not None and resolution is None:
        raise ValueError("checkpoint= requires resolution= (ViT-L V2 = 644)")

    kw = {
        k: v
        for k, v in (("desync", desync), ("tau_fit", tau_fit), ("tau_margin", tau_margin))
        if v is not None
    }
    recognizer: Recognizer | None = None
    written: list[Path] = []
    for entry in _targets(manifest, video_ids):
        vid = entry.video_id
        cp = candidates_path(data_dir, vid)
        if not cp.exists():
            log(f"skip {vid}: not produced (run `annotate produce`)")
            continue
        xp = crosschecked_path(data_dir, vid)
        if xp.exists() and not force:
            log(f"skip {vid}: already cross-checked ({xp.name}); pass force to redo")
            continue
        vfiles = video_files(data_dir, vid, format_id)
        if not vfiles:
            log(f"skip {vid}: video file missing")
            continue
        if recognizer is None:
            if checkpoint is not None:
                log(f"loading recognizer from checkpoint {checkpoint} @ {resolution}px ...")
                recognizer = Recognizer.from_checkpoint(checkpoint, resolution)
            else:
                log(f"loading recognizer from adapter {adapter} ...")
                recognizer = Recognizer.from_adapter(adapter)
        annfile = AnnotationFile.load(cp)
        log(f"\n=== crosscheck {vid} — {entry.title or '(untitled)'} ({len(annfile.annotations)} candidate(s)) ===")
        with FrameReader(probe(vfiles[0], vid, format_id)) as reader:
            result = cc.crosscheck_file(annfile, recognizer, reader, log=log, **kw)
        result.save(xp)
        written.append(xp)
        log(f"wrote {len(result.annotations)} cross-checked annotation(s) -> {xp}")
    return written


def _group_key(rec: dict, split_by: SplitBy) -> str:
    """The leakage-grouping key for one record (see `SplitBy`)."""
    stem = Path(rec["image"]).stem
    if split_by is SplitBy.FRAME:
        return stem
    if split_by is SplitBy.VIDEO:
        return rec.get("video_id") or stem
    if split_by is SplitBy.GAME:
        return rec.get("game_id") or rec.get("video_id") or stem
    raise ValueError(f"unhandled split_by: {split_by}")


def _dedup_key(rec: dict) -> tuple[str, str, str, str]:
    """Near-duplicate identity: same game, same video, same board template *and* the
    same piece placement (FEN field 1). Two frames sharing it are the same position
    from the same viewpoint, differing only in incidental motion — a hand reaching in,
    the clock ticking, camera jitter. A different template (viewpoint) is NOT a
    duplicate, so it stays out of the key and those frames are kept as genuine variation.
    """
    placement = rec["gt_fen"].split(" ", 1)[0]
    return (
        rec.get("game_id") or "",
        rec.get("video_id") or "",
        rec.get("template_id") or "",
        placement,
    )


def _sharpness(path: Path) -> float:
    """Focus proxy = variance of the Laplacian. Higher ⇒ crisper edges ⇒ less motion
    blur / occlusion, so it's the best representative to keep from a near-dup group.
    Returns -1 for an unreadable image so it's never chosen over a readable one."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


@dataclass(frozen=True)
class Deduplication:
    """A record set split by the near-duplicate collapse: the frames that survive and
    the blurrier copies removed (empty when nothing collapsed)."""

    kept: list[dict]
    dropped: list[dict]


def dedup_near_duplicates(records: list[dict], image_root: Path) -> Deduplication:
    """Collapse each near-duplicate group (see :func:`_dedup_key`) to its single
    sharpest frame. Groups are keyed on game+video+template+placement, so the same
    position under a different viewpoint survives untouched."""
    from collections import defaultdict

    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[_dedup_key(r)].append(r)

    kept: list[dict] = []
    dropped: list[dict] = []
    for grp in groups.values():
        if len(grp) == 1:
            kept.append(grp[0])
            continue
        best = max(grp, key=lambda r: _sharpness(Path(image_root) / r["image"]))
        kept.append(best)
        dropped.extend(r for r in grp if r is not best)
    return Deduplication(kept=kept, dropped=dropped)


@dataclass(frozen=True)
class Diversity:
    """How much *distinct* material a set of exported records carries. Many shots of one
    game are one game, so the game/video counts say how representative the frames are."""

    n_frames: int
    n_games: int
    n_videos: int


def diversity(records: list[dict]) -> Diversity:
    """Frame/game/video counts over manifest records."""
    return Diversity(
        n_frames=len(records),
        n_games=len({r["game_id"] for r in records if r.get("game_id")}),
        n_videos=len({r["video_id"] for r in records}),
    )


@dataclass(frozen=True)
class ExistingInternalExport:
    """Validated state needed from a prior internal dataset export."""

    records: tuple[dict, ...] = ()
    deduplicated_by_video: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class InternalExportPlan:
    """Reviewed annotations mapped to one transactional reconstruction operation."""

    records: tuple[ReconstructionRecord, ...]
    reviewed_video_ids: tuple[str, ...]
    prior: ExistingInternalExport
    format_ids: tuple[tuple[str, str], ...]
    grouping_key: str
    rebuild_all: bool


def _load_existing_internal_export(manifest_path: Path) -> ExistingInternalExport:
    if not manifest_path.is_file():
        return ExistingInternalExport()
    try:
        payload = json.loads(manifest_path.read_text())
        records = payload["samples"]
        deduplicated = payload.get("provenance", {}).get("deduplicated_by_video", {})
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ReconstructionError(f"cannot read existing dataset manifest {manifest_path}") from exc
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ReconstructionError(f"existing dataset manifest {manifest_path} has invalid samples")
    ids = [loader_sample_id(record) for record in records]
    if len(ids) != len(set(ids)):
        raise ReconstructionError(f"existing dataset manifest {manifest_path} has duplicate samples")
    if not isinstance(deduplicated, dict) or any(
        not isinstance(video_id, str)
        or not isinstance(sample_ids, list)
        or any(not isinstance(sample_id, str) for sample_id in sample_ids)
        for video_id, sample_ids in deduplicated.items()
    ):
        raise ReconstructionError(
            f"existing dataset manifest {manifest_path} has invalid deduplication provenance"
        )
    return ExistingInternalExport(
        records=tuple(records),
        deduplicated_by_video=tuple(
            (video_id, tuple(sample_ids))
            for video_id, sample_ids in deduplicated.items()
        ),
    )


def _transaction_grouping_key(split_by: SplitBy) -> str:
    return {
        SplitBy.GAME: "game_id",
        SplitBy.VIDEO: "video_id",
        SplitBy.FRAME: "sample_id",
    }[split_by]


def _build_internal_export_plan(
    annotation_files: dict[str, AnnotationFile],
    prior: ExistingInternalExport,
    ratio: "SplitRatio",
    *,
    verified_only: bool,
    format_id: str,
    seed: int,
    split_by: SplitBy,
    regroup: bool,
    rebuild_all: bool,
) -> InternalExportPlan:
    from chessqueries.data.slcc import plan_splits

    for video_id, annfile in annotation_files.items():
        mismatched = [
            annotation.video_id
            for annotation in annfile.annotations
            if annotation.video_id != video_id
        ]
        if mismatched:
            raise ReconstructionError(
                f"reviewed file for {video_id} contains annotation for {mismatched[0]}"
            )
    annotations = [
        annotation
        for annfile in annotation_files.values()
        for annotation in annfile.annotations
        if annotation.verified_by_human or not verified_only
    ]
    selected_rows = {
        stable_sample_id(annotation): {
            "sample_id": stable_sample_id(annotation),
            "image": f"images/{stable_sample_id(annotation)}.jpg",
            "video_id": annotation.video_id,
            "game_id": annotation.game_id,
        }
        for annotation in annotations
    }
    if len(selected_rows) != len(annotations):
        raise ReconstructionError("reviewed annotations contain duplicate sample identities")

    planning_rows = {loader_sample_id(record): record for record in prior.records}
    planning_rows.update(selected_rows)
    groups = {
        sample_id: _group_key(record, split_by)
        for sample_id, record in planning_rows.items()
    }
    existing = (
        {
            loader_sample_id(record): record["split"]
            for record in prior.records
            if "split" in record
        }
        if not regroup
        else {}
    )
    splits = plan_splits(groups, ratio=ratio, seed=seed, existing=existing)
    records = tuple(
        ReconstructionRecord(
            stable_sample_id(annotation),
            Split(splits[stable_sample_id(annotation)]),
            annotation,
        )
        for annotation in annotations
    )
    formats = tuple(
        (video_id, str(annfile.provenance.get("format_id", format_id)))
        for video_id, annfile in annotation_files.items()
    )
    return InternalExportPlan(
        records=records,
        reviewed_video_ids=tuple(annotation_files),
        prior=prior,
        format_ids=formats,
        grouping_key=_transaction_grouping_key(split_by),
        rebuild_all=rebuild_all,
    )


def export_dataset(
    manifest: Manifest,
    ratio: "SplitRatio",
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    out_dir: Path | None = None,
    video_ids: list[str] | None = None,
    verified_only: bool = True,
    seed: int = 0,
    split_by: SplitBy | str = SplitBy.GAME,
    regroup: bool = False,
    dedup: bool = True,
    rebuild_all: bool = False,
    format_id: str = video_mod.DEFAULT_FORMAT_ID,
    log=print,
) -> Path:
    """Assemble reviewed videos into the versioned, split-tagged SLCC dataset that
    the :class:`~chessqueries.data.slcc.SLCC` loader reads.

    A scoped ``video_ids`` operation replaces those videos while preserving every
    other manifest record and split. An intentional whole-dataset replacement requires
    ``rebuild_all=True``. All crops and the merged manifest are staged, validated, and
    atomically committed by :class:`ReconstructionTransaction`.

    ``dedup`` (default on) collapses near-duplicate frames — same game, viewpoint
    (template) and piece placement, differing only in a moving hand / ticking clock /
    camera jitter — to the single sharpest frame, deleting the redundant crops. Pass
    ``dedup=False`` to ship every verified frame. The same position under a *different*
    viewpoint is kept; this only removes truly redundant same-view repeats.
    """
    out_dir = Path(out_dir or DEFAULT_DATASET_DIR)
    if rebuild_all == bool(video_ids):
        raise ValueError("pass video_ids for a scoped export or rebuild_all=True, but not both")
    if regroup and not rebuild_all:
        raise ValueError("regroup=True requires rebuild_all=True")

    targets = _targets(manifest, None if rebuild_all else video_ids)
    # Only ship fully-reviewed videos by default: one with candidates still pending
    # waits (its verified frames go out once review finishes). --include-unverified
    # (verified_only=False) lifts the gate to also ship kept-but-unverified frames.
    annotation_files: dict[str, AnnotationFile] = {}
    for e in targets:
        rp = reviewed_path(data_dir, e.video_id)
        if not rp.exists():
            continue
        annfile = AnnotationFile.load(rp)
        pending = (
            sum(not annotation.verified_by_human for annotation in annfile.annotations)
            if verified_only
            else 0
        )
        if pending:
            log(
                f"skip {e.video_id}: {pending} candidate(s) still pending review "
                f"(finish `annotate review`, or pass --include-unverified)"
            )
            continue
        annotation_files[e.video_id] = annfile
    if not annotation_files:
        log("no fully-reviewed videos to export yet (finish `annotate review` first).")
        raise SystemExit(0)

    prior = _load_existing_internal_export(out_dir / MANIFEST_NAME)
    split_by = SplitBy(split_by)
    if regroup:
        log("--regroup: re-planning ALL splits by group (prior assignments ignored)")
    plan = _build_internal_export_plan(
        annotation_files,
        prior,
        ratio,
        verified_only=verified_only,
        format_id=format_id,
        seed=seed,
        split_by=split_by,
        regroup=regroup,
        rebuild_all=rebuild_all,
    )
    reviewed_set = set(plan.reviewed_video_ids)
    format_ids = dict(plan.format_ids)
    preserved_video_ids = [
        record.get("video_id")
        for record in plan.prior.records
        if not plan.rebuild_all and record.get("video_id") not in reviewed_set
    ]
    dataset_video_ids = list(
        dict.fromkeys(
            [
                *(video_id for video_id in preserved_video_ids if video_id),
                *plan.reviewed_video_ids,
            ]
        )
    )
    deduplicated_by_video = {
        video_id: set(sample_ids)
        for video_id, sample_ids in plan.prior.deduplicated_by_video
        if not plan.rebuild_all and video_id not in reviewed_set
    }

    replace_existing = None
    if not plan.rebuild_all:

        def in_replacement_scope(record: dict) -> bool:
            return record.get("video_id") in reviewed_set

        replace_existing = in_replacement_scope

    collapsed_stems: set[str] = set()
    with ReconstructionTransaction(
        list(plan.records),
        out_dir,
        preserve_unselected=not plan.rebuild_all,
        replace_existing=replace_existing,
        grouping_key=plan.grouping_key,
    ) as transaction:
        pending_by_video: dict[str, list[ReconstructionRecord]] = {}
        for record in transaction.pending_records:
            pending_by_video.setdefault(record.annotation.video_id, []).append(record)
        for video_id, pending in pending_by_video.items():
            video = video_mod.download(video_id, data_dir, format_id=format_ids[video_id])
            with video_mod.FrameReader(video) as reader:
                for record in pending:
                    frame = reader.frame_at_index(record.annotation.frame_index)
                    crop = Rect.from_list(record.annotation.crop_bbox).crop(frame)
                    transaction.write_image(record, crop)

        if dedup:
            deduped = dedup_near_duplicates(
                [record.loader_record() for record in plan.records], transaction.stage_dir
            )
            collapsed = deduped.dropped
            if collapsed:
                collapsed_stems = {loader_sample_id(record) for record in collapsed}
                transaction.discard_records(collapsed_stems)
                for record in collapsed:
                    deduplicated_by_video.setdefault(record["video_id"], set()).add(
                        loader_sample_id(record)
                    )
                n_groups = len({_dedup_key(record) for record in collapsed})
                log(
                    f"\n⊚ collapsed {len(collapsed)} near-duplicate frame(s) across "
                    f"{n_groups} position(s) — kept the sharpest of each "
                    f"(same game+viewpoint+position, differing only in hand/clock/jitter):"
                )
                for record in collapsed:
                    log(f"    {record['image']}")

        report = transaction.commit(
            {
                "reviewed_videos": dataset_video_ids,
                "split_by": split_by.value,
                "ratio": [ratio.train, ratio.val, ratio.test],
                "seed": seed,
                "rebuild_all": plan.rebuild_all,
                "operation_video_ids": list(plan.reviewed_video_ids),
                "deduplicated_by_video": {
                    video_id: sorted(sample_ids)
                    for video_id, sample_ids in sorted(deduplicated_by_video.items())
                },
            }
        )
    if not report.committed:
        raise ReconstructionError("internal reconstruction did not produce every planned crop")

    samples = json.loads((out_dir / MANIFEST_NAME).read_text())["samples"]
    prior_by_id = {loader_sample_id(record): record for record in plan.prior.records}
    by_id = {loader_sample_id(record): record for record in samples}
    selected_after = {
        sample_id: record
        for sample_id, record in by_id.items()
        if record.get("video_id") in reviewed_set
    }
    n_new = sum(sample_id not in prior_by_id for sample_id in selected_after)

    # A re-review can de-accept a frame that was already shipped: it won't be
    # reconstructed this run, so it falls out of the rewritten manifest. Surface
    # that (don't silently shrink the dataset) — scoped to videos we re-exported.
    replaced_prior = {
        sample_id: record
        for sample_id, record in prior_by_id.items()
        if plan.rebuild_all or record.get("video_id") in reviewed_set
    }
    dropped = sorted(
        sample_id
        for sample_id in replaced_prior
        if sample_id not in by_id and sample_id not in collapsed_stems
    )
    if dropped:
        log(
            f"⚠ {len(dropped)} previously-shipped frame(s) no longer verified — "
            f"removed by the atomic dataset replacement:"
        )
        for sample_id in dropped:
            log(f"    {replaced_prior[sample_id].get('image', sample_id)}")

    for video_id in plan.reviewed_video_ids:
        count = sum(record.get("video_id") == video_id for record in samples)
        log(f"{video_id}: {count} exported frame(s)")

    SPLITS = tuple(s.value for s in Split)

    # Report frame *and diversity* counts: a split's eval is only as representative as
    # the number of distinct games/videos behind it (many shots of one game ≈ one sample).
    def _by_split(rows) -> dict[str, Diversity]:
        grouped: dict[str, list[dict]] = {s: [] for s in SPLITS}
        for r in rows:
            grouped[r["split"]].append(r)
        return {s: diversity(rs) for s, rs in grouped.items()}

    per_split = _by_split(samples)
    # `existing` is non-empty only on a true incremental append (a prior export exists
    # and --regroup wasn't passed, so old groups kept their split). There, report the
    # per-split delta this run *added*, not just the running totals.
    incremental = bool(plan.prior.records) and not regroup
    added = _by_split(
        [record for sample_id, record in by_id.items() if sample_id not in prior_by_id]
    )
    log(
        f"\nSLCC dataset -> {out_dir / MANIFEST_NAME}  "
        f"({n_new} new frame(s) split {ratio.train}/{ratio.val}/{ratio.test}; atomic commit)"
    )
    for s in SPLITS:
        delta = f" (+{added[s].n_frames:>3})" if incremental else ""
        gdelta = f" (+{added[s].n_games})" if incremental else ""
        log(f"  {s:<5} {per_split[s].n_frames:>4} frame(s){delta}  "
            f"{per_split[s].n_games:>3} game(s){gdelta}  {per_split[s].n_videos} video(s)")
    if plan.prior.records:
        before, after = diversity(list(plan.prior.records)), diversity(samples)
        log(f"  before {before.n_frames} frame(s) / {before.n_games} game(s) / "
            f"{before.n_videos} video(s)  ->  "
            f"after {after.n_frames} / {after.n_games} / {after.n_videos}")
    # A non-empty eval split carried by a single game is high-variance and narrow — it
    # measures "can the model read this one board", not generalization. Surface it.
    thin = [s for s in ("val", "test") if per_split[s].n_frames and per_split[s].n_games < 2]
    if thin:
        log(f"⚠ {', '.join(thin)} split(s) rest on a single game — low variation; add more "
            f"games/videos (or split by video) before trusting eval numbers there.")
    return out_dir / MANIFEST_NAME
