"""Human-in-the-loop template labeling.

Reuse-first: load the shared (video-independent) layout registry, classify a new
video's shots against it, and surface only the shots that match *no* known template
as candidate new templates. The human grades each (useful / hard / junk) and drags
its rects — board crop, the digital clock overlay, and separate white/black
nameplates; the result is appended to the shared registry so
the next broadcast reuses it. The Gradio UI is a thin shell over the testable
`prepare_labeling_batch` / `commit_new_layouts` functions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from chessqueries.annotate.classifier import JUNK_SUGGEST_THRESHOLD, JunkClassifier
from chessqueries.annotate.templates import (
    CLUSTER_MAX_DISTANCE,
    DEFAULT_REGISTRY_PATH,
    MIN_SIMILARITY,
    RECT_FIELDS,
    REGION_KINDS,
    Layout,
    LayoutRegistry,
    Quality,
    Rect,
    Shot,
    partition_templates,
)
from chessqueries.annotate.video import FrameReader, VideoFile

# Overlay regions (clock + nameplates) are broadcast furniture: the production puts
# them at a near-fixed screen position across almost every composition, so we offer
# a one-press prefill computed from what's already been labeled. The board crop is
# template-specific (varies with camera angle) and is never defaulted.
DEFAULT_REGION_KINDS = ("digital_clock", "white_name", "black_name")
_RECT_COLORS = {
    "board": (0, 255, 0),
    "digital_clock": (255, 128, 0),
    "white_name": (255, 255, 255),
    "black_name": (180, 0, 255),
}

# gradio is optional (viz group). It is imported lazily in build_app, but the name
# must live at module scope so Gradio can resolve the ``gr.SelectData`` event
# annotation via get_type_hints (PEP 563 stringifies it against module globals).
gr = None


def proposed_ids(registry: LayoutRegistry, n: int, prefix: str = "slcc_t") -> list[str]:
    """`n` fresh template ids not already in the registry."""
    out: list[str] = []
    i = 0
    while len(out) < n:
        candidate = f"{prefix}{len(registry.layouts) + i}"
        if candidate not in registry.layouts and candidate not in out:
            out.append(candidate)
        i += 1
    return out


def default_region_rects(registry: LayoutRegistry) -> dict[str, list[int]]:
    """Per-coordinate median rect for each overlay region across labeled templates.

    The clock/nameplate overlays sit at a near-fixed broadcast position, so the
    median over every template that already carries one is a good prefill (the
    median shrugs off the occasional off-composition outlier). Regions nobody has
    labeled yet are omitted; the board is excluded (it's template-specific)."""
    import statistics

    out: dict[str, list[int]] = {}
    for kind in DEFAULT_REGION_KINDS:
        rects = [getattr(lyt, f"{kind}_rect") for lyt in registry.layouts.values()]
        rects = [r for r in rects if r is not None]
        if rects:
            out[kind] = [round(statistics.median(r.as_list()[j] for r in rects)) for j in range(4)]
    return out


def overlay_rects(frame: np.ndarray, rects: dict[str, Rect]) -> np.ndarray:
    """Draw labeled rectangles on a copy of ``frame`` (for the labeling preview)."""
    canvas = frame.copy()
    for kind, rect in rects.items():
        if rect is None:
            continue
        color = _RECT_COLORS.get(kind, (255, 255, 255))
        cv2.rectangle(canvas, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), color, 2)
        cv2.putText(
            canvas,
            kind,
            (rect.x, max(0, rect.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
    return canvas


@dataclass(frozen=True)
class NewCluster:
    """A candidate new template: its proposed id, centroid, and sample images."""

    proposed_id: str
    centroid: list[float]
    n_shots: int
    keyframe_image: str  # full-frame keyframe (for drawing rects), relative to out_dir
    sample_images: list[str]  # montage thumbnails, relative to out_dir
    p_board: float | None = None  # classifier's suggested P(board); None if unscored


def _batch_label(videos: list[tuple[VideoFile, list[Shot], np.ndarray]]) -> str:
    return f"{len(videos)} video(s): " + ", ".join(vf.video_id for vf, _s, _d in videos)


def prepare_labeling_batch(
    videos: list[tuple[VideoFile, list[Shot], np.ndarray]],
    out_dir: Path,
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    min_similarity: float = MIN_SIMILARITY,
    max_distance: float = CLUSTER_MAX_DISTANCE,
    samples_per_cluster: int = 6,
    show_progress: bool = True,
) -> list[NewCluster]:
    """Pool every shot unmatched by the shared registry **across all videos** and
    cluster them jointly, so a composition seen in several broadcasts is one template
    to label, not many. Writes one queue (samples drawn from whichever videos a
    cluster spans) + ``session.json`` that `build_app` serves.

    ``videos`` is ``(VideoFile, shots, descriptors)`` per already-segmented video
    (produced by `pipeline.segment_and_fingerprint`)."""
    from contextlib import ExitStack

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = LayoutRegistry.load(registry_path)
    log = print if show_progress else (lambda *a, **k: None)
    descriptors_per_video = [d for _vf, _shots, d in videos]
    log(f"pooling new templates across {len(videos)} video(s)...")
    clusters = partition_templates(
        descriptors_per_video, registry, min_similarity=min_similarity, max_distance=max_distance
    ).new_clusters

    from tqdm import tqdm

    new_clusters: list[NewCluster] = []
    if clusters:
        ids = proposed_ids(registry, len(clusters))
        with ExitStack() as stack:
            readers = {
                vi: stack.enter_context(FrameReader(vf)) for vi, (vf, _s, _d) in enumerate(videos)
            }
            for cid, cluster in enumerate(
                tqdm(clusters, desc="extracting keyframes", unit="template", disable=not show_progress)
            ):
                tid = ids[cid]
                (out_dir / tid).mkdir(exist_ok=True)
                rep_vi, rep_si = max(
                    cluster.members,
                    key=lambda m: float(descriptors_per_video[m[0]][m[1]] @ cluster.centroid),
                )
                key_rel = f"{tid}/keyframe.jpg"
                cv2.imwrite(
                    str(out_dir / key_rel),
                    readers[rep_vi].frame_at_index(videos[rep_vi][1][rep_si].keyframe_index),
                )
                sample_rels = []
                for vi, si in cluster.members[:samples_per_cluster]:
                    shot = videos[vi][1][si]
                    rel = f"{tid}/v{vi}_shot{shot.index}.jpg"
                    cv2.imwrite(str(out_dir / rel), readers[vi].frame_at_index(shot.keyframe_index))
                    sample_rels.append(rel)
                new_clusters.append(
                    NewCluster(
                        proposed_id=tid,
                        centroid=[float(x) for x in cluster.centroid],
                        n_shots=len(cluster.members),
                        keyframe_image=key_rel,
                        sample_images=sample_rels,
                    )
                )
        log(f"{len(new_clusters)} new template(s) pooled across {len(videos)} video(s) -> {out_dir}")
    else:
        log("no new templates: every shot matches the shared registry")

    # Score board-first: surface the real boards at the top of the queue and pre-suggest
    # the junk tail (the labeler acts on `p_board`). Assist-only — the human still confirms
    # every cluster. Falls back to the clustered order when the registry can't train a
    # classifier (empty / single-class); the labeler then just shows no suggestion.
    clf = JunkClassifier.from_registry(registry)
    if clf is not None and new_clusters:
        new_clusters = sorted(
            (replace(nc, p_board=round(clf.score_board(nc.centroid), 4)) for nc in new_clusters),
            key=lambda nc: nc.p_board,
            reverse=True,
        )
        log(
            f"ranked board-first: {sum(nc.p_board >= 0.5 for nc in new_clusters)} likely-board, "
            f"{sum(nc.p_board < JUNK_SUGGEST_THRESHOLD for nc in new_clusters)} pre-suggested junk"
        )

    (out_dir / "session.json").write_text(
        json.dumps(
            {
                "video_id": _batch_label(videos),
                "registry_path": str(registry_path),
                "new_clusters": [nc.__dict__ for nc in new_clusters],
            },
            indent=2,
        )
    )
    return new_clusters


def commit_new_layouts(registry_path: Path, decisions: list[dict]) -> LayoutRegistry:
    """Append human-labeled templates to the shared registry and save it.

    Each decision: ``{id, centroid, quality, <rect fields>}`` where ``quality`` is
    ``"useful"``/``"hard"``/``"junk"`` and rects (``board_rect`` etc., see
    ``RECT_FIELDS``) are ``[x,y,w,h]`` lists or None. A decision with no/empty
    ``quality`` is deferred (not stored) so undecided clusters aren't lost as junk.

    Idempotent: re-committing an id (a second "Save all", or a long session whose
    earlier templates already reached the registry) overwrites in place rather
    than failing, so a single colliding id can't abort the whole save.
    """
    registry_path = Path(registry_path)
    registry = LayoutRegistry.load(registry_path)
    rect = lambda v: Rect.from_list(v) if v else None  # noqa: E731
    for d in decisions:
        if not d.get("quality"):
            continue  # undecided -> defer
        layout = Layout(
            id=str(d["id"]),
            quality=Quality(d["quality"]),
            **{f: rect(d.get(f)) for f in RECT_FIELDS},
        )
        registry.upsert(layout, np.asarray(d["centroid"], dtype=np.float32))
    registry.save(registry_path)
    return registry


def boxed_ungraded(decisions: list[dict]) -> list[int]:
    """Indices of decisions that have a region box drawn but aren't graded useful/hard.

    The save invariant: a drawn box means a processable board, so it must be useful or
    hard (never junk/undecided). Returned in decision order == queue order, so the
    caller can point the human at each offender's position (the board-first sort means
    ids aren't contiguous, so a position is what makes it findable)."""
    graded = (Quality.USEFUL.value, Quality.HARD.value)
    return [
        i
        for i, d in enumerate(decisions)
        if any(d.get(f"{k}_rect") for k in REGION_KINDS) and d.get("quality") not in graded
    ]


# Injected so keys (outside a text field) click buttons — grade and navigate
# templates without reaching for the mouse: A prev, W accept (useful), E prefill
# clock/name defaults, S save, D next, X discard-and-advance (junk + next).
_HOTKEYS = {
    "a": "prev-btn",
    "w": "accept-btn",
    "e": "add-default-btn",
    "s": "save-btn",
    "d": "next-btn",
    "x": "discard-btn",
}
_HOTKEYS_JS = f"""
<script>
var MAP = {json.dumps(_HOTKEYS)};
document.addEventListener('keydown', function (e) {{
  var tag = (document.activeElement && document.activeElement.tagName) || '';
  if (tag === 'INPUT' || tag === 'TEXTAREA') return;
  var id = MAP[e.key.toLowerCase()];
  if (!id) return;
  var btn = document.querySelector('#' + id + ' button') || document.getElementById(id);
  if (btn) btn.click();
}});
</script>
"""


def build_app(out_dir: Path, registry_path: Path = DEFAULT_REGISTRY_PATH):
    """Gradio app to label the new templates produced by `prepare_labeling_batch`.

    Pick a quality first (useful/hard/junk); junk disables region drawing. For a
    USEFUL/HARD template, choose a region and two-click its rect (top-left then
    bottom-right), or press ``E`` to prefill the clock/nameplate rects at their
    usual broadcast position (median of what's already labeled). Keyboard: ``A``
    prev, ``W`` accept (useful), ``E`` add default clock/name rects, ``S`` save,
    ``D`` next, ``X`` discard (junk) **and advance**. Save commits all decisions
    to the shared registry.
    """
    global gr
    if gr is None:
        try:
            import gradio
        except ImportError as e:  # gradio lives in the optional `viz` group
            raise ImportError("labeler UI needs gradio: `poetry install --with viz`") from e
        gr = gradio

    out_dir = Path(out_dir)
    session = json.loads((out_dir / "session.json").read_text())
    clusters = session["new_clusters"]

    # Prefill values for the clock/nameplate overlays, derived once from the
    # already-labeled templates (empty until at least one is labeled).
    registry = LayoutRegistry.load(registry_path)
    defaults = default_region_rects(registry)

    def keyframe(i: int) -> np.ndarray:
        bgr = cv2.imread(str(out_dir / clusters[i]["keyframe_image"]))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _suggested_quality(c: dict) -> str | None:
        # Pre-select "junk" only for confidently-junk clusters, so a real board is
        # never pre-graded away; everything else starts ungraded (human must decide).
        p = c.get("p_board")
        return Quality.JUNK.value if (p is not None and p < JUNK_SUGGEST_THRESHOLD) else None

    # decisions[i] accumulates the human's choices for cluster i.
    decisions = {
        i: {
            "id": c["proposed_id"],
            "centroid": c["centroid"],
            "quality": _suggested_quality(c),
            **{f: None for f in RECT_FIELDS},
        }
        for i, c in enumerate(clusters)
    }
    pending = {"corner": None}  # first click of a two-click rect

    def status_text(i: int) -> str:
        d = decisions[i]
        drawn = [k for k in REGION_KINDS if d[f"{k}_rect"]]
        p = clusters[i].get("p_board")
        hint = ""
        if p is not None:
            hint = f" — P(board)={p:.2f}"
            if p < JUNK_SUGGEST_THRESHOLD:
                hint += " ⟵ suggested junk"
        return (
            f"Cluster {i+1}/{len(clusters)} — id `{d['id']}` — "
            f"quality: {d['quality'] or '—'} — rects: {drawn}{hint}"
        )

    with gr.Blocks(title="SLCC template labeler", head=_HOTKEYS_JS) as demo:
        scored = any(c.get("p_board") is not None for c in clusters)
        order_note = " · sorted board-first (P(board) shown; low scores pre-suggested junk)" if scored else ""
        gr.Markdown(f"### {len(clusters)} new template(s) in `{session['video_id']}`{order_note}")
        idx = gr.State(0)
        # Quality is the first decision; junk turns the region picker off. Seeded from
        # the classifier's suggestion for cluster 0 (None unless confidently junk).
        quality = gr.Radio(
            [q.value for q in Quality],
            value=decisions[0]["quality"] if clusters else None,
            label="quality — useful (clean) / hard (tough angle → curate) / junk (discard)",
        )
        with gr.Row():
            accept_btn = gr.Button("Accept — useful (W)", elem_id="accept-btn")
            discard_btn = gr.Button("Discard — junk (X)", elem_id="discard-btn")
        region = gr.Radio(list(REGION_KINDS), value="board", label="Region to draw")
        with gr.Row():
            add_default_btn = gr.Button(
                "Add default clock/name rects (E)", elem_id="add-default-btn"
            )
            clear_btn = gr.Button("Clear selected region's rect")
        canvas = gr.Image(value=keyframe(0), label="click top-left then bottom-right", type="numpy")
        status = gr.Markdown(status_text(0))
        with gr.Row():
            prev_btn = gr.Button("Prev (A)", elem_id="prev-btn")
            next_btn = gr.Button("Next (D)", elem_id="next-btn")
            save_btn = gr.Button("Save all (S)", elem_id="save-btn")

        def render_canvas(i):
            d = decisions[i]
            rects = {k: Rect.from_list(d[f"{k}_rect"]) for k in REGION_KINDS if d[f"{k}_rect"]}
            return cv2.cvtColor(
                overlay_rects(cv2.cvtColor(keyframe(i), cv2.COLOR_RGB2BGR), rects),
                cv2.COLOR_BGR2RGB,
            )

        def render_nav(i):
            # full refresh on navigation: image, stored quality, status, and a region
            # reset back to "board" (interactive unless this template is junk).
            region_upd = gr.update(
                value="board", interactive=decisions[i]["quality"] != Quality.JUNK.value
            )
            return render_canvas(i), decisions[i]["quality"], status_text(i), region_upd

        def on_click(i, reg, evt: gr.SelectData):
            x, y = int(evt.index[0]), int(evt.index[1])
            if pending["corner"] is None:
                pending["corner"] = (x, y)
                return gr.update(), gr.update(), f"first corner ({x},{y}); click opposite corner"
            x0, y0 = pending["corner"]
            pending["corner"] = None
            decisions[i][f"{reg}_rect"] = [min(x0, x), min(y0, y), abs(x - x0), abs(y - y0)]
            # A completed box means this template shows a board, so lift a pre-suggested
            # "junk" (or blank) grade to useful — a box must never coexist with junk, or
            # the save guard trips on a template the human clearly meant to keep. Never
            # downgrade an explicit useful/hard.
            q_upd = gr.update()
            if decisions[i]["quality"] not in (Quality.USEFUL.value, Quality.HARD.value):
                decisions[i]["quality"] = Quality.USEFUL.value
                q_upd = gr.update(value=Quality.USEFUL.value)
            return render_canvas(i), q_upd, status_text(i)

        def step(i, quality_v, delta):
            decisions[i]["quality"] = quality_v
            j = max(0, min(len(clusters) - 1, i + delta))
            pending["corner"] = None
            return (j, *render_nav(j))

        def clear_region(i, reg):
            # undo a rect drawn by mistake (e.g. a name this template doesn't show).
            decisions[i][f"{reg}_rect"] = None
            pending["corner"] = None
            return render_canvas(i), status_text(i)

        def accept(i):
            # grade the current template "useful" and keep region drawing enabled.
            decisions[i]["quality"] = Quality.USEFUL.value
            return gr.update(value=Quality.USEFUL.value), gr.update(interactive=True)

        def discard_next(i):
            # grade junk and immediately advance — junk shots are the common case,
            # so X is a one-key "nope, next". Junk carries no board, so drop any boxes
            # drawn on it: keeps the box⇒board invariant so a discard can't leave a
            # boxed-but-junk template the save guard would then reject.
            decisions[i]["quality"] = Quality.JUNK.value
            for k in REGION_KINDS:
                decisions[i][f"{k}_rect"] = None
            j = max(0, min(len(clusters) - 1, i + 1))
            pending["corner"] = None
            return (j, *render_nav(j))

        def add_defaults(i):
            # one press fills the clock/nameplate rects at their usual broadcast
            # spot; only empties, so a hand-drawn box is never clobbered. No-op on
            # junk (those carry no rects) or before anything has been labeled.
            if decisions[i]["quality"] != Quality.JUNK.value:
                for kind, rect in defaults.items():
                    if not decisions[i][f"{kind}_rect"]:
                        decisions[i][f"{kind}_rect"] = list(rect)
            pending["corner"] = None
            return render_canvas(i), status_text(i)

        def save_all(i, quality_v):
            decisions[i]["quality"] = quality_v
            ordered = list(decisions.values())
            bad = boxed_ungraded(ordered)
            if bad:
                n = len(ordered)
                where = ", ".join(f"{ordered[k]['id']} (cluster {k + 1}/{n})" for k in bad)
                return (
                    "⚠ not saved — these have boxes but aren't useful/hard: "
                    f"{where}. Navigate there (A/D) and set their quality, or clear their boxes."
                )
            commit_new_layouts(registry_path, ordered)
            stored = sum(bool(d["quality"]) for d in ordered)
            return f"saved {stored} template(s) to {registry_path}"

        nav_out = [idx, canvas, quality, status, region]
        quality.change(
            lambda q: gr.update(interactive=q != Quality.JUNK.value), [quality], [region]
        )
        accept_btn.click(accept, [idx], [quality, region])
        discard_btn.click(discard_next, [idx], nav_out)
        add_default_btn.click(add_defaults, [idx], [canvas, status])
        clear_btn.click(clear_region, [idx, region], [canvas, status])
        canvas.select(on_click, [idx, region], [canvas, quality, status])
        prev_btn.click(lambda i, q: step(i, q, -1), [idx, quality], nav_out)
        next_btn.click(lambda i, q: step(i, q, +1), [idx, quality], nav_out)
        save_btn.click(save_all, [idx, quality], [status])

    return demo


def main(argv: list[str] | None = None) -> None:
    """CLI: `prepare` a video's new-template candidates, then `serve` the labeler."""
    import argparse

    from chessqueries.annotate import video as video_mod

    parser = argparse.ArgumentParser(description="SLCC template labeling")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="segment + cluster a video; emit new-template candidates")
    p.add_argument("--video-path", required=True, type=Path)
    p.add_argument("--video-id", required=True)
    p.add_argument("--format-id", default=video_mod.DEFAULT_FORMAT_ID)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--registry", default=DEFAULT_REGISTRY_PATH, type=Path)
    p.add_argument("--max-frames", default=None, type=int, help="only scan the first N frames")
    p.add_argument(
        "--frame-skip", default=0, type=int, help="skip N frames between checks (faster, coarser)"
    )

    s = sub.add_parser("serve", help="launch the Gradio labeler over prepared candidates")
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--registry", default=DEFAULT_REGISTRY_PATH, type=Path)

    args = parser.parse_args(argv)
    if args.cmd == "prepare":
        from chessqueries.annotate.pipeline import DESCRIPTORS_EXT, segment_and_fingerprint

        vf = video_mod.probe(args.video_path, args.video_id, args.format_id)
        fingerprinted = segment_and_fingerprint(
            vf,
            shots_cache=vf.path.with_suffix(".shots.json"),
            descriptors_cache=vf.path.with_suffix(DESCRIPTORS_EXT),
            max_frames=args.max_frames,
            frame_skip=args.frame_skip,
        )
        new = prepare_labeling_batch([(vf, fingerprinted.shots, fingerprinted.descriptors)],
                                     args.out, registry_path=args.registry)
        print(f"{len(new)} new template(s) -> {args.out}/session.json")
        for nc in new:
            print(f"  {nc.proposed_id}: {nc.n_shots} shots")
    elif args.cmd == "serve":
        build_app(args.out, args.registry).launch()


if __name__ == "__main__":
    main()
