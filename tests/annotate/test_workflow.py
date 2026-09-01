"""Workflow state machine: survey from on-disk markers + cached descriptors, the
pending-action dashboard, and cross-video template pooling. All headless (no video,
no network) — markers are synthesized on disk."""

from pathlib import Path

import numpy as np
from annotate_fixtures import make_annotation

from chessqueries.annotate.manifest import Manifest, VideoEntry
from chessqueries.annotate.schema import Annotation, AnnotationFile
from chessqueries.annotate.templates import (
    Layout,
    LayoutRegistry,
    Quality,
    Rect,
    Shot,
    partition_templates,
    save_shots,
)
from chessqueries.annotate.workflow import (
    RecognizerKind,
    RecognizerRef,
    candidates_path,
    descriptors_path,
    ingest,
    pending_actions,
    pending_reviews,
    produce,
    render_status,
    review_queue_lines,
    reviewed_path,
    save_recognizer,
    shots_path,
    survey,
)

FMT = "137"


def _registry(path):
    """Two templates: one full (board+clock+names), one board-only; orthogonal centroids."""
    full = Layout(
        "t_full",
        Quality.USEFUL,
        board_rect=Rect(0, 0, 10, 10),
        digital_clock_rect=Rect(0, 0, 5, 5),
        white_name_rect=Rect(0, 0, 5, 5),
        black_name_rect=Rect(0, 0, 5, 5),
    )
    board = Layout("t_board", Quality.USEFUL, board_rect=Rect(0, 0, 10, 10))
    reg = LayoutRegistry(
        layouts={"t_full": full, "t_board": board},
        centroids={"t_full": [1.0, 0, 0, 0], "t_board": [0, 1.0, 0, 0]},
    )
    reg.save(path)
    return reg


def _download(data_dir, vid):
    (data_dir / f"{vid}.{FMT}.mp4").write_bytes(b"x")  # presence is all survey checks


def _segment(data_dir, vid, descriptors):
    save_shots(
        [Shot(i, i * 10, (i + 1) * 10) for i in range(len(descriptors))],
        shots_path(data_dir, vid, FMT),
    )
    np.save(descriptors_path(data_dir, vid, FMT), np.asarray(descriptors, dtype=np.float32))


FULL, BOARD, UNKNOWN = [1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]


def _manifest():
    e = lambda v, **k: VideoEntry(v, **k)  # noqa: E731
    return Manifest(
        path=None,
        entries={
            "vidregistr1": e("vidregistr1", tournaments=["T"]),  # nothing on disk
            "vidsegment1": e("vidsegment1", tournaments=["T"]),  # has an unlabeled template
            "vidready001": e("vidready001", tournaments=["T"]),  # ready to produce
            "vidnomapp01": e("vidnomapp01"),  # ready but no relay mapping
            "vidprodncd1": e("vidprodncd1", tournaments=["T"]),  # produced, not reviewed
        },
    )


def _build(tmp_path):
    data_dir = tmp_path / "slcc"
    data_dir.mkdir()
    reg_path = tmp_path / "registry.json"
    _registry(reg_path)
    _download(data_dir, "vidsegment1")
    _segment(data_dir, "vidsegment1", [FULL, BOARD, UNKNOWN])  # 1 board-only gap, 1 unmatched
    _download(data_dir, "vidready001")
    _segment(data_dir, "vidready001", [FULL, FULL])
    _download(data_dir, "vidnomapp01")
    _segment(data_dir, "vidnomapp01", [FULL, FULL])
    _download(data_dir, "vidprodncd1")
    _segment(data_dir, "vidprodncd1", [FULL])
    AnnotationFile(provenance={"stage": "candidates"}, annotations=[]).save(
        candidates_path(data_dir, "vidprodncd1")
    )
    return _manifest(), data_dir, reg_path


def test_survey_stages(tmp_path):
    manifest, data_dir, reg_path = _build(tmp_path)
    sv = survey(manifest, data_dir=data_dir, registry_path=reg_path, format_id=FMT)
    by_id = {s.video_id: s for s in sv.states}

    assert by_id["vidregistr1"].stage == "registered"
    assert by_id["vidsegment1"].stage == "needs-templates"
    assert (by_id["vidsegment1"].n_unmatched, by_id["vidsegment1"].n_board_only) == (1, 1)
    assert by_id["vidready001"].stage == "ready-to-produce"
    assert by_id["vidnomapp01"].stage == "needs-relay-map"
    assert by_id["vidprodncd1"].stage == "produced"


def test_pending_actions(tmp_path):
    manifest, data_dir, reg_path = _build(tmp_path)
    sv = survey(manifest, data_dir=data_dir, registry_path=reg_path, format_id=FMT)
    titles = " ".join(a.title for a in pending_actions(sv))

    assert "to ingest" in titles  # vidregistr1
    assert "new template" in titles  # vidsegment1's unmatched shot pooled
    assert "ready to auto-label" in titles  # vidready001
    assert "need a relay mapping" in titles  # vidnomapp01
    assert "to review" in titles  # vidprodncd1
    assert "board-only" in titles  # the gap


def test_render_status_is_a_string(tmp_path):
    manifest, data_dir, reg_path = _build(tmp_path)
    sv = survey(manifest, data_dir=data_dir, registry_path=reg_path, format_id=FMT)
    out = render_status(manifest, sv)
    assert "5 video(s)" in out
    assert "annotate ingest" in out


def test_render_status_shows_registry_and_coverage(tmp_path):
    """The dashboard surfaces the registry size and how close still-unlabeled videos
    are to unlocking produce (the gate is 0 unmatched shots)."""
    manifest, data_dir, reg_path = _build(tmp_path)
    sv = survey(manifest, data_dir=data_dir, registry_path=reg_path, format_id=FMT)
    assert sv.n_templates == 2  # _registry defines t_full + t_board
    out = render_status(manifest, sv)
    assert "registry: 2 labeled template(s)" in out
    # vidsegment1 is the lone needs-templates video: 1 unmatched of 3 shots (67% matched).
    assert "coverage: 1 video(s) not fully labeled · 1 shot(s) unmatched (67% matched)" in out
    assert "unlock produce" in out


def _ann(fi, *, verified=False):
    return make_annotation(frame_index=fi, verified_by_human=verified)


def test_partial_review_is_not_done(tmp_path):
    """A reviewed file with kept-but-unverified candidates is *partial*, not reviewed:
    it must still surface in `to_review` (and be resumable), not as releasable."""
    data_dir = tmp_path / "slcc"
    data_dir.mkdir()
    reg_path = tmp_path / "registry.json"
    _registry(reg_path)
    vid = "vidpartial1"
    _download(data_dir, vid)
    _segment(data_dir, vid, [FULL])
    # 3 candidates; reviewer accepted 1 (verified) and rejected 1 (dropped); 1 untouched.
    AnnotationFile(provenance={"stage": "candidates"}, annotations=[_ann(0), _ann(1), _ann(2)]).save(
        candidates_path(data_dir, vid)
    )
    AnnotationFile(
        provenance={"stage": "reviewed"}, annotations=[_ann(0, verified=True), _ann(2)]
    ).save(reviewed_path(data_dir, vid))

    manifest = Manifest(path=None, entries={vid: VideoEntry(vid, tournaments=["T"])})
    sv = survey(manifest, data_dir=data_dir, registry_path=reg_path, format_id=FMT)
    st = sv.states[0]
    assert st.review_started and not st.reviewed
    assert (st.n_verified, st.n_pending_review) == (1, 1)  # frame 2 still pending
    titles = " ".join(a.title for a in pending_actions(sv))
    assert "1 candidate(s) to review" in titles
    assert "video(s) reviewed" not in titles  # not releasable yet


def test_totals_count_accepted_pending_rejected(tmp_path):
    """survey tallies a reviewed file as accepted (verified) + pending (kept, unverified)
    + rejected (candidates dropped), and render_status surfaces the tally."""
    data_dir = tmp_path / "slcc"
    data_dir.mkdir()
    reg_path = tmp_path / "registry.json"
    _registry(reg_path)
    vid = "vidpartial2"
    _download(data_dir, vid)
    _segment(data_dir, vid, [FULL])
    # 4 candidates; review keeps 3 (1 verified, 2 untouched) and drops 1 -> 1 rejected.
    AnnotationFile(
        provenance={"stage": "candidates"},
        annotations=[_ann(0), _ann(1), _ann(2), _ann(3)],
    ).save(candidates_path(data_dir, vid))
    AnnotationFile(
        provenance={"stage": "reviewed"},
        annotations=[_ann(0, verified=True), _ann(1), _ann(2)],
    ).save(reviewed_path(data_dir, vid))

    manifest = Manifest(path=None, entries={vid: VideoEntry(vid, tournaments=["T"])})
    sv = survey(manifest, data_dir=data_dir, registry_path=reg_path, format_id=FMT)
    st = sv.states[0]
    assert (st.n_verified, st.n_pending_review, st.n_rejected) == (1, 2, 1)
    out = render_status(manifest, sv)
    assert "4 auto-labeled across 1 video(s)" in out
    assert "1 accepted · 2 pending · 1 rejected" in out


def _ann_vid(vid, fi, *, verified=False):
    """Like _ann but with an explicit (11-char) video id, so frame stems
    ``<vid>_<fi>`` line up with a dataset manifest's records."""
    a = _ann(fi, verified=verified)
    return Annotation.from_dict({**a.to_dict(), "video_id": vid})


def test_dashboard_reports_new_already_shipped_and_dropped(tmp_path):
    """The 📦 line reports what `reconstruct` would do: new frames to ship, how many
    are already in the dataset, and any previously-shipped frame a re-review now drops."""
    import json

    data_dir = tmp_path / "slcc"
    data_dir.mkdir()
    reg_path = tmp_path / "registry.json"
    _registry(reg_path)
    vid = "vidship0001"
    _download(data_dir, vid)
    _segment(data_dir, vid, [FULL])
    # Fully reviewed: 3 verified frames (0, 1, 2).
    AnnotationFile(
        provenance={"stage": "candidates"},
        annotations=[_ann_vid(vid, 0), _ann_vid(vid, 1), _ann_vid(vid, 2)],
    ).save(candidates_path(data_dir, vid))
    AnnotationFile(
        provenance={"stage": "reviewed"},
        annotations=[_ann_vid(vid, i, verified=True) for i in (0, 1, 2)],
    ).save(reviewed_path(data_dir, vid))
    # Dataset already holds frame 0 (still verified) + a stale frame 9 (re-review dropped it).
    ds = data_dir / "dataset"
    ds.mkdir()
    (ds / "annotations.json").write_text(json.dumps({"samples": [
        {"image": f"images/{vid}_0.jpg", "split": "train", "video_id": vid},
        {"image": f"images/{vid}_9.jpg", "split": "train", "video_id": vid},
    ]}))

    manifest = Manifest(path=None, entries={vid: VideoEntry(vid, tournaments=["T"])})
    sv = survey(manifest, data_dir=data_dir, registry_path=reg_path, format_id=FMT)
    assert sv.shipped_by_vid[vid] == frozenset({f"{vid}_0", f"{vid}_9"})
    pkg = next(a for a in pending_actions(sv) if a.icon == "📦")
    assert pkg.title == "2 new frame(s) to ship"  # frames 1, 2 (frame 0 already shipped)
    assert pkg.command == f"annotate reconstruct --video {vid}"
    assert "3 verified across 1 reviewed video(s)" in pkg.detail
    assert "1 already in the dataset" in pkg.detail
    assert "1 stale shipped frame(s) would be dropped" in pkg.detail


def test_ingest_skips_already_ingested(tmp_path):
    """A video with video + shots + descriptors caches is fully ingested: ingest reports
    it as skipped and never calls download (so a re-run can't spuriously fail on it)."""
    data_dir = tmp_path / "slcc"
    data_dir.mkdir()
    vid = "viddone0001"
    _download(data_dir, vid)
    _segment(data_dir, vid, [FULL])  # writes shots + descriptors caches
    manifest = Manifest(path=None, entries={vid: VideoEntry(vid, tournaments=["T"])})

    lines: list[str] = []
    ingest(manifest, data_dir=data_dir, format_id=FMT, log=lines.append)
    out = "\n".join(lines)
    assert "already ingested, skipping" in out
    assert "1 already done" in out
    assert "0 failed" in out


def test_recognizer_ref_command_and_roundtrip(tmp_path):
    """A checkpoint ref emits a runnable `--checkpoint … --resolution …` command and
    round-trips through disk; an adapter ref omits resolution; a checkpoint without a
    resolution is rejected at construction."""
    import pytest

    ckpt = RecognizerRef(RecognizerKind.CHECKPOINT, Path("checkpoints/v2/last.ckpt"), 644)
    assert ckpt.crosscheck_command() == (
        "annotate crosscheck --checkpoint checkpoints/v2/last.ckpt --resolution 644"
    )
    lora = RecognizerRef(RecognizerKind.ADAPTER, Path("checkpoints/lora/lora_adapter.pt"))
    assert lora.crosscheck_command() == (
        "annotate crosscheck --adapter checkpoints/lora/lora_adapter.pt"
    )

    p = tmp_path / "recognizer.json"
    ckpt.save(p)
    assert RecognizerRef.load(p) == ckpt

    with pytest.raises(ValueError):
        RecognizerRef(RecognizerKind.CHECKPOINT, Path("x.ckpt"))  # no resolution


def test_dashboard_suggests_remembered_recognizer(tmp_path):
    """With no recognizer remembered, the cross-check line is a placeholder; once one is
    saved, the dashboard suggests that exact, copy-pasteable command."""
    manifest, data_dir, reg_path = _build(tmp_path)  # vidprodncd1 is produced, not reviewed

    sv = survey(manifest, data_dir=data_dir, registry_path=reg_path, format_id=FMT)
    xcheck = next(a for a in pending_actions(sv) if a.icon == "🔬")
    assert "<v2.ckpt>" in xcheck.command  # placeholder until one is set

    ref = RecognizerRef(RecognizerKind.CHECKPOINT, Path("checkpoints/v2/last.ckpt"), 644)
    save_recognizer(ref, data_dir)
    sv = survey(manifest, data_dir=data_dir, registry_path=reg_path, format_id=FMT)
    xcheck = next(a for a in pending_actions(sv) if a.icon == "🔬")
    assert xcheck.command == ref.crosscheck_command()
    assert "last.ckpt" in xcheck.detail


def test_produce_salvage_skips_produced_but_unreviewed(tmp_path):
    """`produce --salvage` only revisits fully-reviewed videos: a produced-but-not-yet-
    reviewed one is skipped (with a note) before any video/OCR work, writing nothing."""
    manifest, data_dir, reg_path = _build(tmp_path)  # vidprodncd1: produced, no reviewed file
    lines: list[str] = []
    written = produce(
        manifest, data_dir=data_dir, registry_path=reg_path, format_id=FMT,
        video_ids=["vidprodncd1"], salvage=True, log=lines.append,
    )
    out = "\n".join(lines)
    assert "not reviewed" in out
    assert written == []


def test_produce_salvage_skips_unfinished_review(tmp_path):
    """A video mid-review (kept-but-unverified candidates) is not salvage-able yet."""
    data_dir = tmp_path / "slcc"
    data_dir.mkdir()
    reg_path = tmp_path / "registry.json"
    _registry(reg_path)
    vid = "vidmidrev01"
    _download(data_dir, vid)
    _segment(data_dir, vid, [FULL])
    AnnotationFile(provenance={"stage": "candidates"}, annotations=[_ann(0), _ann(1)]).save(
        candidates_path(data_dir, vid)
    )
    # frame 0 verified, frame 1 still pending -> review unfinished.
    AnnotationFile(
        provenance={"stage": "reviewed"}, annotations=[_ann(0, verified=True), _ann(1)]
    ).save(reviewed_path(data_dir, vid))

    manifest = Manifest(path=None, entries={vid: VideoEntry(vid, tournaments=["T"])})
    lines: list[str] = []
    written = produce(
        manifest, data_dir=data_dir, registry_path=reg_path, format_id=FMT,
        video_ids=[vid], salvage=True, log=lines.append,
    )
    assert "review unfinished (1 pending)" in "\n".join(lines)
    assert written == []


def test_pending_reviews_orders_resumable_first_with_queue_header(tmp_path):
    """`annotate review` walks a queue: resumable (partially-reviewed) videos come first,
    and the header names the position, how many remain, and the upcoming ids."""
    data_dir = tmp_path / "slcc"
    data_dir.mkdir()
    reg_path = tmp_path / "registry.json"
    _registry(reg_path)
    # A fresh produced video (no reviewed file) and a partially-reviewed one.
    _download(data_dir, "vidfresh001")
    _segment(data_dir, "vidfresh001", [FULL])
    AnnotationFile(provenance={"stage": "candidates"}, annotations=[_ann(0), _ann(1)]).save(
        candidates_path(data_dir, "vidfresh001")
    )
    _download(data_dir, "vidresume01")
    _segment(data_dir, "vidresume01", [FULL])
    AnnotationFile(
        provenance={"stage": "candidates"}, annotations=[_ann(0), _ann(1), _ann(2)]
    ).save(candidates_path(data_dir, "vidresume01"))
    AnnotationFile(
        provenance={"stage": "reviewed"}, annotations=[_ann(0, verified=True), _ann(2)]
    ).save(reviewed_path(data_dir, "vidresume01"))

    manifest = Manifest(path=None, entries={
        "vidfresh001": VideoEntry("vidfresh001", tournaments=["T"]),
        "vidresume01": VideoEntry("vidresume01", tournaments=["T"]),
    })
    sv = survey(manifest, data_dir=data_dir, registry_path=reg_path, format_id=FMT)
    q = pending_reviews(sv)
    assert [s.video_id for s in q] == ["vidresume01", "vidfresh001"]  # resumable first

    head1 = review_queue_lines(q, position=1)
    assert "reviewing 1/2 · 1 video(s) left after this" in head1
    assert "vidresume01  (1 of 3 candidate(s), resuming)" in head1
    assert "next: vidfresh001" in head1

    head2 = review_queue_lines(q, position=2)
    assert "reviewing 2/2 · 0 video(s) left after this" in head2
    assert "next: —" in head2


def test_partition_pools_new_templates_across_videos(tmp_path):
    reg_path = tmp_path / "registry.json"
    _registry(reg_path)
    registry = LayoutRegistry.load(reg_path)
    # An unknown composition appears in two different videos -> one shared cluster.
    video_a = np.asarray([FULL, UNKNOWN], dtype=np.float32)
    video_b = np.asarray([UNKNOWN], dtype=np.float32)
    part = partition_templates([video_a, video_b], registry)
    assert len(part.new_clusters) == 1
    assert set(part.new_clusters[0].members) == {(0, 1), (1, 0)}  # one member from each video
    # FULL matched, the two UNKNOWNs are genuinely new -> per-video unmatched tally.
    assert part.unmatched_per_video == [1, 1]
