"""Pure-logic tests for layout templates (no video needed)."""

import os

import numpy as np
import pytest

from chessqueries.annotate.templates import (
    Layout,
    LayoutRegistry,
    Quality,
    Rect,
    TemplateAssignment,
    cluster_descriptors,
    partition_templates,
)


def test_rect_validation_and_crop():
    with pytest.raises(ValueError):
        Rect(0, 0, 0, 10)
    with pytest.raises(ValueError):
        Rect(-1, 0, 10, 10)
    frame = np.arange(100, dtype=np.uint8).reshape(10, 10)
    crop = Rect(2, 3, 4, 5).crop(frame)
    assert crop.shape == (5, 4)
    assert Rect.from_list([1, 2, 3, 4]).as_list() == [1, 2, 3, 4]


def test_processable_layout_requires_board_rect():
    for q in (Quality.USEFUL, Quality.HARD):
        with pytest.raises(ValueError):
            Layout(id="x", quality=q)  # no board_rect
    Layout(id="x", quality=Quality.JUNK)  # junk needs no board_rect
    # round-trips with all region kinds: board crop, digital clock, split names.
    lay = Layout(
        id="x",
        quality=Quality.HARD,
        board_rect=Rect(0, 0, 8, 8),
        digital_clock_rect=Rect(1, 1, 2, 2),
        white_name_rect=Rect(5, 5, 4, 1),
        black_name_rect=Rect(5, 7, 4, 1),
    )
    assert Layout.from_dict(lay.to_dict()) == lay
    assert lay.processable and lay.requires_curation


def test_registry_roundtrip_and_classify(tmp_path):
    layouts = {
        "board": Layout("board", Quality.USEFUL, board_rect=Rect(0, 0, 8, 8)),
        "hard": Layout("hard", Quality.HARD, board_rect=Rect(0, 0, 8, 8)),
        "studio": Layout("studio", Quality.JUNK),
    }
    centroids = {"board": [1.0, 0.0], "hard": [0.7, 0.7], "studio": [0.0, 1.0]}
    reg = LayoutRegistry(layouts=layouts, centroids=centroids)
    path = tmp_path / "layouts.json"
    reg.save(path)
    back = LayoutRegistry.load(path)
    assert back.layouts == layouts
    processable = {k for k, v in back.layouts.items() if v.processable}
    assert processable == {"board", "hard"}  # JUNK excluded
    # a descriptor near the 'board' centroid classifies as board.
    match = back.classify(np.asarray([0.9, 0.1], dtype=np.float32))
    assert match.template_id == "board" and match.similarity > 0.5


def test_registry_rejects_mismatched_ids():
    with pytest.raises(ValueError):
        LayoutRegistry(layouts={"a": Layout("a", Quality.JUNK)}, centroids={"b": [1.0]})


def _reg(ids):
    return LayoutRegistry(
        layouts={i: Layout(i, Quality.JUNK) for i in ids},
        centroids={i: [1.0] for i in ids},
    )


def test_save_is_atomic_and_leaves_no_temp(tmp_path):
    path = tmp_path / "layouts.json"
    _reg(["a", "b"]).save(path)
    assert not path.with_name(path.name + ".tmp").exists()  # temp renamed away
    assert set(LayoutRegistry.load(path).layouts) == {"a", "b"}


def test_save_rotates_previous_generation_to_bak(tmp_path):
    path = tmp_path / "layouts.json"
    bak = path.with_name(path.name + ".bak")
    _reg(["v1"]).save(path)
    assert not bak.exists()  # nothing to rotate on first save
    _reg(["v2"]).save(path)  # second save rotates the first out
    assert set(LayoutRegistry.load(path).layouts) == {"v2"}
    assert set(LayoutRegistry.load(bak).layouts) == {"v1"}  # prior generation preserved


def test_load_tolerates_missing_or_empty_file(tmp_path):
    # A save that never completed once left a 0-byte file and crashed the whole CLI.
    assert LayoutRegistry.load(tmp_path / "absent.json").layouts == {}
    empty = tmp_path / "empty.json"
    empty.write_text("")
    assert LayoutRegistry.load(empty).layouts == {}
    empty.write_text("   \n")
    assert LayoutRegistry.load(empty).layouts == {}


def test_interrupted_save_cannot_destroy_existing_registry(tmp_path, monkeypatch):
    # Simulate a crash after the temp file is written but before the rename lands:
    # the live registry (and its .bak) must survive intact.
    path = tmp_path / "layouts.json"
    _reg(["good"]).save(path)
    calls = {"n": 0}
    real_replace = os.replace

    def boom(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:  # the path->.bak rotation; die right after, before tmp->path
            real_replace(src, dst)
            raise KeyboardInterrupt
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(KeyboardInterrupt):
        _reg(["new"]).save(path)
    # The good data is never lost: it's in .bak, and load() ignores the stray tmp.
    assert set(LayoutRegistry.load(path.with_name(path.name + ".bak")).layouts) == {"good"}


def test_registry_reuse_assign_and_append():
    """A shared registry classifies known templates and flags new ones for labeling."""
    reg = LayoutRegistry.empty()
    # empty registry -> everything is new
    assert reg.assign(np.asarray([[1.0, 0.0]], dtype=np.float32),
                      min_similarity=0.9) == TemplateAssignment([None], [0])
    reg.upsert(Layout("board", Quality.USEFUL, board_rect=Rect(0, 0, 8, 8)), np.asarray([1.0, 0.0]))
    descs = np.asarray([[0.99, 0.14], [0.0, 1.0]], dtype=np.float32)  # one match, one new
    assigned = reg.assign(descs, min_similarity=0.9)
    assert assigned.template_ids == ["board", None]
    assert assigned.unmatched == [1]


def test_assign_cache_invalidates_on_growth():
    """The cached centroid matrix must refresh when a template is added, or a newly
    saved layout would be invisible to matching (and re-surface every run)."""
    reg = LayoutRegistry.empty()
    reg.upsert(Layout("a", Quality.JUNK), np.asarray([1.0, 0.0]))
    d = np.asarray([[0.0, 1.0]], dtype=np.float32)
    # builds + caches the matrix
    assert reg.assign(d, min_similarity=0.9) == TemplateAssignment([None], [0])
    reg.upsert(Layout("b", Quality.JUNK), np.asarray([0.0, 1.0]))  # must invalidate the cache
    # now matches the new template
    assert reg.assign(d, min_similarity=0.9) == TemplateAssignment(["b"], [])
    # upsert (same id, moved centroid) must also refresh the cache
    reg.upsert(Layout("b", Quality.JUNK), np.asarray([1.0, 0.0]))
    assert reg.assign(d, min_similarity=0.9) == TemplateAssignment([None], [0])


def test_assign_accepts_single_1d_descriptor():
    reg = LayoutRegistry.empty()
    reg.upsert(Layout("a", Quality.JUNK), np.asarray([1.0, 0.0]))
    assert reg.assign(np.asarray([1.0, 0.0], dtype=np.float32),
                      min_similarity=0.9) == TemplateAssignment(["a"], [])


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_assign_absorbs_recurring_junk_in_the_gap():
    """A shot below min_similarity but near a *junk* template (its nearest) is absorbed
    when junk_similarity is set — so recurring known-junk stops re-surfacing. A shot
    nearest a real composition, or below junk_similarity, stays unmatched."""
    reg = LayoutRegistry.empty()
    reg.upsert(Layout("junk1", Quality.JUNK), _unit([1.0, 0.0, 0.0]))
    reg.upsert(Layout("useful1", Quality.USEFUL, board_rect=Rect(0, 0, 8, 8)), _unit([0.0, 1.0, 0.0]))

    # already unit, so cos to the axis template == the named component.
    shell = _unit([0.86, 0.5103, 0.0])  # nearest = junk1 at 0.86, in [0.80, 0.92)
    near_useful = _unit([0.5103, 0.86, 0.0])  # nearest = useful1 at 0.86, must NOT be absorbed
    far = _unit([0.0, 0.0, 1.0])  # nearest sim ~0, below junk_similarity

    descs = np.stack([shell, near_useful, far])
    # Without junk_similarity: all three are unmatched (strict).
    assert reg.assign(descs, min_similarity=0.92).unmatched == [0, 1, 2]
    # With it: only the junk shell is absorbed; the useful-nearest and far shots remain.
    assigned = reg.assign(descs, min_similarity=0.92, junk_similarity=0.80)
    assert assigned.template_ids[0] == "junk1"
    assert assigned.unmatched == [1, 2]


def test_assign_skips_blank_frames():
    """A uniform/black frame zero-means to the zero vector (cosine 0 to everything); it
    must be skipped, not surfaced as an unmatched template to label forever."""
    reg = LayoutRegistry.empty()
    reg.upsert(Layout("a", Quality.JUNK), _unit([1.0, 0.0]))
    descs = np.stack([_unit([1.0, 0.0]), np.zeros(2, dtype=np.float32), _unit([0.0, 1.0])])
    assigned = reg.assign(descs, min_similarity=0.9)
    assert assigned.template_ids == ["a", None, None]
    assert assigned.unmatched == [2]  # the blank (index 1) is skipped, not unmatched
    # also skipped with no templates yet (empty registry).
    assert LayoutRegistry.empty().assign(
        descs, min_similarity=0.9) == TemplateAssignment([None, None, None], [0, 2])


def test_partition_unifies_both_suppression_levels():
    """One pass: junk absorbed per-shot, a recurring useful composition dropped per-cluster,
    and BOTH leave the per-video unmatched tally — so neither re-surfaces nor blocks produce.
    Only the genuinely-new cluster is queued and counted."""
    dim = 16
    e = np.eye(dim, dtype=np.float32)
    reg = LayoutRegistry.empty()
    reg.upsert(Layout("U", Quality.USEFUL, board_rect=Rect(0, 0, 8, 8)), e[0])  # cropped composition
    reg.upsert(Layout("J", Quality.JUNK), e[5])  # no-crop composition

    rim = [0.9 * e[0] + 0.436 * e[k] for k in range(1, 7)]  # 6 useful-rim shots, mean -> U
    junk_gap = [0.86 * e[5] + 0.5103 * e[8]]  # nearest J at 0.86, in [0.80, 0.92) -> absorbed
    fresh = [0.97 * e[10] + 0.24 * e[k] for k in (11, 12, 13)]  # genuinely-new cluster
    descs = np.stack(rim + junk_gap + fresh).astype(np.float32)

    # Explicit thresholds: the synthetic descriptors' geometry is tuned to these, so the
    # test checks the two-level suppression logic independent of production tuning.
    part = partition_templates([descs], reg, min_similarity=0.92, max_distance=0.30)
    assert len(part.new_clusters) == 1  # only the genuinely-new composition surfaces
    assert reg.classify(part.new_clusters[0].centroid).similarity < 0.5
    # junk shot (per-shot) + 6 rim shots (per-cluster) suppressed; only the 3 fresh remain.
    assert part.unmatched_per_video == [3]


def test_partition_suppression_is_scope_dependent():
    """A composition's rim is recognized only once enough rim shots are pooled: a couple
    in one video can miss the cluster-mean bar, yet pooled across videos they clear it.
    This is why the produce gate must pool across all videos, exactly like the dashboard —
    a per-video count would skip a video the survey calls ready."""
    dim = 16
    e = np.eye(dim, dtype=np.float32)
    reg = LayoutRegistry.empty()
    reg.upsert(Layout("U", Quality.USEFUL, board_rect=Rect(0, 0, 8, 8)), e[0])
    vid_a = np.stack([0.9 * e[0] + 0.436 * e[1], 0.9 * e[0] + 0.436 * e[2]]).astype(np.float32)
    vid_b = np.stack([0.9 * e[0] + 0.436 * e[3], 0.9 * e[0] + 0.436 * e[4]]).astype(np.float32)
    # too few rim shots in one video -> mean misses 0.95 -> NOT suppressed
    kw = dict(min_similarity=0.92, max_distance=0.30)  # geometry tuned to these
    assert partition_templates([vid_a], reg, **kw).unmatched_per_video == [2]
    # pooled across videos -> mean clears 0.95 -> suppressed for both
    assert partition_templates([vid_a, vid_b], reg, **kw).unmatched_per_video == [0, 0]


def test_clustering_groups_similar_descriptors():
    rng = np.random.default_rng(0)
    dim = 64
    bases = [rng.standard_normal(dim) for _ in range(3)]
    bases = [b / np.linalg.norm(b) for b in bases]
    descs = []
    truth = []
    for k, b in enumerate(bases):
        for _ in range(5):
            v = b + 0.01 * rng.standard_normal(dim)
            descs.append(v / np.linalg.norm(v))
            truth.append(k)
    clustering = cluster_descriptors(np.asarray(descs), max_distance=0.30)
    assert len(clustering) == 3
    # same-truth descriptors share a label.
    for k in range(3):
        block = [lab for i, lab in enumerate(clustering.labels) if truth[i] == k]
        assert len(set(block)) == 1
