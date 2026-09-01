"""Pure-logic tests for the template labeler (no video / no Gradio)."""

import numpy as np

from chessqueries.annotate.labeler import (
    boxed_ungraded,
    commit_new_layouts,
    default_region_rects,
    overlay_rects,
    proposed_ids,
)
from chessqueries.annotate.templates import RECT_FIELDS, Layout, LayoutRegistry, Quality, Rect


def test_proposed_ids_avoid_collisions():
    reg = LayoutRegistry.empty()
    assert proposed_ids(reg, 2) == ["slcc_t0", "slcc_t1"]
    reg.upsert(Layout("slcc_t0", Quality.JUNK), np.asarray([1.0]))
    ids = proposed_ids(reg, 2)
    assert "slcc_t0" not in ids and len(set(ids)) == 2


def test_default_region_rects_medians_and_excludes_board():
    reg = LayoutRegistry.empty()
    # three templates whose clock overlay clusters tightly (+ one off-position
    # outlier the median should ignore); names only on some.
    reg.upsert(Layout("a", Quality.USEFUL, board_rect=Rect(1, 1, 9, 9),
                   digital_clock_rect=Rect(10, 10, 4, 4), white_name_rect=Rect(2, 2, 2, 2)),
            np.asarray([1.0]))
    reg.upsert(Layout("b", Quality.USEFUL, board_rect=Rect(5, 5, 9, 9),
                   digital_clock_rect=Rect(12, 12, 6, 6), white_name_rect=Rect(4, 4, 2, 2)),
            np.asarray([0.0]))
    reg.upsert(Layout("c", Quality.USEFUL, board_rect=Rect(7, 7, 9, 9),
                   digital_clock_rect=Rect(200, 200, 8, 8)),  # outlier clock
            np.asarray([0.5]))
    d = default_region_rects(reg)
    assert d["digital_clock"] == [12, 12, 6, 6]  # median per coordinate, outlier shrugged off
    assert d["white_name"] == [3, 3, 2, 2]  # median of the two labeled
    assert "black_name" not in d  # never labeled -> omitted
    assert "board" not in d  # board is template-specific, never defaulted


def test_default_region_rects_empty_registry():
    assert default_region_rects(LayoutRegistry.empty()) == {}


def test_overlay_rects_draws_without_mutating_input():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = overlay_rects(frame, {"board": Rect(10, 10, 50, 50)})
    assert out.shape == frame.shape
    assert out.sum() > 0  # something drawn
    assert frame.sum() == 0  # original untouched


def _decision(id_, centroid, quality, **rects):
    base = {"id": id_, "centroid": centroid, "quality": quality}
    base.update(
        {
            k: None
            for k in (
                "board_rect",
                "digital_clock_rect",
                "white_name_rect",
                "black_name_rect",
            )
        }
    )
    base.update(rects)
    return base


def test_commit_stores_quality_and_defers_undecided(tmp_path):
    reg_path = tmp_path / "reg.json"
    LayoutRegistry.empty().save(reg_path)
    decisions = [
        _decision(
            "slcc_t0",
            [1.0, 0.0],
            "useful",
            board_rect=[0, 0, 8, 8],
            digital_clock_rect=[1, 1, 2, 2],
            white_name_rect=[5, 5, 4, 1],
        ),
        _decision("slcc_t1", [0.0, 1.0], "junk"),  # stored so it auto-discards later
        _decision("slcc_t2", [0.3, 0.7], None),  # undecided -> deferred
    ]
    commit_new_layouts(reg_path, decisions)
    reg = LayoutRegistry.load(reg_path)
    assert list(reg.layouts) == ["slcc_t0", "slcc_t1"]  # t2 deferred
    assert reg.layouts["slcc_t0"].quality == Quality.USEFUL
    assert reg.layouts["slcc_t0"].board_rect == Rect(0, 0, 8, 8)
    assert reg.layouts["slcc_t0"].digital_clock_rect == Rect(1, 1, 2, 2)
    assert reg.layouts["slcc_t1"].quality == Quality.JUNK
    # committing more appends (reuse across videos) rather than overwriting.
    commit_new_layouts(
        reg_path, [_decision("slcc_t3", [0.5, 0.5], "hard", board_rect=[1, 1, 4, 4])]
    )
    reg2 = LayoutRegistry.load(reg_path)
    assert list(reg2.layouts) == ["slcc_t0", "slcc_t1", "slcc_t3"]
    assert reg2.layouts["slcc_t3"].requires_curation


def test_commit_is_idempotent_on_existing_ids(tmp_path):
    # Re-saving a session whose early templates already landed in the registry
    # must not abort the whole save on the first collision (it used to raise).
    reg_path = tmp_path / "reg.json"
    LayoutRegistry.empty().save(reg_path)
    commit_new_layouts(reg_path, [_decision("slcc_t0", [1.0, 0.0], "junk")])
    decisions = [
        _decision("slcc_t0", [1.0, 0.0], "useful", board_rect=[0, 0, 8, 8]),  # already stored
        _decision("slcc_t1", [0.0, 1.0], "useful", board_rect=[1, 1, 4, 4]),  # new
    ]
    commit_new_layouts(reg_path, decisions)  # would have raised before the upsert fix
    reg = LayoutRegistry.load(reg_path)
    assert list(reg.layouts) == ["slcc_t0", "slcc_t1"]
    assert reg.layouts["slcc_t0"].quality == Quality.USEFUL  # overwritten in place
    assert reg.layouts["slcc_t0"].board_rect == Rect(0, 0, 8, 8)


# --- save invariant: a box must be graded useful/hard, never junk/undecided --------
def _dec(quality, *, box=False):
    d = {"id": "x", "quality": quality, **{f: None for f in RECT_FIELDS}}
    if box:
        d["board_rect"] = [0, 0, 10, 10]
    return d


def test_boxed_ungraded_flags_box_with_junk_or_no_grade():
    decs = [
        _dec("useful", box=True),  # ok
        _dec("hard", box=True),  # ok
        _dec("junk", box=True),  # BAD: box + junk (the bug that blocked saving)
        _dec(None, box=True),  # BAD: box + undecided
        _dec("junk"),  # ok: junk, no box
        _dec(None),  # ok: undecided, no box
    ]
    assert boxed_ungraded(decs) == [2, 3]  # positions, in queue order


def test_boxed_ungraded_detects_any_region_and_is_empty_when_clean():
    clock_on_junk = {"id": "x", "quality": "junk", **{f: None for f in RECT_FIELDS}}
    clock_on_junk["digital_clock_rect"] = [1, 1, 2, 2]  # non-board box still conflicts with junk
    assert boxed_ungraded([clock_on_junk]) == [0]
    assert boxed_ungraded([_dec("useful", box=True), _dec("junk"), _dec(None)]) == []
