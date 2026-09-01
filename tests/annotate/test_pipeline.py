"""Shot triage for `produce`: which shots go to clock-OCR labeling, and the gate
counts for the rest. The labeling half of `produce_one` needs video+OCR and is
exercised end-to-end elsewhere."""

import numpy as np

from chessqueries.annotate.pipeline import triage_shots
from chessqueries.annotate.templates import Layout, LayoutRegistry, Quality, Rect, Shot


def _unit(i: int, dim: int = 4) -> list[float]:
    v = [0.0] * dim
    v[i] = 1.0
    return v


def _registry() -> LayoutRegistry:
    layouts = {
        "board": Layout("board", Quality.USEFUL, board_rect=Rect(0, 0, 8, 8),
                        digital_clock_rect=Rect(0, 8, 8, 2)),
        "hard": Layout("hard", Quality.HARD, board_rect=Rect(0, 0, 8, 8),
                       digital_clock_rect=Rect(0, 8, 8, 2)),
        "boardonly": Layout("boardonly", Quality.USEFUL, board_rect=Rect(0, 0, 8, 8)),
        "studio": Layout("studio", Quality.JUNK),
    }
    centroids = {tid: _unit(i) for i, tid in enumerate(layouts)}
    return LayoutRegistry(layouts=layouts, centroids=centroids)


def _shots(n: int) -> list[Shot]:
    return [Shot(index=i, start_frame=i * 100, end_frame=(i + 1) * 100) for i in range(n)]


def test_triage_routes_each_quality():
    reg = _registry()
    descriptors = np.array(
        [_unit(0), _unit(1), _unit(2), _unit(3), [0.5, 0.5, 0.5, 0.5]],
        dtype=np.float32,
    )  # board, hard, board-only, junk, below-threshold
    triage = triage_shots(reg, _shots(5), descriptors, min_similarity=0.85)

    assert [(t.index, t.template_id) for t in triage.tasks] == [(0, "board"), (1, "hard")]
    assert triage.tasks[0].shot.start_frame == 0  # carries the shot, not just its index
    assert triage.n_processable == 2
    assert triage.n_hard == 1
    assert triage.n_board_only == 1
    assert triage.n_unknown == 1  # junk is dropped without counting anywhere


def test_triage_empty_registry_counts_all_unknown():
    triage = triage_shots(LayoutRegistry.empty(), _shots(3), np.zeros((3, 4), dtype=np.float32))
    assert triage.tasks == []
    assert triage.n_unknown == 3
    assert triage.n_board_only == 0
