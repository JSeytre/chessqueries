"""Board-vs-junk k-NN scorer: fit guards, ranking, and batch scoring on synthetic
centroids (the real registry is local-only, so tests never depend on it)."""

import numpy as np

from chessqueries.annotate.classifier import JunkClassifier
from chessqueries.annotate.templates import LayoutRegistry, Layout, Quality, Rect


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _registry(board_dirs, junk_dirs) -> LayoutRegistry:
    reg = LayoutRegistry.empty()
    for i, d in enumerate(board_dirs):
        reg.upsert(Layout(id=f"b{i}", quality=Quality.USEFUL, board_rect=Rect(0, 0, 10, 10)), _unit(d))
    for i, d in enumerate(junk_dirs):
        reg.upsert(Layout(id=f"j{i}", quality=Quality.JUNK), _unit(d))
    return reg


def test_from_registry_needs_both_classes():
    assert JunkClassifier.from_registry(LayoutRegistry.empty()) is None
    only_junk = _registry([], [[0, 1, 0, 0], [0, 1, 0, 0.1]])
    assert JunkClassifier.from_registry(only_junk) is None
    only_board = _registry([[1, 0, 0, 0], [1, 0.1, 0, 0]], [])
    assert JunkClassifier.from_registry(only_board) is None


def test_ranks_board_above_junk():
    # boards cluster near +x, junk near +y; k small since few templates.
    reg = _registry(
        board_dirs=[[1, 0, 0, 0], [0.95, 0.1, 0, 0], [0.9, 0, 0.1, 0]],
        junk_dirs=[[0, 1, 0, 0], [0.1, 0.95, 0, 0], [0, 0.9, 0.1, 0]],
    )
    clf = JunkClassifier.from_registry(reg, k=3)
    p_board = clf.score_board(_unit([0.98, 0.05, 0, 0]))  # board-like query
    p_junk = clf.score_board(_unit([0.05, 0.98, 0, 0]))  # junk-like query
    assert p_board > 0.5 > p_junk
    assert 0.0 <= p_junk <= p_board <= 1.0


def test_k_clamps_to_registry_size_and_batch_matches_scalar():
    reg = _registry([[1, 0, 0, 0]], [[0, 1, 0, 0]])
    clf = JunkClassifier.from_registry(reg, k=99)  # k > N
    assert clf.k == 2
    queries = np.stack([_unit([1, 0.1, 0, 0]), _unit([0.1, 1, 0, 0])])
    batch = clf.score_many(queries)
    assert batch.shape == (2,)
    assert np.isclose(batch[0], clf.score_board(queries[0]))
    assert batch[0] > batch[1]  # first is board-like, second junk-like
