"""Board-recognition metrics.

Headline metric (comparable to the ChessReD paper) is **exact board accuracy**:
the fraction of boards with all 64 squares correct. We also report the tolerance
curve (boards with <= t wrong squares), occupancy/piece breakdowns, and the mean
number of wrong squares (Hamming distance — a human-facing "squares to fix").
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Sequence, Union

from chessqueries.core import Board, Piece

EMPTY = int(Piece.EMPTY)
Labels = Sequence[int]
BoardLike = Union[Board, Labels]


def _as_ints(board: BoardLike) -> list[int]:
    out = board.labels if isinstance(board, Board) else [int(x) for x in board]
    if len(out) != 64:
        raise ValueError(f"Expected 64 labels, got {len(out)}")
    return out


@dataclass
class BoardMetrics:
    """Per-board scoring primitives (counts, not rates)."""

    n_squares_correct: int          # out of 64
    n_occupancy_correct: int        # squares with correct empty/occupied status
    n_occupied_gt: int              # squares occupied in ground truth
    n_piece_correct_given_occupied: int  # of n_occupied_gt, class fully correct

    @property
    def n_wrong(self) -> int:
        return 64 - self.n_squares_correct

    @property
    def exact(self) -> bool:
        return self.n_squares_correct == 64

    def within(self, tolerance: int) -> bool:
        return self.n_wrong <= tolerance


def score_board(pred: BoardLike, gt: BoardLike) -> BoardMetrics:
    pred, gt = _as_ints(pred), _as_ints(gt)
    squares_correct = sum(p == g for p, g in zip(pred, gt))
    occ_correct = sum((p != EMPTY) == (g != EMPTY) for p, g in zip(pred, gt))
    occupied_gt = [(p, g) for p, g in zip(pred, gt) if g != EMPTY]
    piece_correct = sum(p == g for p, g in occupied_gt)
    return BoardMetrics(
        n_squares_correct=squares_correct,
        n_occupancy_correct=occ_correct,
        n_occupied_gt=len(occupied_gt),
        n_piece_correct_given_occupied=piece_correct,
    )


def count_wrong_squares(pred: BoardLike, gt: BoardLike) -> int:
    """Number of squares that differ = edits a human needs to fix the board."""
    return score_board(pred, gt).n_wrong


def aggregate(
    preds: Sequence[BoardLike],
    gts: Sequence[BoardLike],
    tolerances: Sequence[int] = (0, 1, 2, 3, 5, 10),
) -> dict:
    """Aggregate metrics over a set of boards.

    Returns a dict with:
        n_boards
        board_accuracy            (exact, == board_accuracy@0)
        board_accuracy@{t}        for each tolerance t
        per_square_accuracy       (micro, over all 64*N squares)
        occupancy_accuracy        (micro)
        piece_accuracy_given_occupied (micro, over occupied-in-GT squares)
        mean_wrong_squares        (mean wrong squares per board)
    """
    if len(preds) != len(gts):
        raise ValueError(f"{len(preds)} preds vs {len(gts)} gts")
    n = len(preds)
    if n == 0:
        raise ValueError("No boards to score")

    scores = [score_board(p, g) for p, g in zip(preds, gts)]

    tot_sq = sum(s.n_squares_correct for s in scores)
    tot_occ = sum(s.n_occupancy_correct for s in scores)
    tot_occupied = sum(s.n_occupied_gt for s in scores)
    tot_piece = sum(s.n_piece_correct_given_occupied for s in scores)
    tot_wrong = sum(s.n_wrong for s in scores)

    out = {
        "n_boards": n,
        "board_accuracy": sum(s.exact for s in scores) / n,
        "per_square_accuracy": tot_sq / (64 * n),
        "occupancy_accuracy": tot_occ / (64 * n),
        "piece_accuracy_given_occupied": (tot_piece / tot_occupied) if tot_occupied else float("nan"),
        "mean_wrong_squares": tot_wrong / n,
    }
    for t in tolerances:
        out[f"board_accuracy@{t}"] = sum(s.within(t) for s in scores) / n
    return out


def aggregate_subsets(
    preds: Sequence[BoardLike],
    gts: Sequence[BoardLike],
    group_keys: Sequence[Hashable],
    tolerances: Sequence[int] = (0, 1, 2, 3, 5, 10),
) -> dict:
    """Aggregate metrics overall and per subset, grouping boards by ``group_keys``.

    ``group_keys[i]`` is the (hashable) bucket label for board ``i`` — e.g. a
    "low"/"high" resolution band. Returns::

        {"overall": {<aggregate>}, "subsets": {<key>: {<aggregate>}, ...}}

    Subsets are keyed by their group label (sorted for stable output); each is a
    full ``aggregate`` over just its members, so the headline metric and any
    breakdown stay defined the same way.
    """
    if not (len(preds) == len(gts) == len(group_keys)):
        raise ValueError(f"{len(preds)} preds vs {len(gts)} gts vs {len(group_keys)} keys")

    by_group: dict[Hashable, list[int]] = defaultdict(list)
    for i, key in enumerate(group_keys):
        by_group[key].append(i)

    subsets = {
        key: aggregate([preds[i] for i in idx], [gts[i] for i in idx], tolerances)
        for key, idx in sorted(by_group.items())
    }
    return {"overall": aggregate(preds, gts, tolerances), "subsets": subsets}


class ErrorDecomposition:
    """Streaming, mutually-exclusive per-square error split over (B,64) label
    batches: **occupancy** errors (empty <-> piece) vs **piece-type** errors (both
    occupied, wrong piece). Also accumulates the per-square wrong counts for
    error-localization heatmaps. Accepts torch tensors or numpy arrays."""

    def __init__(self) -> None:
        self.occ_wrong = 0  # empty <-> piece confusions
        self.piece_wrong = 0  # both occupied, wrong piece
        self.occupied = 0  # truly-occupied squares seen
        self.n_boards = 0
        self.wrong_per_square = [0] * 64

    def update(self, preds, labels) -> None:
        pe, ge = preds == EMPTY, labels == EMPTY
        wrong = preds != labels
        self.occ_wrong += int((pe != ge).sum())
        self.piece_wrong += int((wrong & ~pe & ~ge).sum())
        self.occupied += int((~ge).sum())
        self.n_boards += len(labels)
        per_square = wrong.sum(0)
        for i in range(64):
            self.wrong_per_square[i] += int(per_square[i])

    @property
    def n_squares(self) -> int:
        return 64 * self.n_boards

    @property
    def occupancy_error(self) -> float:
        return self.occ_wrong / self.n_squares

    @property
    def piece_type_error(self) -> float:
        return self.piece_wrong / self.n_squares

    @property
    def per_square_error(self) -> float:
        return (self.occ_wrong + self.piece_wrong) / self.n_squares

    @property
    def piece_error_given_occupied(self) -> float:
        return self.piece_wrong / max(self.occupied, 1)

    @property
    def per_square_error_rates(self) -> list[float]:
        """(64,) any-error rate per square, for heatmaps."""
        return [w / self.n_boards for w in self.wrong_per_square]
