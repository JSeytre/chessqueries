"""Content-pin checks: an annotation's FEN must actually match the relay at the
ply it claims, and a game's labeled frames must progress by legal moves.

This is the guard the CVChess off-by-one taught us: a globally shifted labeling is
still a *coherent* game, so coherence alone can't catch it — every frame must be
anchored to the relay position it claims.
"""

from __future__ import annotations

import chess

from chessqueries.annotate.relay import GameTimeline
from chessqueries.annotate.schema import Annotation, AnnotationFile


def round_median_times(
    annotations: list[Annotation], *, min_confidence: float = 0.8
) -> dict[str, float]:
    """Each round's median video timestamp, from its confident frames. The median
    is robust to a few misattributed frames (unlike a min/max window)."""
    from statistics import median

    by_round: dict[str, list[float]] = {}
    for a in annotations:
        if a.confidence >= min_confidence:
            by_round.setdefault(a.round_id, []).append(a.timestamp_s)
    return {r: median(ts) for r, ts in by_round.items()}


def chronology_outliers(annotations: list[Annotation], *, min_confidence: float = 0.8) -> list[int]:
    """Indices of annotations that violate round chronology.

    Rounds are sequential in the broadcast (Round 2 after Round 1, blitz after rapid),
    so each occupies a contiguous time span. A frame is an outlier if its timestamp is
    nearest a *different* round's median time than the round it was attributed to —
    catching a same-matchup-different-round misID the clock alone might miss.
    """
    medians = round_median_times(annotations, min_confidence=min_confidence)
    if len(medians) < 2:
        return []
    out: list[int] = []
    for i, a in enumerate(annotations):
        nearest = min(medians, key=lambda r: abs(a.timestamp_s - medians[r]))
        if nearest != a.round_id:
            out.append(i)
    return out


def _legal_or_same(from_fen: str, to_placement: str) -> bool:
    if from_fen.split(" ")[0] == to_placement:
        return True
    board = chess.Board(from_fen)
    for mv in board.legal_moves:
        board.push(mv)
        reached = board.board_fen() == to_placement
        board.pop()
        if reached:
            return True
    return False


def check_annotations(
    annfile: AnnotationFile, timelines_by_game: dict[str, GameTimeline]
) -> list[str]:
    """Return a list of problems (empty == clean).

    Per annotation: the claimed ply's relay placement and side-to-move must equal
    the annotation. Per game: frames ordered by ply must be legal/identical
    transitions (catches a coherent global ply shift).
    """
    problems: list[str] = []
    for a in annfile.annotations:
        tl = timelines_by_game.get(a.game_id)
        if tl is None:
            problems.append(f"{a.game_id}@{a.frame_index}: no timeline for game")
            continue
        if not 0 <= a.ply < len(tl.positions):
            problems.append(f"{a.game_id}@{a.frame_index}: ply {a.ply} out of range")
            continue
        pos = tl.position_at(a.ply)
        if pos.placement != a.placement:
            problems.append(f"{a.game_id}@{a.frame_index}: placement != relay at ply {a.ply}")
        expected_side = pos.turn.fen
        if a.side_to_move != expected_side:
            problems.append(
                f"{a.game_id}@{a.frame_index}: side_to_move {a.side_to_move} != {expected_side}"
            )

    for game, group in annfile.by_game().items():
        ordered = sorted(group, key=lambda a: a.ply)
        for prev, nxt in zip(ordered, ordered[1:]):
            if prev.ply == nxt.ply:
                continue
            if not _legal_or_same(prev.fen, nxt.placement):
                problems.append(f"{game}: ply {prev.ply}->{nxt.ply} not a legal progression")
    return problems
