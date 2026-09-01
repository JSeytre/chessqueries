"""Parse a relay PGN with clocks into a per-game timeline (no network)."""


import chess

from chessqueries.annotate.relay import (
    START_PLACEMENT,
    parse_clock,
    parse_round_pgn,
    TimeControl,
    parse_time_control,
)
from chessqueries.core import Color

# Two-game fixture with [%clk] comments and a 25+10 time control. The clocks go
# *up* (increment exceeds time spent) — the non-monotonic case alignment must handle.
PGN = """[Event "Test Cup"]
[White "Alice"]
[Black "Bob"]
[Result "*"]
[TimeControl "25+10"]

1. e4 {[%clk 0:25:05]} e5 {[%clk 0:25:07]} 2. Nf3 {[%clk 0:25:12]} *

[Event "Test Cup"]
[White "Carol"]
[Black "Dave"]
[Result "1-0"]
[TimeControl "25+10"]

1. d4 {[%clk 0:25:03]} d5 {[%clk 0:25:04]} 1-0
"""


def test_clock_and_tc_parsing():
    assert parse_clock("[%eval 0.1] [%clk 0:25:17]") == 25 * 60 + 17
    assert parse_clock("no clock here") is None
    assert parse_time_control("25+10") == TimeControl(base_s=1500, increment_s=10)
    assert parse_time_control("") == TimeControl(base_s=None, increment_s=0)


def test_round_splits_into_games():
    games = parse_round_pgn(PGN)
    assert len(games) == 2
    assert (games[0].white, games[0].black) == ("Alice", "Bob")
    assert (games[1].white, games[1].black) == ("Carol", "Dave")


def test_timeline_invariants_and_clock_carry():
    g = parse_round_pgn(PGN)[0]
    # ply 0..3 contiguous, start placement pinned (else __post_init__ raises).
    assert [p.ply for p in g.positions] == [0, 1, 2, 3]
    assert g.positions[0].placement == START_PLACEMENT
    # clocks: mover's from %clk, opponent carried forward.
    assert (g.positions[0].white_clk_s, g.positions[0].black_clk_s) == (1500, 1500)
    assert (g.positions[1].white_clk_s, g.positions[1].black_clk_s) == (1505, 1500)  # White moved
    assert (g.positions[2].white_clk_s, g.positions[2].black_clk_s) == (1505, 1507)  # Black moved
    assert (g.positions[3].white_clk_s, g.positions[3].black_clk_s) == (1512, 1507)  # White moved
    # turn alternates W after even plies.
    assert [p.turn for p in g.positions] == [Color.WHITE, Color.BLACK, Color.WHITE, Color.BLACK]


def _one_move_apart(from_fen: str, to_placement: str) -> bool:
    board = chess.Board(from_fen)
    for mv in board.legal_moves:
        board.push(mv)
        reached = board.board_fen() == to_placement
        board.pop()
        if reached:
            return True
    return False


def test_transitions_are_legal_moves():
    """Every consecutive placement is one legal move apart (the alignment backbone)."""
    g = parse_round_pgn(PGN)[0]
    for a, b in zip(g.positions, g.positions[1:]):
        assert _one_move_apart(a.fen, b.placement), f"ply {a.ply}->{b.ply} not a legal transition"
