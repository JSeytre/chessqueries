"""Relay PGN -> per-game timeline of (ply, FEN, clocks): the ground truth that
video frames are aligned against.

Broadcast relays (Lichess, chess.com, official) record every game of a round move
by move with ``[%clk H:MM:SS]`` comments. We parse those into one `GameTimeline`
per game; alignment (which board, which ply) lives in `identify`/`align`.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field, replace

import chess
import chess.pgn
import requests

from chessqueries.core import Board, Color

START_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

# Lichess broadcast round PGN export (public, no auth).
LICHESS_ROUND_PGN_URL = "https://lichess.org/api/broadcast/round/{round_id}.pgn"
# Broadcast tournament metadata (its list of rounds).
LICHESS_TOURNAMENT_URL = "https://lichess.org/api/broadcast/{tour_id}"
# Per-video raw-input metadata: which relay rounds each YouTube video covers, so
# the pipeline resolves rounds from the video id instead of being passed them.

_CLK_RE = re.compile(r"%clk\s+(\d+):(\d+):(\d+)")
_TC_BASE_RE = re.compile(r"^(\d+)(?:\+(\d+))?")


def parse_clock(comment: str) -> int | None:
    """Seconds remaining from a PGN ``[%clk H:MM:SS]`` comment, or None."""
    m = _CLK_RE.search(comment)
    if not m:
        return None
    h, mnt, s = (int(g) for g in m.groups())
    return h * 3600 + mnt * 60 + s


@dataclass(frozen=True)
class TimeControl:
    """A parsed PGN ``TimeControl`` header, in seconds. ``base_s`` is None when the
    header is missing or unparseable; the increment then reads 0."""

    base_s: int | None
    increment_s: int


def parse_time_control(tc: str) -> TimeControl:
    """``"25+10"`` -> 1500s base, 10s increment."""
    m = _TC_BASE_RE.match(tc or "")
    if not m:
        return TimeControl(base_s=None, increment_s=0)
    return TimeControl(base_s=int(m.group(1)) * 60,
                       increment_s=int(m.group(2)) if m.group(2) else 0)


@dataclass(frozen=True)
class Position:
    """One node in a game: the board after ``ply`` half-moves, with both clocks
    as they stand (the side that just moved from its ``%clk``, the other carried
    forward from its previous move)."""

    ply: int  # 0 = start position, 1 = after White's first move, ...
    fen: str  # full FEN
    placement: str  # FEN placement field only
    turn: Color  # side to move at this position
    last_san: str | None  # SAN of the move that produced this position
    white_clk_s: int | None
    black_clk_s: int | None

    def __post_init__(self) -> None:
        if self.ply < 0:
            raise ValueError(f"ply must be >= 0, got {self.ply}")
        Board.from_fen(self.placement)  # full structural validation (8 ranks x 8 squares)
        for clk in (self.white_clk_s, self.black_clk_s):
            if clk is not None and clk < 0:
                raise ValueError(f"clock seconds must be >= 0, got {clk}")


@dataclass(frozen=True)
class GameTimeline:
    """All positions of one relayed game, indexed contiguously from ply 0."""

    white: str
    black: str
    result: str
    time_control: str
    event: str
    round: str
    positions: list[Position] = field(default_factory=list)
    source_round_id: str | None = None
    increment_s: int = 0  # Fischer increment (s/move): 2 blitz, 10 rapid, 30 classical

    def __post_init__(self) -> None:
        if not self.positions:
            raise ValueError("GameTimeline needs at least the start position")
        if [p.ply for p in self.positions] != list(range(len(self.positions))):
            raise ValueError("positions must be contiguous from ply 0")
        if self.positions[0].placement != START_PLACEMENT:
            raise ValueError(
                f"first position must be the start placement, got {self.positions[0].placement!r}"
            )

    def position_at(self, ply: int) -> Position:
        return self.positions[ply]


def parse_round_pgn(pgn_text: str) -> list[GameTimeline]:
    """Parse a multi-game round PGN into one `GameTimeline` per game."""
    timelines: list[GameTimeline] = []
    stream = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        timelines.append(_timeline_from_game(game))
    return timelines


def _timeline_from_game(game: chess.pgn.Game) -> GameTimeline:
    h = game.headers
    time_control = parse_time_control(h.get("TimeControl", ""))

    board = game.board()
    positions = [
        Position(
            ply=0,
            fen=board.fen(),
            placement=board.board_fen(),
            turn=Color.WHITE,
            last_san=None,
            white_clk_s=time_control.base_s,
            black_clk_s=time_control.base_s,
        )
    ]
    white_clk = black_clk = time_control.base_s
    for i, node in enumerate(game.mainline(), start=1):
        san = board.san(node.move)
        board.push(node.move)
        mover_is_white = i % 2 == 1
        clk = parse_clock(node.comment)
        if clk is not None:
            if mover_is_white:
                white_clk = clk
            else:
                black_clk = clk
        positions.append(
            Position(
                ply=i,
                fen=board.fen(),
                placement=board.board_fen(),
                turn=Color.WHITE if board.turn else Color.BLACK,
                last_san=san,
                white_clk_s=white_clk,
                black_clk_s=black_clk,
            )
        )

    return GameTimeline(
        white=h.get("White", "?"),
        black=h.get("Black", "?"),
        result=h.get("Result", "*"),
        time_control=h.get("TimeControl", ""),
        event=h.get("Event", ""),
        round=h.get("Round", ""),
        positions=positions,
        source_round_id=None,
        increment_s=time_control.increment_s,
    )


def fetch_round_pgn(round_id: str) -> str:
    """Download a Lichess broadcast round PGN (all its games, with clocks)."""
    resp = requests.get(LICHESS_ROUND_PGN_URL.format(round_id=round_id), timeout=30)
    resp.raise_for_status()
    return resp.text


def load_round(round_id: str) -> list[GameTimeline]:
    """Fetch + parse a round, tagging each timeline with its source round id."""
    timelines = parse_round_pgn(fetch_round_pgn(round_id))
    return [replace(t, source_round_id=round_id) for t in timelines]


@dataclass(frozen=True)
class RoundInfo:
    id: str
    name: str
    starts_at_ms: int | None
    finished: bool


def tournament_rounds(tour_id: str) -> list[RoundInfo]:
    """List a broadcast tournament's rounds (auto-pull, no hand-copying round ids)."""
    resp = requests.get(LICHESS_TOURNAMENT_URL.format(tour_id=tour_id), timeout=30)
    resp.raise_for_status()
    rounds = resp.json().get("rounds", [])
    return [
        RoundInfo(r["id"], r.get("name", ""), r.get("startsAt"), bool(r.get("finished")))
        for r in rounds
    ]
