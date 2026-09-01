"""Identify which board (game) and ply a useful-template frame shows, from the
on-screen overlay text — never the analysis board (commentators move its pieces).

Two cues, cross-checked against the round's relay timelines (`relay.GameTimeline`):
- player names on the nameplate -> which of the 5+ simultaneous games;
- the clock pair -> which ply. The *static* (non-ticking) clock equals an exact
  recorded ``%clk``; the *running* clock has only ticked down since its last
  ``%clk``. Increment-safe, because ``%clk`` already bakes the increment in.

The matching core is pure (testable against real timelines); the OCR reader is a thin
lazy wrapper over the local engine (which engine, and the benchmark behind it, lives in
``ocr_bench``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from chessqueries.annotate.relay import GameTimeline
from chessqueries.core import Board, Color

_CLK_TEXT_RE = re.compile(r"(\d{1,2})[:.](\d{2})(?:[:.](\d{2}))?")


def surname(full_name: str) -> str:
    """Broadcast nameplate key from a relay name: ``"Niemann, Hans Moke" -> "niemann"``,
    ``"Vachier-Lagrave, Maxime" -> "vachier-lagrave"``, ``"Gukesh D" -> "gukesh"``."""
    head = full_name.split(",")[0].strip() if "," in full_name else full_name.strip()
    tokens = head.split()
    # No comma: prefer the longest token (drops single-letter initials like "D").
    pick = head if "," in full_name else max(tokens, key=len, default=head)
    return pick.lower()


def parse_clock_text(text: str) -> int | None:
    """Seconds from an OCR'd clock like ``"25:17"``, ``"1:23"``, ``"0:25:17"``.

    Tolerates ``.`` misread for ``:``. Returns None if no clock-like token found.
    """
    m = _CLK_TEXT_RE.search(text.replace(" ", ""))
    if not m:
        return None
    a, b, c = m.groups()
    if c is None:  # M:SS
        return int(a) * 60 + int(b)
    return int(a) * 3600 + int(b) * 60 + int(c)  # H:MM:SS


@dataclass(frozen=True)
class Nameplates:
    """OCR'd nameplate text, kept split by region so White/Black orientation survives.

    The fixed overlay puts the White player on the White nameplate (left) and Black on
    the Black nameplate (right) — the same screen order the clock pair is trusted in. We
    keep the two regions apart instead of one blob so a pair who meet twice with reversed
    colours (a blitz double-round) can be told apart by who is on which nameplate.
    """

    white: tuple[str, ...] = ()
    black: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.white or self.black)

    @property
    def white_blob(self) -> str:
        return " ".join(self.white).lower()

    @property
    def black_blob(self) -> str:
        return " ".join(self.black).lower()

    @property
    def blob(self) -> str:
        return " ".join((*self.white, *self.black)).lower()


def name_in_blob(name: str, blob: str) -> bool:
    """Whether a relay surname is present in an OCR'd nameplate blob, tolerant of OCR slop.

    Exact substring first; otherwise any whitespace token that contains, or is contained
    by, the surname differing by at most two chars — so a clipped leading letter
    (``"achier-lagrave"`` for ``"vachier-lagrave"``, the rect cut off the ``V``) or a
    trailing smudge still matches. Names under four chars (``"so"``, ``"li"``) demand an
    exact token, since a loose substring would match far too much.
    """
    if name in blob:
        return True
    for tok in blob.split():
        short, long = sorted((name, tok), key=len)
        if len(short) < 4:
            if name == tok:
                return True
        elif short in long and len(long) - len(short) <= 2:
            return True
    return False


def match_games_by_name(nameplates: Nameplates, timelines: list[GameTimeline]) -> list[int]:
    """Game indices whose players match the OCR'd nameplates.

    Orientation-aware: the White nameplate names White and the Black nameplate names
    Black, so a game matches only when its White surname is on the White plate *and* its
    Black surname on the Black plate. That is what separates the two meetings of a pair
    who play twice with reversed colours — both have the same surnames, only the
    orientation differs. Falls back to unordered membership (either surname anywhere)
    when orientation can't decide: a region failed to OCR, or both names landed in one.
    Surname presence is matched leniently (`name_in_blob`) so a single OCR'd character
    dropped from a long name doesn't silently knock the right game out of contention.
    """
    oriented = [
        i
        for i, t in enumerate(timelines)
        if name_in_blob(surname(t.white), nameplates.white_blob)
        and name_in_blob(surname(t.black), nameplates.black_blob)
    ]
    if oriented:
        return oriented
    blob = nameplates.blob
    return [
        i
        for i, t in enumerate(timelines)
        if name_in_blob(surname(t.white), blob) and name_in_blob(surname(t.black), blob)
    ]


@dataclass(frozen=True)
class PlyMatch:
    ply: int
    static_side: Color  # the side whose clock matched a %clk exactly
    residual: float  # |observed - recorded| seconds on the static side


def candidate_plies(
    timeline: GameTimeline, white_obs: int, black_obs: int, *, tol_exact: int = 2, tol_run: int = 3
) -> list[PlyMatch]:
    """Plies consistent with an observed (white, black) clock pair, best first.

    The on-screen clock order is trusted: ``white_obs`` is White's clock and
    ``black_obs`` Black's (callers read them left-to-right from the fixed overlay). At
    each position the side that just moved is **static** — its clock equals the recorded
    ``%clk`` (post-increment), within ``tol_exact``. The side to move (``pos.turn``) is
    **running**: its live clock sits between its next recorded value minus the increment
    (the instant before the next press) and its current value (it has only ticked down),
    within ``tol_run``. That two-sided, increment-aware bound is what separates plies in
    time scramble, where one player's clock hovers near the increment for many moves.
    """
    inc = timeline.increment_s
    positions = timeline.positions
    n = len(positions)
    out: list[PlyMatch] = []
    for pos in positions:
        w, b = pos.white_clk_s, pos.black_clk_s
        if w is None or b is None:
            continue
        white_running = pos.turn == Color.WHITE
        if white_running:
            static_side, static_obs, static_rec = Color.BLACK, black_obs, b
            run_obs, run_now = white_obs, w
            nxt = positions[pos.ply + 1].white_clk_s if pos.ply + 1 < n else None
        else:
            static_side, static_obs, static_rec = Color.WHITE, white_obs, w
            run_obs, run_now = black_obs, b
            nxt = positions[pos.ply + 1].black_clk_s if pos.ply + 1 < n else None
        if abs(static_obs - static_rec) > tol_exact:
            continue
        # Running clock window: [next_recorded - increment, current], padded by tol_run.
        # No next move (or a missing %clk) -> only the upper bound applies (floor at 0).
        lower = (nxt - inc) if nxt is not None else 0
        if not (lower - tol_run <= run_obs <= run_now + tol_run):
            continue
        out.append(PlyMatch(pos.ply, static_side, abs(static_obs - static_rec)))
    out.sort(key=lambda m: m.residual)
    return out


def candidate_window(
    timeline: GameTimeline,
    matched_ply: int,
    clocks: tuple[int | None, int | None],
    *,
    desync: int = 3,
    tol_exact: int = 2,
    tol_run: int = 3,
) -> list[int]:
    """The plies to score visually for one frame: the clock-consistent set (a whole
    scramble cluster, since increment makes many plies share a clock) padded by
    ``+/-desync`` to absorb a broadcast-clock that lags the camera. Returns sorted,
    unique, in-range plies — the search space the recognizer ranks the image against.

    Empirically the true ply sits within +/-3 of the clock match (broadcast desync),
    so a small ``desync`` keeps the window tight while still bracketing the truth.
    """
    n = len(timeline.positions)
    anchors = {matched_ply}
    if clocks[0] is not None and clocks[1] is not None:
        anchors |= {
            m.ply
            for m in candidate_plies(
                timeline, clocks[0], clocks[1], tol_exact=tol_exact, tol_run=tol_run
            )
        }
    window: set[int] = set()
    for p in anchors:
        window.update(range(max(0, p - desync), min(n - 1, p + desync) + 1))
    return sorted(window)


@dataclass(frozen=True)
class Identification:
    game_index: int  # index into the round's timelines list
    ply: int
    fen: str
    placement: str
    white: str
    black: str
    static_side: Color
    residual: float
    name_matched: bool
    confidence: float


def identify_position(
    timelines: list[GameTimeline],
    clocks: tuple[int, int],
    names: Nameplates | None = None,
    *,
    tol_exact: int = 2,
    tol_run: int = 3,
) -> Identification | None:
    """Best (game, ply) for an observed (white, black) clock pair + optional nameplates.

    The clock pair is in screen order (White, Black) — the fixed overlay layout makes
    that reliable, and trusting it avoids the phantom matches a both-orderings search
    invites. Restricts to name-matched games when names disambiguate; the orientation in
    ``names`` separates a pair who meet twice with reversed colours. Returns None if
    nothing matches cleanly.

    Crucial guard: if a nameplate *was* read but matches no game, we do **not** fall back
    to clock-only across every game/round — that is how a frame gets a confident-looking
    label for the wrong players entirely (a coincidental clock collision). We only let
    the clock pick the game blind when there were no names to read at all (a board+clock
    layout). A read-but-unmatched nameplate -> None (leave it for the model/human).
    """
    if names:
        candidate_indices = match_games_by_name(names, timelines)
        if not candidate_indices:
            return None  # nameplate read but matched nothing -> don't guess from the clock
        name_matched = True
    else:
        candidate_indices = list(range(len(timelines)))
        name_matched = False
    white_obs, black_obs = clocks

    best: Identification | None = None
    for gi in candidate_indices:
        t = timelines[gi]
        matches = candidate_plies(t, white_obs, black_obs, tol_exact=tol_exact, tol_run=tol_run)
        if not matches:
            continue
        m = matches[0]  # sorted; first (lowest residual) is enough
        pos = t.position_at(m.ply)
        conf = _confidence(m.residual, name_matched=name_matched, tol_exact=tol_exact)
        if best is None or conf > best.confidence:
            best = Identification(
                game_index=gi,
                ply=m.ply,
                fen=pos.fen,
                placement=pos.placement,
                white=t.white,
                black=t.black,
                static_side=m.static_side,
                residual=m.residual,
                name_matched=name_matched,
                confidence=conf,
            )
    return best


def _confidence(residual: float, *, name_matched: bool, tol_exact: int) -> float:
    clock_score = max(0.0, 1.0 - residual / max(tol_exact, 1))
    return 0.5 * clock_score + 0.5 if name_matched else 0.5 * clock_score


@dataclass(frozen=True)
class BoardMatch:
    """A board-content match into the relay: the nearest legal position and how far
    off the (noisy) prediction was, in squares."""

    game_index: int
    ply: int
    diff: int  # number of squares differing from the relay position
    fen: str
    placement: str
    confidence: float


def match_board_to_relay(
    timelines: list[GameTimeline], placement: str, *, max_diff: int = 4
) -> BoardMatch | None:
    """Nearest relay position to a predicted board placement, across all games.

    The label stays the *relay* FEN (authoritative); a recognizer's prediction is
    only the retrieval key. Used for board-only HARD frames with no clock/name to
    OCR. Returns None if even the closest position differs by more than ``max_diff``
    squares (too unsure -> leave for a human). ``game_index`` + ``ply`` identify
    both which of the simultaneous boards and the move.
    """
    predicted = Board.from_fen(placement)
    best: BoardMatch | None = None
    for gi, t in enumerate(timelines):
        for pos in t.positions:
            diff = len(predicted.diff(Board.from_fen(pos.placement)))
            if best is None or diff < best.diff:
                best = BoardMatch(
                    game_index=gi,
                    ply=pos.ply,
                    diff=diff,
                    fen=pos.fen,
                    placement=pos.placement,
                    confidence=max(0.0, 1.0 - diff / (max_diff + 1)),
                )
                if diff == 0:
                    return best
    if best is None or best.diff > max_diff:
        return None
    return best


class OcrReader:
    """Reads clocks + nameplates as plain strings via the benchmark-winning engine
    from ``ocr_bench`` (PaddleOCR). The heavy engine loads at construction — build
    one reader and reuse it across a video."""

    def __init__(self, lang: str = "en") -> None:
        from chessqueries.annotate.ocr_bench import PaddleOcrEngine

        self._engine = PaddleOcrEngine(lang=lang)

    def _detect(self, image_bgr: np.ndarray) -> list[tuple[float, str]]:
        """``(x_left, text)`` for every detected box, ordered left-to-right by x.

        Left-to-right order makes the first clock White's and the second Black's,
        matching the fixed broadcast overlay (White nameplate on the left)."""
        return sorted(self._engine.detect(image_bgr), key=lambda d: d[0])

    def texts(self, image_bgr: np.ndarray) -> list[str]:
        return [text for _x, text in self._detect(image_bgr)]

    def clocks_in(self, image_bgr: np.ndarray) -> list[int]:
        """Clock-like values (seconds) in a region, ordered left-to-right by x."""
        return [c for _x, t in self._detect(image_bgr) if (c := parse_clock_text(t)) is not None]
