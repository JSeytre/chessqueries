"""Render a `Board` to SVG via python-chess, optionally tinting wrong squares."""
from __future__ import annotations

from collections.abc import Iterable

import chess
import chess.svg

from chessqueries.core import Board, Square

# Flat, opaque tints: the same red/green whatever the square underneath, so a
# wrong square reads identically on light and dark squares. Mid-tone on purpose,
# so both black and white piece glyphs stay legible over them.
WRONG_FILL = "#ef5350"   # predicted square disagrees with ground truth
RIGHT_FILL = "#66bb6a"   # same squares highlighted on the ground-truth board


def _to_chess_board(board: Board) -> chess.Board:
    cb = chess.Board(None)  # empty
    cb.set_board_fen(board.placement)
    return cb


def board_svg(board: Board, wrong: Iterable[Square] | None = None, size: int = 360,
              fill_color: str = WRONG_FILL, flipped: bool = False) -> str:
    """SVG string for a board; squares in `wrong` are filled with `fill_color`.

    `flipped` views the board from Black's side. It is a *view* transform only —
    the position is unchanged, so the FEN of a flipped render is identical.
    """
    cb = _to_chess_board(board)
    fill = {chess.parse_square(sq.name): fill_color for sq in (wrong or ())}
    return chess.svg.board(cb, size=size, fill=fill, flipped=flipped)
