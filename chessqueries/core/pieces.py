"""Piece vocabulary: colors, piece types, and the 13-class `Piece` enum."""
from __future__ import annotations

from enum import Enum, IntEnum


class Color(Enum):
    WHITE = "white"
    BLACK = "black"

    @property
    def fen(self) -> str:
        """The FEN side-to-move char: ``"w"`` / ``"b"``."""
        return "w" if self is Color.WHITE else "b"


class PieceType(Enum):
    PAWN = "pawn"
    KNIGHT = "knight"
    BISHOP = "bishop"
    ROOK = "rook"
    QUEEN = "queen"
    KING = "king"

    @property
    def letter(self) -> str:
        return _TYPE_LETTER[self]


_TYPE_LETTER = {
    PieceType.PAWN: "P", PieceType.KNIGHT: "N", PieceType.BISHOP: "B",
    PieceType.ROOK: "R", PieceType.QUEEN: "Q", PieceType.KING: "K",
}
_ORDER = [PieceType.PAWN, PieceType.KNIGHT, PieceType.BISHOP,
          PieceType.ROOK, PieceType.QUEEN, PieceType.KING]


class Piece(IntEnum):
    """A square's content. Integer values are the model class ids (0=empty),
    ordered empty, white P/N/B/R/Q/K, black p/n/b/r/q/k — matching FEN symbols.
    """

    EMPTY = 0
    WHITE_PAWN = 1
    WHITE_KNIGHT = 2
    WHITE_BISHOP = 3
    WHITE_ROOK = 4
    WHITE_QUEEN = 5
    WHITE_KING = 6
    BLACK_PAWN = 7
    BLACK_KNIGHT = 8
    BLACK_BISHOP = 9
    BLACK_ROOK = 10
    BLACK_QUEEN = 11
    BLACK_KING = 12

    @property
    def is_empty(self) -> bool:
        return self is Piece.EMPTY

    @property
    def color(self) -> Color | None:
        if self.is_empty:
            return None
        return Color.WHITE if self.value <= 6 else Color.BLACK

    @property
    def piece_type(self) -> PieceType | None:
        if self.is_empty:
            return None
        return _ORDER[(self.value - 1) % 6]

    @property
    def symbol(self) -> str:
        if self.is_empty:
            return "."
        letter = self.piece_type.letter
        return letter if self.color is Color.WHITE else letter.lower()

    @classmethod
    def of(cls, color: Color, piece_type: PieceType) -> "Piece":
        base = 1 + _ORDER.index(piece_type)
        return cls(base + (0 if color is Color.WHITE else 6))

    @classmethod
    def from_symbol(cls, symbol: str) -> "Piece":
        try:
            return _SYMBOL_TO_PIECE[symbol]
        except KeyError:
            raise ValueError(f"Unknown piece symbol: {symbol!r}") from None


NUM_PIECES = len(Piece)  # 13
_SYMBOL_TO_PIECE = {p.symbol: p for p in Piece}
