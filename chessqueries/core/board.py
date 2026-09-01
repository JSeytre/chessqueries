"""The `Board` value object: 64 validated squares in FEN order.

The single representation shared across datasets, models, metrics, and viz.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from chessqueries.core.pieces import Piece
from chessqueries.core.square import ALL_SQUARES, Square

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class Board:
    pieces: tuple[Piece, ...]

    def __post_init__(self) -> None:
        if len(self.pieces) != 64:
            raise ValueError(f"Board needs exactly 64 squares, got {len(self.pieces)}")
        if not all(isinstance(p, Piece) for p in self.pieces):
            raise TypeError("Board pieces must all be Piece instances")

    # -- constructors --------------------------------------------------------
    @classmethod
    def from_labels(cls, labels: Sequence[int | Piece]) -> "Board":
        return cls(tuple(p if isinstance(p, Piece) else Piece(int(p)) for p in labels))

    @classmethod
    def from_tensor(cls, tensor: "torch.Tensor") -> "Board":
        return cls.from_labels(tensor.tolist())

    @classmethod
    def from_fen(cls, fen: str) -> "Board":
        placement = fen.strip().split(" ")[0]
        rows = placement.split("/")
        if len(rows) != 8:
            raise ValueError(f"FEN placement needs 8 ranks, got {len(rows)}: {placement!r}")
        pieces: list[Piece] = []
        for row in rows:
            rank: list[Piece] = []
            for ch in row:
                if ch.isdigit():
                    rank.extend([Piece.EMPTY] * int(ch))
                else:
                    rank.append(Piece.from_symbol(ch))
            if len(rank) != 8:
                raise ValueError(f"FEN rank needs 8 squares, got {len(rank)}: {row!r}")
            pieces.extend(rank)
        return cls(tuple(pieces))

    @classmethod
    def empty(cls) -> "Board":
        return cls((Piece.EMPTY,) * 64)

    # -- accessors -----------------------------------------------------------
    @property
    def labels(self) -> list[int]:
        return [int(p) for p in self.pieces]

    def piece_at(self, square: Square) -> Piece:
        return self.pieces[square.index]

    def __iter__(self) -> Iterator[tuple[Square, Piece]]:
        for sq in ALL_SQUARES:
            yield sq, self.pieces[sq.index]

    def diff(self, other: "Board") -> set[Square]:
        """Squares where this board differs from ``other``."""
        return {sq for sq in ALL_SQUARES if self.pieces[sq.index] != other.pieces[sq.index]}

    # -- rendering -----------------------------------------------------------
    def to_fen(self, side_to_move: str = "w") -> str:
        """Full FEN with empties run-length pooled. Castling/ep/clocks are
        placeholders (unknowable from a still image)."""
        ranks: list[str] = []
        for r in range(8):
            row, empty = "", 0
            for p in self.pieces[r * 8 : r * 8 + 8]:
                if p.is_empty:
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    row += p.symbol
            if empty:
                row += str(empty)
            ranks.append(row)
        return f"{'/'.join(ranks)} {side_to_move} - - 0 1"

    @property
    def placement(self) -> str:
        return self.to_fen().split(" ")[0]

    def lichess_url(self) -> str:
        return f"https://lichess.org/editor/{self.placement}"
