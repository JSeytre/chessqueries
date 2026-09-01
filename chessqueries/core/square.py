"""Board squares and the FEN-order indexing convention (a8=0 … h1=63)."""
from __future__ import annotations

from dataclasses import dataclass

FILES = "abcdefgh"


@dataclass(frozen=True)
class Square:
    """A board square. ``file`` is 0–7 (a–h), ``rank`` is 1–8."""

    file: int
    rank: int

    def __post_init__(self) -> None:
        if not 0 <= self.file < 8:
            raise ValueError(f"file out of range [0,8): {self.file}")
        if not 1 <= self.rank <= 8:
            raise ValueError(f"rank out of range [1,8]: {self.rank}")

    @property
    def index(self) -> int:
        """Position in FEN order: a8=0, h8=7, a1=56, h1=63."""
        return (8 - self.rank) * 8 + self.file

    @classmethod
    def from_index(cls, index: int) -> "Square":
        if not 0 <= index < 64:
            raise ValueError(f"index out of range [0,64): {index}")
        return cls(file=index % 8, rank=8 - index // 8)

    @classmethod
    def from_name(cls, name: str) -> "Square":
        name = name.strip().lower()
        if len(name) != 2 or name[0] not in FILES or not name[1].isdigit():
            raise ValueError(f"Invalid square name: {name!r}")
        return cls(file=FILES.index(name[0]), rank=int(name[1]))

    @property
    def name(self) -> str:
        return f"{FILES[self.file]}{self.rank}"

    @property
    def is_light(self) -> bool:
        return (self.file + self.rank) % 2 == 0

    def __str__(self) -> str:
        return self.name


ALL_SQUARES: tuple[Square, ...] = tuple(Square.from_index(i) for i in range(64))
