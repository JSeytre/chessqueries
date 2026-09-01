"""Rasterize `Board`s and compose input-photo/predicted-board figures.

The lichess-style SVG from `chessqueries.viz.board` rendered to pixels, plus the
two-row (input above, board below) layout the paper figures and the demo share.
Needs the `viz` extras (matplotlib, cairosvg).
"""
from __future__ import annotations

import io
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cairosvg
import numpy as np
from PIL import Image

from chessqueries.core import Board, Square
from chessqueries.viz.board import RIGHT_FILL, WRONG_FILL, board_svg

if TYPE_CHECKING:
    from matplotlib.figure import Figure

DEFAULT_BOARD_PX = 480
__all__ = ["WRONG_FILL", "RIGHT_FILL"]  # re-exported from .board for callers


def render_board_png(
    board: Board,
    wrong: Iterable[Square] = (),
    *,
    size: int = DEFAULT_BOARD_PX,
    fill_color: str = WRONG_FILL,
    flipped: bool = False,
) -> np.ndarray:
    """Rasterize a board as a lichess-style RGB array, filling `wrong` squares.

    `flipped` renders it from Black's side; the position itself is unchanged.
    """
    svg = board_svg(board, wrong, size=size, fill_color=fill_color, flipped=flipped)
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=size, output_height=size)
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def draw_board(
    ax,
    board: Board,
    wrong: Iterable[Square] = (),
    *,
    fill_color: str = WRONG_FILL,
) -> None:
    """Draw a lichess-style board into a matplotlib axis."""
    ax.imshow(render_board_png(board, wrong, fill_color=fill_color))
    ax.axis("off")


def pad_to_aspect(image_path: Path | str, target_ar: float) -> np.ndarray:
    """Load a photo and pad it (black margins) to a target width/height aspect ratio.

    Shows the photo at its true aspect ratio instead of the model's distorted square
    resize. Padding every tile to the SAME `target_ar` makes a row of mixed-shape
    photos read as one homogeneous strip. Nothing is ever cropped.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if w / h < target_ar:          # too narrow -> widen the canvas
        new_w, new_h = int(round(h * target_ar)), h
    else:                          # too wide -> heighten the canvas
        new_w, new_h = w, int(round(w / target_ar))
    canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))  # black padding
    canvas.paste(img, ((new_w - w) // 2, (new_h - h) // 2))
    return np.asarray(canvas)


@dataclass(frozen=True)
class BoardPanel:
    """One column of a side-by-side figure: the input photo above, its board below."""

    image_path: Path
    board: Board
    wrong: frozenset[Square] = field(default_factory=frozenset)
    title: str | None = None


def composite_pair(
    image_path: Path | str,
    board: Board,
    *,
    height: int = DEFAULT_BOARD_PX,
    wrong: Iterable[Square] = (),
    flipped: bool = False,
) -> np.ndarray:
    """A single `input | board` strip, PIL only (no matplotlib) — the demo gallery item."""
    photo = Image.open(image_path).convert("RGB")
    photo = photo.resize((max(1, round(photo.width * height / photo.height)), height))
    tile = Image.fromarray(render_board_png(board, wrong, size=height, flipped=flipped))
    strip = Image.new("RGB", (photo.width + tile.width, height), (255, 255, 255))
    strip.paste(photo, (0, 0))
    strip.paste(tile, (photo.width, 0))
    return np.asarray(strip)


def side_by_side_figure(
    panels: Sequence[BoardPanel],
    *,
    input_ar: float = 1.0,
    row_labels: tuple[str, str] = ("input", "output"),
    col_width: float = 2.75,
) -> "Figure":
    """The paper's hero layout: a row of input photos above a row of predicted boards.

    Input tiles are padded to `input_ar` so the top row lines up with the square
    board tiles below it. The caller saves/closes the returned figure.
    """
    if not panels:
        raise ValueError("side_by_side_figure needs at least one panel")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(panels), figsize=(col_width * len(panels), 5.8), squeeze=False)
    for col, panel in enumerate(panels):
        axes[0, col].imshow(pad_to_aspect(panel.image_path, input_ar))
        axes[0, col].axis("off")
        if panel.title:
            axes[0, col].set_title(panel.title, fontsize=12)
        draw_board(axes[1, col], panel.board, panel.wrong)
    # Row labels down the left edge; the spines stay hidden so only the text shows.
    for row, text in enumerate(row_labels):
        axes[row, 0].set_ylabel(text, fontsize=12)
        axes[row, 0].axis("on")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for spine in axes[row, 0].spines.values():
            spine.set_visible(False)
    fig.tight_layout()
    return fig
