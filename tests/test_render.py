"""Board rasterization and the input/board figure layout: shapes, the wrong-square
tint, aspect padding, and the two-row composition the hero figure and demo share."""
import pytest
from PIL import Image

from chessqueries.core import Board, Square

pytest.importorskip("cairosvg")
pytest.importorskip("matplotlib")

from chessqueries.viz.render import (  # noqa: E402  (after importorskip)
    DEFAULT_BOARD_PX,
    BoardPanel,
    composite_pair,
    pad_to_aspect,
    render_board_png,
    side_by_side_figure,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


@pytest.fixture
def photo(tmp_path):
    """A landscape photo — the shape that needs padding to sit above a square board."""
    path = tmp_path / "photo.png"
    Image.new("RGB", (80, 40), (30, 60, 90)).save(path)
    return path


def test_render_board_png_shape_and_dtype():
    img = render_board_png(Board.from_fen(START_FEN))

    assert img.shape == (DEFAULT_BOARD_PX, DEFAULT_BOARD_PX, 3)
    assert img.dtype.kind == "u"


def test_render_board_png_honors_size():
    assert render_board_png(Board.empty(), size=120).shape == (120, 120, 3)


def test_wrong_squares_change_pixels():
    """The tint is what marks an error, so it has to actually alter the raster."""
    plain = render_board_png(Board.empty())
    tinted = render_board_png(Board.empty(), [Square.from_name("e4")])

    assert (plain != tinted).any()


def test_flipped_changes_the_render():
    """Viewing from Black's side is a different picture (orientation + coordinate
    labels) of an unchanged position — the caller's Board is never mutated."""
    board = Board.from_fen(START_FEN)
    plain = render_board_png(board)
    flipped = render_board_png(board, flipped=True)

    assert plain.shape == flipped.shape
    assert (plain != flipped).any()
    assert board.to_fen().startswith(START_FEN)   # rendering does not touch the position


def test_composite_pair_accepts_flipped(photo):
    strip = composite_pair(photo, Board.from_fen(START_FEN), height=100, flipped=True)
    assert strip.shape == (100, 300, 3)


def test_pad_to_aspect_reaches_target_ratio(photo):
    square = pad_to_aspect(photo, 1.0)
    h, w = square.shape[:2]
    assert w == h == 80  # widened canvas keeps the original width, pads height

    wide = pad_to_aspect(photo, 4.0)
    h, w = wide.shape[:2]
    assert round(w / h, 2) == 4.0


def test_pad_to_aspect_never_crops(photo):
    """Padding only ever grows the canvas."""
    padded = pad_to_aspect(photo, 1.0)
    original = Image.open(photo)
    assert padded.shape[1] >= original.width
    assert padded.shape[0] >= original.height


def test_composite_pair_is_photo_beside_board(photo):
    strip = composite_pair(photo, Board.from_fen(START_FEN), height=100)

    assert strip.shape[0] == 100
    # photo scaled to height 100 (80x40 -> 200x100) then the 100px board tile
    assert strip.shape[1] == 200 + 100


def test_side_by_side_figure_has_two_rows_per_panel(photo):
    panels = [
        BoardPanel(image_path=photo, board=Board.empty(), title="a"),
        BoardPanel(
            image_path=photo,
            board=Board.from_fen(START_FEN),
            wrong=frozenset({Square.from_name("e4")}),
        ),
    ]
    fig = side_by_side_figure(panels)

    assert len(fig.axes) == 2 * len(panels)
    assert fig.axes[0].get_title() == "a"
    assert fig.axes[0].get_ylabel() == "input"


def test_side_by_side_figure_single_panel(photo):
    """squeeze=False keeps the 2xN indexing valid for one column."""
    fig = side_by_side_figure([BoardPanel(image_path=photo, board=Board.empty())])
    assert len(fig.axes) == 2


def test_side_by_side_figure_rejects_empty():
    with pytest.raises(ValueError, match="at least one panel"):
        side_by_side_figure([])
