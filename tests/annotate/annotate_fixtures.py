"""Shared builders for the annotate test suite (importable because pytest puts each
test file's directory on sys.path)."""
from chessqueries.annotate.relay import START_PLACEMENT as START
from chessqueries.annotate.schema import Annotation

_BASE = dict(
    video_id="v",
    frame_index=1,
    timestamp_s=0.0,
    template_id="t",
    crop_bbox=[0, 0, 8, 8],
    game_id="g",
    round_id="r",
    ply=0,
    fen=f"{START} w - - 0 1",
    placement=START,
    side_to_move="w",
    white="W",
    black="B",
    white_clk_s=1,
    black_clk_s=1,
    confidence=0.5,
)


def make_annotation(**overrides) -> Annotation:
    """A valid `Annotation` at the start position; override any field."""
    return Annotation(**{**_BASE, **overrides})


__all__ = ["START", "make_annotation"]
