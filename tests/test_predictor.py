"""Predictor turns arbitrary image paths into Boards: order preserved across
batches, odd input modes (alpha/grayscale) coerced to RGB, records serialized."""
import json

import pytest
import torch
from PIL import Image

from chessqueries.core import Board, Piece
from chessqueries.models.base import BoardRecognizer
from chessqueries.models.predictor import PAPER_RESOLUTION, Prediction, Predictor

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


class StubRecognizer(BoardRecognizer):
    """Returns one canned board per image and records the batch shapes it saw."""

    def __init__(self, boards: list[Board]) -> None:
        super().__init__()
        self.boards = boards
        self.batch_shapes: list[tuple[int, ...]] = []
        self._next = 0

    def predict_labels(self, x: torch.Tensor) -> torch.Tensor:
        self.batch_shapes.append(tuple(x.shape))
        taken = self.boards[self._next : self._next + x.shape[0]]
        self._next += x.shape[0]
        return torch.tensor([b.labels for b in taken])


@pytest.fixture
def images(tmp_path):
    """Three tiny images of different sizes/modes — none of them dataset-shaped."""
    paths = []
    for i, (size, mode) in enumerate([((40, 30), "RGB"), ((30, 40), "RGBA"), ((25, 25), "L")]):
        path = tmp_path / f"img{i}.png"
        Image.new(mode, size).save(path)
        paths.append(path)
    return paths


@pytest.fixture
def boards():
    return [Board.empty(), Board.from_fen(START_FEN), Board.empty()]


def test_predict_preserves_order_and_paths(images, boards):
    stub = StubRecognizer(boards)
    preds = Predictor(stub, resolution=64).predict(images)

    assert [p.image_path for p in preds] == images
    assert [p.board for p in preds] == boards


def test_predict_batches_and_resizes(images, boards):
    """Every image is resized to the square resolution, so they stack; batch_size splits."""
    stub = StubRecognizer(boards)
    Predictor(stub, resolution=64).predict(images, batch_size=2)

    assert stub.batch_shapes == [(2, 3, 64, 64), (1, 3, 64, 64)]


def test_alpha_and_grayscale_become_three_channels(images, boards):
    stub = StubRecognizer(boards)
    Predictor(stub, resolution=32).predict(images, batch_size=1)

    assert {shape[1] for shape in stub.batch_shapes} == {3}


def test_empty_input_short_circuits(boards):
    stub = StubRecognizer(boards)
    assert Predictor(stub, resolution=64).predict([]) == []
    assert stub.batch_shapes == []


def test_prediction_record_is_json_serializable(tmp_path):
    pred = Prediction(image_path=tmp_path / "x.jpg", board=Board.from_fen(START_FEN))

    record = pred.to_record()
    assert json.loads(json.dumps(record)) == record
    assert record["fen"] == f"{START_FEN} w - - 0 1"
    assert record["lichess_url"].endswith(START_FEN)
    assert record["image"] == str(tmp_path / "x.jpg")


def test_paper_resolution_is_the_shipped_default():
    """Paper checkpoints are ViT-L@644 and resolution is not stored in the ckpt."""
    assert PAPER_RESOLUTION == 644


def test_predictions_round_trip_through_board(images):
    """A non-trivial board survives tensor -> Board -> FEN."""
    board = Board.from_fen(START_FEN)
    stub = StubRecognizer([board] * len(images))
    preds = Predictor(stub, resolution=32).predict(images)

    assert preds[0].fen.startswith(START_FEN)
    assert preds[0].board.pieces[0] is Piece.from_symbol("r")
