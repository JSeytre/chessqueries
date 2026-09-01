"""Dependency-light sanity tests for the independent labels and evaluator."""

import tempfile
from pathlib import Path

import data as data_module
from data import fen_to_labels, load_slcc
from eval import score_boards


START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def test_perfect_prediction():
    board = list(fen_to_labels(START))
    metrics = score_boards([board, board], [board, board])
    assert metrics == {
        "n_boards": 2,
        "n_squares": 128,
        "n_squares_correct": 128,
        "per_square_accuracy": 1.0,
        "n_boards_exact": 2,
        "board_accuracy": 1.0,
    }


def test_one_wrong_square_on_one_board():
    ground_truth = list(fen_to_labels(START))
    wrong = ground_truth.copy()
    wrong[0] = 0
    metrics = score_boards([wrong, ground_truth], [ground_truth, ground_truth])
    assert metrics["n_squares"] == 128
    assert metrics["n_squares_correct"] == 127
    assert metrics["per_square_accuracy"] == 127 / 128
    assert metrics["n_boards_exact"] == 1
    assert metrics["board_accuracy"] == 0.5


def test_multi_board_aggregation():
    ground_truth = list(fen_to_labels(START))
    two_wrong = ground_truth.copy()
    two_wrong[10] = 5
    two_wrong[63] = 12
    all_wrong = [(label + 1) % 13 for label in ground_truth]
    metrics = score_boards(
        [ground_truth, two_wrong, all_wrong],
        [ground_truth, ground_truth, ground_truth],
    )
    assert metrics["n_boards"] == 3
    assert metrics["n_squares"] == 192
    assert metrics["n_squares_correct"] == 64 + 62
    assert metrics["n_boards_exact"] == 1
    assert abs(metrics["board_accuracy"] - 1 / 3) < 1e-12


def test_fen_order():
    labels = fen_to_labels(START)
    assert len(labels) == 64
    assert labels[0] == 10  # a8 = black rook
    assert labels[4] == 12  # e8 = black king
    assert labels[60] == 6  # e1 = white king
    assert labels[63] == 4  # h1 = white rook
    assert all(label == 0 for label in labels[16:48])
    assert fen_to_labels("8/8/8/8/8/8/8/8") == (0,) * 64


def test_missing_slcc_manifest_has_actionable_error():
    original_data_root = data_module.DATA_ROOT
    with tempfile.TemporaryDirectory() as directory:
        data_module.DATA_ROOT = Path(directory)
        try:
            try:
                load_slcc("train")
            except FileNotFoundError as exc:
                message = str(exc)
            else:
                raise AssertionError("missing SLCC manifest should fail")
        finally:
            data_module.DATA_ROOT = original_data_root

    assert "strict reconstruction" in message
    assert "minimal_reproduction/README.md" in message
    assert "Do not use --allow-partial" in message


if __name__ == "__main__":
    tests = [
        test_perfect_prediction,
        test_one_wrong_square_on_one_board,
        test_multi_board_aggregation,
        test_fen_order,
        test_missing_slcc_manifest_has_actionable_error,
    ]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print("all evaluator tests passed")
