"""Shared eval reporting helpers (metrics/report.py) and the predict_all loop."""
import json

import pytest
import torch
from PIL import Image

from chessqueries.core import Board, Split
from chessqueries.data.base import BoardImageDataset, BoardSample, DatasetName
from chessqueries.metrics.report import eval_tag, print_metrics, resolve_split, write_eval_outputs
from chessqueries.models.base import BoardRecognizer, EvalPredictions, predict_all


def test_eval_tag():
    assert eval_tag("chessred", Split.TEST) == "chessred_test"
    assert eval_tag("cvchess", None) == "cvchess_all"
    assert eval_tag("chessred", Split.TEST, sep="/") == "chessred/test"


def test_resolve_split_split_datasets_and_splitless():
    from chessqueries.data import get_dataset

    chessred = get_dataset(DatasetName.CHESSRED)
    assert resolve_split(chessred, "test") is Split.TEST
    with pytest.raises(SystemExit):
        resolve_split(chessred, None)  # split required

    cvchess = get_dataset(DatasetName.CVCHESS)
    assert resolve_split(cvchess, None) is None  # split-less dataset


def _samples(tmp_path, n=2):
    out = []
    for i in range(n):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (8, 8), (i * 40, 0, 0)).save(p)
        out.append(BoardSample(image_path=p, board=Board.empty(),
                               dataset=DatasetName.CVCHESS, sample_id=f"s{i}"))
    return out


def test_write_eval_outputs_shapes(tmp_path):
    samples = _samples(tmp_path)
    preds = [[0] * 64, [1] * 64]
    out = write_eval_outputs(tmp_path / "out", "cvchess_all", {"board_accuracy": 0.5},
                             samples, preds)
    metrics = json.loads((out / "metrics_cvchess_all.json").read_text())
    assert metrics == {"board_accuracy": 0.5}
    written = json.loads((out / "preds_cvchess_all.json").read_text())
    assert written == {"s0": [0] * 64, "s1": [1] * 64}


@pytest.mark.parametrize("n_samples,n_preds", [(1, 2), (2, 1)])
def test_write_eval_outputs_rejects_misaligned_lengths_before_writing(
    tmp_path, n_samples, n_preds
):
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=r"zip\(\) argument"):
        write_eval_outputs(
            out,
            "cvchess_all",
            {"board_accuracy": 0.5},
            _samples(tmp_path, n_samples),
            [[0] * 64 for _ in range(n_preds)],
        )
    assert not out.exists()


def test_write_eval_outputs_rejects_duplicate_ids_before_writing(tmp_path):
    samples = _samples(tmp_path)
    samples[1].sample_id = samples[0].sample_id
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="duplicate sample IDs.*s0"):
        write_eval_outputs(
            out,
            "cvchess_all",
            {"board_accuracy": 0.5},
            samples,
            [[0] * 64, [1] * 64],
        )
    assert not out.exists()


def test_print_metrics_formats_floats_and_ints(capsys):
    print_metrics({"board_accuracy": 0.5, "n_boards": 3}, "header")
    out = capsys.readouterr().out
    assert "=== header ===" in out
    assert "0.5000" in out and " 3" in out


class _ConstantRecognizer(BoardRecognizer):
    def predict_labels(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones(len(x), 64, dtype=torch.long)


def test_predict_all_returns_label_lists(tmp_path):
    ds = BoardImageDataset(_samples(tmp_path))
    predicted = predict_all(_ConstantRecognizer(), ds, batch_size=1, workers=0)
    assert predicted.preds == [[1] * 64, [1] * 64]
    assert predicted.gts == [[0] * 64, [0] * 64]  # Board.empty() ground truth
    assert len(predicted) == 2


def test_eval_predictions_rejects_misaligned_lengths():
    with pytest.raises(ValueError, match="2 preds vs 1 gts"):
        EvalPredictions(preds=[[0] * 64, [0] * 64], gts=[[0] * 64])
