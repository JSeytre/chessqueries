"""Strict dataset inventories and the explicit diagnostic partial mode."""

from pathlib import Path

import pytest

from chessqueries.core import Board, Split
from chessqueries.data.base import (
    BoardSample,
    ChessDataset,
    DatasetIncompleteError,
    DatasetName,
)
from chessqueries.metrics.report import evaluation_inventory


class _InventoryDataset(ChessDataset):
    name = None
    splits = (Split.TEST,)
    expected_samples = {Split.TEST: 3}

    def __init__(self, samples: list[BoardSample]) -> None:
        self.samples = samples

    def _load_samples(self, split: Split | None) -> list[BoardSample]:
        return list(self.samples)


_InventoryDataset.name = DatasetName.SLCC


def _sample(path: Path, sample_id: str) -> BoardSample:
    return BoardSample(path, Board.empty(), DatasetName.SLCC, sample_id, Split.TEST)


def test_strict_loading_rejects_missing_labelled_images(tmp_path):
    present = tmp_path / "present.jpg"
    present.write_bytes(b"image")
    dataset = _InventoryDataset(
        [_sample(present, "present"), _sample(tmp_path / "missing.jpg", "missing")]
    )

    with pytest.raises(DatasetIncompleteError, match="expected 3.*1 available"):
        dataset.load_samples(Split.TEST)


def test_allow_partial_filters_missing_images_and_reports_counts(tmp_path, capsys):
    present = tmp_path / "present.jpg"
    present.write_bytes(b"image")
    dataset = _InventoryDataset(
        [_sample(present, "present"), _sample(tmp_path / "missing.jpg", "missing")]
    )

    loaded = dataset.load_with_report(Split.TEST, allow_partial=True)
    inventory = evaluation_inventory(loaded.completeness, evaluated_samples=1)

    assert [sample.sample_id for sample in loaded.samples] == ["present"]
    assert inventory == {
        "dataset": "slcc",
        "split": "test",
        "mode": "allow_partial",
        "scope": "full_split",
        "data_complete": False,
        "expected_samples": 3,
        "expected_evaluated_samples": 3,
        "labelled_samples": 2,
        "available_samples": 1,
        "actual_samples": 1,
        "missing_images": 1,
        "structural_issues": [],
    }
    assert "WARNING [allow_partial]" in capsys.readouterr().out


def test_allow_partial_cannot_turn_an_empty_split_into_an_evaluation(tmp_path):
    dataset = _InventoryDataset([_sample(tmp_path / "missing.jpg", "missing")])
    with pytest.raises(DatasetIncompleteError, match="no available images"):
        dataset.load_samples(Split.TEST, allow_partial=True)


def test_strict_loading_rejects_an_unexpected_record_count(tmp_path):
    samples = []
    for index in range(2):
        path = tmp_path / f"{index}.jpg"
        path.write_bytes(b"image")
        samples.append(_sample(path, str(index)))
    dataset = _InventoryDataset(samples)

    with pytest.raises(DatasetIncompleteError, match="expected 3, found 2 labelled"):
        dataset.load_samples(Split.TEST)


def test_duplicate_ids_are_invalid_even_in_partial_mode(tmp_path):
    path = tmp_path / "image.jpg"
    path.write_bytes(b"image")
    dataset = _InventoryDataset([_sample(path, "same"), _sample(path, "same")])

    with pytest.raises(ValueError, match="duplicate sample IDs.*same"):
        dataset.load_samples(Split.TEST, allow_partial=True)


def test_label_count_mismatch_is_partial_even_if_available_count_matches(tmp_path):
    samples = []
    for index in range(4):
        path = tmp_path / f"{index}.jpg"
        if index < 3:
            path.write_bytes(b"image")
        samples.append(_sample(path, str(index)))
    dataset = _InventoryDataset(samples)

    loaded = dataset.load_with_report(Split.TEST, allow_partial=True)

    assert loaded.completeness.available_samples == 3
    assert loaded.completeness.expected_samples == 3
    assert loaded.completeness.partial
