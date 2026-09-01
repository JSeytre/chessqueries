"""Atomic staging, merging, and report behavior of the shared reconstruction core."""

import json

import cv2
import numpy as np
import pytest
from annotate_fixtures import make_annotation

from chessqueries.annotate import reconstruction as reconstruction_module
from chessqueries.annotate.reconstruction import (
    MANIFEST_NAME,
    ReconstructionError,
    ReconstructionRecord,
    ReconstructionTransaction,
)
from chessqueries.core import Split


def _record(video: str, frame: int, split: Split, game: str) -> ReconstructionRecord:
    annotation = make_annotation(
        video_id=video,
        frame_index=frame,
        game_id=game,
        crop_bbox=[0, 0, 8, 8],
        verified_by_human=True,
    )
    return ReconstructionRecord(f"{video}_{frame}", split, annotation)


def _image(value: int = 0) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


def test_core_merges_selected_records_without_dropping_unselected(tmp_path):
    destination = tmp_path / "dataset"
    images = destination / "images"
    images.mkdir(parents=True)
    cv2.imwrite(str(images / "old_1.jpg"), _image(10))
    old = {
        "sample_id": "old_1",
        "image": "images/old_1.jpg",
        "split": "val",
        "game_id": "old-game",
    }
    (destination / MANIFEST_NAME).write_text(json.dumps({"samples": [old]}))
    new = _record("vidnew00001", 2, Split.TEST, "new-game")

    with ReconstructionTransaction([new], destination, preserve_unselected=True) as transaction:
        transaction.write_image(new, _image(20))
        report = transaction.commit()

    samples = json.loads((destination / MANIFEST_NAME).read_text())["samples"]
    assert report.committed and not report.partial
    assert {record["sample_id"] for record in samples} == {"old_1", new.sample_id}
    assert next(record for record in samples if record["sample_id"] == "old_1")["split"] == "val"
    assert (destination / "images" / "old_1.jpg").is_file()


def test_destination_reuse_requires_matching_structural_fingerprint(tmp_path):
    destination = tmp_path / "dataset"
    old = _record("vidfirst001", 1, Split.TRAIN, "game-a")
    with ReconstructionTransaction([old], destination) as transaction:
        transaction.write_image(old, _image(10))
        transaction.commit()

    moved_annotation = make_annotation(
        video_id="vidfirst001",
        frame_index=1,
        game_id="game-a",
        crop_bbox=[1, 0, 8, 8],
        verified_by_human=True,
    )
    moved = ReconstructionRecord(old.sample_id, Split.TRAIN, moved_annotation)
    assert moved.crop_size == old.crop_size
    assert moved.fingerprint != old.fingerprint

    with ReconstructionTransaction([moved], destination) as transaction:
        assert transaction.pending_records == (moved,)
        assert not (transaction.stage_dir / moved.image).exists()
        transaction.write_image(moved, _image(20))
        report = transaction.commit()

    manifest = json.loads((destination / MANIFEST_NAME).read_text())
    assert report.reused_sample_ids == ()
    assert manifest["samples"][0]["reconstruction_fingerprint"] == moved.fingerprint
    assert int(cv2.imread(str(destination / moved.image)).mean()) == 20


def test_incomplete_strict_run_leaves_destination_untouched(tmp_path):
    destination = tmp_path / "dataset"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("original")
    records = [
        _record("vidfirst001", 1, Split.TRAIN, "game-a"),
        _record("vidsecond01", 2, Split.TEST, "game-b"),
    ]

    with ReconstructionTransaction(records, destination) as transaction:
        transaction.write_image(records[0], _image())
        transaction.note_unavailable_video("vidsecond01", "source unavailable")
        transaction.note_advisory_mismatch(records[0].sample_id, "reference pixels differ")
        report = transaction.commit()

    assert not report.committed and report.partial
    assert report.missing_sample_ids == (records[1].sample_id,)
    assert report.unavailable_videos[0].video_id == "vidsecond01"
    assert report.advisory_mismatches[0].sample_id == records[0].sample_id
    assert sentinel.read_text() == "original"
    assert not (destination / MANIFEST_NAME).exists()
    assert (tmp_path / ".dataset.reconstruction-cache").is_dir()

    with ReconstructionTransaction(records, destination) as transaction:
        assert [record.sample_id for record in transaction.pending_records] == [
            records[1].sample_id
        ]
        transaction.write_image(records[1], _image())
        resumed = transaction.commit()

    assert resumed.committed and not resumed.partial
    assert not (tmp_path / ".dataset.reconstruction-cache").exists()
    assert len(json.loads((destination / MANIFEST_NAME).read_text())["samples"]) == 2


def test_explicit_partial_run_records_expected_and_actual_counts(tmp_path):
    destination = tmp_path / "dataset"
    records = [
        _record("vidfirst001", 1, Split.TRAIN, "game-a"),
        _record("vidsecond01", 2, Split.TEST, "game-b"),
    ]

    with ReconstructionTransaction(records, destination, allow_partial=True) as transaction:
        transaction.write_image(records[0], _image())
        transaction.note_unavailable_video("vidsecond01", "source unavailable")
        report = transaction.commit()

    manifest = json.loads((destination / MANIFEST_NAME).read_text())
    assert report.committed and report.partial
    assert manifest["provenance"]["partial"] is True
    assert manifest["provenance"]["expected_counts"] == {
        "records": 2,
        "splits": {"test": 1, "train": 1},
    }
    assert manifest["provenance"]["actual_counts"] == {
        "records": 1,
        "splits": {"train": 1},
    }
    assert manifest["provenance"]["missing_sample_ids"] == [records[1].sample_id]


def test_group_split_violation_fails_before_replacing_destination(tmp_path):
    destination = tmp_path / "dataset"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("original")
    records = [
        _record("vidfirst001", 1, Split.TRAIN, "same-game"),
        _record("vidsecond01", 2, Split.TEST, "same-game"),
    ]

    with pytest.raises(ReconstructionError, match="cross splits"):
        with ReconstructionTransaction(records, destination) as transaction:
            for record in records:
                transaction.write_image(record, _image())
            transaction.commit()

    assert sentinel.read_text() == "original"
    assert not (destination / MANIFEST_NAME).exists()


def test_wrong_crop_dimensions_are_rejected_in_staging(tmp_path):
    record = _record("vidfirst001", 1, Split.TRAIN, "game-a")
    with ReconstructionTransaction([record], tmp_path / "dataset") as transaction:
        with pytest.raises(ReconstructionError, match="expected 8x8"):
            transaction.write_image(record, np.zeros((7, 8, 3), dtype=np.uint8))


def test_failed_prepare_removes_staging_directory(tmp_path):
    destination = tmp_path / "dataset"
    destination.mkdir()
    (destination / MANIFEST_NAME).write_text("not json")
    record = _record("vidfirst001", 1, Split.TRAIN, "game-a")

    with pytest.raises(ReconstructionError, match="malformed"):
        with ReconstructionTransaction(
            [record], destination, preserve_unselected=True
        ) as transaction:
            raise AssertionError(f"unexpected transaction: {transaction}")

    assert list(tmp_path.glob(".dataset.stage-*")) == []


def test_malformed_resume_image_path_is_ignored(tmp_path):
    destination = tmp_path / "dataset"
    cache = tmp_path / ".dataset.reconstruction-cache"
    (cache / "images").mkdir(parents=True)
    record = _record("vidfirst001", 1, Split.TRAIN, "game-a")
    (cache / "resume.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "sample_id": record.sample_id,
                        "image": "../outside.jpg",
                        "fingerprint": record.fingerprint,
                    }
                ]
            }
        )
    )

    with ReconstructionTransaction([record], destination) as transaction:
        assert transaction.pending_records == (record,)


def test_commit_rename_failure_restores_prior_destination(tmp_path, monkeypatch):
    destination = tmp_path / "dataset"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("original")
    record = _record("vidfirst001", 1, Split.TRAIN, "game-a")
    original_replace = reconstruction_module.os.replace
    calls = 0

    def fail_second_replace(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated commit interruption")
        return original_replace(source, target)

    monkeypatch.setattr(reconstruction_module.os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="simulated"):
        with ReconstructionTransaction([record], destination) as transaction:
            transaction.write_image(record, _image())
            transaction.commit()

    assert sentinel.read_text() == "original"
    assert list(tmp_path.glob(".dataset.backup-*")) == []
    assert list(tmp_path.glob(".dataset.stage-*")) == []
