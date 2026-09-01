"""Synthetic multi-video coverage for the public SLCC bundle command."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from annotate_fixtures import make_annotation

from chessqueries.annotate import reconstruct as reconstruct_module
from chessqueries.annotate.reconstruct import reconstruct_bundle
from chessqueries.annotate.reconstruction import ReconstructionRecord
from chessqueries.core import Split
from chessqueries.data.slcc import SLCC


def _record(video: str, frame: int, split: Split, game: str) -> ReconstructionRecord:
    annotation = make_annotation(
        video_id=video,
        frame_index=frame,
        game_id=game,
        crop_bbox=[2, 1, 4, 3],
        verified_by_human=True,
    )
    return ReconstructionRecord(f"{video}_{frame}", split, annotation)


class _Bundle:
    def __init__(self, root: Path, records: list[ReconstructionRecord]) -> None:
        self.root = root
        self._records = records
        self.manifest = SimpleNamespace(
            release_version="synthetic-v1",
            schema_version=1,
            grouping=SimpleNamespace(key="game_id"),
        )

    def reconstruction_records(self):
        return list(self._records)

    def file_spec(self, video_id: str):
        return SimpleNamespace(
            video_id=video_id,
            format_id="137",
            source_width=10,
            source_height=8,
            source_fps=30.0,
        )


class _Reader:
    def __init__(self, video) -> None:
        self.video = video

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def frame_at_index(self, index: int):
        return np.full((8, 10, 3), index, dtype=np.uint8)


def _install_bundle(monkeypatch, tmp_path, records):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "manifest.json").write_text("{}")
    bundle = _Bundle(root, records)
    monkeypatch.setattr(reconstruct_module.ReleaseBundle, "load", lambda path: bundle)
    monkeypatch.setattr(reconstruct_module, "FrameReader", _Reader)
    return bundle


def _video(tmp_path: Path, video_id: str):
    path = tmp_path / f"{video_id}.mp4"
    path.write_bytes(b"synthetic")
    return SimpleNamespace(
        path=path,
        video_id=video_id,
        format_id="137",
        width=10,
        height=8,
        fps=30.0,
    )


def test_public_command_reconstructs_multiple_videos_and_is_restartable(tmp_path, monkeypatch):
    records = [
        _record("vidfirst001", 1, Split.TRAIN, "game-a"),
        _record("vidsecond01", 2, Split.TEST, "game-b"),
    ]
    bundle = _install_bundle(monkeypatch, tmp_path, records)
    calls: list[str] = []

    def fake_download(video_id, video_dir, *, format_id, access):
        assert access is None
        calls.append(video_id)
        return _video(tmp_path, video_id)

    monkeypatch.setattr(reconstruct_module, "download", fake_download)
    destination = tmp_path / "dataset"
    report = reconstruct_bundle(bundle.root, destination, video_dir=tmp_path / "videos")
    first_hash = hashlib.sha256((destination / "annotations.json").read_bytes()).hexdigest()

    assert report.committed and not report.partial
    assert calls == ["vidfirst001", "vidsecond01"]
    assert [
        sample.sample_id
        for sample in SLCC(destination).load_samples(Split.TRAIN, allow_partial=True)
    ] == [records[0].sample_id]
    assert [
        sample.sample_id
        for sample in SLCC(destination).load_samples(Split.TEST, allow_partial=True)
    ] == [records[1].sample_id]

    def no_download(*args, **kwargs):
        raise AssertionError("a complete reconstruction should reuse its validated crops")

    monkeypatch.setattr(reconstruct_module, "download", no_download)
    second = reconstruct_bundle(bundle.root, destination, video_dir=tmp_path / "videos")
    second_hash = hashlib.sha256((destination / "annotations.json").read_bytes()).hexdigest()
    assert second.committed and len(second.reused_sample_ids) == 2
    assert second_hash == first_hash


def test_missing_video_is_separate_and_requires_explicit_partial_mode(tmp_path, monkeypatch):
    records = [
        _record("vidfirst001", 1, Split.TRAIN, "game-a"),
        _record("vidsecond01", 2, Split.TEST, "game-b"),
    ]
    bundle = _install_bundle(monkeypatch, tmp_path, records)

    calls: list[str] = []

    def one_missing(video_id, video_dir, *, format_id, access):
        assert access is None
        calls.append(video_id)
        if video_id == "vidsecond01":
            raise RuntimeError("video no longer available")
        return _video(tmp_path, video_id)

    monkeypatch.setattr(reconstruct_module, "download", one_missing)
    destination = tmp_path / "dataset"
    strict = reconstruct_bundle(bundle.root, destination, video_dir=tmp_path / "videos")
    assert not strict.committed
    assert strict.missing_sample_ids == (records[1].sample_id,)
    assert [item.video_id for item in strict.unavailable_videos] == ["vidsecond01"]
    assert strict.advisory_mismatches == ()
    assert not destination.exists()
    assert calls == ["vidfirst001", "vidsecond01"]
    assert (tmp_path / ".dataset.reconstruction-cache").is_dir()

    calls.clear()
    partial = reconstruct_bundle(
        bundle.root,
        destination,
        video_dir=tmp_path / "videos",
        allow_partial=True,
    )
    manifest = json.loads((destination / "annotations.json").read_text())
    assert partial.committed and partial.partial
    assert manifest["provenance"]["partial"] is True
    assert manifest["provenance"]["actual_counts"]["records"] == 1
    assert manifest["provenance"]["expected_counts"]["records"] == 2
    assert calls == ["vidsecond01"]
    assert not (tmp_path / ".dataset.reconstruction-cache").exists()


def test_public_command_rejects_source_cache_inside_replaced_output(tmp_path):
    destination = tmp_path / "dataset"
    with pytest.raises(ValueError, match="must not be inside"):
        reconstruct_bundle(
            tmp_path / "bundle",
            destination,
            video_dir=destination / "videos",
        )


def test_bundle_resolution_fetches_hugging_face_unless_local_path_is_explicit(
    tmp_path, monkeypatch
):
    fetched = tmp_path / "fetched"
    calls = []

    def fake_fetch(*, repo_id, revision, local_dir):
        calls.append((repo_id, revision, local_dir))
        return SimpleNamespace(root=fetched)

    monkeypatch.setattr(reconstruct_module, "fetch_release_bundle", fake_fetch)
    cache = tmp_path / "cache"
    assert (
        reconstruct_module.resolve_bundle_path(
            None,
            hf_repo="owner/slcc",
            hf_revision="commit",
            bundle_cache=cache,
        )
        == fetched
    )
    assert calls == [("owner/slcc", "commit", cache)]

    local = tmp_path / "offline-bundle"
    assert reconstruct_module.resolve_bundle_path(local) == local
    assert len(calls) == 1


def test_cli_explains_that_an_incomplete_run_did_not_publish_dataset(tmp_path, monkeypatch, capsys):
    destination = tmp_path / "dataset"
    report = SimpleNamespace(
        committed=False,
        missing_sample_ids=("missing-sample",),
        expected_sample_ids=("missing-sample",),
        manifest_path=destination / "annotations.json",
        to_dict=lambda include_complete_ids=False: {"committed": False},
    )
    monkeypatch.setattr(reconstruct_module, "reconstruct_bundle", lambda *args, **kwargs: report)

    with pytest.raises(SystemExit) as caught:
        reconstruct_module.main(
            [
                "--bundle",
                str(tmp_path / "bundle"),
                "--out",
                str(destination),
                "--video-dir",
                str(tmp_path / "videos"),
            ]
        )

    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert "reconstruction was not committed" in error
    assert "intentionally not created or replaced" in error
    assert "training must not be started" in error
