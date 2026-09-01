"""Transactional guarantees of the maintainer-facing SLCC export."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from annotate_fixtures import make_annotation

from chessqueries.annotate import reconstruction as reconstruction_module
from chessqueries.annotate import workflow
from chessqueries.annotate.__main__ import main as annotate_main
from chessqueries.annotate.manifest import Manifest, VideoEntry
from chessqueries.annotate.reconstruction import ReconstructionError
from chessqueries.annotate.schema import AnnotationFile
from chessqueries.data.slcc import MANIFEST_NAME, SplitRatio

VID_A = "vidAAAAAAAA"
VID_B = "vidBBBBBBBB"


def _manifest(tmp_path, *video_ids: str) -> Manifest:
    return Manifest(
        path=tmp_path / "videos.json",
        entries={video_id: VideoEntry(video_id=video_id) for video_id in video_ids},
    )


def _reviewed(
    data_dir,
    video_id: str,
    frames: list[int],
    *,
    games: int = 1,
    verified: set[int] | None = None,
    duplicate_templates: bool = False,
) -> None:
    accepted = set(frames) if verified is None else verified
    annotations = [
        make_annotation(
            video_id=video_id,
            frame_index=frame,
            game_id=f"{video_id}-game-{frame % games}",
            template_id="shared" if duplicate_templates and frame < 3 else f"template-{frame}",
            crop_bbox=[0, 0, 8, 8],
            verified_by_human=frame in accepted,
        )
        for frame in frames
    ]
    AnnotationFile(
        provenance={"video_id": video_id, "format_id": "137"},
        annotations=annotations,
    ).save(workflow.reviewed_path(data_dir, video_id))


def _install_video_stubs(monkeypatch, *, images=None, interrupt_at=None):
    calls: list[str] = []
    images = images or {}

    def fake_download(video_id, data_dir, *, format_id):
        calls.append(video_id)
        return SimpleNamespace(video_id=video_id, format_id=format_id)

    class Reader:
        def __init__(self, video):
            self.video = video

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def frame_at_index(self, frame_index):
            if (self.video.video_id, frame_index) == interrupt_at:
                raise KeyboardInterrupt("simulated interruption")
            return images.get(
                (self.video.video_id, frame_index),
                np.full((10, 10, 3), frame_index % 255, dtype=np.uint8),
            )

    monkeypatch.setattr(workflow.video_mod, "download", fake_download)
    monkeypatch.setattr(workflow.video_mod, "FrameReader", Reader)
    return calls


def _samples(out_dir):
    return json.loads((out_dir / MANIFEST_NAME).read_text())["samples"]


def _export(manifest, data_dir, out_dir, **kwargs):
    return workflow.export_dataset(
        manifest,
        SplitRatio(60, 20, 20),
        data_dir=data_dir,
        out_dir=out_dir,
        dedup=False,
        log=lambda *args: None,
        **kwargs,
    )


def test_scoped_export_preserves_other_video_and_its_split(tmp_path, monkeypatch):
    data_dir, out_dir = tmp_path / "slcc", tmp_path / "dataset"
    data_dir.mkdir()
    manifest = _manifest(tmp_path, VID_A, VID_B)
    _reviewed(data_dir, VID_A, [0, 1])
    _reviewed(data_dir, VID_B, [2])
    _install_video_stubs(monkeypatch)

    _export(manifest, data_dir, out_dir, rebuild_all=True)
    before_b = next(record for record in _samples(out_dir) if record["video_id"] == VID_B)
    before_pixels = (out_dir / before_b["image"]).read_bytes()

    _reviewed(data_dir, VID_A, [3])
    _export(manifest, data_dir, out_dir, video_ids=[VID_A])
    payload = json.loads((out_dir / MANIFEST_NAME).read_text())
    after = payload["samples"]
    after_b = next(record for record in after if record["video_id"] == VID_B)

    assert {_id(record) for record in after if record["video_id"] == VID_A} == {f"{VID_A}_3"}
    assert after_b == before_b
    assert (out_dir / after_b["image"]).read_bytes() == before_pixels
    assert set(payload["provenance"]["reviewed_videos"]) == {VID_A, VID_B}
    assert payload["provenance"]["operation_video_ids"] == [VID_A]
    assert payload["provenance"]["preserved_records"] == 1
    assert payload["provenance"]["dataset_counts"]["records"] == 2


def test_repeated_export_is_idempotent_and_reuses_validated_crops(tmp_path, monkeypatch):
    data_dir, out_dir = tmp_path / "slcc", tmp_path / "dataset"
    data_dir.mkdir()
    manifest = _manifest(tmp_path, VID_A)
    _reviewed(data_dir, VID_A, [0, 1], games=2)
    calls = _install_video_stubs(monkeypatch)

    _export(manifest, data_dir, out_dir, rebuild_all=True)
    first_hash = hashlib.sha256((out_dir / MANIFEST_NAME).read_bytes()).hexdigest()
    _export(manifest, data_dir, out_dir, rebuild_all=True)
    second_hash = hashlib.sha256((out_dir / MANIFEST_NAME).read_bytes()).hexdigest()

    assert first_hash == second_hash
    assert calls == [VID_A]


def test_write_failure_leaves_the_previous_dataset_usable(tmp_path, monkeypatch):
    data_dir, out_dir = tmp_path / "slcc", tmp_path / "dataset"
    data_dir.mkdir()
    manifest = _manifest(tmp_path, VID_A)
    _reviewed(data_dir, VID_A, [0])
    _install_video_stubs(monkeypatch)
    _export(manifest, data_dir, out_dir, rebuild_all=True)
    original_manifest = (out_dir / MANIFEST_NAME).read_bytes()
    original_image = (out_dir / "images" / f"{VID_A}_0.jpg").read_bytes()

    _reviewed(data_dir, VID_A, [1])
    monkeypatch.setattr(reconstruction_module.cv2, "imwrite", lambda *args: False)
    with pytest.raises(ReconstructionError, match="failed to write"):
        _export(manifest, data_dir, out_dir, video_ids=[VID_A])

    assert (out_dir / MANIFEST_NAME).read_bytes() == original_manifest
    assert (out_dir / "images" / f"{VID_A}_0.jpg").read_bytes() == original_image
    assert not list(tmp_path.glob(".dataset.stage-*"))


def test_interruption_leaves_the_previous_dataset_usable(tmp_path, monkeypatch):
    data_dir, out_dir = tmp_path / "slcc", tmp_path / "dataset"
    data_dir.mkdir()
    manifest = _manifest(tmp_path, VID_A)
    _reviewed(data_dir, VID_A, [0])
    _install_video_stubs(monkeypatch)
    _export(manifest, data_dir, out_dir, rebuild_all=True)
    original_manifest = (out_dir / MANIFEST_NAME).read_bytes()

    _reviewed(data_dir, VID_A, [1])
    _install_video_stubs(monkeypatch, interrupt_at=(VID_A, 1))
    with pytest.raises(KeyboardInterrupt, match="simulated"):
        _export(manifest, data_dir, out_dir, video_ids=[VID_A])

    assert (out_dir / MANIFEST_NAME).read_bytes() == original_manifest
    assert (out_dir / "images" / f"{VID_A}_0.jpg").is_file()
    assert not list(tmp_path.glob(".dataset.stage-*"))


def test_export_keeps_games_in_one_split(tmp_path, monkeypatch):
    data_dir, out_dir = tmp_path / "slcc", tmp_path / "dataset"
    data_dir.mkdir()
    manifest = _manifest(tmp_path, VID_A)
    _reviewed(data_dir, VID_A, list(range(30)), games=4)
    _install_video_stubs(monkeypatch)

    _export(manifest, data_dir, out_dir, rebuild_all=True)
    by_game: dict[str, set[str]] = {}
    for record in _samples(out_dir):
        by_game.setdefault(record["game_id"], set()).add(record["split"])

    assert all(len(splits) == 1 for splits in by_game.values())


def test_regroup_requires_and_performs_an_explicit_full_rebuild(tmp_path, monkeypatch):
    data_dir, out_dir = tmp_path / "slcc", tmp_path / "dataset"
    data_dir.mkdir()
    manifest = _manifest(tmp_path, VID_A)
    _reviewed(data_dir, VID_A, list(range(12)), games=2)
    _install_video_stubs(monkeypatch)

    _export(manifest, data_dir, out_dir, rebuild_all=True, split_by=workflow.SplitBy.FRAME)
    before: dict[str, set[str]] = {}
    for record in _samples(out_dir):
        before.setdefault(record["game_id"], set()).add(record["split"])
    assert any(len(splits) > 1 for splits in before.values())

    with pytest.raises(ValueError, match="requires rebuild_all"):
        _export(manifest, data_dir, out_dir, video_ids=[VID_A], regroup=True)
    _export(manifest, data_dir, out_dir, rebuild_all=True, regroup=True)
    after: dict[str, set[str]] = {}
    for record in _samples(out_dir):
        after.setdefault(record["game_id"], set()).add(record["split"])
    assert all(len(splits) == 1 for splits in after.values())


def test_deduplication_discards_only_staged_losers(tmp_path, monkeypatch):
    data_dir, out_dir = tmp_path / "slcc", tmp_path / "dataset"
    data_dir.mkdir()
    manifest = _manifest(tmp_path, VID_A)
    _reviewed(data_dir, VID_A, [0, 1, 2, 3], duplicate_templates=True)
    rng = np.random.default_rng(0)
    images = {
        (VID_A, 0): np.full((10, 10, 3), 128, dtype=np.uint8),
        (VID_A, 1): rng.integers(0, 256, (10, 10, 3), dtype=np.uint8),
        (VID_A, 2): np.full((10, 10, 3), 128, dtype=np.uint8),
        (VID_A, 3): np.full((10, 10, 3), 64, dtype=np.uint8),
    }
    _install_video_stubs(monkeypatch, images=images)

    workflow.export_dataset(
        manifest,
        SplitRatio(100, 0, 0),
        data_dir=data_dir,
        out_dir=out_dir,
        rebuild_all=True,
        log=lambda *args: None,
    )

    payload = json.loads((out_dir / MANIFEST_NAME).read_text())
    assert {_id(record) for record in payload["samples"]} == {f"{VID_A}_1", f"{VID_A}_3"}
    assert payload["provenance"]["deduplicated_by_video"] == {
        VID_A: [f"{VID_A}_0", f"{VID_A}_2"]
    }
    assert not (out_dir / "images" / f"{VID_A}_0.jpg").exists()
    assert not (out_dir / "images" / f"{VID_A}_2.jpg").exists()


def test_dashboard_treats_deduplicated_frames_as_accounted_for():
    state = workflow.VideoState(entry=VideoEntry(video_id=VID_A), reviewed=True)
    state.verified_stems = frozenset(f"{VID_A}_{frame}" for frame in range(4))
    surveyed = workflow.Survey(
        states=[state],
        shipped_by_vid={VID_A: frozenset({f"{VID_A}_1", f"{VID_A}_3"})},
        deduplicated_by_vid={VID_A: frozenset({f"{VID_A}_0", f"{VID_A}_2"})},
    )

    assert not [action for action in workflow.pending_actions(surveyed) if action.icon == "📦"]


def test_partially_reviewed_video_is_skipped_unless_explicitly_included(
    tmp_path, monkeypatch
):
    data_dir, out_dir = tmp_path / "slcc", tmp_path / "dataset"
    data_dir.mkdir()
    manifest = _manifest(tmp_path, VID_A)
    _reviewed(data_dir, VID_A, [0, 1], verified={0})
    _install_video_stubs(monkeypatch)

    with pytest.raises(SystemExit):
        _export(manifest, data_dir, out_dir, video_ids=[VID_A])
    assert not (out_dir / MANIFEST_NAME).exists()

    workflow.export_dataset(
        manifest,
        SplitRatio(100, 0, 0),
        data_dir=data_dir,
        out_dir=out_dir,
        video_ids=[VID_A],
        verified_only=False,
        dedup=False,
        log=lambda *args: None,
    )
    assert len(_samples(out_dir)) == 2


def test_export_requires_an_explicit_scope(tmp_path):
    manifest = _manifest(tmp_path, VID_A)
    with pytest.raises(ValueError, match="scoped export"):
        workflow.export_dataset(manifest, SplitRatio(100, 0, 0))
    with pytest.raises(ValueError, match="not both"):
        workflow.export_dataset(
            manifest,
            SplitRatio(100, 0, 0),
            video_ids=[VID_A],
            rebuild_all=True,
        )


def test_reconstruct_cli_requires_video_scope_or_rebuild_all():
    with pytest.raises(SystemExit) as exc:
        annotate_main(["reconstruct"])
    assert exc.value.code == 2


def test_reconstruct_cli_reports_scoped_regroup_as_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        annotate_main(["reconstruct", "--video", VID_A, "--regroup"])
    assert exc.value.code == 2


def _id(record: dict) -> str:
    return record.get("sample_id") or record["image"].rsplit("/", 1)[-1].removesuffix(".jpg")
