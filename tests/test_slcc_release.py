"""SLCC release identity and Hugging Face distribution contract."""

from pathlib import Path

import pytest

from chessqueries.annotate import release as release_module
from chessqueries.annotate.release import (
    DEFAULT_HF_REPO_ID,
    DEFAULT_HF_REVISION,
    EXPECTED_GROUPS,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_RECORDS,
    EXPECTED_SOURCE_SHA256,
    EXPECTED_SPLITS,
    EXPECTED_VIDEOS,
    ReleaseBundle,
)


def test_frozen_release_identity_is_pinned_without_a_tracked_payload():
    assert DEFAULT_HF_REPO_ID == "joelseytre/slcc"
    assert DEFAULT_HF_REVISION == "41bf6ca7fe50b6e3ed781e651b4fa53fcc9a7c25"
    assert EXPECTED_RECORDS == 2_174
    assert EXPECTED_VIDEOS == 20
    assert EXPECTED_GROUPS == 152
    assert EXPECTED_SPLITS == {"train": 1_475, "val": 326, "test": 373}
    assert EXPECTED_SOURCE_SHA256 == (
        "057f247ae92b134ca2b172317335919df01b22cfaa7472ddaf53393c2515ab75"
    )
    assert EXPECTED_MANIFEST_SHA256 == (
        "3c6137508de17472bd97470cd3f7d218e585db9f24a77574165029d16401e375"
    )
    assert not (Path(__file__).parents[1] / "slcc-v1").exists()


def test_fetch_uses_the_dataset_repo_revision_and_local_cache(tmp_path, monkeypatch):
    calls = []

    def fake_snapshot(repo_id, revision, local_dir):
        calls.append((repo_id, revision, local_dir))
        (local_dir / "downloaded.txt").write_text("complete")
        return str(local_dir)

    expected = object()
    monkeypatch.setattr(release_module, "_snapshot_download", fake_snapshot)
    monkeypatch.setattr(
        ReleaseBundle,
        "load",
        classmethod(lambda cls, root: expected),
    )

    actual = release_module.fetch_release_bundle(
        repo_id="owner/release",
        revision="abc123",
        local_dir=tmp_path / "cache",
    )

    assert actual is expected
    assert len(calls) == 1
    assert calls[0][:2] == ("owner/release", "abc123")
    assert calls[0][2].parent == tmp_path
    assert calls[0][2].name.startswith(".cache.part-")
    assert (tmp_path / "cache" / "downloaded.txt").is_file()


def test_fetch_reuses_a_validated_local_bundle_without_network(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    expected = object()
    monkeypatch.setattr(
        ReleaseBundle,
        "load",
        classmethod(lambda cls, root: expected),
    )
    monkeypatch.setattr(
        release_module,
        "_snapshot_download",
        lambda *args, **kwargs: pytest.fail("valid cache triggered a download"),
    )

    assert release_module.fetch_release_bundle(local_dir=cache) is expected


def test_failed_fetch_validation_leaves_prior_cache_untouched(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    sentinel = cache / "keep.txt"
    sentinel.write_text("prior cache")

    def fake_snapshot(repo_id, revision, local_dir):
        (local_dir / "manifest.json").write_text("incomplete")
        return str(local_dir)

    def reject(cls, root):
        raise ValueError(f"invalid bundle at {root}")

    monkeypatch.setattr(release_module, "_snapshot_download", fake_snapshot)
    monkeypatch.setattr(ReleaseBundle, "load", classmethod(reject))

    with pytest.raises(ValueError, match="invalid bundle"):
        release_module.fetch_release_bundle(local_dir=cache)

    assert sentinel.read_text() == "prior cache"
    assert list(tmp_path.glob(".cache.part-*")) == []


def test_release_rejects_a_manifest_other_than_the_pinned_one(tmp_path):
    for name in release_module.REQUIRED_DOCUMENTS:
        (tmp_path / name).write_text("metadata only")
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "checksums.sha256").write_text("")

    with pytest.raises(ValueError, match="not the frozen SLCC v1 manifest"):
        ReleaseBundle.load(tmp_path)


def test_hugging_face_local_cache_metadata_is_not_release_payload(tmp_path):
    (tmp_path / "README.md").write_text("release")
    metadata = tmp_path / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    (metadata / "README.md.metadata").write_text("hub cache state")

    paths = [
        path.relative_to(tmp_path).as_posix() for path in release_module._bundle_files(tmp_path)
    ]
    assert paths == ["README.md"]


def test_hugging_face_card_is_not_part_of_frozen_bundle_checksums(tmp_path):
    (tmp_path / "README.md").write_text("mutable Hub card")
    (tmp_path / "LICENSE-ANNOTATIONS.md").write_text("frozen notice")

    paths = [relative for _, relative in release_module._checksum_entries(tmp_path)]

    assert paths == ["LICENSE-ANNOTATIONS.md"]


def test_failed_regeneration_leaves_prior_bundle_untouched(tmp_path, monkeypatch):
    output = tmp_path / "slcc-v1"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("original")

    def fail_in_staging(dataset_manifest, reviewed_dir, stage, **kwargs):
        (stage / "partial.json").write_text("{}")
        raise ValueError("invalid frozen input")

    monkeypatch.setattr(release_module, "_build_release_bundle", fail_in_staging)
    with pytest.raises(ValueError, match="invalid frozen input"):
        release_module.build_release_bundle(
            tmp_path / "missing-manifest.json",
            tmp_path / "missing-reviewed",
            output,
        )

    assert sentinel.read_text() == "original"
    assert list(tmp_path.glob(".slcc-v1.stage-*")) == []


def test_fetch_explains_anonymized_placeholder(tmp_path, monkeypatch):
    """A placeholder repo id fails with the review explanation; others re-raise."""
    import sys
    import types

    from chessqueries.data.download import AnonymizedArtifactError

    class _RepoMissing(Exception):
        pass

    fake_errors = types.ModuleType("huggingface_hub.errors")
    fake_errors.RepositoryNotFoundError = _RepoMissing
    fake_errors.RevisionNotFoundError = type("RevisionNotFoundError", (Exception,), {})

    def _missing_repo(**kwargs):
        raise _RepoMissing("404: repository not found")

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = _missing_repo
    fake_hub.errors = fake_errors
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", fake_errors)

    with pytest.raises(AnonymizedArtifactError, match="anonymized review copy"):
        release_module._snapshot_download("anonymous/slcc", "some-revision", tmp_path)

    with pytest.raises(_RepoMissing):
        release_module._snapshot_download("example/other", "some-revision", tmp_path)


def test_vendored_release_bundle_validates():
    """The local bundle at data/slcc/releases/slcc-v1 loads end to end.

    The bundle is not tracked in git: make_supplementary.sh vendors it into the
    reviewer archive from this local cache (doc files rewritten, checksums
    refreshed). In the archive its presence is guaranteed (the smoke test
    hard-fails without it) and this validates it; in a checkout without the
    local data cache there is nothing to validate, so skip. Offline by design.
    """
    bundle_dir = Path(__file__).parents[1] / "data" / "slcc" / "releases" / "slcc-v1"
    if not bundle_dir.is_dir():
        pytest.skip("local SLCC release cache not present (fetch_release_bundle builds it)")
    bundle = release_module.ReleaseBundle.load(bundle_dir)

    records = bundle.reconstruction_records()
    assert len(records) == EXPECTED_RECORDS
    split_counts = {}
    for record in records:
        split_counts[record.split.value] = split_counts.get(record.split.value, 0) + 1
    assert split_counts == EXPECTED_SPLITS
