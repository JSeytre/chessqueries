"""Atomic, hash-verified dataset and checkpoint downloads."""

import hashlib
import zipfile

import pytest

from chessqueries.data import download


class _Response:
    headers = {}

    def __init__(self, chunks, error=None):
        self.chunks = chunks
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield from self.chunks
        if self.error:
            raise self.error


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_http_download_installs_only_after_hash_verification(tmp_path, monkeypatch):
    payload = b"complete artifact"
    monkeypatch.setattr(
        download.requests, "get", lambda *args, **kwargs: _Response([payload])
    )
    destination = tmp_path / "artifact.bin"

    assert download._download("https://example.test/artifact", destination, _digest(payload)) == destination
    assert destination.read_bytes() == payload
    assert not (tmp_path / "artifact.bin.part").exists()


def test_interrupted_http_download_never_appears_at_final_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        download.requests,
        "get",
        lambda *args, **kwargs: _Response([b"partial"], RuntimeError("connection lost")),
    )
    destination = tmp_path / "artifact.bin"

    with pytest.raises(RuntimeError, match="connection lost"):
        download._download("https://example.test/artifact", destination, _digest(b"complete"))

    assert not destination.exists()
    assert (tmp_path / "artifact.bin.part").read_bytes() == b"partial"


def test_hash_mismatch_stays_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(
        download.requests, "get", lambda *args, **kwargs: _Response([b"wrong"])
    )
    destination = tmp_path / "artifact.bin"

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        download._download("https://example.test/artifact", destination, _digest(b"right"))

    assert not destination.exists()
    assert (tmp_path / "artifact.bin.part").is_file()


def test_gdrive_download_uses_part_path_and_verifies(tmp_path, monkeypatch):
    payload = b"checkpoint"
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        with open(kwargs["output"], "wb") as handle:
            handle.write(payload)
        return kwargs["output"]

    monkeypatch.setattr("gdown.download", fake_download)
    destination = tmp_path / "model.ckpt"

    download._download_gdrive("file-id", destination, _digest(payload))

    assert calls[0]["output"].endswith("model.ckpt.part")
    assert destination.read_bytes() == payload
    assert not download._part_path(destination).exists()


def test_gdrive_hash_mismatch_discards_corrupt_partial(tmp_path, monkeypatch):
    def fake_download(**kwargs):
        with open(kwargs["output"], "wb") as handle:
            handle.write(b"wrong")
        return kwargs["output"]

    monkeypatch.setattr("gdown.download", fake_download)
    destination = tmp_path / "model.ckpt"

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        download._download_gdrive("file-id", destination, _digest(b"right"))

    assert not destination.exists()
    assert not download._part_path(destination).exists()


def test_failed_extraction_keeps_existing_dataset_directory(tmp_path, monkeypatch):
    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("images/new.jpg", b"new")
    target = tmp_path / "images"
    target.mkdir()
    (target / "old.jpg").write_bytes(b"old")

    def fail_extract(self, member, path=None, pwd=None):
        raise RuntimeError("disk full")

    monkeypatch.setattr(zipfile.ZipFile, "extract", fail_extract)
    with pytest.raises(RuntimeError, match="disk full"):
        download._extract_zip(archive, tmp_path, "images")

    assert (target / "old.jpg").read_bytes() == b"old"
    assert list(tmp_path.glob(".images.part-*")) == []


def test_chessred_removes_a_corrupt_unhashed_zip_before_retry(tmp_path, monkeypatch):
    archive = tmp_path / "images_1024.zip"
    archive.write_bytes(b"Google Drive returned something other than a ZIP")
    downloads = []

    def fake_gdrive(file_id, destination, expected_sha256=None):
        if not destination.exists():
            downloads.append(file_id)
            with zipfile.ZipFile(destination, "w") as zf:
                zf.writestr("images/one.jpg", b"image")
        return destination

    monkeypatch.setattr(download, "_download", lambda *args, **kwargs: None)
    monkeypatch.setattr(download, "_download_gdrive", fake_gdrive)
    monkeypatch.setattr(download, "CHESSRED_IMAGE_COUNT", 1)

    with pytest.raises(RuntimeError, match="failed ZIP validation and was removed"):
        download.download_chessred(tmp_path)

    assert not archive.exists()
    assert downloads == []

    assert download.download_chessred(tmp_path) == tmp_path
    assert downloads == [download.CHESSRED_IMAGES_GDRIVE_ID]
    assert (tmp_path / "images" / "one.jpg").read_bytes() == b"image"
    assert not archive.exists()


def _failing_download(*args, **kwargs):
    import requests

    raise requests.HTTPError("401 Client Error: Unauthorized")


def test_release_checkpoint_explains_anonymized_placeholder(tmp_path, monkeypatch):
    """In the review export the URL points at anonymous/; the failure must say why."""
    monkeypatch.setattr(
        download,
        "RELEASE_CHECKPOINT_URL",
        "https://huggingface.co/anonymous/chessqueries/resolve/main/x.ckpt",
    )
    monkeypatch.setattr(download, "_download", _failing_download)

    with pytest.raises(download.AnonymizedArtifactError, match="anonymized review copy"):
        download.download_release_checkpoint(tmp_path / "x.ckpt")


def test_release_checkpoint_reraises_real_http_errors(tmp_path, monkeypatch):
    """From a non-placeholder host, an HTTP failure surfaces unchanged."""
    import requests

    # A neutral URL rather than the real constant: the review export rewrites
    # the project's Hugging Face account to the placeholder everywhere,
    # including in this test file, and the test must pass in both trees.
    monkeypatch.setattr(
        download, "RELEASE_CHECKPOINT_URL", "https://example.test/x.ckpt"
    )
    monkeypatch.setattr(download, "_download", _failing_download)

    with pytest.raises(requests.HTTPError, match="401"):
        download.download_release_checkpoint(tmp_path / "x.ckpt")
