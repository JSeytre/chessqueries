"""The chessqueries-predict CLI: TSV and JSON output, the --viz figure, and a
useful error when the viz extras are missing."""
import json

import pytest
import torch
from PIL import Image

from chessqueries.core import Board
from chessqueries.models import predictor as predictor_module
from chessqueries.models.base import BoardRecognizer
from chessqueries.models.predictor import Predictor, main, resolve_checkpoint

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


class StubRecognizer(BoardRecognizer):
    def predict_labels(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([Board.from_fen(START_FEN).labels] * x.shape[0])


@pytest.fixture
def images(tmp_path):
    paths = []
    for i in range(2):
        path = tmp_path / f"board{i}.png"
        Image.new("RGB", (60, 40)).save(path)
        paths.append(path)
    return paths


@pytest.fixture(autouse=True)
def stub_checkpoint_loading(monkeypatch):
    """Never touch a real checkpoint: from_checkpoint yields a stub-backed Predictor."""
    monkeypatch.setattr(
        Predictor,
        "from_checkpoint",
        classmethod(lambda cls, ckpt, *, resolution, device=None:
                    cls(StubRecognizer(), resolution=32)),
    )


def test_tsv_output_is_path_then_fen(images, capsys):
    main(["--checkpoint", "unused.ckpt", *map(str, images)])

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    for line, image in zip(lines, images):
        path, fen = line.split("\t")
        assert path == str(image)
        assert fen == f"{START_FEN} w - - 0 1"


def test_json_output_parses_to_records(images, capsys):
    main(["--checkpoint", "unused.ckpt", "--json", *map(str, images)])

    records = json.loads(capsys.readouterr().out)
    assert [sorted(r) for r in records] == [["fen", "image", "lichess_url"]] * 2
    assert records[0]["image"] == str(images[0])


def test_cli_uses_release_checkpoint_by_default(images, tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "release.ckpt"
    monkeypatch.setattr(predictor_module, "download_release_checkpoint", lambda: checkpoint)

    main([str(images[0])])

    assert capsys.readouterr().out.startswith(f"{images[0]}\t")


def test_default_checkpoint_is_downloaded_without_polluting_stdout(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "release.ckpt"

    def download():
        print("download status")
        return checkpoint

    monkeypatch.setattr(predictor_module, "download_release_checkpoint", download)

    assert resolve_checkpoint(None) == checkpoint
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "download status\n"


def test_explicit_checkpoint_skips_release_download(tmp_path, monkeypatch):
    checkpoint = tmp_path / "custom.ckpt"
    monkeypatch.setattr(
        predictor_module,
        "download_release_checkpoint",
        lambda: pytest.fail("explicit checkpoint should not trigger a download"),
    )

    assert resolve_checkpoint(checkpoint) == checkpoint


def test_viz_writes_a_figure(images, tmp_path, capsys):
    pytest.importorskip("cairosvg")
    pytest.importorskip("matplotlib")
    out = tmp_path / "figures" / "sxs.png"

    main(["--checkpoint", "unused.ckpt", "--viz", str(out), *map(str, images)])

    assert out.exists() and out.stat().st_size > 0   # parent dir created too
    assert str(out) in capsys.readouterr().out


def test_missing_viz_extras_explains_the_install(tmp_path, monkeypatch):
    """A missing viz extra becomes an actionable SystemExit, not a raw ImportError."""
    import builtins

    real_import = builtins.__import__

    def fail_render(name, *args, **kwargs):
        if name == "chessqueries.viz.render":
            raise ImportError("No module named 'cairosvg'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_render)
    with pytest.raises(SystemExit, match="poetry install --with viz"):
        predictor_module._write_figure([], tmp_path / "x.png")
