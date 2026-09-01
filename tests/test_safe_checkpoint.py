"""Safe release weights carry enough configuration for strict model loading."""

from __future__ import annotations

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from chessqueries.models import checkpoint


def _metadata(**overrides: str) -> dict[str, str]:
    values = {
        "format": checkpoint.FORMAT_NAME,
        "encoder_name": "tiny-encoder",
        "decoder_layers": "2",
        "nheads": "4",
        "aux_heads": "false",
        "drop_path_rate": "0.1",
        "head_type": "query",
        "freeze_encoder": "false",
        "resolution": "64",
        "normalization": "imagenet",
        "license": checkpoint.WEIGHTS_LICENSE,
    }
    values.update(overrides)
    return values


class _TinyModel(nn.Module):
    last_kwargs = None

    def __init__(self, **kwargs) -> None:
        super().__init__()
        type(self).last_kwargs = kwargs
        self.weight = nn.Parameter(torch.zeros(2, 3))


def test_load_safetensors_reconstructs_and_strictly_loads(tmp_path, monkeypatch):
    path = tmp_path / "model.safetensors"
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    save_file({"weight": expected}, path, metadata=_metadata())
    monkeypatch.setattr(checkpoint, "ChessQueriesModel", _TinyModel)

    model = checkpoint.load_safetensors_model(path)

    assert torch.equal(model.weight, expected)
    assert _TinyModel.last_kwargs == {
        "encoder_name": "tiny-encoder",
        "pretrained": False,
        "freeze_encoder": False,
        "decoder_layers": 2,
        "nheads": 4,
        "aux_heads": False,
        "drop_path_rate": 0.1,
        "head_type": "query",
    }
    assert not model.training


def test_load_safetensors_rejects_missing_configuration(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.zeros(1)}, path, metadata={"format": checkpoint.FORMAT_NAME})

    try:
        checkpoint.read_safetensors_metadata(path)
    except ValueError as exc:
        assert "metadata is missing" in str(exc)
        assert "encoder_name" in str(exc)
    else:  # pragma: no cover - failure message is clearer than a bare assert
        raise AssertionError("incomplete metadata was accepted")


def test_export_strips_lightning_and_optimizer_state(tmp_path):
    source = tmp_path / "source.ckpt"
    output = tmp_path / "model.safetensors"
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch.save(
        {
            "hyper_parameters": {
                "encoder_name": "tiny-encoder",
                "decoder_layers": 2,
                "freeze_encoder": False,
                "aux_weight": 0.0,
            },
            "state_dict": {
                "model.weight": expected,
                "class_weight": torch.ones(13),
            },
            "optimizer_states": [{"large": torch.ones(10)}],
        },
        source,
    )

    metadata = checkpoint.export_lightning_safetensors(source, output, resolution=64)

    with safe_open(output, framework="pt", device="cpu") as handle:
        assert list(handle.keys()) == ["weight"]
        assert torch.equal(handle.get_tensor("weight"), expected)
        assert handle.metadata() == metadata
    assert metadata["source_checkpoint_sha256"] == checkpoint.sha256_file(source)
