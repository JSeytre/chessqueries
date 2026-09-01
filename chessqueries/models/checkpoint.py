"""Safe, tensor-only checkpoints for released ChessQueries models."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from chessqueries.models.chessqueries_model import ChessQueriesModel, HeadType

FORMAT_NAME = "chessqueries-safetensors-v1"
WEIGHTS_LICENSE = "PolyForm-Noncommercial-1.0.0"
REQUIRED_METADATA = {
    "format",
    "encoder_name",
    "decoder_layers",
    "nheads",
    "aux_heads",
    "drop_path_rate",
    "head_type",
    "freeze_encoder",
    "resolution",
    "normalization",
    "license",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: str, field: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"invalid safetensors {field}: {value!r}")


def read_safetensors_metadata(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    missing = sorted(REQUIRED_METADATA - metadata.keys())
    if missing:
        raise ValueError(f"ChessQueries safetensors metadata is missing: {missing}")
    if metadata["format"] != FORMAT_NAME:
        raise ValueError(f"unsupported ChessQueries safetensors format: {metadata['format']!r}")
    return metadata


def load_safetensors_model(path: Path, *, device: str = "cpu") -> ChessQueriesModel:
    """Reconstruct a ChessQueries model from a tensor-only release artifact."""
    path = Path(path)
    metadata = read_safetensors_metadata(path)
    model = ChessQueriesModel(
        encoder_name=metadata["encoder_name"],
        pretrained=False,
        freeze_encoder=_as_bool(metadata["freeze_encoder"], "freeze_encoder"),
        decoder_layers=int(metadata["decoder_layers"]),
        nheads=int(metadata["nheads"]),
        aux_heads=_as_bool(metadata["aux_heads"], "aux_heads"),
        drop_path_rate=float(metadata["drop_path_rate"]),
        head_type=metadata["head_type"],
    )
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    return model.to(device).eval()


def _metadata_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def export_lightning_safetensors(
    checkpoint: Path,
    output: Path,
    *,
    resolution: int,
    normalization: str = "imagenet",
) -> dict[str, str]:
    """Export only ``LitChessQueriesModel.model`` from a Lightning checkpoint."""
    # Importing the module registers the project's trusted enum values for
    # torch's weights-only checkpoint loader.
    from chessqueries.train import lit as _lit  # noqa: F401

    checkpoint = Path(checkpoint)
    output = Path(output)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    hparams = dict(payload.get("hyper_parameters", {}))
    lightning_state = payload.get("state_dict")
    if not isinstance(lightning_state, dict):
        raise ValueError("Lightning checkpoint has no state_dict")
    state = {
        key.removeprefix("model."): tensor.contiguous()
        for key, tensor in lightning_state.items()
        if key.startswith("model.") and isinstance(tensor, torch.Tensor)
    }
    if not state:
        raise ValueError("Lightning checkpoint contains no model.* tensors")

    metadata = {
        "format": FORMAT_NAME,
        "encoder_name": _metadata_value(hparams.get("encoder_name")),
        "decoder_layers": str(hparams.get("decoder_layers", 4)),
        "nheads": "8",
        "aux_heads": str(float(hparams.get("aux_weight", 0.0)) > 0).lower(),
        "drop_path_rate": str(hparams.get("drop_path_rate", 0.0)),
        "head_type": _metadata_value(hparams.get("head_type", HeadType.QUERY)),
        "freeze_encoder": str(bool(hparams.get("freeze_encoder", False))).lower(),
        "resolution": str(resolution),
        "normalization": normalization,
        "license": WEIGHTS_LICENSE,
        "source_checkpoint_sha256": sha256_file(checkpoint),
    }
    if metadata["encoder_name"] in {"None", ""}:
        raise ValueError("Lightning checkpoint does not record encoder_name")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(dict(sorted(state.items())), str(output), metadata=metadata)
    return metadata
