"""Guards for the ChessReD baseline reproduction fix (upstream issue
tmasouris/end-to-end-chess-recognition#5): we ship the authors' 1024x1024 images
as the single image set, so the inference transform must NOT resize."""

import torch

from chessqueries.data.chessred import IMAGE_SIZE
from chessqueries.models.chessred_resnext import INFERENCE_TRANSFORM


def test_inference_transform_does_not_resize():
    # No runtime Resize: an already-1024 image is passed through unchanged, and a
    # differently-sized one is NOT coerced to 1024 (which is what collapsed the
    # checkpoint on the raw images).
    for size in (IMAGE_SIZE, 800):
        img = torch.randint(0, 256, (3, size, size), dtype=torch.float32)
        out = INFERENCE_TRANSFORM(img)
        assert out.shape == (3, size, size), f"{size} -> {tuple(out.shape)}"


def test_checkpoint_missing_keys_raise(tmp_path):
    """A checkpoint that doesn't cover the model must fail loudly — with
    strict=False the model would otherwise evaluate on random weights."""
    import pytest

    from chessqueries.models.chessred_resnext import ChessReDResNeXt

    state = ChessReDResNeXt().state_dict()
    state.pop(next(iter(state)))
    p = tmp_path / "bad.ckpt"
    torch.save({"state_dict": state}, p)
    with pytest.raises(ValueError, match="missing"):
        ChessReDResNeXt.from_checkpoint(p)


def test_checkpoint_unexpected_keys_warn_but_load_safely(tmp_path, monkeypatch):
    import pytest

    from chessqueries.models.chessred_resnext import ChessReDResNeXt

    state = ChessReDResNeXt().state_dict()
    state["criterion.weight"] = torch.zeros(1)
    p = tmp_path / "extra.ckpt"
    torch.save({"state_dict": state}, p)

    real_load = torch.load
    load_kwargs = []

    def recording_load(*args, **kwargs):
        load_kwargs.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    with pytest.warns(UserWarning, match="unused"):
        model = ChessReDResNeXt.from_checkpoint(p)
    assert load_kwargs[0]["weights_only"] is True
    assert not model.training  # loaded and in eval mode
