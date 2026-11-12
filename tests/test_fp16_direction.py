"""Released safetensors preserve board and direction through the public predictor."""

from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from chessqueries.core import Color
from chessqueries.models.checkpoint import load_safetensors_model, read_safetensors_metadata
from chessqueries.models.chessqueries_model import ChessQueriesModel
from chessqueries.models.direction import unit_direction
from chessqueries.models.predictor import Predictor


class Encoder(torch.nn.Module):
    embed_dim = 8

    def forward_features(self, images):
        return images.mean((2, 3))[:, :, None].expand(-1, -1, self.embed_dim)


@pytest.fixture
def released_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("chessqueries.models.chessqueries_model.timm.create_model",
                        lambda *a, **k: Encoder())

    def create(mode="auxiliary", **overrides):
        model = ChessQueriesModel(pretrained=False, decoder_layers=1, nheads=2,
                                  orientation_mode=mode).half().eval()
        if mode == "auxiliary":
            with torch.no_grad():
                for parameter in model.orientation_predictor.parameters():
                    parameter.zero_()
                model.orientation_predictor.net[-1].bias.copy_(torch.tensor([-3., 4.]))
        metadata = {
            "format": "chessqueries-safetensors-v1", "precision": "fp16",
            "encoder_name": "tiny", "decoder_layers": "1", "nheads": "2",
            "aux_heads": "false", "drop_path_rate": "0", "head_type": "query",
            "freeze_encoder": "true", "resolution": "8", "normalization": "imagenet",
            "license": "PolyForm-Noncommercial-1.0.0",
        }
        if mode != "none":
            metadata.update(orientation_mode=mode, orientation_representation="direction")
        metadata.update(overrides)
        path = tmp_path / "model.safetensors"
        save_file(model.state_dict(), path, metadata=metadata)
        return path

    return create


def test_fp16_checkpoint_predicts_board_and_direction_in_one_encoder_pass(released_checkpoint, monkeypatch):
    path = released_checkpoint()
    loaded = load_safetensors_model(path)
    assert next(loaded.parameters()).dtype == torch.float16
    predictor = Predictor.from_checkpoint(path, resolution=8, device="cpu")
    assert next(predictor.model.parameters()).dtype == torch.float32
    monkeypatch.setattr(predictor, "prepare_image", lambda _: torch.ones(3, 8, 8))
    with patch.object(predictor.model, "encode", wraps=predictor.model.encode) as encode:
        predictions = predictor.predict(["a", "b"], batch_size=2)
    encode.assert_called_once()
    with torch.no_grad():
        expected = loaded.float().predict_labels(torch.ones(2, 3, 8, 8)).tolist()
    assert [p.board.labels for p in predictions] == expected
    for result in predictions:
        assert result.orientation is Color.BLACK
        assert result.viewing_direction == pytest.approx((-0.6, 0.8))
        assert result.lichess_url.endswith("?color=black")


def test_legacy_safetensors_still_have_no_direction(released_checkpoint):
    path = released_checkpoint("none")
    model = load_safetensors_model(path).float()
    assert model.recognize(torch.ones(1, 3, 8, 8)).directions is None
    assert not any(k.startswith("orientation_predictor.") for k in model.state_dict())


@pytest.mark.parametrize("metadata", [
    {"orientation_mode": "conditioned"},
    {"orientation_representation": "axis_direction"},
])
def test_unsupported_release_orientation_rejected(released_checkpoint, metadata):
    path = released_checkpoint(**metadata)
    with pytest.raises(ValueError):
        read_safetensors_metadata(path)


def test_missing_direction_weights_are_rejected(released_checkpoint):
    path = released_checkpoint("none", orientation_mode="auxiliary")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_safetensors_model(path)


def test_fp16_direction_normalizes_in_float32():
    actual = unit_direction(torch.tensor([[0., 0.], [-3., 4.]], dtype=torch.float16))
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, torch.tensor([[1., 0.], [-.6, .8]]))
