"""The encoder_name knob selects the backbone (default ViT-B, opt-in ViT-L)."""
from chessqueries.models.chessqueries_model import ChessQueriesModel


def test_default_is_vit_base():
    m = ChessQueriesModel(pretrained=False, freeze_encoder=True)
    assert m.encoder.embed_dim == 768  # ViT-B/14


def test_encoder_name_selects_vit_large():
    m = ChessQueriesModel(encoder_name="vit_large_patch14_dinov2.lvd142m",
                      pretrained=False, freeze_encoder=True)
    assert m.encoder.embed_dim == 1024  # ViT-L/14 — decoder adapts to enc_dim
