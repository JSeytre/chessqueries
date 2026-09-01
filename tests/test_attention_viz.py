"""extract_attention / extract_encoder_attention return maps that are shaped,
normalized, and side-effect-free (logits match a plain forward; the model is
left untouched)."""
import pytest
import torch
import torch.nn.functional as F

from chessqueries.models.chessqueries_model import ChessQueriesModel
from chessqueries.viz.attention import (
    extract_attention,
    extract_encoder_attention,
    pool_bins,
)


@pytest.fixture(scope="module")
def model():
    return ChessQueriesModel(pretrained=False, freeze_encoder=True).eval()


@pytest.fixture(scope="module")
def batch():
    return torch.randn(2, 3, 224, 224)


@pytest.fixture(scope="module")
def extracted(model, batch):
    return extract_attention(model, batch)


def test_shapes(model, extracted):
    logits, maps = extracted
    n_layers = len(model.decoder.layers)
    heads = model.decoder.layers[0].multihead_attn.num_heads
    grid = 224 // 14
    n_tokens = model.encoder.num_prefix_tokens + grid * grid

    assert logits.shape == (2, 64, 13)
    assert len(maps.cross) == len(maps.self_attn) == n_layers
    assert maps.cross[0].shape == (2, heads, 64, n_tokens)
    assert maps.self_attn[0].shape == (2, heads, 64, 64)
    assert maps.grid_hw == (grid, grid)


def test_cross_grid_is_normalized(extracted):
    _, maps = extracted
    grid = maps.cross_grid()  # (B, 64, H, W), head-averaged, prefix dropped
    assert grid.shape[:2] == (2, 64)
    assert torch.allclose(grid.sum(dim=(-2, -1)), torch.ones(2, 64), atol=1e-5)
    per_head = maps.cross_grid(layer=0, head=0)
    assert per_head.shape == grid.shape


def test_centroids_in_unit_square(extracted):
    _, maps = extracted
    c = maps.cross_centroids()
    assert c.shape == (2, 64, 2)
    assert (c >= 0).all() and (c <= 1).all()


def test_logits_match_plain_forward_and_model_is_restored(model, batch, extracted):
    logits, _ = extracted
    with torch.no_grad():
        plain = model(batch)  # runs after extraction: forward must be restored
    assert torch.allclose(logits, plain, atol=1e-4)
    for layer in model.decoder.layers:
        assert "forward" not in layer.multihead_attn.__dict__
        assert "forward" not in layer.self_attn.__dict__


@pytest.fixture(scope="module")
def linear_model():
    return ChessQueriesModel(pretrained=False, freeze_encoder=True, head_type="linear").eval()


@pytest.fixture(scope="module")
def encoder_extracted(linear_model, batch):
    return extract_encoder_attention(linear_model, batch, layers=(-1, 0))


def test_pool_bins_match_adaptive_avg_pool():
    x = torch.randn(1, 3, 16, 16)
    pooled = F.adaptive_avg_pool2d(x, (8, 8))
    manual = torch.stack([
        torch.stack([x[..., r0:r1, c0:c1].mean(dim=(-2, -1)) for c0, c1 in pool_bins(16)], dim=-1)
        for r0, r1 in pool_bins(16)
    ], dim=-2)
    assert torch.allclose(pooled, manual, atol=1e-6)


def test_encoder_attention_shapes(linear_model, encoder_extracted):
    logits, maps = encoder_extracted
    grid = 224 // 14
    heads = linear_model.encoder.blocks[-1].attn.num_heads
    n_tokens = linear_model.encoder.num_prefix_tokens + grid * grid

    assert logits.shape == (2, 64, 13)
    assert maps.layers == (0, len(linear_model.encoder.blocks) - 1)
    assert len(maps.self_attn) == 2
    assert maps.self_attn[-1].shape == (2, heads, n_tokens, n_tokens)
    assert maps.grid_hw == (grid, grid)


def test_cell_grid_is_normalized(encoder_extracted):
    _, maps = encoder_extracted
    grid = maps.cell_grid()  # (B, 64, H, W), head-averaged, prefix dropped
    assert grid.shape[:2] == (2, 64)
    assert torch.allclose(grid.sum(dim=(-2, -1)), torch.ones(2, 64), atol=1e-5)
    per_head = maps.cell_grid(layer=0, head=0)
    assert per_head.shape == grid.shape
    c = maps.cell_centroids()
    assert c.shape == (2, 64, 2)
    assert (c >= 0).all() and (c <= 1).all()


def test_encoder_logits_match_plain_forward_and_model_is_restored(linear_model, batch, encoder_extracted):
    logits, _ = encoder_extracted
    with torch.no_grad():
        plain = linear_model(batch)  # runs after extraction: fused path must be back
    assert torch.allclose(logits, plain, atol=1e-4)
    for blk in linear_model.encoder.blocks:
        assert blk.attn.fused_attn
        assert not blk.attn.attn_drop._forward_hooks
