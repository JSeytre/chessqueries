"""ShotEmbedder + Pca: the descriptor space and its blank/determinism contracts.

The heavy DINOv2 forward pass is replaced by a tiny deterministic fake model, so these
run with no GPU and no pretrained-weight download — they exercise the grid-pool + PCA
projection path and the blank/1-D contracts the registry relies on.
"""

import numpy as np
import pytest

from chessqueries.annotate import embedding
from chessqueries.annotate.embedding import Pca, ShotEmbedder, shot_descriptors


def test_pca_fit_transform_shapes_and_norm():
    x = np.random.default_rng(0).standard_normal((60, 300)).astype(np.float32)
    pca = Pca.fit(x, 16)
    assert pca.dim == 16 and pca.components.shape == (16, 300)
    z = pca.transform(x)
    assert z.shape == (60, 16)
    assert np.allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-4)
    # a single 1-D input round-trips to 1-D.
    assert pca.transform(x[0]).shape == (16,)


def test_pca_blank_row_stays_zero():
    # A zero (blank) input must not become a spurious non-zero descriptor: the registry
    # skips zero-norm rows, so mean-centering a blank into a real vector would break that.
    x = np.random.default_rng(1).standard_normal((5, 40)).astype(np.float32)
    x[2] = 0.0
    z = Pca.fit(x, 8).transform(x)
    assert np.allclose(z[2], 0.0)
    assert not np.allclose(z[0], 0.0)


def test_pca_save_load_roundtrip(tmp_path):
    x = np.random.default_rng(2).standard_normal((40, 50)).astype(np.float32)
    pca = Pca.fit(x, 12)
    p = tmp_path / "pca.npz"
    pca.save(p)
    back = Pca.load(p)
    assert np.allclose(back.transform(x), pca.transform(x), atol=1e-5)


def test_pca_validates_shapes():
    with pytest.raises(ValueError):
        Pca(mean=np.zeros(10, np.float32), components=np.zeros((4, 9), np.float32))  # dim mismatch
    with pytest.raises(ValueError):
        Pca(mean=np.zeros((2, 2), np.float32), components=np.zeros((4, 4), np.float32))  # 2-D mean


class _FakeDino:
    """Deterministic stand-in for the DINOv2 model: 1 prefix token + 49 patch tokens of
    width C, tokens fixed by each frame's mean so the same frame always embeds the same."""

    num_prefix_tokens = 1

    def __init__(self, channels: int) -> None:
        self.c = channels

    def forward_features(self, x):
        import torch

        b = x.shape[0]
        seed = x.reshape(b, -1).mean(dim=1).view(b, 1, 1)  # per-frame scalar
        base = torch.arange(50 * self.c, dtype=torch.float32).reshape(1, 50, self.c)
        return base + seed  # [B, 1+49, C]


def _embedder_with_fake(tmp_path, channels=16):
    # grid3 dim = GRID*GRID*channels; fit a PCA of that width so transform lines up.
    grid_dim = embedding.GRID * embedding.GRID * channels
    pca = Pca.fit(
        np.random.default_rng(3).standard_normal((grid_dim + 5, grid_dim)).astype(np.float32), 8
    )
    p = tmp_path / "pca.npz"
    pca.save(p)
    emb = ShotEmbedder(device="cpu", pca_path=p)
    emb._model = _FakeDino(channels)  # skip timm load / weight download
    emb._n_prefix = _FakeDino.num_prefix_tokens
    return emb


def _frame(val, rng=None):
    if rng is None:  # uniform -> blank (gray std 0 < BLANK_STD)
        return np.full((64, 64, 3), val, dtype=np.uint8)
    return (rng.random((64, 64, 3)) * 255).astype(np.uint8)


def test_embedder_output_dim_norm_and_determinism(tmp_path):
    emb = _embedder_with_fake(tmp_path)
    rng = np.random.default_rng(4)
    frames = [_frame(0, rng), _frame(0, rng)]
    z = emb.embed_frames(frames)
    assert z.shape == (2, emb.pca.dim)
    assert np.allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-4)
    # deterministic: same frames -> identical descriptors.
    assert np.allclose(z, emb.embed_frames(frames))


def test_embedder_blank_frame_is_zero(tmp_path):
    emb = _embedder_with_fake(tmp_path)
    rng = np.random.default_rng(5)
    z = emb.embed_frames([_frame(0, rng), _frame(128)])  # second is uniform -> blank
    assert not np.allclose(z[0], 0.0)
    assert np.allclose(z[1], 0.0)


def test_embed_frames_empty(tmp_path):
    emb = _embedder_with_fake(tmp_path)
    assert emb.embed_frames([]).shape == (0, emb.pca.dim)


def test_shot_descriptors_delegates_to_default_embedder(monkeypatch):
    sentinel = np.zeros((3, embedding.DESCRIPTOR_DIM), dtype=np.float32)
    calls = {}

    class _Stub:
        def embed_shots(self, video, shots, *, show_progress=False):
            calls["args"] = (video, shots, show_progress)
            return sentinel

    monkeypatch.setattr(embedding, "default_embedder", lambda: _Stub())
    out = shot_descriptors("VIDEO", ["s0", "s1"], show_progress=True)
    assert out is sentinel
    assert calls["args"] == ("VIDEO", ["s0", "s1"], True)
