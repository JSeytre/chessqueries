"""Attention extraction for the ChessQueries model.

Two sources, one per head type: the query decoder's query->patch cross-attention
(`extract_attention`), and — for the linear (grid-pool) head, which has no
decoder — the encoder's own self-attention read out per pooled cell
(`extract_encoder_attention`)."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:  # annotation-only: viz has no runtime dependency on models
    from chessqueries.models.chessqueries_model import ChessQueriesModel


@dataclass(frozen=True)
class AttentionMaps:
    """Attention captured from one forward pass.

    cross[i]:     (B, heads, 64, N_tokens) — layer-i query->memory attention.
    self_attn[i]: (B, heads, 64, 64)       — layer-i query->query attention.
    grid_hw:      encoder patch-grid (rows, cols) at this input size.
    num_prefix_tokens: leading memory tokens that are not patches (CLS etc.).

    Query index == square index in FEN order (a8=0 .. h1=63), by construction.
    """

    cross: tuple[Tensor, ...]
    self_attn: tuple[Tensor, ...]
    grid_hw: tuple[int, int]
    num_prefix_tokens: int

    def cross_grid(self, layer: int = -1, head: int | None = None) -> Tensor:
        """Cross-attention as spatial maps: (B, 64, H, W).

        Prefix (non-patch) tokens are dropped and the remaining mass is
        renormalized over patches. ``head=None`` averages over heads.
        """
        w = self.cross[layer]
        w = w.mean(dim=1) if head is None else w[:, head]
        w = w[..., self.num_prefix_tokens:]
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rows, cols = self.grid_hw
        return w.reshape(w.shape[0], w.shape[1], rows, cols)

    def cross_centroids(self, layer: int = -1) -> Tensor:
        """Attention centroid per query in normalized [0,1]² image coords:
        (B, 64, 2) as (x, y). Useful for the query->square alignment plot."""
        return grid_centroids(self.cross_grid(layer))


def grid_centroids(grid: Tensor) -> Tensor:
    """Centroids of spatial attention maps (B, 64, H, W) in normalized [0,1]²
    image coords: (B, 64, 2) as (x, y)."""
    b, q, rows, cols = grid.shape
    ys = (torch.arange(rows, device=grid.device, dtype=grid.dtype) + 0.5) / rows
    xs = (torch.arange(cols, device=grid.device, dtype=grid.dtype) + 0.5) / cols
    mass = grid.sum(dim=(-2, -1)).clamp_min(1e-12)
    cy = (grid.sum(dim=-1) * ys).sum(dim=-1) / mass
    cx = (grid.sum(dim=-2) * xs).sum(dim=-1) / mass
    return torch.stack([cx, cy], dim=-1)


def pool_bins(size: int, out: int = 8) -> list[tuple[int, int]]:
    """Half-open index ranges of ``F.adaptive_avg_pool2d``'s bins along one
    dimension (size -> out). Mirrors PyTorch's start/end formula so cell i's
    token bin is exactly the region averaged into pooled cell i."""
    return [(i * size // out, -(-(i + 1) * size // out)) for i in range(out)]


@dataclass(frozen=True)
class EncoderAttentionMaps:
    """Encoder self-attention captured from one forward pass (linear-head viz).

    self_attn[j]: (B, heads, N, N) — self-attention of captured block layers[j],
                  rows = attending tokens, cols = attended tokens (incl. prefix).
    layers:       encoder block indices captured (same order as ``self_attn``).
    grid_hw:      encoder patch-grid (rows, cols) at this input size.
    num_prefix_tokens: leading tokens that are not patches (CLS etc.).

    Cell index == the linear head's output order: raster over the 8x8 pooled
    grid, which is the same order the labels/loss use.
    """

    self_attn: tuple[Tensor, ...]
    layers: tuple[int, ...]
    grid_hw: tuple[int, int]
    num_prefix_tokens: int

    def cell_grid(self, layer: int = -1, head: int | None = None) -> Tensor:
        """Per-cell attention as spatial maps: (B, 64, H, W).

        For each pooled 8x8 cell, the (head-averaged) attention of the patch
        tokens in that cell's pooling bin over all patch tokens; prefix columns
        are dropped and the remaining mass renormalized. ``layer`` indexes the
        *captured* layers (default: last captured)."""
        w = self.self_attn[layer]
        w = w.mean(dim=1) if head is None else w[:, head]
        rows, cols = self.grid_hw
        p = self.num_prefix_tokens
        by_token = w[:, p:, :].reshape(w.shape[0], rows, cols, -1)
        cells = [
            by_token[:, r0:r1, c0:c1].mean(dim=(1, 2))
            for r0, r1 in pool_bins(rows)
            for c0, c1 in pool_bins(cols)
        ]
        w = torch.stack(cells, dim=1)[..., p:]  # (B, 64, patches)
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return w.reshape(w.shape[0], 64, rows, cols)

    def cell_centroids(self, layer: int = -1) -> Tensor:
        """Attention centroid per pooled cell, (B, 64, 2) as (x, y) in [0,1]²."""
        return grid_centroids(self.cell_grid(layer))


@contextmanager
def _force_weights(mha: nn.MultiheadAttention, store: list[Tensor]):
    """Temporarily make an MHA module return per-head weights and record them.

    ``nn.TransformerDecoderLayer`` calls its attention blocks with
    ``need_weights=False``; overriding the kwargs here flips both flags without
    touching the layer's math.
    """
    orig = mha.forward

    def wrapped(query, key, value, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        out, weights = orig(query, key, value, **kwargs)
        store.append(weights.detach())
        return out, weights

    mha.forward = wrapped
    try:
        yield
    finally:
        del mha.forward  # restore the bound method


def _patch_grid(encoder: nn.Module, x: Tensor) -> tuple[int, int]:
    ph, pw = encoder.patch_embed.patch_size
    return x.shape[-2] // ph, x.shape[-1] // pw


@torch.no_grad()
def extract_attention(model: ChessQueriesModel, x: Tensor) -> tuple[Tensor, AttentionMaps]:
    """Run a forward pass and capture all decoder attention maps.

    Returns ``(logits, maps)`` where logits match ``model(x)``.
    """
    model.eval()
    cross: list[Tensor] = []
    self_attn: list[Tensor] = []
    with ExitStack() as stack:
        for layer in model.decoder.layers:
            stack.enter_context(_force_weights(layer.multihead_attn, cross))
            stack.enter_context(_force_weights(layer.self_attn, self_attn))
        logits = model(x)

    rows, cols = _patch_grid(model.encoder, x)
    num_prefix = model.encoder.num_prefix_tokens
    n_tokens = cross[0].shape[-1]
    if n_tokens != num_prefix + rows * cols:
        raise RuntimeError(
            f"memory tokens ({n_tokens}) != prefix ({num_prefix}) + grid ({rows}x{cols})"
        )
    return logits, AttentionMaps(tuple(cross), tuple(self_attn), (rows, cols), num_prefix)


@torch.no_grad()
def extract_encoder_attention(
    model: ChessQueriesModel, x: Tensor, layers: tuple[int, ...] = (-1,)
) -> tuple[Tensor, EncoderAttentionMaps]:
    """Run a forward pass and capture encoder self-attention for ``layers``.

    Works for either head type (it only touches the encoder), but exists for the
    linear head, which has no decoder to attend with. Only the requested blocks
    are captured — a full (N, N) map per layer is large — and weights are moved
    to CPU as they are recorded. Returns ``(logits, maps)`` with logits matching
    ``model(x)``.

    timm's fused attention never materializes the weights, so the captured
    blocks are temporarily switched to the math path (identical result, plus a
    softmax matrix to hook) and restored afterwards.
    """
    model.eval()
    blocks = list(model.encoder.blocks)
    idxs = tuple(sorted({i % len(blocks) for i in layers}))
    captured: dict[int, Tensor] = {}
    handles, fused_prior = [], {}

    def _hook(index: int):
        def hook(_mod, _inp, out):
            captured[index] = out.detach().cpu()
        return hook

    for i in idxs:
        attn = blocks[i].attn
        fused_prior[i] = attn.fused_attn
        attn.fused_attn = False
        handles.append(attn.attn_drop.register_forward_hook(_hook(i)))
    try:
        logits = model(x)
    finally:
        for h in handles:
            h.remove()
        for i, fused in fused_prior.items():
            blocks[i].attn.fused_attn = fused

    rows, cols = _patch_grid(model.encoder, x)
    num_prefix = model.encoder.num_prefix_tokens
    n_tokens = captured[idxs[0]].shape[-1]
    if n_tokens != num_prefix + rows * cols:
        raise RuntimeError(
            f"encoder tokens ({n_tokens}) != prefix ({num_prefix}) + grid ({rows}x{cols})"
        )
    return logits, EncoderAttentionMaps(
        tuple(captured[i] for i in idxs), idxs, (rows, cols), num_prefix
    )
