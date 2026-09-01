"""ChessQueries model: ViT encoder + 64 semantic square queries + transformer decoder
with a shared per-square classification head."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import timm
import torch
from torch import nn
from torch.nn import functional as F

from chessqueries.core import NUM_PIECES, ALL_SQUARES, Color, Piece, PieceType
from chessqueries.models.base import BoardRecognizer


class HeadType(str, Enum):
    """How the 64 squares are read off the encoder."""

    QUERY = "query"    # learned square queries + transformer decoder
    LINEAR = "linear"  # 8x8 avg-pool of the patch grid + shared head (no-decoder ablation)

NUM_CLASSES = NUM_PIECES
DEFAULT_ENCODER = "vit_base_patch14_dinov2.lvd142m"

# Aux-head target decompositions of the 13-class label (list index == Piece value).
_TYPE_ORDER = [PieceType.PAWN, PieceType.KNIGHT, PieceType.BISHOP,
               PieceType.ROOK, PieceType.QUEEN, PieceType.KING]
NUM_TYPES = len(_TYPE_ORDER) + 1   # 0=empty, 1..6 = piece type
NUM_COLORS = 3                     # 0=empty, 1=white, 2=black
LABEL_TO_TYPE = [0 if p.is_empty else _TYPE_ORDER.index(p.piece_type) + 1 for p in Piece]
LABEL_TO_COLOR = [0 if p.is_empty else (1 if p.color is Color.WHITE else 2) for p in Piece]


@dataclass(frozen=True)
class SquareGeometry:
    """Embedding-lookup indices for the 64 squares in FEN order (a8=0..h1=63), one
    (64,) long tensor per property: file 0-7, rank 0-7, color 0=light/1=dark."""

    files: torch.Tensor
    ranks: torch.Tensor
    colors: torch.Tensor


def _square_geometry() -> SquareGeometry:
    """Per-square file/rank/color indices in FEN order (a8=0..h1=63)."""
    return SquareGeometry(
        files=torch.tensor([sq.file for sq in ALL_SQUARES], dtype=torch.long),
        ranks=torch.tensor([sq.rank - 1 for sq in ALL_SQUARES], dtype=torch.long),
        colors=torch.tensor([0 if sq.is_light else 1 for sq in ALL_SQUARES], dtype=torch.long),
    )


class SquareQueries(nn.Module):
    """64 learned queries = square_id + file + rank + square_color embeddings."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.square_id = nn.Embedding(64, dim)
        self.file_emb = nn.Embedding(8, dim)
        self.rank_emb = nn.Embedding(8, dim)
        self.color_emb = nn.Embedding(2, dim)
        geometry = _square_geometry()
        self.register_buffer("file_idx", geometry.files)
        self.register_buffer("rank_idx", geometry.ranks)
        self.register_buffer("color_idx", geometry.colors)
        self.register_buffer("square_ids", torch.arange(64))

    def forward(self, batch_size: int) -> torch.Tensor:
        q = (
            self.square_id(self.square_ids)
            + self.file_emb(self.file_idx)
            + self.rank_emb(self.rank_idx)
            + self.color_emb(self.color_idx)
        )  # (64, dim)
        return q.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 64, dim)


class ChessQueriesModel(BoardRecognizer):
    def __init__(
        self,
        encoder_name: str = DEFAULT_ENCODER,
        pretrained: bool = True,
        freeze_encoder: bool = True,
        decoder_layers: int = 4,
        nheads: int = 8,
        aux_heads: bool = False,
        drop_path_rate: float = 0.0,
        head_type: HeadType | str = HeadType.QUERY,
    ) -> None:
        super().__init__()
        # str accepted for checkpoint hparams round-trips; unknown values raise here.
        self.head_type = HeadType(head_type)
        self.encoder = timm.create_model(
            encoder_name, pretrained=pretrained, num_classes=0, dynamic_img_size=True,
            drop_path_rate=drop_path_rate,
        )
        dim = self.encoder.embed_dim

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Linear-head (no-query-decoder) ablation: reads the 64 squares off an 8x8
        # average-pool of the encoder patch grid, so only the shared head remains
        # and spatial correspondence is kept.
        if self.head_type is HeadType.QUERY:
            self.queries = SquareQueries(dim)
            layer = nn.TransformerDecoderLayer(
                d_model=dim, nhead=nheads, dim_feedforward=4 * dim, batch_first=True
            )
            self.decoder = nn.TransformerDecoder(layer, num_layers=decoder_layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, NUM_CLASSES)  # shared across all 64 squares

        # Auxiliary piece-type / colour heads (multi-task): extra structured
        # supervision targeting tall-piece->pawn confusions. Off by default.
        self.aux_heads = aux_heads
        if aux_heads:
            self.type_head = nn.Linear(dim, NUM_TYPES)
            self.color_head = nn.Linear(dim, NUM_COLORS)
            self.register_buffer("label_to_type", torch.tensor(LABEL_TO_TYPE, dtype=torch.long))
            self.register_buffer("label_to_color", torch.tensor(LABEL_TO_COLOR, dtype=torch.long))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # forward_features -> (B, num_tokens, enc_dim), tokens incl. prefix.
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                feats = self.encoder.forward_features(x)
        else:
            feats = self.encoder.forward_features(x)
        return feats

    def features(self, x: torch.Tensor) -> torch.Tensor:
        memory = self.encode(x)  # (B, N, dim)
        if self.head_type is HeadType.LINEAR:
            patch = memory[:, self.encoder.num_prefix_tokens:, :]  # drop CLS/register tokens
            b, p, d = patch.shape
            g = int(round(p**0.5))  # square patch grid (res/patch), row-major
            grid = patch.transpose(1, 2).reshape(b, d, g, g)  # (B, dim, g, g)
            pooled = F.adaptive_avg_pool2d(grid, (8, 8))  # (B, dim, 8, 8) -> one cell per square
            h = pooled.flatten(2).transpose(1, 2)  # (B, 64, dim), FEN order a8..h1
            return self.norm(h)
        q = self.queries(x.shape[0])  # (B, 64, dim)
        h = self.decoder(q, memory)  # (B, 64, dim)
        return self.norm(h)  # (B, 64, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))  # (B, 64, 13)

    def forward_aux(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Main + auxiliary logits (training only). Main is the same tensor
        `forward` returns; `predict_labels` still uses `forward`."""
        h = self.features(x)
        out = {"main": self.head(h)}
        if self.aux_heads:
            out["type"] = self.type_head(h)
            out["color"] = self.color_head(h)
        return out

    @torch.no_grad()
    def predict_labels(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=-1)  # (B, 64), already our label space
