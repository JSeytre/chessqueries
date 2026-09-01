"""Minimal independent implementation of the paper's ChessQueries model."""
from __future__ import annotations

import timm
import torch
from torch import nn


NUM_CLASSES = 13


def _square_index_tables() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return file, rank, and color lookup indices in FEN square order."""
    files = torch.tensor([index % 8 for index in range(64)], dtype=torch.long)
    ranks = torch.tensor([(8 - index // 8) - 1 for index in range(64)], dtype=torch.long)
    colors = torch.tensor(
        [
            0 if (index % 8 + 8 - index // 8) % 2 == 0 else 1
            for index in range(64)
        ],
        dtype=torch.long,
    )
    return files, ranks, colors


class SquareQueries(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.square_id = nn.Embedding(64, dim)
        self.file_emb = nn.Embedding(8, dim)
        self.rank_emb = nn.Embedding(8, dim)
        self.color_emb = nn.Embedding(2, dim)
        files, ranks, colors = _square_index_tables()
        self.register_buffer("file_idx", files)
        self.register_buffer("rank_idx", ranks)
        self.register_buffer("color_idx", colors)
        self.register_buffer("square_ids", torch.arange(64))

    def forward(self, batch_size: int) -> torch.Tensor:
        queries = (
            self.square_id(self.square_ids)
            + self.file_emb(self.file_idx)
            + self.rank_emb(self.rank_idx)
            + self.color_emb(self.color_idx)
        )
        return queries.unsqueeze(0).expand(batch_size, -1, -1)


class ChessQueryNet(nn.Module):
    """DINOv2 ViT + 64 semantic queries + shared 13-class linear head."""

    def __init__(
        self,
        encoder_name: str = "vit_large_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        decoder_layers: int = 4,
        nheads: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
        )
        dim = self.encoder.embed_dim
        self.queries = SquareQueries(dim)
        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=nheads,
            dim_feedforward=4 * dim,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=decoder_layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, NUM_CLASSES)
        self.encoder_frozen = False

    def set_encoder_frozen(self, frozen: bool) -> None:
        self.encoder_frozen = frozen
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not frozen
        if frozen:
            self.encoder.eval()
        else:
            self.encoder.train(self.training)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.encoder_frozen:
            self.encoder.eval()
            with torch.no_grad():
                memory = self.encoder.forward_features(images)
        else:
            memory = self.encoder.forward_features(images)
        queries = self.queries(images.shape[0])
        decoded = self.norm(self.decoder(queries, memory))
        return self.head(decoded)
