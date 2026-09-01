"""Adaptation modes: how much of a trained recognizer may move when fitting a
new domain."""
from __future__ import annotations

from enum import Enum

from torch import nn

from chessqueries.models.lora import inject_lora

# LoRA goes on the encoder's attention + MLP projections.
LORA_TARGETS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")

# Everything downstream of the encoder: square queries, decoder, final norm,
# shared head. `head_type="linear"` bases lack the queries/decoder, so missing
# attributes are skipped rather than required.
READOUT_MODULES = ("queries", "decoder", "norm", "head")


class AdaptMode(str, Enum):
    """A capacity ladder over the same base checkpoint and support set."""

    ZERO_SHOT = "zero_shot"      # nothing trains: the floor
    LORA = "lora"                # low-rank adapters on the encoder only
    HEAD_FT = "head_ft"          # frozen encoder, readout trains: features vs readout
    ENCODER_FT = "encoder_ft"    # encoder trains full-rank, readout frozen: converse of head_ft
    FULL_FT = "full_ft"          # everything trains: the unconstrained ceiling


# Per-mode LRs tuned on target val under best-on-val selection (each an interior
# optimum of its own probe). Adapters start as a no-op and only add a low-rank delta,
# so they tolerate a hotter LR; the FT modes move pretrained weights directly and are
# an order of magnitude cooler. Re-probe before trusting these on a new target domain.
DEFAULT_LR = {
    AdaptMode.ZERO_SHOT: 0.0,
    AdaptMode.LORA: 1e-4,
    AdaptMode.HEAD_FT: 2e-4,
    AdaptMode.ENCODER_FT: 5e-6,
    AdaptMode.FULL_FT: 5e-6,
}


def configure_trainable(
    model: nn.Module,
    mode: AdaptMode,
    *,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> list[nn.Parameter]:
    """Freeze/unfreeze `model` in place for `mode`; return its trainable params."""
    for prm in model.parameters():
        prm.requires_grad = False

    if mode is AdaptMode.ZERO_SHOT:
        return []

    if mode is AdaptMode.LORA:
        # A base trained with staged unfreeze keeps freeze_encoder=True, which makes
        # encode() run the encoder under torch.no_grad() -- the adapters would get
        # no gradient. We adapt the encoder, so its forward must build a graph.
        model.freeze_encoder = False
        return inject_lora(model.encoder, targets=LORA_TARGETS, r=rank, alpha=alpha,
                           dropout=dropout)

    if mode is AdaptMode.HEAD_FT:
        # The encoder stays frozen *and* stays in no-grad mode, so its forward runs
        # at inference cost: only the readout moves.
        model.freeze_encoder = True
        params: list[nn.Parameter] = []
        for name in READOUT_MODULES:
            module = getattr(model, name, None)
            if module is None:
                continue
            for prm in module.parameters():
                prm.requires_grad = True
                params.append(prm)
        if not params:
            raise ValueError(f"no readout parameters found among {READOUT_MODULES}")
        return params

    if mode is AdaptMode.ENCODER_FT:
        # Same trap as LORA: a staged-unfreeze base keeps freeze_encoder=True and would
        # run the very module we are adapting under no_grad.
        model.freeze_encoder = False
        params = list(model.encoder.parameters())
        if not params:
            raise ValueError("no encoder parameters found")
        for prm in params:
            prm.requires_grad = True
        return params

    if mode is AdaptMode.FULL_FT:
        model.freeze_encoder = False
        for prm in model.parameters():
            prm.requires_grad = True
        return list(model.parameters())

    raise ValueError(f"unhandled adaptation mode: {mode}")
