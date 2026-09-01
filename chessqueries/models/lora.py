"""Minimal LoRA for `nn.Linear`, used to probe few-shot domain transfer (adapt a
trained recognizer to a new image distribution from a handful of examples without
touching the base weights)."""
from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Wraps a frozen `nn.Linear` with a low-rank residual ``B @ A`` (scaled).
    Initialised to a no-op (B=0) so the wrapped model is unchanged at step 0."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # B stays 0 -> delta starts at 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * (self.dropout(x) @ self.A.t()) @ self.B.t()


def _matches(qualified: str, target: str) -> bool:
    """True when `qualified` equals `target` or ends with it at a dot boundary
    (so ``attn.proj`` matches ``blocks.0.attn.proj`` but not ``xattn.proj``)."""
    return qualified == target or qualified.endswith(f".{target}")


def inject_lora(root: nn.Module, targets: tuple[str, ...], r: int = 8, alpha: int = 16,
                dropout: float = 0.0) -> list[nn.Parameter]:
    """Replace every `nn.Linear` whose qualified name ends in one of `targets`
    with a `LoRALinear`. Returns the new trainable LoRA parameters."""
    replaced: list[nn.Parameter] = []
    for name, module in list(root.named_modules()):
        for child_name, child in list(module.named_children()):
            qualified = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(_matches(qualified, t) for t in targets):
                lora = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                setattr(module, child_name, lora)
                replaced.extend([lora.A, lora.B])
    if not replaced:
        raise ValueError(f"inject_lora matched no linear layers for targets={targets}")
    return replaced
