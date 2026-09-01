"""Few-shot adaptation loop: evaluation, best-on-val selection, early stopping.

Lives in the package (not the CLI script) so the selection and eval-mode contracts
are unit-testable without a GPU or a checkpoint.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.data import DataLoader

from chessqueries.data.base import BoardImageDataset, BoardSample
from chessqueries.data.transforms import build_transform
from chessqueries.metrics import aggregate


def autocast_ctx(device: str, bf16: bool):
    """bf16 autocast on CUDA, matching the main training recipe."""
    if not bf16 or not str(device).startswith("cuda"):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


@torch.no_grad()
def evaluate(model, samples: list[BoardSample], res: int, device: str, bf16: bool,
             batch_size: int = 16) -> dict:
    """Metrics for `samples`. Forces eval mode: measuring a model that is still in
    train mode silently reports it with dropout and drop-path active."""
    model.eval()
    ds = BoardImageDataset(samples, build_transform(res, train=False))
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4)
    preds, gts = [], []
    for imgs, labels in loader:
        with autocast_ctx(device, bf16):
            preds.extend(model.predict_labels(imgs.to(device)).cpu().tolist())
        gts.extend(labels.tolist())
    return aggregate(preds, gts)


def _snapshot(params_by_name: dict) -> dict:
    """Copy just the trainable tensors — the frozen ones cannot have moved."""
    return {n: p.detach().cpu().clone() for n, p in params_by_name.items()}


def adapt(model, samples, val_probe, res, device, steps, lr, batch_size, bf16, params,
          eval_every, patience, eval_batch_size, verbose=True) -> dict:
    """Train up to `steps`, returning the *best-on-val* weights rather than the last.

    A k-image support set is memorised long before the step budget runs out, after
    which the model keeps drifting away from every other domain. Selecting on val
    (and stopping once it stalls) reports each mode at its own best point, so the
    comparison is not decided by who overfits fastest.
    """
    ds = BoardImageDataset(samples, build_transform(res, train=True))
    # A k-image support set makes an epoch a handful of batches, so `steps` spans
    # hundreds of epochs. Without persistent workers the loader re-forks its pool
    # every epoch -- hundreds of fork/teardown cycles alongside a live CUDA
    # context, which deadlocks in worker startup.
    n_workers = min(4, len(samples))
    loader = DataLoader(ds, batch_size=min(batch_size, len(samples)), shuffle=True,
                        num_workers=n_workers, drop_last=False,
                        persistent_workers=n_workers > 0)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}

    # Select on per-square, not board accuracy: board accuracy is all-64-correct, so
    # at low k it sits at ~0 for every candidate and a couple of lucky base-model
    # boards would outrank a model whose per-square is 14 points better. Board
    # accuracy stays the *reported* headline; it is just too coarse to select on.
    def val_score():
        m = evaluate(model, val_probe, res, device, bf16, eval_batch_size)
        model.train()
        return (m["per_square_accuracy"], m["board_accuracy"]), m

    select = bool(val_probe) and eval_every > 0
    best = {"score": None, "step": 0, "metrics": {}, "state": None}
    if select:
        best["score"], best["metrics"] = val_score()   # step 0 = the base model
        best["state"] = _snapshot(named)

    model.train()
    step, last, stale, stopped_early = 0, float("nan"), 0, False
    while step < steps and not stopped_early:
        for imgs, labels in loader:
            with autocast_ctx(device, bf16):
                logits = model(imgs.to(device))
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.to(device).reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            last, step = float(loss.detach()), step + 1

            if select and step % eval_every == 0:
                score, metrics = val_score()
                if score > best["score"]:
                    best.update(score=score, step=step, metrics=metrics,
                                state=_snapshot(named))
                    stale = 0
                else:
                    stale += 1
                    if patience and stale >= patience:
                        stopped_early = True
                if verbose:
                    print(f"  step {step:5d}/{steps}  loss {last:.4f}  "
                          f"val board {metrics['board_accuracy']:.4f} "
                          f"per-sq {metrics['per_square_accuracy']:.4f}"
                          f"{'  <- best' if stale == 0 else ''}", flush=True)
            if step >= steps or stopped_early:
                break

    if select and best["state"] is not None:
        with torch.no_grad():
            for n, p in named.items():
                p.copy_(best["state"][n].to(p.device))
    model.eval()  # hand back a model that is ready to be measured
    return {"final_train_loss": last, "steps_run": step, "best_step": best["step"],
            "selected_val_probe": best["metrics"], "stopped_early": stopped_early,
            "selection_enabled": select}
