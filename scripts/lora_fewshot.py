"""Few-shot domain adaptation of a trained recognizer: LoRA vs head-, encoder- and
full-FT.

Adapts a base checkpoint to a target domain from k images and reports three things
per run: the zero-shot baseline, the adapted target-domain accuracy, and a
ChessReD/ChessCog retention probe -- so the gain and the forgetting it costs are
legible side by side. Every mode gets the same step ceiling and the same
best-on-val checkpoint selection, so the ladder compares each mode at its own
best point rather than at whichever step happens to end the run.

Usage:
    poetry run python scripts/lora_fewshot.py \
        --dataset slcc --checkpoint checkpoints/v3-data-noSLCC-s1/last.ckpt \
        --mode lora --k 10 --data-seed 0 --resolution 644 --batch-size 4
"""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from chessqueries.config import get_config
from chessqueries.core import Split
from chessqueries.data import DatasetIncompleteError, DatasetName, get_dataset
from chessqueries.data.fewshot import FULL_SPLIT, adaptation_targets, load_support
from chessqueries.metrics.report import evaluation_inventory, print_evaluation_inventory
from chessqueries.models.adapt import (DEFAULT_LR, LORA_TARGETS, AdaptMode,
                                      configure_trainable)
from chessqueries.train.fewshot import adapt, evaluate
from chessqueries.train.lit import LitChessQueriesModel

# In-domain sets used only to measure forgetting; never adaptation targets.
RETENTION_DATASETS = (DatasetName.CHESSRED, DatasetName.CHESSCOG)


@dataclass(frozen=True)
class EvalSweep:
    """One model state scored everywhere it matters: the target domain's val and test
    splits, plus per-dataset metrics on the in-domain sets kept to measure forgetting."""

    val: dict
    test: dict
    retention: dict[str, dict]


def fmt(m: dict) -> str:
    return (f"board_acc={m['board_accuracy']:.4f}  per_square={m['per_square_accuracy']:.4f}  "
            f"wrong/board={m['mean_wrong_squares']:.3f}  (n={m['n_boards']})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    targets = [d for d in adaptation_targets() if d not in RETENTION_DATASETS]
    p.add_argument("--dataset", required=True, type=DatasetName, choices=targets,
                   metavar="{" + ",".join(d.value for d in targets) + "}",
                   help="target domain to adapt to")
    p.add_argument("--mode", required=True, type=AdaptMode, choices=list(AdaptMode),
                   metavar="{" + ",".join(m.value for m in AdaptMode) + "}")
    p.add_argument("--k", type=int, default=FULL_SPLIT,
                   help=f"support-set size ({FULL_SPLIT} = full train split)")
    p.add_argument("--data-seed", type=int, default=0, help="seeds the support-set draw")
    p.add_argument("--name", default=None, help="output dir under CHECKPOINTS_ROOT")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=None,
                   help=f"per-mode default: { {m.value: lr for m, lr in DEFAULT_LR.items()} }")
    p.add_argument("--eval-every", type=int, default=100,
                   help="val-probe cadence in steps for best-checkpoint selection "
                        "(0 = off, report the final step instead)")
    p.add_argument("--patience", type=int, default=5,
                   help="stop after this many consecutive val probes without "
                        "improvement (0 = never stop early)")
    p.add_argument("--val-probe-limit", type=int, default=160,
                   help="val boards used for in-training selection; the final "
                        "reported val metrics always use the full split")
    p.add_argument("--retention-limit", type=int, default=400,
                   help="Eval the adapted model on this many CR/CC test boards to measure "
                        "in-domain forgetting (0 disables).")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--resolution", type=int, default=644)
    p.add_argument("--fp32", action="store_true",
                   help="disable bf16 autocast (slower; bf16 matches the training recipe)")
    p.add_argument("--save-weights", action="store_true",
                   help="always persist trainable weights; by default only LoRA adapters "
                        "are saved (head/full-FT deltas are GBs per run)")
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Diagnostic only: use available images from incomplete datasets and mark "
        "the report non-paper-comparable.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if not args.checkpoint.is_file():
        p.error(f"checkpoint not found: {args.checkpoint}")

    bf16 = not args.fp32
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.mode]
    name = args.name or f"fewshot-{args.dataset.value}-{args.mode.value}-k{args.k}-s{args.data_seed}"
    started = time.monotonic()

    try:
        support = load_support(
            args.dataset, args.k, args.data_seed, allow_partial=args.allow_partial
        )
    except DatasetIncompleteError as exc:
        p.error(str(exc))
    print(f"{args.dataset.value}: {len(support.train)} support / {len(support.val)} val / "
          f"{len(support.test)} test frame(s)")

    # Retention probe: a fixed subsample of CR/CC test, so we can see how much
    # adaptation dents in-domain accuracy.
    retention = {}
    retention_completeness = {}
    if args.retention_limit:
        for ret_name in RETENTION_DATASETS:
            try:
                loaded = get_dataset(ret_name).load_with_report(
                    Split.TEST, allow_partial=args.allow_partial
                )
            except DatasetIncompleteError as exc:
                p.error(str(exc))
            retention[ret_name.value] = loaded.samples[: args.retention_limit]
            retention_completeness[ret_name.value] = loaded.completeness

    data_inventory = {
        "target": {
            Split.TRAIN.value: evaluation_inventory(
                support.completeness[Split.TRAIN]
            ),
            Split.VAL.value: evaluation_inventory(
                support.completeness[Split.VAL], evaluated_samples=len(support.val)
            ),
            Split.TEST.value: evaluation_inventory(
                support.completeness[Split.TEST], evaluated_samples=len(support.test)
            ),
        },
        "retention": {
            name: evaluation_inventory(
                retention_completeness[name],
                expected_evaluated_samples=len(samples),
                evaluated_samples=len(samples),
            )
            for name, samples in retention.items()
        },
    }
    data_inventory["target"][Split.TRAIN.value]["selected_support_samples"] = len(
        support.train
    )
    for split_name, inventory in data_inventory["target"].items():
        print(f"target {split_name}: ", end="")
        print_evaluation_inventory(inventory)
    for retention_name, inventory in data_inventory["retention"].items():
        print(f"retention {retention_name}: ", end="")
        print_evaluation_inventory(inventory)

    lit = LitChessQueriesModel.load_from_checkpoint(args.checkpoint, map_location=args.device)
    model = lit.model.to(args.device).eval()

    def eval_all(tag: str) -> EvalSweep:
        val = evaluate(model, support.val, args.resolution, args.device, bf16,
                       args.eval_batch_size)
        test = evaluate(model, support.test, args.resolution, args.device, bf16,
                        args.eval_batch_size)
        print(f"[{tag}] val : {fmt(val)}")
        print(f"[{tag}] test: {fmt(test)}")
        ret = {k: evaluate(model, s, args.resolution, args.device, bf16, args.eval_batch_size)
               for k, s in retention.items()}
        for k, m in ret.items():
            print(f"[{tag}] {k:8s}: {fmt(m)}")
        return EvalSweep(val=val, test=test, retention=ret)

    # 1) Zero-shot baseline (base model, no adaptation) -- always reported first.
    print()
    zero_shot = eval_all("zero-shot")

    # 2) Freeze/unfreeze for the requested mode and adapt on the support set.
    params = configure_trainable(model, args.mode, rank=args.rank, alpha=args.alpha,
                                 dropout=args.dropout)
    model.to(args.device)
    n_trainable = sum(prm.numel() for prm in params)
    print(f"\nmode={args.mode.value}: {n_trainable:,} trainable params, lr={lr:g}"
          + (f" ({len(params)//2} LoRA adapters, rank={args.rank}, alpha={args.alpha})"
             if args.mode is AdaptMode.LORA else ""))

    val_probe = support.val[: args.val_probe_limit]
    if args.mode is not AdaptMode.ZERO_SHOT and not val_probe:
        print("WARNING: --val-probe-limit 0 -> selecting the final step, not the best one")

    train_started = time.monotonic()
    fit = {"final_train_loss": float("nan"), "steps_run": 0, "best_step": 0,
           "selected_val_probe": {}, "stopped_early": False, "selection_enabled": False}
    if args.mode is not AdaptMode.ZERO_SHOT:
        fit = adapt(model, support.train, val_probe, args.resolution, args.device,
                    args.steps, lr, args.batch_size, bf16, params,
                    args.eval_every, args.patience, args.eval_batch_size)
    train_seconds = time.monotonic() - train_started
    final_loss = fit["final_train_loss"]

    # 3) Re-evaluate the adapted model on the held-out target split (+ retention delta).
    adapted = EvalSweep(val={}, test=zero_shot.test, retention=zero_shot.retention)
    if args.mode is not AdaptMode.ZERO_SHOT:
        print(f"\n(final train loss {final_loss:.4f}, {train_seconds/60:.1f} min; "
              f"selected step {fit['best_step']}/{fit['steps_run']}"
              f"{', stopped early' if fit['stopped_early'] else ''})")
        adapted = eval_all(args.mode.value)
        for k, m in adapted.retention.items():
            print(f"[{args.mode.value}] {k} retention: "
                  f"{zero_shot.retention[k]['board_accuracy']:.4f} "
                  f"-> {m['board_accuracy']:.4f}")

    out_dir = get_config().CHECKPOINTS_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode is AdaptMode.LORA or args.save_weights:
        weights = {n: prm.detach().cpu() for n, prm in model.named_parameters()
                   if prm.requires_grad}
        torch.save(
            {"adapter": weights, "mode": args.mode.value, "targets": LORA_TARGETS,
             "rank": args.rank, "alpha": args.alpha, "dropout": args.dropout,
             "base_checkpoint": str(args.checkpoint),
             "encoder_name": lit.hparams.encoder_name, "resolution": args.resolution},
            out_dir / "adapter.pt",
        )
    report = {
        "name": name, "checkpoint": str(args.checkpoint), "dataset": args.dataset.value,
        "mode": args.mode.value, "k": args.k, "data_seed": args.data_seed,
        "support_ids": support.train_ids,
        "steps": args.steps, "lr": lr, "rank": args.rank, "alpha": args.alpha,
        "batch_size": args.batch_size, "bf16": bf16, "resolution": args.resolution,
        "n_trainable_params": n_trainable,
        "n_train": len(support.train), "n_val": len(support.val), "n_test": len(support.test),
        "final_train_loss": final_loss,
        "eval_every": args.eval_every, "patience": args.patience,
        "val_probe_limit": args.val_probe_limit, "n_val_probe": len(val_probe),
        "selection_enabled": fit["selection_enabled"], "best_step": fit["best_step"],
        "steps_run": fit["steps_run"], "stopped_early": fit["stopped_early"],
        "selected_val_probe": fit["selected_val_probe"],
        "train_seconds": round(train_seconds, 1),
        "total_seconds": round(time.monotonic() - started, 1),
        "zero_shot_val": zero_shot.val, "zero_shot_test": zero_shot.test,
        "adapted_val": adapted.val, "adapted_test": adapted.test,
        "retention_limit": args.retention_limit,
        "zero_shot_retention": zero_shot.retention,
        "adapted_retention": adapted.retention,
        "evaluation": data_inventory,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved report -> {out_dir}")


if __name__ == "__main__":
    main()
