"""Evaluate the ChessReD ResNeXt baseline on any dataset split.

Usage:
    poetry run python scripts/eval_baseline.py \
        --checkpoint checkpoints/chessred_resnext.ckpt --split test

    # cross-dataset: the baseline on SLCC broadcast crops (out-of-distribution —
    # the board does not fill the frame, so this is a zero-shot transfer floor).
    poetry run python scripts/eval_baseline.py \
        --checkpoint checkpoints/chessred_resnext.ckpt --dataset slcc --split test

Writes predictions + metrics under outputs/ for the visualizer and logbook.
"""
import argparse
from pathlib import Path

import torch

from chessqueries.config import get_config
from chessqueries.data import DatasetIncompleteError, DatasetName, get_dataset
from chessqueries.data.chessred import category_names_in_id_order
from chessqueries.data.transforms import Normalization, build_transform
from chessqueries.metrics import aggregate
from chessqueries.metrics.report import (
    eval_tag,
    evaluation_inventory,
    print_evaluation_inventory,
    print_metrics,
    resolve_split,
    write_eval_outputs,
)
from chessqueries.models.base import predict_all
from chessqueries.models.chessred_resnext import INFERENCE_TRANSFORM, ChessReDResNeXt


def _positive_limit(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--limit must be greater than zero")
    return parsed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--dataset", type=DatasetName, choices=list(DatasetName), default=DatasetName.CHESSRED)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resize", type=int, default=None,
                   help="Resize to RxR before normalization. ChessReD's own images "
                        "are native 1024 (leave unset); cross-dataset crops vary in size, so pass "
                        "--resize 1024 (the size the weights were trained on) to feed + batch them.")
    p.add_argument("--from-lightning", action="store_true",
                   help="Checkpoint is from train_resnext.py: head predicts in canonical Piece "
                        "order, no class remap. Default: the authors' released weights "
                        "(ChessReD category order, remapped via set_class_order).")
    p.add_argument("--norm", type=Normalization, choices=list(Normalization), default=Normalization.CHESSRED,
                   help="Normalization regime to feed the model. Use 'imagenet' for a model trained "
                        "with the V2 recipe; 'chessred' (default) for the faithful recipe / authors' weights.")
    p.add_argument("--limit", type=_positive_limit, default=None, help="Debug: cap #samples.")
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Diagnostic only: evaluate available images from an incomplete dataset and "
        "mark the output non-paper-comparable.",
    )
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Directory for metrics/preds JSONs (default: outputs/chessred_baseline). "
                        "Set a distinct dir to run several checkpoints concurrently without clobber.")
    args = p.parse_args()
    if args.norm is Normalization.IMAGENET and not args.resize:
        p.error("--norm imagenet requires --resize R (the training resolution, e.g. 644)")
    if not args.checkpoint.is_file():
        p.error(f"checkpoint not found: {args.checkpoint}")

    cfg = get_config()
    dataroot = cfg.chessred_root  # category-id->Piece remap table is ChessReD's, regardless of eval set.

    # ChessReD ships the authors' 1024x1024 images, so the default transform does
    # not resize (see chessred_resnext.INFERENCE_TRANSFORM). On other datasets the
    # images keep their own resolution unless --resize squares them for batching.
    if args.resize:
        transform = build_transform(args.resize, train=False, normalization=args.norm)
    else:
        transform = INFERENCE_TRANSFORM
    ds_obj = get_dataset(args.dataset)
    split = resolve_split(ds_obj, args.split)
    try:
        ds = ds_obj.torch_dataset(
            split, transform=transform, allow_partial=args.allow_partial
        )
    except DatasetIncompleteError as exc:
        p.error(str(exc))
    if args.limit:
        ds.samples = ds.samples[: args.limit]

    if args.from_lightning:
        # Retrained head: already in canonical Piece order — no class remap.
        model = ChessReDResNeXt.from_lightning_checkpoint(args.checkpoint, map_location=args.device)
    else:
        model = ChessReDResNeXt.from_checkpoint(args.checkpoint, map_location=args.device)
        model.set_class_order(category_names_in_id_order(dataroot))
    model.to(args.device)

    predicted = predict_all(model, ds, device=args.device, batch_size=args.batch_size,
                            workers=args.workers, desc=f"eval[{args.split}]")

    metrics = aggregate(predicted.preds, predicted.gts)
    tag = eval_tag(args.dataset.value, split)
    print_metrics(metrics, f"ChessReD baseline | {tag} | n={len(predicted)}")
    inventory = evaluation_inventory(
        ds.completeness,
        expected_evaluated_samples=len(ds.samples),
        evaluated_samples=len(predicted),
    )
    print_evaluation_inventory(inventory)
    metrics["evaluation"] = inventory

    out_dir = write_eval_outputs(args.out_dir or cfg.OUTPUTS_ROOT / "chessred_baseline",
                                 tag, metrics, ds.samples, predicted.preds)
    print(f"\nWrote predictions + metrics to {out_dir}")


if __name__ == "__main__":
    main()
