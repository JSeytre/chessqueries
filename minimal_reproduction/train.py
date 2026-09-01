"""Train the paper's reported recipe from scratch in a plain PyTorch loop."""
from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from data import (
    BoardDataset,
    eval_transform,
    load_chesscog,
    load_chessred,
    load_slcc,
    train_transform,
)
from eval import score_boards
from model import ChessQueryNet


# Fixed paper recipe and defaults confirmed against the released checkpoint.
ENCODER = "vit_large_patch14_dinov2.lvd142m"
RESOLUTION = 644
BATCH_SIZE = 6
WORKERS = 8
EPOCHS = 45
LR = 1.4e-4
ENCODER_LR = 1.4e-5
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 3
UNFREEZE_EPOCH = 5
LR_FLOOR = 0.001
GRAD_CLIP_NORM = 1.0

TRAIN_PARTS = (
    ("chessred/train", lambda: load_chessred("train"), 6_479),
    ("chesscog/train", lambda: load_chesscog("train"), 4_400),
    ("slcc/train", lambda: load_slcc("train"), 1_475),
)
VAL_PARTS = (
    ("chessred/val", lambda: load_chessred("val"), 2_192),
    ("chesscog/val", lambda: load_chesscog("val"), 146),
    ("slcc/val", lambda: load_slcc("val"), 326),
)


def lr_factor(epoch: int, start_epoch: int) -> float:
    """Staged linear warmup followed by cosine decay to ``LR_FLOOR``."""
    if epoch < start_epoch:
        return 0.0
    local_epoch = epoch - start_epoch
    if local_epoch < WARMUP_EPOCHS:
        return (local_epoch + 1) / WARMUP_EPOCHS
    denominator = max(1, EPOCHS - start_epoch - WARMUP_EPOCHS)
    progress = min(1.0, (local_epoch - WARMUP_EPOCHS) / denominator)
    return LR_FLOOR + (1.0 - LR_FLOOR) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def complete_samples(name: str, load, expected: int):
    samples = load()
    if len(samples) != expected:
        raise ValueError(f"{name}: found {len(samples)} labels, expected {expected}")
    missing = [sample.image_path for sample in samples if not sample.image_path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{name}: {len(missing)} images are missing (first: {missing[0]})"
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Run directory (default: minimal_reproduction/runs/reproduction-seed<seed>).",
    )
    args = parser.parse_args()

    if not args.device.startswith("cuda"):
        parser.error("the paper recipe uses CUDA bf16; --device must name a CUDA device")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; from-scratch reproduction requires a CUDA GPU")

    torch.set_float32_matmul_precision("high")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_dir = args.out or Path(__file__).parent / "runs" / f"reproduction-seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_parts = [
            BoardDataset(complete_samples(name, load, expected), train_transform(RESOLUTION))
            for name, load, expected in TRAIN_PARTS
        ]
        val_parts = [
            BoardDataset(complete_samples(name, load, expected), eval_transform(RESOLUTION))
            for name, load, expected in VAL_PARTS
        ]
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    train_data = ConcatDataset(train_parts)
    val_data = ConcatDataset(val_parts)
    assert len(train_data) == 12_354 and len(val_data) == 2_664
    print(f"train n={len(train_data)}  val n={len(val_data)}")

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = ChessQueryNet(ENCODER).to(args.device)
    model.set_encoder_frozen(True)

    encoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("encoder.")
    ]
    new_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": ENCODER_LR},
            {"params": new_parameters, "lr": LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    group_starts = [UNFREEZE_EPOCH, 0]

    log_path = out_dir / "log.csv"
    with log_path.open("w", newline="") as file:
        csv.writer(file).writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_board_acc",
                "val_per_square_acc",
                "lr_encoder",
                "lr_new",
                "seconds",
            ]
        )

    best_accuracy, best_epoch = -1.0, -1
    for epoch in range(EPOCHS):
        start_time = time.time()
        if epoch >= UNFREEZE_EPOCH and model.encoder_frozen:
            model.set_encoder_frozen(False)
            print(f"[staged-unfreeze] encoder unfrozen at epoch {epoch}")
        for group, start_epoch in zip(optimizer.param_groups, group_starts):
            peak = ENCODER_LR if start_epoch == UNFREEZE_EPOCH else LR
            group["lr"] = peak * lr_factor(epoch, start_epoch)

        model.train()
        running_loss, steps = 0.0, 0
        for images, labels in train_loader:
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, 13), labels.reshape(-1)
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            running_loss += loss.item()
            steps += 1

        model.eval()
        predictions: list[list[int]] = []
        ground_truth: list[list[int]] = []
        validation_loss, validation_steps = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for images, labels in val_loader:
                images = images.to(args.device, non_blocking=True)
                device_labels = labels.to(args.device, non_blocking=True)
                logits = model(images)
                validation_loss += nn.functional.cross_entropy(
                    logits.reshape(-1, 13), device_labels.reshape(-1)
                ).item()
                validation_steps += 1
                predictions.extend(logits.argmax(-1).cpu().tolist())
                ground_truth.extend(labels.tolist())
        validation = score_boards(predictions, ground_truth)
        accuracy = validation["board_accuracy"]
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        row = [
            epoch,
            running_loss / steps,
            validation_loss / validation_steps,
            accuracy,
            validation["per_square_accuracy"],
            learning_rates[0],
            learning_rates[1],
            round(time.time() - start_time),
        ]
        with log_path.open("a", newline="") as file:
            csv.writer(file).writerow(row)
        print(
            f"epoch {epoch:02d}  train_loss {running_loss / steps:.4f}  "
            f"val_board_acc {accuracy:.4f}  ({row[-1]}s)"
        )

        state = {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "val_board_acc": accuracy,
            "seed": args.seed,
            "encoder": ENCODER,
            "resolution": RESOLUTION,
        }
        torch.save(state, out_dir / "last.pt")
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            torch.save(state, out_dir / "best.pt")

    print(
        f"best val_board_acc {best_accuracy:.4f} at epoch {best_epoch} "
        f"-> {out_dir / 'best.pt'}"
    )


if __name__ == "__main__":
    main()
