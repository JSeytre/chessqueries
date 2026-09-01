"""Independent fp32 evaluation for the paper's four test datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from data import (
    BoardDataset,
    Sample,
    eval_transform,
    load_chesscog,
    load_chessred,
    load_cvchess,
    load_slcc,
)
from model import ChessQueryNet


def score_boards(predictions: list[list[int]], ground_truth: list[list[int]]) -> dict:
    """Compute micro square accuracy and strict 64-of-64 board accuracy."""
    assert len(predictions) == len(ground_truth) and predictions
    squares_correct = 0
    boards_exact = 0
    for predicted, expected in zip(predictions, ground_truth):
        assert len(predicted) == 64 and len(expected) == 64
        correct = sum(left == right for left, right in zip(predicted, expected))
        squares_correct += correct
        boards_exact += correct == 64
    n_boards = len(predictions)
    n_squares = 64 * n_boards
    return {
        "n_boards": n_boards,
        "n_squares": n_squares,
        "n_squares_correct": squares_correct,
        "per_square_accuracy": squares_correct / n_squares,
        "n_boards_exact": boards_exact,
        "board_accuracy": boards_exact / n_boards,
    }


def load_model(checkpoint: Path, device: str) -> ChessQueryNet:
    """Load either a minimal-reproduction checkpoint or the Lightning release."""
    if checkpoint.suffix == ".safetensors":
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        state = load_file(str(checkpoint), device="cpu")
        encoder = metadata.get("encoder_name")
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["state_dict"]
        if any(key.startswith("model.") for key in state):
            state = {
                key.removeprefix("model."): value
                for key, value in state.items()
                if key.startswith("model.")
            }
        encoder = payload.get("encoder") or payload.get("hyper_parameters", {}).get(
            "encoder_name"
        )
    if not encoder:
        raise ValueError("checkpoint does not record its encoder name")
    model = ChessQueryNet(encoder, pretrained=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


TestLoader = Callable[[], list[Sample]]
TEST_SETS: dict[str, tuple[str, TestLoader, int]] = {
    "chessred": ("chessred_test", lambda: load_chessred("test"), 2_129),
    "chesscog": ("chesscog_test", lambda: load_chesscog("test"), 342),
    "slcc": ("slcc_test", lambda: load_slcc("test"), 373),
    "cvchess": ("cvchess_all", load_cvchess, 352),
}


def complete_samples(dataset: str, load: TestLoader, expected: int) -> list[Sample]:
    samples = load()
    if len(samples) != expected:
        raise ValueError(f"{dataset}: found {len(samples)} labels, expected {expected}")
    missing = [sample.image_path for sample in samples if not sample.image_path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f"{dataset}: {len(missing)} of {expected} images are missing (first: {preview})"
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=TEST_SETS,
        default=list(TEST_SETS),
        help="Datasets to evaluate (default: all four).",
    )
    parser.add_argument("--resolution", type=int, default=644)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Metrics directory (default: <checkpoint-directory>/eval).",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    # Validate every requested inventory before allocating the 1.5 GB model.
    inventories = {}
    for dataset in args.datasets:
        output_name, load, expected = TEST_SETS[dataset]
        inventories[dataset] = (
            output_name,
            complete_samples(dataset, load, expected),
        )

    torch.set_float32_matmul_precision("high")
    model = load_model(args.checkpoint, args.device)
    out_dir = args.out or args.checkpoint.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        output_name, samples = inventories[dataset]
        data = BoardDataset(samples, eval_transform(args.resolution))
        loader = DataLoader(
            data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=args.device.startswith("cuda"),
        )
        predictions: list[list[int]] = []
        ground_truth: list[list[int]] = []
        with torch.no_grad():
            for images, labels in loader:
                logits = model(images.to(args.device, non_blocking=True))
                predictions.extend(logits.argmax(-1).cpu().tolist())
                ground_truth.extend(labels.tolist())
        metrics = score_boards(predictions, ground_truth)
        (out_dir / f"metrics_{output_name}.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )
        print(
            f"{output_name:14s} boards "
            f"{metrics['n_boards_exact']:4d}/{metrics['n_boards']:4d} "
            f"= {metrics['board_accuracy']:.4%}   squares "
            f"{metrics['n_squares_correct']}/{metrics['n_squares']} "
            f"= {metrics['per_square_accuracy']:.4%}"
        )
    print(f"wrote metrics to {out_dir}")


if __name__ == "__main__":
    main()
