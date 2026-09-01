"""Single-image inference latency benchmark for the ChessQueries model.

Times `predict_labels` on one image and reports per-image latency. With
--checkpoint it benchmarks the trained model end to end; without one it builds
a randomly initialised model of the same shape (identical timing, no weights to
download or copy), so it runs on a bare checkout on any machine. The input is
the bundled example photo (assets/example_input.jpg) unless --image overrides
it; random pixels if the example is missing too. Every run also writes the
model's prediction on that input (FEN + lichess link + run metadata) to
outputs/bench_inference/last_run.json — meaningless when the weights are
random, and flagged as such in the record. When the read is real (trained
weights + an actual photo), an input-vs-predicted-board figure is saved
alongside as last_run.png and linked from the record.

Usage:
    poetry run python scripts/bench_inference.py                    # paper model, CPU/GPU auto
    poetry run python scripts/bench_inference.py --device mps       # Apple-silicon Metal
    poetry run python scripts/bench_inference.py --checkpoint <ckpt> --image photo.jpg
    # The paper's latency (19 ms/board on an RTX 4090) is measured in bf16:
    poetry run python scripts/bench_inference.py --checkpoint <ckpt> --precision bf16
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from statistics import median

import torch

from chessqueries.config import get_config
from chessqueries.core import Board
from chessqueries.data.transforms import build_transform
from chessqueries.models.base import BoardRecognizer
from chessqueries.models.predictor import PAPER_RESOLUTION, Predictor
from chessqueries.models.chessqueries_model import ChessQueriesModel

PAPER_ENCODER = "vit_large_patch14_dinov2.lvd142m"
EXAMPLE_IMAGE = Path(__file__).resolve().parent.parent / "assets" / "example_input.jpg"
RECORD_PATH = get_config().OUTPUTS_ROOT / "bench_inference" / "last_run.json"
VIZ_PATH = RECORD_PATH.with_suffix(".png")


class Precision(str, Enum):
    """Numeric regime for the timed forward pass."""

    FP32 = "fp32"  # plain eager, the default
    BF16 = "bf16"  # autocast; the paper's reported latency (19 ms on an RTX 4090)

    def autocast(self, device: str):
        if self is Precision.FP32:
            return nullcontext()
        return torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16)


@dataclass(frozen=True)
class BenchResult:
    """Per-image `predict_labels` latency over the timed runs, in seconds."""

    median_s: float
    min_s: float
    max_s: float
    runs: int

    def __str__(self) -> str:
        return (
            f"median {self.median_s * 1000:.0f} ms/image "
            f"(min {self.min_s * 1000:.0f}, max {self.max_s * 1000:.0f}, n={self.runs})"
        )


def resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(checkpoint: Path | None, encoder: str, device: str) -> BoardRecognizer:
    if checkpoint is not None:
        return Predictor.from_checkpoint(checkpoint, device=device).model
    return ChessQueriesModel(encoder_name=encoder, pretrained=False).eval().to(device)


def load_input(image: Path | None, resolution: int, device: str) -> torch.Tensor:
    if image is None:
        return torch.randn(1, 3, resolution, resolution, device=device)
    from torchvision.io import ImageReadMode, read_image

    pixels = read_image(str(image), mode=ImageReadMode.RGB).float()
    return build_transform(resolution, train=False)(pixels).unsqueeze(0).to(device)


def _sync(device: str) -> None:
    # Async backends buffer kernel launches; without a sync the timer stops early.
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


@torch.no_grad()
def bench(
    model: BoardRecognizer,
    x: torch.Tensor,
    *,
    warmup: int,
    runs: int,
    precision: Precision = Precision.FP32,
) -> BenchResult:
    device = str(x.device)
    with precision.autocast(device):
        for _ in range(warmup):
            model.predict_labels(x)
        times = []
        for _ in range(runs):
            _sync(device)
            start = time.perf_counter()
            model.predict_labels(x)
            _sync(device)
            times.append(time.perf_counter() - start)
    return BenchResult(median_s=median(times), min_s=min(times), max_s=max(times), runs=runs)


def write_viz(board: Board, image: Path) -> Path | None:
    """Input-vs-predicted-board figure (the `--viz` layout); needs the viz extras."""
    try:
        from chessqueries.viz.render import BoardPanel, side_by_side_figure
    except ImportError as exc:
        print(f"viz skipped ({exc}); install the extras: poetry install --with viz")
        return None
    fig = side_by_side_figure([BoardPanel(image_path=image, board=board, title=image.name)])
    VIZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(VIZ_PATH, dpi=200, bbox_inches="tight")
    return VIZ_PATH


def write_record(
    board: Board, image: Path | None, viz: Path | None, weights: str,
    random_weights: bool, device: str, resolution: int, precision: Precision,
    result: BenchResult,
) -> None:
    """Persist the run's prediction + timing (the JSON payload, local-only)."""
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image": str(image) if image else None,
        "fen": board.to_fen(),
        "lichess_url": board.lichess_url(),
        "viz": str(viz) if viz else None,
        "weights": weights,
        "random_weights": random_weights,
        "device": device,
        "resolution": resolution,
        "precision": precision.value,
        "latency_ms": {
            "median": round(result.median_s * 1000, 1),
            "min": round(result.min_s * 1000, 1),
            "max": round(result.max_s * 1000, 1),
            "runs": result.runs,
        },
    }
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORD_PATH.write_text(json.dumps(record, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="bench_inference",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="default: random weights of the paper architecture (same latency)")
    p.add_argument("--image", type=Path, default=None,
                   help="default: the bundled example photo, or random pixels without it")
    p.add_argument("--encoder", default=PAPER_ENCODER,
                   help="timm encoder for the no-checkpoint model")
    p.add_argument("--resolution", type=int, default=PAPER_RESOLUTION,
                   help="must match the checkpoint's training resolution")
    p.add_argument("--device", default=None, help="cpu / cuda / mps; default: cuda when available")
    p.add_argument("--precision", type=Precision, choices=list(Precision), default=Precision.FP32,
                   help="'bf16' (autocast) is what the paper's 19 ms/board on an RTX 4090 "
                        "is measured in; 'fp32' (default) is plain eager execution.")
    p.add_argument("--threads", type=int, default=None,
                   help="torch CPU threads; default: torch's own choice (~physical cores)")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=5)
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    model = build_model(args.checkpoint, args.encoder, device)
    image = args.image or (EXAMPLE_IMAGE if EXAMPLE_IMAGE.exists() else None)
    x = load_input(image, args.resolution, device)

    params = sum(p.numel() for p in model.parameters())
    weights = str(args.checkpoint) if args.checkpoint else f"random ({args.encoder})"
    print(f"model: {params / 1e6:.0f}M params | weights: {weights}")
    print(f"device: {device} | resolution: {args.resolution} | precision: {args.precision.value} | "
          f"input: {image or 'random pixels'} | threads: {torch.get_num_threads()}")
    result = bench(model, x, warmup=args.warmup, runs=args.runs, precision=args.precision)
    print(result)

    with args.precision.autocast(device):
        board = Board.from_tensor(model.predict_labels(x)[0].cpu())
    # A figure only when the read is real (trained weights) and there is a photo
    # to show next to it; a stale one from an earlier run must not outlive its record.
    viz = None
    if args.checkpoint is not None and image is not None:
        viz = write_viz(board, image)
    if viz is None:
        VIZ_PATH.unlink(missing_ok=True)
    write_record(board, image, viz, weights, args.checkpoint is None,
                 device, args.resolution, args.precision, result)
    tag = " (random weights — not a real read)" if args.checkpoint is None else ""
    print(f"fen: {board.to_fen()}{tag}")
    print(f"-> {RECORD_PATH}" + (f"\n-> {viz}" if viz else ""))


if __name__ == "__main__":
    main()
