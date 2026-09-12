"""Run the released ChessQueriesLite ViT-S/14 ONNX models on a photo using CPU inference.

Standalone dependencies: numpy, onnxruntime, and Pillow. Run this file directly;
the training package and PyTorch are not needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

RESOLUTION = 644
CLASS_SYMBOLS = ".PNBRQKpnbrqk"
MODEL_SHA256 = {
    "fp32": "c085f7060ce848c33389ca6e90db5b1d7e41d56fc0566f5a1e4c96cf1460f897",
    "int8": "1c7b2968263b9a51b4405f4cad8228a81283fdc458031f13f0c53fa2c147c402",
}


def preprocess(path: Path, *, exif: bool = False) -> np.ndarray:
    """Resize RGB float channels before normalization, preserving the export reference."""
    with Image.open(path) as source:
        rgb = (ImageOps.exif_transpose(source) if exif else source).convert("RGB")
        channels = [
            np.asarray(
                band.convert("F").resize(
                    (RESOLUTION, RESOLUTION), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
            for band in rgb.split()
        ]
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return np.ascontiguousarray(((np.stack(channels) / np.float32(255) - mean) / std)[None])


def placement(labels: list[int]) -> str:
    """Encode 64 class IDs in a8-to-h1 order as FEN piece placement."""
    if len(labels) != 64 or any(
        type(label) is not int or not 0 <= label < len(CLASS_SYMBOLS) for label in labels
    ):
        raise ValueError("Expected 64 valid square labels")
    ranks = []
    for start in range(0, 64, 8):
        row, empty = "", 0
        for label in labels[start:start + 8]:
            if label == 0:
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                row += CLASS_SYMBOLS[label]
        ranks.append(row + (str(empty) if empty else ""))
    return "/".join(ranks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", required=True, type=Path, help="FP32 or INT8 release ONNX file")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--exif", action="store_true", help="Apply photo EXIF orientation")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")

    with args.model.open("rb") as handle:
        actual_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual_hash not in MODEL_SHA256.values():
        raise ValueError("Model SHA-256 does not match either ViT-S/14 644 release")

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session = ort.InferenceSession(
        str(args.model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    logits = session.run(["logits"], {"image": preprocess(args.image, exif=args.exif)})[0]
    if logits.shape != (1, 64, 13) or not np.isfinite(logits).all():
        raise ValueError("Unexpected model output")
    labels = logits[0].argmax(axis=-1).tolist()
    print(json.dumps({"labels": labels, "placement": placement(labels)}, indent=2))


if __name__ == "__main__":
    main()
