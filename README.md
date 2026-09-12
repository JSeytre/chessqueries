# ChessQueries

Chess board recognition: provide a photo of a real-life chessboard and receive
its chess position.

📄 [Paper](https://arxiv.org/abs/2608.30762) ·
📊 [Dataset](https://huggingface.co/datasets/joelseytre/slcc) ·
🤖 [Model weights](https://huggingface.co/joelseytre/chessqueries)

![ChessQueries predictions across ChessReD, ChessCog, SLCC, and CVChess](assets/readme/hero.jpg)

## Model

ChessQueries consists of a 304M-parameter ViT-L/14 encoder that maps a 644 × 644
input image to patch tokens. Sixty-four learned queries—one for each board
square—cross-attend to those tokens through a DETR-style decoder. A single
shared linear head maps each decoded query to its per-square class.

Best exact-board accuracy reported in the paper for a model trained on the
ChessReD, ChessCog, and SLCC training splits:

| ChessReD | ChessCog | SLCC | CVChess (zero-shot) |
|---:|---:|---:|---:|
| **99.5%** | **98.5%** | **87.1%** | **87.6%** |

## Quick start

ChessQueries requires Python 3.12 and [Poetry](https://python-poetry.org/).
Install the visualizer and launch the local Gradio demo:

```bash
poetry install --with viz
poetry run chessqueries-demo
```

You can also predict directly from one or more images:

```bash
poetry run chessqueries-predict photo.jpg --viz prediction.png
```

On first use, both commands download and verify the 1.49 GB
[safetensors checkpoint](https://huggingface.co/joelseytre/chessqueries), then
cache it under `checkpoints/release/`.

## ChessQueriesLite: ViT-S/14 ONNX

The smaller ViT-S/14 model accepts 644 × 644 images and is available as two
self-contained [ONNX files](https://huggingface.co/joelseytre/chessqueries#chessquerieslite-vit-s14-onnx):
FP32 (127.57 MB) and dynamic INT8 (36.03 MB). Both contain the trained encoder
and decoder. The accuracy table above describes the ViT-L model.

For CPU inference with Python 3.12, run these commands from this repository:

```bash
python3.12 -m venv .venv-onnx
.venv-onnx/bin/python -m pip install numpy==1.26.4 onnxruntime==1.23.2 Pillow==10.4.0
curl --fail --location \
  https://huggingface.co/joelseytre/chessqueries/resolve/main/chessquerieslite-vits-644-int8.onnx \
  --output chessquerieslite-vits-644-int8.onnx
.venv-onnx/bin/python chessqueries/models/predict_onnx.py photo.jpg \
  --model chessquerieslite-vits-644-int8.onnx
```

For FP32, replace `int8` with `fp32` in the download URL and both file paths.
Only the selected model needs to be downloaded. This standalone runner needs
NumPy, ONNX Runtime, and Pillow; it verifies the model's SHA-256 before loading.
It prints JSON containing 64 class IDs and FEN piece placement. Side to move,
castling rights, en passant, and move counters cannot be inferred from a photo.

The runner converts to RGB, resizes each floating-point channel to 644 × 644
using antialiased bilinear interpolation, then applies ImageNet normalization.
It ignores EXIF orientation to match the export reference; add `--exif` to apply
the photo's orientation. The full input/output contract and model hashes are in
the [model card](https://huggingface.co/joelseytre/chessqueries#chessquerieslite-vit-s14-onnx).

## Citation

```bibtex
@misc{seytre2026chessqueries,
  title         = {ChessQueries: Toward Better Chess Board Recognition},
  author        = {Seytre, Jo{\"e}l},
  year          = {2026},
  eprint        = {2608.30762},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2608.30762},
  url           = {https://arxiv.org/abs/2608.30762}
}
```

## Licenses

- **Code** — [PolyForm Noncommercial 1.0.0](LICENSE), including the required
  notice: Copyright (c) 2026 the chessqueries authors.
- **Model weights** — [PolyForm Noncommercial 1.0.0](https://huggingface.co/joelseytre/chessqueries#license).
- **SLCC annotations** — [CC BY-NC 4.0](DATA_LICENSE). The source broadcasts
  and reconstructed frames are not distributed and remain the property of
  their respective rights holders.

Commercial use is not licensed.

## Going further

See [DEVELOPMENT.md](DEVELOPMENT.md) for repository layout, data preparation,
training and evaluation commands, experiment utilities, and checks.

See the [minimal reproduction guide](minimal_reproduction/README.md) for a
self-contained implementation of the model, training, evaluation, and leakage
checks used in the paper.
