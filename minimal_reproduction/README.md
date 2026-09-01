# Minimal independent reproduction

This folder independently reimplements the data loading, model, training, and
evaluation from the [ChessQueries paper](https://arxiv.org/abs/2608.30762)
without importing `chessqueries`. Run commands from the repository root. Set
`CHESSQUERIES_DATA_ROOT=/path/to/data` to use another data directory.

## 1. Base data

```bash
poetry install --with slcc
poetry run python scripts/download_data.py chessred chesscog
```

The downloader verifies and reuses complete datasets already on disk.

## 2. Check the released checkpoint

```bash
poetry run python minimal_reproduction/test_eval.py
poetry run python scripts/download_data.py --release-checkpoint
poetry run python minimal_reproduction/eval.py \
  --checkpoint checkpoints/release/chessqueries-vitL14-644-joint.safetensors \
  --datasets chessred
```

Expected: **2121/2129 = 99.62% exact-board accuracy** (all 64 squares correct).

## 3. Reconstruct SLCC

Check the 20 pinned YouTube sources before downloading the videos:

```bash
poetry run python -m chessqueries.annotate.reconstruct \
  --bundle data/slcc/releases/slcc-v1 \
  --check-sources
```

If YouTube requests sign-in, add `--cookies-from-browser firefox` (using your
browser) or `--cookies FILE`. Use the same option below. Never commit cookies.

```bash
poetry run python -m chessqueries.annotate.reconstruct \
  --bundle data/slcc/releases/slcc-v1 \
  --out data/slcc/dataset \
  --video-dir data/slcc/videos
```

Continue only after the source check reports `"complete": true` and
reconstruction reports `"committed": true`. Do not use `--allow-partial` for
training.

## 4. Reproduce training

The fixed recipe is DINOv2 ViT-L/14 at 644 px, batch size 6, ChessReD +
ChessCog + SLCC, and 45 epochs (about 9 hours and 17 GB VRAM on an RTX 4090).

```bash
poetry run python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.is_bf16_supported(), "bf16 unsupported"; print(torch.cuda.get_device_name(0))'
CUDA_VISIBLE_DEVICES=0 poetry run python minimal_reproduction/train.py
```

## 5. Evaluate the reproduced checkpoint

```bash
poetry run python minimal_reproduction/eval.py \
  --checkpoint minimal_reproduction/runs/reproduction-seed1/best.pt \
  --datasets chessred chesscog slcc
```

Reference reproduction: **99.15 / 98.54 / 84.72%**. CUDA training is
nondeterministic, so exact results may differ.

## 6. Evaluate on CVChess (optional)

CVChess is zero-shot and is not used for training. Seed its labels, then follow
the printed instructions to place the manually downloaded images:

```bash
poetry run python scripts/download_data.py cvchess
poetry run python minimal_reproduction/eval.py \
  --checkpoint minimal_reproduction/runs/reproduction-seed1/best.pt \
  --datasets cvchess
```

| Exact-board accuracy | ChessReD | ChessCog | SLCC | CVChess zero-shot |
|---|---:|---:|---:|---:|
| Released checkpoint | 99.62% | 98.54% | 87.67% | 87.22% |
| Reference reproduction | 99.15% | 98.54% | 84.72% | 82.10% |

With all four datasets present, the optional leakage audit is:

```bash
poetry run python minimal_reproduction/leakage_audit.py --hash-images
```

Raw metrics are in `reference_results.json`; the learning curve is in
`reference_training_log.csv`.
