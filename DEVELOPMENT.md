# Development

This is the operational guide for working with the ChessQueries codebase. For
the compact, independent paper reproduction, use
[`minimal_reproduction/`](minimal_reproduction/README.md).

Run all commands from the repository root.

## Environment

ChessQueries uses Python 3.12 and Poetry with an in-project `.venv`.

```bash
# Core model, training, and evaluation
poetry install

# Everything used by the public codebase
poetry install --with viz,dev,slcc,annotate,llm
```

Optional dependency groups can also be installed individually:

| Group | Purpose |
|---|---|
| `viz` | Gradio apps, rendering, and visualizations |
| `slcc` | Downloading and reconstructing the public SLCC release |
| `annotate` | Producing SLCC annotations, including the OCR stack |
| `llm` | Frontier multimodal-model baselines |
| `notebooks` | Jupyter and exploratory analysis |
| `dev` | Tests, linting, and pre-commit checks |

## Inference and visualization

```bash
# Local Gradio demo; the terminal prints the URL
poetry run chessqueries-demo

# One or more images to FEN, with an optional rendered comparison
poetry run chessqueries-predict photo1.jpg photo2.jpg --viz side_by_side.png

# Browse precomputed predictions, worst boards first
poetry run python -m chessqueries.viz.app \
  --dataset chessred --split test \
  --predictions outputs/chessred_baseline/preds_test.json
```

Both inference entrypoints accept `--checkpoint PATH` and `--resolution N` for
weights trained at a different input resolution. Run either command with
`--help` for network binding and device options.

## Data and released weights

Download the automatically available datasets, the ChessReD baseline, and the
released ChessQueries checkpoint:

```bash
poetry run python scripts/download_data.py all \
  --checkpoint --release-checkpoint
```

| Dataset | Splits used here | Acquisition |
|---|---|---|
| ChessReD | train/validation/test; 2,129 test images | automatic, about 2.5 GB |
| ChessCog | train/validation/test; 342 test images | automatic |
| SLCC | 1,475/326/373 train/validation/test frames | annotations automatic; frames reconstructed locally |
| CVChess | evaluation only; 352 images from one game | labels automatic; images manual |

`--checkpoint` fetches the published ChessReD ResNeXt baseline.
`--release-checkpoint` fetches the ChessQueries safetensors model, verifies SHA-256
`6151bdd98fbe25f32080c097eba7ae75808b5615160e4867240931085fc762e5`, and
places it in `checkpoints/release/`.

The CVChess downloader seeds `annotations.json` and prints instructions for
placing its manually downloaded images under `data/cvchess/images/`.

### Reconstruct SLCC

SLCC distributes annotations and reconstruction metadata, not broadcast pixels.
Install the `slcc` group and [Deno](https://deno.com/) or another JavaScript
runtime supported by yt-dlp, then check that the 20 pinned sources are available:

```bash
poetry install --with slcc
poetry run python -m chessqueries.annotate.reconstruct \
  --bundle data/slcc/releases/slcc-v1 \
  --check-sources
```

If the check is complete, reconstruct the crops:

```bash
poetry run python -m chessqueries.annotate.reconstruct \
  --bundle data/slcc/releases/slcc-v1 \
  --out data/slcc/dataset \
  --video-dir data/slcc/videos
```

YouTube may require `--cookies FILE`, `--cookies-from-browser BROWSER`, or a
yt-dlp PO-token provider. Never commit cookies. Third-party source availability
cannot be guaranteed.

## Training and evaluation

The paper configuration uses DINOv2 ViT-L/14 at 644 pixels, batch size 6, and
ChessReD + ChessCog + SLCC:

```bash
poetry run python scripts/train_chessqueries.py \
  --name experiment --seed 1 \
  --encoder vit_large_patch14_dinov2.lvd142m \
  --resolution 644 --batch-size 6 \
  --datasets chessred,chesscog,slcc
```

Evaluate the best checkpoint from that run across all acquired domains:

```bash
bash scripts/eval_all_domains.sh checkpoints/experiment experiment 644
```

Evaluation writes `outputs/chessqueries_<name>/metrics_<domain>.json`. It checks
expected dataset sizes, available images, non-empty splits, and complete SLCC
reconstruction before running. `--allow-partial` is only for debugging incomplete
local data; its results are marked partial and are not comparable to the paper.

Useful experiment entrypoints:

| Task | Command or script |
|---|---|
| Train ChessQueries | `scripts/train_chessqueries.py` |
| Evaluate one domain | `scripts/eval_cross_dataset.py` |
| Train the ChessReD ResNeXt baseline | `scripts/train_resnext.py` |
| Evaluate the ResNeXt baseline | `scripts/eval_baseline.py` |
| Query vs. linear head | `scripts/train_chessqueries.py --head-type query\|linear` |
| Resolution, encoder, or augmentation ablations | `scripts/train_chessqueries.py --help` |
| Few-shot and LoRA adaptation | `scripts/lora_fewshot.py` |
| Frontier multimodal-model baseline | `scripts/eval_llm_baseline.py` |
| Inference latency | `scripts/bench_inference.py` |

The multimodal-model baseline requires provider credentials from `.env`; see
`.env.example`. It can incur API charges, so inspect `--help` before running it.

## Checks

```bash
poetry install --with dev
poetry run ruff check .
poetry run pytest
poetry run pre-commit run --all-files
```

CI runs Ruff, focused SLCC reconstruction tests, and the full test suite on
Python 3.12.

## Repository layout

```text
chessqueries/
  core/               chess types, FEN ordering, and square vocabulary
  data/               dataset loaders, transforms, and downloaders
  models/             ChessQueries, comparison models, and prediction API
  train/              training and multi-dataset input pipeline
  metrics/            board accuracy, tolerance curve, and edit distance
  annotate/           SLCC annotation and reconstruction pipeline
  viz/                Gradio apps, board rendering, and attention maps
scripts/               download, training, evaluation, and experiment entrypoints
minimal_reproduction/ independent model, loaders, training, evaluation, and audits
tests/                 unit and integration tests
```
