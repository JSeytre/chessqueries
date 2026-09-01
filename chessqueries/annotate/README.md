# `chessqueries.annotate` — broadcast dataset workflow

Turn open St. Louis Chess Club / Grand Chess Tour YouTube broadcasts into chess
board-recognition labels. We release **only annotations** (video id + frame index +
crop bbox + FEN) and reconstruct frames Kinetics-style — no pixels redistributed.
Auto-FEN comes from the tournament's **relay PGN**, never the on-screen analysis
board.

## One command to rule them all

```bash
poetry install --with annotate,viz     # producer OCR deps + gradio for the UIs
chessqueries-annotate                    # == `status`: what's pending, and what to run next
```

`status` surveys the manifest against what's on disk and tells you the next move:

```
SLCC broadcast dataset — 14 video(s) · 3 tournament(s)

  ⬇  14 video(s) to ingest               → annotate ingest
       13 not downloaded; the rest downloaded but not segmented
  🏷  5 new template(s) to label          → annotate label
       pooled across 4 video(s)
  ⚙  9 video(s) ready to auto-label       → annotate produce
  ✅  120 candidate(s) to review           → annotate review
  🧩  ~40 board-only shot(s) await the model → annotate model-pass  (deferred)
  📦  6 video(s) reviewed                  → annotate reconstruct --video ID1 ID2
```

Run the command each line names; re-run `status` to see progress. (`chessqueries-annotate`
and `python -m chessqueries.annotate` are the same entry point.)

## The phases

| phase | command | what it does | human? |
|---|---|---|---|
| add | `annotate add <url|id>…` | register videos (yt-dlp title/date); set `--tournament`/`--round-id` | quick |
| **ingest** | `annotate ingest [--video ID]` | download (pinned fmt) + segment + fingerprint; caches shots & descriptors | no |
| **label** | `annotate label` | grade the **pooled** new templates once across all videos (Gradio) | yes |
| **produce** | `annotate produce [--video ID]` | clock-OCR align ready videos → `<id>.candidates.json` | no |
| **crosscheck** | `annotate crosscheck --adapter\|--checkpoint …` | recognizer vs. clock consensus → triage accept/review/quarantine → `<id>.crosschecked.json` | no |
| **review** | `annotate review [--video ID]` | crop-vs-relay verify → `<id>.reviewed.json` | yes |
| internal export | `annotate reconstruct --video ID` or `--rebuild-all` | transactionally assemble the maintainer's reviewed working files | no |
| **model-pass** | *(deferred)* | fill board-only gaps via a v1 recognizer + relay retrieval | later |

Commands default to "do everything pending"; `--video ID` scopes to one. Internal
reconstruction is the exception: it requires either one or more `--video` IDs, which
preserves all other exported videos and their splits, or an explicit `--rebuild-all`.

`crosscheck` takes the recognizer as either a LoRA `--adapter` (from `scripts/lora_fewshot.py`)
or a full `--checkpoint` (a `LitChessQueriesModel`, e.g. the joint V2 model — pass its eval
`--resolution`, ViT-L V2 = 644, since the ckpt doesn't store it).

## Adding videos

```bash
chessqueries-annotate add https://www.youtube.com/watch?v=<id> --tournament <lichessTourId>
```

`add` records title/date via yt-dlp and writes a manifest entry. The one manual bit is
the **relay mapping**: which Lichess broadcast a video covers. Map it coarsely to the
*tournament* (one id covers many videos — the clock+name+chronology matcher picks the
round per frame), and optionally narrow with `--round-id` (e.g. a single classical
round). List a tournament's rounds to copy ids from:

```bash
python -c "from chessqueries.annotate.relay import tournament_rounds as r; print([(x.id,x.name) for x in r('<tourId>')])"
```

Mappings live in `resources/slcc_videos.json` (committed). The large labeled **layout
registry** (`resources/slcc_layouts.json`) is local-only/gitignored — rebuilt by `label`.

## GPU

The pipeline is CPU-only: segmentation is CPU-bound, and the `produce` step's OCR runs on
CPU (the `annotate` group installs the CPU `paddlepaddle` build, which is fast enough — see
`ocr_bench`). Nothing here competes with training for a GPU. To run OCR on a spare card,
install `paddlepaddle-gpu` instead.

The optional EasyOCR comparison is intentionally run in a separate environment:
EasyOCR requires its own headless OpenCV wheel, which must not be installed beside
the production `opencv-contrib-python`/Paddle stack.

## Files per video (gitignored `data/slcc/`)

| file | meaning |
|---|---|
| `<id>.<fmt>.mp4` | downloaded video (discardable after reconstruct) |
| `<id>.<fmt>.shots.json` · `.descriptors.npy` | segmentation caches (the slow step; reused by produce + status + model-pass) |
| `<id>.candidates.json` | auto-produced labels, **WIP** (`stage = candidates`) |
| `<id>.reviewed.json` | human-confirmed, **released** (`stage = reviewed`) |

Each `Annotation` carries a `source` (`clock_ocr` / `model_retrieval` / `manual`), so the
deferred model pass can fill the board-only gaps (label = relay FEN, never the model's
output) through the same review step, distinguishable in any eval split.

## Public SLCC v1 reconstruction

Consumers should use the pinned
[`joelseytre/slcc`](https://huggingface.co/datasets/joelseytre/slcc) metadata release
rather than the maintainer workflow above:

```bash
poetry install --with slcc
# Install Deno (recommended) or another yt-dlp-supported JavaScript runtime.
poetry run python -m chessqueries.annotate.reconstruct --check-sources \
  --report outputs/slcc-source-check.json
poetry run python -m chessqueries.annotate.reconstruct \
  --out data/slcc/dataset --video-dir data/slcc/videos
```

This fetches and validates all 20 annotation files from Hugging Face, processes them
through the shared staged reconstruction core, retains the frozen split map, and refuses
an incomplete commit unless `--allow-partial` is explicit. `--bundle PATH` remains
available for offline metadata validation, but reconstruction itself cannot be offline:
the release contains no broadcast pixels. If YouTube rejects an unauthenticated request,
pass `--cookies FILE` or `--cookies-from-browser BROWSER`; some environments may also
need a yt-dlp PO-token provider plugin. `--js-runtime RUNTIME[:PATH]` overrides automatic
runtime detection. The source preflight writes one structured availability/error record
per video without downloading the multi-GB streams.

Source broadcasts and pinned formats are controlled by YouTube and may become unavailable;
the metadata release and transactional code cannot guarantee access. Failed strict runs do
not replace an existing dataset. See the dataset card for source-availability and rights
limitations. The internal `chessqueries-annotate reconstruct` command uses the same staging,
validation, and atomic-commit core. Scoped exports preserve other videos; whole-dataset
replacement requires `--rebuild-all`.

Migration note: manifests produced before the transactional exporter do not contain
crop fingerprints. The first `--rebuild-all` therefore needs every source video in the
local ingest cache and re-extracts every crop; take a backup of the frozen dataset before
that one-time migration. Later full runs reuse validated crops, and scoped runs only
touch the selected videos.

## Notes

- **Yield is low by design** — broadcast coverage is sparse and fragmented; expect tens
  of confident full-board frames per video. Scale comes from many videos.
- **Resumable.** Re-running a video reuses its caches and the already-downloaded file, so
  a crashed run just re-runs cheaply. `status` is always safe to run.
- **Descriptor scope.** DINOv2 3×3 grid descriptors and their locally fitted PCA support
  annotation-time shot/template discovery only. They are not in SLCC v1 and are not
  required for public crop reconstruction.
