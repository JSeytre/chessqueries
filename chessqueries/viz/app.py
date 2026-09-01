"""Gradio visualizer, served on a port.

Browse a dataset split alongside the rendered board. If a predictions file is
supplied (the JSON produced by scripts/eval_baseline.py), the app shows the
predicted board next to ground truth, highlights wrong squares, reports the
per-board count of wrong squares, and can filter to the worst boards.

Usage:
    poetry run python -m chessqueries.viz.app --dataset chessred --split test
    poetry run python -m chessqueries.viz.app --dataset chessred --split test \
        --predictions outputs/chessred_baseline/preds_test.json --port 7860
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gradio as gr

from chessqueries.core import Board, Split
from chessqueries.data import DatasetName, get_dataset
from chessqueries.metrics import count_wrong_squares
from chessqueries.viz.board import board_svg


def _load(dataset: DatasetName, split: str):
    ds = get_dataset(dataset)
    return ds.load_samples(
        Split(split) if ds.splits else None,
        allow_partial=True,
    )


def build_app(dataset: DatasetName, split: str, predictions_path: Path | None):
    samples = _load(dataset, split)
    preds: dict[str, list[int]] = {}
    if predictions_path:
        preds = json.loads(Path(predictions_path).read_text())

    # Order: by descending wrong-square count when we have predictions (worst first).
    order = list(range(len(samples)))
    if preds:
        def err(i: int) -> int:
            s = samples[i]
            pred = preds.get(s.sample_id)
            return count_wrong_squares(pred, s.board.labels) if pred else 0

        order.sort(key=err, reverse=True)

    def view(pos: int):
        pos = max(0, min(pos, len(order) - 1))
        s = samples[order[pos]]
        gt_svg = board_svg(s.board)
        photo = str(s.image_path)
        header = f"**{s.dataset.value} #{s.sample_id}** ({split}) — sample {pos + 1}/{len(order)}"
        link = f"[Open in lichess]({s.board.lichess_url()})"
        if preds and s.sample_id in preds:
            pred_board = Board.from_labels(preds[s.sample_id])
            wrong = pred_board.diff(s.board)
            pred_svg = board_svg(pred_board, wrong=wrong)
            info = f"{header}\n\n**{len(wrong)}** wrong squares\n\n{link}"
            return photo, gt_svg, pred_svg, info, pos
        return photo, gt_svg, gt_svg, f"{header}\n\n{link}", pos

    with gr.Blocks(title="chessqueries viz") as app:
        gr.Markdown(f"# chessqueries — {dataset.value}/{split}" + (" (worst-first)" if preds else ""))
        pos_state = gr.State(0)
        with gr.Row():
            prev_btn = gr.Button("← Prev")
            idx_box = gr.Number(value=0, label="index", precision=0)
            next_btn = gr.Button("Next →")
        info_md = gr.Markdown()
        with gr.Row():
            photo = gr.Image(label="Photo", height=360)
            gt = gr.HTML(label="Ground truth")
            pred = gr.HTML(label="Prediction")

        def go(p):
            ph, g, pr, inf, np_ = view(int(p))
            return ph, g, pr, inf, np_, np_

        outputs = [photo, gt, pred, info_md, pos_state, idx_box]
        prev_btn.click(lambda p: go(p - 1), pos_state, outputs)
        next_btn.click(lambda p: go(p + 1), pos_state, outputs)
        idx_box.submit(go, idx_box, outputs)
        app.load(lambda: go(0), None, outputs)
    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DatasetName.CHESSRED, type=DatasetName, choices=list(DatasetName))
    p.add_argument("--split", default="test", choices=[s.value for s in Split])
    p.add_argument("--predictions", type=Path, default=None)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    app = build_app(args.dataset, args.split, args.predictions)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
