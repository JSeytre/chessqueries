"""Live-inference demo: upload photos, get predicted boards, FENs and lichess links.

Companion to `chessqueries.viz.app` (which browses a dataset split against
precomputed predictions); this one runs a checkpoint on arbitrary uploads.
Gradio also exposes the predict function over HTTP, so the running app doubles as
an API — see the README for `gradio_client` and curl examples.

Usage:
    poetry run chessqueries-demo
    poetry run python -m chessqueries.viz.demo --checkpoint <ckpt> --resolution 644
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr
import numpy as np

from chessqueries.models.predictor import (
    PAPER_RESOLUTION,
    Prediction,
    Predictor,
    resolve_checkpoint,
)
from chessqueries.viz.render import composite_pair

MAX_UPLOADS = 32  # keep one predict call (and its gallery) to a sane size
EMPTY_PROMPT = "_Upload an image to get started._"


@dataclass(frozen=True)
class DemoResult:
    """One render pass, shaped for the demo's three output components."""

    gallery: list[tuple[np.ndarray, str]]  # (input | board strip, FEN caption)
    details_md: str                        # per-image filename, FEN, lichess link
    records: list[dict[str, str]]          # the JSON/API payload


@dataclass
class DemoSession:
    """A session's predictions plus each board's orientation and rendered strip.

    Caching the strips lets the per-image orientation toggle redraw exactly one
    board instead of re-rendering (let alone re-predicting) the whole batch.
    """

    preds: list[Prediction] = field(default_factory=list)
    flips: list[bool] = field(default_factory=list)
    strips: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not len(self.preds) == len(self.flips) == len(self.strips):
            raise ValueError(
                f"ragged session: {len(self.preds)} preds, {len(self.flips)} flips, "
                f"{len(self.strips)} strips"
            )

    @classmethod
    def build(cls, preds: Sequence[Prediction], *, flipped: bool = False) -> "DemoSession":
        preds = list(preds)
        return cls(
            preds=preds,
            flips=[flipped] * len(preds),
            strips=[composite_pair(p.image_path, p.board, flipped=flipped) for p in preds],
        )

    def set_flipped(self, index: int, flipped: bool) -> None:
        """Re-render a single board at the requested orientation."""
        if not 0 <= index < len(self.preds):
            return  # nothing selected, or a stale index: leave the view alone
        pred = self.preds[index]
        self.flips[index] = flipped
        self.strips[index] = composite_pair(pred.image_path, pred.board, flipped=flipped)

    def is_flipped(self, index: int) -> bool:
        return 0 <= index < len(self.flips) and self.flips[index]

    def result(self) -> DemoResult:
        if not self.preds:
            return DemoResult(gallery=[], details_md=EMPTY_PROMPT, records=[])
        return DemoResult(
            gallery=[(strip, pred.fen) for strip, pred in zip(self.strips, self.preds)],
            details_md="\n".join(
                f"**{p.image_path.name}**\n\n`{p.fen}`\n\n[Open in lichess]({p.lichess_url})\n"
                for p in self.preds
            ),
            records=[p.to_record() for p in self.preds],
        )


def run_predictions(predictor: Predictor, files: list[str] | None) -> list[Prediction]:
    """Predict the uploaded files, rejecting an oversized batch."""
    paths = [Path(f) for f in (files or [])]
    if not paths:
        return []
    if len(paths) > MAX_UPLOADS:
        raise gr.Error(f"{len(paths)} images; this demo takes at most {MAX_UPLOADS} at a time.")
    return predictor.predict(paths)


def predict_files(
    predictor: Predictor, files: list[str] | None, *, flipped: bool = False
) -> DemoResult:
    """Run the model over uploaded file paths and shape the three outputs."""
    return DemoSession.build(run_predictions(predictor, files), flipped=flipped).result()


def build_demo(predictor: Predictor) -> gr.Blocks:
    """The Blocks app; `api_name=` is what makes a handler callable over HTTP."""
    with gr.Blocks(title="chessqueries demo") as demo:
        gr.Markdown("# chessqueries — read a chessboard off a photo")
        # Predictions and rendered strips are cached per session, so flipping a
        # board is a redraw of that one image rather than another forward pass.
        session = gr.State(DemoSession())
        # A hidden component rather than gr.State: State is not exposed to API
        # callers, and the index has to be a real parameter of `set_view`.
        selected = gr.Number(value=0, precision=0, visible=False, label="index")
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.Files(
                    label="Board photos",
                    file_types=["image"],
                    file_count="multiple",
                    type="filepath",
                )
                run = gr.Button("Read boards", variant="primary")
            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="Input and predicted board",
                    columns=1,
                    # preview mode = the enlarged view with a thumbnail rail. Without
                    # it the first render is a cropped, clipped grid tile until the
                    # user clicks, so the initial view differs from every later one.
                    preview=True,
                    # object_fit applies to the thumbnails, which are far wider
                    # than tall; the default ("cover") crops them to fill the tile.
                    object_fit="contain",
                )
                flip = gr.Checkbox(
                    label="View this board from Black's side",
                    info="Applies to the selected image; the FEN is unchanged.",
                    value=False,
                )
        details = gr.Markdown(label="Details")
        records = gr.JSON(label="Predictions (API payload)")

        def _outputs(result: DemoResult, index: int = 0):
            # Gradio wants a positional tuple; DemoResult keeps the shape named
            # everywhere except this boundary.
            #
            # selected_index puts the gallery in preview mode. Left unset it
            # renders a thumbnail grid until the first click, so the initial view
            # differs from every later one; pinning it keeps them the same. It
            # also preserves the selection when a single board is re-rendered.
            return (
                gr.Gallery(value=result.gallery,
                           selected_index=index if result.gallery else None),
                result.details_md,
                result.records,
            )

        def run_predict(images, flipped):
            # The parameter names are the API's keyword arguments; name them for callers.
            state = DemoSession.build(run_predictions(predictor, images), flipped=flipped)
            return state, 0, flipped, *_outputs(state.result())

        def set_view(state, index, flipped):
            """Re-render only the selected board."""
            index = int(index)
            state.set_flipped(index, bool(flipped))
            return state, *_outputs(state.result(), index)

        def on_select(state, event: gr.SelectData):
            """Track the selection and show that board's current orientation."""
            index = int(event.index) if event.index is not None else 0
            return index, state.is_flipped(index)

        run.click(
            run_predict,
            inputs=[files, flip],
            outputs=[session, selected, flip, gallery, details, records],
            api_name="predict",
        )
        # `.input` (not `.change`) so syncing the box on selection cannot feed
        # back in and re-toggle the newly selected image.
        flip.input(
            set_view,
            inputs=[session, selected, flip],
            outputs=[session, gallery, details, records],
            api_name="set_view",
        )
        # api_name=False: selecting is a UI gesture, not a useful HTTP endpoint.
        # (Gradio 6 renames this to api_visibility="private", absent in 5.x.)
        gallery.select(on_select, inputs=session, outputs=[selected, flip], api_name=False)
    return demo


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="chessqueries-demo",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="custom checkpoint (default: download/cache the released weights)",
    )
    p.add_argument("--resolution", type=int, default=PAPER_RESOLUTION,
                   help="must match the checkpoint's training resolution")
    p.add_argument("--device", default=None, help="default: cuda when available")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--host", default="127.0.0.1",
                   help="loopback by default; pass 0.0.0.0 to expose on the network")
    p.add_argument("--share", action="store_true", help="public Gradio tunnel")
    args = p.parse_args(argv)

    predictor = Predictor.from_checkpoint(
        resolve_checkpoint(args.checkpoint),
        resolution=args.resolution,
        device=args.device,
    )
    build_demo(predictor).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
