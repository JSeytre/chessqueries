"""The demo's predict path: uploads become gallery strips, markdown details and an
API payload, without launching a server."""
import pytest
import torch
from PIL import Image

from chessqueries.core import Board
from chessqueries.models.base import BoardRecognizer
from chessqueries.models.predictor import Predictor

pytest.importorskip("gradio")
pytest.importorskip("cairosvg")

from chessqueries.viz import demo as demo_module  # noqa: E402
from chessqueries.viz.demo import (  # noqa: E402  (after importorskip)
    MAX_UPLOADS,
    DemoSession,
    build_demo,
    predict_files,
    run_predictions,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


class StubRecognizer(BoardRecognizer):
    def predict_labels(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([Board.from_fen(START_FEN).labels] * x.shape[0])


@pytest.fixture
def predictor():
    return Predictor(StubRecognizer(), resolution=32)


@pytest.fixture
def uploads(tmp_path):
    paths = []
    for i in range(2):
        path = tmp_path / f"upload{i}.png"
        Image.new("RGB", (60, 40)).save(path)
        paths.append(str(path))
    return paths


def test_predict_files_shapes_all_three_outputs(predictor, uploads):
    result = predict_files(predictor, uploads)

    assert len(result.gallery) == len(result.records) == 2
    strip, caption = result.gallery[0]
    assert strip.ndim == 3 and strip.shape[2] == 3       # an RGB input|board strip
    assert caption.startswith(START_FEN)                 # caption is the FEN
    assert "upload0.png" in result.details_md
    assert "lichess.org/editor" in result.details_md


def test_records_are_the_api_payload(predictor, uploads):
    records = predict_files(predictor, uploads).records

    assert [sorted(r) for r in records] == [["fen", "image", "lichess_url"]] * 2
    assert records[0]["fen"] == f"{START_FEN} w - - 0 1"


def test_no_upload_is_a_prompt_not_a_crash(predictor):
    for empty in (None, []):
        result = predict_files(predictor, empty)
        assert result.gallery == [] and result.records == []
        assert "Upload" in result.details_md


def test_too_many_uploads_is_rejected(predictor, uploads):
    import gradio as gr

    with pytest.raises(gr.Error):
        predict_files(predictor, uploads * (MAX_UPLOADS // 2 + 1))


def test_flip_changes_the_picture_but_not_the_position(predictor, uploads):
    """The model reads absolute coordinates, so a flipped view keeps the same FEN."""
    plain = predict_files(predictor, uploads)
    flipped = predict_files(predictor, uploads, flipped=True)

    assert flipped.records == plain.records          # same FENs, same lichess links
    assert flipped.details_md == plain.details_md
    assert (flipped.gallery[0][0] != plain.gallery[0][0]).any()   # different image
    assert flipped.gallery[0][1] == plain.gallery[0][1]           # same caption


def test_session_flip_is_per_image(predictor, uploads):
    """Toggling one board must not change how the others are drawn."""
    state = DemoSession.build(run_predictions(predictor, uploads))
    before = [strip.copy() for strip in state.strips]

    state.set_flipped(1, True)

    assert state.flips == [False, True]
    assert (state.strips[0] == before[0]).all()      # untouched image is byte-identical
    assert (state.strips[1] != before[1]).any()      # only the selected one redrew


def test_session_flip_keeps_the_position(predictor, uploads):
    """A per-image flip changes the picture, never the FEN or the payload."""
    state = DemoSession.build(run_predictions(predictor, uploads))
    plain = state.result()

    state.set_flipped(0, True)
    flipped = state.result()

    assert flipped.records == plain.records
    assert flipped.details_md == plain.details_md
    assert flipped.gallery[0][1] == plain.gallery[0][1]   # caption is still the FEN


def test_session_reports_and_round_trips_orientation(predictor, uploads):
    state = DemoSession.build(run_predictions(predictor, uploads))
    assert not state.is_flipped(0)

    state.set_flipped(0, True)
    assert state.is_flipped(0)

    state.set_flipped(0, False)
    assert not state.is_flipped(0)
    assert state.flips == [False, False]


def test_session_ignores_out_of_range_selection(predictor, uploads):
    """No selection yet (or a stale index) must not raise or alter the view."""
    state = DemoSession.build(run_predictions(predictor, uploads))
    before = [strip.copy() for strip in state.strips]

    state.set_flipped(-1, True)
    state.set_flipped(99, True)

    assert state.flips == [False, False]
    assert all((a == b).all() for a, b in zip(state.strips, before))
    assert not DemoSession().is_flipped(0)


def test_session_rejects_ragged_state():
    with pytest.raises(ValueError, match="ragged session"):
        DemoSession(preds=[], flips=[True], strips=[])


def test_empty_session_prompts():
    result = DemoSession().result()
    assert result.gallery == [] and result.records == []
    assert "Upload" in result.details_md


def test_build_demo_exposes_the_predict_api(predictor):
    """The app builds without launching, and the predict fn is API-callable."""
    import gradio as gr

    demo = build_demo(predictor)
    assert isinstance(demo, gr.Blocks)

    api_names = [
        dep.get("api_name") for dep in demo.get_config_file()["dependencies"]
    ]
    assert "predict" in api_names
    assert "set_view" in api_names


def test_demo_uses_release_checkpoint_by_default(predictor, monkeypatch):
    checkpoint = object()
    launch_args = {}

    monkeypatch.setattr(demo_module, "resolve_checkpoint", lambda value: checkpoint)
    monkeypatch.setattr(
        Predictor,
        "from_checkpoint",
        classmethod(
            lambda cls, value, *, resolution, device: (
                predictor if value is checkpoint else pytest.fail("wrong checkpoint")
            )
        ),
    )

    class Demo:
        def launch(self, **kwargs):
            launch_args.update(kwargs)

    monkeypatch.setattr(demo_module, "build_demo", lambda value: Demo())

    demo_module.main([])

    assert launch_args == {
        "server_name": "127.0.0.1",
        "server_port": 7861,
        "share": False,
    }


def test_gallery_fits_the_wide_strips(predictor):
    """The strips are wider than tall; "cover" would crop them into a broken zoom."""
    demo = build_demo(predictor)
    galleries = [
        block for block in demo.blocks.values() if type(block).__name__ == "Gallery"
    ]
    assert [g.object_fit for g in galleries] == ["contain"]
    # preview mode: without it the first render is a clipped grid tile and only
    # looks right after the user clicks into it.
    assert [g.preview for g in galleries] == [True]


def test_no_blurb_above_the_controls(predictor):
    """Only the title; the explanatory paragraphs were noise."""
    demo = build_demo(predictor)
    texts = [
        block.value for block in demo.blocks.values()
        if type(block).__name__ == "Markdown" and block.value
    ]
    assert texts == ["# chessqueries — read a chessboard off a photo"]
