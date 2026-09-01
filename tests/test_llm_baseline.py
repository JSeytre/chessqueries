"""Offline tests for the prompted-VLM baseline: placement parsing, image prep,
usage/cost accounting, and the retry loop (with a stubbed client)."""
import json

import pytest
from llm_stubs import _decoded_size, _Msg, _payload, _reader  # noqa: E402
from PIL import Image

from chessqueries.core import Board, Piece
from chessqueries.models.llm import (
    MAX_LONG_EDGE,
    MODELS,
    PROMPT_SHA256,
    PROMPT_VERSION,
    SCHEMA_SHA256,
    SCHEMA_VERSION,
    ImagePrep,
    LLMName,
    LLMPrediction,
    Usage,
    parse_placement,
)

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def test_prompt_and_schema_versions_are_explicit():
    assert (PROMPT_VERSION, PROMPT_SHA256) == (
        "v1",
        "31b04841325ed5d7d4eee72a12c5b6125c2bae182012daf583c0ef8724594a03",
    )
    assert (SCHEMA_VERSION, SCHEMA_SHA256) == (
        "v1",
        "1957bb3c4cfe37fd3f2e7feebaa571923afd6934bd9291babffb230ae6b38d1c",
    )


def test_parse_placement_round_trips():
    assert parse_placement(START).labels == Board.from_fen(START).labels


def test_parse_placement_accepts_a_full_fen():
    assert parse_placement(f"{START} w KQkq - 0 1").labels == Board.from_fen(START).labels


@pytest.mark.parametrize("bad", [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP",          # 7 ranks
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN",  # last rank spans 7
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNRR",  # last rank spans 9
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNX",  # unknown symbol
])
def test_parse_placement_rejects_malformed(bad):
    """A rank that does not span 8 squares must fail here, not silently later."""
    with pytest.raises(ValueError):
        parse_placement(bad)


def test_image_prep_spec_parsing():
    assert ImagePrep.parse("native").square is None
    assert ImagePrep.parse("644").square == 644
    for bad in ("0", "-1", "medium"):
        with pytest.raises(ValueError):
            ImagePrep.parse(bad)


def test_square_prep_matches_the_vit_transform(tmp_path):
    """`--image-size N` must send exactly the NxN square resize the ViT eval
    transform applies, so the matched-input comparison really is matched."""
    path = tmp_path / "wide.png"
    Image.new("RGB", (1200, 800)).save(path)
    image = ImagePrep(square=644).encode(path)
    assert image.media_type == "image/jpeg"
    assert _decoded_size(image.data) == (644, 644)


def test_native_prep_downscales_only_above_the_cap(tmp_path):
    big, small = tmp_path / "big.png", tmp_path / "small.png"
    Image.new("RGB", (4284, 4284)).save(big)
    Image.new("RGB", (550, 280)).save(small)
    assert _decoded_size(ImagePrep().encode(big).data) == (MAX_LONG_EDGE, MAX_LONG_EDGE)
    assert _decoded_size(ImagePrep().encode(small).data) == (550, 280)


def test_usage_adds_and_prices():
    total = Usage(1000, 200) + Usage(500, 100)
    assert total.input_tokens == 1500 and total.output_tokens == 300
    # Opus 5: $5/MTok in, $25/MTok out.
    assert total.cost_usd(MODELS[LLMName.CLAUDE_OPUS_5].pricing) == pytest.approx(0.015)


# --------------------------------------------------------------------------- #
# Retry loop, against a stub client (stubs shared with the batch/provider tests)
# --------------------------------------------------------------------------- #
def test_read_returns_a_board(tmp_path):
    path = tmp_path / "b.png"
    Image.new("RGB", (64, 64)).save(path)
    pred = _reader([_Msg(_payload(START))]).read(path, "s0")
    assert pred.ok and pred.attempts == 1
    assert pred.board.labels == Board.from_fen(START).labels
    assert pred.usage == Usage(1000, 100, 0)


def test_read_retries_a_malformed_placement_and_bills_both_attempts(tmp_path):
    path = tmp_path / "b.png"
    Image.new("RGB", (64, 64)).save(path)
    pred = _reader([_Msg(_payload("8/8/8")), _Msg(_payload(START))]).read(path, "s0")
    assert pred.ok and pred.attempts == 2
    assert pred.usage == Usage(2000, 200, 0)  # the failed attempt is paid for too


def test_read_gives_up_after_max_attempts(tmp_path):
    path = tmp_path / "b.png"
    Image.new("RGB", (64, 64)).save(path)
    pred = _reader([_Msg(None, "refusal")] * 3, max_attempts=3).read(path, "s0")
    assert not pred.ok and pred.board is None
    assert pred.attempts == 3 and "refusal" in pred.error
    assert pred.usage == Usage(3000, 300, 0)  # every refused turn is still billed


def test_truncation_is_not_retried(tmp_path):
    """A max_tokens cut-off recurs on retry, so retrying only re-spends the whole
    output budget. One attempt, one bill."""
    path = tmp_path / "b.png"
    Image.new("RGB", (64, 64)).save(path)
    reader = _reader([_Msg(None, "max_tokens")] * 3, max_attempts=3)
    pred = reader.read(path, "s0")
    assert not pred.ok and pred.attempts == 1
    assert reader.client.calls == 1
    assert pred.usage == Usage(1000, 100, 0)
    assert "max_tokens" in pred.error


def test_unreadable_image_fails_only_that_sample(tmp_path):
    path = tmp_path / "not-an-image.png"
    path.write_text("garbage")
    reader = _reader([])
    pred = reader.read(path, "s0")
    assert not pred.ok and reader.client.calls == 0
    assert pred.usage == Usage()


def test_prediction_survives_a_cache_round_trip(tmp_path):
    path = tmp_path / "b.png"
    Image.new("RGB", (64, 64)).save(path)
    pred = _reader([_Msg(_payload(START))]).read(path, "s0")
    back = LLMPrediction.from_dict(json.loads(json.dumps(pred.as_dict())))
    assert back.sample_id == "s0" and back.usage == pred.usage
    assert back.board.labels == pred.board.labels


def test_empty_board_is_not_mistaken_for_a_failure(tmp_path):
    """An all-empty prediction round-trips as a real board, not as `None` --
    `labels` is a 64-long list of zeros, which is falsy-adjacent but valid."""
    path = tmp_path / "b.png"
    Image.new("RGB", (64, 64)).save(path)
    pred = _reader([_Msg(_payload("8/8/8/8/8/8/8/8"))]).read(path, "s0")
    back = LLMPrediction.from_dict(json.loads(json.dumps(pred.as_dict())))
    assert back.ok and back.board.labels == [int(Piece.EMPTY)] * 64


def test_never_run_is_distinguished_from_a_model_failure(tmp_path):
    """A request the model never answered (batch canceled/expired, unreadable
    image) must not be scored: counting it as wrong would invent a result. A turn
    the model *did* produce but we could not parse is a real failure and is."""
    path = tmp_path / "b.png"
    Image.new("RGB", (64, 64)).save(path)

    # Model produced turns we could not use -> a genuine failure, still scored.
    refused = _reader([_Msg(None, "refusal")] * 3, max_attempts=3).read(path, "s0")
    assert not refused.ok and refused.attempted

    # Never sent (unreadable file) -> excluded from scoring.
    bad = tmp_path / "not-an-image.png"
    bad.write_text("garbage")
    never = _reader([]).read(bad, "s1")
    assert not never.ok and not never.attempted

    # The flag survives the cache round-trip, so a resumed run keeps the split.
    for pred in (refused, never):
        back = LLMPrediction.from_dict(json.loads(json.dumps(pred.as_dict())))
        assert back.attempted == pred.attempted


def test_legacy_cache_records_default_to_attempted():
    """Cache lines written before `attempted` existed are real model turns."""
    back = LLMPrediction.from_dict({"sample_id": "s0", "labels": None, "placement": None,
                                    "usage": {}, "attempts": 3, "error": "boom"})
    assert back.attempted
