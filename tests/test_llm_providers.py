"""LLM baseline, provider dispatch + the OpenAI reader (request shape, retry
semantics, reasoning-token accounting) — against stubbed clients."""
import json

import pytest
from llm_stubs import START, _decoded_size, _openai_reader, _openai_response  # noqa: E402
from PIL import Image

from chessqueries.core import Board
from chessqueries.models.llm import (
    MODELS,
    SYSTEM_PROMPT,
    USER_PROMPT,
    ClaudeBoardReader,
    Effort,
    ImagePrep,
    LLMName,
    LLMPrediction,
    OpenAIBoardReader,
    Provider,
    Usage,
    get_reader,
)


# --------------------------------------------------------------------------- #
# Provider dispatch
# --------------------------------------------------------------------------- #
def test_get_reader_dispatches_on_the_model_provider():
    assert isinstance(get_reader(LLMName.CLAUDE_OPUS_5, client=object()), ClaudeBoardReader)
    assert isinstance(get_reader(LLMName.GPT_5_6_TERRA, client=object()), OpenAIBoardReader)
    assert MODELS[LLMName.GPT_5_6_SOL].provider is Provider.OPENAI


def test_a_reader_refuses_another_providers_model():
    """Mismatched pairings must fail at construction, not on the first billed call."""
    with pytest.raises(ValueError, match="served by"):
        ClaudeBoardReader(model=LLMName.GPT_5_6_TERRA, client=object())
    with pytest.raises(ValueError, match="served by"):
        OpenAIBoardReader(model=LLMName.CLAUDE_OPUS_5, client=object())


def test_unsupported_effort_is_rejected_at_construction():
    """`none` exists for GPT-5.6 but not for Claude; spending a batch to find that
    out is exactly the kind of mistake this check is here to prevent."""
    with pytest.raises(ValueError, match="does not support effort=none"):
        ClaudeBoardReader(effort=Effort.NONE, client=object())
    OpenAIBoardReader(effort=Effort.NONE, client=object())  # accepted


def test_api_key_resolution(monkeypatch):
    """`.env` wins; unset *and* blank both fall back to the SDK's own env lookup.

    Blank matters: `.env.example` ships `ANTHROPIC_API_KEY=`, which parses as an
    empty secret rather than a missing one, and passing "" to a client shadows a
    perfectly good shell export with a credential that cannot work.
    """
    from pydantic import SecretStr

    import chessqueries.config as config_mod
    from chessqueries.models.llm import api_key

    settings = config_mod.Settings(_env_file=None, OPENAI_API_KEY=SecretStr("sk-from-dotenv"),
                                   ANTHROPIC_API_KEY=SecretStr(""))
    monkeypatch.setattr(config_mod, "get_config", lambda: settings)

    assert api_key("OPENAI_API_KEY") == "sk-from-dotenv"
    assert api_key("ANTHROPIC_API_KEY") is None      # blank -> fall back
    assert api_key("NONEXISTENT_KEY") is None        # absent -> fall back
    # The secret must not leak into a repr we might paste into a log or an issue.
    assert "sk-from-dotenv" not in repr(settings)


def test_openai_pricing_matches_the_published_rates():
    # Terra: $2/MTok in, $12/MTok out.
    assert Usage(1_000_000, 1_000_000).cost_usd(
        MODELS[LLMName.GPT_5_6_TERRA].pricing) == pytest.approx(14.0)


# --------------------------------------------------------------------------- #
# OpenAI request shape
# --------------------------------------------------------------------------- #
def test_openai_request_params_match_the_documented_shape(tmp_path):
    path = tmp_path / "b.png"
    Image.new("RGB", (1200, 800)).save(path)
    reader = _openai_reader(prep=ImagePrep(square=644), effort=Effort.LOW, max_tokens=32000)
    params = reader.request_params(path)

    assert params["model"] == "gpt-5.6-terra"
    assert params["max_output_tokens"] == 32000
    assert params["reasoning"] == {"effort": "low"}
    fmt = params["text"]["format"]
    # Strict mode is refused without a name and additionalProperties:false.
    assert fmt["type"] == "json_schema" and fmt["strict"] is True and fmt["name"]
    assert fmt["schema"]["additionalProperties"] is False

    content = params["input"][0]["content"]
    image, text = content[0], content[1]
    assert image["type"] == "input_image" and text["type"] == "input_text"
    assert image["image_url"].startswith("data:image/jpeg;base64,")
    # The square resize must survive the data-URL wrapper, or the matched-input
    # comparison against our ViT silently stops being matched.
    assert _decoded_size(image["image_url"].split(",", 1)[1]) == (644, 644)


def test_both_providers_ask_the_identical_question(tmp_path):
    """The experiment is the prompt + schema; only the wire format may differ.
    If these drift, the two rows in the paper are no longer comparable."""
    path = tmp_path / "b.png"
    Image.new("RGB", (300, 300)).save(path)
    prep = ImagePrep(square=644)
    claude = ClaudeBoardReader(client=object(), prep=prep).request_params(path)
    openai_ = _openai_reader(prep=prep).request_params(path)

    assert claude["system"] == openai_["instructions"] == SYSTEM_PROMPT
    assert (claude["messages"][0]["content"][1]["text"]
            == openai_["input"][0]["content"][1]["text"] == USER_PROMPT)
    assert (claude["output_config"]["format"]["schema"]
            == openai_["text"]["format"]["schema"])
    # Same pixels, too.
    assert (claude["messages"][0]["content"][0]["source"]["data"]
            == openai_["input"][0]["content"][0]["image_url"].split(",", 1)[1])


# --------------------------------------------------------------------------- #
# OpenAI interpretation
# --------------------------------------------------------------------------- #
def test_openai_read_returns_a_board_and_records_reasoning_tokens(tmp_path):
    path = tmp_path / "b.png"
    Image.new("RGB", (64, 64)).save(path)
    pred = _openai_reader([_openai_response()]).read(path, "s0")
    assert pred.ok and pred.board.labels == Board.from_fen(START).labels
    assert pred.usage == Usage(900, 500, 0, 420)
    # Thinking is billed as output, so it must not be charged a second time.
    assert pred.usage.cost_usd(MODELS[LLMName.GPT_5_6_TERRA].pricing) == pytest.approx(
        (900 * 2.0 + 500 * 12.0) / 1e6)


def test_openai_truncation_is_not_retried(tmp_path):
    """Mirrors the Claude rule: an exhausted output budget will be exhausted
    again, so one attempt, one bill."""
    path = tmp_path / "b.png"
    Image.new("RGB", (64, 64)).save(path)
    reader = _openai_reader([_openai_response(status="incomplete",
                                              reason="max_output_tokens")] * 3,
                            max_attempts=3)
    pred = reader.read(path, "s0")
    assert not pred.ok and pred.attempts == 1 and reader.client.calls == 1
    assert "max_output_tokens" in pred.error


def test_openai_content_filter_is_retried(tmp_path):
    """An incomplete response for any *other* reason may succeed on retry."""
    path = tmp_path / "b.png"
    Image.new("RGB", (64, 64)).save(path)
    reader = _openai_reader([_openai_response(status="incomplete", reason="content_filter"),
                             _openai_response()], max_attempts=3)
    pred = reader.read(path, "s0")
    assert pred.ok and pred.attempts == 2


def test_openai_malformed_placement_is_a_scored_failure(tmp_path):
    """Structured outputs guarantee a string, not that its ranks add up to 8 --
    which is where 85% of the observed failures actually come from."""
    path = tmp_path / "b.png"
    Image.new("RGB", (64, 64)).save(path)
    pred = _openai_reader([_openai_response(placement="8/8/8")] * 3,
                          max_attempts=3).read(path, "s0")
    assert not pred.ok and pred.attempted and pred.attempts == 3


def test_usage_reasoning_tokens_survive_the_cache_and_legacy_lines_still_load():
    usage = Usage(900, 500, 0, 420)
    back = LLMPrediction.from_dict(json.loads(json.dumps(
        LLMPrediction("s0", None, None, usage, 1).as_dict())))
    assert back.usage == usage
    # Cache lines written before `reasoning_tokens` existed must still parse.
    legacy = LLMPrediction.from_dict({"sample_id": "s0", "labels": None, "placement": None,
                                      "usage": {"input_tokens": 5, "output_tokens": 7,
                                                "cache_read_input_tokens": 0},
                                      "attempts": 1})
    assert legacy.usage == Usage(5, 7, 0, 0)
