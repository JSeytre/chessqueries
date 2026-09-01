"""LLM baseline, `Provider.LOCAL`: a self-hosted OpenAI-compatible server. Same
prompt/schema as the hosted providers, plus served-model discovery and pass@k
sampling — against a stubbed server."""
import pytest
from llm_stubs import START, _chat_response, _decoded_size, _local_reader  # noqa: E402
from PIL import Image

from chessqueries.core import Board
from chessqueries.models import llm_batch
from chessqueries.models.llm import (
    MODELS,
    SYSTEM_PROMPT,
    USER_PROMPT,
    ClaudeBoardReader,
    Effort,
    ImagePrep,
    LLMName,
    LocalBoardReader,
    Provider,
    Usage,
    get_reader,
)


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "b.png"
    Image.new("RGB", (1200, 800)).save(path)
    return path


# --------------------------------------------------------------------------- #
# Registration and cost
# --------------------------------------------------------------------------- #
def test_get_reader_dispatches_to_the_local_reader():
    reader = get_reader(LLMName.LOCAL, client=_local_reader().client, effort=Effort.NONE)
    assert isinstance(reader, LocalBoardReader)
    assert MODELS[LLMName.LOCAL].provider is Provider.LOCAL


def test_a_local_run_is_free():
    """The GPU is already paid for, so the eval script's spend guard rails have
    nothing to guard and a local row must never report a dollar figure."""
    assert Usage(10_000_000, 10_000_000, 10_000_000).cost_usd(
        MODELS[LLMName.LOCAL].pricing) == 0.0


def test_only_effort_none_is_accepted():
    """Local weights have no reasoning-effort dial; an `--effort high` that would be
    silently dropped must fail instead, so the recorded protocol stays honest."""
    with pytest.raises(ValueError, match="does not support effort=high"):
        _local_reader(effort=Effort.HIGH)
    _local_reader(effort=Effort.NONE)  # accepted


def test_local_has_no_batch_api():
    with pytest.raises(ValueError, match="has no Batch API"):
        llm_batch.backend_for(_local_reader())


# --------------------------------------------------------------------------- #
# Served-model discovery
# --------------------------------------------------------------------------- #
def test_the_served_checkpoint_is_discovered_and_recorded():
    """`LLMName.LOCAL` covers a base model and every fine-tune of it, so the run
    record — not the enum — is what says which checkpoint answered."""
    reader = _local_reader(served=("out/qwen3vl-sft-merged",))
    assert reader.served_model == "out/qwen3vl-sft-merged"
    extra = reader.run_info_extra()
    assert extra["served_model"] == "out/qwen3vl-sft-merged"
    assert extra["base_url"].endswith("/v1") and extra["temperature"] == 0.0


def test_an_ambiguous_or_empty_server_fails_at_construction():
    """Fail before the first image, not 2000 images in."""
    with pytest.raises(RuntimeError, match="serves no model"):
        _local_reader(served=())
    with pytest.raises(RuntimeError, match="pass served_model="):
        _local_reader(served=("a", "b"))
    # Explicitly named, discovery is skipped and ambiguity is moot.
    assert _local_reader(served=("a", "b"), served_model="b").served_model == "b"


def test_an_unreachable_server_says_so():
    reader_cls_kwargs = dict(effort=Effort.NONE, base_url="http://localhost:9/v1")

    class _Dead:
        def __init__(self):
            self.models = self

        def list(self):
            raise ConnectionError("refused")

    with pytest.raises(RuntimeError, match="no OpenAI-compatible server reachable"):
        LocalBoardReader(client=_Dead(), **reader_cls_kwargs)


# --------------------------------------------------------------------------- #
# Request shape
# --------------------------------------------------------------------------- #
def test_request_params_match_the_chat_completions_shape(image):
    reader = _local_reader(prep=ImagePrep(square=644), max_tokens=512)
    params = reader.request_params(image)

    assert params["model"] == "qwen3-vl-8b"          # the discovered id, not "local"
    assert params["max_tokens"] == 512
    assert params["temperature"] == 0.0              # greedy: a local eval is reproducible
    fmt = params["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True

    system, user = params["messages"]
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    img, text = user["content"]
    assert img["type"] == "image_url" and text["text"] == USER_PROMPT
    assert img["image_url"]["url"].startswith("data:image/jpeg;base64,")
    # The square resize must survive the data-URL wrapper, or the matched-input
    # comparison against the hosted rows silently stops being matched.
    assert _decoded_size(img["image_url"]["url"].split(",", 1)[1]) == (644, 644)


def test_local_asks_the_identical_question_to_the_hosted_providers(image):
    """The experiment is the prompt + schema + pixels; only the wire format may
    differ. If these drift, a local row is no longer comparable to the API rows."""
    prep = ImagePrep(square=644)
    local = _local_reader(prep=prep).request_params(image)
    claude = ClaudeBoardReader(client=object(), prep=prep).request_params(image)

    assert local["messages"][0]["content"] == claude["system"] == SYSTEM_PROMPT
    assert (local["messages"][1]["content"][1]["text"]
            == claude["messages"][0]["content"][1]["text"] == USER_PROMPT)
    assert (local["response_format"]["json_schema"]["schema"]
            == claude["output_config"]["format"]["schema"])
    assert (local["messages"][1]["content"][0]["image_url"]["url"].split(",", 1)[1]
            == claude["messages"][0]["content"][0]["source"]["data"])


# --------------------------------------------------------------------------- #
# Interpretation
# --------------------------------------------------------------------------- #
def test_read_returns_a_board_and_counts_tokens(image):
    pred = _local_reader([_chat_response()]).read(image, "s0")
    assert pred.ok and pred.board.labels == Board.from_fen(START).labels
    assert pred.usage == Usage(420, 30)


def test_truncation_is_not_retried(image):
    """Mirrors the hosted readers: an exhausted output budget will be exhausted
    again, so one attempt."""
    reader = _local_reader([_chat_response(finish_reason="length")] * 3, max_attempts=3)
    pred = reader.read(image, "s0")
    assert not pred.ok and pred.attempts == 1 and reader.client.calls == 1
    assert "max_tokens" in pred.error


def test_an_empty_message_is_retried(image):
    reader = _local_reader([_chat_response(placements=(None,)), _chat_response()],
                           max_attempts=3)
    pred = reader.read(image, "s0")
    assert pred.ok and pred.attempts == 2


def test_a_malformed_placement_is_a_scored_failure(image):
    """Guided decoding guarantees a string, not that its ranks add up to eight."""
    pred = _local_reader([_chat_response(placements=("8/8/8",))] * 3,
                         max_attempts=3).read(image, "s0")
    assert not pred.ok and pred.attempted and pred.attempts == 3


# --------------------------------------------------------------------------- #
# pass@k sampling
# --------------------------------------------------------------------------- #
def test_read_samples_returns_k_scored_answers_from_one_request(image):
    wrong = "8/8/8/8/8/8/8/8"
    reader = _local_reader([_chat_response(placements=(START, wrong, "nonsense"))])
    preds = reader.read_samples(image, "s0", k=3, temperature=1.0)

    assert reader.client.calls == 1 and reader.client.kwargs["n"] == 3
    assert reader.client.kwargs["temperature"] == 1.0   # overrides the greedy default
    assert [p.ok for p in preds] == [True, True, False]
    assert preds[0].board.labels == Board.from_fen(START).labels
    # One `usage` covers the whole request, so it is attributed once: summing over
    # the k samples must equal what the request actually spent, not k times it.
    assert sum((p.usage for p in preds), Usage()) == Usage(420, 30)


def test_read_samples_rejects_a_nonsensical_k(image):
    with pytest.raises(ValueError, match="k must be >= 1"):
        _local_reader().read_samples(image, "s0", k=0)
