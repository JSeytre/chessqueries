"""LLM baseline, OpenAI batch delivery: the JSONL round trip against a stubbed
files+batches client."""
import json

from llm_stubs import _openai_response, _sample  # noqa: E402

from chessqueries.models import llm_batch
from chessqueries.models.llm import ImagePrep, OpenAIBoardReader, Usage


# --------------------------------------------------------------------------- #
# OpenAI batch delivery
# --------------------------------------------------------------------------- #
class _OpenAIBatchClient:
    """Files + batches, enough of them to exercise the JSONL round trip."""

    def __init__(self, output_lines=(), error_lines=()):
        self.uploaded = []          # list[list[dict]] -- one entry per created file
        self.output_lines = list(output_lines)
        self.error_lines = list(error_lines)
        self.files = self
        self.batches = self

    # --- files ---
    def create(self, **kw):
        if "file" in kw:            # files.create
            payload = kw["file"].getvalue().decode()
            self.uploaded.append([json.loads(x) for x in payload.splitlines() if x.strip()])
            return type("F", (), {"id": f"file_{len(self.uploaded) - 1}"})()
        # batches.create
        self.batch_kwargs = kw
        return type("B", (), {"id": "batch_0"})()

    def content(self, file_id):
        lines = self.output_lines if file_id == "out_0" else self.error_lines
        return type("C", (), {"text": "".join(json.dumps(x) + "\n" for x in lines)})()

    # --- batches ---
    def retrieve(self, bid):
        counts = type("C", (), {"total": 2, "completed": 1, "failed": 1})()
        return type("B", (), {
            "id": bid, "status": "completed", "request_counts": counts,
            "output_file_id": "out_0" if self.output_lines else None,
            "error_file_id": "err_0" if self.error_lines else None})()


def _batch_line(custom_id, resp=None, status_code=200, error=None):
    return {"custom_id": custom_id, "error": error,
            "response": None if resp is None else {
                "status_code": status_code,
                "body": json.loads(resp.model_dump_json()) if resp is not None else None}}


def test_openai_batch_submits_the_documented_jsonl_shape(tmp_path):
    client = _OpenAIBatchClient()
    reader = OpenAIBoardReader(client=client, prep=ImagePrep(square=644))
    samples = [_sample(tmp_path, f"s{i}") for i in range(3)]

    submission = llm_batch.submit(reader, samples)
    assert submission.job.batch_ids == ["batch_0"] and not submission.skipped
    assert client.batch_kwargs["endpoint"] == "/v1/responses"
    assert client.batch_kwargs["completion_window"] == "24h"
    assert client.batch_kwargs["input_file_id"] == "file_0"

    lines = client.uploaded[0]
    assert [x["custom_id"] for x in lines] == ["s0", "s1", "s2"]
    for line in lines:
        assert line["method"] == "POST" and line["url"] == "/v1/responses"
        # The body must be a request the Responses API would accept as-is.
        assert line["body"]["reasoning"]["effort"] and line["body"]["max_output_tokens"]
        assert line["body"]["input"][0]["content"][0]["type"] == "input_image"


def test_openai_batch_separates_never_run_from_model_failures():
    """The error file holds requests that never produced a turn (and were never
    billed) -- they must be re-run. A 200 whose body is `incomplete` IS a billed
    turn and must be scored instead."""
    client = _OpenAIBatchClient(
        output_lines=[
            _batch_line("ok", _openai_response()),
            _batch_line("truncated", _openai_response(status="incomplete",
                                                      reason="max_output_tokens")),
        ],
        error_lines=[
            _batch_line("rejected", _openai_response(), status_code=400),
            _batch_line("boom", None, error={"message": "server exploded"}),
        ],
    )
    reader = OpenAIBoardReader(client=client)
    preds = {p.sample_id: p for p in llm_batch.collect(reader, llm_batch.BatchJob(["batch_0"]))}

    assert preds["ok"].ok and preds["ok"].attempted and preds["ok"].batch
    assert preds["ok"].usage.reasoning_tokens == 420

    # Billed, unusable -> scored as a model failure.
    assert not preds["truncated"].ok and preds["truncated"].attempted
    assert preds["truncated"].usage.output_tokens == 500

    # Never produced a turn -> excluded from scoring, re-run instead.
    for sid in ("rejected", "boom"):
        assert not preds[sid].ok and not preds[sid].attempted
        assert preds[sid].usage == Usage()


def test_openai_batch_status_reports_terminality():
    reader = OpenAIBoardReader(client=_OpenAIBatchClient())
    (st,) = llm_batch.status(reader, llm_batch.BatchJob(["batch_0"]))
    assert st.terminal and st.status == "completed"
    assert st.counts["succeeded"] == 1 and st.counts["errored"] == 1
