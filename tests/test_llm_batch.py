"""LLM baseline, Anthropic batch delivery: chunked submit, resume, and the
outcome breakdown — against a stubbed batches client."""
from pathlib import Path

import pytest
from llm_stubs import START, _Msg, _payload, _sample  # noqa: E402

from chessqueries.core import Board
from chessqueries.models import llm_batch
from chessqueries.models.llm import ClaudeBoardReader, LLMPrediction, Usage, outcome_breakdown


# --------------------------------------------------------------------------- #
# Batch delivery, against a stubbed batches client
# --------------------------------------------------------------------------- #
class _StubBatches:
    """Records created batches and replays scripted results per batch id."""

    def __init__(self, results_by_batch=None, status="ended"):
        self.created = []                       # list[list[custom_id]]
        self.results_by_batch = results_by_batch or {}
        self._status = status

    def create(self, requests):
        bid = f"msgbatch_{len(self.created)}"
        self.created.append([r["custom_id"] for r in requests])
        return type("B", (), {"id": bid})()

    def retrieve(self, bid):
        counts = type("C", (), {"processing": 0, "succeeded": 0, "errored": 0,
                                "canceled": 0, "expired": 0})()
        return type("B", (), {"id": bid, "processing_status": self._status,
                              "request_counts": counts})()

    def results(self, bid):
        return iter(self.results_by_batch.get(bid, []))


class _BatchClient:
    def __init__(self, batches):
        self.messages = type("M", (), {"batches": batches})()


def _entry(custom_id, kind="succeeded", msg=None):
    result = type("R", (), {"type": kind, "message": msg, "error": None})()
    return type("E", (), {"custom_id": custom_id, "result": result})()


def test_batch_submit_persists_ids_after_every_chunk(tmp_path, monkeypatch):
    """A failure part-way through a multi-chunk submission must not orphan the
    batches already created and billed -- their ids have to be on disk."""
    monkeypatch.setattr(llm_batch, "MAX_REQUESTS_PER_BATCH", 2)
    batches = _StubBatches()
    reader = ClaudeBoardReader(client=_BatchClient(batches))
    samples = [_sample(tmp_path, f"s{i}") for i in range(5)]

    seen = []
    llm_batch.submit(reader, samples, on_submitted=lambda j: seen.append(list(j.batch_ids)))
    assert [len(c) for c in batches.created] == [2, 2, 1]
    # Ids are handed over cumulatively as each chunk lands, not only at the end.
    assert seen == [["msgbatch_0"], ["msgbatch_0", "msgbatch_1"],
                    ["msgbatch_0", "msgbatch_1", "msgbatch_2"]]


def test_batch_unreadable_image_is_marked_never_run(tmp_path):
    """Same contract as the live path: a request that never reached the model must
    not be scored as a wrong answer."""
    bad = tmp_path / "bad.png"
    bad.write_text("garbage")
    sample = type("S", (), {"sample_id": "s0", "image_path": bad})()
    batches = _StubBatches()
    reader = ClaudeBoardReader(client=_BatchClient(batches))

    submission = llm_batch.submit(reader, [sample])
    assert submission.job.batch_ids == [] and len(submission.skipped) == 1
    assert not submission.skipped[0].attempted


def test_batch_non_succeeded_results_are_never_run():
    batches = _StubBatches(results_by_batch={"b0": [
        _entry("ok", "succeeded", _Msg(_payload(START))),
        _entry("gone", "expired"),
        _entry("boom", "errored"),
    ]})
    reader = ClaudeBoardReader(client=_BatchClient(batches))
    preds = {p.sample_id: p for p in
             llm_batch.collect(reader, llm_batch.BatchJob(["b0"]))}
    assert preds["ok"].ok and preds["ok"].attempted and preds["ok"].batch
    for sid in ("gone", "boom"):
        assert not preds[sid].ok and not preds[sid].attempted


def test_parse_breakdown_splits_contract_failures_from_vision_failures(tmp_path):
    """The two failure modes must be reported apart: a malformed placement means
    the model could not emit 64 squares, a well-formed wrong one means it could
    not read the board. Conflating them makes a vision result look like a
    formatting bug."""
    gt_board = Board.from_fen(START)
    other = Board.from_fen("8/8/8/4k3/8/8/8/4K3")
    # 1 exact, 2 well-formed but wrong, 1 unparseable.
    scored = [LLMPrediction("s0", gt_board, START, Usage(), 1),
              LLMPrediction("s1", other, "x", Usage(), 1),
              LLMPrediction("s2", other, "x", Usage(), 1),
              LLMPrediction("s3", None, None, Usage(), 1, "ValueError: bad")]
    truth = {f"s{i}": gt_board.labels for i in range(4)}

    out = outcome_breakdown(scored, truth)
    assert out["n_parsed"] == 3
    assert out["n_parsed_wrong"] == 2
    assert out["parsed_wrong_fraction"] == pytest.approx(0.5)
    assert out["n_failed_parse"] == 1
    assert out["failed_parse_fraction"] == pytest.approx(0.25)
    # The three buckets must partition the scored set exactly once each.
    n_exact = out["n_parsed"] - out["n_parsed_wrong"]
    assert n_exact + out["n_parsed_wrong"] + out["n_failed_parse"] == len(scored)


def test_outcome_breakdown_counts_an_empty_prediction_as_parsed(tmp_path):
    """An all-empty board is a real prediction the model could make, not a parse
    failure -- `board` is falsy-adjacent, so the split must key on `ok`."""
    empty = Board.from_fen("8/8/8/8/8/8/8/8")
    out = outcome_breakdown([LLMPrediction("s0", empty, "8/8/8/8/8/8/8/8", Usage(), 1)],
                            {"s0": Board.from_fen(START).labels})
    assert out["n_parsed"] == 1 and out["n_failed_parse"] == 0
    assert out["n_parsed_wrong"] == 1


def test_finished_batch_job_file_is_removed_so_new_work_can_be_submitted(tmp_path):
    """Regression: the job file used to survive collection, so every later run
    re-collected the same finished batches and silently submitted nothing -- a
    cell with expired/errored requests could never be completed."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eval_llm_baseline", Path(__file__).parents[1] / "scripts" / "eval_llm_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    job_path = tmp_path / "batch_slcc_test.json"
    samples = [_sample(tmp_path, "s0"), _sample(tmp_path, "s1")]

    # Round 1: s0 succeeds, s1 expires (never reached the model).
    batches = _StubBatches(results_by_batch={"msgbatch_0": [
        _entry("s0", "succeeded", _Msg(_payload(START))),
        _entry("s1", "expired"),
    ]})
    reader = ClaudeBoardReader(client=_BatchClient(batches))
    recorded = {}

    def record(pred):
        recorded[pred.sample_id] = pred
        return pred

    def needs_run(s):
        pred = recorded.get(s.sample_id)
        return pred is None or not pred.attempted

    assert mod._run_batch(reader, samples, needs_run, job_path, record,
                          wait=False, poll_seconds=0)
    assert not job_path.exists(), "finished job file must not be left behind"
    assert batches.created == [["s0", "s1"]]
    assert needs_run(samples[1]), "the expired sample still needs running"

    # Round 2: the expired sample must be resubmitted, not silently skipped.
    batches.results_by_batch["msgbatch_1"] = [_entry("s1", "succeeded", _Msg(_payload(START)))]
    assert mod._run_batch(reader, samples, needs_run, job_path, record,
                          wait=False, poll_seconds=0)
    assert batches.created[1] == ["s1"]
    assert recorded["s1"].ok and not job_path.exists()
