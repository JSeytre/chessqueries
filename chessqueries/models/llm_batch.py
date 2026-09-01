"""Batch-API delivery for the prompted-VLM baseline: half the price of live calls.

Submission and collection are separate steps, and the batch ids are persisted, so
a job survives a restart: re-running picks the batches back up instead of paying
for the same images twice. Requests are built by `BoardReader.request_params` and
responses scored by `BoardReader.to_prediction`, so batch and live runs prompt the
model identically and are directly comparable.

One backend per provider, because the two APIs differ in shape: Anthropic takes
requests inline and returns results by batch id, while OpenAI wants a JSONL file
uploaded first and splits the results across an output *and* an error file. Both
honour the same contract -- a request that never produced a model turn comes back
``attempted=False`` so it is re-run rather than scored as a wrong answer.
"""
from __future__ import annotations

import io
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import ClassVar, Iterator

from chessqueries.models.llm import BoardReader, LLMPrediction, Provider, Usage

# Anthropic caps a batch at 100k requests / 256 MB; OpenAI at 50k requests /
# 200 MB. Base64 JPEGs dominate the body (~130 KB each at 644px), so chunk on
# request count -- 500 keeps us near 65 MB -- rather than trying to predict bytes.
MAX_REQUESTS_PER_BATCH = 500

# custom_id must be short and URL-safe. Every dataset's sample_id already is
# (`G000_IMG000`, `0cdIkt4Q2zI_355399`, `IMG_6289`, `0046`), so we use it verbatim
# and fail loudly rather than mangling ids into an unmappable form.
_CUSTOM_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def custom_id_for(sample_id: str) -> str:
    if not _CUSTOM_ID.match(sample_id):
        raise ValueError(
            f"sample_id {sample_id!r} is not usable as a batch custom_id "
            f"(need 1-64 chars of [A-Za-z0-9_-]); batch mode cannot map it back."
        )
    return sample_id


@dataclass
class BatchJob:
    """The submitted batch ids for one (dataset, split, config) run."""

    batch_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"batch_ids": self.batch_ids}

    @classmethod
    def from_dict(cls, d: dict) -> "BatchJob":
        return cls(batch_ids=list(d.get("batch_ids", [])))


@dataclass(frozen=True)
class BatchStatus:
    """One batch's state. `terminal` is the provider-agnostic 'stop polling' flag."""

    batch_id: str
    status: str
    terminal: bool
    counts: dict[str, int]


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
BACKENDS: dict[Provider, type["BatchBackend"]] = {}


class BatchBackend(ABC):
    """Provider-specific batch transport. Registers itself by `provider`."""

    provider: ClassVar[Provider]

    def __init_subclass__(cls, **kw) -> None:
        super().__init_subclass__(**kw)
        if getattr(cls, "provider", None) is not None:
            BACKENDS[cls.provider] = cls

    @abstractmethod
    def create(self, reader: BoardReader, chunk: list[tuple[str, dict]]) -> str:
        """Upload one chunk of (custom_id, request_params); return the batch id."""

    @abstractmethod
    def status(self, reader: BoardReader, batch_id: str) -> BatchStatus:
        ...

    @abstractmethod
    def results(self, reader: BoardReader, batch_id: str) -> Iterator[LLMPrediction]:
        """Every finished request in one batch, as a prediction."""


class AnthropicBatch(BatchBackend):
    provider = Provider.ANTHROPIC

    TERMINAL = "ended"

    def create(self, reader, chunk):
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        requests = [Request(custom_id=cid,
                            params=MessageCreateParamsNonStreaming(**params))
                    for cid, params in chunk]
        return reader.client.messages.batches.create(requests=requests).id

    def status(self, reader, batch_id):
        b = reader.client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        return BatchStatus(batch_id, b.processing_status,
                           b.processing_status == self.TERMINAL,
                           {"processing": c.processing, "succeeded": c.succeeded,
                            "errored": c.errored, "canceled": c.canceled,
                            "expired": c.expired})

    def results(self, reader, batch_id):
        for entry in reader.client.messages.batches.results(batch_id):
            result = entry.result
            if result.type == "succeeded":
                yield replace(reader.to_prediction(entry.custom_id, result.message),
                              batch=True)
            else:
                # canceled / expired / errored: no message was created (and none
                # was billed), so this must be re-run, not scored.
                detail = getattr(result, "error", None) or result.type
                yield LLMPrediction(entry.custom_id, None, None, Usage(), 0,
                                    f"batch_{result.type}: {detail}",
                                    attempted=False, batch=True)


class OpenAIBatch(BatchBackend):
    """Files API upload -> batch -> output/error files.

    The error file is read as carefully as the output file: a request that failed
    validation produced no model turn, and dropping it silently would let a cell
    quietly go short of the images it claims.
    """

    provider = Provider.OPENAI

    ENDPOINT = "/v1/responses"
    TERMINAL = frozenset({"completed", "failed", "expired", "cancelled"})

    def create(self, reader, chunk):
        lines = "".join(
            json.dumps({"custom_id": cid, "method": "POST",
                        "url": self.ENDPOINT, "body": params}) + "\n"
            for cid, params in chunk
        )
        buf = io.BytesIO(lines.encode())
        buf.name = "batch.jsonl"   # the SDK infers the upload filename from this
        upload = reader.client.files.create(file=buf, purpose="batch")
        batch = reader.client.batches.create(
            input_file_id=upload.id, endpoint=self.ENDPOINT, completion_window="24h")
        return batch.id

    def status(self, reader, batch_id):
        b = reader.client.batches.retrieve(batch_id)
        c = b.request_counts
        total, completed, failed = (getattr(c, "total", 0) or 0,
                                    getattr(c, "completed", 0) or 0,
                                    getattr(c, "failed", 0) or 0)
        return BatchStatus(batch_id, b.status, b.status in self.TERMINAL,
                           {"processing": max(total - completed - failed, 0),
                            "succeeded": completed, "errored": failed,
                            "canceled": 0, "expired": 0})

    def results(self, reader, batch_id):
        from openai.types.responses import Response

        batch = reader.client.batches.retrieve(batch_id)
        for file_id in (batch.output_file_id, batch.error_file_id):
            if not file_id:
                continue
            for line in reader.client.files.content(file_id).text.splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                sample_id = entry.get("custom_id")
                resp = entry.get("response") or {}
                if entry.get("error") or resp.get("status_code") != 200:
                    detail = entry.get("error") or f"status_code={resp.get('status_code')}"
                    yield LLMPrediction(sample_id, None, None, Usage(), 0,
                                        f"batch_errored: {detail}",
                                        attempted=False, batch=True)
                    continue
                # A 200 whose body says `incomplete` IS a billed model turn, so it
                # goes through `to_prediction` and is scored like any other failure.
                msg = Response.model_validate(resp["body"])
                yield replace(reader.to_prediction(sample_id, msg), batch=True)


def backend_for(reader: BoardReader) -> BatchBackend:
    """The batch transport for `reader`'s provider.

    Not every provider has one -- a self-hosted server has no Batch API -- so this
    says so instead of raising a bare KeyError from inside a submission.
    """
    try:
        return BACKENDS[reader.provider]()
    except KeyError:
        raise ValueError(
            f"{reader.provider.value} has no Batch API; run without --batch "
            f"(available: {sorted(p.value for p in BACKENDS)})"
        ) from None


# --------------------------------------------------------------------------- #
# Provider-agnostic entry points
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Submission:
    """What a batch upload produced: the job tracking the created batches, and
    placeholder predictions for the samples that never made it into one (an unreadable
    image), marked ``attempted=False`` so they are re-run rather than scored as answers
    the model never gave."""

    job: BatchJob
    skipped: list[LLMPrediction]


def submit(reader: BoardReader, samples, on_submitted=None) -> Submission:
    """Upload every sample as batch requests.

    ``on_submitted(job)`` is called after **each** chunk is created, so the caller
    can persist the ids as they appear. Without that, a failure part-way through a
    multi-chunk submission would orphan the batches already created and billed --
    a re-run would pay for them a second time.
    """
    backend = backend_for(reader)
    requests, failed = [], []
    for s in samples:
        try:
            params = reader.request_params(s.image_path)
        except Exception as exc:
            failed.append(LLMPrediction(s.sample_id, None, None, Usage(), 0,
                                        f"{type(exc).__name__}: {exc}", attempted=False))
            continue
        requests.append((custom_id_for(s.sample_id), params))

    job = BatchJob()
    for i in range(0, len(requests), MAX_REQUESTS_PER_BATCH):
        job.batch_ids.append(backend.create(reader, requests[i : i + MAX_REQUESTS_PER_BATCH]))
        if on_submitted is not None:
            on_submitted(job)
    return Submission(job=job, skipped=failed)


def status(reader: BoardReader, job: BatchJob) -> list[BatchStatus]:
    """Each batch's state; poll until every one is `terminal`."""
    backend = backend_for(reader)
    return [backend.status(reader, bid) for bid in job.batch_ids]


def collect(reader: BoardReader, job: BatchJob) -> Iterator[LLMPrediction]:
    """Stream every finished batch's results as predictions.

    Results arrive in arbitrary order, so each is keyed by its own ``custom_id``
    rather than by position.
    """
    backend = backend_for(reader)
    for bid in job.batch_ids:
        yield from backend.results(reader, bid)
