"""Prompted-VLM baseline: ask a frontier multimodal LLM to read the position.

The model is behind an HTTP API, so it is not a `BoardRecognizer` (no tensors,
no batch); it consumes an image file and emits a validated `Board`, which the
same metrics score. Token usage is accumulated per call — including retried
attempts — so a run reports what it actually cost.

One `BoardReader` subclass per provider: everything that defines the *experiment*
(prompt, schema, image prep, parsing, retry policy) lives on the base class, so
the providers can only differ in wire format, never in what the model is asked.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from PIL import Image

from chessqueries.core import Board


class Provider(str, Enum):
    """API a model is served behind. Selects the reader, and the batch backend for
    the providers that have a Batch API (`LOCAL` does not -- see `llm_batch`)."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    LOCAL = "local"


class LLMName(str, Enum):
    """Models we can prompt. Values are the API model ids, except `LOCAL`, which is
    a selector: one member covers every self-hosted checkpoint, and the served id
    is discovered from the server (see `LocalBoardReader`).
    """

    CLAUDE_OPUS_5 = "claude-opus-5"
    CLAUDE_SONNET_5 = "claude-sonnet-5"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5"
    GPT_5_6_SOL = "gpt-5.6-sol"
    GPT_5_6_TERRA = "gpt-5.6-terra"
    GPT_5_6_LUNA = "gpt-5.6-luna"
    LOCAL = "local"


class Effort(str, Enum):
    """Reasoning-effort levels, cheapest first.

    Not every provider accepts every level (`none`/`minimal` are OpenAI-only),
    so each reader declares the subset it supports and rejects the rest at
    construction rather than at the first billed request.
    """

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float

    def scaled(self, factor: float) -> "Pricing":
        return Pricing(self.input_per_mtok * factor, self.output_per_mtok * factor,
                       self.cache_read_per_mtok * factor)


# Both providers' Batch APIs bill every token at half the standard rate.
BATCH_DISCOUNT = 0.5


@dataclass(frozen=True)
class ModelInfo:
    """Where a model is served and what it costs."""

    provider: Provider
    pricing: Pricing


MODELS: dict[LLMName, ModelInfo] = {
    LLMName.CLAUDE_OPUS_5: ModelInfo(Provider.ANTHROPIC, Pricing(5.0, 25.0, 0.5)),
    LLMName.CLAUDE_SONNET_5: ModelInfo(Provider.ANTHROPIC, Pricing(3.0, 15.0, 0.3)),
    LLMName.CLAUDE_HAIKU_4_5: ModelInfo(Provider.ANTHROPIC, Pricing(1.0, 5.0, 0.1)),
    LLMName.GPT_5_6_SOL: ModelInfo(Provider.OPENAI, Pricing(5.0, 30.0, 0.5)),
    LLMName.GPT_5_6_TERRA: ModelInfo(Provider.OPENAI, Pricing(2.0, 12.0, 0.2)),
    LLMName.GPT_5_6_LUNA: ModelInfo(Provider.OPENAI, Pricing(0.2, 1.2, 0.02)),
    # Self-hosted: the GPU is already paid for, so a run's `cost_usd` is 0 and the
    # spend guard rails in the eval script have nothing to guard.
    LLMName.LOCAL: ModelInfo(Provider.LOCAL, Pricing(0.0, 0.0, 0.0)),
}


@dataclass(frozen=True)
class Usage:
    """Billable tokens. Adds across the attempts of one sample and across a run.

    ``reasoning_tokens`` is a *subset* of ``output_tokens`` (thinking is billed as
    output), so it is recorded for analysis but never priced separately. Only
    OpenAI reports it; it stays 0 elsewhere.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
            self.reasoning_tokens + other.reasoning_tokens,
        )

    def cost_usd(self, pricing: Pricing) -> float:
        """USD for these tokens at `pricing`."""
        return (
            self.input_tokens * pricing.input_per_mtok
            + self.output_tokens * pricing.output_per_mtok
            + self.cache_read_input_tokens * pricing.cache_read_per_mtok
        ) / 1e6

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


# Claude's high-resolution ceiling: images are downscaled to this long edge
# server-side anyway, so sending anything larger only costs upload bandwidth.
# GPT-5.6 resizes to a patch budget instead, and is comfortably under this.
MAX_LONG_EDGE = 2576


@dataclass(frozen=True)
class EncodedImage:
    """One image ready for the wire: base64 payload plus the MIME type that labels it
    (a base64 blob for Claude, the two spliced into a data URL for OpenAI)."""

    data: str
    media_type: str


@dataclass(frozen=True)
class ImagePrep:
    """How a dataset image becomes the bytes we send.

    ``square`` mirrors the ViT eval transform (a square resize to NxN, aspect
    ratio not preserved) so the LLM sees the *same* pixels our model does.
    ``square=None`` sends the native image, downscaled only if it exceeds
    :data:`MAX_LONG_EDGE`.
    """

    square: int | None = None
    jpeg_quality: int = 95

    @classmethod
    def parse(cls, spec: str) -> "ImagePrep":
        """``"native"`` -> native resolution; ``"644"`` -> 644x644 square resize."""
        if spec == "native":
            return cls()
        if not spec.isdigit() or int(spec) <= 0:
            raise ValueError(f"--image-size must be 'native' or a positive integer, got {spec!r}")
        return cls(square=int(spec))

    @property
    def label(self) -> str:
        return "native" if self.square is None else str(self.square)

    def encode(self, image_path: Path) -> EncodedImage:
        """Read, resize, JPEG-encode into the wire form both providers take."""
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if self.square is not None:
                img = img.resize((self.square, self.square), Image.BICUBIC)
            elif max(img.size) > MAX_LONG_EDGE:
                scale = MAX_LONG_EDGE / max(img.size)
                img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BICUBIC)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return EncodedImage(data=base64.standard_b64encode(buf.getvalue()).decode(),
                            media_type="image/jpeg")


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
# The task is orientation-free: ground truth is always canonical FEN order
# (a8 first), whatever the camera angle, so the model has to work out which way
# the board faces. That, plus the output contract, is the whole prompt -- the
# schema below enforces the shape, so the prompt does not restate it.
PROMPT_VERSION = "v1"
SCHEMA_VERSION = "v1"

SYSTEM_PROMPT = (
    "You read chess positions off images. Report the position in the standard "
    "FEN piece-placement orientation: ranks from 8 down to 1, each rank from "
    "file a to file h, White's pieces uppercase and Black's lowercase. "
    "That orientation is fixed regardless of where the camera is -- infer which "
    "way the board faces from the image and report the position accordingly. "
    "Every image shows a legal position from a real game."
)

USER_PROMPT = (
    "Give the FEN piece-placement field for this position: eight ranks "
    "separated by '/', with runs of empty squares written as digits."
)

_PLACEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "placement": {
            "type": "string",
            "description": "FEN piece-placement field only, e.g. "
                           "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
        },
    },
    "required": ["placement"],
    "additionalProperties": False,
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


# Exact protocol fingerprints supplement the human-readable versions in run
# records. The regression test pins each v1 hash, forcing intentional protocol
# edits to bump the corresponding version.
PROMPT_SHA256 = _canonical_sha256({"system": SYSTEM_PROMPT, "user": USER_PROMPT})
SCHEMA_SHA256 = _canonical_sha256(_PLACEMENT_SCHEMA)


def parse_placement(placement: str) -> Board:
    """Validate a FEN piece-placement field and lift it into a `Board`.

    Stricter than `Board.from_fen`, which is happy with ranks that do not add up
    to eight squares (they would only surface later as a 64-square mismatch).
    """
    field = placement.strip().split(" ")[0]
    ranks = field.split("/")
    if len(ranks) != 8:
        raise ValueError(f"expected 8 ranks, got {len(ranks)}: {field!r}")
    for i, rank in enumerate(ranks):
        width = sum(int(c) if c.isdigit() else 1 for c in rank)
        if width != 8:
            raise ValueError(f"rank {8 - i} spans {width} squares, expected 8: {rank!r}")
    return Board.from_fen(field)


# --------------------------------------------------------------------------- #
# Reader
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LLMPrediction:
    """One sample's outcome. `usage` covers every attempt, since we paid for all.

    Three states, and the difference matters for scoring:

    * ``ok`` -- the model returned a parseable placement.
    * ``attempted`` but not ``ok`` -- the model produced a turn we could not use
      (refusal, truncation, malformed placement). That is a genuine model
      failure, so it is scored (as an empty board) and counted.
    * not ``attempted`` -- no turn was ever produced (batch canceled/expired, or
      the image was unreadable). Scoring it as a wrong answer would invent a
      result the model never gave, so it is excluded from the metrics entirely.

    ``batch`` records how the turn was delivered, so a cell that mixes batch and
    live calls (a batch run topped up live, say) is still costed at the right rate
    per sample instead of one rate for the whole cell.
    """

    sample_id: str
    board: Board | None
    placement: str | None
    usage: Usage
    attempts: int
    error: str | None = None
    attempted: bool = True
    batch: bool = False   # delivered via the Batch API, so billed at half rate

    @property
    def ok(self) -> bool:
        return self.board is not None

    def as_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "placement": self.placement,
            "labels": self.board.labels if self.board else None,
            "usage": self.usage.as_dict(),
            "attempts": self.attempts,
            "error": self.error,
            "attempted": self.attempted,
            "batch": self.batch,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LLMPrediction":
        return cls(
            sample_id=d["sample_id"],
            board=Board.from_labels(d["labels"]) if d.get("labels") else None,
            placement=d.get("placement"),
            usage=Usage(**d.get("usage", {})),
            attempts=int(d.get("attempts", 1)),
            error=d.get("error"),
            attempted=bool(d.get("attempted", True)),
            batch=bool(d.get("batch", False)),
        )


@dataclass(frozen=True)
class Turn:
    """A usable answer from one API round-trip: the raw JSON text the model emitted
    and the tokens it billed. The provider-specific unwrapping stops here."""

    text: str
    usage: Usage


class TurnError(RuntimeError):
    """A turn that completed but produced no usable placement.

    Carries the tokens it burned, since the API bills a refused or truncated
    turn like any other. ``retryable=False`` marks a failure a second attempt
    cannot fix -- hitting ``max_tokens`` just spends the whole budget again.
    """

    def __init__(self, message: str, usage: Usage, retryable: bool = True) -> None:
        super().__init__(message)
        self.usage = usage
        self.retryable = retryable


READERS: dict[Provider, type["BoardReader"]] = {}


def api_key(field: str) -> str | None:
    """A provider credential from `.env`, or None to let the SDK read the
    environment itself.

    pydantic-settings parses `.env` without exporting into ``os.environ``, so a
    key that lives only in the file would otherwise be invisible to the SDKs.
    Returning None rather than "" preserves the fallback for keys supplied as
    shell exports -- and a blank line in `.env` (which `.env.example` ships, and
    which parses as an empty secret, not a missing one) must count as unset or it
    would shadow a working export with an empty credential.
    """
    from chessqueries.config import get_config

    secret = getattr(get_config(), field, None)
    return (secret.get_secret_value() if secret is not None else None) or None


class BoardReader(ABC):
    """Prompts one model per image and returns a validated `Board`.

    Subclasses supply only the provider's wire format -- how to build a request
    (`request_params`), how to read a returned turn (`interpret`), and how to make
    one live call (`_call`). Everything above that (prompt, schema, retry policy,
    scoring) lives here, so two providers cannot drift into asking different
    questions. Instances are safe to share across threads.

    Subclasses register themselves by `provider`, so adding one is a class
    definition and a `MODELS` entry -- nothing else needs editing.
    """

    provider: ClassVar[Provider]
    supported_efforts: ClassVar[frozenset[Effort]]

    def __init_subclass__(cls, **kw) -> None:
        super().__init_subclass__(**kw)
        if getattr(cls, "provider", None) is not None:
            READERS[cls.provider] = cls

    def __init__(
        self,
        model: LLMName | None = None,
        effort: Effort = Effort.HIGH,
        prep: ImagePrep | None = None,
        max_tokens: int = 32000,
        max_attempts: int = 3,
        client=None,
    ) -> None:
        model = model if model is not None else self.default_model
        info = MODELS[model]
        if info.provider is not self.provider:
            raise ValueError(
                f"{type(self).__name__} serves {self.provider.value} models, but "
                f"{model.value} is served by {info.provider.value}"
            )
        if effort not in self.supported_efforts:
            raise ValueError(
                f"{model.value} does not support effort={effort.value}; "
                f"choose from {sorted(e.value for e in self.supported_efforts)}"
            )
        self.client = self._default_client() if client is None else client
        self.model = model
        self.effort = effort
        self.prep = prep or ImagePrep()
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts

    @property
    def default_model(self) -> LLMName:
        raise NotImplementedError

    @property
    def pricing(self) -> Pricing:
        return MODELS[self.model].pricing

    def run_info_extra(self) -> dict:
        """Extra fields for a run record, beyond the model id and effort the caller
        already knows. Empty for the hosted APIs, whose `LLMName` pins the weights
        exactly; a local server does not, so `LocalBoardReader` names what answered.
        """
        return {}

    @abstractmethod
    def _default_client(self):
        """The provider SDK client, built lazily so the extra stays optional."""

    @abstractmethod
    def request_params(self, image_path: Path) -> dict:
        """The request body for one image. Shared by the live and batch paths, so
        the two can never drift into prompting the model differently."""

    @abstractmethod
    def interpret(self, msg) -> Turn:
        """Unwrap a returned turn. Raises `TurnError` on a turn that produced no
        usable text. Shared by the live and batch paths."""

    @abstractmethod
    def _call(self, params: dict) -> Turn:
        """One live API round-trip."""

    def to_prediction(self, sample_id: str, msg, usage_so_far: Usage | None = None,
                      attempt: int = 1) -> LLMPrediction:
        """Score one returned message into a prediction (no API call)."""
        total = usage_so_far or Usage()
        placement = None
        try:
            turn = self.interpret(msg)
            total = total + turn.usage
            placement = json.loads(turn.text)["placement"]
            return LLMPrediction(sample_id, parse_placement(placement), placement, total, attempt)
        except TurnError as exc:
            return LLMPrediction(sample_id, None, None, total + exc.usage, attempt,
                                 f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            return LLMPrediction(sample_id, None, placement, total, attempt,
                                 f"{type(exc).__name__}: {exc}")

    def read(self, image_path: Path, sample_id: str) -> LLMPrediction:
        try:
            params = self.request_params(image_path)
        except Exception as exc:  # unreadable file: fail this sample, not the run
            return LLMPrediction(sample_id, None, None, Usage(), 0,
                                 f"{type(exc).__name__}: {exc}", attempted=False)
        total = Usage()
        placement = None
        attempt = 0
        error = "no attempt made"
        for attempt in range(1, self.max_attempts + 1):
            try:
                turn = self._call(params)
                total = total + turn.usage
                placement = json.loads(turn.text)["placement"]
                return LLMPrediction(sample_id, parse_placement(placement), placement,
                                     total, attempt)
            except TurnError as exc:
                total, error = total + exc.usage, f"{type(exc).__name__}: {exc}"
                if not exc.retryable:
                    break
            except Exception as exc:  # transport failure, or an unusable placement
                error = f"{type(exc).__name__}: {exc}"
        return LLMPrediction(sample_id, None, placement, total, attempt, error)


class ClaudeBoardReader(BoardReader):
    """Anthropic Messages API: image + text in one user turn, `output_config`
    carrying both the reasoning effort and the JSON-schema response format."""

    provider = Provider.ANTHROPIC
    supported_efforts = frozenset({Effort.LOW, Effort.MEDIUM, Effort.HIGH,
                                   Effort.XHIGH, Effort.MAX})

    @property
    def default_model(self) -> LLMName:
        return LLMName.CLAUDE_OPUS_5

    def _default_client(self):
        import anthropic  # imported lazily: the `llm` group is optional

        # A long run fans out enough concurrent requests to hit the rate limit;
        # the SDK's default 2 retries give up too early for a job we do not want
        # to restart (and re-pay for).
        return anthropic.Anthropic(api_key=api_key("ANTHROPIC_API_KEY"), max_retries=8)

    def request_params(self, image_path: Path) -> dict:
        image = self.prep.encode(Path(image_path))
        return dict(
            model=self.model.value,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            output_config={
                "effort": self.effort.value,
                "format": {"type": "json_schema", "schema": _PLACEMENT_SCHEMA},
            },
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": image.media_type,
                                "data": image.data}},
                    {"type": "text", "text": USER_PROMPT},
                ],
            }],
        )

    def interpret(self, msg) -> Turn:
        usage = Usage(
            input_tokens=msg.usage.input_tokens or 0,
            output_tokens=msg.usage.output_tokens or 0,
            cache_read_input_tokens=msg.usage.cache_read_input_tokens or 0,
        )
        if msg.stop_reason == "max_tokens":
            raise TurnError(f"hit max_tokens={self.max_tokens}", usage, retryable=False)
        if msg.stop_reason == "refusal":
            raise TurnError(f"refusal ({getattr(msg, 'stop_details', None)})", usage)
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if not text:
            raise TurnError(f"no text block (stop_reason={msg.stop_reason})", usage)
        return Turn(text=text, usage=usage)

    def _call(self, params: dict) -> Turn:
        with self.client.messages.stream(**params) as stream:
            return self.interpret(stream.get_final_message())


class OpenAIBoardReader(BoardReader):
    """OpenAI Responses API. Same prompt and schema as the Claude path; the
    differences are all wire format:

    * the system prompt is ``instructions`` rather than ``system``;
    * effort moves to ``reasoning.effort`` and the schema to ``text.format``,
      which additionally requires a schema ``name`` and ``strict: true``;
    * the image is a data URL rather than a base64 block;
    * truncation surfaces as ``status="incomplete"`` instead of a stop reason.

    ``detail="high"`` is set explicitly: the default (``auto``) lets the server
    pick a downscale we do not control, which would silently break the
    matched-input comparison that `ImagePrep(square=644)` exists to guarantee.
    """

    provider = Provider.OPENAI
    supported_efforts = frozenset(Effort)   # none/minimal through max

    @property
    def default_model(self) -> LLMName:
        return LLMName.GPT_5_6_TERRA

    def _default_client(self):
        import openai  # imported lazily: the `llm` group is optional

        return openai.OpenAI(api_key=api_key("OPENAI_API_KEY"), max_retries=8)

    def request_params(self, image_path: Path) -> dict:
        image = self.prep.encode(Path(image_path))
        return dict(
            model=self.model.value,
            max_output_tokens=self.max_tokens,
            instructions=SYSTEM_PROMPT,
            reasoning={"effort": self.effort.value},
            text={"format": {"type": "json_schema", "name": "placement",
                             "schema": _PLACEMENT_SCHEMA, "strict": True}},
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_image",
                     "image_url": f"data:{image.media_type};base64,{image.data}",
                     "detail": "high"},
                    {"type": "input_text", "text": USER_PROMPT},
                ],
            }],
        )

    def interpret(self, resp) -> Turn:
        u = resp.usage
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_input_tokens=getattr(
                getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0,
            reasoning_tokens=getattr(
                getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0,
        )
        if resp.status == "incomplete":
            reason = getattr(resp.incomplete_details, "reason", None)
            # Same rule as the Claude path: a budget exhausted by thinking will be
            # exhausted again on retry, so do not re-spend it.
            raise TurnError(f"incomplete ({reason})", usage,
                            retryable=reason != "max_output_tokens")
        if resp.status != "completed":
            raise TurnError(f"status={resp.status}", usage)
        text = _openai_output_text(resp)
        if not text:
            raise TurnError(f"no output text (status={resp.status})", usage)
        return Turn(text=text, usage=usage)

    def _call(self, params: dict) -> Turn:
        with self.client.responses.stream(**params) as stream:
            return self.interpret(stream.get_final_response())


def _openai_output_text(resp) -> str | None:
    """The assistant's text from a Responses object.

    Walks ``output`` rather than using the SDK's ``output_text`` helper: batch
    results arrive as plain JSON that we revive with ``Response.model_validate``,
    and reasoning items sit in the same list, so the message block has to be
    picked out explicitly either way.
    """
    for item in getattr(resp, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", None) or []:
            if getattr(block, "type", None) == "output_text" and block.text:
                return block.text
    return None


class LocalBoardReader(BoardReader):
    """A model served on this machine behind an OpenAI-compatible HTTP API (vLLM,
    SGLang, llama.cpp).

    Chat Completions rather than the Responses API: every local server implements
    it, and its ``response_format`` carries the same placement schema the hosted
    providers are held to, so a local row stays comparable to theirs.

    Two things differ from a hosted provider, and both are recorded in
    `run_info_extra` because `LLMName.LOCAL` does not pin them: the *served model*
    (discovered from ``/v1/models``, so one enum member covers a base model and
    every fine-tune of it) and the *sampling* (greedy by default -- a local eval
    should be reproducible, and pass@k has `read_samples` instead).
    """

    provider = Provider.LOCAL
    # Local weights have no reasoning-effort dial. Declaring only `none` rejects a
    # run whose `--effort` would otherwise be silently dropped, and keeps the
    # recorded protocol honest.
    supported_efforts = frozenset({Effort.NONE})

    def __init__(self, *args, base_url: str | None = None, temperature: float = 0.0,
                 top_p: float = 1.0, served_model: str | None = None, **kwargs) -> None:
        from chessqueries.config import get_config  # lazy, as in `api_key`

        # Set before `super().__init__`, which builds the client from it.
        self.base_url = base_url or get_config().LOCAL_BASE_URL
        self.temperature = temperature
        self.top_p = top_p
        super().__init__(*args, **kwargs)
        self.served_model = served_model or self._discover_served_model()

    @property
    def default_model(self) -> LLMName:
        return LLMName.LOCAL

    def _default_client(self):
        import openai  # imported lazily: the `llm` group is optional

        # Local servers ignore the credential but the SDK insists on one.
        return openai.OpenAI(base_url=self.base_url, api_key="EMPTY", max_retries=8)

    def _discover_served_model(self) -> str:
        """The id the server answers to, so the wire `model` field and the run record
        name the actual checkpoint. Fails here rather than on the first image."""
        try:
            served = [m.id for m in self.client.models.list().data]
        except Exception as exc:
            raise RuntimeError(
                f"no OpenAI-compatible server reachable at {self.base_url} "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        if not served:
            raise RuntimeError(f"{self.base_url} serves no model")
        if len(served) > 1:
            raise RuntimeError(
                f"{self.base_url} serves {served}; pass served_model= to pick one"
            )
        return served[0]

    def request_params(self, image_path: Path) -> dict:
        image = self.prep.encode(Path(image_path))
        return dict(
            model=self.served_model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "placement", "strict": True,
                                             "schema": _PLACEMENT_SCHEMA}},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{image.media_type};base64,{image.data}"}},
                    {"type": "text", "text": USER_PROMPT},
                ]},
            ],
        )

    def interpret(self, resp) -> Turn:
        u = getattr(resp, "usage", None)
        # Tokens are free here, but they still say how much the model spent on an
        # image, which is the comparison against the hosted rows.
        usage = Usage(
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
        )
        if not resp.choices:
            raise TurnError("no choices returned", usage)
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            # Same rule as the hosted readers: a budget already exhausted will be
            # exhausted again.
            raise TurnError(f"hit max_tokens={self.max_tokens}", usage, retryable=False)
        text = choice.message.content
        if not text:
            raise TurnError(f"empty message (finish_reason={choice.finish_reason})", usage)
        return Turn(text=text, usage=usage)

    def _call(self, params: dict) -> Turn:
        # Not streamed: the hosted readers stream to survive long thinking turns on
        # a remote connection, which a localhost request does not need.
        return self.interpret(self.client.chat.completions.create(**params))

    def read_samples(self, image_path: Path, sample_id: str, k: int,
                     temperature: float | None = None) -> list[LLMPrediction]:
        """`k` independent answers for one image, from a single ``n=k`` request.

        Local-only on purpose: sampling k times off a metered API is a spending
        decision, while here it is free -- and pass@k is what tells us whether RL
        has any headroom to convert.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        params = self.request_params(Path(image_path))
        params.update(n=k, temperature=self.temperature if temperature is None
                      else temperature)
        resp = self.client.chat.completions.create(**params)
        # One `usage` covers the whole request, so it is attributed to the first
        # sample rather than split k ways: a sum over samples then still equals what
        # the request actually spent. Each choice is scored through the normal path
        # by handing `to_prediction` a response holding just that choice.
        return [
            self.to_prediction(
                sample_id,
                SimpleNamespace(choices=[choice], usage=resp.usage if i == 0 else None),
            )
            for i, choice in enumerate(resp.choices)
        ]

    def run_info_extra(self) -> dict:
        return {
            "base_url": self.base_url,
            "served_model": self.served_model,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


def get_reader(model: LLMName, **kwargs) -> BoardReader:
    """The reader for `model`, chosen by the provider it is served behind."""
    return READERS[MODELS[model].provider](model=model, **kwargs)


def outcome_breakdown(scored: list[LLMPrediction],
                      gt: dict[str, list[int]]) -> dict[str, float]:
    """Split scored predictions into the two ways this baseline can be wrong:
    a malformed placement is an *output-contract* failure (the model could not emit
    64 squares); a well-formed placement that is not the position is a *vision*
    failure. The three buckets (exact / well-formed-but-wrong / unparseable)
    partition the scored set exactly once each. Requires every prediction to be
    `attempted`; never-run samples are excluded upstream and are not a model outcome.
    """
    if not scored:
        raise ValueError("no scored predictions to break down")
    failed = [p for p in scored if not p.ok]
    n_exact = sum(1 for p in scored if p.ok and p.board.labels == gt[p.sample_id])
    n_parsed = len(scored) - len(failed)
    return {
        "n_parsed": n_parsed,
        "n_parsed_wrong": n_parsed - n_exact,
        "parsed_wrong_fraction": (n_parsed - n_exact) / len(scored),
        "n_failed_parse": len(failed),
        "failed_parse_fraction": len(failed) / len(scored),
    }
