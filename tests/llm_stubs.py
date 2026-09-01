"""Stubbed provider clients + tiny builders shared by the LLM-baseline tests
(importable because pytest puts each test file's directory on sys.path)."""
import json

from PIL import Image

from chessqueries.models.llm import ClaudeBoardReader, LocalBoardReader, OpenAIBoardReader

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def _payload(placement):
    return json.dumps({"placement": placement})


def _decoded_size(b64: str) -> tuple[int, int]:
    import base64
    import io

    with Image.open(io.BytesIO(base64.standard_b64decode(b64))) as img:
        return img.size


def _sample(tmp_path, sid):
    path = tmp_path / f"{sid}.png"
    Image.new("RGB", (64, 64)).save(path)
    return type("S", (), {"sample_id": sid, "image_path": path})()


# --- Anthropic ---------------------------------------------------------------
class _StubStream:
    def __init__(self, msg):
        self._msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._msg


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Msg:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_Block(text)] if text is not None else []
        self.stop_reason = stop_reason
        self.stop_details = None
        self.usage = type("U", (), {"input_tokens": 1000, "output_tokens": 100,
                                    "cache_read_input_tokens": 0})()


class _StubClient:
    """Replays a scripted list of messages/exceptions, one per API call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.messages = self

    def stream(self, **kwargs):
        self.kwargs = kwargs
        item = self.script[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return _StubStream(item)


def _reader(script, **kw):
    return ClaudeBoardReader(client=_StubClient(script), **kw)


# --- OpenAI -------------------------------------------------------------------
class _OpenAIStream:
    def __init__(self, resp):
        self._resp = resp

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_response(self):
        return self._resp


class _OpenAIStubClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.responses = self

    def stream(self, **kwargs):
        self.kwargs = kwargs
        item = self.script[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return _OpenAIStream(item)


def _openai_reader(script=(), **kw):
    return OpenAIBoardReader(client=_OpenAIStubClient(script), **kw)


def _openai_response(placement=START, status="completed", reason=None,
                     in_tok=900, out_tok=500, reasoning=420):
    """A real `Response` object, revived the same way batch results are."""
    from openai.types.responses import Response

    body = {
        "id": "resp_1", "object": "response", "created_at": 0,
        "model": "gpt-5.6-terra", "status": status,
        "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
        "output": [{"type": "message", "id": "m1", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "annotations": [],
                                 "text": json.dumps({"placement": placement})}]}],
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok,
                  "total_tokens": in_tok + out_tok,
                  "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": reasoning}},
    }
    if reason is not None:
        body["incomplete_details"] = {"reason": reason}
    return Response.model_validate(body)


# --- Local (OpenAI-compatible server) ----------------------------------------
class _LocalStubClient:
    """A stubbed OpenAI-compatible server: a scripted list of chat completions plus
    the ``/v1/models`` listing the reader discovers the served id from."""

    def __init__(self, script=(), served=("qwen3-vl-8b",)):
        self.script = list(script)
        self.served = list(served)
        self.calls = 0
        self.chat = self
        self.completions = self
        self.models = self

    def list(self):
        return type("Page", (), {"data": [type("M", (), {"id": m})() for m in self.served]})()

    def create(self, **kwargs):
        self.kwargs = kwargs
        item = self.script[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


def _local_reader(script=(), served=("qwen3-vl-8b",), **kw):
    from chessqueries.models.llm import Effort

    kw.setdefault("effort", Effort.NONE)
    return LocalBoardReader(client=_LocalStubClient(script, served), **kw)


def _chat_response(placements=(START,), finish_reason="stop",
                   in_tok=420, out_tok=30):
    """A real `ChatCompletion`, one choice per placement (``n=k`` returns k choices).

    ``placements`` entries may be None to model an empty message.
    """
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate({
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0,
        "model": "qwen3-vl-8b",
        "choices": [
            {"index": i, "finish_reason": finish_reason,
             "message": {"role": "assistant",
                         "content": None if pl is None else _payload(pl)}}
            for i, pl in enumerate(placements)
        ],
        "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok,
                  "total_tokens": in_tok + out_tok},
    })
