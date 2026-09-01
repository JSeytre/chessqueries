import sys
from types import SimpleNamespace

import numpy as np
import pytest

from chessqueries.annotate import video
from chessqueries.annotate.ocr_bench import PaddleOcrEngine


def test_runtime_auto_detection_prefers_deno(monkeypatch):
    installed = {"deno": "/usr/bin/deno", "node": "/usr/bin/node"}
    monkeypatch.setattr(video.shutil, "which", installed.get)

    assert video.resolve_js_runtime() == "deno"


def test_missing_runtime_has_actionable_error(monkeypatch):
    monkeypatch.setattr(video.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="Install Deno"):
        video.resolve_js_runtime()


def test_cookie_sources_are_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError, match="either"):
        video.YoutubeAccess(
            cookie_file=tmp_path / "cookies.txt",
            cookies_from_browser="firefox",
        )


def test_download_preserves_underlying_ytdlp_error(tmp_path, monkeypatch):
    class FailingYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def extract_info(self, url, download):
            raise RuntimeError("Sign in to confirm you're not a bot")

    monkeypatch.setitem(
        sys.modules,
        "yt_dlp",
        SimpleNamespace(YoutubeDL=FailingYoutubeDL),
    )
    monkeypatch.setattr(video, "youtube_options", lambda access: {})

    with pytest.raises(RuntimeError) as caught:
        video.download("example-video", tmp_path, format_id="137")

    message = str(caught.value)
    assert "pinned format 137" in message
    assert "Sign in to confirm you're not a bot" in message
    assert "--cookies-from-browser" in message
    assert str(caught.value.__cause__) == "Sign in to confirm you're not a bot"


def test_paddle_v3_result_is_converted_to_ordered_detections():
    class FakePaddle:
        def predict(self, image):
            assert image.shape == (4, 6, 3)
            return [
                {
                    "rec_polys": np.array(
                        [
                            [[12, 0], [20, 0], [20, 4], [12, 4]],
                            [[2, 0], [8, 0], [8, 4], [2, 4]],
                        ]
                    ),
                    "rec_texts": ["black", "white"],
                }
            ]

    engine = PaddleOcrEngine.__new__(PaddleOcrEngine)
    engine._ocr = FakePaddle()

    assert engine.detect(np.zeros((4, 6, 3), dtype=np.uint8)) == [
        (12.0, "black"),
        (2.0, "white"),
    ]
