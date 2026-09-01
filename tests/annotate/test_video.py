"""FrameReader threading safety: the review UI shares one reader across Gradio's
worker threads, and the underlying FFmpeg decoder aborts if entered concurrently.
A fake capture stands in for the codec so we can assert the seek+read is serialized
(no overlap, no interleaved seek returning the wrong frame) without a real video."""

import threading
import time

import cv2

from chessqueries.annotate.video import FrameReader, VideoFile


class _FakeCap:
    """Emulates cv2.VideoCapture's stateful seek/read: read() returns whatever index
    the last set() moved to. Flags any overlapping read() (the real async_lock abort)
    and any seek/read that got interleaved by another thread."""

    def __init__(self):
        self._pos = 0
        self._active = 0
        self.max_active = 0
        self._guard = threading.Lock()  # protects the test's own counters only

    def isOpened(self):
        return True

    def set(self, prop, val):
        self._pos = int(val)
        return True

    def read(self):
        with self._guard:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        time.sleep(0.001)  # widen the window an unsynchronized reader would race in
        pos = self._pos
        with self._guard:
            self._active -= 1
        return True, pos  # the "frame" is just the index that was sought

    def release(self):
        pass


def _video(tmp_path) -> VideoFile:
    path = tmp_path / "v.137.mp4"
    path.write_bytes(b"x")  # VideoFile only checks the path exists
    return VideoFile(path=path, video_id="v", format_id="137", width=16, height=16,
                     fps=30.0, frame_count=1000)


def test_frame_reader_serializes_concurrent_reads(tmp_path, monkeypatch):
    fake = _FakeCap()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_a, **_k: fake)
    reader = FrameReader(_video(tmp_path))

    errors: list[str] = []

    def hammer(base: int):
        for i in range(base, base + 40):
            frame = reader.frame_at_index(i)
            if frame != i:  # a concurrent seek slipped between our set() and read()
                errors.append(f"wanted {i}, got {frame}")

    threads = [threading.Thread(target=hammer, args=(b,)) for b in (0, 100, 200, 300, 400, 500)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:5]
    assert fake.max_active == 1  # reads never overlapped -> the decoder is never re-entered
