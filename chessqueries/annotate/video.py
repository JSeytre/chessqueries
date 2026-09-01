"""Download a broadcast video at a pinned format and read frames by index.

Frames are addressed by integer index (``round(t * fps)``), not wall-clock seek,
so a released annotation has a stable structural identity and crop. Decoder builds
need not produce bit-identical pixels. OpenCV decodes the single-stream mp4 directly,
so no separate ffmpeg binary is required.
"""

from __future__ import annotations

import json
import shutil
import threading
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

import cv2

# 1080p H.264 (avc1) video-only stream: universally decodable, enough detail for
# small pieces in wide shots, and a single stream (no ffmpeg merge needed). Pinned
# so reconstructed crops refer to the same source stream downstream.
DEFAULT_FORMAT_ID = "137"
YOUTUBE_WATCH_URL = "https://www.youtube.com/watch?v={video_id}"

JS_RUNTIME_EXECUTABLES = {
    "deno": "deno",
    "node": "node",
    "bun": "bun",
    "quickjs": "qjs",
}


@dataclass(frozen=True)
class YoutubeAccess:
    """Local access configuration passed through to yt-dlp.

    Cookies are optional and never copied into provenance.  A JavaScript runtime
    is selected automatically (Deno first, then another supported installed
    runtime), or may be set explicitly with yt-dlp's ``RUNTIME[:PATH]`` syntax.
    """

    cookie_file: Path | None = None
    cookies_from_browser: str | None = None
    js_runtime: str | None = None

    def __post_init__(self) -> None:
        if self.cookie_file is not None and self.cookies_from_browser is not None:
            raise ValueError("use either a cookie file or browser cookies, not both")


def resolve_js_runtime(requested: str | None = None) -> str:
    """Return an installed yt-dlp runtime specification or fail actionably."""
    if requested:
        name, separator, explicit_path = requested.partition(":")
        if name not in JS_RUNTIME_EXECUTABLES:
            choices = ", ".join(JS_RUNTIME_EXECUTABLES)
            raise RuntimeError(f"unsupported JavaScript runtime {name!r}; choose {choices}")
        executable = explicit_path if separator else shutil.which(JS_RUNTIME_EXECUTABLES[name])
        if executable is None or not Path(executable).is_file():
            raise RuntimeError(
                f"JavaScript runtime {requested!r} is not installed. Install Deno "
                "(recommended by yt-dlp), or pass --js-runtime RUNTIME[:PATH]."
            )
        return requested

    for name, executable_name in JS_RUNTIME_EXECUTABLES.items():
        if shutil.which(executable_name):
            return name
    raise RuntimeError(
        "No supported JavaScript runtime was found. Install Deno (recommended by "
        "yt-dlp), or pass --js-runtime RUNTIME[:PATH]."
    )


def youtube_options(access: YoutubeAccess | None = None) -> dict:
    """Translate reviewer-facing access flags into yt-dlp's Python options."""
    import yt_dlp

    access = access or YoutubeAccess()
    try:
        metadata.version("yt-dlp-ejs")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "yt-dlp's EJS challenge solver is missing; install the locked `slcc` "
            "dependency group before reconstructing."
        ) from exc

    runtime = resolve_js_runtime(access.js_runtime)
    args = ["--js-runtimes", runtime]
    if access.cookies_from_browser:
        args.extend(["--cookies-from-browser", access.cookies_from_browser])
    parsed = yt_dlp.parse_options(args).ydl_opts
    options = {"js_runtimes": parsed["js_runtimes"]}
    if access.cookies_from_browser:
        options["cookiesfrombrowser"] = parsed["cookiesfrombrowser"]
    if access.cookie_file:
        if not access.cookie_file.is_file():
            raise FileNotFoundError(access.cookie_file)
        options["cookiefile"] = str(access.cookie_file)
    return options


def probe_source(
    video_id: str,
    *,
    format_id: str = DEFAULT_FORMAT_ID,
    access: YoutubeAccess | None = None,
) -> dict:
    """Check that one pinned source format is currently resolvable without downloading."""
    import yt_dlp

    options = {
        "format": format_id,
        "quiet": True,
        "noprogress": True,
        "skip_download": True,
        **youtube_options(access),
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(YOUTUBE_WATCH_URL.format(video_id=video_id), download=False)
        return {
            "video_id": video_id,
            "format_id": format_id,
            "available": True,
            "resolved_format_id": str(info.get("format_id", "")),
        }
    except Exception as exc:
        return {
            "video_id": video_id,
            "format_id": format_id,
            "available": False,
            "error": str(exc),
        }


@dataclass(frozen=True)
class VideoFile:
    """A downloaded video plus the metadata needed to address frames reproducibly."""

    path: Path
    video_id: str
    format_id: str
    width: int
    height: int
    fps: float
    frame_count: int

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps

    def provenance(self) -> dict:
        d = asdict(self)
        d["path"] = self.path.name
        d["duration_s"] = self.duration_s
        return d


def probe(path: Path, video_id: str, format_id: str) -> VideoFile:
    """Build a `VideoFile` from an already-downloaded file (read props via OpenCV)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    try:
        return VideoFile(
            path=Path(path),
            video_id=video_id,
            format_id=format_id,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def download(
    video_id: str,
    dest_dir: Path,
    *,
    format_id: str = DEFAULT_FORMAT_ID,
    sections: list[tuple[float, float]] | None = None,
    overwrite: bool = False,
    access: YoutubeAccess | None = None,
) -> VideoFile:
    """Download one video at ``format_id`` into ``dest_dir`` and write provenance.

    ``sections`` (start, end) seconds restricts the download for cheap tests; it
    needs ffmpeg, so leave it None for the real pinned-format pull. Re-downloads
    are skipped unless ``overwrite``.
    """
    import yt_dlp

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{video_id}.{format_id}"

    # Only the downloaded media counts as "present". Exclude provenance/cache sidecars
    # (`.json`, the `.npy` descriptor cache) and leftover `.part`/`.ytdl` partials —
    # otherwise OpenCV would try to open a cache file as the video and fail.
    non_video_suffixes = (".json", ".npy", ".part", ".ytdl")

    def complete_files() -> list[Path]:
        return [p for p in dest_dir.glob(f"{stem}.*") if p.suffix.lower() not in non_video_suffixes]

    existing = complete_files()
    if existing and not overwrite:
        video = probe(existing[0], video_id, format_id)
    else:
        base_opts = {
            "format": format_id,
            "outtmpl": str(dest_dir / f"{stem}.%(ext)s"),
            "quiet": True,
            "noprogress": True,
            "overwrites": overwrite,
            # Broadcast pulls are multi-GB; ride out transient network drops and resume
            # the partial file rather than failing the whole batch.
            "continuedl": True,
            "retries": 10,
            "fragment_retries": 10,
            "file_access_retries": 5,
            "socket_timeout": 30,
            **youtube_options(access),
        }
        if sections:
            from yt_dlp.utils import download_range_func

            base_opts["download_ranges"] = download_range_func(None, list(sections))

        url = YOUTUBE_WATCH_URL.format(video_id=video_id)
        try:
            with yt_dlp.YoutubeDL(base_opts) as ydl:
                ydl.extract_info(url, download=True)
        except Exception as exc:
            detail = str(exc).strip() or type(exc).__name__
            raise RuntimeError(
                f"yt-dlp could not download YouTube video {video_id} at pinned format "
                f"{format_id}. Underlying yt-dlp error: {detail}. Run the reconstruction "
                "command with --check-sources first. If YouTube requests sign-in, pass "
                "--cookies-from-browser BROWSER or --cookies FILE to both the source "
                "check and reconstruction commands."
            ) from exc

        downloaded = complete_files()
        if not downloaded:
            raise RuntimeError(f"download produced no file for {stem} in {dest_dir}")
        video = probe(downloaded[0], video_id, format_id)

    (dest_dir / f"{stem}.provenance.json").write_text(json.dumps(video.provenance(), indent=2))
    return video


class FrameReader:
    """Random-access frame reader; addresses frames by integer index for determinism.

    Use as a context manager. Returns BGR ``numpy`` arrays (OpenCV convention).

    The seek+read pair is guarded by a lock: ``cv2.VideoCapture`` wraps a single,
    non-reentrant FFmpeg decoder, so unsynchronized concurrent access aborts the
    process (uncatchable) or returns the wrong frame.
    """

    def __init__(self, video: VideoFile) -> None:
        self.video = video
        self._cap = cv2.VideoCapture(str(video.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV could not open {video.path}")
        self._lock = threading.Lock()

    def frame_at_index(self, index: int):
        if not 0 <= index < self.video.frame_count:
            raise IndexError(f"frame {index} out of range [0, {self.video.frame_count})")
        with self._lock:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError(f"failed to read frame {index} of {self.video.path}")
        return frame

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
