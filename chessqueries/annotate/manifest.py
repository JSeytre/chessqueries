"""The video manifest: which open broadcasts are in the dataset and how each maps
to its Lichess relay (one or more *tournaments*, optionally narrowed to specific
rounds). Small, committed provenance — distinct from the large, local-only layout
registry (`resources/slcc_layouts.json`).

Mapping a YouTube video to its relay is the one irreducibly manual step: a video
covers some rounds of some tournament, and there is no reliable automatic link. We
keep it coarse — tag the *tournament* (one id usually covers many videos) and let
`identify`/`align` pick the round per frame — with optional `round_ids` to narrow.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from chessqueries.annotate.relay import tournament_rounds

DEFAULT_VIDEOS_PATH = Path(__file__).parent / "resources" / "slcc_videos.json"

_VIDEO_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")
_YT_URL_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")


def video_id_from_url(url_or_id: str) -> str:
    """Extract the 11-char YouTube id from a watch URL, or accept a bare id."""
    s = url_or_id.strip()
    m = _YT_URL_RE.search(s)
    if m:
        return m.group(1)
    if _VIDEO_ID_RE.fullmatch(s):
        return s
    raise ValueError(f"could not parse a YouTube id from {url_or_id!r}")


@dataclass(frozen=True)
class VideoEntry:
    """One registered broadcast video and its relay mapping."""

    video_id: str
    title: str = ""
    date: str = ""  # YYYY-MM-DD if known
    tournaments: list[str] = field(default_factory=list)  # Lichess broadcast tour ids
    round_ids: list[str] = field(default_factory=list)  # optional: narrow to these rounds

    def __post_init__(self) -> None:
        if not _VIDEO_ID_RE.fullmatch(self.video_id):
            raise ValueError(f"video_id must be an 11-char YouTube id, got {self.video_id!r}")

    @property
    def has_relay_mapping(self) -> bool:
        """Whether `produce` can resolve relay rounds for this video."""
        return bool(self.tournaments or self.round_ids)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "date": self.date,
            "tournaments": list(self.tournaments),
            "round_ids": list(self.round_ids),
        }

    @classmethod
    def from_dict(cls, video_id: str, d: dict) -> "VideoEntry":
        # Back-compat: an older entry may carry a singular ``tournament`` string.
        tournaments = list(d.get("tournaments") or [])
        if not tournaments and d.get("tournament"):
            tournaments = [d["tournament"]]
        return cls(
            video_id=video_id,
            title=d.get("title", ""),
            date=d.get("date", ""),
            tournaments=tournaments,
            round_ids=list(d.get("round_ids") or []),
        )


@dataclass
class Manifest:
    """The set of registered videos, persisted as ``resources/slcc_videos.json``."""

    path: Path
    entries: dict[str, VideoEntry] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = DEFAULT_VIDEOS_PATH) -> "Manifest":
        path = Path(path)
        data = json.loads(path.read_text()) if path.exists() else {}
        return cls(path=path, entries={k: VideoEntry.from_dict(k, v) for k, v in data.items()})

    def save(self) -> None:
        ordered = dict(sorted(self.entries.items(), key=lambda kv: (kv[1].date, kv[0])))
        self.path.write_text(
            json.dumps({k: v.to_dict() for k, v in ordered.items()}, indent=2) + "\n"
        )

    def add(self, entry: VideoEntry) -> bool:
        """Register a video; returns False (no-op) if already present."""
        if entry.video_id in self.entries:
            return False
        self.entries[entry.video_id] = entry
        return True

    def __contains__(self, video_id: str) -> bool:
        return video_id in self.entries

    def __getitem__(self, video_id: str) -> VideoEntry:
        return self.entries[video_id]

    def __iter__(self):
        return iter(self.entries.values())

    def __len__(self) -> int:
        return len(self.entries)


def fetch_youtube_metadata(video_id: str) -> dict:
    """``{"title", "date"}`` for a video via yt-dlp (no download). Best-effort:
    returns empty strings if yt-dlp or the network is unavailable."""
    try:
        import yt_dlp

        from chessqueries.annotate.video import YOUTUBE_WATCH_URL

        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(YOUTUBE_WATCH_URL.format(video_id=video_id), download=False)
        upload = info.get("upload_date") or ""  # YYYYMMDD
        date = f"{upload[:4]}-{upload[4:6]}-{upload[6:]}" if len(upload) == 8 else ""
        return {"title": info.get("title", ""), "date": date}
    except Exception:
        return {"title": "", "date": ""}


def rounds_for_video(video_id: str, *, videos_path: Path = DEFAULT_VIDEOS_PATH) -> list[str]:
    """Relay round ids a YouTube video covers, from the video manifest.

    Resolution: explicit ``round_ids`` if given, else every round of the mapped
    ``tournaments`` (coarse — the matcher picks the right round per frame). Raises if
    the video is unregistered or has no relay mapping (then pass round ids explicitly).
    """
    entry = Manifest.load(videos_path).entries.get(video_id)
    if entry is None:
        raise KeyError(f"video {video_id!r} not in {videos_path}; pass round ids explicitly")
    if entry.round_ids:
        return list(entry.round_ids)
    if not entry.tournaments:
        raise KeyError(f"video {video_id!r} has no relay mapping (tournaments or round_ids)")
    return [r.id for t in entry.tournaments for r in tournament_rounds(t)]
