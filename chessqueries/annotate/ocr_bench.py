"""Benchmark harness for the OCR the producer relies on: reading game **clocks** off
the broadcast overlay and player **names** off the nameplate (`identify.OcrReader`).

Lets us compare candidate OCR engines on real crops to decide which to wire in: an
`OcrEngine` ABC + registry (adding an engine is one subclass), the crop / ground-truth
value objects the extract, annotate, and benchmark scripts share, and the scoring.

Engines emit ``(x_left, text)`` detections. Clocks: `read_clocks` sorts them
left-to-right and parses each with `identify.parse_clock_text`, so the first value is
White's and the second Black's — `OcrReader.clocks_in`'s contract. Names: `read_text`
joins them into the blob `match_games_by_name` keys a surname on.

Crops, ground truth, and results are large/throwaway and live under `data/`
(local-only); only this core and its thin entry-point scripts are committed.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from chessqueries.annotate.identify import parse_clock_text, surname
from chessqueries.annotate.pipeline import ShotFingerprints


def fmt_clock(seconds: int | None) -> str:
    """Seconds -> ``"M:SS"`` (or ``"H:MM:SS"`` past an hour); ``""`` for None."""
    if seconds is None:
        return ""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


@dataclass(frozen=True)
class ClockCrop:
    """One extracted clock image and where it came from.

    ``rect`` is the source ``[x, y, w, h]`` in full-frame pixels; ``image_path`` is
    the PNG path relative to the benchmark root. ``baseline`` is the optional EasyOCR
    suggestion — the raw clock strings it read, left-to-right — saved at extract time
    to speed annotation.
    """

    crop_id: str
    video_id: str
    frame_index: int
    template_id: str
    rect: list[int]
    image_path: str
    baseline: list[str] | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError(f"frame_index must be non-negative, got {self.frame_index}")
        if len(self.rect) != 4:
            raise ValueError(f"rect must be [x, y, w, h], got {self.rect}")

    def to_dict(self) -> dict:
        return {
            "crop_id": self.crop_id,
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            "template_id": self.template_id,
            "rect": list(self.rect),
            "image_path": self.image_path,
            "baseline": list(self.baseline) if self.baseline is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ClockCrop":
        return cls(
            crop_id=str(d["crop_id"]),
            video_id=str(d["video_id"]),
            frame_index=int(d["frame_index"]),
            template_id=str(d["template_id"]),
            rect=[int(v) for v in d["rect"]],
            image_path=str(d["image_path"]),
            baseline=list(d["baseline"]) if d.get("baseline") is not None else None,
        )


def save_crops(crops: list[ClockCrop], path: Path) -> None:
    """Write the crop manifest as JSONL (one crop per line)."""
    Path(path).write_text("\n".join(json.dumps(c.to_dict()) for c in crops) + "\n")


def load_crops(path: Path) -> list[ClockCrop]:
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    return [ClockCrop.from_dict(json.loads(ln)) for ln in lines]


@dataclass(frozen=True)
class ClockLabel:
    """Ground truth for one crop: the (white, black) clock strings *exactly as shown*
    (``"0:19.0"``, ``"0:37"``) — annotation stays lossless. Seconds are derived on
    demand via `parse_clock_text`, so normalization is a scoring choice, not baked in.

    A clock not visible in the crop is the empty string. ``unreadable`` marks a crop
    no engine could fairly be asked to read (blurred, mid-transition) — excluded from
    scoring rather than counted as a failure for every engine.
    """

    white_text: str
    black_text: str
    unreadable: bool = False

    @property
    def white_s(self) -> int | None:
        return parse_clock_text(self.white_text)

    @property
    def black_s(self) -> int | None:
        return parse_clock_text(self.black_text)

    def to_dict(self) -> dict:
        return {
            "white_text": self.white_text,
            "black_text": self.black_text,
            "unreadable": self.unreadable,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ClockLabel":
        return cls(
            white_text=str(d.get("white_text", "")),
            black_text=str(d.get("black_text", "")),
            unreadable=bool(d.get("unreadable", False)),
        )


def save_labels(labels: dict[str, ClockLabel], path: Path) -> None:
    """Persist ground truth as ``{crop_id: label}`` JSON."""
    Path(path).write_text(json.dumps({k: v.to_dict() for k, v in labels.items()}, indent=2) + "\n")


def load_labels(path: Path) -> dict[str, ClockLabel]:
    if not Path(path).exists():
        return {}
    raw = json.loads(Path(path).read_text())
    return {k: ClockLabel.from_dict(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# OCR engines: ABC + auto-registry. A new engine is one `@register_engine`d
# subclass implementing `detect`.
# ---------------------------------------------------------------------------

ENGINES: dict[str, type["OcrEngine"]] = {}


def register_engine(name: str):
    """Class decorator registering an `OcrEngine` under ``name``."""

    def deco(cls: type["OcrEngine"]) -> type["OcrEngine"]:
        if name in ENGINES:
            raise ValueError(f"OCR engine {name!r} already registered")
        cls.name = name
        ENGINES[name] = cls
        return cls

    return deco


class OcrEngine(ABC):
    """A clock OCR backend. Subclasses load their model lazily in ``__init__`` and
    implement ``detect``; ``read_clocks`` is shared so every engine parses identically."""

    name: str = "?"

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> list[tuple[float, str]]:
        """``(x_left, text)`` for each text box found in the (BGR) crop."""

    def read_clock_texts(self, image_bgr: np.ndarray) -> list[str]:
        """Raw clock-like strings, ordered left-to-right -> [White, Black].

        Keeps the engine's verbatim text (e.g. ``"0:19.0"``) so the benchmark can
        score raw fidelity; filters to tokens `parse_clock_text` recognizes as clocks.
        """
        dets = sorted(self.detect(image_bgr), key=lambda d: d[0])
        return [t for _x, t in dets if parse_clock_text(t) is not None]

    def read_clocks(self, image_bgr: np.ndarray) -> list[int]:
        """Clock values in seconds, ordered left-to-right -> [White, Black]."""
        return [
            s for t in self.read_clock_texts(image_bgr) if (s := parse_clock_text(t)) is not None
        ]

    def read_text(self, image_bgr: np.ndarray) -> str:
        """All detected text, left-to-right, space-joined — the blob a nameplate yields
        (mirrors how `align._read_names` feeds `match_games_by_name`)."""
        dets = sorted(self.detect(image_bgr), key=lambda d: d[0])
        return " ".join(t for _x, t in dets)


@register_engine("easyocr")
class EasyOcrEngine(OcrEngine):
    """EasyOCR — the baseline engine (loses the clock benchmark to PaddleOCR)."""

    def __init__(self, gpu: bool | None = None) -> None:
        import easyocr
        import torch

        use_gpu = torch.cuda.is_available() if gpu is None else gpu
        self._reader = easyocr.Reader(["en"], gpu=use_gpu)

    def detect(self, image_bgr: np.ndarray) -> list[tuple[float, str]]:
        return [
            (float(min(pt[0] for pt in bbox)), text)
            for bbox, text, _conf in self._reader.readtext(image_bgr)
        ]


@register_engine("tesseract")
class TesseractEngine(OcrEngine):
    """Tesseract via pytesseract. CPU-only; upscales + thresholds the crop first,
    since the engine is weak on small low-contrast overlay text."""

    def __init__(self, psm: int = 6) -> None:
        import pytesseract  # noqa: F401  (fail early if missing)

        self._pt = pytesseract
        self._config = f"--psm {psm}"

    def detect(self, image_bgr: np.ndarray) -> list[tuple[float, str]]:
        import cv2

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        _t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        data = self._pt.image_to_data(binary, config=self._config, output_type=self._pt.Output.DICT)
        out: list[tuple[float, str]] = []
        for text, left in zip(data["text"], data["left"]):
            if text.strip():
                out.append((float(left), text))
        return out


@register_engine("paddleocr")
class PaddleOcrEngine(OcrEngine):
    """PaddleOCR (PP-OCR) — the benchmark winner and the `OcrReader` backend."""

    def __init__(self, lang: str = "en") -> None:
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            lang=lang,
            ocr_version="PP-OCRv4",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def detect(self, image_bgr: np.ndarray) -> list[tuple[float, str]]:
        result = self._ocr.predict(image_bgr)
        out: list[tuple[float, str]] = []
        for page in result or []:
            try:
                boxes, texts = page["rec_polys"], page["rec_texts"]
            except KeyError:
                # Serialized PaddleX results wrap the pipeline fields in `res`.
                payload = page["res"]
                boxes, texts = payload["rec_polys"], payload["rec_texts"]
            for box, text in zip(boxes, texts):
                x_left = float(min(pt[0] for pt in box))
                out.append((x_left, text))
        return out


def available_engines() -> dict[str, type[OcrEngine]]:
    """Registered engines whose backing library (and binary, for Tesseract) is present."""
    import importlib.util
    import shutil

    deps = {"easyocr": "easyocr", "tesseract": "pytesseract", "paddleocr": "paddleocr"}

    def ready(name: str) -> bool:
        if importlib.util.find_spec(deps.get(name, name)) is None:
            return False
        if name == "tesseract" and shutil.which("tesseract") is None:
            return False  # pytesseract imports but the engine binary is missing
        return True

    return {n: cls for n, cls in ENGINES.items() if ready(n)}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Match:
    """One prediction scored at two levels: raw-string fidelity (``*_exact``) and the
    pipeline-relevant seconds equivalence (``*_secs``, what `parse_clock_text` yields)."""

    white_exact: bool
    black_exact: bool
    white_secs: bool
    black_secs: bool

    @property
    def pair_exact(self) -> bool:
        return self.white_exact and self.black_exact

    @property
    def pair_secs(self) -> bool:
        return self.white_secs and self.black_secs


@dataclass(frozen=True)
class CropResult:
    crop_id: str
    predicted: list[str]  # raw clock texts, left-to-right
    match: Match
    latency_ms: float


@dataclass(frozen=True)
class EngineScore:
    """An engine's accuracy over the labeled crops, at both scoring levels."""

    name: str
    n: int  # scored crops (readable ground truth only)
    white_exact: int
    black_exact: int
    pair_exact: int
    white_secs: int
    black_secs: int
    pair_secs: int
    mean_latency_ms: float
    results: list[CropResult]

    def _acc(self, count: int) -> float:
        return count / self.n if self.n else 0.0

    @property
    def white_secs_acc(self) -> float:
        return self._acc(self.white_secs)

    @property
    def black_secs_acc(self) -> float:
        return self._acc(self.black_secs)

    @property
    def pair_secs_acc(self) -> float:
        return self._acc(self.pair_secs)

    @property
    def pair_exact_acc(self) -> float:
        return self._acc(self.pair_exact)


def _norm(text: str) -> str:
    return text.strip().replace(" ", "")


def score_prediction(predicted: list[str], label: ClockLabel) -> Match:
    """Score a predicted (White, Black) text pair against the labeled clocks.

    Position is significant: ``predicted[0]`` is White, ``predicted[1]`` Black (the
    left-to-right overlay order the producer relies on). Missing predictions count as
    empty. ``*_exact`` is whitespace-insensitive raw-string equality; ``*_secs`` is
    equality after `parse_clock_text` (tolerates a ``.``/``:`` swap and tenths)."""
    pw = predicted[0] if len(predicted) >= 1 else ""
    pb = predicted[1] if len(predicted) >= 2 else ""
    return Match(
        white_exact=_norm(pw) == _norm(label.white_text),
        black_exact=_norm(pb) == _norm(label.black_text),
        white_secs=parse_clock_text(pw) == label.white_s,
        black_secs=parse_clock_text(pb) == label.black_s,
    )


# ---------------------------------------------------------------------------
# Names track — one nameplate per crop. The producer only needs the surname to be
# recoverable (`match_games_by_name` checks ``surname in blob``), so that, not a
# verbatim read, is the headline metric.
# ---------------------------------------------------------------------------


class NameSide(str, Enum):
    WHITE = "white_name"
    BLACK = "black_name"


@dataclass(frozen=True)
class NameCrop:
    """One nameplate crop. ``baseline`` is EasyOCR's read text (annotation aid)."""

    crop_id: str
    video_id: str
    frame_index: int
    template_id: str
    side: NameSide
    rect: list[int]
    image_path: str
    baseline: str | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError(f"frame_index must be non-negative, got {self.frame_index}")
        if len(self.rect) != 4:
            raise ValueError(f"rect must be [x, y, w, h], got {self.rect}")

    def to_dict(self) -> dict:
        return {
            "crop_id": self.crop_id,
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            "template_id": self.template_id,
            "side": self.side.value,
            "rect": list(self.rect),
            "image_path": self.image_path,
            "baseline": self.baseline,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NameCrop":
        return cls(
            crop_id=str(d["crop_id"]),
            video_id=str(d["video_id"]),
            frame_index=int(d["frame_index"]),
            template_id=str(d["template_id"]),
            side=NameSide(d["side"]),
            rect=[int(v) for v in d["rect"]],
            image_path=str(d["image_path"]),
            baseline=d.get("baseline"),
        )


@dataclass(frozen=True)
class NameLabel:
    """Ground truth: the nameplate text *exactly as shown*. The matchable surname is
    derived with the same `identify.surname` heuristic the producer uses."""

    text: str
    unreadable: bool = False

    @property
    def surname(self) -> str:
        return surname(self.text) if self.text else ""

    def to_dict(self) -> dict:
        return {"text": self.text, "unreadable": self.unreadable}

    @classmethod
    def from_dict(cls, d: dict) -> "NameLabel":
        return cls(text=str(d.get("text", "")), unreadable=bool(d.get("unreadable", False)))


def save_names(crops: list[NameCrop], path: Path) -> None:
    Path(path).write_text("\n".join(json.dumps(c.to_dict()) for c in crops) + "\n")


def load_names(path: Path) -> list[NameCrop]:
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    return [NameCrop.from_dict(json.loads(ln)) for ln in lines]


def save_name_labels(labels: dict[str, NameLabel], path: Path) -> None:
    Path(path).write_text(json.dumps({k: v.to_dict() for k, v in labels.items()}, indent=2) + "\n")


def load_name_labels(path: Path) -> dict[str, NameLabel]:
    if not Path(path).exists():
        return {}
    raw = json.loads(Path(path).read_text())
    return {k: NameLabel.from_dict(v) for k, v in raw.items()}


@dataclass(frozen=True)
class NameMatch:
    surname_ok: bool  # the matchable surname appears in the read text (the pipeline metric)
    exact: bool  # nameplate read verbatim (letters only, case-insensitive)


def _alnum_upper(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def score_name(predicted_text: str, label: NameLabel) -> NameMatch:
    """Score a read nameplate. ``surname_ok`` mirrors `match_games_by_name`: the
    ground-truth surname, lowercased, occurs in the (lowercased) read text."""
    sur = label.surname
    return NameMatch(
        surname_ok=bool(sur) and sur in predicted_text.lower(),
        exact=_alnum_upper(predicted_text) == _alnum_upper(label.text),
    )


# ---------------------------------------------------------------------------
# Sampling helpers shared by the extract scripts (clocks + names).
# ---------------------------------------------------------------------------


def cached_videos(data_dir: Path) -> list[str]:
    """Video ids in ``data_dir`` with all three artifacts (mp4 + shots + descriptors)."""
    from chessqueries.annotate.video import DEFAULT_FORMAT_ID
    from chessqueries.annotate.workflow import descriptors_path, shots_path

    out = []
    for mp4 in sorted(Path(data_dir).glob(f"*.{DEFAULT_FORMAT_ID}.mp4")):
        vid = mp4.name[: -len(f".{DEFAULT_FORMAT_ID}.mp4")]
        if (
            shots_path(data_dir, vid, DEFAULT_FORMAT_ID).exists()
            and descriptors_path(data_dir, vid, DEFAULT_FORMAT_ID).exists()
        ):
            out.append(vid)
    return out


def load_cached_shots(data_dir: Path, vid: str) -> ShotFingerprints:
    """Load a video's cached shots + keyframe descriptors from ``data_dir``."""
    from chessqueries.annotate.templates import load_shots
    from chessqueries.annotate.video import DEFAULT_FORMAT_ID
    from chessqueries.annotate.workflow import descriptors_path, shots_path

    shots = load_shots(shots_path(data_dir, vid, DEFAULT_FORMAT_ID))
    descriptors = np.load(descriptors_path(data_dir, vid, DEFAULT_FORMAT_ID))
    return ShotFingerprints(shots=shots, descriptors=descriptors)


def spread_order(items: list) -> list:
    """Reorder so any prefix stays evenly spread: endpoints first, then recursive
    midpoints. Round-robining group prefixes then samples each group across time."""
    n = len(items)
    if n <= 2:
        return list(items)
    from collections import deque

    order = [0, n - 1]
    seen = {0, n - 1}
    q = deque([(0, n - 1)])
    while q:
        lo, hi = q.popleft()
        mid = (lo + hi) // 2
        if mid not in seen:
            seen.add(mid)
            order.append(mid)
        if mid - lo > 1:
            q.append((lo, mid))
        if hi - mid > 1:
            q.append((mid, hi))
    return [items[i] for i in order]


def diversify(cands: list, k: int, group_key, time_key) -> list:
    """Pick up to ``k`` items, round-robin across ``group_key`` groups (variety of
    composition); within a group, prefixes are spread by ``time_key``. Uneven group
    sizes still fill to ``k`` (large groups keep contributing once small ones dry up)."""
    if k <= 0:
        return []
    groups: dict = {}
    for c in cands:
        groups.setdefault(group_key(c), []).append(c)
    ordered = {key: spread_order(sorted(items, key=time_key)) for key, items in groups.items()}
    keys = sorted(ordered)
    picked: list = []
    pos = {key: 0 for key in keys}
    progressed = True
    while len(picked) < k and progressed:
        progressed = False
        for key in keys:
            if pos[key] < len(ordered[key]):
                picked.append(ordered[key][pos[key]])
                pos[key] += 1
                progressed = True
                if len(picked) >= k:
                    break
    return picked


@dataclass(frozen=True)
class NameScore:
    """One engine's aggregate on the name track (mirror of `EngineScore`)."""

    name: str
    n: int
    surname_ok: int
    exact: int
    mean_latency_ms: float
    per_crop: dict[str, dict]

    @property
    def surname_acc(self) -> float:
        return self.surname_ok / self.n if self.n else 0.0

    @property
    def exact_acc(self) -> float:
        return self.exact / self.n if self.n else 0.0


# ---------------------------------------------------------------------------
# Helpers shared by the clock + name benchmark scripts (extract / annotate / run).
# ---------------------------------------------------------------------------


def classified_shots(data_dir: Path, videos: list[str], registry, min_similarity: float):
    """Yield ``(video_id, shot, layout)`` for every cached shot whose descriptor
    matches a processable template at/above ``min_similarity``."""
    for vid in videos:
        for shot, desc in load_cached_shots(data_dir, vid).pairs():
            match = registry.classify(desc)
            if match.similarity < min_similarity:
                continue
            layout = registry.layouts[match.template_id]
            if layout.processable:
                yield vid, shot, layout


class ReaderPool:
    """Lazy per-video `FrameReader` cache for the extract scripts; close() releases all."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self._readers: dict = {}

    def frame(self, vid: str, frame_index: int):
        from chessqueries.annotate.video import DEFAULT_FORMAT_ID, FrameReader, probe

        if vid not in self._readers:
            mp4 = self.data_dir / f"{vid}.{DEFAULT_FORMAT_ID}.mp4"
            self._readers[vid] = FrameReader(probe(mp4, vid, DEFAULT_FORMAT_ID))
        return self._readers[vid].frame_at_index(frame_index)

    def close(self) -> None:
        for r in self._readers.values():
            r.close()

    def __enter__(self) -> "ReaderPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def selected_engines(requested: list[str] | None) -> list[str]:
    """Engine names to benchmark: the requested (or all installed) ones, announcing
    each skip (unknown name / backing library not installed)."""
    installed = available_engines()
    out = []
    for name in requested or list(installed):
        if name not in ENGINES:
            print(f"skip {name!r}: unknown (known: {', '.join(ENGINES)})")
        elif name not in installed:
            print(f"skip {name!r}: backing library/binary not installed")
        else:
            out.append(name)
    return out


def readable(crops, labels) -> list[tuple]:
    """``(crop, label)`` pairs with readable ground truth — the scoreable set."""
    return [
        (c, lab) for c in crops if (lab := labels.get(c.crop_id)) is not None and not lab.unreadable
    ]


def write_results(
    results_dir: Path, summary: list[dict], ground_truth: dict, predictions: dict
) -> Path:
    """The per-crop results JSON both benchmark tracks write for error inspection."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "results.json"
    path.write_text(
        json.dumps(
            {"summary": summary, "ground_truth": ground_truth, "predictions": predictions},
            indent=2,
        )
    )
    return path


# ---------------------------------------------------------------------------
# The shared hand-labeling app (clock + name tracks differ only in their fields).
# ---------------------------------------------------------------------------


class LabelFilter(str, Enum):
    """Which crops the labeling app steps through."""

    REVIEW = "To review (labeled)"
    TODO = "To do (unlabeled)"
    ALL = "All"


def filtered_indices(crop_ids: list[str], labeled: set[str], mode: LabelFilter) -> list[int]:
    """Crop indices the filter lets you navigate, in manifest order."""
    if mode == LabelFilter.REVIEW:
        return [i for i, cid in enumerate(crop_ids) if cid in labeled]
    if mode == LabelFilter.TODO:
        return [i for i, cid in enumerate(crop_ids) if cid not in labeled]
    return list(range(len(crop_ids)))


def neighbor_index(cur: int, act: list[int], step: int) -> int:
    """Move ``step`` within the filtered set; if ``cur`` just left the set (saved in
    TODO mode), land on the nearest remaining crop in the direction of travel."""
    if not act:
        return cur
    if cur in act:
        return act[max(0, min(len(act) - 1, act.index(cur) + step))]
    after = [x for x in act if x > cur]
    before = [x for x in act if x < cur]
    if step >= 0:
        return after[0] if after else before[-1]
    return before[-1] if before else after[0]


@dataclass(frozen=True)
class LabelTrack:
    """Everything track-specific about a labeling app: manifest/label IO, the text
    fields, and how a crop prefills them. The navigation shell is shared."""

    title: str
    image_label: str
    image_height: int
    instructions: str
    field_labels: tuple[str, ...]  # one Textbox per entry, in order
    load_crops: object  # Path -> list[crop]
    load_labels: object  # Path -> dict[crop_id, label]
    save_labels: object  # (labels, Path) -> None
    label_fields: object  # label -> tuple[str, ...] (existing label -> field values)
    baseline_fields: object  # crop -> tuple[str, ...] (fresh crop -> prefill)
    make_label: object  # (fields: tuple[str, ...], unreadable: bool) -> label
    caption: object  # (crop, labels) -> str extra status-line detail


def build_label_app(out_dir: Path, track: LabelTrack):
    """The shared ground-truth labeling app: filterable prev/save-next navigation
    over the manifest, persisting labels to ``ground_truth.json`` as you go."""
    import cv2
    import gradio as gr

    out_dir = Path(out_dir)
    crops = track.load_crops(out_dir / "manifest.jsonl")
    labels_path = out_dir / "ground_truth.json"
    if not crops:
        raise SystemExit(f"no crops in {out_dir / 'manifest.jsonl'} — run the extract script first")
    crop_ids = [c.crop_id for c in crops]

    def active(mode: str, labels: dict) -> list[int]:
        return filtered_indices(crop_ids, set(labels), LabelFilter(mode))

    def load_image(i: int):
        bgr = cv2.imread(str(out_dir / crops[i].image_path))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def fields_for(i: int, labels: dict) -> tuple:
        c = crops[i]
        if c.crop_id in labels:
            lab = labels[c.crop_id]
            return (*track.label_fields(lab), lab.unreadable)
        return (*track.baseline_fields(c), False)

    def status(i: int, mode: str, labels: dict) -> str:
        act = active(mode, labels)
        where = f"{act.index(i) + 1} / {len(act)}" if i in act else "—"
        done = sum(1 for cid in crop_ids if cid in labels)
        return (
            f"### {mode}: {where}  ·  {done} / {len(crops)} labeled overall\n"
            f"{track.caption(crops[i], labels)}"
        )

    init_labels = track.load_labels(labels_path)
    default_mode = (
        LabelFilter.REVIEW if any(cid in init_labels for cid in crop_ids) else LabelFilter.TODO
    ).value
    init_cur = (active(default_mode, init_labels) or [0])[0]

    with gr.Blocks(title=track.title) as app:
        labels_state = gr.State(init_labels)
        cur_state = gr.State(init_cur)

        mode_in = gr.Radio([f.value for f in LabelFilter], value=default_mode, label="Filter")
        info = gr.Markdown(status(init_cur, default_mode, init_labels))
        image = gr.Image(load_image(init_cur), label=track.image_label, height=track.image_height)
        gr.Markdown(track.instructions)
        init_fields = fields_for(init_cur, init_labels)
        with gr.Row():
            field_ins = [gr.Textbox(label=lbl, scale=2) for lbl in track.field_labels]
            unreadable_in = gr.Checkbox(label="unreadable", scale=1)
        for widget, val in zip((*field_ins, unreadable_in), init_fields):
            widget.value = val
        with gr.Row():
            prev_btn = gr.Button("◀ Prev")
            save_btn = gr.Button("Save & Next ▶", variant="primary")

        outputs = [cur_state, labels_state, image, *field_ins, unreadable_in, info]

        def show(cur: int, mode: str, labels: dict):
            return cur, labels, load_image(cur), *fields_for(cur, labels), status(cur, mode, labels)

        def go(cur, labels, mode, *rest):
            *fields, unreadable, step, do_save = rest
            if do_save:
                labels = dict(labels)
                labels[crops[cur].crop_id] = track.make_label(
                    tuple(f.strip() for f in fields), bool(unreadable)
                )
                track.save_labels(labels, labels_path)
            return show(neighbor_index(cur, active(mode, labels), step), mode, labels)

        nav_inputs = [cur_state, labels_state, mode_in, *field_ins, unreadable_in]
        save_btn.click(lambda *a: go(*a, +1, True), nav_inputs, outputs)
        prev_btn.click(lambda *a: go(*a, -1, False), nav_inputs, outputs)
        # Switching filter jumps to the first crop in the newly selected set.
        mode_in.change(
            lambda mode, labels, cur: show((active(mode, labels) or [cur])[0], mode, labels),
            [mode_in, labels_state, cur_state],
            outputs,
        )
    return app
