"""OCR benchmark core: crop/label value objects, manifest IO, engine registry, scoring."""

import numpy as np
import pytest

from chessqueries.annotate.ocr_bench import (
    ENGINES,
    ClockCrop,
    ClockLabel,
    NameCrop,
    NameLabel,
    NameSide,
    OcrEngine,
    fmt_clock,
    load_crops,
    load_labels,
    load_name_labels,
    load_names,
    save_crops,
    save_labels,
    save_name_labels,
    save_names,
    score_name,
    score_prediction,
)


def _crop(crop_id="v_10_digital", **kw):
    defaults = dict(
        crop_id=crop_id,
        video_id="v",
        frame_index=10,
        template_id="slcc_t2",
        rect=[694, 948, 532, 127],
        image_path="crops/v_10_digital.png",
    )
    defaults.update(kw)
    return ClockCrop(**defaults)


def test_fmt_clock():
    assert fmt_clock(None) == ""
    assert fmt_clock(37) == "0:37"
    assert fmt_clock(83) == "1:23"
    assert fmt_clock(3661) == "1:01:01"


def test_crop_validation():
    with pytest.raises(ValueError):
        _crop(frame_index=-1)
    with pytest.raises(ValueError):
        _crop(rect=[1, 2, 3])


def test_crop_manifest_roundtrip(tmp_path):
    crops = [_crop(baseline=["0:19", "0:37"]), _crop("v_20_digital", frame_index=20)]
    path = tmp_path / "manifest.jsonl"
    save_crops(crops, path)
    assert load_crops(path) == crops


def test_label_seconds_derived():
    """Ground truth keeps the raw string; seconds are derived, tolerating tenths."""
    lab = ClockLabel(white_text="0:19.0", black_text="0:37")
    assert (lab.white_s, lab.black_s) == (19, 37)
    assert ClockLabel(white_text="", black_text="").white_s is None


def test_labels_roundtrip(tmp_path):
    labels = {
        "a": ClockLabel(white_text="0:19.0", black_text="0:37"),
        "b": ClockLabel(white_text="", black_text="", unreadable=True),
    }
    path = tmp_path / "gt.json"
    save_labels(labels, path)
    assert load_labels(path) == labels


def test_load_labels_missing(tmp_path):
    assert load_labels(tmp_path / "absent.json") == {}


def test_engines_registered():
    assert {"easyocr", "tesseract", "paddleocr"} <= set(ENGINES)


def test_read_clock_texts_orders_left_to_right():
    """The shared read_clock_texts sorts detections by x and keeps raw clock strings."""

    class FakeEngine(OcrEngine):
        def detect(self, image_bgr):
            # Deliberately out of order; black (x=300) before white (x=10).
            return [(300.0, "0:37"), (10.0, "0:19.0"), (150.0, "garbage")]

    eng = FakeEngine()
    img = np.zeros((1, 1, 3), np.uint8)
    assert eng.read_clock_texts(img) == ["0:19.0", "0:37"]  # raw text preserved
    assert eng.read_clocks(img) == [19, 37]  # parsed to seconds


def test_score_prediction_exact_and_secs():
    label = ClockLabel(white_text="0:19.0", black_text="0:37")
    # Verbatim match: both levels pass.
    m = score_prediction(["0:19.0", "0:37"], label)
    assert (m.pair_exact, m.pair_secs) == (True, True)
    # Tenths dropped: not verbatim, but the same number of seconds.
    m = score_prediction(["0:19", "0:37"], label)
    assert (m.white_exact, m.white_secs, m.pair_secs) == (False, True, True)
    # Wrong value: fails both.
    m = score_prediction(["0:20", "0:37"], label)
    assert (m.white_secs, m.black_secs, m.pair_secs) == (False, True, False)
    # Black missing entirely.
    m = score_prediction(["0:19.0"], label)
    assert (m.white_secs, m.black_secs) == (True, False)


# --- names track ---


def _name_crop(crop_id="v_10_white_name", **kw):
    defaults = dict(
        crop_id=crop_id,
        video_id="v",
        frame_index=10,
        template_id="slcc_t2",
        side=NameSide.WHITE,
        rect=[418, 990, 268, 85],
        image_path="crops/v_10_white_name.png",
    )
    defaults.update(kw)
    return NameCrop(**defaults)


def test_name_crop_roundtrip(tmp_path):
    crops = [_name_crop(baseline="CARLSEN"), _name_crop("v_20_black_name", side=NameSide.BLACK)]
    path = tmp_path / "manifest.jsonl"
    save_names(crops, path)
    assert load_names(path) == crops


def test_name_label_surname_derived():
    assert NameLabel(text="Carlsen").surname == "carlsen"
    assert NameLabel(text="Vachier-Lagrave").surname == "vachier-lagrave"
    assert NameLabel(text="").surname == ""


def test_name_labels_roundtrip(tmp_path):
    labels = {"a": NameLabel(text="Carlsen"), "b": NameLabel(text="", unreadable=True)}
    path = tmp_path / "gt.json"
    save_name_labels(labels, path)
    assert load_name_labels(path) == labels


def test_read_text_joins_left_to_right():
    class FakeEngine(OcrEngine):
        def detect(self, image_bgr):
            return [(50.0, "Moke"), (10.0, "Niemann,"), (30.0, "Hans")]

    assert FakeEngine().read_text(np.zeros((1, 1, 3), np.uint8)) == "Niemann, Hans Moke"


def test_score_name():
    label = NameLabel(text="Nepomniachtchi")
    # Verbatim read, letters-only & case-insensitive: both pass.
    m = score_name("nepomniachtchi", label)
    assert (m.surname_ok, m.exact) == (True, True)
    # Surname embedded in a noisier read: pipeline metric passes, verbatim fails.
    m = score_name("I. Nepomniachtchi 2771", label)
    assert (m.surname_ok, m.exact) == (True, False)
    # Misread surname fails both.
    m = score_name("Nepomntachtch", label)
    assert (m.surname_ok, m.exact) == (False, False)


def test_read_clocks_keeps_a_flagged_zero_clock():
    """A clock at 0:00 parses to 0 seconds — falsy, but a legitimate value."""

    class FakeEngine(OcrEngine):
        def detect(self, image_bgr):
            return [(10.0, "0:00"), (300.0, "0:37")]

    img = np.zeros((1, 1, 3), np.uint8)
    assert FakeEngine().read_clocks(img) == [0, 37]


def _write_video_cache(data_dir, vid, descriptors_name=None):
    from chessqueries.annotate.pipeline import DESCRIPTORS_EXT
    from chessqueries.annotate.templates import Shot, save_shots
    from chessqueries.annotate.video import DEFAULT_FORMAT_ID

    stem = f"{vid}.{DEFAULT_FORMAT_ID}"
    (data_dir / f"{stem}.mp4").write_bytes(b"")
    save_shots([Shot(index=0, start_frame=0, end_frame=10)], data_dir / f"{stem}.shots.json")
    np.save(data_dir / (descriptors_name or f"{stem}{DESCRIPTORS_EXT}"), np.zeros((1, 4), np.float32))
    return stem


def test_cached_videos_and_load_use_the_live_descriptor_cache(tmp_path):
    from chessqueries.annotate.ocr_bench import cached_videos, load_cached_shots

    _write_video_cache(tmp_path, "vidA")
    assert cached_videos(tmp_path) == ["vidA"]
    cached = load_cached_shots(tmp_path, "vidA")
    assert [s.start_frame for s in cached.shots] == [0]
    assert cached.descriptors.shape == (1, 4)


def test_cached_videos_ignores_a_stale_descriptor_space(tmp_path):
    """A cache written in a different descriptor space (different extension tag) must
    not count as ingested — mixing spaces would silently misclassify every shot."""
    from chessqueries.annotate.ocr_bench import cached_videos
    from chessqueries.annotate.video import DEFAULT_FORMAT_ID

    _write_video_cache(tmp_path, "vidA", descriptors_name=f"vidA.{DEFAULT_FORMAT_ID}.descriptors.npy")
    assert cached_videos(tmp_path) == []


def test_label_filter_indices():
    from chessqueries.annotate.ocr_bench import LabelFilter, filtered_indices

    ids = ["a", "b", "c"]
    labeled = {"b"}
    assert filtered_indices(ids, labeled, LabelFilter.REVIEW) == [1]
    assert filtered_indices(ids, labeled, LabelFilter.TODO) == [0, 2]
    assert filtered_indices(ids, labeled, LabelFilter.ALL) == [0, 1, 2]


def test_neighbor_index_steps_and_reentry():
    from chessqueries.annotate.ocr_bench import neighbor_index

    act = [0, 2, 5]
    assert neighbor_index(2, act, +1) == 5
    assert neighbor_index(2, act, -1) == 0
    assert neighbor_index(5, act, +1) == 5  # clamped at the end
    # cur just left the set (saved in TODO mode): land on the next in travel direction
    assert neighbor_index(1, act, +1) == 2
    assert neighbor_index(1, act, -1) == 0
    assert neighbor_index(6, act, +1) == 5  # nothing after -> nearest before
    assert neighbor_index(3, [], +1) == 3  # empty set: stay put


def test_readable_filters_missing_and_unreadable():
    from chessqueries.annotate.ocr_bench import readable

    crops = [_crop("c1"), _crop("c2"), _crop("c3")]
    labels = {"c1": ClockLabel("0:19", "0:37"), "c2": ClockLabel("", "", unreadable=True)}
    assert [(c.crop_id, lab.white_text) for c, lab in readable(crops, labels)] == [("c1", "0:19")]


def test_selected_engines_skips_unknown(capsys):
    from chessqueries.annotate.ocr_bench import selected_engines

    out = selected_engines(["definitely-not-an-engine"])
    assert out == []
    assert "unknown" in capsys.readouterr().out


def test_write_results_layout(tmp_path):
    import json

    from chessqueries.annotate.ocr_bench import write_results

    path = write_results(tmp_path / "results", [{"engine": "e", "n": 1}],
                         {"c1": {"white": "0:19"}}, {"e": {"c1": {"predicted": ["0:19"]}}})
    d = json.loads(path.read_text())
    assert set(d) == {"summary", "ground_truth", "predictions"}
    assert d["summary"][0]["engine"] == "e"


def test_name_score_accuracies():
    from chessqueries.annotate.ocr_bench import NameScore

    s = NameScore(name="e", n=4, surname_ok=3, exact=1, mean_latency_ms=2.0, per_crop={})
    assert s.surname_acc == 0.75 and s.exact_acc == 0.25
    empty = NameScore(name="e", n=0, surname_ok=0, exact=0, mean_latency_ms=0.0, per_crop={})
    assert empty.surname_acc == 0.0
