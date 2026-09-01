"""Intra-shot reconciliation of per-frame identifications (no video/OCR)."""

import numpy as np

from chessqueries.annotate.align import (
    FrameObservation,
    _read_clocks,
    _read_names,
    _sample_indices,
    reconcile_shot,
)
from chessqueries.annotate.identify import Identification
from chessqueries.annotate.templates import Layout, Quality, Rect, Shot
from chessqueries.core import Color


class _FakeOcr:
    """Returns preset results in call order (clocks per call; white name first, then black)."""

    def __init__(self, clocks=None, names=None):
        self._clocks = list(clocks or [])
        self._names = list(names or [])

    def clocks_in(self, _crop):
        return self._clocks.pop(0)

    def texts(self, _crop):
        return self._names.pop(0)


def _clock_layout() -> Layout:
    return Layout(
        "t",
        Quality.USEFUL,
        board_rect=Rect(0, 0, 8, 8),
        digital_clock_rect=Rect(0, 0, 10, 10),
    )


def test_reliable_digital_clock_is_used():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    ocr = _FakeOcr(clocks=[[200, 300]])
    assert _read_clocks(ocr, frame, _clock_layout()) == [200, 300]


def test_incomplete_digital_clock_returns_no_clock():
    # Fewer than two readable values -> can't form a pair, so don't match the frame.
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    ocr = _FakeOcr(clocks=[[100]])
    assert _read_clocks(ocr, frame, _clock_layout()) == []


def test_desynced_digital_zero_returns_no_clock():
    """A 0:0 overlay (desynced/not-loaded) would match a bogus ply -> return no clock so
    the frame is left unidentified for the review/model pass."""
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    ocr = _FakeOcr(clocks=[[0, 250]])
    assert _read_clocks(ocr, frame, _clock_layout()) == []


def test_names_read_from_separate_regions():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    layout = Layout(
        "t",
        Quality.USEFUL,
        board_rect=Rect(0, 0, 8, 8),
        white_name_rect=Rect(0, 0, 10, 5),
        black_name_rect=Rect(0, 6, 10, 5),
    )
    ocr = _FakeOcr(names=[["NIEMANN"], ["SO"]])
    plates = _read_names(ocr, frame, layout)
    assert plates.white == ("NIEMANN",) and plates.black == ("SO",)


def _ident(game: int, ply: int, conf: float = 1.0) -> Identification:
    return Identification(
        game_index=game,
        ply=ply,
        fen=f"fen{ply}",
        placement=f"plc{ply}",
        white="W",
        black="B",
        static_side=Color.WHITE,
        residual=0.0,
        name_matched=True,
        confidence=conf,
    )


def _obs(t: float, ident) -> FrameObservation:
    return FrameObservation(frame_index=int(t * 30), timestamp_s=t, identification=ident)


def test_distinct_positions_all_emitted():
    # A long take advancing through plies 10..15 -> one label per distinct position.
    obs = [_obs(float(i), _ident(1, 10 + i)) for i in range(6)]
    labels = reconcile_shot(obs)
    assert [lab.ply for lab in labels] == [10, 11, 12, 13, 14, 15]
    assert all(lab.game_index == 1 for lab in labels)
    assert all(abs(lab.confidence - 1.0) < 1e-9 for lab in labels)  # per-frame confidence


def test_same_ply_deduped_to_best_confidence():
    # Camera lingers on one position -> a single label, the highest-confidence frame.
    obs = [_obs(float(i), _ident(1, 10, conf=0.5 + 0.1 * i)) for i in range(6)]
    labels = reconcile_shot(obs)
    assert len(labels) == 1 and labels[0].ply == 10
    assert abs(labels[0].confidence - 1.0) < 1e-9  # best confidence kept


def test_multiple_games_in_one_shot_all_kept():
    # The featured board switches mid-shot: game 1 (plies 10-12) then game 2 (50-52).
    obs = [_obs(float(i), _ident(1, 10 + i)) for i in range(3)]
    obs += [_obs(float(3 + i), _ident(2, 50 + i)) for i in range(3)]
    labels = reconcile_shot(obs, min_game_samples=3)
    assert {lab.game_index for lab in labels} == {1, 2}
    assert sorted(lab.ply for lab in labels) == [10, 11, 12, 50, 51, 52]


def test_isolated_game_misread_dropped():
    # A lone game-2 read among game-1 samples is below min_game_samples -> dropped.
    obs = [_obs(float(i), _ident(1, 10 + i)) for i in range(5)] + [_obs(5.0, _ident(2, 99))]
    labels = reconcile_shot(obs, min_game_samples=3)
    assert {lab.game_index for lab in labels} == {1}
    assert [lab.ply for lab in labels] == [10, 11, 12, 13, 14]


def test_ply_regression_dropped():
    obs = [_obs(float(i), _ident(1, p)) for i, p in enumerate([20, 21, 22, 23, 5, 24])]
    labels = reconcile_shot(obs, max_ply_regression=2)
    assert [lab.ply for lab in labels] == [20, 21, 22, 23, 24]  # the ply-5 misread is dropped


def test_no_identifications_returns_empty():
    assert reconcile_shot([_obs(float(i), None) for i in range(6)]) == []


def test_sample_indices_floor_for_short_shots():
    fps = 30.0
    # 1s shot: 1/0.5 = 2, but floored to min_samples=5 (sampled closer than 0.5s).
    short = _sample_indices(Shot(0, 0, 30), fps, interval_s=0.5, min_samples=5)
    assert len(short) == 5 and all(0 <= i < 30 for i in short)
    # 10s shot: 10/0.5 = 20 samples.
    assert len(_sample_indices(Shot(1, 0, 300), fps, interval_s=0.5, min_samples=5)) == 20
    # very short shot with fewer frames than min_samples: capped at frame count.
    assert len(_sample_indices(Shot(2, 0, 3), fps, interval_s=0.5, min_samples=5)) == 3
