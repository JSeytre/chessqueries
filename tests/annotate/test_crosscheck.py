"""Pure cross-check logic: candidate windows, window scoring, gate triage, and the
CrossCheck schema round-trip (no model, no network)."""

import json

import numpy as np
from annotate_fixtures import make_annotation

from chessqueries.annotate.crosscheck import gate, mark_duplicates, score_window
from chessqueries.annotate.identify import candidate_window
from chessqueries.annotate.relay import parse_round_pgn
from chessqueries.annotate.schema import Annotation, Bucket, CrossCheck

PGN = """[Event "T"]
[White "Niemann, Hans Moke"]
[Black "Vachier-Lagrave, Maxime"]
[Result "*"]
[TimeControl "25+10"]

1. e4 {[%clk 0:25:00]} e5 {[%clk 0:24:40]} 2. Nf3 {[%clk 0:24:30]} Nc6 {[%clk 0:24:20]} *
"""



# --- candidate_window -------------------------------------------------------
def test_window_is_matched_ply_padded_by_desync():
    g = parse_round_pgn(PGN)[0]  # 5 positions (ply 0..4)
    assert candidate_window(g, 2, (None, None), desync=1) == [1, 2, 3]


def test_window_clamps_at_timeline_edges():
    g = parse_round_pgn(PGN)[0]
    assert candidate_window(g, 2, (None, None), desync=3) == [0, 1, 2, 3, 4]


def test_window_includes_clock_matched_plies():
    g = parse_round_pgn(PGN)[0]
    # (1470, 1476) clock-matches ply 3 (White just moved); desync=0 keeps anchors only.
    win = candidate_window(g, 0, (1470, 1476), desync=0)
    assert 0 in win and 3 in win


# --- score_window + gate ----------------------------------------------------
def _logp(pred_labels):
    """Sharp log-prob array whose argmax is ``pred_labels`` (0 at the predicted
    class, -10 elsewhere) — value at any label doubles as a confidence proxy."""
    a = np.full((64, 13), -10.0)
    for sq, c in enumerate(pred_labels):
        a[sq, c] = 0.0
    return a


def _boards():
    a = [0] * 64  # bare board
    b = a.copy()
    b[0] = 1  # differs from A at square 0
    c = a.copy()
    c[1] = 1  # differs from A/B at square 1
    return a, b, c


def test_accept_when_model_matches_one_ply_cleanly():
    a, b, c = _boards()
    cands = {10: a, 11: b, 12: c}
    d = gate(score_window(_logp(b), cands), cands)
    assert d.bucket is Bucket.ACCEPT
    assert d.chosen.ply == 11 and d.chosen.fit_diff == 0
    assert d.margin >= 5 and not d.repetition


def test_repetition_does_not_block_accept():
    a, b, c = _boards()
    cands = {10: a, 11: b, 12: c, 13: b.copy()}  # ply 13 repeats ply 11's board
    d = gate(score_window(_logp(b), cands), cands)
    assert d.bucket is Bucket.ACCEPT and d.repetition
    assert d.margin >= 5  # measured against the *different* boards, not the repeat


def test_review_when_fit_fails_but_margin_holds():
    a, b, c = _boards()
    pred = b.copy()
    for sq in range(5, 11):  # 6 wrong squares outside the differing set -> fit_diff > tau_fit
        pred[sq] = 1
    cands = {10: a, 11: b, 12: c}
    d = gate(score_window(_logp(pred), cands), cands)
    assert d.chosen.fit_diff > 4 and d.margin >= 5
    assert d.bucket is Bucket.REVIEW


def test_quarantine_when_both_gates_fail():
    a, b, c = _boards()
    cands = {10: a, 11: b, 12: c}
    d = gate(score_window(_logp([2] * 64), cands), cands)  # model predicts a class no candidate has
    assert d.bucket is Bucket.QUARANTINE
    assert d.chosen.fit_diff > 4 and (d.margin is None or d.margin < 5)


# --- dedup ------------------------------------------------------------------
def _acc(fi, *, game, placement, fit, ts, margin=20.0, bucket=Bucket.ACCEPT):
    cc = CrossCheck(bucket, chosen_ply=fi, clock_ply=fi, fit_diff=fit, margin=margin, window=[fi])
    return Annotation(
        video_id="v", frame_index=fi, timestamp_s=ts, template_id="t", crop_bbox=[0, 0, 4, 4],
        game_id=game, round_id="r", ply=fi, fen=f"{placement} w - - 0 1", placement=placement,
        side_to_move="w", white="W", black="B", white_clk_s=1, black_clk_s=1, confidence=1.0,
        crosscheck=cc,
    )


P1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
P2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"  # a different placement


def test_dedup_keeps_best_fit_copy_per_game_placement():
    anns = [
        _acc(1, game="g", placement=P1, fit=2, ts=10.0),  # same FEN, worse fit -> dup
        _acc(2, game="g", placement=P1, fit=0, ts=20.0),  # same FEN, best fit -> kept
        _acc(3, game="g", placement=P2, fit=0, ts=30.0),  # different FEN -> kept
    ]
    dup = {a.frame_index: a.crosscheck.duplicate for a in mark_duplicates(anns)}
    assert dup == {1: True, 2: False, 3: False}  # best-fit (frame 2) survives, not the first


def test_dedup_is_per_game():
    anns = [
        _acc(1, game="a", placement=P1, fit=0, ts=1.0),
        _acc(2, game="b", placement=P1, fit=0, ts=2.0),  # same FEN, other game -> not a dup
    ]
    out = {a.frame_index: a.crosscheck.duplicate for a in mark_duplicates(anns)}
    assert out == {1: False, 2: False}


def test_dedup_spans_buckets_and_accept_outranks_review():
    """All buckets dedup (a cross-check can repoint two review frames onto the same FEN),
    and a confident accept always keeps the slot over a shakier review of that position."""
    anns = [
        _acc(1, game="g", placement=P1, fit=6, ts=1.0, bucket=Bucket.REVIEW),  # review, same FEN
        _acc(2, game="g", placement=P1, fit=4, ts=2.0, bucket=Bucket.ACCEPT),  # accept wins
        _acc(3, game="g", placement=P1, fit=5, ts=3.0, bucket=Bucket.REVIEW),  # review, same FEN
        _acc(4, game="g", placement=P2, fit=8, ts=4.0, bucket=Bucket.QUARANTINE),  # other FEN -> kept
    ]
    out = {a.frame_index: a.crosscheck.duplicate for a in mark_duplicates(anns)}
    assert out == {1: True, 2: False, 3: True, 4: False}


def test_dedup_collapses_two_reviews_to_best_fit():
    """Two review frames repointed onto one position keep only the lower-fit copy."""
    anns = [
        _acc(1, game="g", placement=P1, fit=7, ts=1.0, bucket=Bucket.REVIEW),
        _acc(2, game="g", placement=P1, fit=6, ts=2.0, bucket=Bucket.REVIEW),  # lower fit -> kept
    ]
    out = {a.frame_index: a.crosscheck.duplicate for a in mark_duplicates(anns)}
    assert out == {1: True, 2: False}


# --- schema round-trip ------------------------------------------------------
def _ann(**kw):
    base = dict(timestamp_s=1.0, crop_bbox=[0, 0, 4, 4], ply=4,
                white_clk_s=10, black_clk_s=10, confidence=1.0)
    return make_annotation(**{**base, **kw})


def test_crosscheck_round_trips_through_json():
    cc = CrossCheck(Bucket.ACCEPT, chosen_ply=5, clock_ply=4, fit_diff=1, margin=7.5,
                    window=[3, 4, 5], repetition=False)
    back = Annotation.from_dict(json.loads(json.dumps(_ann(crosscheck=cc).to_dict())))
    assert back.crosscheck.bucket is Bucket.ACCEPT
    assert back.crosscheck.margin == 7.5 and back.crosscheck.window == [3, 4, 5]


def test_crosscheck_margin_none_round_trips():
    cc = CrossCheck(Bucket.REVIEW, chosen_ply=4, clock_ply=4, fit_diff=0, margin=None, window=[4])
    back = Annotation.from_dict(json.loads(json.dumps(_ann(crosscheck=cc).to_dict())))
    assert back.crosscheck.margin is None

    plain = Annotation.from_dict(json.loads(json.dumps(_ann().to_dict())))
    assert plain.crosscheck is None  # backward-compatible: absent stays None


# --- model-source validation (LoRA adapter vs. full checkpoint, e.g. V2) -----
def test_crosscheck_requires_exactly_one_model_source(tmp_path):
    import pytest

    from chessqueries.annotate import workflow
    from chessqueries.annotate.manifest import Manifest

    manifest = Manifest(path=tmp_path / "videos.json")
    with pytest.raises(ValueError, match="exactly one"):
        workflow.crosscheck(manifest)  # neither adapter nor checkpoint
    with pytest.raises(ValueError, match="exactly one"):
        workflow.crosscheck(manifest, adapter=tmp_path / "a.pt", checkpoint=tmp_path / "m.ckpt")
    with pytest.raises(ValueError, match="resolution"):
        workflow.crosscheck(manifest, checkpoint=tmp_path / "m.ckpt")  # missing resolution
