"""Pure-logic tests for annotation review (no video/Gradio)."""

import numpy as np
from annotate_fixtures import START, make_annotation

from chessqueries.annotate.relay import parse_round_pgn
from chessqueries.annotate.review import (
    apply_reviews,
    board_svg,
    corrected_annotation,
    crop_data_uri,
    last_move,
    lichess_url,
    neighbor_plies,
    side_to_move,
)
from chessqueries.annotate.schema import Annotation, AnnotationFile

PGN = """[Event "T"]
[White "Niemann, Hans Moke"]
[Black "So, Wesley"]
[Result "*"]
[TimeControl "25+10"]

1. e4 {[%clk 0:25:00]} e5 {[%clk 0:24:40]} 2. Nf3 {[%clk 0:24:30]} Nc6 {[%clk 0:24:20]} *
"""


def _ann(frame_index: int) -> Annotation:
    return make_annotation(frame_index=frame_index)


def test_apply_reviews_accept_reject():
    file = AnnotationFile(provenance={}, annotations=[_ann(100), _ann(200), _ann(300)])
    out = apply_reviews(file, accepted={100}, rejected={200})
    by_idx = {a.frame_index: a for a in out.annotations}
    assert set(by_idx) == {100, 300}  # 200 dropped
    assert by_idx[100].verified_by_human is True
    assert by_idx[300].verified_by_human is False
    assert out.provenance["stage"] == "reviewed"


def test_save_is_atomic_and_overwrites(tmp_path):
    out = tmp_path / "v.reviewed.json"
    AnnotationFile(provenance={"stage": "reviewed"}, annotations=[_ann(100)]).save(out)
    # Round-trips, overwrites a prior file, and leaves no .tmp sibling behind (the rename
    # that makes a mid-write crash safe also cleans up after itself).
    AnnotationFile(provenance={"stage": "reviewed"}, annotations=[_ann(100), _ann(200)]).save(out)
    reloaded = AnnotationFile.load(out)
    assert [a.frame_index for a in reloaded.annotations] == [100, 200]
    assert not list(tmp_path.glob("*.tmp"))


def test_lichess_url():
    url = lichess_url(START)
    assert "lichess.org" in url


def test_board_svg_and_crop_uri():
    assert "<svg" in board_svg(START + " w - - 0 1")
    uri = crop_data_uri(np.zeros((20, 20, 3), dtype=np.uint8))
    assert uri.startswith("data:image/png;base64,")


def test_neighbor_plies_clamped():
    tl = parse_round_pgn(PGN)[0]  # plies 0..4
    assert [p for p, _ in neighbor_plies(tl, 0, k=2)] == [0, 1, 2]  # clamped low
    assert [p for p, _ in neighbor_plies(tl, 2, k=2)] == [0, 1, 2, 3, 4]


def test_neighbor_plies_asymmetric_window():
    tl = parse_round_pgn(PGN)[0]  # plies 0..4
    # widen only forward (time-trouble: true position is several plies ahead)
    assert [p for p, _ in neighbor_plies(tl, 2, back=0, fwd=2)] == [2, 3, 4]
    # widen only backward
    assert [p for p, _ in neighbor_plies(tl, 2, back=2, fwd=0)] == [0, 1, 2]
    # both sides clamp to the game bounds
    assert [p for p, _ in neighbor_plies(tl, 2, back=10, fwd=10)] == [0, 1, 2, 3, 4]


def test_youtube_url_has_timestamp():
    from chessqueries.annotate.review import youtube_url

    assert youtube_url("hunt9gfNW48", 95.7) == "https://www.youtube.com/watch?v=hunt9gfNW48&t=95s"


def test_side_to_move():
    assert side_to_move(START + " w - - 0 1") == "White"
    assert side_to_move(START + " b - - 0 1") == "Black"


def test_last_move_highlights_relay_move():
    import chess

    tl = parse_round_pgn(PGN)[0]  # 1.e4 e5 2.Nf3 Nc6
    assert last_move(tl, 0) is None  # start position has no preceding move
    assert last_move(tl, 1) == chess.Move.from_uci("e2e4")  # White's 1st
    assert last_move(tl, 3) == chess.Move.from_uci("g1f3")  # White's 2nd (Nf3)
    assert last_move(tl, 99) is None  # out of range -> no move


def test_corrected_annotation_repoints_to_new_ply():
    tl = parse_round_pgn(PGN)[0]
    a = _ann(100)  # claims START (ply 0); correct it to ply 2
    fixed = corrected_annotation(a, tl, 2)
    assert fixed.ply == 2
    assert fixed.placement == tl.position_at(2).placement
    assert fixed.verified_by_human is True and fixed.requires_review is False
    assert fixed.frame_index == 100  # same frame, new label


def test_apply_reviews_with_corrections():
    tl = parse_round_pgn(PGN)[0]
    file = AnnotationFile(provenance={}, annotations=[_ann(100), _ann(200)])
    fix = corrected_annotation(_ann(100), tl, 2)
    out = apply_reviews(file, accepted=set(), rejected={200}, corrections={100: fix})
    by_idx = {a.frame_index: a for a in out.annotations}
    assert set(by_idx) == {100}  # 200 rejected
    assert by_idx[100].ply == 2 and by_idx[100].verified_by_human is True


# --- cross-check bucket integration -----------------------------------------
from chessqueries.annotate.review import bucket_sorted, seed_decisions  # noqa: E402
from chessqueries.annotate.schema import Bucket, CrossCheck  # noqa: E402


def _xann(fi, ts, bucket):
    cc = CrossCheck(bucket, chosen_ply=fi, clock_ply=fi, fit_diff=0, margin=9.0, window=[fi])
    return make_annotation(frame_index=fi, timestamp_s=ts, crop_bbox=[0, 0, 4, 4],
                           ply=fi, confidence=1.0, crosscheck=cc)


def test_bucket_sorted_review_first_quarantine_last():
    anns = [
        _xann(1, 30.0, Bucket.ACCEPT),
        _xann(2, 10.0, Bucket.QUARANTINE),
        _xann(3, 20.0, Bucket.REVIEW),
        _xann(4, 5.0, Bucket.ACCEPT),
    ]
    order = [a.frame_index for a in bucket_sorted(anns)]
    assert order == [3, 4, 1, 2]  # review, then accepts by time, then quarantine


def test_seed_decisions_from_buckets():
    anns = [_xann(1, 0, Bucket.ACCEPT), _xann(2, 0, Bucket.QUARANTINE), _xann(3, 0, Bucket.REVIEW)]
    d = seed_decisions(anns)
    assert d == {1: "accept", 2: "reject"}  # review stays undecided


def test_seed_decisions_rejects_duplicates():
    a = _xann(1, 0, Bucket.ACCEPT)
    dup = _xann(2, 0, Bucket.ACCEPT)
    object.__setattr__(dup, "crosscheck", CrossCheck(
        Bucket.ACCEPT, chosen_ply=2, clock_ply=2, fit_diff=0, margin=9.0, window=[2], duplicate=True))
    assert seed_decisions([a, dup]) == {1: "accept", 2: "reject"}  # redundant dup skipped


def test_visible_for_review_hides_duplicates():
    """Same-game FEN duplicates are dropped from the queue but still seed a reject, so a
    Save over the full set drops them rather than keeping a second copy of a position."""
    from chessqueries.annotate.review import visible_for_review

    a = _xann(1, 0, Bucket.ACCEPT)
    dup = _xann(2, 0, Bucket.ACCEPT)
    object.__setattr__(dup, "crosscheck", CrossCheck(
        Bucket.ACCEPT, chosen_ply=2, clock_ply=2, fit_diff=0, margin=9.0, window=[2], duplicate=True))
    rev = _xann(3, 0, Bucket.REVIEW)
    assert [a.frame_index for a in visible_for_review([a, dup, rev])] == [1, 3]  # dup hidden
    assert seed_decisions([a, dup, rev]).get(2) == "reject"  # but still saved as reject


def test_review_bucket_duplicate_is_hidden_and_rejected():
    """A REVIEW frame the cross-check repointed onto a FEN we already have (same game) is
    flagged duplicate; it must be hidden from the queue AND seeded reject, or an undecided
    review frame would silently survive a Save as an unverified second copy."""
    from chessqueries.annotate.review import visible_for_review

    keep = _xann(1, 0, Bucket.REVIEW)
    dup = _xann(2, 0, Bucket.REVIEW)
    object.__setattr__(dup, "crosscheck", CrossCheck(
        Bucket.REVIEW, chosen_ply=2, clock_ply=2, fit_diff=0, margin=9.0, window=[2], duplicate=True))
    assert [a.frame_index for a in visible_for_review([keep, dup])] == [1]  # dup hidden
    assert seed_decisions([keep, dup]) == {2: "reject"}  # non-dup review undecided; dup rejected


def test_resume_verified_verdict_outranks_duplicate_flag():
    """Cross-checking a salvage-merged file re-runs the dedup over old + new frames
    together; a new copy of a position can win the quality tiebreak and flag the
    previously-VERIFIED copy as duplicate. Resume must keep the human's accept (and
    ply correction) — not silently un-verify it — while an unverified dup still drops."""
    from chessqueries.annotate.review import resume_decisions

    kept = _xann(1, 0, Bucket.ACCEPT)  # verified, now dup-flagged and ply-repointed
    object.__setattr__(kept, "crosscheck", CrossCheck(
        Bucket.ACCEPT, chosen_ply=5, clock_ply=1, fit_diff=0, margin=9.0, window=[5], duplicate=True))
    object.__setattr__(kept, "ply", 5)
    newcopy = _xann(2, 1, Bucket.ACCEPT)  # the unverified keeper that beat it -> pending

    prior_kept = _xann(1, 0, Bucket.ACCEPT)  # human verified it at ply 1
    object.__setattr__(prior_kept, "verified_by_human", True)
    prior_new = _xann(2, 1, Bucket.ACCEPT)  # preseeded pending
    prior = {1: prior_kept, 2: prior_new}

    resumed = resume_decisions([kept, newcopy], prior)
    # verified survives its dup flag; new copy stays pending
    assert resumed.decisions == {1: "accept"}
    # and keeps the human's ply, not the repointed one
    assert resumed.corrections == {1: prior_kept}


def test_pending_first_surfaces_salvaged_frames_before_verified_ones():
    """After `produce --salvage`, the merged candidates mix old verified frames (still
    carrying their cross-check) with new crosscheck-less ones — which bucket_sorted
    interleaves by timestamp. pending_first must pull every undecided frame ahead of
    the decided tail, preserving relative order within each half."""
    from chessqueries.annotate.review import pending_first, resume_decisions

    def _new(fi, ts):  # a salvaged frame: no cross-check
        a = _xann(fi, ts, Bucket.REVIEW)
        object.__setattr__(a, "crosscheck", None)
        return a

    old1, old2 = _xann(1, 10.0, Bucket.REVIEW), _xann(2, 40.0, Bucket.REVIEW)
    new1, new2 = _new(3, 20.0), _new(4, 30.0)
    prior1, prior2 = _xann(1, 10.0, Bucket.REVIEW), _xann(2, 40.0, Bucket.REVIEW)
    object.__setattr__(prior1, "verified_by_human", True)
    object.__setattr__(prior2, "verified_by_human", True)

    anns = bucket_sorted([old1, old2, new1, new2])
    assert [a.frame_index for a in anns] == [1, 3, 4, 2]  # interleaved by timestamp
    resumed = resume_decisions(anns, {1: prior1, 2: prior2, 3: _new(3, 20.0), 4: _new(4, 30.0)})
    assert [a.frame_index for a in pending_first(anns, resumed.decisions)] == [3, 4, 1, 2]


def test_pending_first_keeps_fresh_crosschecked_order():
    """On a fresh cross-checked review only the review bucket is undecided — and it
    already sorts first, so pending_first must be a no-op there."""
    from chessqueries.annotate.review import pending_first

    anns = bucket_sorted([
        _xann(1, 30.0, Bucket.ACCEPT),
        _xann(2, 10.0, Bucket.QUARANTINE),
        _xann(3, 20.0, Bucket.REVIEW),
        _xann(4, 5.0, Bucket.ACCEPT),
    ])
    order = [a.frame_index for a in pending_first(anns, seed_decisions(anns))]
    assert order == [a.frame_index for a in anns] == [3, 4, 1, 2]


def test_resume_rejects_duplicates_flagged_after_first_review():
    """A dup flagged *after* a file was first reviewed (a later dedup pass) must be rejected
    on resume — even kept-but-unverified — else it re-saves itself every launch and the
    pending count never clears. Verified non-dups stay accepted; rejected (dropped from the
    prior file) stay rejected."""
    from chessqueries.annotate.review import resume_decisions

    keep = _xann(1, 0, Bucket.ACCEPT)  # verified in prior -> accept
    newdup = _xann(2, 0, Bucket.REVIEW)  # kept-but-unverified, now flagged duplicate
    object.__setattr__(newdup, "crosscheck", CrossCheck(
        Bucket.REVIEW, chosen_ply=2, clock_ply=2, fit_diff=0, margin=9.0, window=[2], duplicate=True))
    gone = _xann(3, 0, Bucket.REVIEW)  # not in prior (was rejected) -> reject

    prior_keep, prior_dup = _xann(1, 0, Bucket.ACCEPT), _xann(2, 0, Bucket.REVIEW)
    object.__setattr__(prior_keep, "verified_by_human", True)
    object.__setattr__(prior_dup, "verified_by_human", False)  # kept, never verified
    prior = {1: prior_keep, 2: prior_dup}

    resumed = resume_decisions([keep, newdup, gone], prior)
    # the stuck dup now drops
    assert resumed.decisions == {1: "accept", 2: "reject", 3: "reject"}
    assert resumed.corrections == {}


def test_fit_color_scale():
    from chessqueries.annotate.review import fit_color

    assert fit_color(0) == "#22c55e"  # green: perfect read
    assert fit_color(1) == "#eab308"  # yellow: within 2
    assert fit_color(2) == "#eab308"
    assert fit_color(3) == "#ef4444"  # red: further off
    assert fit_color(-1) == "#9ca3af"  # gray: no timeline


def test_margin_color_scale():
    from chessqueries.annotate.review import margin_color

    assert margin_color(None) == "#22c55e"  # green: ∞ (only repetitions nearby)
    assert margin_color(5.0) == "#22c55e"  # green: at the accept gate
    assert margin_color(4.9) == "#eab308"  # yellow: thin
    assert margin_color(2.0) == "#eab308"
    assert margin_color(1.9) == "#ef4444"  # red: candidates barely separate


def test_crosscheck_bar_html_gray_na_without_crosscheck():
    from chessqueries.annotate.review import crosscheck_bar_html

    html = crosscheck_bar_html(None)
    assert "fit" in html and "margin" in html  # bar still present
    assert "n/a" in html and "#9ca3af" in html  # gray n/a rather than vanishing


def test_crosscheck_bar_html_shows_values_and_colors():
    from chessqueries.annotate.review import crosscheck_bar_html

    cc = CrossCheck(Bucket.ACCEPT, chosen_ply=1, clock_ply=1, fit_diff=0, margin=None, window=[1])
    html = crosscheck_bar_html(cc)
    assert "fit" in html and "margin" in html
    assert ">0<" in html and "∞" in html  # fit value and ∞ margin rendered
    assert "#22c55e" in html  # both green for a clean accept


def test_margin_color_respects_the_file_gate():
    """The green threshold is the run's actual tau_margin, not a hardcoded 5.0."""
    from chessqueries.annotate.review import margin_color

    assert margin_color(3.5, tau_margin=3.0) == "#22c55e"  # at/above a custom gate
    assert margin_color(2.5, tau_margin=3.0) == "#eab308"  # below the gate, above thin
    assert margin_color(3.5, tau_margin=7.0) == "#eab308"  # same value, stricter gate


def test_crosscheck_bar_html_uses_the_file_gate():
    from chessqueries.annotate.review import crosscheck_bar_html

    cc = CrossCheck(Bucket.ACCEPT, chosen_ply=1, clock_ply=1, fit_diff=0, margin=4.0, window=[1])
    assert "#22c55e" in crosscheck_bar_html(cc, tau_margin=4.0)  # green at a 4.0 gate
    green_chips = crosscheck_bar_html(cc, tau_margin=5.0).count("#22c55e")
    assert green_chips == 1  # fit stays green; the 4.0 margin is below the 5.0 gate


def test_legend_markdown_renders_the_file_gates():
    from chessqueries.annotate.review import GateThresholds, legend_markdown

    md = legend_markdown(GateThresholds(tau_fit=3, tau_margin=7.5))
    assert "≤3 auto-accepts" in md
    assert "≥7.5" in md
    assert "≥5" not in md.replace("≥7.5", "")  # no stale default leaks through


def test_gate_thresholds_read_from_provenance_with_defaults():
    from chessqueries.annotate.crosscheck import DEFAULT_TAU_FIT, DEFAULT_TAU_MARGIN
    from chessqueries.annotate.review import GateThresholds, gate_thresholds

    prov = {"crosscheck": {"tau_fit": 2, "tau_margin": 8.0}}
    assert gate_thresholds(prov) == GateThresholds(tau_fit=2, tau_margin=8.0)
    assert gate_thresholds({}) == GateThresholds(DEFAULT_TAU_FIT, DEFAULT_TAU_MARGIN)  # pre-gate
