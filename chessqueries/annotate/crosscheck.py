"""Cross-check clock-OCR candidates against the visual recognizer: score every ply
in a frame's :func:`~chessqueries.annotate.identify.candidate_window` and gate on

- **fit** — the model's board within ``tau_fit`` squares of the chosen position, and
- **margin** — the chosen ply beats the best *different* position by ``tau_margin``
  log-prob over their distinguishing squares (repetitions don't count against it).

Both pass -> ``accept`` (the model may shift the ply off the clock match, fixing
desync); one fails -> ``review``; both fail -> ``quarantine``. The label's authority
always stays the relay FEN; the model only confirms or relocates it. ``score_window``
and ``gate`` are pure and tested in isolation; the orchestrator pulls crops.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from chessqueries.annotate.identify import candidate_window
from chessqueries.annotate.pipeline import game_id
from chessqueries.annotate.relay import GameTimeline
from chessqueries.annotate.schema import Annotation, AnnotationFile, Bucket, CrossCheck, Stage
from chessqueries.annotate.templates import Rect
from chessqueries.core import Board

# Calibrated on the first reviewed video (held-out val): correct plies diff <=2
# squares, broken/occluded frames >=6; adjacent plies separate cleanly by log-prob
# (true-ply margin >=~5) where the raw 2-square diff is too thin to trust.
DEFAULT_DESYNC = 3
DEFAULT_TAU_FIT = 4
DEFAULT_TAU_MARGIN = 5.0


@dataclass(frozen=True)
class PlyScore:
    ply: int
    fit_diff: int  # squares the model's argmax board differs from this candidate
    score: float  # log-prob of this candidate over the window's differing squares


def score_window(log_probs: np.ndarray, candidates: dict[int, list[int]]) -> list[PlyScore]:
    """Score each candidate ply against the model's per-square distribution.

    ``fit_diff`` compares the model's argmax board to the candidate over all 64
    squares; ``score`` is the candidate's log-prob over only the squares that
    *differ* across the window — the moves that actually distinguish the plies, so
    the static majority of the board can't drown the signal.
    """
    pred = log_probs.argmax(axis=1)
    plies = sorted(candidates)
    differing = [sq for sq in range(64) if len({candidates[p][sq] for p in plies}) > 1]
    out: list[PlyScore] = []
    for p in plies:
        labels = candidates[p]
        fit = int(np.count_nonzero(pred != np.asarray(labels)))
        score = float(sum(log_probs[sq, labels[sq]] for sq in differing))
        out.append(PlyScore(p, fit, score))
    return out


@dataclass(frozen=True)
class Decision:
    bucket: Bucket
    chosen: PlyScore
    margin: float | None  # None when the only rivals are repetitions of the chosen board
    repetition: bool


def gate(
    scores: list[PlyScore],
    candidates: dict[int, list[int]],
    *,
    tau_fit: int = DEFAULT_TAU_FIT,
    tau_margin: float = DEFAULT_TAU_MARGIN,
) -> Decision:
    """Triage one frame from its scored window. The chosen ply is the highest-scoring
    candidate; the margin is measured only against candidates with a *different*
    placement, so repeated positions never block an accept."""
    chosen = max(scores, key=lambda s: s.score)
    chosen_labels = candidates[chosen.ply]
    others = [s for s in scores if candidates[s.ply] != chosen_labels]
    repetition = any(s.ply != chosen.ply and candidates[s.ply] == chosen_labels for s in scores)
    margin = None if not others else chosen.score - max(o.score for o in others)

    fit_ok = chosen.fit_diff <= tau_fit
    margin_ok = margin is None or margin >= tau_margin
    if fit_ok and margin_ok:
        bucket = Bucket.ACCEPT
    elif fit_ok or margin_ok:
        bucket = Bucket.REVIEW
    else:
        bucket = Bucket.QUARANTINE
    return Decision(bucket=bucket, chosen=chosen, margin=margin, repetition=repetition)


# Tiebreak rank when several copies of one position survive: an accept is a more
# trustworthy keeper than a review, which beats a quarantine.
_BUCKET_RANK = {Bucket.ACCEPT: 0, Bucket.REVIEW: 1, Bucket.QUARANTINE: 2}


def mark_duplicates(annotations: list[Annotation]) -> list[Annotation]:
    """Flag redundant copies: within one game, frames that share a placement are the
    broadcast lingering on (or cutting back to) an unmoved board — and after cross-check
    they can even arrive at the same position from *different* clock plies. Keep one copy
    per ``(game, placement)`` and mark the rest ``duplicate``.

    The keeper is the most trustworthy copy: an ``accept`` outranks a ``review`` outranks
    a ``quarantine``, then lowest ``fit_diff``, then widest margin, then earliest. So a
    confident accept always wins over a shakier copy of the same position, and if a *wrong*
    accept ever collides on a FEN with a correct one, the copy whose image actually matches
    the board survives.

    All buckets dedup, not just accepts: a second image of a position we already have from
    the same game adds nothing to review either, so it's hidden and reject-seeded."""

    def quality(a: Annotation) -> tuple:
        cc = a.crosscheck
        margin = cc.margin if cc.margin is not None else float("inf")
        return (_BUCKET_RANK.get(cc.bucket, 3), cc.fit_diff, -margin, a.timestamp_s)

    best: dict[tuple[str, str], Annotation] = {}
    for a in annotations:
        if a.crosscheck is None:
            continue
        key = (a.game_id, a.placement)
        if key not in best or quality(a) < quality(best[key]):
            best[key] = a

    out: list[Annotation] = []
    for a in annotations:
        cc = a.crosscheck
        if cc is None:
            out.append(a)
            continue
        is_dup = best[(a.game_id, a.placement)] is not a
        if is_dup == cc.duplicate:
            out.append(a)
        else:
            out.append(
                Annotation.from_dict({**a.to_dict(), "crosscheck": {**cc.to_dict(), "duplicate": is_dup}})
            )
    return out


def timelines_by_game(round_ids: list[str]) -> dict[str, GameTimeline]:
    """Load every game of every round, keyed like ``Annotation.game_id`` so a
    candidate can be matched back to its relay timeline."""
    from chessqueries.annotate import relay

    out: dict[str, GameTimeline] = {}
    for rid in round_ids:
        for tl in relay.load_round(rid):
            out[game_id(rid, tl)] = tl
    return out


def _repoint(ann: Annotation, timeline: GameTimeline, ply: int, cc: CrossCheck) -> Annotation:
    """Attach the cross-check verdict, relocating the FEN to the model-chosen ply
    (pulled from the relay — still authoritative) when it shifted off the clock."""
    if ply == ann.ply:
        return Annotation.from_dict({**ann.to_dict(), "crosscheck": cc.to_dict()})
    pos = timeline.position_at(ply)
    return Annotation.from_dict(
        {
            **ann.to_dict(),
            "ply": ply,
            "fen": pos.fen,
            "placement": pos.placement,
            "side_to_move": pos.turn.fen,
            "white_clk_s": pos.white_clk_s,
            "black_clk_s": pos.black_clk_s,
            "crosscheck": cc.to_dict(),
        }
    )


def crosscheck_annotation(
    ann: Annotation,
    timeline: GameTimeline,
    recognizer,
    reader,
    *,
    desync: int = DEFAULT_DESYNC,
    tau_fit: int = DEFAULT_TAU_FIT,
    tau_margin: float = DEFAULT_TAU_MARGIN,
) -> Annotation:
    window = candidate_window(timeline, ann.ply, (ann.white_clk_s, ann.black_clk_s), desync=desync)
    candidates = {p: Board.from_fen(timeline.position_at(p).placement).labels for p in window}
    crop = Rect.from_list(ann.crop_bbox).crop(reader.frame_at_index(ann.frame_index))
    pred = recognizer.predict_crop(crop)
    scores = score_window(pred.log_probs, candidates)
    d = gate(scores, candidates, tau_fit=tau_fit, tau_margin=tau_margin)
    cc = CrossCheck(
        bucket=d.bucket,
        chosen_ply=d.chosen.ply,
        clock_ply=ann.ply,
        fit_diff=d.chosen.fit_diff,
        margin=d.margin,
        window=window,
        repetition=d.repetition,
    )
    return _repoint(ann, timeline, d.chosen.ply, cc)


def crosscheck_file(
    annfile: AnnotationFile,
    recognizer,
    reader,
    *,
    desync: int = DEFAULT_DESYNC,
    tau_fit: int = DEFAULT_TAU_FIT,
    tau_margin: float = DEFAULT_TAU_MARGIN,
    log=print,
) -> AnnotationFile:
    """Cross-check every candidate in one video's file, triaging into buckets. Frames
    whose game has no loadable relay timeline are left for review (can't cross-check)."""
    from tqdm import tqdm

    games = timelines_by_game(annfile.provenance.get("round_ids", []))
    out: list[Annotation] = []
    counts: Counter[str] = Counter()
    for a in tqdm(annfile.annotations, desc="crosscheck", unit="frame"):
        tl = games.get(a.game_id)
        if tl is None:
            cc = CrossCheck(Bucket.REVIEW, a.ply, a.ply, fit_diff=-1, margin=None, window=[a.ply])
            out.append(Annotation.from_dict({**a.to_dict(), "crosscheck": cc.to_dict()}))
            counts["no-timeline"] += 1
            continue
        new = crosscheck_annotation(
            a, tl, recognizer, reader, desync=desync, tau_fit=tau_fit, tau_margin=tau_margin
        )
        out.append(new)
        counts[new.crosscheck.bucket.value] += 1
        counts["shifted"] += int(new.crosscheck.chosen_ply != new.crosscheck.clock_ply)

    out = mark_duplicates(out)
    n_dup = sum(bool(a.crosscheck and a.crosscheck.duplicate) for a in out)

    log(
        f"SUMMARY: {len(out)} frame(s) -> {counts['accept']} accept · "
        f"{counts['review'] + counts['no-timeline']} review · {counts['quarantine']} quarantine "
        f"({n_dup} same-game FEN dup(s) hidden, {counts['shifted']} ply-shifted by the model, "
        f"{counts['no-timeline']} had no timeline)"
    )
    prov = {
        **annfile.provenance,
        "stage": Stage.CROSSCHECKED.value,
        "crosscheck": {
            "desync": desync,
            "tau_fit": tau_fit,
            "tau_margin": tau_margin,
            "n_accept": counts["accept"],
            "n_duplicate": n_dup,
            "n_review": counts["review"] + counts["no-timeline"],
            "n_quarantine": counts["quarantine"],
            "n_shifted": counts["shifted"],
        },
    }
    return AnnotationFile(provenance=prov, annotations=out)
