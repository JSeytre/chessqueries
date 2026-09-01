"""Human verification: for each kept frame, compare the real crop against the relay
position rendered as a board, accept/reject (green/red), and — when wrong — correct
to one of the ±2 neighbouring plies (the usual failure is an off-by-a-few clock read).

The automatic signals are good but not infallible, so verification is first-class.
Pure helpers (`apply_reviews`, `neighbor_plies`, `corrected_annotation`, renderers)
are testable; the Gradio UI is a thin shell.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import chess
import chess.svg
import cv2

from chessqueries.annotate.crosscheck import DEFAULT_TAU_FIT, DEFAULT_TAU_MARGIN, timelines_by_game
from chessqueries.annotate.relay import GameTimeline
from chessqueries.annotate.schema import Annotation, AnnotationFile, Bucket, CrossCheck, Stage
from chessqueries.annotate.templates import Rect
from chessqueries.annotate.video import FrameReader, probe
from chessqueries.core import Board

# keyboard shortcuts (outside text fields): navigate + verdict without the mouse.
_HOTKEYS = {
    "q": "earlier-btn",  # widen window backward
    "w": "accept-btn",
    "e": "later-btn",  # widen window forward
    "a": "prev-btn",  # previous image
    "s": "save-btn",
    "d": "next-btn",  # next image
    "x": "discard-btn",
}
gr = None


# Cross-checked review order: review bucket first (needs a human call), then accepts
# (a quick skim), then quarantine (handled last).
_BUCKET_ORDER = {Bucket.REVIEW: 0, Bucket.ACCEPT: 1, Bucket.QUARANTINE: 2}


def bucket_sorted(anns: list[Annotation]) -> list[Annotation]:
    """Order annotations for review by triage bucket then time; a pass with no
    cross-check (plain candidates) keeps its original order."""
    if not any(a.crosscheck for a in anns):
        return list(anns)
    return sorted(
        anns,
        key=lambda a: (_BUCKET_ORDER.get(a.crosscheck.bucket, 0) if a.crosscheck else 0, a.timestamp_s),
    )


def pending_first(anns: list[Annotation], decisions: dict[int, Verdict]) -> list[Annotation]:
    """Stable re-sort putting undecided frames ahead of already-decided ones, so a
    resumed review (e.g. after `produce --salvage` folded new frames into a finished
    file) walks the pending frames contiguously instead of stepping over verified
    ones interleaved by timestamp. Order within each half is preserved."""
    return sorted(anns, key=lambda a: a.frame_index in decisions)


def visible_for_review(anns: list[Annotation]) -> list[Annotation]:
    """Drop same-game FEN duplicates from the review queue: the cross-check kept the
    best-fit copy and flagged the rest, and a second image of a position we already have
    from the same game adds nothing. They stay reject-seeded (so Save drops them) — the
    queue just never shows them."""
    return [a for a in anns if not (a.crosscheck and a.crosscheck.duplicate)]


def seed_decisions(anns: list[Annotation]) -> dict[int, Verdict]:
    """Pre-seed the gate's verdict so a fresh cross-checked review only needs the
    human to adjudicate the review bucket: accepts default to accept, quarantine and
    redundant duplicates (in any bucket) to reject. Frames with no cross-check stay
    undecided — as does the review bucket, unless it's a hidden duplicate."""
    out: dict[int, str] = {}
    for a in anns:
        cc = a.crosscheck
        if cc is None:
            continue
        if cc.duplicate:
            out[a.frame_index] = Verdict.REJECT
        elif cc.bucket is Bucket.ACCEPT:
            out[a.frame_index] = Verdict.ACCEPT
        elif cc.bucket is Bucket.QUARANTINE:
            out[a.frame_index] = Verdict.REJECT
    return out


@dataclass(frozen=True)
class ResumedReview:
    """Review state rebuilt from an already-reviewed file, both keyed by frame index:
    the verdict each frame carries, and the human-corrected annotation for the frames
    whose ply the reviewer changed."""

    decisions: dict[int, Verdict]
    corrections: dict[int, Annotation]


def resume_decisions(anns: list[Annotation], prior: dict[int, Annotation]) -> ResumedReview:
    """Rebuild decisions/corrections from an existing reviewed file so Save merges with
    earlier work. In that file rejected candidates were dropped, verified ones accepted,
    a ply shift is a correction. The human verdict outranks the automatic dedup: a
    verified frame stays accepted even if a later cross-check flagged it duplicate (the
    keeper that beat it may itself be unverified — silently un-verifying human work is
    worse than letting the reviewer adjudicate the other copy). A kept-but-unverified
    duplicate is still always rejected — even one that predates the dedup — else it
    re-saves itself every resume and never clears the pending count."""
    decisions: dict[int, Verdict] = {}
    corrections: dict[int, Annotation] = {}
    for a in anns:
        p = prior.get(a.frame_index)
        if p is not None and p.verified_by_human:
            decisions[a.frame_index] = Verdict.ACCEPT
            if p.ply != a.ply:
                corrections[a.frame_index] = p
        elif a.crosscheck and a.crosscheck.duplicate:
            decisions[a.frame_index] = Verdict.REJECT
        elif p is None:
            decisions[a.frame_index] = Verdict.REJECT
    return ResumedReview(decisions=decisions, corrections=corrections)


def lichess_url(placement: str) -> str:
    return Board.from_fen(placement).lichess_url()


def lichess_broadcast_url(round_id: str) -> str:
    """The live Lichess broadcast round (lichess redirects the placeholder slugs)."""
    return f"https://lichess.org/broadcast/-/-/{round_id}"


def youtube_url(video_id: str, timestamp_s: float) -> str:
    """The source YouTube video seeked to this sample's timestamp."""
    return f"https://www.youtube.com/watch?v={video_id}&t={int(timestamp_s)}s"


class Verdict(str, Enum):
    """A reviewer's decision on one frame."""

    ACCEPT = "accept"
    REJECT = "reject"


# Colours for the fit/margin bar shown directly above the crops.
_GREEN, _YELLOW, _RED, _GRAY = "#22c55e", "#eab308", "#ef4444", "#9ca3af"


def fit_color(fit_diff: int) -> str:
    """Green: a perfect read (0 squares off). Yellow: within 2. Red: further off.
    Gray: no cross-check timeline (``fit_diff < 0``)."""
    if fit_diff < 0:
        return _GRAY
    if fit_diff == 0:
        return _GREEN
    if fit_diff <= 2:
        return _YELLOW
    return _RED


_MARGIN_THIN = 2.0  # yellow band floor: below this the model barely separates candidates


def margin_color(margin: float | None, tau_margin: float = DEFAULT_TAU_MARGIN) -> str:
    """Green: a decisive win — at/above the accept gate ``tau_margin``, or ``∞``
    (``None``: the only rivals are repetitions, so picking either is harmless). Yellow: a
    thin margin (``>= 2``). Red: below that the model barely separates the candidates."""
    if margin is None or margin >= tau_margin:
        return _GREEN
    if margin >= _MARGIN_THIN:
        return _YELLOW
    return _RED


@dataclass(frozen=True)
class GateThresholds:
    """The accept gates a file was cross-checked with: the maximum squares a read may
    differ by, and the minimum log-prob margin between the winning candidate and its rival."""

    tau_fit: int
    tau_margin: float


def gate_thresholds(provenance: dict) -> GateThresholds:
    """The gates this file was cross-checked with, from its provenance; crosscheck
    defaults for files that predate the recorded gates."""
    cc = provenance.get("crosscheck", {})
    return GateThresholds(tau_fit=cc.get("tau_fit", DEFAULT_TAU_FIT),
                          tau_margin=cc.get("tau_margin", DEFAULT_TAU_MARGIN))


def legend_markdown(gates: GateThresholds) -> str:
    """Bottom-of-page legend, rendered from the gate thresholds this file was actually
    cross-checked with (so a run with a custom ``--tau-margin`` doesn't show stale numbers)."""
    import math

    odds = math.exp(gates.tau_margin)
    return (
        f"**legend** · `fit N`: squares the model's read differs from the position "
        f"(≤{gates.tau_fit} auto-accepts) · "
        "`margin M`: how much more the model prefers this ply than the next *different* position — "
        "summed log-probability over the squares that differ between candidate plies "
        f"(≥{gates.tau_margin:g} ≈ {odds:.0f}× more likely; ∞ = only repetitions nearby) · "
        "`⇄a→b`: vision moved the ply from the clock's *a* to *b*."
        "\n\n**colour bar** (above the crops) · fit: 🟢 0 · 🟡 ≤2 · 🔴 >2 · ⬜ n/a · "
        f"margin: 🟢 ≥{gates.tau_margin:g} or ∞ · 🟡 ≥{_MARGIN_THIN:g} · 🔴 <{_MARGIN_THIN:g}"
    )


def crosscheck_bar_html(cc: CrossCheck | None, tau_margin: float = DEFAULT_TAU_MARGIN) -> str:
    """Color-coded fit/margin chips, shown right above the crops so the reviewer reads the
    two automatic signals at a glance. A frame with no cross-check shows a gray ``n/a`` bar
    (rather than vanishing) so the strip is always present."""
    if cc is None:
        fit_txt, margin_txt, fcol, mcol = "n/a", "n/a", _GRAY, _GRAY
    else:
        fit_txt = "n/a" if cc.fit_diff < 0 else str(cc.fit_diff)
        margin_txt = "∞" if cc.margin is None else f"{cc.margin:.1f}"
        fcol, mcol = fit_color(cc.fit_diff), margin_color(cc.margin, tau_margin)

    def chip(label: str, value: str, color: str) -> str:
        return (
            f'<span style="flex:1;background:{color};color:#fff;border-radius:4px;'
            'padding:1px 10px;text-align:center;font-weight:700;font-size:13px;line-height:1.5">'
            f'{label} <b style="font-size:15px">{value}</b></span>'
        )

    return (
        '<div style="display:flex;gap:6px">'
        + chip("fit", fit_txt, fcol)
        + chip("margin", margin_txt, mcol)
        + "</div>"
    )


def board_svg(fen: str, size: int = 320, *, lastmove: chess.Move | None = None) -> str:
    """Render a FEN as an SVG board (python-chess; no extra deps, inlines in HTML).

    ``lastmove`` highlights the from/to squares of the move that produced the position."""
    board = chess.Board(fen)
    check = board.king(board.turn) if board.is_check() else None
    return chess.svg.board(board, size=size, lastmove=lastmove, check=check)


def side_to_move(fen: str) -> str:
    """``"White"``/``"Black"`` from a FEN's active-colour field."""
    return "White" if fen.split()[1] == "w" else "Black"


def last_move(timeline: GameTimeline | None, ply: int) -> chess.Move | None:
    """The move that produced position ``ply`` (parsed from the relay's SAN against the
    previous position), so the board can highlight where the last piece moved."""
    if timeline is None or ply <= 0 or ply >= len(timeline.positions):
        return None
    san = timeline.positions[ply].last_san
    if not san:
        return None
    try:
        return chess.Board(timeline.positions[ply - 1].fen).parse_san(san)
    except ValueError:
        return None


def crop_data_uri(bgr) -> str:
    """PNG data URI for a BGR crop, so it embeds directly in the review HTML."""
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("failed to PNG-encode crop")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def neighbor_plies(
    timeline: GameTimeline, ply: int, k: int = 2, *, back: int | None = None, fwd: int | None = None
) -> list[tuple[int, str]]:
    """``(ply, fen)`` for plies around ``ply`` (clamped), including ``ply``.

    Symmetric by ``k``; pass ``back``/``fwd`` to widen one side independently (the
    review UI grows these on demand — time-trouble + increment can push the true
    position well beyond ±2 plies)."""
    back = k if back is None else back
    fwd = k if fwd is None else fwd
    lo, hi = max(0, ply - back), min(len(timeline.positions) - 1, ply + fwd)
    return [(p, timeline.position_at(p).fen) for p in range(lo, hi + 1)]


def corrected_annotation(ann: Annotation, timeline: GameTimeline, new_ply: int) -> Annotation:
    """Re-point an annotation at a different ply of the same game (a human fix),
    pulling that ply's FEN/clocks from the relay and marking it verified."""
    pos = timeline.position_at(new_ply)
    return Annotation.from_dict(
        {
            **ann.to_dict(),
            "ply": new_ply,
            "fen": pos.fen,
            "placement": pos.placement,
            "side_to_move": pos.turn.fen,
            "white_clk_s": pos.white_clk_s,
            "black_clk_s": pos.black_clk_s,
            "verified_by_human": True,
            "requires_review": False,
        }
    )


def apply_reviews(
    annfile: AnnotationFile,
    accepted: set[int],
    rejected: set[int],
    corrections: dict[int, Annotation] | None = None,
) -> AnnotationFile:
    """New annotation file: drop ``rejected``, replace ``corrections`` (by frame
    index), mark accepted/corrected as ``verified_by_human``, keep the rest."""
    corrections = corrections or {}
    kept: list[Annotation] = []
    for a in annfile.annotations:
        if a.frame_index in rejected:
            continue
        if a.frame_index in corrections:
            kept.append(corrections[a.frame_index])
            continue
        verified = a.frame_index in accepted or a.verified_by_human
        kept.append(Annotation.from_dict({**a.to_dict(), "verified_by_human": verified}))
    prov = {**annfile.provenance, "stage": Stage.REVIEWED.value}
    return AnnotationFile(provenance=prov, annotations=kept)


def build_app(annotation_path: Path, video_path: Path, output_path: Path | None = None):
    """Gradio review app: this sample's crop beside the last-accepted crop and the
    relay board (last move highlighted, side-to-move + game link shown), with an
    expandable ply window for corrections. Shortcuts: Q/E widen the window
    earlier/later, W accept, X discard, A/D prev/next, S save.

    Saves the confirmed result to ``output_path`` (default ``<video_id>.reviewed.json``
    next to the input), leaving the candidates file untouched. Every accept/reject
    autosaves (atomically), so a crash never loses more than the frame in hand; the Save
    button stays for an explicit checkpoint + confirmation message."""
    global gr
    if gr is None:
        try:
            import gradio
        except ImportError as e:
            raise ImportError("review UI needs gradio: `poetry install --with viz`") from e
        gr = gradio

    import json


    annfile = AnnotationFile.load(annotation_path)
    anns_all = bucket_sorted(annfile.annotations)
    anns = visible_for_review(anns_all)  # hide same-game dups (still reject-seeded below)
    n_hidden_dups = len(anns_all) - len(anns)
    prov = annfile.provenance
    gates = gate_thresholds(prov)
    video = probe(Path(video_path), prov["video_id"], prov["format_id"])
    reader = FrameReader(video)
    out_path = (
        Path(output_path)
        if output_path
        else Path(annotation_path).parent / (f"{prov['video_id']}.reviewed.json")
    )

    # relay timelines for this file's rounds, keyed like annotation.game_id, so we
    # can render neighbouring plies for corrections.
    timelines = timelines_by_game(prov.get("round_ids", []))

    decisions: dict[int, Verdict] = {}  # frame_index -> reviewer verdict
    corrections: dict[int, Annotation] = {}
    last_accepted: list[Annotation] = []  # most recently accepted (effective) annotation

    # Resume a partial review (over the FULL set incl. hidden dups, so their reject is
    # kept and saved); navigation/display below uses the dup-free `anns`.
    if out_path.exists():
        prior = {a.frame_index: a for a in AnnotationFile.load(out_path).annotations}
        resumed = resume_decisions(anns_all, prior)
        decisions.update(resumed.decisions)
        corrections.update(resumed.corrections)
    else:
        # Fresh cross-checked review: pre-seed the gate's verdict so the human only
        # adjudicates the review bucket; untouched defaults still save as decided.
        decisions.update(seed_decisions(anns_all))

    # Undecided frames come first (a fresh cross-checked review already leads with the
    # undecided review bucket, so only a resume actually moves anything — the decided
    # frames drop behind the pending ones instead of interleaving by timestamp).
    anns = pending_first(anns, decisions)
    # seed the "last accepted" reference so the comparison crop shows on resume
    accepted = [
        corrections.get(a.frame_index, a)
        for a in anns
        if decisions.get(a.frame_index) is Verdict.ACCEPT
    ]
    if accepted:
        last_accepted[:] = [accepted[-1]]

    hotkeys_js = (
        "<script>var M=" + json.dumps(_HOTKEYS) + ";document.addEventListener('keydown',"
        "function(e){var t=(document.activeElement||{}).tagName||'';"
        "if(t==='INPUT'||t==='TEXTAREA')return;var id=M[e.key.toLowerCase()];if(!id)return;"
        "var b=document.querySelector('#'+id+' button')||document.getElementById(id);"
        "if(b)b.click();});</script>"
    )

    def crop_bgr(a: Annotation):
        return Rect.from_list(a.crop_bbox).crop(reader.frame_at_index(a.frame_index))

    def neighbor_choices(a: Annotation, back: int = 2, fwd: int = 2) -> list[int]:
        tl = timelines.get(a.game_id)
        return [p for p, _ in neighbor_plies(tl, a.ply, back=back, fwd=fwd)] if tl else [a.ply]

    def verdict_color(fi: int) -> str:
        return {Verdict.ACCEPT: _GREEN, Verdict.REJECT: _RED}.get(decisions.get(fi), "#888")

    def panel_html(i: int, back: int = 2, fwd: int = 2) -> str:
        a = anns[i]
        shown = corrections.get(a.frame_index, a)  # reflect an accepted correction
        color = verdict_color(a.frame_index)
        def captioned_crop(ann: Annotation, *, border: str, width: int, label: str) -> str:
            mm, ss = divmod(ann.timestamp_s, 60)
            yt = youtube_url(ann.video_id, ann.timestamp_s)
            return (
                f'<div style="border:8px solid {border};padding:2px">'
                f'<img src="{crop_data_uri(crop_bgr(ann))}" style="max-width:{width}px"/>'
                f'<div style="font-size:11px;color:#888;text-align:center">'
                f"{label} · frame {ann.frame_index} · "
                f'<a href="{yt}" target="_blank">▶ {int(mm):d}:{ss:05.2f} on youtube</a>'
                "</div></div>"
            )

        # Side-by-side with the last accepted position so a static-board duplicate (only
        # the players' hands moved) is obvious. Only meaningful within the same game and
        # template/composition — a different shot is not a duplicate, so don't compare.
        CROP_W = 400
        prev_crop = ""
        p = last_accepted[-1] if last_accepted else None
        if (
            p is not None
            and p.frame_index != a.frame_index
            and p.game_id == a.game_id
            and p.template_id == a.template_id
        ):
            prev_crop = captioned_crop(
                p, border="#22c55e", width=CROP_W, label=f"last accepted · ply {p.ply}"
            )
        crop = captioned_crop(a, border=color, width=CROP_W, label="this sample")
        tag = f" → corrected to ply {shown.ply}" if shown is not a else ""
        tl = timelines.get(a.game_id)
        game_link = (
            f'<a href="{lichess_broadcast_url(a.round_id)}" target="_blank">📺 lichess broadcast</a>'
            f' · <a href="{lichess_url(shown.placement)}" target="_blank">analysis board</a>'
        )
        claimed = (
            f"<div>relay · ply {a.ply}{tag} · {a.white} vs {a.black}"
            f"<br><b>{side_to_move(shown.fen)} to move</b> · {game_link}"
            f"<br>{board_svg(shown.fen, 300, lastmove=last_move(tl, shown.ply))}</div>"
        )
        neigh = ""
        if tl:
            cells = "".join(
                f'<div style="text-align:center;font-size:12px">ply {p}'
                f'{" ◀current" if p == shown.ply else ""} · {side_to_move(fen)} to move'
                f"<br>{board_svg(fen, 150, lastmove=last_move(tl, p))}</div>"
                for p, fen in neighbor_plies(tl, a.ply, back=back, fwd=fwd)
            )
            span = neighbor_plies(tl, a.ply, back=back, fwd=fwd)
            edges = f"plies {span[0][0]}–{span[-1][0]}"
            neigh = (
                f'<div style="margin-top:10px;font-size:12px;color:#888">{edges} '
                "(use ⏮/⏭ below to widen)</div>"
                f'<div style="margin-top:4px;display:flex;flex-wrap:wrap;gap:8px">{cells}</div>'
            )
        row = f'<div style="display:flex;gap:16px;align-items:flex-start">{prev_crop}{crop}{claimed}</div>'
        return f"{row}{neigh}"

    def status(i: int) -> str:
        a = anns[i]
        d = decisions.get(a.frame_index)
        dcolor = {Verdict.ACCEPT: _GREEN, Verdict.REJECT: _RED}.get(d, _GRAY)
        badge = (
            f"<span style='background:{dcolor};color:#fff;padding:1px 8px;border-radius:4px;"
            f"font-weight:700'>{(d or 'undecided').upper()}</span>"
        )
        flag = " · ⚠needs review" if a.requires_review else ""
        cc = a.crosscheck
        cctag = ""
        if cc is not None:
            # fit/margin live in the color-coded bar above the crops; keep just the
            # bucket and any ply-shift here for context.
            shift = f" ⇄{cc.clock_ply}→{cc.chosen_ply}" if cc.chosen_ply != cc.clock_ply else ""
            cctag = f" · {cc.bucket.value}{shift}"
        return (
            f"{badge} · **{i+1}/{len(anns)}** · conf {a.confidence:.2f}{flag}{cctag} · "
            f"[lichess]({lichess_url(a.placement)})"
        )

    WINDOW_STEP = 4  # plies added to one side per "see more" press
    start = next((k for k, a in enumerate(anns) if a.frame_index not in decisions), 0)

    def in_bucket(a, v: str) -> bool:
        if v == "all":
            return True
        if v == "pending":  # verdict status, orthogonal to the cross-check buckets
            return a.frame_index not in decisions
        return bool(a.crosscheck) and a.crosscheck.bucket.value == v

    def bucket_count(v: str) -> int:
        return sum(1 for a in anns if in_bucket(a, v))

    def counts_markdown() -> str:
        """Header + live bucket tallies. Recomputed on demand so Save refreshes the
        pending count as frames get decided (the cross-check buckets are fixed)."""
        return (
            f"### Review {prov['video_id']} — {len(anns)} frame(s)"
            + (f" · {n_hidden_dups} same-game duplicate(s) hidden" if n_hidden_dups else "")
            + "\n\n"
            f"🕐 **pending {bucket_count('pending')}** · "
            f"🟡 review {bucket_count('review')} · 🟢 accept {bucket_count('accept')} · "
            f"🔴 quarantine {bucket_count('quarantine')}"
        )
    legend_md = legend_markdown(gates)

    # Strip the default block chrome (padding/border/min-height) off the fit/margin
    # strip and the crop panel so the bar sits tight against the crops instead of
    # floating in a tall empty block.
    css = (
        "#ccbar{padding:0!important;border:none!important;min-height:0!important;"
        "background:transparent!important}"
        "#ccbar>*{margin:0!important}"
        "#review-panel{padding:0!important;border:none!important;min-height:0!important}"
    )

    with gr.Blocks(title="SLCC annotation review", head=hotkeys_js, css=css) as demo:
        idx = gr.State(start)
        back = gr.State(2)  # plies shown before the claimed ply (grows on demand)
        fwd = gr.State(2)  # plies shown after the claimed ply
        md = gr.Markdown(status(start))  # per-frame verdict + fit/margin — kept at the very top
        counts = gr.Markdown(counts_markdown())
        view = gr.Radio(
            ["pending", "review", "accept", "quarantine", "all"],
            value="pending",
            label="jump to",
        )
        # the two automatic signals, color-coded, right above the crops
        ccbar = gr.HTML(crosscheck_bar_html(anns[start].crosscheck, gates.tau_margin), elem_id="ccbar")
        panel = gr.HTML(panel_html(start), elem_id="review-panel")
        correct_to = gr.Radio(
            neighbor_choices(anns[start]),
            value=anns[start].ply,
            label="correct to ply (Accept keeps current unless changed)",
        )
        with gr.Row():
            more_back = gr.Button(f"⏮ Earlier plies +{WINDOW_STEP} (Q)", elem_id="earlier-btn")
            more_fwd = gr.Button(f"⏭ Later plies +{WINDOW_STEP} (E)", elem_id="later-btn")
        with gr.Row():
            accept = gr.Button("Accept (W)", elem_id="accept-btn")
            reject = gr.Button("Discard (X)", elem_id="discard-btn")
        with gr.Row():
            prev_btn = gr.Button("◀ Prev image (A)", elem_id="prev-btn")
            next_btn = gr.Button("Next image ▶ (D)", elem_id="next-btn")
            save_btn = gr.Button("Save (S)", elem_id="save-btn")
        gr.Markdown(legend_md)  # reference, kept at the very bottom of the page

        def show(i, b, f):
            a = anns[i]
            return (
                panel_html(i, b, f),
                status(i),
                gr.update(choices=neighbor_choices(a, b, f), value=a.ply),
                crosscheck_bar_html(a.crosscheck, gates.tau_margin),
            )

        def write_out() -> None:
            """Persist current decisions/corrections to the reviewed file (atomic)."""
            accepted = {fi for fi, v in decisions.items() if v is Verdict.ACCEPT}
            rejected = {fi for fi, v in decisions.items() if v is Verdict.REJECT}
            apply_reviews(annfile, accepted, rejected, corrections).save(out_path)

        def do_accept(i, chosen_ply, b, f):
            a = anns[i]
            tl = timelines.get(a.game_id)
            if tl is not None and chosen_ply != a.ply:
                corrections[a.frame_index] = corrected_annotation(a, tl, int(chosen_ply))
            else:
                corrections.pop(a.frame_index, None)
            decisions[a.frame_index] = Verdict.ACCEPT
            last_accepted[:] = [corrections.get(a.frame_index, a)]
            write_out()  # autosave every decision -> a crash costs at most the open frame
            return panel_html(i, b, f), status(i), counts_markdown()

        def do_reject(i, b, f):
            a = anns[i]
            corrections.pop(a.frame_index, None)
            decisions[a.frame_index] = Verdict.REJECT
            write_out()  # autosave every decision -> a crash costs at most the open frame
            return panel_html(i, b, f), status(i), counts_markdown()

        def step(i, delta):
            j = max(0, min(len(anns) - 1, i + delta))
            return (j, 2, 2, *show(j, 2, 2))  # fresh ±2 window per annotation

        def jump(v):
            # jump to the first frame of the chosen group. "pending" reads the live
            # decisions, so it always lands on the next still-undecided frame — the
            # way back after wandering into the already-decided tail.
            members = [k for k, a in enumerate(anns) if in_bucket(a, v)]
            j = members[0] if members else 0
            return (j, 2, 2, *show(j, 2, 2))

        def widen(i, b, f, chosen, side):
            a = anns[i]
            b2, f2 = (b + WINDOW_STEP, f) if side < 0 else (b, f + WINDOW_STEP)
            keep = int(chosen) if chosen is not None else a.ply
            return (
                b2,
                f2,
                panel_html(i, b2, f2),
                gr.update(choices=neighbor_choices(a, b2, f2), value=keep),
            )

        def save():
            write_out()
            accepted = sum(v is Verdict.ACCEPT for v in decisions.values())
            rejected = sum(v is Verdict.REJECT for v in decisions.values())
            msg = (
                f"saved: {accepted} accepted ({len(corrections)} corrected), "
                f"{rejected} rejected -> {out_path}"
            )
            return msg, counts_markdown()  # refresh the pending/bucket tallies too

        accept.click(do_accept, [idx, correct_to, back, fwd], [panel, md, counts])
        reject.click(do_reject, [idx, back, fwd], [panel, md, counts])
        prev_btn.click(lambda i: step(i, -1), [idx], [idx, back, fwd, panel, md, correct_to, ccbar])
        next_btn.click(lambda i: step(i, +1), [idx], [idx, back, fwd, panel, md, correct_to, ccbar])
        view.change(jump, [view], [idx, back, fwd, panel, md, correct_to, ccbar])
        more_back.click(
            lambda i, b, f, c: widen(i, b, f, c, -1),
            [idx, back, fwd, correct_to],
            [back, fwd, panel, correct_to],
        )
        more_fwd.click(
            lambda i, b, f, c: widen(i, b, f, c, +1),
            [idx, back, fwd, correct_to],
            [back, fwd, panel, correct_to],
        )
        save_btn.click(save, [], [md, counts])

    return demo


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Review SLCC annotation candidates")
    parser.add_argument("--annotations", required=True, type=Path, help="a .candidates.json file")
    parser.add_argument("--video-path", required=True, type=Path)
    parser.add_argument(
        "--out", type=Path, default=None, help="default <video-id>.reviewed.json next to input"
    )
    args = parser.parse_args(argv)
    build_app(args.annotations, args.video_path, args.out).launch()


if __name__ == "__main__":
    main()
