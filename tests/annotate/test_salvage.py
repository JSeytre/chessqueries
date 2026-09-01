"""`annotate produce --salvage`: re-produce an already-reviewed video with the current
registry, keep every prior human decision, and surface for review ONLY the frames the
better matching newly rescued. The pure merge (`merge_salvage`) carries the logic; the
`produce_one` half needs video+OCR and is exercised end-to-end elsewhere (kept out here,
matching the rest of the suite)."""

from annotate_fixtures import make_annotation

from chessqueries.annotate.pipeline import merge_salvage, salvage_key
from chessqueries.annotate.review import resume_decisions
from chessqueries.annotate.schema import AnnotationFile


def _ann(fi, ply, *, verified=False, game="g", vid="v"):
    return make_annotation(video_id=vid, frame_index=fi, timestamp_s=float(fi),
                           game_id=game, ply=ply, verified_by_human=verified)


def _file(anns, stage="candidates"):
    return AnnotationFile(provenance={"stage": stage}, annotations=anns)


def test_salvage_key_is_video_game_ply():
    a = _ann(10, 3, game="rd-a-b", vid="vid00000001")
    assert salvage_key(a) == ("vid00000001", "rd-a-b", 3)


def test_merge_surfaces_only_new_frames_verified_kept_drift_robust():
    # Old run captured plies 0,1,2 (frames 10,11,12); review kept 0 & 2, rejected 1.
    old = _file([_ann(10, 0), _ann(11, 1), _ann(12, 2)])
    reviewed = _file([_ann(10, 0, verified=True), _ann(12, 2, verified=True)], stage="reviewed")
    # Fresh run: same ply-0 shot but its frame drifted (10 -> 13); the rejected ply-1 shot
    # also re-appears drifted (11 -> 14); plus two genuinely-new salvaged plies (5, 6).
    fresh = _file([_ann(13, 0), _ann(14, 1), _ann(20, 5), _ann(21, 6)])

    merged = merge_salvage(old, reviewed, fresh, log=lambda *_: None)
    candidates, preseed = merged.candidates, merged.reviewed

    assert candidates.provenance["n_salvaged_new"] == 2
    assert candidates.provenance["n_verified_kept"] == 2
    assert candidates.provenance["salvage"] is True
    # Kept verified frames (verbatim) + the two new plies; drift-recaptured & rejected out.
    assert {(a.frame_index, a.ply) for a in candidates.annotations} == {
        (10, 0), (12, 2), (20, 5), (21, 6)
    }
    # Candidates substrate carries no verdicts; the pre-seeded reviewed file does.
    assert all(not a.verified_by_human for a in candidates.annotations)
    verified = {a.frame_index for a in preseed.annotations if a.verified_by_human}
    assert verified == {10, 12}


def test_preseeded_review_resumes_onto_only_the_new():
    """The write-out the workflow does — candidates + pre-seeded reviewed — makes
    `review`'s resume path accept the prior-verified frames and leave the salvaged ones
    undecided (surfaced), touching nothing the human already judged."""
    old = _file([_ann(10, 0), _ann(12, 2)])
    reviewed = _file([_ann(10, 0, verified=True), _ann(12, 2, verified=True)], stage="reviewed")
    fresh = _file([_ann(20, 5), _ann(21, 6)])

    merged = merge_salvage(old, reviewed, fresh, log=lambda *_: None)
    candidates, preseed = merged.candidates, merged.reviewed
    prior = {a.frame_index: a for a in preseed.annotations}
    resumed = resume_decisions(candidates.annotations, prior)

    # verified auto-accepted, not shown
    assert resumed.decisions == {10: "accept", 12: "accept"}
    surfaced = [a.frame_index for a in candidates.annotations
                if a.frame_index not in resumed.decisions]
    assert surfaced == [20, 21]  # only the newly-salvaged frames need a human call
    assert resumed.corrections == {}


def test_merge_keeps_human_ply_correction_verbatim():
    """A verified frame whose ply the human corrected is kept at the corrected ply, and
    the fresh run's original-ply candidate for that shot is not re-surfaced."""
    old = _file([_ann(10, 4)])  # clock-OCR originally read ply 4
    reviewed = _file([_ann(10, 7, verified=True)], stage="reviewed")  # human corrected 4 -> 7
    fresh = _file([_ann(10, 4)])  # re-produce reads the same original ply 4 again

    candidates = merge_salvage(old, reviewed, fresh, log=lambda *_: None).candidates

    assert candidates.provenance["n_salvaged_new"] == 0
    assert [(a.frame_index, a.ply) for a in candidates.annotations] == [(10, 7)]


def test_merge_skips_frameindex_collision_against_kept_frame():
    old = _file([_ann(10, 0)])
    reviewed = _file([_ann(10, 0, verified=True)], stage="reviewed")
    # A fresh candidate with a NEW key but the SAME frame index as the kept frame 10.
    fresh = _file([_ann(10, 9)])
    logged: list[str] = []

    candidates = merge_salvage(old, reviewed, fresh, log=logged.append).candidates

    assert candidates.provenance["n_salvaged_new"] == 0
    assert [(a.frame_index, a.ply) for a in candidates.annotations] == [(10, 0)]
    assert any("collides" in m for m in logged)
