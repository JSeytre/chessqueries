"""Content-pin checks on produced SLCC annotations (no video/network).

Mirrors tests/test_cvchess_labels.py: anchor every label to the relay position it
claims, so a coherent global ply shift (the CVChess off-by-one) is caught.
"""

import numpy as np
import pytest

from chessqueries.annotate.pipeline import game_id
from chessqueries.annotate.reconstruction import ReconstructionRecord, stable_sample_id
from chessqueries.annotate.templates import Rect
from chessqueries.annotate.relay import parse_round_pgn
from chessqueries.annotate.schema import Annotation, AnnotationFile, Source
from chessqueries.annotate.validate import check_annotations, chronology_outliers
from chessqueries.core import Color, Split

PGN = """[Event "T"]
[White "Niemann, Hans Moke"]
[Black "So, Wesley"]
[Result "*"]
[TimeControl "25+10"]

1. e4 {[%clk 0:25:00]} e5 {[%clk 0:24:40]} 2. Nf3 {[%clk 0:24:30]} Nc6 {[%clk 0:24:20]} *
"""
ROUND = "rTEST"


def _ann_at(tl, ply: int, frame_index: int, *, placement: str | None = None) -> Annotation:
    pos = tl.position_at(ply)
    return Annotation(
        video_id="vid",
        frame_index=frame_index,
        timestamp_s=frame_index / 30,
        template_id="slcc_t0",
        crop_bbox=[10, 20, 800, 800],
        game_id=game_id(ROUND, tl),
        round_id=ROUND,
        ply=ply,
        fen=pos.fen,
        placement=placement or pos.placement,
        side_to_move="w" if pos.turn == Color.WHITE else "b",
        white=tl.white,
        black=tl.black,
        white_clk_s=pos.white_clk_s,
        black_clk_s=pos.black_clk_s,
        confidence=0.9,
    )


def _build(timelines):
    tl = timelines[0]
    anns = [_ann_at(tl, p, frame_index=p * 100) for p in range(len(tl.positions))]
    return AnnotationFile(provenance={"round_id": ROUND}, annotations=anns), {
        game_id(ROUND, tl): tl
    }


def test_clean_annotations_have_no_problems():
    file, by_game = _build(parse_round_pgn(PGN))
    assert check_annotations(file, by_game) == []


def test_off_by_one_placement_is_caught():
    """Claim ply p but show ply p+1's placement -> flagged (the CVChess failure)."""
    tl = parse_round_pgn(PGN)[0]
    bad = AnnotationFile(
        provenance={},
        annotations=[_ann_at(tl, 1, 100, placement=tl.position_at(2).placement)],
    )
    problems = check_annotations(bad, {game_id(ROUND, tl): tl})
    assert any("placement" in p for p in problems)


def test_wrong_side_to_move_is_caught():
    tl = parse_round_pgn(PGN)[0]
    a = _ann_at(tl, 1, 100)
    flipped = Annotation.from_dict(
        {**a.to_dict(), "side_to_move": "w" if a.side_to_move == "b" else "b"}
    )
    problems = check_annotations(AnnotationFile({}, [flipped]), {game_id(ROUND, tl): tl})
    assert any("side_to_move" in p for p in problems)


def test_annotation_validation():
    with pytest.raises(ValueError):  # bad crop_bbox (len 3)
        Annotation(
            "v",
            0,
            0.0,
            "t",
            [1, 2, 3],
            "g",
            "r",
            0,
            "f",
            "8/8/8/8/8/8/8/8",
            "w",
            "W",
            "B",
            1,
            1,
            0.5,
        )
    with pytest.raises(ValueError):  # bad placement
        Annotation(
            "v", 0, 0.0, "t", [1, 2, 3, 4], "g", "r", 0, "f", "bad", "w", "W", "B", 1, 1, 0.5
        )


def test_annotation_file_roundtrip(tmp_path):
    file, _ = _build(parse_round_pgn(PGN))
    path = tmp_path / "ann.json"
    file.save(path)
    back = AnnotationFile.load(path)
    assert len(back.annotations) == len(file.annotations)
    assert back.annotations[0] == file.annotations[0]


def test_source_defaults_and_serializes(tmp_path):
    tl = parse_round_pgn(PGN)[0]
    a = _ann_at(tl, 1, 100)
    assert a.source == Source.CLOCK_OCR  # default
    assert a.to_dict()["source"] == "clock_ocr"  # serialized as the value string
    # a model-retrieval label round-trips and stays distinguishable.
    m = Annotation.from_dict({**a.to_dict(), "source": "model_retrieval"})
    assert m.source == Source.MODEL_RETRIEVAL
    file = AnnotationFile(provenance={}, annotations=[a, m])
    file.save(tmp_path / "f.json")
    back = AnnotationFile.load(tmp_path / "f.json")
    assert [x.source for x in back.annotations] == [Source.CLOCK_OCR, Source.MODEL_RETRIEVAL]


def _ann_rt(round_id: str, t: float, conf: float = 0.9) -> Annotation:
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    return Annotation(
        video_id="v",
        frame_index=int(t * 30),
        timestamp_s=t,
        template_id="t",
        crop_bbox=[0, 0, 8, 8],
        game_id=f"{round_id}-w-b",
        round_id=round_id,
        ply=0,
        fen=start + " w - - 0 1",
        placement=start,
        side_to_move="w",
        white="W",
        black="B",
        white_clk_s=1,
        black_clk_s=1,
        confidence=conf,
    )


def test_chronology_flags_round_time_mismatch():
    anns = [
        _ann_rt("r1", 10.0),
        _ann_rt("r1", 20.0),
        _ann_rt("r1", 30.0),  # round 1 window 10..30
        _ann_rt("r2", 110.0),
        _ann_rt("r2", 120.0),  # round 2 window 110..120
        _ann_rt("r2", 20.0),  # outlier: labeled r2 but sits inside r1's window
    ]
    outliers = chronology_outliers(anns)
    assert outliers == [5]


def test_reconstruct_helpers_are_pure():
    frame = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    assert Rect.from_list([10, 20, 30, 40]).crop(frame).shape == (40, 30, 3)
    tl = parse_round_pgn(PGN)[0]
    annotation = _ann_at(tl, 2, 200)
    rec = ReconstructionRecord(stable_sample_id(annotation), Split.TEST, annotation).loader_record()
    assert rec["image"] == "images/vid_200.jpg"
    assert rec["gt_fen"] == tl.position_at(2).fen
    assert rec["players"] == [tl.white, tl.black]


def test_annotation_rejects_malformed_placement():
    """Placement is validated through Board.from_fen — a 64-square total with
    mis-shaped ranks (9+7) must be rejected, not just a wrong rank count."""
    import pytest

    tl = parse_round_pgn(PGN)[0]
    good = _ann_at(tl, 0, 1)
    with pytest.raises(ValueError):
        _ann_at(tl, 0, 1, placement="ppppppppp/7/8/8/8/8/8/8")
    assert good.placement.count("/") == 7


def test_annotation_confidence_must_be_a_probability():
    import pytest

    tl = parse_round_pgn(PGN)[0]
    with pytest.raises(ValueError, match="confidence"):
        _ann_at(tl, 0, 1).__class__(**{**_ann_at(tl, 0, 1).to_dict(), "confidence": 1.5})
