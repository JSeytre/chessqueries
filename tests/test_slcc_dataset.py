"""SLCC loader + split planner (no pixels/network)."""
import json

import pytest

from chessqueries.core import Split
from chessqueries.data import DatasetName, get_dataset
from chessqueries.data.base import DATASET_REGISTRY, DatasetIncompleteError
from chessqueries.data.slcc import MANIFEST_NAME, SLCC, SplitRatio, plan_splits

FEN = "3b4/3B2k1/5pb1/p1pN3p/P1P1P2P/1P3K2/8/8 b - - 2 65"


def _rec(
    sid: str, split: str, video_id: str = "vidA", game_id: str = "g"
) -> dict:
    return {"image": f"images/{sid}.jpg", "gt_fen": FEN, "split": split,
            "video_id": video_id, "game_id": game_id, "ply": 1, "side_to_move": "b",
            "players": ["W", "B"], "template_id": "t", "confidence": 0.9,
            "verified_by_human": True}


def _write(root, recs):
    (root).mkdir(parents=True, exist_ok=True)
    (root / MANIFEST_NAME).write_text(json.dumps({"version": "v1", "samples": recs}))
    for record in recs:
        image = root / record["image"]
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"synthetic image")


def test_registered():
    assert DATASET_REGISTRY[DatasetName.SLCC] is SLCC
    assert get_dataset(DatasetName.SLCC).splits == (Split.TRAIN, Split.VAL, Split.TEST)


def test_loader_filters_by_split(tmp_path):
    root = tmp_path / "dataset"
    _write(root, [_rec("a", "train"), _rec("b", "train"), _rec("c", "val")])
    ds = SLCC(root)
    assert len(ds.load_samples(Split.TRAIN, allow_partial=True)) == 2
    val = ds.load_samples(Split.VAL, allow_partial=True)
    assert len(val) == 1
    s = val[0]
    assert s.dataset is DatasetName.SLCC and s.split is Split.VAL
    assert s.sample_id == "c"
    assert s.board.placement == FEN.split()[0]  # placement round-trips through Board
    assert s.meta["verified_by_human"] is True


def test_complete_requested_split_does_not_hide_partial_full_reconstruction(tmp_path):
    root = tmp_path / "dataset"
    _write(root, [_rec(f"test-{index}", "test") for index in range(373)])
    dataset = SLCC(root)

    with pytest.raises(DatasetIncompleteError, match="full split counts"):
        dataset.load_samples(Split.TEST)

    loaded = dataset.load_with_report(Split.TEST, allow_partial=True)
    assert len(loaded.samples) == 373
    assert loaded.completeness.structural_issues


def test_manifest_rejects_a_game_that_crosses_splits(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    _write(
        root,
        [
            _rec("train", "train", game_id="shared"),
            _rec("val", "val", game_id="val-game"),
            _rec("test", "test", game_id="shared"),
        ],
    )
    monkeypatch.setattr(
        SLCC,
        "expected_samples",
        {Split.TRAIN: 1, Split.VAL: 1, Split.TEST: 1},
    )

    with pytest.raises(DatasetIncompleteError, match="game IDs cross splits"):
        SLCC(root).load_samples(Split.TEST)


def test_manifest_reports_duplicate_sample_and_image_identities(tmp_path):
    root = tmp_path / "dataset"
    _write(
        root,
        [
            _rec("same", "train", game_id="train-game"),
            _rec("same", "val", game_id="val-game"),
        ],
    )

    issues = SLCC(root)._structural_issues()

    assert any("duplicate sample IDs" in issue for issue in issues)
    assert any("duplicate image paths" in issue for issue in issues)


def test_split_ratio_parse_and_validate():
    assert SplitRatio.from_str("60/30/10") == SplitRatio(60, 30, 10)
    for bad in ("60/40", "50/50/10", "a/b/c", "60/30/-10"):
        with pytest.raises(ValueError):
            SplitRatio.from_str(bad)


def test_plan_splits_is_deterministic_and_honors_ratio():
    groups = {f"s{i}": f"s{i}" for i in range(100)}  # each frame its own group
    r = SplitRatio(60, 30, 10)
    a = plan_splits(groups, ratio=r, seed=0)
    assert a == plan_splits(groups, ratio=r, seed=0)
    counts = {s: sum(v == s for v in a.values()) for s in ("train", "val", "test")}
    assert counts == {"train": 60, "val": 30, "test": 10}
    assert plan_splits(groups, ratio=r, seed=1) != a  # seed actually matters


def test_plan_splits_is_append_only():
    groups = {f"s{i}": f"s{i}" for i in range(10)}
    first = plan_splits(groups, ratio=SplitRatio(80, 20, 0), seed=0)
    # Grow: add 5 new singleton groups; old assignments must not move, new -> test.
    grown = plan_splits({**groups, **{f"t{i}": f"t{i}" for i in range(5)}},
                        ratio=SplitRatio(0, 0, 100), seed=0, existing=first)
    assert all(grown[k] == first[k] for k in groups)  # nothing reshuffled
    assert all(grown[f"t{i}"] == "test" for i in range(5))


def test_plan_splits_keeps_groups_whole():
    # Four games of differing size; no game may straddle two splits.
    games = {"g_a": 4, "g_b": 5, "g_c": 6, "g_d": 16}
    groups = {f"{g}_{i}": g for g, n in games.items() for i in range(n)}
    out = plan_splits(groups, ratio=SplitRatio(60, 20, 20), seed=0)
    for g in games:
        splits = {out[f"{g}_{i}"] for i in range(games[g])}
        assert len(splits) == 1, f"game {g} leaked across {splits}"


def test_plan_splits_stratifies_by_group_size():
    # 15 games each of 3 / 8 / 50 frames; an even three-way split must put 5 games of
    # EACH size in every split — not pile all the big (or small) games into one split.
    groups, gid = {}, 0
    for size, n in ((3, 15), (8, 15), (50, 15)):
        for _ in range(n):
            for i in range(size):
                groups[f"g{gid}_{i}"] = f"g{gid}"
            gid += 1
    out = plan_splits(groups, ratio=SplitRatio(34, 33, 33), seed=0)
    game_split, game_size = {}, {}
    for sid, sp in out.items():
        game_split[groups[sid]] = sp
        game_size[groups[sid]] = game_size.get(groups[sid], 0) + 1
    per = {}
    for g, sp in game_split.items():
        per[(sp, game_size[g])] = per.get((sp, game_size[g]), 0) + 1
    for size in (3, 8, 50):
        for sp in ("train", "val", "test"):
            assert per.get((sp, size), 0) == 5, f"{sp}/{size}f = {per.get((sp, size), 0)}, want 5"


def test_plan_splits_append_only_keeps_a_group_together():
    # A game already in TRAIN must keep its *new* frames in TRAIN too (no leak on growth).
    groups = {f"g_a_{i}": "g_a" for i in range(3)}
    first = plan_splits(groups, ratio=SplitRatio(100, 0, 0), seed=0)
    grown = plan_splits({**groups, "g_a_3": "g_a", "g_b_0": "g_b"},
                        ratio=SplitRatio(0, 0, 100), seed=0, existing=first)
    assert grown["g_a_3"] == "train"   # joins its game, not the new-frame ratio
    assert grown["g_b_0"] == "test"    # genuinely new group follows the ratio
