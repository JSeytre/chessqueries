"""Freeze pins for the released SLCC manifest (``data/slcc/dataset/annotations.json``).

The manifest lives under the gitignored ``data/`` tree, so these tests **skip**
when it isn't present (e.g. CI without the dataset) and act as a drift guard for
anyone who does have it. Bumping the dataset is a deliberate act: regenerate the
manifest, confirm the new numbers, then update the constants below in the same
commit.
"""
import collections
import hashlib
import json

import chess
import pytest

from chessqueries.core import NUM_PIECES, Board
from chessqueries.data.slcc import SLCC

# --- Frozen release identity (SLCC v1) -------------------------------------
SHA256 = "057f247ae92b134ca2b172317335919df01b22cfaa7472ddaf53393c2515ab75"
N_SAMPLES = 2174
SPLIT_COUNTS = {"train": 1475, "val": 326, "test": 373}
N_GAMES = 152
N_VIDEOS = 20


@pytest.fixture(scope="module")
def manifest():
    path = SLCC().manifest_path
    if not path.is_file():
        pytest.skip(f"SLCC manifest not present at {path} (data/ is local-only)")
    return path


def test_manifest_checksum_is_frozen(manifest):
    """Byte-exact pin of the released file — the release identifier."""
    got = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert got == SHA256, (
        f"SLCC manifest checksum changed ({got}). If this is an intentional "
        f"dataset bump, update SHA256 and the counts in this file."
    )


def test_sample_and_split_counts(manifest):
    samples = json.loads(manifest.read_text())["samples"]
    assert len(samples) == N_SAMPLES
    counts = collections.Counter(s["split"] for s in samples)
    assert dict(counts) == SPLIT_COUNTS
    assert len({s["video_id"] for s in samples}) == N_VIDEOS
    assert len({s["game_id"] for s in samples}) == N_GAMES


def test_splits_are_group_disjoint_by_game(manifest):
    """No game may straddle splits — the leakage guarantee for the benchmark."""
    samples = json.loads(manifest.read_text())["samples"]
    by_game = collections.defaultdict(set)
    for s in samples:
        by_game[s["game_id"]].add(s["split"])
    straddling = {g for g, sp in by_game.items() if len(sp) > 1}
    assert not straddling, f"{len(straddling)} game(s) leak across splits: {straddling}"


def test_every_fen_is_legal_and_human_verified(manifest):
    samples = json.loads(manifest.read_text())["samples"]
    for s in samples:
        chess.Board(s["gt_fen"])  # full FEN is legal
        labels = Board.from_fen(s["gt_fen"]).labels
        assert len(labels) == 64 and all(0 <= c < NUM_PIECES for c in labels)
        assert s["verified_by_human"] is True
