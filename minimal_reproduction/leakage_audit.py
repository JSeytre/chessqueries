"""Audit split isolation and frozen data identities for the paper experiment."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from data import (
    CHESSRED_ANNOTATIONS_SHA256,
    CVCHESS_ANNOTATIONS_SHA256,
    DATA_ROOT,
    SLCC_MANIFEST_SHA256,
    TRAIN_SOURCES,
)


failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_hashes(paths: list[Path], label: str) -> set[str]:
    print(f"HASH  {label}: {len(paths)} images")
    return {file_sha256(path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hash-images",
        action="store_true",
        help="Also rule out exact-byte image duplicates across train/val and test.",
    )
    args = parser.parse_args()

    # 1. ChessReD: official ID splits and frozen annotation identity.
    annotation_path = DATA_ROOT / "chessred" / "annotations.json"
    annotation_sha = file_sha256(annotation_path)
    check(
        "chessred annotations match the frozen checksum",
        annotation_sha == CHESSRED_ANNOTATIONS_SHA256,
        annotation_sha,
    )
    annotations = json.loads(annotation_path.read_text())
    chessred_ids = {
        split: set(annotations["splits"][split]["image_ids"])
        for split in ("train", "val", "test")
    }
    check(
        "chessred split sizes are 6479/2192/2129",
        tuple(len(chessred_ids[split]) for split in ("train", "val", "test"))
        == (6_479, 2_192, 2_129),
    )
    check(
        "chessred train/val/test IDs are pairwise disjoint",
        not chessred_ids["train"] & chessred_ids["val"]
        and not chessred_ids["train"] & chessred_ids["test"]
        and not chessred_ids["val"] & chessred_ids["test"],
    )
    chessred_path_by_id = {
        image["id"]: DATA_ROOT / "chessred" / image["path"]
        for image in annotations["images"]
    }
    chessred_images = {
        split: [chessred_path_by_id[image_id] for image_id in chessred_ids[split]]
        for split in chessred_ids
    }

    # 2. ChessCog: directory-defined official splits.
    chesscog_annotations = {
        split: sorted((DATA_ROOT / "chesscog" / "render" / split).glob("*.json"))
        for split in ("train", "val", "test")
    }
    chesscog_stems = {
        split: {path.stem for path in paths}
        for split, paths in chesscog_annotations.items()
    }
    check(
        "chesscog split sizes are 4400/146/342",
        tuple(len(chesscog_stems[split]) for split in ("train", "val", "test"))
        == (4_400, 146, 342),
    )
    check(
        "chesscog train/val/test stems are pairwise disjoint",
        not chesscog_stems["train"] & chesscog_stems["val"]
        and not chesscog_stems["train"] & chesscog_stems["test"]
        and not chesscog_stems["val"] & chesscog_stems["test"],
    )
    chesscog_images = {
        split: [path.with_suffix(".png") for path in paths]
        for split, paths in chesscog_annotations.items()
    }

    # 3. SLCC: immutable manifest, expected counts, and game-level isolation.
    slcc_root = DATA_ROOT / "slcc" / "dataset"
    manifest_path = slcc_root / "annotations.json"
    manifest_sha = file_sha256(manifest_path)
    check(
        "slcc manifest matches the pre-run frozen-v1 checksum",
        manifest_sha == SLCC_MANIFEST_SHA256,
        manifest_sha,
    )
    samples = json.loads(manifest_path.read_text())["samples"]
    split_counts = Counter(sample["split"] for sample in samples)
    check(
        "slcc split sizes are 1475/326/373",
        tuple(split_counts[split] for split in ("train", "val", "test"))
        == (1_475, 326, 373),
    )
    game_splits: dict[str, set[str]] = {}
    for sample in samples:
        game_splits.setdefault(sample["game_id"], set()).add(sample["split"])
    straddling_games = {
        game_id for game_id, splits in game_splits.items() if len(splits) > 1
    }
    check(
        "slcc no game straddles splits",
        not straddling_games,
        f"{len(game_splits)} games",
    )
    slcc_images = {
        split: [
            slcc_root / sample["image"]
            for sample in samples
            if sample["split"] == split
        ]
        for split in ("train", "val", "test")
    }
    check(
        "slcc train/val/test image paths are pairwise disjoint",
        not set(slcc_images["train"]) & set(slcc_images["val"])
        and not set(slcc_images["train"]) & set(slcc_images["test"])
        and not set(slcc_images["val"]) & set(slcc_images["test"]),
    )

    # 4. CVChess: corrected, frozen labels and structural zero-shot boundary.
    cvchess_annotation_path = DATA_ROOT / "cvchess" / "annotations.json"
    cvchess_sha = file_sha256(cvchess_annotation_path)
    check(
        "cvchess annotations match the frozen checksum",
        cvchess_sha == CVCHESS_ANNOTATIONS_SHA256,
        cvchess_sha,
    )
    cvchess = json.loads(cvchess_annotation_path.read_text())
    check("cvchess contains 352 labelled images", len(cvchess) == 352, str(len(cvchess)))
    check(
        "training sources are exactly chessred/chesscog/slcc (no cvchess)",
        tuple(load.__name__ for load in TRAIN_SOURCES)
        == ("load_chessred", "load_chesscog", "load_slcc"),
    )
    cvchess_images = [
        DATA_ROOT / "cvchess" / "images" / entry["image"] for entry in cvchess
    ]

    all_images = [
        *chessred_images["train"],
        *chessred_images["val"],
        *chessred_images["test"],
        *chesscog_images["train"],
        *chesscog_images["val"],
        *chesscog_images["test"],
        *slcc_images["train"],
        *slcc_images["val"],
        *slcc_images["test"],
        *cvchess_images,
    ]
    missing = [path for path in all_images if not path.is_file()]
    check(
        "all 18214 referenced images exist",
        not missing and len(all_images) == 18_214,
        f"missing={len(missing)}",
    )

    if args.hash_images and not missing:
        # The training-influencing side includes train and validation because
        # validation selects the checkpoint. The held-out side is every test.
        influential = [
            *chessred_images["train"],
            *chessred_images["val"],
            *chesscog_images["train"],
            *chesscog_images["val"],
            *slcc_images["train"],
            *slcc_images["val"],
        ]
        held_out = [
            *chessred_images["test"],
            *chesscog_images["test"],
            *slcc_images["test"],
            *cvchess_images,
        ]
        influential_hashes = image_hashes(influential, "train+validation")
        held_out_hashes = image_hashes(held_out, "all held-out tests")
        overlap = influential_hashes & held_out_hashes
        check(
            "no exact-byte image duplicate crosses into a held-out test",
            not overlap,
            f"overlap={len(overlap)}",
        )

    print()
    if failures:
        raise SystemExit(f"{len(failures)} check(s) FAILED: {failures}")
    print("all leakage checks passed")


if __name__ == "__main__":
    main()
