"""Transactional core for rebuilding loader-ready SLCC crops from annotations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

import cv2
import numpy as np

from chessqueries.annotate.schema import Annotation
from chessqueries.core import Split

MANIFEST_NAME = "annotations.json"
REPORT_NAME = "reconstruction-report.json"
RESUME_MANIFEST_NAME = "resume.json"
PIXEL_ADVISORY_NOTE = (
    "Pixel content is advisory: OpenCV/FFmpeg decoder builds can produce harmless "
    "pixel differences. Structural checks cover IDs, splits, crop dimensions, and files."
)


class ReconstructionError(RuntimeError):
    """Raised when a reconstruction cannot be validated or committed safely."""


def stable_sample_id(annotation: Annotation) -> str:
    """Return the public, decoder-independent identity of an annotated frame."""
    return f"{annotation.video_id}_{annotation.frame_index}"


@dataclass(frozen=True)
class ReconstructionRecord:
    """One released annotation paired with its canonical split and sample identity."""

    sample_id: str
    split: Split
    annotation: Annotation

    def __post_init__(self) -> None:
        if not isinstance(self.split, Split):
            object.__setattr__(self, "split", Split(self.split))
        expected_id = stable_sample_id(self.annotation)
        if self.sample_id != expected_id:
            raise ValueError(
                f"sample_id {self.sample_id!r} does not match annotation identity {expected_id!r}"
            )
        _, _, width, height = self.annotation.crop_bbox
        if width <= 0 or height <= 0:
            raise ValueError(
                f"sample {self.sample_id} has a non-positive crop size {width}x{height}"
            )

    @property
    def image(self) -> str:
        return f"images/{self.sample_id}.jpg"

    @property
    def crop_size(self) -> tuple[int, int]:
        return self.annotation.crop_bbox[2], self.annotation.crop_bbox[3]

    @property
    def fingerprint(self) -> str:
        """Pin every structural input that determines this record and crop."""
        payload = json.dumps(
            {"split": self.split.value, "annotation": self.annotation.to_dict()},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def loader_record(self) -> dict:
        """Serialize all consumer-facing label and provenance fields."""
        annotation = self.annotation
        return {
            "sample_id": self.sample_id,
            "image": self.image,
            "gt_fen": annotation.fen,
            "video_id": annotation.video_id,
            "frame_index": annotation.frame_index,
            "timestamp_s": annotation.timestamp_s,
            "game_id": annotation.game_id,
            "round_id": annotation.round_id,
            "ply": annotation.ply,
            "side_to_move": annotation.side_to_move,
            "players": [annotation.white, annotation.black],
            "template_id": annotation.template_id,
            "crop_bbox": annotation.crop_bbox,
            "confidence": annotation.confidence,
            "source": annotation.source.value,
            "requires_review": annotation.requires_review,
            "verified_by_human": annotation.verified_by_human,
            "split": self.split.value,
            "reconstruction_fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class UnavailableVideo:
    video_id: str
    reason: str


@dataclass(frozen=True)
class AdvisoryMismatch:
    sample_id: str
    reason: str


@dataclass(frozen=True)
class ReconstructionReport:
    """Structured outcome, separating structural gaps from pixel-level advisories."""

    committed: bool
    partial: bool
    expected_sample_ids: tuple[str, ...]
    complete_sample_ids: tuple[str, ...]
    missing_sample_ids: tuple[str, ...]
    reused_sample_ids: tuple[str, ...]
    written_sample_ids: tuple[str, ...]
    unavailable_videos: tuple[UnavailableVideo, ...]
    advisory_mismatches: tuple[AdvisoryMismatch, ...]
    manifest_path: Path

    def to_dict(self, *, include_complete_ids: bool = True) -> dict:
        if self.committed:
            status = "partial" if self.partial else "complete"
        else:
            status = "not_committed"
        structural = {
            "expected_count": len(self.expected_sample_ids),
            "complete_count": len(self.complete_sample_ids),
            "missing_count": len(self.missing_sample_ids),
            "missing_sample_ids": list(self.missing_sample_ids),
            "reused_count": len(self.reused_sample_ids),
            "written_count": len(self.written_sample_ids),
        }
        if include_complete_ids:
            structural["complete_sample_ids"] = list(self.complete_sample_ids)
        return {
            "status": status,
            "committed": self.committed,
            "partial": self.partial,
            "manifest": str(self.manifest_path),
            "structural": structural,
            "unavailable_videos": [asdict(item) for item in self.unavailable_videos],
            "pixel_content": {
                "status": "advisory",
                "note": PIXEL_ADVISORY_NOTE,
                "mismatches": [asdict(item) for item in self.advisory_mismatches],
            },
        }


def loader_sample_id(record: dict) -> str:
    """Return a loader record's explicit or image-derived sample identity."""
    sample_id = record.get("sample_id")
    if sample_id:
        return str(sample_id)
    image = record.get("image")
    if not image:
        raise ReconstructionError("loader record has neither sample_id nor image")
    return Path(image).stem


def _safe_image_path(root: Path, image: str) -> Path:
    rel = PurePosixPath(image)
    if rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 2 or rel.parts[0] != "images":
        raise ReconstructionError(f"unsafe or non-canonical image path: {image!r}")
    return root / Path(*rel.parts)


def _readable_image(path: Path, expected_size: tuple[int, int] | None = None) -> bool:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return False
    if expected_size is None:
        return True
    width, height = expected_size
    return image.shape[1] == width and image.shape[0] == height


def _group_value(record: dict, grouping_key: str) -> str:
    if grouping_key == "sample_id":
        return loader_sample_id(record)
    value = record.get(grouping_key)
    if not value:
        raise ReconstructionError(
            f"sample {loader_sample_id(record)} has no reconstruction grouping key "
            f"{grouping_key!r}"
        )
    return str(value)


def validate_loader_records(
    records: list[dict], root: Path, *, grouping_key: str = "game_id"
) -> None:
    """Validate IDs, splits, group isolation, file references, and crop dimensions."""
    seen: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        sample_id = loader_sample_id(record)
        if sample_id in seen:
            raise ReconstructionError(f"duplicate loader sample_id: {sample_id}")
        seen.add(sample_id)
        image = record.get("image")
        if not image or Path(image).stem != sample_id:
            raise ReconstructionError(f"sample {sample_id} has an inconsistent image identity")
        try:
            split = Split(record["split"]).value
        except (KeyError, ValueError) as exc:
            raise ReconstructionError(f"sample {sample_id} has an invalid split") from exc
        group_splits[_group_value(record, grouping_key)].add(split)

        image_path = _safe_image_path(root, image)
        bbox = record.get("crop_bbox")
        expected_size = None
        if bbox is not None:
            if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                raise ReconstructionError(f"sample {sample_id} has an invalid crop_bbox")
            expected_size = int(bbox[2]), int(bbox[3])
        if not image_path.is_file() or not _readable_image(image_path, expected_size):
            raise ReconstructionError(
                f"sample {sample_id} has a missing, unreadable, or wrongly-sized image"
            )

    straddling = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if straddling:
        raise ReconstructionError(
            f"{len(straddling)} {grouping_key} group(s) cross splits: {straddling[:5]}"
        )


class ReconstructionTransaction:
    """Stage, validate, and atomically replace a loader dataset directory."""

    def __init__(
        self,
        records: list[ReconstructionRecord],
        dest_dir: Path,
        *,
        allow_partial: bool = False,
        preserve_unselected: bool = False,
        replace_existing: Callable[[dict], bool] | None = None,
        grouping_key: str = "game_id",
    ) -> None:
        self.records = tuple(records)
        self.dest_dir = Path(dest_dir)
        self.allow_partial = allow_partial
        self.preserve_unselected = preserve_unselected
        self.replace_existing = replace_existing
        self.grouping_key = grouping_key
        if replace_existing is not None and not preserve_unselected:
            raise ValueError("replace_existing requires preserve_unselected=True")
        by_id = {record.sample_id: record for record in self.records}
        if len(by_id) != len(self.records):
            counts = Counter(record.sample_id for record in self.records)
            duplicate = next(sample_id for sample_id, count in counts.items() if count > 1)
            raise ReconstructionError(f"duplicate reconstruction sample_id: {duplicate}")
        for record in self.records:
            _safe_image_path(Path("."), record.image)
        self._by_id = by_id
        self._stage: Path | None = None
        self._complete: set[str] = set()
        self._reused: set[str] = set()
        self._written: set[str] = set()
        self._preserved: list[dict] = []
        self._unavailable: dict[str, UnavailableVideo] = {}
        self._advisory: list[AdvisoryMismatch] = []
        self._discarded: set[str] = set()

    @property
    def stage_dir(self) -> Path:
        if self._stage is None:
            raise ReconstructionError("reconstruction transaction has not been entered")
        return self._stage

    @property
    def resume_dir(self) -> Path:
        return self.dest_dir.with_name(f".{self.dest_dir.name}.reconstruction-cache")

    @property
    def pending_records(self) -> tuple[ReconstructionRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.sample_id not in self._complete
            and record.sample_id not in self._discarded
        )

    def __enter__(self) -> "ReconstructionTransaction":
        parent = self.dest_dir.parent
        if self.dest_dir.exists() and (not self.dest_dir.is_dir() or self.dest_dir.is_symlink()):
            raise ReconstructionError(
                f"destination is not a replaceable directory: {self.dest_dir}"
            )
        if self.resume_dir.exists() and (
            not self.resume_dir.is_dir() or self.resume_dir.is_symlink()
        ):
            raise ReconstructionError(
                f"resume cache is not a replaceable directory: {self.resume_dir}"
            )
        parent.mkdir(parents=True, exist_ok=True)
        self._stage = Path(tempfile.mkdtemp(prefix=f".{self.dest_dir.name}.stage-", dir=parent))
        (self.stage_dir / "images").mkdir()
        try:
            self._reuse_existing()
        except Exception:
            self.close()
            raise
        return self

    def _reuse_existing(self) -> None:
        destination_exists = self.dest_dir.is_dir()
        existing = self._existing_loader_records() if destination_exists else []
        existing_by_id: dict[str, dict] = {}
        try:
            for loader_record in existing:
                sample_id = loader_sample_id(loader_record)
                if sample_id in existing_by_id:
                    raise ReconstructionError(f"duplicate existing sample_id: {sample_id}")
                existing_by_id[sample_id] = loader_record
        except (KeyError, TypeError, ReconstructionError) as exc:
            if self.preserve_unselected:
                raise ReconstructionError(
                    f"cannot preserve records from malformed {self.dest_dir / MANIFEST_NAME}"
                ) from exc
            existing = []
            existing_by_id = {}

        if destination_exists:
            for record in self.records:
                prior = existing_by_id.get(record.sample_id)
                if (
                    prior is None
                    or prior.get("image") != record.image
                    or prior.get("reconstruction_fingerprint") != record.fingerprint
                ):
                    continue
                source = _safe_image_path(self.dest_dir, prior["image"])
                if source.is_file() and _readable_image(source, record.crop_size):
                    destination = self.stage_dir / record.image
                    shutil.copy2(source, destination)
                    self._complete.add(record.sample_id)
                    self._reused.add(record.sample_id)

        self._reuse_resume_cache()

        if not self.preserve_unselected or not destination_exists:
            return
        for loader_record in existing:
            sample_id = loader_sample_id(loader_record)
            if sample_id in self._by_id:
                continue
            if self.replace_existing is not None:
                try:
                    if self.replace_existing(loader_record):
                        continue
                except Exception as exc:
                    raise ReconstructionError(
                        f"could not determine replacement scope for existing sample {sample_id}"
                    ) from exc
            source = _safe_image_path(self.dest_dir, loader_record["image"])
            if not source.is_file() or not _readable_image(source):
                raise ReconstructionError(f"cannot preserve {sample_id}: its image is unavailable")
            destination = _safe_image_path(self.stage_dir, loader_record["image"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            self._preserved.append(loader_record)

    def _existing_loader_records(self) -> list[dict]:
        manifest = self.dest_dir / MANIFEST_NAME
        if not manifest.is_file():
            return []
        try:
            records = json.loads(manifest.read_text())["samples"]
            if not isinstance(records, list) or any(
                not isinstance(record, dict) for record in records
            ):
                raise TypeError("samples must be a list of objects")
            return records
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            if self.preserve_unselected:
                raise ReconstructionError(
                    f"cannot preserve records from malformed {manifest}"
                ) from exc
            return []

    def _reuse_resume_cache(self) -> None:
        manifest = self.resume_dir / RESUME_MANIFEST_NAME
        if not manifest.is_file():
            return
        try:
            cached = json.loads(manifest.read_text())["records"]
            by_id = {item["sample_id"]: item for item in cached}
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return
        for record in self.records:
            if record.sample_id in self._complete:
                continue
            item = by_id.get(record.sample_id)
            if item is None or item.get("fingerprint") != record.fingerprint:
                continue
            try:
                source = _safe_image_path(self.resume_dir, item.get("image", ""))
            except (ReconstructionError, TypeError):
                continue
            if not source.is_file() or not _readable_image(source, record.crop_size):
                continue
            destination = self.stage_dir / record.image
            shutil.copy2(source, destination)
            self._complete.add(record.sample_id)
            self._reused.add(record.sample_id)

    def write_image(self, record: ReconstructionRecord, image: np.ndarray) -> None:
        """Write and re-read one crop in staging before marking it structurally complete."""
        if record.sample_id not in self._by_id:
            raise ReconstructionError(f"record {record.sample_id} is outside this transaction")
        width, height = record.crop_size
        if image.ndim not in (2, 3) or image.shape[1] != width or image.shape[0] != height:
            actual = "x".join(str(dimension) for dimension in reversed(image.shape[:2]))
            raise ReconstructionError(
                f"sample {record.sample_id} crop is {actual or 'dimensionless'}, expected "
                f"{width}x{height}"
            )
        output = self.stage_dir / record.image
        if not cv2.imwrite(str(output), image):
            raise ReconstructionError(f"OpenCV failed to write sample {record.sample_id}")
        if not _readable_image(output, record.crop_size):
            raise ReconstructionError(f"written crop for {record.sample_id} failed validation")
        self._complete.add(record.sample_id)
        self._written.add(record.sample_id)

    def discard_records(self, sample_ids: set[str]) -> None:
        """Remove selected records and their staged crops before the atomic commit."""
        unknown = sample_ids - self._by_id.keys()
        if unknown:
            raise ReconstructionError(
                f"cannot discard records outside this transaction: {sorted(unknown)[:5]}"
            )
        for sample_id in sample_ids:
            record = self._by_id[sample_id]
            (self.stage_dir / record.image).unlink(missing_ok=True)
            self._complete.discard(sample_id)
            self._reused.discard(sample_id)
            self._written.discard(sample_id)
        self._discarded.update(sample_ids)

    def note_unavailable_video(self, video_id: str, reason: str) -> None:
        self._unavailable[video_id] = UnavailableVideo(video_id, reason)

    def note_advisory_mismatch(self, sample_id: str, reason: str) -> None:
        if sample_id not in self._by_id:
            raise ReconstructionError(f"advisory sample {sample_id} is outside this transaction")
        self._advisory.append(AdvisoryMismatch(sample_id, reason))

    def _report(self, committed: bool) -> ReconstructionReport:
        active = [
            record for record in self.records if record.sample_id not in self._discarded
        ]
        expected = tuple(record.sample_id for record in active)
        complete = tuple(sample_id for sample_id in expected if sample_id in self._complete)
        missing = tuple(sample_id for sample_id in expected if sample_id not in self._complete)
        return ReconstructionReport(
            committed=committed,
            partial=bool(missing),
            expected_sample_ids=expected,
            complete_sample_ids=complete,
            missing_sample_ids=missing,
            reused_sample_ids=tuple(sorted(self._reused)),
            written_sample_ids=tuple(sorted(self._written)),
            unavailable_videos=tuple(self._unavailable[key] for key in sorted(self._unavailable)),
            advisory_mismatches=tuple(self._advisory),
            manifest_path=self.dest_dir / MANIFEST_NAME,
        )

    def commit(self, provenance: dict | None = None) -> ReconstructionReport:
        """Commit a complete dataset, or an explicitly authorized partial one."""
        preliminary = self._report(committed=False)
        if preliminary.missing_sample_ids and not self.allow_partial:
            self._save_resume_cache()
            return preliminary

        complete = self._complete
        active = [
            record for record in self.records if record.sample_id not in self._discarded
        ]
        selected = [record.loader_record() for record in active if record.sample_id in complete]
        samples = [*self._preserved, *selected]
        validate_loader_records(samples, self.stage_dir, grouping_key=self.grouping_key)
        split_counts = Counter(record["split"] for record in selected)
        dataset_split_counts = Counter(record["split"] for record in samples)
        expected_split_counts = Counter(record.split.value for record in active)
        manifest_provenance = {
            **(provenance or {}),
            "partial": bool(preliminary.missing_sample_ids),
            "expected_counts": {
                "records": len(active),
                "splits": dict(sorted(expected_split_counts.items())),
            },
            "actual_counts": {
                "records": len(selected),
                "splits": dict(sorted(split_counts.items())),
            },
            "dataset_counts": {
                "records": len(samples),
                "splits": dict(sorted(dataset_split_counts.items())),
            },
            "preserved_records": len(self._preserved),
            "missing_sample_ids": list(preliminary.missing_sample_ids),
            "unavailable_videos": [asdict(item) for item in preliminary.unavailable_videos],
            "pixel_content": {
                "status": "advisory",
                "note": PIXEL_ADVISORY_NOTE,
                "mismatches": [asdict(item) for item in preliminary.advisory_mismatches],
            },
        }
        manifest_payload = {"version": "v1", "provenance": manifest_provenance, "samples": samples}
        (self.stage_dir / MANIFEST_NAME).write_text(json.dumps(manifest_payload, indent=2) + "\n")

        report = self._report(committed=True)
        (self.stage_dir / REPORT_NAME).write_text(json.dumps(report.to_dict(), indent=2) + "\n")
        self._commit_stage()
        return report

    def _save_resume_cache(self) -> None:
        cached = [
            {
                "sample_id": record.sample_id,
                "image": record.image,
                "fingerprint": record.fingerprint,
            }
            for record in self.records
            if record.sample_id in self._complete
        ]
        (self.stage_dir / RESUME_MANIFEST_NAME).write_text(
            json.dumps({"records": cached}, indent=2) + "\n"
        )
        backup: Path | None = None
        if self.resume_dir.exists():
            if not self.resume_dir.is_dir() or self.resume_dir.is_symlink():
                raise ReconstructionError(
                    f"resume cache is not a replaceable directory: {self.resume_dir}"
                )
            backup = self.resume_dir.with_name(f".{self.resume_dir.name}.backup-{uuid.uuid4().hex}")
            os.replace(self.resume_dir, backup)
        try:
            os.replace(self.stage_dir, self.resume_dir)
            self._stage = None
        except Exception:
            if backup is not None and not self.resume_dir.exists():
                os.replace(backup, self.resume_dir)
            raise
        if backup is not None:
            shutil.rmtree(backup)

    def _commit_stage(self) -> None:
        stage = self.stage_dir
        backup: Path | None = None
        if self.dest_dir.exists():
            if not self.dest_dir.is_dir() or self.dest_dir.is_symlink():
                raise ReconstructionError(
                    f"destination is not a replaceable directory: {self.dest_dir}"
                )
            backup = self.dest_dir.with_name(f".{self.dest_dir.name}.backup-{uuid.uuid4().hex}")
            os.replace(self.dest_dir, backup)
        try:
            os.replace(stage, self.dest_dir)
            self._stage = None
        except Exception:
            if backup is not None and not self.dest_dir.exists():
                os.replace(backup, self.dest_dir)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        if self.resume_dir.is_dir() and not self.resume_dir.is_symlink():
            shutil.rmtree(self.resume_dir)

    def close(self) -> None:
        if self._stage is not None and self._stage.exists():
            shutil.rmtree(self._stage)
        self._stage = None

    def __exit__(self, *exc) -> None:
        self.close()
