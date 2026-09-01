"""Versioned metadata bundle and validation contract for the frozen SLCC release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import date
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chessqueries.annotate.manifest import DEFAULT_VIDEOS_PATH, Manifest
from chessqueries.annotate.reconstruction import ReconstructionRecord, stable_sample_id
from chessqueries.annotate.schema import Annotation, AnnotationFile
from chessqueries.core import Split
from chessqueries.data.base import DatasetName
from chessqueries.data.download import AnonymizedArtifactError, is_anonymized_placeholder
from chessqueries.data.inventory import FROZEN_SAMPLE_COUNTS

RELEASE_VERSION = "slcc-v1"
SCHEMA_VERSION = 1
RECONSTRUCTION_TOOL_VERSION = "1"
DEFAULT_HF_REPO_ID = "joelseytre/slcc"
# Immutable Hub commit for the canonical release; the manifest hash pins it again.
DEFAULT_HF_REVISION = "41bf6ca7fe50b6e3ed781e651b4fa53fcc9a7c25"
DEFAULT_BUNDLE_PATH = Path("data/slcc/releases") / RELEASE_VERSION
DEFAULT_DOCUMENT_SOURCE = Path(__file__).with_name("release_assets")
DEFAULT_FREEZE_DATE = date(2026, 7, 17)
EXPECTED_RECORDS = 2_174
EXPECTED_VIDEOS = 20
EXPECTED_GROUPS = 152
EXPECTED_SPLITS = {
    split.value: count
    for split, count in FROZEN_SAMPLE_COUNTS[DatasetName.SLCC].items()
}
EXPECTED_SOURCE_SHA256 = "057f247ae92b134ca2b172317335919df01b22cfaa7472ddaf53393c2515ab75"
EXPECTED_MANIFEST_SHA256 = "3c6137508de17472bd97470cd3f7d218e585db9f24a77574165029d16401e375"
GROUPING_KEY = "game_id"
GROUPING_DESCRIPTION = (
    "Global game_id from the relay round and player slug. The video_id is deliberately "
    "excluded, so one game spanning two broadcast parts remains one leakage group."
)
NO_PIXELS_STATEMENT = (
    "This bundle contains annotation and provenance metadata only. It contains no source "
    "video, frame, crop, thumbnail, descriptor, or PCA payload."
)
HUB_CARD_FILENAME = "README.md"
REQUIRED_DOCUMENTS = ("LICENSE-ANNOTATIONS.md",)
REQUIRED_BUNDLE_FILES = (*REQUIRED_DOCUMENTS, "manifest.json", "checksums.sha256")
HF_ALLOW_PATTERNS = (*REQUIRED_BUNDLE_FILES, "annotations/*.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AnnotationFileSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str = Field(pattern=r"^[A-Za-z0-9_-]{11}$")
    file: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_count: int = Field(ge=1)
    format_id: str
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    source_fps: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_path(self) -> "AnnotationFileSpec":
        if self.file != f"annotations/{self.video_id}.json":
            raise ValueError(f"annotation path does not match video identity: {self.file}")
        return self


class ReleaseRecordSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_id: str
    video_id: str = Field(pattern=r"^[A-Za-z0-9_-]{11}$")
    annotation_file: str
    frame_index: int = Field(ge=0)
    split: Split
    group_id: str
    image: str
    crop_size: tuple[int, int]

    @model_validator(mode="after")
    def validate_identity(self) -> "ReleaseRecordSpec":
        expected = f"{self.video_id}_{self.frame_index}"
        if self.sample_id != expected:
            raise ValueError(f"sample_id {self.sample_id!r} must be {expected!r}")
        if self.image != f"images/{self.sample_id}.jpg":
            raise ValueError(f"sample {self.sample_id} has a non-canonical image path")
        if self.annotation_file != f"annotations/{self.video_id}.json":
            raise ValueError(f"sample {self.sample_id} has a non-canonical annotation path")
        if any(dimension <= 0 for dimension in self.crop_size):
            raise ValueError(f"sample {self.sample_id} has a non-positive crop size")
        return self


class GroupingSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    description: str
    count: int
    video_scoped_count: int
    cross_video_groups: list[str]


class ReleaseCounts(BaseModel):
    model_config = ConfigDict(frozen=True)

    records: int
    videos: int
    canonical_groups: int
    splits: dict[str, int]


class ReconstructionToolSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: str
    module: str


class ReleaseManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    release_version: str
    schema_version: int
    freeze_date: date
    metadata_only: bool
    no_pixels_statement: str
    source_loader_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    grouping: GroupingSpec
    counts: ReleaseCounts
    reconstruction_tool: ReconstructionToolSpec
    annotation_files: list[AnnotationFileSpec]
    records: list[ReleaseRecordSpec]

    @model_validator(mode="after")
    def validate_release(self) -> "ReleaseManifest":
        if self.release_version != RELEASE_VERSION or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported SLCC release/schema version")
        if self.freeze_date != DEFAULT_FREEZE_DATE:
            raise ValueError("manifest does not carry the frozen SLCC v1 date")
        if not self.metadata_only:
            raise ValueError("SLCC release must be explicitly metadata-only")
        if self.grouping.key != GROUPING_KEY:
            raise ValueError(f"canonical grouping key must be {GROUPING_KEY!r}")
        if (
            self.reconstruction_tool.version != RECONSTRUCTION_TOOL_VERSION
            or self.reconstruction_tool.module != "chessqueries.annotate.reconstruct"
        ):
            raise ValueError("manifest names an unsupported reconstruction tool")

        sample_ids = [record.sample_id for record in self.records]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("release records contain duplicate sample IDs")
        videos = {record.video_id for record in self.records}
        groups = {record.group_id for record in self.records}
        splits = Counter(record.split.value for record in self.records)
        annotation_files = {spec.file for spec in self.annotation_files}
        annotation_videos = {spec.video_id for spec in self.annotation_files}
        if len(annotation_files) != len(self.annotation_files):
            raise ValueError("release annotation-file list contains duplicates")
        if len(annotation_videos) != len(self.annotation_files):
            raise ValueError("release contains multiple annotation files for one video")
        if annotation_videos != videos:
            raise ValueError("release annotation-file videos do not match record videos")
        if any(record.annotation_file not in annotation_files for record in self.records):
            raise ValueError("release record refers to an unlisted annotation file")

        actual = ReleaseCounts(
            records=len(self.records),
            videos=len(videos),
            canonical_groups=len(groups),
            splits=dict(sorted(splits.items())),
        )
        if actual != self.counts:
            raise ValueError(f"declared release counts do not match records: {actual}")
        if self.grouping.count != len(groups):
            raise ValueError("declared grouping count does not match records")

        game_videos: dict[str, set[str]] = defaultdict(set)
        for record in self.records:
            game_videos[record.group_id].add(record.video_id)
        video_scoped_count = len({(record.video_id, record.group_id) for record in self.records})
        cross_video = sorted(group for group, assigned in game_videos.items() if len(assigned) > 1)
        if self.grouping.video_scoped_count != video_scoped_count:
            raise ValueError("declared video-scoped group count does not match records")
        if self.grouping.cross_video_groups != cross_video:
            raise ValueError("declared cross-video groups do not match records")

        if (
            self.counts.records != EXPECTED_RECORDS
            or self.counts.videos != EXPECTED_VIDEOS
            or self.counts.canonical_groups != EXPECTED_GROUPS
            or self.counts.splits != EXPECTED_SPLITS
            or self.source_loader_manifest_sha256 != EXPECTED_SOURCE_SHA256
        ):
            raise ValueError("manifest does not match the frozen SLCC v1 identity")

        group_splits: dict[str, set[Split]] = defaultdict(set)
        for record in self.records:
            group_splits[record.group_id].add(record.split)
        straddling = [group for group, assigned in group_splits.items() if len(assigned) > 1]
        if straddling:
            raise ValueError(f"canonical groups cross splits: {straddling[:5]}")
        return self


class ReleaseBundle:
    """A validated release manifest plus its per-video annotation records."""

    def __init__(self, root: Path, manifest: ReleaseManifest) -> None:
        self.root = Path(root)
        self.manifest = manifest

    @classmethod
    def load(cls, root: Path) -> "ReleaseBundle":
        root = Path(root)
        missing = [name for name in REQUIRED_BUNDLE_FILES if not (root / name).is_file()]
        if missing:
            raise ValueError(f"release bundle is missing required files: {missing}")
        manifest_path = root / "manifest.json"
        manifest_sha256 = sha256_file(manifest_path)
        if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
            raise ValueError(
                f"release manifest is not the frozen SLCC v1 manifest: {manifest_sha256}"
            )
        manifest = ReleaseManifest.model_validate_json(manifest_path.read_text())
        assert_metadata_only(root)
        verify_checksums(root)
        bundle = cls(root, manifest)
        bundle.reconstruction_records()
        return bundle

    def reconstruction_records(self) -> list[ReconstructionRecord]:
        """Join release split specs to full annotations and validate the join exactly."""
        annotations: dict[tuple[str, str], Annotation] = {}
        for spec in self.manifest.annotation_files:
            path = self.root / spec.file
            if sha256_file(path) != spec.sha256:
                raise ValueError(f"annotation checksum does not match manifest: {spec.file}")
            annotation_file = AnnotationFile.load(path)
            _assert_public_provenance(annotation_file.provenance)
            if annotation_file.provenance.get("video_id") != spec.video_id:
                raise ValueError(f"annotation provenance video mismatch: {spec.file}")
            if str(annotation_file.provenance.get("format_id")) != spec.format_id:
                raise ValueError(f"annotation format mismatch: {spec.file}")
            source_properties = (
                int(annotation_file.provenance.get("width", 0)),
                int(annotation_file.provenance.get("height", 0)),
                float(annotation_file.provenance.get("fps", 0)),
            )
            expected_properties = (spec.source_width, spec.source_height, spec.source_fps)
            if source_properties != expected_properties:
                raise ValueError(f"annotation source properties mismatch: {spec.file}")
            if len(annotation_file.annotations) != spec.record_count:
                raise ValueError(f"annotation record count mismatch: {spec.file}")
            for annotation in annotation_file.annotations:
                key = (spec.file, stable_sample_id(annotation))
                if key in annotations:
                    raise ValueError(f"duplicate annotation identity in {spec.file}: {key[1]}")
                annotations[key] = annotation

        output: list[ReconstructionRecord] = []
        used: set[tuple[str, str]] = set()
        for spec in self.manifest.records:
            key = (spec.annotation_file, spec.sample_id)
            try:
                annotation = annotations[key]
            except KeyError as exc:
                raise ValueError(
                    f"release record has no matching annotation: {spec.sample_id}"
                ) from exc
            if annotation.video_id != spec.video_id or annotation.frame_index != spec.frame_index:
                raise ValueError(f"release identity mismatch for {spec.sample_id}")
            if annotation.game_id != spec.group_id:
                raise ValueError(f"release group mismatch for {spec.sample_id}")
            fen_fields = annotation.fen.split()
            if (
                len(fen_fields) != 6
                or fen_fields[0] != annotation.placement
                or fen_fields[1] != annotation.side_to_move
            ):
                raise ValueError(f"release FEN fields disagree for {spec.sample_id}")
            if (annotation.crop_bbox[2], annotation.crop_bbox[3]) != spec.crop_size:
                raise ValueError(f"release crop-size mismatch for {spec.sample_id}")
            if not annotation.verified_by_human:
                raise ValueError(f"release annotation is not human-verified: {spec.sample_id}")
            output.append(ReconstructionRecord(spec.sample_id, spec.split, annotation))
            used.add(key)
        unused = set(annotations) - used
        if unused:
            raise ValueError(f"annotation files contain {len(unused)} records absent from manifest")
        return output

    def file_spec(self, video_id: str) -> AnnotationFileSpec:
        matches = [spec for spec in self.manifest.annotation_files if spec.video_id == video_id]
        if len(matches) != 1:
            raise ValueError(
                f"expected one annotation file for video {video_id}, got {len(matches)}"
            )
        return matches[0]


def _snapshot_download(repo_id: str, revision: str, local_dir: Path) -> str:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Hugging Face download support is unavailable; install the annotation "
            "dependencies with `poetry install --with annotate`"
        ) from exc

    try:
        downloaded = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=local_dir,
            allow_patterns=list(HF_ALLOW_PATTERNS),
        )
    except (RepositoryNotFoundError, RevisionNotFoundError) as exc:
        if not is_anonymized_placeholder(repo_id):
            raise
        raise AnonymizedArtifactError(
            f"SLCC annotation release ({repo_id}@{revision})",
            "the review archive already ships this bundle at "
            f"{DEFAULT_BUNDLE_PATH}/ — reaching this download means that local "
            "copy is missing or failed validation, so restore it by re-unzipping "
            "the archive rather than re-downloading.",
        ) from exc
    if not isinstance(downloaded, str):  # snapshot_download returns a list only for dry runs
        raise RuntimeError("Hugging Face returned no downloaded snapshot path")
    return downloaded


def _replace_directory(stage: Path, output: Path) -> None:
    """Atomically install a staged directory, restoring the prior one on failure."""
    backup: Path | None = None
    if output.exists():
        backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
        os.replace(output, backup)
    try:
        os.replace(stage, output)
    except Exception:
        if backup is not None and not output.exists():
            os.replace(backup, output)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def fetch_release_bundle(
    *,
    repo_id: str = DEFAULT_HF_REPO_ID,
    revision: str = DEFAULT_HF_REVISION,
    local_dir: Path = DEFAULT_BUNDLE_PATH,
) -> "ReleaseBundle":
    """Fetch and validate the pinned release before installing its local directory."""
    local_dir = Path(local_dir)
    if local_dir.exists() and (not local_dir.is_dir() or local_dir.is_symlink()):
        raise ValueError(f"release cache is not a replaceable directory: {local_dir}")
    if local_dir.exists():
        try:
            return ReleaseBundle.load(local_dir)
        except (OSError, ValueError):
            print(f"  [redo] cached SLCC release failed validation: {local_dir}")

    parent = local_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{local_dir.name}.part-", dir=parent))
    try:
        root = Path(_snapshot_download(repo_id, revision, stage))
        if root.resolve() != stage.resolve():
            raise RuntimeError(
                f"Hugging Face downloaded outside the requested staging directory: {root}"
            )
        ReleaseBundle.load(stage)
        _replace_directory(stage, local_dir)
        return ReleaseBundle.load(local_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _bundle_files(root: Path) -> Iterator[Path]:
    """Yield payload files, excluding Hugging Face's local-dir cache metadata."""
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ".cache":
            continue
        if path.is_file():
            yield path


def _checksum_entries(root: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for path in _bundle_files(root):
        relative = path.relative_to(root).as_posix()
        if path.name != "checksums.sha256" and relative != HUB_CARD_FILENAME:
            entries.append((sha256_file(path), relative))
    return entries


def write_checksums(root: Path) -> None:
    lines = [f"{digest}  {relative}" for digest, relative in _checksum_entries(root)]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n")


def verify_checksums(root: Path) -> None:
    checksum_path = root / "checksums.sha256"
    expected: dict[str, str] = {}
    for line in checksum_path.read_text().splitlines():
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"malformed checksum line: {line!r}") from exc
        rel = PurePosixPath(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe checksum path: {relative!r}")
        if relative in expected:
            raise ValueError(f"duplicate checksum path: {relative}")
        # Early SLCC v1 bundles included the mutable Hugging Face dataset card
        # in their checksums. Keep those pinned revisions readable while
        # treating the live Hub card as metadata outside the frozen bundle.
        if relative == HUB_CARD_FILENAME:
            continue
        expected[relative] = digest
    actual = dict((relative, digest) for digest, relative in _checksum_entries(root))
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        changed = sorted(
            path for path in expected.keys() & actual.keys() if expected[path] != actual[path]
        )
        raise ValueError(
            f"release checksum mismatch; missing={missing}, unexpected={unexpected}, changed={changed}"
        )


MEDIA_SIGNATURES = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"BM",
    b"RIFF",
    b"\x1aE\xdf\xa3",
)
_METADATA_SUFFIXES = {".json", ".md", ".sha256"}


def has_media_signature(path: Path) -> bool:
    """Return whether a file starts with a common image or video signature."""
    with path.open("rb") as handle:
        prefix = handle.read(16)
    return any(prefix.startswith(signature) for signature in MEDIA_SIGNATURES) or (
        len(prefix) >= 12 and prefix[4:8] == b"ftyp"
    )


def assert_metadata_only(root: Path) -> None:
    """Reject media extensions, signatures, symlinks, and unrecognized payloads."""
    for path in _bundle_files(root):
        if path.is_symlink():
            raise ValueError(f"release bundle may not contain symlinks: {path}")
        if path.suffix.lower() not in _METADATA_SUFFIXES:
            raise ValueError(f"release bundle contains a non-metadata extension: {path}")
        if has_media_signature(path):
            raise ValueError(f"release bundle contains a media payload: {path}")


def _assert_public_provenance(value: object, context: str = "provenance") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_public_provenance(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_public_provenance(child, f"{context}[{index}]")
    elif isinstance(value, str) and value.startswith("/"):
        raise ValueError(f"absolute local path in release {context}: {value}")


def _loader_sample_id(record: dict) -> str:
    return str(record.get("sample_id") or Path(record["image"]).stem)


def _build_release_bundle(
    dataset_manifest: Path,
    reviewed_dir: Path,
    output: Path,
    *,
    videos_manifest: Path = DEFAULT_VIDEOS_PATH,
    freeze_date: date = DEFAULT_FREEZE_DATE,
) -> ReleaseBundle:
    """Generate SLCC v1 metadata from the byte-frozen loader and reviewed labels."""
    dataset_manifest = Path(dataset_manifest)
    reviewed_dir = Path(reviewed_dir)
    output = Path(output)
    loader = json.loads(dataset_manifest.read_text())
    loader_records = loader["samples"]
    sample_ids = [_loader_sample_id(record) for record in loader_records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("frozen loader manifest contains duplicate sample IDs")
    by_id = dict(zip(sample_ids, loader_records, strict=True))
    videos = Manifest.load(videos_manifest)

    output.mkdir(parents=True, exist_ok=True)
    annotations_dir = output / "annotations"
    annotations_dir.mkdir(exist_ok=True)
    for stale in annotations_dir.glob("*.json"):
        stale.unlink()

    video_order = list(loader.get("provenance", {}).get("reviewed_videos", []))
    if set(video_order) != {record["video_id"] for record in loader_records}:
        video_order = sorted({record["video_id"] for record in loader_records})
    unregistered = sorted(set(video_order) - videos.entries.keys())
    if unregistered:
        raise ValueError(f"release videos are absent from the source manifest: {unregistered}")

    annotation_specs: list[AnnotationFileSpec] = []
    annotations_by_id: dict[str, tuple[str, Annotation]] = {}
    for video_id in video_order:
        source = reviewed_dir / f"{video_id}.reviewed.json"
        annotation_file = AnnotationFile.load(source)
        _assert_public_provenance(annotation_file.provenance)
        selected = [
            annotation
            for annotation in annotation_file.annotations
            if stable_sample_id(annotation) in by_id
        ]
        selected_ids = {stable_sample_id(annotation) for annotation in selected}
        expected_ids = {
            sample_id for sample_id, record in by_id.items() if record["video_id"] == video_id
        }
        if selected_ids != expected_ids:
            raise ValueError(
                f"reviewed annotations do not match frozen loader for {video_id}: "
                f"missing={sorted(expected_ids - selected_ids)}, extra={sorted(selected_ids - expected_ids)}"
            )
        if len(selected) != len(selected_ids):
            raise ValueError(f"reviewed annotations contain duplicate identities for {video_id}")
        if any(not annotation.verified_by_human for annotation in selected):
            raise ValueError(f"unverified annotation selected for release: {video_id}")

        relative = f"annotations/{video_id}.json"
        destination = output / relative
        release_provenance = {
            **annotation_file.provenance,
            "release_version": RELEASE_VERSION,
            "release_record_count": len(selected),
        }
        AnnotationFile(release_provenance, selected).save(destination)
        provenance = annotation_file.provenance
        annotation_specs.append(
            AnnotationFileSpec(
                video_id=video_id,
                file=relative,
                sha256=sha256_file(destination),
                record_count=len(selected),
                format_id=str(provenance["format_id"]),
                source_width=int(provenance["width"]),
                source_height=int(provenance["height"]),
                source_fps=float(provenance["fps"]),
            )
        )
        for annotation in selected:
            sample_id = stable_sample_id(annotation)
            annotations_by_id[sample_id] = (relative, annotation)

    release_records: list[ReleaseRecordSpec] = []
    for sample_id, loader_record in zip(sample_ids, loader_records, strict=True):
        annotation_file, annotation = annotations_by_id[sample_id]
        expected_pairs = {
            "video_id": annotation.video_id,
            "frame_index": annotation.frame_index,
            "gt_fen": annotation.fen,
            "game_id": annotation.game_id,
            "crop_bbox": annotation.crop_bbox,
        }
        changed = [
            key for key, expected in expected_pairs.items() if loader_record.get(key) != expected
        ]
        if changed:
            raise ValueError(
                f"frozen loader and reviewed annotation disagree for {sample_id}: {changed}"
            )
        release_records.append(
            ReleaseRecordSpec(
                sample_id=sample_id,
                video_id=annotation.video_id,
                annotation_file=annotation_file,
                frame_index=annotation.frame_index,
                split=Split(loader_record["split"]),
                group_id=annotation.game_id,
                image=f"images/{sample_id}.jpg",
                crop_size=(annotation.crop_bbox[2], annotation.crop_bbox[3]),
            )
        )

    game_videos: dict[str, set[str]] = defaultdict(set)
    for record in release_records:
        game_videos[record.group_id].add(record.video_id)
    cross_video = sorted(game for game, assigned in game_videos.items() if len(assigned) > 1)
    counts = ReleaseCounts(
        records=len(release_records),
        videos=len({record.video_id for record in release_records}),
        canonical_groups=len(game_videos),
        splits=dict(sorted(Counter(record.split.value for record in release_records).items())),
    )
    if (
        counts.records != EXPECTED_RECORDS
        or counts.videos != EXPECTED_VIDEOS
        or counts.canonical_groups != EXPECTED_GROUPS
        or counts.splits != EXPECTED_SPLITS
    ):
        raise ValueError(f"input does not match frozen SLCC v1 counts: {counts}")

    video_scoped_count = len({(record.video_id, record.group_id) for record in release_records})
    manifest = ReleaseManifest(
        release_version=RELEASE_VERSION,
        schema_version=SCHEMA_VERSION,
        freeze_date=freeze_date,
        metadata_only=True,
        no_pixels_statement=NO_PIXELS_STATEMENT,
        source_loader_manifest_sha256=sha256_file(dataset_manifest),
        grouping=GroupingSpec(
            key=GROUPING_KEY,
            description=GROUPING_DESCRIPTION,
            count=counts.canonical_groups,
            video_scoped_count=video_scoped_count,
            cross_video_groups=cross_video,
        ),
        counts=counts,
        reconstruction_tool=ReconstructionToolSpec(
            version=RECONSTRUCTION_TOOL_VERSION,
            module="chessqueries.annotate.reconstruct",
        ),
        annotation_files=annotation_specs,
        records=release_records,
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n"
    )
    write_checksums(output)
    return ReleaseBundle.load(output)


def build_release_bundle(
    dataset_manifest: Path,
    reviewed_dir: Path,
    output: Path,
    *,
    videos_manifest: Path = DEFAULT_VIDEOS_PATH,
    freeze_date: date = DEFAULT_FREEZE_DATE,
) -> ReleaseBundle:
    """Generate in staging and replace the prior bundle only after validation."""
    output = Path(output)
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and (not output.is_dir() or output.is_symlink()):
        raise ValueError(f"release output is not a replaceable directory: {output}")

    document_source = DEFAULT_DOCUMENT_SOURCE
    missing_documents = [
        name for name in REQUIRED_DOCUMENTS if not (document_source / name).is_file()
    ]
    if missing_documents:
        raise ValueError(f"release documentation templates are missing: {missing_documents}")

    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=parent))
    try:
        for name in REQUIRED_DOCUMENTS:
            shutil.copy2(document_source / name, stage / name)
        _build_release_bundle(
            dataset_manifest,
            reviewed_dir,
            stage,
            videos_manifest=videos_manifest,
            freeze_date=freeze_date,
        )
        _replace_directory(stage, output)
        return ReleaseBundle.load(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build or validate the metadata-only SLCC v1 release"
    )
    parser.add_argument(
        "--dataset-manifest", type=Path, default=Path("data/slcc/dataset/annotations.json")
    )
    parser.add_argument("--reviewed-dir", type=Path, default=Path("data/slcc"))
    parser.add_argument("--videos-manifest", type=Path, default=DEFAULT_VIDEOS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument(
        "--check", action="store_true", help="validate the existing bundle without rebuilding"
    )
    args = parser.parse_args(argv)
    if args.check:
        bundle = ReleaseBundle.load(args.out)
    else:
        bundle = build_release_bundle(
            args.dataset_manifest,
            args.reviewed_dir,
            args.out,
            videos_manifest=args.videos_manifest,
        )
    counts = bundle.manifest.counts
    print(
        f"{bundle.root}: {counts.records} records, {counts.videos} videos, "
        f"{counts.canonical_groups} canonical games"
    )


if __name__ == "__main__":
    main()
