"""Public reconstruction of loader-ready SLCC crops from the metadata release."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from chessqueries.annotate.reconstruction import (
    ReconstructionRecord,
    ReconstructionReport,
    ReconstructionTransaction,
)
from chessqueries.annotate.release import (
    DEFAULT_BUNDLE_PATH,
    DEFAULT_HF_REPO_ID,
    DEFAULT_HF_REVISION,
    ReleaseBundle,
    fetch_release_bundle,
    sha256_file,
)
from chessqueries.annotate.templates import Rect
from chessqueries.annotate.video import FrameReader, YoutubeAccess, download, probe_source


def reconstruct_bundle(
    bundle_path: Path,
    dest_dir: Path,
    *,
    video_dir: Path,
    allow_partial: bool = False,
    discard_videos: bool = False,
    youtube_access: YoutubeAccess | None = None,
) -> ReconstructionReport:
    """Reconstruct every available release video through one validated transaction."""
    dest_dir = Path(dest_dir)
    video_dir = Path(video_dir)
    resolved_dest = dest_dir.resolve()
    resolved_video_dir = video_dir.resolve()
    if (
        resolved_dest == Path(resolved_dest.anchor)
        or resolved_dest == Path.home().resolve()
        or (resolved_dest / ".git").exists()
    ):
        raise ValueError(f"refusing unsafe reconstruction destination: {dest_dir}")
    if resolved_video_dir == resolved_dest or resolved_dest in resolved_video_dir.parents:
        raise ValueError("--video-dir must not be inside --out, which is replaced atomically")
    if dest_dir.is_dir():
        allowed = {"images", "annotations.json", "reconstruction-report.json"}
        unexpected = sorted(path.name for path in dest_dir.iterdir() if path.name not in allowed)
        if unexpected:
            raise ValueError(f"refusing to replace non-dataset files in {dest_dir}: {unexpected}")

    bundle = ReleaseBundle.load(bundle_path)
    records = bundle.reconstruction_records()
    downloaded = []
    provenance = {
        "release_version": bundle.manifest.release_version,
        "release_schema_version": bundle.manifest.schema_version,
        "release_manifest_sha256": sha256_file(bundle.root / "manifest.json"),
        "grouping_key": bundle.manifest.grouping.key,
    }

    with ReconstructionTransaction(
        records,
        dest_dir,
        allow_partial=allow_partial,
        preserve_unselected=False,
    ) as transaction:
        by_video: dict[str, list[ReconstructionRecord]] = defaultdict(list)
        for record in transaction.pending_records:
            by_video[record.annotation.video_id].append(record)

        for video_id, pending in by_video.items():
            spec = bundle.file_spec(video_id)
            try:
                video = download(
                    video_id,
                    video_dir,
                    format_id=spec.format_id,
                    access=youtube_access,
                )
                downloaded.append(video)
                if (video.width, video.height) != (spec.source_width, spec.source_height):
                    raise RuntimeError(
                        f"source dimensions are {video.width}x{video.height}, expected "
                        f"{spec.source_width}x{spec.source_height}"
                    )
                if abs(video.fps - spec.source_fps) > 1e-3:
                    raise RuntimeError(f"source fps is {video.fps}, expected {spec.source_fps}")
                with FrameReader(video) as reader:
                    for record in pending:
                        frame = reader.frame_at_index(record.annotation.frame_index)
                        crop = Rect.from_list(record.annotation.crop_bbox).crop(frame)
                        transaction.write_image(record, crop)
            except Exception as exc:
                transaction.note_unavailable_video(video_id, str(exc))

        report = transaction.commit(provenance)

    if discard_videos and report.committed:
        for video in downloaded:
            video.path.unlink(missing_ok=True)
    return report


def resolve_bundle_path(
    bundle_path: Path | None,
    *,
    hf_repo: str = DEFAULT_HF_REPO_ID,
    hf_revision: str = DEFAULT_HF_REVISION,
    bundle_cache: Path = DEFAULT_BUNDLE_PATH,
) -> Path:
    """Use an explicit local bundle or fetch the pinned public release."""
    if bundle_path is not None:
        return Path(bundle_path)
    return fetch_release_bundle(
        repo_id=hf_repo,
        revision=hf_revision,
        local_dir=bundle_cache,
    ).root


def check_sources(bundle_path: Path, access: YoutubeAccess | None = None) -> dict:
    """Probe every frozen YouTube source and return a machine-readable report."""
    bundle = ReleaseBundle.load(bundle_path)
    sources = [
        probe_source(spec.video_id, format_id=spec.format_id, access=access)
        for spec in bundle.manifest.annotation_files
    ]
    available = sum(source["available"] for source in sources)
    return {
        "release_version": bundle.manifest.release_version,
        "available": available,
        "unavailable": len(sources) - available,
        "complete": available == len(sources),
        "sources": sources,
    }


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Reconstruct SLCC frames from annotations")
    parser.add_argument(
        "--bundle",
        type=Path,
        help="local annotation bundle; otherwise fetch the pinned release from Hugging Face",
    )
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--hf-revision", default=DEFAULT_HF_REVISION)
    parser.add_argument(
        "--bundle-cache",
        type=Path,
        default=DEFAULT_BUNDLE_PATH,
        help="local directory for the fetched Hugging Face annotation bundle",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--video-dir", type=Path, default=Path("data/slcc/videos"))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--discard-videos", action="store_true")
    parser.add_argument("--report", type=Path, help="also write the structured outcome here")
    parser.add_argument(
        "--check-sources",
        action="store_true",
        help="probe every pinned source/format without downloading video",
    )
    cookies = parser.add_mutually_exclusive_group()
    cookies.add_argument("--cookies", type=Path, help="Netscape-format cookie file for yt-dlp")
    cookies.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER[:PROFILE]",
        help="let yt-dlp read cookies from an installed browser profile",
    )
    parser.add_argument(
        "--js-runtime",
        metavar="RUNTIME[:PATH]",
        help="yt-dlp JavaScript runtime; auto-detected when omitted",
    )
    args = parser.parse_args(argv)

    bundle_path = resolve_bundle_path(
        args.bundle,
        hf_repo=args.hf_repo,
        hf_revision=args.hf_revision,
        bundle_cache=args.bundle_cache,
    )

    access = YoutubeAccess(
        cookie_file=args.cookies,
        cookies_from_browser=args.cookies_from_browser,
        js_runtime=args.js_runtime,
    )
    if args.check_sources:
        source_report = check_sources(bundle_path, access)
        payload = json.dumps(source_report, indent=2)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(payload + "\n")
        print(payload)
        if not source_report["complete"]:
            raise SystemExit(2)
        return
    if args.out is None:
        parser.error("--out is required unless --check-sources is used")

    report = reconstruct_bundle(
        bundle_path,
        args.out,
        video_dir=args.video_dir,
        allow_partial=args.allow_partial,
        discard_videos=args.discard_videos,
        youtube_access=access,
    )
    payload = json.dumps(report.to_dict(), indent=2)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n")
    print(json.dumps(report.to_dict(include_complete_ids=False), indent=2))
    if not report.committed:
        print(
            "ERROR: SLCC reconstruction was not committed: "
            f"{len(report.missing_sample_ids)}/{len(report.expected_sample_ids)} samples "
            f"are missing. {report.manifest_path} was intentionally not created or "
            "replaced, so training must not be started. Run --check-sources and resolve "
            "the reported yt-dlp errors; if YouTube requests sign-in, pass "
            "--cookies-from-browser BROWSER or --cookies FILE to both commands.",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
