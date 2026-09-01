"""Single front door for the broadcast-dataset workflow:

    python -m chessqueries.annotate            # == status: what's pending
    python -m chessqueries.annotate add <url>… # register videos
    python -m chessqueries.annotate ingest      # download + segment + detect templates
    python -m chessqueries.annotate label       # grade the new templates (Gradio)
    python -m chessqueries.annotate produce      # clock-OCR auto-label -> candidates
    python -m chessqueries.annotate crosscheck   # visual model vs. clock -> accept/review/quarantine
    python -m chessqueries.annotate review       # verify candidates (Gradio)
    python -m chessqueries.annotate reconstruct --rebuild-all  # assemble the full SLCC dataset

`status` is the only command you must remember; it names the next one to run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chessqueries.annotate import workflow
from chessqueries.annotate import video as video_mod
from chessqueries.annotate.manifest import (
    DEFAULT_VIDEOS_PATH,
    Manifest,
    VideoEntry,
    fetch_youtube_metadata,
    video_id_from_url,
)
from chessqueries.annotate.pipeline import DEFAULT_PRODUCE_WORKERS
from chessqueries.annotate.templates import DEFAULT_REGISTRY_PATH


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--manifest", type=Path, default=DEFAULT_VIDEOS_PATH)
    p.add_argument("--data-dir", type=Path, default=workflow.DEFAULT_DATA_DIR)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    p.add_argument("--format-id", default=video_mod.DEFAULT_FORMAT_ID)


def _cmd_status(args) -> None:
    manifest = Manifest.load(args.manifest)
    sv = workflow.survey(
        manifest,
        data_dir=args.data_dir,
        registry_path=args.registry,
        format_id=args.format_id,
        show_progress=True,
    )
    print(workflow.render_status(manifest, sv))


def _cmd_add(args) -> None:
    manifest = Manifest.load(args.manifest)
    added = 0
    for url in args.urls:
        vid = video_id_from_url(url)
        if vid in manifest:
            print(f"  • {vid} already registered")
            continue
        meta = {"title": args.title or "", "date": ""} if args.no_fetch else fetch_youtube_metadata(vid)
        manifest.add(
            VideoEntry(
                video_id=vid,
                title=args.title or meta["title"],
                date=meta["date"],
                tournaments=list(args.tournaments or []),
                round_ids=list(args.round_ids or []),
            )
        )
        added += 1
        print(f"  + {vid}  {meta['title'] or '(no title)'}")
    if added:
        manifest.save()
        print(f"registered {added} video(s) -> {args.manifest}")
    if not (args.tournaments or args.round_ids):
        print("note: set `tournaments` (or `round_ids`) in the manifest before `produce`.")


def _cmd_ingest(args) -> None:
    manifest = Manifest.load(args.manifest)
    workflow.ingest(
        manifest,
        data_dir=args.data_dir,
        video_ids=args.video,
        format_id=args.format_id,
        max_frames=args.max_frames,
    )


def _cmd_label(args) -> None:
    manifest = Manifest.load(args.manifest)
    queue_dir = args.queue_dir or (args.data_dir / "_label_queue")
    print("scanning segmented videos for new templates (clustering + keyframes)...")
    n = workflow.build_label_queue(
        manifest,
        queue_dir,
        data_dir=args.data_dir,
        registry_path=args.registry,
        format_id=args.format_id,
    )
    if n == 0:
        print("no new templates to label.")
        return
    from chessqueries.annotate.labeler import build_app

    print(f"serving labeler for {n} new template(s) from {queue_dir}")
    build_app(queue_dir, registry_path=args.registry).launch()


def _cmd_produce(args) -> None:
    manifest = Manifest.load(args.manifest)
    workflow.produce(
        manifest,
        data_dir=args.data_dir,
        registry_path=args.registry,
        video_ids=args.video,
        format_id=args.format_id,
        force=args.force,
        salvage=args.salvage,
        workers=args.workers,
    )


def _cmd_crosscheck(args) -> None:
    from chessqueries.annotate.workflow import (
        RecognizerKind,
        RecognizerRef,
        load_recognizer,
        save_recognizer,
    )

    manifest = Manifest.load(args.manifest)
    if args.checkpoint:
        ref = RecognizerRef(RecognizerKind.CHECKPOINT, args.checkpoint, args.resolution)
    elif args.adapter:
        ref = RecognizerRef(RecognizerKind.ADAPTER, args.adapter)
    else:  # neither given -> reuse the recognizer remembered from a prior run
        ref = load_recognizer(args.data_dir)
        if ref is None:
            raise SystemExit(
                "no recognizer given and none remembered — pass --checkpoint <v2.ckpt> "
                "(--resolution defaults to 644) or --adapter <lora.pt> once; it'll be saved."
            )
        print(f"using remembered recognizer: {ref.crosscheck_command()}")
    save_recognizer(ref, args.data_dir)  # remember the latest choice for next time
    is_ckpt = ref.kind is RecognizerKind.CHECKPOINT
    workflow.crosscheck(
        manifest,
        adapter=None if is_ckpt else ref.path,
        checkpoint=ref.path if is_ckpt else None,
        resolution=ref.resolution if is_ckpt else None,
        data_dir=args.data_dir,
        video_ids=args.video,
        format_id=args.format_id,
        desync=args.desync,
        tau_fit=args.tau_fit,
        tau_margin=args.tau_margin,
        force=args.force,
    )


def _cmd_review(args) -> None:
    from chessqueries.annotate.review import build_app

    manifest = Manifest.load(args.manifest)
    sv = workflow.survey(
        manifest, data_dir=args.data_dir, registry_path=args.registry, format_id=args.format_id
    )
    queue = workflow.pending_reviews(sv)
    if args.video:  # an explicit video overrides the queue (review just that one)
        queue = [s for s in queue if s.video_id == args.video]
        if not queue:
            print(f"{args.video}: nothing to review (not produced, or already fully reviewed).")
            return
    if not queue:
        print("✓ nothing to review — all produced videos are fully reviewed.")
        return

    # Walk the queue in one session: launch each review non-blocking, wait at the terminal,
    # and load the next on Enter (the UI autosaves, so nothing is lost between videos).
    for position, st in enumerate(queue, 1):
        vid = st.video_id
        print("\n" + workflow.review_queue_lines(queue, position=position))
        src = workflow.crosschecked_path(args.data_dir, vid)
        if not src.exists():  # fall back to raw candidates if no cross-check was run
            src = workflow.candidates_path(args.data_dir, vid)
        vpath = workflow.video_files(args.data_dir, vid, args.format_id)[0]
        print(f"reviewing from {src.name}")
        demo = build_app(src, vpath)
        demo.launch(prevent_thread_lock=True)
        url = getattr(demo, "local_url", "http://127.0.0.1:7860")
        last = position == len(queue)
        prompt = (
            f"\n▶ open {url}\n  finished? press Enter to "
            + ("finish" if last else f"load the next video ({queue[position].video_id})")
            + " — Ctrl-C to stop here\n"
        )
        try:
            input(prompt)
        except (KeyboardInterrupt, EOFError):
            print("\nstopped — rerun `annotate review` to pick up where you left off.")
            demo.close()
            return
        demo.close()
        if not last:
            print("↻ refresh the browser tab for the next video")
    print("\n✓ reached the end of the review queue.")


def _cmd_reconstruct(args) -> None:
    from chessqueries.data.slcc import SplitRatio

    manifest = Manifest.load(args.manifest)
    workflow.export_dataset(
        manifest,
        SplitRatio.from_str(args.ratio),
        data_dir=args.data_dir,
        out_dir=args.out,
        video_ids=args.video,
        verified_only=not args.include_unverified,
        seed=args.seed,
        split_by=args.split_by,
        regroup=args.regroup,
        dedup=not args.keep_duplicates,
        rebuild_all=args.rebuild_all,
        format_id=args.format_id,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="annotate", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("status", help="what's pending (default)")
    _add_common(p)
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser("add", help="register videos by URL/id")
    _add_common(p)
    p.add_argument("urls", nargs="+", help="YouTube watch URLs or 11-char ids")
    p.add_argument("--tournament", dest="tournaments", nargs="+", help="Lichess broadcast tour id(s)")
    p.add_argument("--round-id", dest="round_ids", nargs="+", help="narrow to specific round ids")
    p.add_argument("--title", default=None)
    p.add_argument("--no-fetch", action="store_true", help="don't call yt-dlp for title/date")
    p.set_defaults(func=_cmd_add)

    p = sub.add_parser("ingest", help="download + segment + detect templates")
    _add_common(p)
    p.add_argument("--video", nargs="+", help="limit to these video ids (default: all)")
    p.add_argument("--max-frames", type=int, default=None, help="cap decode (quick smoke test)")
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser("label", help="grade the pooled new templates (Gradio)")
    _add_common(p)
    p.add_argument("--queue-dir", type=Path, default=None)
    p.set_defaults(func=_cmd_label)

    p = sub.add_parser("produce", help="clock-OCR auto-label -> candidates")
    _add_common(p)
    p.add_argument("--video", nargs="+", help="limit to these video ids (default: all ready)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true", help="re-produce even if candidates exist")
    mode.add_argument("--salvage", action="store_true",
                      help="revisit produced+reviewed videos; keep prior decisions and surface "
                           "ONLY frames newly rescued by the current matcher")
    p.add_argument("--workers", type=int, default=DEFAULT_PRODUCE_WORKERS,
                   help="shot-labeling processes (decode+OCR are single-core; 1 = in-process)")
    p.set_defaults(func=_cmd_produce)

    p = sub.add_parser("crosscheck", help="visual model vs. clock consensus -> triage candidates")
    _add_common(p)
    src = p.add_mutually_exclusive_group(required=False)  # omit both -> reuse remembered recognizer
    src.add_argument("--adapter", type=Path, help="LoRA adapter (.pt) from lora_fewshot.py")
    src.add_argument("--checkpoint", type=Path, help="full LitChessQueriesModel checkpoint (e.g. joint V2)")
    p.add_argument("--resolution", type=int, default=644,
                   help="eval resolution for --checkpoint (must match training; ViT-L V2 = 644)")
    p.add_argument("--video", nargs="+", help="limit to these video ids (default: all produced)")
    p.add_argument("--desync", type=int, default=None, help="ply window padding (default 3)")
    p.add_argument("--tau-fit", type=int, default=None, help="max wrong squares to accept (default 4)")
    p.add_argument("--tau-margin", type=float, default=None, help="min log-prob margin (default 5)")
    p.add_argument("--force", action="store_true", help="re-run even if crosschecked exists")
    p.set_defaults(func=_cmd_crosscheck)

    p = sub.add_parser("review", help="verify candidates (Gradio)")
    _add_common(p)
    p.add_argument("--video", help="review just this video id (default: walk all pending)")
    p.set_defaults(func=_cmd_review)

    p = sub.add_parser("reconstruct", help="assemble reviewed labels into the split-tagged SLCC dataset")
    _add_common(p)
    p.add_argument("ratio", nargs="?", default="80/10/10",
                   help="train/val/test split (percentage points, sum 100) for NEW frames; "
                        "existing frames keep their split. e.g. 60/30/10, or 0/0/100 for all-test")
    scope = p.add_mutually_exclusive_group(required=True)
    scope.add_argument("--video", nargs="+", help="replace only these videos")
    scope.add_argument(
        "--rebuild-all",
        action="store_true",
        help="intentionally replace the complete dataset from all reviewed videos",
    )
    p.add_argument("--out", type=Path, default=None, help="dataset dir (default: data/slcc/dataset)")
    p.add_argument("--include-unverified", action="store_true",
                   help="also export frames kept on review but not human-verified")
    p.add_argument("--split-by", type=workflow.SplitBy, choices=list(workflow.SplitBy), default=workflow.SplitBy.GAME,
                   help="leakage-grouping unit: keep a whole game (default) / video / "
                        "nothing (frames shuffle independently; permits same-game leakage)")
    p.add_argument("--regroup", action="store_true",
                   help="one-time clean re-split: re-plan ALL splits by group, ignoring "
                        "prior assignments (otherwise existing groups keep their split)")
    p.add_argument("--keep-duplicates", action="store_true",
                   help="ship every verified frame; skip collapsing same-view near-duplicate "
                        "positions (same game+viewpoint+placement) to their sharpest frame")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_reconstruct)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):  # bare `annotate` -> status
        args = parser.parse_args(["status", *(argv or [])])
    if args.cmd == "reconstruct" and args.regroup and not args.rebuild_all:
        parser.error("reconstruct --regroup requires --rebuild-all")
    args.func(args)


if __name__ == "__main__":
    main()
