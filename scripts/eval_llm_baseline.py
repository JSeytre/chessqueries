"""Evaluate a prompted frontier VLM on any dataset split, zero-shot.

Sends each test image to the model with an instruction to read off the position,
then scores the returned FEN with the same metrics as our own recognizers, and
writes the same ``metrics_<domain>_<split>.json`` / ``preds_...json`` pair, so an
LLM row drops straight into the paper tables.

Every completed sample is appended to a JSONL cache in the output directory:
re-running the same command resumes rather than re-paying for finished images.
Needs the optional ``llm`` group (``poetry install --with llm``) and the API key
for the chosen model's provider -- ``ANTHROPIC_API_KEY`` for ``claude-*``,
``OPENAI_API_KEY`` for ``gpt-*``.

Usage:
    # costing probe: two images live, print the per-image cost
    poetry run python scripts/eval_llm_baseline.py --dataset slcc --split test \
        --limit 2 --effort low --image-size 644

    # a paper cell, via the Batch API (half price, asynchronous)
    poetry run python scripts/eval_llm_baseline.py --dataset chessred --split test \
        --limit 400 --effort low --image-size 644 --batch --wait

    # the same cell from a different provider: only --model changes
    poetry run python scripts/eval_llm_baseline.py --dataset chessred --split test \
        --limit 400 --model gpt-5.6-terra --effort low --image-size 644 --batch --wait
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from chessqueries.config import get_config
from chessqueries.core import Board
from chessqueries.data import DatasetIncompleteError, DatasetName, get_dataset
from chessqueries.metrics import aggregate
from chessqueries.metrics.report import (
    eval_tag,
    evaluation_inventory,
    print_evaluation_inventory,
    print_metrics,
    resolve_split,
)
from chessqueries.models import llm_batch
from chessqueries.models.llm import (
    BATCH_DISCOUNT,
    PROMPT_SHA256,
    PROMPT_VERSION,
    SCHEMA_SHA256,
    SCHEMA_VERSION,
    Effort,
    ImagePrep,
    LLMName,
    LLMPrediction,
    Usage,
    get_reader,
    outcome_breakdown,
)


def _load_cache(path: Path) -> dict[str, LLMPrediction]:
    """Read the append-only JSONL cache; later records win over earlier ones."""
    if not path.is_file():
        return {}
    out: dict[str, LLMPrediction] = {}
    torn = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            pred = LLMPrediction.from_dict(json.loads(line))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # A record cut short by a kill mid-write, or one a newer version wrote
            # with fields we cannot read. Skip the line and re-run that sample --
            # never abort the whole run (and re-pay for it) over one bad line.
            torn += 1
            continue
        out[pred.sample_id] = pred
    if torn:
        print(f"cache: skipped {torn} truncated record(s) in {path}")
    return out


def _await_and_collect(reader, job, job_path: Path, record, *, wait: bool,
                       poll_seconds: int) -> bool:
    """Poll `job` to completion and record its results.

    Returns False if work is still processing and we were told not to wait -- the
    job file stays on disk so a later run resumes it instead of resubmitting.
    On success the job file is REMOVED: leaving it behind would make every later
    invocation re-collect the same finished batches and silently never submit new
    work (so a cell with expired/errored requests could never be finished).
    """
    while True:
        states = llm_batch.status(reader, job)
        for st in states:
            print(f"  {st.batch_id}  {st.status}  {st.counts}")
        if all(st.terminal for st in states):
            break
        if not wait:
            print(f"still processing -- re-run this command (or add --wait) to collect.\n"
                  f"batch ids recorded in {job_path}")
            return False
        time.sleep(poll_seconds)

    n = 0
    for pred in llm_batch.collect(reader, job):
        record(pred)
        n += 1
    print(f"collected {n} result(s) from {len(job.batch_ids)} batch(es)")
    job_path.unlink(missing_ok=True)
    return True


def _run_batch(reader, samples, needs_run, job_path: Path, record, *,
               wait: bool, poll_seconds: int) -> bool:
    """Finish any outstanding submission, then submit whatever still needs running.

    Two phases, in that order, because an outstanding job must be collected before
    we can tell what is left: a resumed run recomputes `needs_run` *after*
    recording the old batch's results, so it neither resubmits work already paid
    for nor skips work that batch never covered (e.g. a smaller `--limit` probe).
    """
    if job_path.is_file():
        job = llm_batch.BatchJob.from_dict(json.loads(job_path.read_text()))
        print(f"resuming batches {job.batch_ids}")
        if not _await_and_collect(reader, job, job_path, record,
                                  wait=wait, poll_seconds=poll_seconds):
            return False

    todo = [s for s in samples if needs_run(s)]
    if not todo:
        return True

    def persist(job) -> None:
        job_path.write_text(json.dumps(job.as_dict(), indent=2))

    submission = llm_batch.submit(reader, todo, on_submitted=persist)
    job = submission.job
    for pred in submission.skipped:
        record(pred)
    if not job.batch_ids:
        return True
    print(f"submitted {len(job.batch_ids)} batch(es) for {len(todo)} sample(s): {job.batch_ids}")
    return _await_and_collect(reader, job, job_path, record,
                              wait=wait, poll_seconds=poll_seconds)


_SPEND_WARNING = """
COSTS REAL MONEY. Every sample not already in the cache is a paid API call, and
samples that never reached the model (a canceled/expired batch) are re-queued by
default -- so a command you think only re-scores can start spending. Guard rails:

  --score-only    score the cache, make NO API calls at all (works without a key)
  --limit N       N evenly-spaced samples; check the printed $/image first
  --batch         half price, asynchronous
  --wait          absent, --batch submits and returns; re-run to collect

Thinking tokens dominate cost and scale with how legible the board is, not with
--effort alone -- a per-image cost measured on one dataset does NOT transfer to
another. Probe each dataset with --limit first (measured per-image references
live in the local results log).
"""


def _positive_limit(value: str) -> int:
    limit = int(value)
    if limit <= 0:
        raise argparse.ArgumentTypeError("--limit must be greater than zero")
    return limit


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__ + _SPEND_WARNING,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, type=DatasetName, choices=list(DatasetName))
    p.add_argument("--split", default=None, choices=["train", "val", "test"])
    p.add_argument("--model", type=LLMName, choices=list(LLMName), default=LLMName.CLAUDE_OPUS_5)
    p.add_argument("--effort", type=Effort, choices=list(Effort), default=Effort.HIGH)
    p.add_argument("--image-size", default="native",
                   help="'native' (downscaled only to Claude's 2576px cap) or an integer N "
                        "for an NxN square resize matching the ViT eval transform.")
    p.add_argument("--max-tokens", type=int, default=32000)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=8, help="Live mode only.")
    p.add_argument("--batch", action="store_true",
                   help="Deliver via the Batch API: half price, asynchronous. Submits on the "
                        "first invocation and records the batch ids; re-run the same command "
                        "to poll and collect (add --wait to block until they finish).")
    p.add_argument("--wait", action="store_true",
                   help="With --batch: poll until every batch ends instead of returning.")
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--retry-failed", action="store_true",
                   help="Also re-run samples where the model produced a turn we could not use "
                        "(refusal, truncation, malformed placement). Samples that never reached "
                        "the model are always re-run -- that is finishing the job, not a retry.")
    p.add_argument("--score-only", action="store_true",
                   help="Score whatever the cache already holds and make NO API calls. Use this "
                        "to (re)write a cell's metrics without spending anything.")
    p.add_argument("--allow-partial", action="store_true",
                   help="Diagnostic only: allow missing dataset images or model responses. "
                        "The output is visibly marked non-paper-comparable.")
    p.add_argument("--limit", type=_positive_limit, default=None,
                   help="Costing probe: score N evenly-spaced samples instead of the whole "
                        "split, so the estimate is not dominated by one game.")
    p.add_argument("--name", default=None,
                   help="Run tag (default: <model>_<effort>_<image-size>). Names the output dir.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: outputs/llm_<name>. Holds metrics, preds and the resume cache.")
    p.add_argument("--no-cache", action="store_true", help="Ignore and do not write the resume cache.")
    args = p.parse_args(argv)

    cfg = get_config()
    prep = ImagePrep.parse(args.image_size)
    ds_obj = get_dataset(args.dataset)
    split = resolve_split(ds_obj, args.split)

    try:
        loaded = ds_obj.load_with_report(split, allow_partial=args.allow_partial)
    except DatasetIncompleteError as exc:
        p.error(str(exc))
    samples = loaded.samples
    n_total = len(samples)
    if args.limit is not None and args.limit < n_total:
        stride = n_total / args.limit
        samples = [samples[int(i * stride)] for i in range(args.limit)]

    name = args.name or f"{args.model.value}_{args.effort.value}_{prep.label}"
    out_dir = args.out_dir or cfg.OUTPUTS_ROOT / f"llm_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = eval_tag(args.dataset.value, split)
    cache_path = out_dir / f"cache_{slug}.jsonl"

    reader = get_reader(args.model, effort=args.effort, prep=prep,
                        max_tokens=args.max_tokens, max_attempts=args.max_attempts)

    cached = {} if args.no_cache else _load_cache(cache_path)

    def needs_run(sample) -> bool:
        pred = cached.get(sample.sample_id)
        if pred is None:
            return True
        if not pred.attempted:
            return True   # never reached the model: finishing the job, not a retry
        return args.retry_failed and not pred.ok

    todo = [] if args.score_only else [s for s in samples if needs_run(s)]
    mode = ("score-only (no API calls)" if args.score_only
            else "batch" if args.batch else f"live (concurrency={args.concurrency})")
    print(f"{args.model.value} effort={args.effort.value} image={prep.label} "
          f"max_tokens={args.max_tokens} -> {out_dir}")
    print(f"{slug}: {len(samples)} samples, {len(cached)} cached, {len(todo)} to run [{mode}]")

    sink = None if args.no_cache else cache_path.open("a")
    lock = threading.Lock()

    def record(pred: LLMPrediction) -> LLMPrediction:
        with lock:
            cached[pred.sample_id] = pred
            if sink is not None:
                sink.write(json.dumps(pred.as_dict()) + "\n")
                sink.flush()
        return pred

    try:
        if args.score_only:
            pass  # deliberately no API calls: score the cache as it stands
        elif args.batch:
            if not _run_batch(reader, samples, needs_run, out_dir / f"batch_{slug}.json",
                              record, wait=args.wait, poll_seconds=args.poll_seconds):
                return  # still processing; re-run to collect
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                results = pool.map(lambda s: record(reader.read(s.image_path, s.sample_id)), todo)
                for _ in tqdm(results, total=len(todo), desc=slug):
                    pass
    finally:
        if sink is not None:
            sink.close()

    # Only samples the model actually produced a turn for can be scored. A
    # never-run request (batch canceled/expired, unreadable image) is NOT scored:
    # counting it as a wrong answer would invent a result the model never gave.
    scored = [cached[s.sample_id] for s in samples
              if s.sample_id in cached and cached[s.sample_id].attempted]
    scored_ids = {pr.sample_id for pr in scored}
    not_run = [s.sample_id for s in samples if s.sample_id not in scored_ids]
    if not_run:
        print(f"\n{len(not_run)} of {len(samples)} sample(s) were never run "
              f"(e.g. {not_run[:3]}) -- re-run with --retry-failed to finish them.")
        if not args.allow_partial:
            raise SystemExit("refusing to write a partial cell; pass --allow-partial "
                             "to score only what ran (n is recorded in the metrics).")
    if not scored:
        raise SystemExit("nothing to score yet.")

    # A turn the model produced but we could not parse IS a model failure, so it
    # is scored as an empty board -- a prediction it could legitimately have made
    # -- and counted separately, never silently a free pass.
    gt = {s.sample_id: s.board.labels for s in samples}
    failed = [pr for pr in scored if not pr.ok]
    metrics = aggregate([(pr.board or Board.empty()).labels for pr in scored],
                        [gt[pr.sample_id] for pr in scored])
    metrics.update(outcome_breakdown(scored, gt))
    metrics["n_requested"] = len(samples)
    metrics["n_not_run"] = len(not_run)
    inventory = evaluation_inventory(
        loaded.completeness,
        expected_evaluated_samples=len(samples),
        evaluated_samples=len(scored),
    )
    metrics["evaluation"] = inventory

    # Price each sample by how *it* was delivered: a cell that mixes a batch run
    # with a live top-up would otherwise be costed entirely at one rate.
    usage = sum((pr.usage for pr in scored), Usage())
    live, batch_price = reader.pricing, reader.pricing.scaled(BATCH_DISCOUNT)
    cost = sum(pr.usage.cost_usd(batch_price if pr.batch else live) for pr in scored)
    n_batch = sum(1 for pr in scored if pr.batch)
    run_info = {
        "model": args.model.value, "provider": reader.provider.value,
        "prompt_version": PROMPT_VERSION, "prompt_sha256": PROMPT_SHA256,
        "schema_version": SCHEMA_VERSION, "schema_sha256": SCHEMA_SHA256,
        "effort": args.effort.value,
        "delivery": ("batch" if n_batch == len(scored) else
                     "live" if n_batch == 0 else f"mixed ({n_batch} batch)"),
        "image_size": prep.label, "max_tokens": args.max_tokens,
        "dataset": args.dataset.value, "split": split.value if split else None,
        "n_scored": len(scored), "n_requested": len(samples), "n_in_split": n_total,
        "n_not_run": len(not_run), "n_failed_parse": len(failed),
        "evaluation": inventory,
        "usage": usage.as_dict(), "cost_usd": cost,
        "cost_usd_per_image": cost / len(scored),
        "projected_cost_usd_full_split": cost / len(scored) * n_total,
    }
    # Provider-specific identity (a local server's served checkpoint and sampling
    # settings, which `--model` does not pin).
    run_info.update(reader.run_info_extra())

    (out_dir / f"metrics_{slug}.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / f"preds_{slug}.json").write_text(json.dumps(
        {pr.sample_id: (pr.board or Board.empty()).labels for pr in scored}))
    (out_dir / f"run_{slug}.json").write_text(json.dumps(run_info, indent=2))

    print_metrics(
        {key: value for key, value in metrics.items() if key != "evaluation"},
        f"{args.model.value} ({name}) on {slug} | n={len(scored)}"
        f"{f' of {len(samples)} requested' if not_run else ''}",
    )
    print_evaluation_inventory(inventory)
    print(f"\n  {metrics['n_parsed_wrong']} of {len(scored)} "
          f"({100 * metrics['parsed_wrong_fraction']:.1f}%) were well-formed FEN but the "
          f"wrong position; {len(failed)} ({100 * metrics['failed_parse_fraction']:.1f}%) "
          f"were unparseable")
    reasoning = (f"  reasoning={usage.reasoning_tokens / len(scored):.0f}"
                 if usage.reasoning_tokens else "")
    print(f"\n  tokens/image  in={usage.input_tokens / len(scored):.0f}  "
          f"out={usage.output_tokens / len(scored):.0f}{reasoning}")
    print(f"  cost          ${cost:.4f} total  ${cost / len(scored):.4f}/image  "
          f"-> ${cost / len(scored) * n_total:.2f} for the full {n_total}-image split")
    if failed:
        print(f"  {len(failed)} unparseable (scored as empty boards), e.g. {failed[0].error}")
    print(f"\nWrote metrics + predictions to {out_dir} ({slug})")


if __name__ == "__main__":
    main()
