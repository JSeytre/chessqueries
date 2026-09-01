"""Shared eval-script reporting: the metric printout and the metrics/preds JSON
pair every eval entry point writes (consumed by the visualizer and the paper
tooling), plus the CLI split resolution they all repeat."""
from __future__ import annotations

import json
from pathlib import Path

from chessqueries.core import Split
from chessqueries.data.base import BoardSample, ChessDataset, DatasetCompleteness


def eval_tag(dataset_value: str, split: Split | None, sep: str = "_") -> str:
    """The ``<dataset>_<split>`` slug that names metrics/preds files."""
    return f"{dataset_value}{sep}{split.value if split else 'all'}"


def resolve_split(ds_obj: ChessDataset, split_arg: str | None) -> Split | None:
    """CLI boundary: the dataset's `Split` for a ``--split`` string, or None for
    split-less datasets. Exits with a usage message when a required split is missing."""
    split = Split(split_arg) if (ds_obj.splits and split_arg) else None
    if ds_obj.splits and split is None:
        raise SystemExit(
            f"{ds_obj.name.value} requires --split {[s.value for s in ds_obj.splits]}"
        )
    return split


def print_metrics(metrics: dict, header: str) -> None:
    print(f"\n=== {header} ===")
    for k, v in metrics.items():
        print(f"  {k:32s} {v:.4f}" if isinstance(v, float) else f"  {k:32s} {v}")


def evaluation_inventory(
    completeness: DatasetCompleteness,
    *,
    evaluated_samples: int | None = None,
    expected_evaluated_samples: int | None = None,
) -> dict:
    """Serializable strict/partial-data status for an evaluation output."""
    return completeness.as_dict(
        evaluated_samples=evaluated_samples,
        expected_evaluated_samples=expected_evaluated_samples,
    )


def print_evaluation_inventory(inventory: dict) -> None:
    """Make strict, subset, and diagnostic partial runs unmistakable in logs."""
    dataset = inventory.get("dataset") or "unknown"
    split = inventory.get("split") or "all"
    mode = inventory["mode"].upper()
    paper_comparable = (
        inventory["mode"] == "strict"
        and inventory["actual_samples"] == inventory["expected_samples"]
    )
    status = "PAPER-COMPARABLE" if paper_comparable else "NON-PAPER"
    scope = inventory["scope"].replace("_", " ").upper()
    print(
        f"data {dataset}[{split}] [{mode}; {scope}; {status}]: "
        f"expected={inventory['expected_samples']}  "
        f"labelled={inventory['labelled_samples']}  "
        f"available={inventory['available_samples']}  "
        f"requested={inventory['expected_evaluated_samples']}  "
        f"evaluated={inventory['actual_samples']}"
    )


def write_eval_outputs(
    out_dir: Path,
    tag: str,
    metrics: dict,
    samples: list[BoardSample],
    preds: list[list[int]],
) -> Path:
    """Write ``metrics_<tag>.json`` + ``preds_<tag>.json`` (sample_id -> 64 labels)."""
    pairs = list(zip(samples, preds, strict=True))
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for sample, _ in pairs:
        if sample.sample_id in seen_ids:
            duplicate_ids.add(sample.sample_id)
        seen_ids.add(sample.sample_id)
    if duplicate_ids:
        raise ValueError(f"duplicate sample IDs: {sorted(duplicate_ids)}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))
    predictions = {sample.sample_id: pred for sample, pred in pairs}
    (out_dir / f"preds_{tag}.json").write_text(json.dumps(predictions))
    return out_dir
