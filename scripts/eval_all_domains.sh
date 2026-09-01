#!/usr/bin/env bash
# Evaluate one ChessQueries checkpoint across every test domain and write the
# per-domain metrics JSONs (outputs/chessqueries_<name>/metrics_<domain>.json).
#
# Usage: eval_all_domains.sh <checkpoint_file_or_dir> <name> [resolution]
# A directory selects its highest-val_board_acc checkpoint (then last.ckpt).
set -euo pipefail

checkpoint="$1"; name="$2"; res="${3:-644}"
if [[ -f "$checkpoint" ]]; then
  best="$checkpoint"
else
  # LC_ALL=C: under a comma-decimal locale (LC_NUMERIC=fr_FR) `sort -g` parses
  # "0.9655" as 0, every key ties, and the pick degenerates to highest-epoch.
  best=$(ls "$checkpoint"/*val_board_acc=*.ckpt 2>/dev/null | LC_ALL=C sort -t= -k3 -g | tail -1 || true)
  [ -z "${best:-}" ] && best="$checkpoint/last.ckpt"
fi
if [[ ! -f "$best" ]]; then
  echo "ERROR: checkpoint not found: $best" >&2
  exit 2
fi
echo "eval_all_domains: name=$name res=$res checkpoint=$best"

for pair in "chessred test" "chesscog test" "slcc test"; do
  set -- $pair
  poetry run python scripts/eval_cross_dataset.py \
    --checkpoint "$best" --dataset "$1" --split "$2" --resolution "$res" --name "$name"
done
# CVChess has no split (zero-shot, one game).
poetry run python scripts/eval_cross_dataset.py \
  --checkpoint "$best" --dataset cvchess --resolution "$res" --name "$name"
echo "eval_all_domains DONE: $name"
