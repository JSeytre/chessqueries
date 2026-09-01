"""Recognition metrics."""
from chessqueries.metrics.recognition import (
    BoardMetrics,
    aggregate,
    aggregate_subsets,
    count_wrong_squares,
    score_board,
)

__all__ = ["BoardMetrics", "aggregate", "aggregate_subsets", "count_wrong_squares", "score_board"]
