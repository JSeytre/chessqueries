"""Board-vs-junk scorer over layout descriptors, so the labeler can sort new template
clusters board-first and pre-suggest the obvious junk (interviews, city/crowd shots) a
human would otherwise grade one by one.

A new venue is mostly *new* junk — the shared registry only auto-suppresses junk it has
already seen — so grading is the labeling bottleneck. This reuses the registry's own
labeled centroids as training data and scores a new cluster by a distance-weighted
k-NN over them, in the same unit-norm cosine space `templates` already clusters in (no
new dependency — matching `templates.cluster_descriptors`' deliberate no-sklearn stance).
Ranking is what matters (board-first sort); the score is only *suggested* — the human
still confirms every cluster, so a mis-scored board is caught, never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chessqueries.annotate.templates import LayoutRegistry, Quality

DEFAULT_K = 15
# Below this suggested P(board), the labeler pre-selects "junk" (one keypress to confirm).
# Conservative: the spike put k-NN junk-precision ~99% at this end, so a pre-suggested
# junk is almost never a real board — and the human still sees it regardless.
JUNK_SUGGEST_THRESHOLD = 0.10


@dataclass(frozen=True)
class JunkClassifier:
    """Distance-weighted cosine k-NN over the registry's labeled template centroids."""

    centroids: np.ndarray  # [N, D] unit-norm layout descriptors
    is_board: np.ndarray  # [N] float, 1.0 for useful/hard, 0.0 for junk
    k: int = DEFAULT_K

    @classmethod
    def from_registry(cls, registry: LayoutRegistry, k: int = DEFAULT_K) -> "JunkClassifier | None":
        """Fit on every labeled template. Returns None when there's nothing to learn
        from — an empty registry, or one with only a single class (all junk / all
        board) — so callers fall back to the unsorted queue."""
        ids = list(registry.centroids)
        if not ids:
            return None
        mat = np.asarray([registry.centroids[i] for i in ids], dtype=np.float32)
        board = np.asarray(
            [registry.layouts[i].quality is not Quality.JUNK for i in ids], dtype=np.float32
        )
        if board.min() == board.max():  # only one class present -> can't rank
            return None
        return cls(centroids=mat, is_board=board, k=min(k, len(ids)))

    def score_board(self, centroid) -> float:
        """P(board) for one unit-norm cluster centroid."""
        return float(self.score_many(centroid)[0])

    def score_many(self, centroids) -> np.ndarray:
        """P(board) for a batch of centroids ([M, D] -> [M])."""
        c = np.asarray(centroids, dtype=np.float32)
        if c.ndim == 1:
            c = c[None, :]
        sims = c @ self.centroids.T  # [M, N] cosine (both sides unit-norm)
        idx = np.argpartition(-sims, kth=self.k - 1, axis=1)[:, : self.k]  # top-k neighbours
        rows = np.arange(c.shape[0])[:, None]
        w = np.clip(sims[rows, idx], 0.0, None)  # weight by similarity; opposites contribute 0
        lbl = self.is_board[idx]
        wsum = w.sum(axis=1)
        # If every top-k neighbour is orthogonal/opposite (wsum==0), fall back to the
        # unweighted neighbour majority rather than dividing by zero.
        return np.where(wsum > 0, (w * lbl).sum(axis=1) / np.where(wsum > 0, wsum, 1), lbl.mean(axis=1))
