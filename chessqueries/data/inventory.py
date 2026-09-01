"""Canonical sample inventories for the frozen paper datasets."""

from chessqueries.core import Split
from chessqueries.data.base import DatasetName

FROZEN_SAMPLE_COUNTS: dict[DatasetName, dict[Split | None, int]] = {
    DatasetName.CHESSRED: {
        Split.TRAIN: 6_479,
        Split.VAL: 2_192,
        Split.TEST: 2_129,
    },
    DatasetName.CHESSCOG: {
        Split.TRAIN: 4_400,
        Split.VAL: 146,
        Split.TEST: 342,
    },
    DatasetName.CVCHESS: {None: 352},
    DatasetName.SLCC: {
        Split.TRAIN: 1_475,
        Split.VAL: 326,
        Split.TEST: 373,
    },
}
