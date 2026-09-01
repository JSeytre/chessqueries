"""Dataset splits."""
from __future__ import annotations

from enum import Enum


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
