"""Installation invariants for native packages that share process-global state."""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest


OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
)


def _installed(distribution: str) -> bool:
    try:
        version(distribution)
    except PackageNotFoundError:
        return False
    return True


def test_at_most_one_opencv_distribution_is_installed():
    installed = [name for name in OPENCV_DISTRIBUTIONS if _installed(name)]
    assert len(installed) <= 1, (
        "multiple OpenCV wheels install the same cv2 package and can crash native "
        f"imports: {installed}"
    )


def test_lock_contains_one_opencv_distribution():
    lock = Path(__file__).parents[1] / "poetry.lock"
    packages = re.findall(r'^name = "(opencv[^\"]*)"$', lock.read_text(), re.MULTILINE)
    assert packages == ["opencv-contrib-python"]


@pytest.mark.skipif(
    any(
        importlib.util.find_spec(module) is None
        for module in ("cv2", "paddle", "paddleocr")
    ),
    reason="the optional annotation stack is not installed",
)
@pytest.mark.parametrize(
    "modules",
    [("cv2", "paddle", "paddleocr"), ("paddleocr", "paddle", "cv2")],
)
def test_opencv_and_paddle_import_safely_in_either_order(modules, tmp_path):
    environment = {
        **os.environ,
        "PADDLE_PDX_CACHE_HOME": str(tmp_path / "paddlex"),
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
    }
    completed = subprocess.run(
        [sys.executable, "-c", "; ".join(f"import {name}" for name in modules)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, (
        f"native import order {modules} exited {completed.returncode}: "
        f"{completed.stderr[-1000:]}"
    )
