"""CLI guards that must fail before data, models, or paid providers are touched."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from chessqueries.core import Board
from chessqueries.data.base import BoardSample, DatasetName
from chessqueries.models.llm import (
    PROMPT_SHA256,
    SCHEMA_SHA256,
    LLMPrediction,
    Pricing,
    Provider,
    Usage,
)

SCRIPTS = Path(__file__).parents[1] / "scripts"
MAINTAINER_ONLY_SCRIPTS = {"export_chesscog_manifest", "run_chesscog_on_manifest"}


def _load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    if not path.is_file():
        if name in MAINTAINER_ONLY_SCRIPTS:
            pytest.skip(f"{name} is maintainer-only and absent from the release tree")
        pytest.fail(f"shipped script is missing from the tree: {name}")
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_llm_limit_must_be_positive_before_setup(limit, monkeypatch, capsys):
    module = _load_script("eval_llm_baseline")
    monkeypatch.setattr(
        module, "get_config", lambda: pytest.fail("setup ran after an invalid --limit")
    )

    with pytest.raises(SystemExit) as exc_info:
        module.main(["--dataset", "cvchess", "--limit", limit])

    assert exc_info.value.code == 2
    assert "--limit must be greater than zero" in capsys.readouterr().err


@pytest.mark.parametrize("limit", ["0", "-1"])
@pytest.mark.parametrize(
    "script,args",
    [
        ("eval_baseline", ["--checkpoint", "unused.ckpt"]),
        ("export_chesscog_manifest", ["--dataset", "cvchess"]),
        ("run_chesscog_on_manifest", ["--manifest", "unused.json", "--out", "out.json"]),
    ],
)
def test_other_evaluation_limits_must_be_positive(
    script, args, limit, monkeypatch, capsys
):
    module = _load_script(script)
    monkeypatch.setattr(sys, "argv", [f"{script}.py", *args, "--limit", limit])

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert exc_info.value.code == 2
    assert "--limit must be greater than zero" in capsys.readouterr().err


def test_llm_run_record_includes_prompt_and_schema_versions(tmp_path, monkeypatch):
    module = _load_script("eval_llm_baseline")
    sample = BoardSample(
        image_path=tmp_path / "unused.png",
        board=Board.empty(),
        dataset=DatasetName.CVCHESS,
        sample_id="s0",
    )

    class Dataset:
        name = DatasetName.CVCHESS
        splits = ()

        @staticmethod
        def load_with_report(split, *, allow_partial=False):
            assert split is None
            from chessqueries.data.base import DatasetCompleteness, DatasetLoad

            completeness = DatasetCompleteness(
                dataset=DatasetName.CVCHESS,
                split=None,
                expected_samples=1,
                labelled_samples=1,
                available_samples=1,
                missing_sample_ids=(),
                allow_partial=allow_partial,
            )
            return DatasetLoad([sample], completeness)

    class Reader:
        provider = Provider.LOCAL
        pricing = Pricing(0.0, 0.0, 0.0)

        @staticmethod
        def read(image_path, sample_id):
            return LLMPrediction(sample_id, Board.empty(), "8/8/8/8/8/8/8/8", Usage(), 1)

        @staticmethod
        def run_info_extra():
            return {}

    monkeypatch.setattr(module, "get_config", object)
    monkeypatch.setattr(module, "get_dataset", lambda name: Dataset())
    monkeypatch.setattr(module, "get_reader", lambda *args, **kwargs: Reader())

    out_dir = tmp_path / "run"
    module.main(
        [
            "--dataset",
            "cvchess",
            "--model",
            "local",
            "--effort",
            "none",
            "--out-dir",
            str(out_dir),
            "--no-cache",
        ]
    )

    run_info = json.loads((out_dir / "run_cvchess_all.json").read_text())
    assert run_info["prompt_version"] == "v1"
    assert run_info["prompt_sha256"] == PROMPT_SHA256
    assert run_info["schema_version"] == "v1"
    assert run_info["schema_sha256"] == SCHEMA_SHA256
    assert run_info["evaluation"] == {
        "dataset": "cvchess",
        "split": None,
        "mode": "strict",
        "scope": "full_split",
        "data_complete": True,
        "expected_samples": 1,
        "expected_evaluated_samples": 1,
        "labelled_samples": 1,
        "available_samples": 1,
        "actual_samples": 1,
        "missing_images": 0,
        "structural_issues": [],
    }


@pytest.mark.parametrize("devices", ["-1", "0", "2"])
@pytest.mark.parametrize(
    "script,args",
    [
        ("train_chessqueries", ["--name", "smoke"]),
        ("train_resnext", ["--recipe", "faithful", "--name", "smoke"]),
    ],
)
def test_training_clis_reject_unsupported_devices_before_setup(
    script, args, devices, monkeypatch, capsys
):
    module = _load_script(script)
    monkeypatch.setattr(
        module, "get_config", lambda: pytest.fail("setup ran with unsupported --devices")
    )

    with pytest.raises(SystemExit) as exc_info:
        module.main([*args, "--devices", devices])

    assert exc_info.value.code == 2
    assert "released training recipe supports one GPU" in capsys.readouterr().err


def test_download_data_requires_a_target(capsys):
    module = _load_script("download_data")

    with pytest.raises(SystemExit) as exc_info:
        module.main([])

    assert exc_info.value.code == 2
    assert "nothing to do" in capsys.readouterr().err


def test_download_data_checkpoint_flags_work_without_datasets(monkeypatch):
    module = _load_script("download_data")
    fetched = []
    monkeypatch.setattr(
        module.dl, "download_chessred_checkpoint", lambda: fetched.append("baseline")
    )
    monkeypatch.setattr(
        module.dl, "download_release_checkpoint", lambda: fetched.append("release")
    )

    module.main(["--checkpoint", "--release-checkpoint"])

    assert fetched == ["baseline", "release"]
