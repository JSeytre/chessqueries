"""Checkpoints with enum hparams survive torch>=2.6 weights_only loading."""
import sys
from enum import Enum
from types import ModuleType

import torch

from chessqueries.data.base import DatasetName
from chessqueries.data.transforms import Augment, Normalization
from chessqueries.models.chessqueries_model import HeadType
from chessqueries.train import lit  # noqa: F401  (import registers the globals)
from chessqueries.train.lit import LEGACY_HEAD_TYPE_PATH
from chessqueries.train.baseline_recipes import Optimizer, ResNeXtLoss, Schedule

HPARAM_ENUMS = {Augment, DatasetName, HeadType, Normalization,
                Optimizer, ResNeXtLoss, Schedule}


def test_hparam_enums_are_weights_only_safe():
    safe = set(torch.serialization.get_safe_globals())
    assert HPARAM_ENUMS <= safe


def test_enum_hparams_round_trip_weights_only(tmp_path):
    path = tmp_path / "hparams.ckpt"
    torch.save({"lr_schedule": Schedule.COSINE, "head_type": HeadType.LINEAR,
                "optimizer": Optimizer.ADAMW, "loss": ResNeXtLoss.SOFTMAX_CE},
               path)
    loaded = torch.load(path, weights_only=True)
    assert loaded["lr_schedule"] is Schedule.COSINE
    assert loaded["head_type"] is HeadType.LINEAR


def test_renamed_head_type_loads_from_legacy_checkpoint_path(tmp_path, monkeypatch):
    """Existing checkpoints should not require the retired source module."""
    module_name, _, class_name = LEGACY_HEAD_TYPE_PATH.rpartition(".")
    legacy_module = ModuleType(module_name)
    legacy_head_type = Enum(
        class_name,
        {"QUERY": "query", "LINEAR": "linear"},
        type=str,
        module=module_name,
    )
    setattr(legacy_module, class_name, legacy_head_type)
    monkeypatch.setitem(sys.modules, module_name, legacy_module)

    path = tmp_path / "legacy-hparams.ckpt"
    torch.save({"head_type": legacy_head_type.LINEAR}, path)
    monkeypatch.delitem(sys.modules, module_name)

    loaded = torch.load(path, weights_only=True)
    assert loaded["head_type"] is HeadType.LINEAR
