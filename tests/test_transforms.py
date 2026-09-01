"""Validate the augmentation presets exposed by build_transform."""

import torch

import pytest

from chessqueries.data.transforms import Augment, _GEOMETRIC_PRESETS, build_transform


def test_every_augment_builds_and_runs():
    x = torch.randint(0, 256, (3, 64, 64)).float()
    for augment in Augment:
        out = build_transform(resolution=56, train=True, augment=augment)(x)
        assert out.shape == (3, 56, 56)


def test_geometric_presets_add_geometric_ops():
    # geometric presets contribute 3 extra ops (rotation/perspective/affine) on
    # top of the photometric pipeline; photometric alone does not.
    photo = build_transform(56, train=True, augment="photometric").transforms
    for augment in _GEOMETRIC_PRESETS:
        geo = build_transform(56, train=True, augment=augment).transforms
        assert len(geo) == len(photo) + 3


def test_mild_is_gentler_than_aggressive():
    mild = _GEOMETRIC_PRESETS[Augment.GEOMETRIC_MILD]
    aggressive = _GEOMETRIC_PRESETS[Augment.GEOMETRIC]
    # guards the "mild" semantics from silently inverting
    assert mild["rotation"] < aggressive["rotation"]
    assert mild["perspective"] < aggressive["perspective"]
    assert mild["scale"][0] > aggressive["scale"][0]  # less zoom-out


def test_none_augment_matches_eval_pipeline():
    # The no-augmentation ablation ("none") must make the train transform
    # identical to eval: no geometric ops and no photometric jitter.
    train_ops = [type(op).__name__ for op in build_transform(56, train=True, augment="none").transforms]
    eval_ops = [type(op).__name__ for op in build_transform(56, train=False, augment="none").transforms]
    assert train_ops == eval_ops
    assert "ColorJitter" not in train_ops


def test_eval_transform_has_no_augmentation():
    # train=False must be deterministic: no rotation/perspective/jitter ops.
    ops = build_transform(56, train=False, augment="geometric").transforms
    names = {type(op).__name__ for op in ops}
    assert not names & {
        "RandomRotation",
        "RandomPerspective",
        "RandomAffine",
        "ColorJitter",
        "RandomApply",
    }


def test_unknown_augment_raises():
    """An unknown preset must fail loudly, not silently degrade to photometric-only."""
    with pytest.raises(ValueError):
        build_transform(56, train=True, augment="geometrik")
