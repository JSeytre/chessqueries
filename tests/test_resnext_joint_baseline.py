"""Guards for the ResNeXt joint-training baseline: recipe specs, ChessReD-faithful
preprocessing, and the round-trip from a trained Lightning checkpoint back into a
canonical-order `ChessReDResNeXt` for evaluation."""

import pytest
import torch

from chessqueries.data.transforms import (
    CHESSRED_MEAN,
    Normalization,
    build_transform,
)
from chessqueries.models.chessred_resnext import NUM_CLASSES, ChessReDResNeXt
from chessqueries.train.baseline_recipes import (
    RECIPES,
    BaselineRecipe,
    Optimizer,
    RecipeSpec,
    ResNeXtLoss,
    Schedule,
)


def test_faithful_recipe_matches_chessred_paper():
    spec = RECIPES[BaselineRecipe.FAITHFUL]
    assert spec.loss is ResNeXtLoss.BCE
    assert spec.optimizer is Optimizer.ADAM and spec.lr == 1e-3
    assert spec.schedule is Schedule.STEP and spec.step_size == 100 and spec.gamma == 0.1
    assert spec.epochs == 200 and spec.batch_size == 8 and spec.resolution == 1024
    assert spec.normalization is Normalization.CHESSRED and spec.augment == "none"
    assert spec.early_stop_patience == 20  # paper: "200 epochs with early stopping"


def test_v2_recipe_transplants_v2_training():
    spec = RECIPES[BaselineRecipe.V2]
    assert spec.loss is ResNeXtLoss.SOFTMAX_CE
    assert spec.optimizer is Optimizer.ADAMW and spec.schedule is Schedule.COSINE
    assert spec.resolution == 644 and spec.normalization is Normalization.IMAGENET
    assert spec.augment == "geometric_mild"
    assert spec.early_stop_patience == 0  # V2 runs the full 45 epochs


def test_recipe_spec_rejects_bad_augment():
    with pytest.raises(ValueError):
        RecipeSpec(ResNeXtLoss.BCE, Optimizer.ADAM, 1e-3, 0.0, Schedule.STEP, 100, 0.1,
                   200, 8, 1024, Normalization.CHESSRED, augment="bogus",
                   empty_weight=1.0, grad_clip=0.0, seed=42, early_stop_patience=0)


def test_recipe_spec_rejects_empty_weight_with_bce():
    with pytest.raises(ValueError):
        RecipeSpec(ResNeXtLoss.BCE, Optimizer.ADAM, 1e-3, 0.0, Schedule.STEP, 100, 0.1,
                   200, 8, 1024, Normalization.CHESSRED, augment="none",
                   empty_weight=2.0, grad_clip=0.0, seed=42, early_stop_patience=0)


def test_chessred_transform_is_square_and_unaugmented():
    # ChessReD-faithful preprocessing: square resize + their channel stats, and
    # deterministic even with train=True (their recipe used no augmentation).
    img = (torch.rand(3, 200, 300) * 255)  # non-square float [0,255], as datasets yield
    tf = build_transform(128, train=True, augment="geometric_mild", normalization=Normalization.CHESSRED)
    names = {type(op).__name__ for op in tf.transforms}
    assert not names & {"RandomRotation", "RandomPerspective", "RandomAffine", "ColorJitter", "RandomApply"}
    assert tf(img).shape == (3, 128, 128)
    assert CHESSRED_MEAN[0] == pytest.approx(0.47225544)


def test_from_lightning_checkpoint_round_trips_safely_in_canonical_order(
    tmp_path, monkeypatch
):
    # A trained head is saved under the LightningModule's `model.` prefix and
    # already predicts in our canonical Piece order — the loader must strip the
    # prefix cleanly (no missing/unexpected keys) and keep an identity class perm.
    trained = ChessReDResNeXt(pretrained=False)
    state = {f"model.{k}": v for k, v in trained.state_dict().items()}
    ckpt = tmp_path / "last.ckpt"
    torch.save({"state_dict": state, "hyper_parameters": {
        "loss": ResNeXtLoss.SOFTMAX_CE,
        "optimizer": Optimizer.ADAMW,
        "schedule": Schedule.COSINE,
    }}, ckpt)

    real_load = torch.load
    load_kwargs = []

    def recording_load(*args, **kwargs):
        load_kwargs.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)

    loaded = ChessReDResNeXt.from_lightning_checkpoint(ckpt)
    assert load_kwargs[0]["weights_only"] is True
    assert torch.equal(loaded._class_perm, torch.arange(NUM_CLASSES))  # identity: no remap
    trained.eval()  # match loaded (BatchNorm running stats vs batch stats)
    x = torch.randn(2, 3, 96, 96)
    # Same weights => identical raw forward, and predict_labels == plain argmax.
    with torch.no_grad():
        assert torch.allclose(loaded(x), trained(x), atol=1e-5)
    expected = trained(x).reshape(-1, 64, NUM_CLASSES).argmax(-1)
    assert torch.equal(loaded.predict_labels(x), expected)
