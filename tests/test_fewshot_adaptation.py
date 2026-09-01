"""Few-shot support-set selection + the adaptation modes."""

import pytest
import torch

from chessqueries.core import Board, Split
from chessqueries.data import fewshot
from chessqueries.data.base import BoardSample, DatasetCompleteness, DatasetLoad, DatasetName
from chessqueries.data.fewshot import FULL_SPLIT, FewShotSplit, adaptation_targets, load_support
from chessqueries.models.adapt import READOUT_MODULES, AdaptMode, configure_trainable
from chessqueries.models.chessqueries_model import ChessQueriesModel
from chessqueries.models.lora import LoRALinear
from chessqueries.train.fewshot import adapt, evaluate

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# A tiny stand-in encoder keeps the mode tests CPU-cheap; 96 = 6*16 patches.
_ENCODER = "vit_tiny_patch16_224"
_X = torch.randint(0, 256, (1, 3, 96, 96)).float() / 255.0


@pytest.fixture(autouse=True)
def in_process_batches(monkeypatch):
    """Keep unit tests focused on adaptation rather than DataLoader IPC."""
    real_loader = torch.utils.data.DataLoader

    def loader(*args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs["persistent_workers"] = False
        return real_loader(*args, **kwargs)

    monkeypatch.setattr("chessqueries.train.fewshot.DataLoader", loader)


def _samples(n: int, prefix: str) -> list[BoardSample]:
    board = Board.from_fen(START_FEN)
    return [
        BoardSample(
            image_path=f"/tmp/{prefix}{i}.png",
            board=board,
            dataset=DatasetName.SLCC,
            sample_id=f"{prefix}{i}",
        )
        for i in range(n)
    ]


class _FakeDataset:
    """Stands in for a registered dataset."""

    def __init__(self, splits, pools):
        self.splits = splits
        self._pools = pools
        self.name = DatasetName.SLCC

    def load_with_report(self, split=None, *, allow_partial=False):
        samples = self._pools[split]
        return DatasetLoad(
            samples,
            DatasetCompleteness(
                dataset=self.name,
                split=split,
                expected_samples=len(samples),
                labelled_samples=len(samples),
                available_samples=len(samples),
                missing_sample_ids=(),
                allow_partial=allow_partial,
            ),
        )


@pytest.fixture
def split_dataset(monkeypatch):
    pools = {
        Split.TRAIN: _samples(50, "tr"),
        Split.VAL: _samples(7, "va"),
        Split.TEST: _samples(9, "te"),
    }
    ds = _FakeDataset((Split.TRAIN, Split.VAL, Split.TEST), pools)
    monkeypatch.setattr(fewshot, "get_dataset", lambda name: ds)
    return ds


# --- support-set selection -------------------------------------------------


def test_k_subsamples_train_only(split_dataset):
    s = load_support(DatasetName.SLCC, k=10, seed=0)
    assert len(s.train) == 10
    assert len(s.val) == 7 and len(s.test) == 9  # eval splits untouched


def test_support_draw_is_seeded(split_dataset):
    a = load_support(DatasetName.SLCC, k=10, seed=0).train_ids
    b = load_support(DatasetName.SLCC, k=10, seed=0).train_ids
    c = load_support(DatasetName.SLCC, k=10, seed=1).train_ids
    assert a == b, "same seed must reproduce the support set"
    assert a != c, "different seeds must draw different support sets"


def test_k_zero_keeps_full_train_split(split_dataset):
    assert len(load_support(DatasetName.SLCC, k=FULL_SPLIT, seed=0).train) == 50


def test_k_above_pool_size_keeps_full_train_split(split_dataset):
    assert len(load_support(DatasetName.SLCC, k=999, seed=0).train) == 50


def test_negative_k_rejected(split_dataset):
    with pytest.raises(ValueError, match="k must be >= 0"):
        load_support(DatasetName.SLCC, k=-1, seed=0)


def test_adaptation_targets_need_train_and_val_splits():
    # CVChess is one game with no official split: nothing to draw a support set
    # from and nothing to select a checkpoint on, so it stays eval-only.
    targets = adaptation_targets()
    assert DatasetName.CVCHESS not in targets
    assert {DatasetName.SLCC, DatasetName.CHESSRED, DatasetName.CHESSCOG} <= set(targets)


def test_dataset_without_splits_rejected_as_target():
    with pytest.raises(ValueError, match="no train/val split"):
        load_support(DatasetName.CVCHESS, k=10, seed=0)


def test_overlapping_support_and_test_rejected():
    shared = _samples(3, "x")
    with pytest.raises(ValueError, match="both support and test"):
        FewShotSplit(train=shared, val=[], test=shared)


def test_val_overlapping_other_sets_rejected():
    # Selecting a checkpoint on frames that are also in train or test would leak.
    shared = _samples(3, "x")
    with pytest.raises(ValueError, match="both support and val"):
        FewShotSplit(train=shared, val=shared, test=_samples(2, "t"))
    with pytest.raises(ValueError, match="both val and test"):
        FewShotSplit(train=_samples(2, "t"), val=shared, test=shared)


def test_empty_sets_rejected():
    with pytest.raises(ValueError, match="support set is empty"):
        FewShotSplit(train=[], val=[], test=_samples(2, "t"))
    with pytest.raises(ValueError, match="evaluation set is empty"):
        FewShotSplit(train=_samples(2, "t"), val=[], test=[])


# --- adaptation modes ------------------------------------------------------


@pytest.fixture
def make_model():
    """Fresh model per call: LoRA injection mutates the encoder in place, so a
    shared instance would leak adapters across modes."""

    def factory():
        # Staged-unfreeze bases persist freeze_encoder=True, which is the state the
        # modes have to cope with (it made LoRA silently receive no gradient).
        return ChessQueriesModel(
            encoder_name=_ENCODER, pretrained=False, freeze_encoder=True, decoder_layers=1
        )

    return factory


@pytest.fixture
def base_model(make_model):
    return make_model()


def _trainable(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}


def _readout_names(model):
    return {
        f"{mod}.{n}"
        for mod in READOUT_MODULES
        if getattr(model, mod, None) is not None
        for n, _ in getattr(model, mod).named_parameters()
    }


def test_zero_shot_trains_nothing(base_model):
    params = configure_trainable(base_model, AdaptMode.ZERO_SHOT)
    assert params == []
    assert _trainable(base_model) == set()


def test_lora_trains_adapters_only(base_model):
    params = configure_trainable(base_model, AdaptMode.LORA, rank=4, alpha=8)
    assert len(params) > 0 and len(params) % 2 == 0  # an A and a B per adapter
    assert all(n.endswith((".A", ".B")) for n in _trainable(base_model))
    assert any(isinstance(m, LoRALinear) for m in base_model.encoder.modules())
    # Encoder must build a graph, else the adapters get no gradient.
    assert base_model.freeze_encoder is False


def test_lora_receives_gradient(base_model):
    params = configure_trainable(base_model, AdaptMode.LORA, rank=4, alpha=8)
    base_model(_X).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)


def test_head_ft_trains_readout_with_frozen_encoder(base_model):
    configure_trainable(base_model, AdaptMode.HEAD_FT)
    assert _trainable(base_model) == _readout_names(base_model)
    # Left in no-grad mode on purpose: the encoder forward stays at inference cost.
    assert base_model.freeze_encoder is True
    assert not any(p.requires_grad for p in base_model.encoder.parameters())


def test_head_ft_receives_gradient_through_frozen_encoder(base_model):
    params = configure_trainable(base_model, AdaptMode.HEAD_FT)
    base_model(_X).sum().backward()
    assert all(p.grad is not None for p in params)


def test_encoder_ft_trains_encoder_only(base_model):
    params = configure_trainable(base_model, AdaptMode.ENCODER_FT)
    trainable = _trainable(base_model)
    assert trainable and all(n.startswith("encoder.") for n in trainable)
    assert trainable.isdisjoint(_readout_names(base_model))
    assert all(p.requires_grad for p in base_model.encoder.parameters())
    assert len(params) == len(list(base_model.encoder.parameters()))
    # Encoder must build a graph, else the weights we are adapting get no gradient.
    assert base_model.freeze_encoder is False


def test_encoder_ft_receives_gradient(base_model):
    params = configure_trainable(base_model, AdaptMode.ENCODER_FT)
    base_model(_X).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)


def test_full_ft_trains_everything(base_model):
    params = configure_trainable(base_model, AdaptMode.FULL_FT)
    assert len(params) == len(list(base_model.parameters()))
    assert all(p.requires_grad for p in base_model.parameters())
    assert base_model.freeze_encoder is False


# --- eval-mode + selection contracts --------------------------------------


@pytest.fixture
def image_samples(tmp_path):
    """Real files on disk: BoardImageDataset decodes image_path."""
    from torchvision.utils import save_image

    board = Board.from_fen(START_FEN)
    out = []
    for i in range(4):
        path = tmp_path / f"img{i}.png"
        save_image(torch.rand(3, 96, 96), path)
        out.append(
            BoardSample(image_path=path, board=board, dataset=DatasetName.SLCC, sample_id=f"img{i}")
        )
    return out


def test_evaluate_forces_eval_mode(base_model, image_samples):
    # Measuring a model left in train mode silently reports it with the decoder's
    # dropout active, which depresses every adapted number and makes evals noisy.
    base_model.train()
    a = evaluate(base_model, image_samples, 96, "cpu", False, batch_size=2)
    assert base_model.training is False
    base_model.train()
    b = evaluate(base_model, image_samples, 96, "cpu", False, batch_size=2)
    assert a["per_square_accuracy"] == b["per_square_accuracy"]
    assert a["board_accuracy"] == b["board_accuracy"]


def _fake_val_scores(monkeypatch, values):
    """Drive selection with a fixed val-metric sequence, so the restore path is
    tested independently of real training dynamics."""
    it = iter(values)

    def fake_eval(model, samples, res, device, bf16, batch_size=16):
        model.eval()
        return {
            "per_square_accuracy": next(it),
            "board_accuracy": 0.0,
            "mean_wrong_squares": 0.0,
            "n_boards": len(samples),
        }

    monkeypatch.setattr("chessqueries.train.fewshot.evaluate", fake_eval)


def test_adapt_restores_base_weights_when_val_never_improves(
    base_model, image_samples, monkeypatch
):
    params = configure_trainable(base_model, AdaptMode.HEAD_FT)
    before = {n: p.detach().clone() for n, p in base_model.named_parameters() if p.requires_grad}
    _fake_val_scores(monkeypatch, [0.9, 0.5, 0.4, 0.3, 0.2])
    fit = adapt(
        base_model,
        image_samples,
        image_samples,
        96,
        "cpu",
        steps=6,
        lr=1e-2,
        batch_size=2,
        bf16=False,
        params=params,
        eval_every=1,
        patience=2,
        eval_batch_size=2,
        verbose=False,
    )
    assert fit["best_step"] == 0
    assert fit["stopped_early"] is True
    assert params, "head_ft should expose trainable params"
    for n, p in base_model.named_parameters():
        if n in before:
            assert torch.equal(p, before[n]), f"{n} was not restored to the base weights"


def test_adapt_keeps_the_improving_step_not_the_last(base_model, image_samples, monkeypatch):
    params = configure_trainable(base_model, AdaptMode.HEAD_FT)
    # Peak at step 2, then decline: selection must not return the final weights.
    _fake_val_scores(monkeypatch, [0.1, 0.2, 0.9, 0.3, 0.2, 0.1])
    fit = adapt(
        base_model,
        image_samples,
        image_samples,
        96,
        "cpu",
        steps=6,
        lr=1e-2,
        batch_size=2,
        bf16=False,
        params=params,
        eval_every=1,
        patience=3,
        eval_batch_size=2,
        verbose=False,
    )
    assert fit["best_step"] == 2
    assert fit["selected_val_probe"]["per_square_accuracy"] == 0.9


def test_adapt_hands_back_a_model_in_eval_mode(base_model, image_samples):
    params = configure_trainable(base_model, AdaptMode.HEAD_FT)
    adapt(
        base_model,
        image_samples,
        image_samples,
        96,
        "cpu",
        steps=2,
        lr=1e-4,
        batch_size=2,
        bf16=False,
        params=params,
        eval_every=2,
        patience=0,
        eval_batch_size=2,
        verbose=False,
    )
    assert base_model.training is False


def test_capacity_ladder_is_ordered(make_model):
    counts = {}
    for mode in AdaptMode:
        params = configure_trainable(make_model(), mode, rank=8, alpha=16)
        counts[mode] = sum(p.numel() for p in params)
    assert counts[AdaptMode.ZERO_SHOT] == 0
    assert counts[AdaptMode.ZERO_SHOT] < counts[AdaptMode.LORA] < counts[AdaptMode.FULL_FT]
    assert counts[AdaptMode.HEAD_FT] < counts[AdaptMode.FULL_FT]
    # encoder_ft and head_ft partition full_ft: same location grid, both directions.
    assert counts[AdaptMode.LORA] < counts[AdaptMode.ENCODER_FT] < counts[AdaptMode.FULL_FT]
    assert counts[AdaptMode.ENCODER_FT] + counts[AdaptMode.HEAD_FT] == counts[AdaptMode.FULL_FT]


def test_unhandled_adapt_mode_raises(base_model):
    """A value outside AdaptMode must raise, not silently full-fine-tune."""
    with pytest.raises(ValueError, match="unhandled"):
        configure_trainable(base_model, "not_a_mode")


def test_inject_lora_matches_targets_at_dot_boundaries_only():
    from torch import nn

    from chessqueries.models.lora import inject_lora

    class Sub(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Sub()
            self.xattn = Sub()  # suffix lookalike: must NOT match "attn.proj"

    net = Net()
    params = inject_lora(net, targets=("attn.proj",))
    assert isinstance(net.attn.proj, LoRALinear)
    assert type(net.xattn.proj) is nn.Linear
    assert len(params) == 2  # one A/B pair, for the single true match
