"""Staged encoder unfreeze + warmup + cosine-floor LR schedule."""
import math
import types
import warnings

from chessqueries.train.lit import LitChessQueriesModel, _warmup_cosine_factor


def test_warmup_then_cosine_to_floor():
    def factor(epoch):
        return _warmup_cosine_factor(
            epoch, start_epoch=0, warmup_epochs=3, total_epochs=10, floor=0.01
        )

    assert math.isclose(factor(0), 1 / 3)
    assert math.isclose(factor(1), 2 / 3)
    assert math.isclose(factor(2), 1.0)  # end of warmup -> peak
    assert math.isclose(factor(3), 1.0)  # cosine progress 0 -> still peak
    assert math.isclose(factor(10), 0.01)  # anneals to the floor fraction, not zero
    mids = [factor(e) for e in range(2, 11)]
    assert all(a >= b for a, b in zip(mids, mids[1:]))  # monotone non-increasing


def test_frozen_group_multiplier_is_zero_until_unfreeze():
    def factor(epoch):
        return _warmup_cosine_factor(
            epoch, start_epoch=5, warmup_epochs=3, total_epochs=20, floor=0.0
        )

    assert factor(0) == 0.0
    assert factor(4) == 0.0  # encoder group parked at LR 0 during phase 1
    assert math.isclose(factor(5), 1 / 3)  # warmup restarts at the unfreeze boundary
    assert math.isclose(factor(7), 1.0)
    assert math.isclose(factor(20), 0.0, abs_tol=1e-9)  # cosine to zero when floor=0


def test_plain_cosine_recovered_with_defaults_off():
    # start=0, warmup=0, floor=0 == plain cosine-to-zero.
    def factor(epoch):
        return _warmup_cosine_factor(
            epoch, start_epoch=0, warmup_epochs=0, total_epochs=10, floor=0.0
        )

    assert math.isclose(factor(0), 1.0)
    assert math.isclose(factor(5), 0.5)
    assert math.isclose(factor(10), 0.0, abs_tol=1e-9)


def test_staged_unfreeze_flips_requires_grad_at_epoch():
    m = LitChessQueriesModel(freeze_encoder=False, unfreeze_epoch=5, warmup_epochs=3)
    enc = list(m.model.encoder.parameters())
    assert m.model.freeze_encoder is True             # starts frozen for phase 1
    assert not any(p.requires_grad for p in enc)
    m._maybe_unfreeze(4)
    assert not any(p.requires_grad for p in enc)      # not yet
    m._maybe_unfreeze(5)
    assert all(p.requires_grad for p in enc)          # phase 2: full fine-tune
    assert m.model.freeze_encoder is False


def test_permanent_freeze_never_unfreezes():
    m = LitChessQueriesModel(freeze_encoder=True, unfreeze_epoch=5)
    m._maybe_unfreeze(99)
    assert m.model.freeze_encoder is True
    assert not any(p.requires_grad for p in m.model.encoder.parameters())


def test_configure_optimizers_stages_encoder_and_floors_lr():
    m = LitChessQueriesModel(freeze_encoder=False, unfreeze_epoch=5, warmup_epochs=3,
                         lr_floor=0.01, lr_schedule="cosine", lr=1.4e-4, encoder_lr=1.4e-5)
    m.trainer = types.SimpleNamespace(max_epochs=10)
    cfg = m.configure_optimizers()
    opt, sched = cfg["optimizer"], cfg["lr_scheduler"]["scheduler"]
    assert [g["initial_lr"] for g in opt.param_groups] == [1.4e-5, 1.4e-4]  # (encoder, new)

    def lrs_at(epoch):
        # Drive the real scheduler to `epoch`; suppress the benign "step before
        # optimizer.step" warning that this manual stepping (no optimizer.step) trips.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sched.last_epoch = epoch - 1
            sched.step()
        return [g["lr"] for g in opt.param_groups]

    enc0, new0 = lrs_at(0)
    assert enc0 == 0.0 and math.isclose(new0, 1.4e-4 / 3)   # encoder parked, decoder warming
    enc4, _ = lrs_at(4)
    assert enc4 == 0.0                                       # still frozen at epoch 4
    enc5, _ = lrs_at(5)
    assert math.isclose(enc5, 1.4e-5 / 3)                   # encoder warmup starts at unfreeze
    enc10, new10 = lrs_at(10)
    assert math.isclose(enc10, 1.4e-5 * 0.01) and math.isclose(new10, 1.4e-4 * 0.01)  # floor, not zero
