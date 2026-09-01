"""Lightning module + data module for the ChessQueries model."""
from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from chessqueries.core import NUM_PIECES, Piece, Split
from chessqueries.data.base import DatasetName, get_dataset
from chessqueries.data.transforms import Augment, Normalization, build_transform
from chessqueries.metrics import aggregate
from chessqueries.models.chessred_resnext import ChessReDResNeXt
from chessqueries.models.chessqueries_model import DEFAULT_ENCODER, ChessQueriesModel, HeadType
from chessqueries.train.baseline_recipes import Optimizer, ResNeXtLoss, Schedule

# save_hyperparameters() pickles these str-enums into checkpoints; torch>=2.6
# loads with weights_only=True by default and rejects any non-allowlisted
# global, so checkpoints written since the enum refactors fail to load without
# this. Plain-data enums, safe to unpickle by construction. Some checkpoints
# predate the model-module rename, so map that serialized path to the new class.
LEGACY_HEAD_TYPE_PATH = "chessqueries.models.query_net.HeadType"
torch.serialization.add_safe_globals(
    [
        Augment,
        DatasetName,
        HeadType,
        (HeadType, LEGACY_HEAD_TYPE_PATH),
        Normalization,
        Optimizer,
        ResNeXtLoss,
        Schedule,
    ]
)

NUM_CLASSES = NUM_PIECES
EMPTY_CLASS = int(Piece.EMPTY)


def _warmup_cosine_factor(
    epoch: int, *, start_epoch: int, warmup_epochs: int, total_epochs: int, floor: float
) -> float:
    """Per-epoch LR multiplier in ``[floor, 1]`` for one param group.

    Zero until ``start_epoch`` (the group is frozen until then), then a linear
    warmup over ``warmup_epochs``, then a cosine anneal to ``floor`` — a fraction
    of the peak LR — by ``total_epochs``. ``start_epoch=0, warmup_epochs=0,
    floor=0`` reproduces a plain cosine-to-zero schedule."""
    if epoch < start_epoch:
        return 0.0
    local = epoch - start_epoch
    if warmup_epochs > 0 and local < warmup_epochs:
        return (local + 1) / warmup_epochs
    denom = max(1, total_epochs - start_epoch - warmup_epochs)
    progress = min(1.0, (local - warmup_epochs) / denom)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


class _ValMetricsMixin:
    """Shared validation loop for our LightningModules: accumulate per-square
    predictions/ground-truth, then report board/per-square accuracy via the
    project metrics. Subclasses provide ``_loss`` and a ``forward`` returning
    ``(B, 64, NUM_CLASSES)`` logits."""

    def _reset_val_buffers(self) -> None:
        self._val_preds: list[list[int]] = []
        self._val_gts: list[list[int]] = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log("val_loss", self._loss(logits, y), prog_bar=True, sync_dist=True)
        self._val_preds.extend(logits.argmax(-1).cpu().tolist())
        self._val_gts.extend(y.cpu().tolist())

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        agg = aggregate(self._val_preds, self._val_gts)
        self.log("val_board_acc", agg["board_accuracy"], prog_bar=True)
        self.log("val_per_square_acc", agg["per_square_accuracy"], prog_bar=True)
        self.log("val_wrong_squares", agg["mean_wrong_squares"])
        self._val_preds.clear()
        self._val_gts.clear()


class JointDataModule(pl.LightningDataModule):
    """Train on the concatenation of several datasets' train splits; validate on
    the concatenation of their val splits (one combined `val_board_acc` drives
    checkpointing). Per-domain test numbers come from `eval_cross_dataset.py`.

    Sources must have an official TRAIN/VAL split (ChessReD, ChessCog). Held-out
    distributions (e.g. CVChess) are deliberately excluded so they stay a clean
    zero-shot generalization target."""

    def __init__(self, sources: tuple[DatasetName, ...], resolution: int, batch_size: int, workers: int,
                 augment: Augment | str = Augment.PHOTOMETRIC,
                 normalization: Normalization = Normalization.IMAGENET) -> None:
        super().__init__()
        self.sources = tuple(sources)
        self.resolution = resolution
        self.batch_size = batch_size
        self.workers = workers
        self.augment = Augment(augment)  # unknown presets fail here, not mid-setup
        self.normalization = normalization

    def setup(self, stage: str | None = None) -> None:
        train_parts, val_parts = [], []
        for name in self.sources:
            ds = get_dataset(name)
            if Split.TRAIN not in ds.splits or Split.VAL not in ds.splits:
                raise ValueError(f"{name.value} lacks TRAIN/VAL splits; cannot join-train on it.")
            train_tf = build_transform(self.resolution, train=True, augment=self.augment, normalization=self.normalization)
            val_tf = build_transform(self.resolution, train=False, normalization=self.normalization)
            train_parts.append(ds.torch_dataset(Split.TRAIN, transform=train_tf))
            val_parts.append(ds.torch_dataset(Split.VAL, transform=val_tf))
        self.train_ds = ConcatDataset(train_parts)
        self.val_ds = ConcatDataset(val_parts)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.workers, pin_memory=True, drop_last=True, persistent_workers=self.workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=self.workers,
            pin_memory=True, persistent_workers=self.workers > 0,
        )


class LitChessQueriesModel(_ValMetricsMixin, pl.LightningModule):
    """Per-square 13-class cross-entropy, with optional down-weighting of the
    dominant 'empty' class. Reports board/per-square accuracy on val/test using
    our shared metrics.

    Cold-start knobs mitigating the decoder's plateau/late-ignition failure mode:
    ``unfreeze_epoch`` keeps the encoder frozen for the first N epochs;
    ``warmup_epochs`` applies linear LR warmup at epoch 0 and again for the encoder
    group at the unfreeze boundary; ``lr_floor`` stops the cosine anneal at a
    fraction of the peak LR instead of 0."""

    def __init__(
        self,
        encoder_name: str = DEFAULT_ENCODER,
        freeze_encoder: bool = True,
        decoder_layers: int = 4,
        lr: float = 1e-4,
        encoder_lr: float = 1e-5,
        weight_decay: float = 0.05,
        empty_weight: float = 1.0,
        lr_schedule: str = Schedule.NONE,  # Schedule.NONE or Schedule.COSINE
        warmup_epochs: int = 0,     # linear LR warmup length (epochs)
        unfreeze_epoch: int = 0,    # staged unfreeze: encoder frozen until this epoch (0 = never freeze)
        lr_floor: float = 0.0,      # cosine floor as a fraction of peak LR
        aux_weight: float = 0.0,    # >0 enables piece-type + colour aux heads
        label_smoothing: float = 0.0,
        drop_path_rate: float = 0.0,  # encoder stochastic depth
        head_type: str = HeadType.QUERY,  # HeadType value; str accepted for ckpt hparams
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # The encoder starts frozen for a permanent probe (``freeze_encoder``) or
        # for the first phase of a staged unfreeze (``unfreeze_epoch > 0``).
        starts_frozen = freeze_encoder or unfreeze_epoch > 0
        self.model = ChessQueriesModel(
            encoder_name=encoder_name, freeze_encoder=starts_frozen, decoder_layers=decoder_layers,
            aux_heads=aux_weight > 0, drop_path_rate=drop_path_rate, head_type=head_type,
        )
        w = torch.ones(NUM_CLASSES)
        w[EMPTY_CLASS] = empty_weight
        self.register_buffer("class_weight", w)
        self._reset_val_buffers()

    def forward(self, x):
        return self.model(x)

    def _loss(self, logits, labels):
        return nn.functional.cross_entropy(
            logits.reshape(-1, NUM_CLASSES), labels.reshape(-1), weight=self.class_weight,
            label_smoothing=self.hparams.label_smoothing,
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.hparams.aux_weight > 0:
            out = self.model.forward_aux(x)
            loss = self._loss(out["main"], y)
            type_t = self.model.label_to_type[y]
            color_t = self.model.label_to_color[y]
            aux = (nn.functional.cross_entropy(out["type"].reshape(-1, out["type"].shape[-1]), type_t.reshape(-1))
                   + nn.functional.cross_entropy(out["color"].reshape(-1, out["color"].shape[-1]), color_t.reshape(-1)))
            loss = loss + self.hparams.aux_weight * aux
            self.log("train_aux", aux, prog_bar=False)
        else:
            loss = self._loss(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _maybe_unfreeze(self, epoch: int) -> None:
        """Staged unfreeze: once ``epoch`` reaches ``unfreeze_epoch``, enable
        encoder grads so phase 2 fine-tunes the whole network. Idempotent, so it
        is safe to call every epoch and after a mid-run resume. Permanent
        ``freeze_encoder`` probes never unfreeze."""
        if self.hparams.freeze_encoder or self.hparams.unfreeze_epoch <= 0:
            return
        if epoch >= self.hparams.unfreeze_epoch and self.model.freeze_encoder:
            for p in self.model.encoder.parameters():
                p.requires_grad = True
            self.model.freeze_encoder = False
            self.model.encoder.train()
            print(f"[staged-unfreeze] encoder unfrozen at epoch {epoch}")

    def on_train_epoch_start(self) -> None:
        self._maybe_unfreeze(self.current_epoch)

    def configure_optimizers(self):
        wd = self.hparams.weight_decay
        if self.hparams.freeze_encoder:
            params = [p for p in self.model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=wd)
            group_starts = [0]
        else:
            # Differential LR: low for the pretrained encoder, higher for new
            # modules. Both groups are always present — even when the encoder
            # starts frozen for a staged unfreeze — so no param-group surgery is
            # needed later: the encoder group simply sits at LR 0 (multiplier 0)
            # until ``unfreeze_epoch``, then warms up and joins the cosine anneal.
            enc, new = [], []
            for n, p in self.model.named_parameters():
                (enc if n.startswith("encoder.") else new).append(p)
            opt = torch.optim.AdamW(
                [
                    {"params": enc, "lr": self.hparams.encoder_lr},
                    {"params": new, "lr": self.hparams.lr},
                ],
                weight_decay=wd,
            )
            group_starts = [self.hparams.unfreeze_epoch, 0]  # (encoder, new)
        schedule = Schedule(self.hparams.lr_schedule)
        if schedule is Schedule.STEP:
            raise ValueError("LitChessQueriesModel supports Schedule.NONE or Schedule.COSINE")
        if schedule is not Schedule.COSINE:
            return opt
        total, warm, floor = self.trainer.max_epochs, self.hparams.warmup_epochs, self.hparams.lr_floor
        lr_lambdas = [
            (lambda start: lambda e: _warmup_cosine_factor(
                e, start_epoch=start, warmup_epochs=warm, total_epochs=total, floor=floor))(s)
            for s in group_starts
        ]
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambdas)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


class LitChessReDResNeXt(_ValMetricsMixin, pl.LightningModule):
    """The ChessReD ResNeXt-101 baseline, trainable on our joint set. The loss
    selects the comparison: ``BCE`` reproduces the authors' recipe exactly,
    ``SOFTMAX_CE`` transplants V2's objective onto the same model. See
    ``baseline_recipes`` for the full recipe bundles."""

    def __init__(
        self,
        loss: ResNeXtLoss = ResNeXtLoss.BCE,
        optimizer: Optimizer = Optimizer.ADAM,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        schedule: Schedule = Schedule.STEP,
        step_size: int = 100,
        gamma: float = 0.1,
        epochs: int = 200,
        empty_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Train from ImageNet init in our canonical Piece order (no class perm).
        self.model = ChessReDResNeXt(pretrained=True)
        w = torch.ones(NUM_CLASSES)
        w[EMPTY_CLASS] = empty_weight
        self.register_buffer("class_weight", w)
        self._reset_val_buffers()

    def forward(self, x):
        return self.model(x).reshape(-1, 64, NUM_CLASSES)  # (B, 64, 13)

    def _loss(self, logits, labels):
        if self.hparams.loss is ResNeXtLoss.BCE:
            # ChessReD-faithful: independent sigmoids vs the one-hot target.
            target = nn.functional.one_hot(labels, NUM_CLASSES).float()
            return nn.functional.binary_cross_entropy_with_logits(logits, target)
        return nn.functional.cross_entropy(
            logits.reshape(-1, NUM_CLASSES), labels.reshape(-1), weight=self.class_weight,
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self._loss(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        cls = torch.optim.AdamW if self.hparams.optimizer is Optimizer.ADAMW else torch.optim.Adam
        opt = cls(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.schedule is Schedule.STEP:
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
        if self.hparams.schedule is Schedule.COSINE:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
        return opt
