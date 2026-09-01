"""Train the ChessReD ResNeXt-101 baseline on the joint CR+CC+SLCC set.

Two recipes (see chessqueries.train.baseline_recipes):
    --recipe faithful   ChessReD's exact recipe, only the data swapped (their
                        model, more data). BCE / Adam 1e-3 step@100 / 1024px / no-aug.
    --recipe v2         V2's training recipe on the same ResNeXt (isolate the
                        model). softmax-CE / AdamW cosine / 644px / geometric_mild.

Examples:
    # faithful baseline (matches Masouris & van Gemert, joint data)
    poetry run python scripts/train_resnext.py --recipe faithful \
        --name baseline-resnext-faithful-joint

    # V2-recipe baseline
    poetry run python scripts/train_resnext.py --recipe v2 \
        --name baseline-resnext-v2recipe-joint

    # CPU wiring smoke test (touches no GPU)
    poetry run python scripts/train_resnext.py --recipe faithful --name smoke \
        --accelerator cpu --fast-dev-run --batch-size 2 --resolution 128 --workers 0
"""
import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from chessqueries.config import get_config
from chessqueries.data.base import DatasetName
from chessqueries.train.baseline_recipes import RECIPES, BaselineRecipe
from chessqueries.train.cli import single_gpu_devices
from chessqueries.train.lit import JointDataModule, LitChessReDResNeXt

torch.set_float32_matmul_precision("high")

# bf16 halves activation memory (ResNeXt-101 @ 1024px is heavy); faithful keeps
# fp32 to match the paper, but pass --precision bf16-mixed if it OOMs at 24GB.
_DEFAULT_PRECISION = {BaselineRecipe.FAITHFUL: "32-true", BaselineRecipe.V2: "bf16-mixed"}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipe", type=BaselineRecipe, choices=list(BaselineRecipe), required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--datasets", default="chessred,chesscog,slcc",
                   help="Comma-separated joint train datasets (each needs TRAIN/VAL splits).")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--devices", type=single_gpu_devices, default=1)
    p.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--precision", default=None, help="Override Trainer precision (default per-recipe).")
    # Recipe overrides (default: the recipe's value).
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--accumulate", type=int, default=1, help="Gradient accumulation (keeps effective batch at OOM).")
    p.add_argument("--fast-dev-run", action="store_true", help="1 train + 1 val batch; for CPU wiring smoke tests.")
    args = p.parse_args(argv)

    spec = RECIPES[args.recipe]
    epochs = args.epochs if args.epochs is not None else spec.epochs
    batch_size = args.batch_size if args.batch_size is not None else spec.batch_size
    resolution = args.resolution if args.resolution is not None else spec.resolution
    seed = args.seed if args.seed is not None else spec.seed
    precision = args.precision or _DEFAULT_PRECISION[args.recipe]

    pl.seed_everything(seed, workers=True)
    cfg = get_config()
    sources = tuple(DatasetName(s.strip()) for s in args.datasets.split(","))
    print(f"[{args.recipe.value}] training on {[s.value for s in sources]} | "
          f"res={resolution} bs={batch_size}x{args.accumulate} epochs={epochs} seed={seed} prec={precision}")

    dm = JointDataModule(
        sources, resolution, batch_size, args.workers,
        augment=spec.augment, normalization=spec.normalization,
    )
    model = LitChessReDResNeXt(
        loss=spec.loss, optimizer=spec.optimizer, lr=spec.lr, weight_decay=spec.weight_decay,
        schedule=spec.schedule, step_size=spec.step_size, gamma=spec.gamma,
        epochs=epochs, empty_weight=spec.empty_weight,
    )

    out_dir = cfg.OUTPUTS_ROOT / "train"
    ckpt_cb = ModelCheckpoint(
        dirpath=cfg.CHECKPOINTS_ROOT / args.name,
        filename="{epoch:02d}-{val_board_acc:.4f}",
        monitor="val_board_acc",
        mode="max",
        save_top_k=2,
        save_last=True,
    )
    callbacks = [ckpt_cb, LearningRateMonitor(logging_interval="epoch")]
    if spec.early_stop_patience > 0:
        callbacks.append(EarlyStopping(monitor="val_board_acc", mode="max", patience=spec.early_stop_patience))

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=precision,
        logger=CSVLogger(str(out_dir), name=args.name),
        callbacks=callbacks,
        log_every_n_steps=20,
        gradient_clip_val=spec.grad_clip or None,
        accumulate_grad_batches=args.accumulate,
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(model, dm)
    print("\nBest val_board_acc:", ckpt_cb.best_model_score, "->", ckpt_cb.best_model_path)


if __name__ == "__main__":
    main()
