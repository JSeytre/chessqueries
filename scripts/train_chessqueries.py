"""Train the ChessQueries model.

Joint ChessReD+ChessCog training, fine-tuned encoder, mild geometric
augmentation, 45 epochs. The default LR recipe de-risks from-scratch runs
against the cold-start plateau (see LitChessQueriesModel): staged encoder unfreeze
(--unfreeze-epoch), linear warmup (--warmup-epochs), and a cosine LR floor
(--lr-floor). Set --unfreeze-epoch 0 --warmup-epochs 0 --lr-floor 0 to recover
the older plain cosine-to-zero schedule (e.g. when warm-starting an already
ignited model via --init-checkpoint).

Examples:
    # default from-scratch recipe (no args needed)
    poetry run python scripts/train_chessqueries.py --name v3

    # frozen-encoder probe (fast, single dataset)
    poetry run python scripts/train_chessqueries.py --freeze-encoder --datasets chessred \
        --augment photometric --lr-schedule none --epochs 20
"""
import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from chessqueries.config import get_config
from chessqueries.data.base import DatasetName
from chessqueries.data.transforms import Augment
from chessqueries.models.chessqueries_model import DEFAULT_ENCODER, HeadType
from chessqueries.train.baseline_recipes import Schedule
from chessqueries.train.cli import single_gpu_devices
from chessqueries.train.lit import JointDataModule, LitChessQueriesModel

torch.set_float32_matmul_precision("high")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resolution", type=int, default=518)
    p.add_argument("--decoder-layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1.4e-4)
    p.add_argument("--encoder-lr", type=float, default=1.4e-5)
    p.add_argument("--empty-weight", type=float, default=1.0, help="Loss weight on the 'empty' class.")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--drop-path", type=float, default=0.0, help="Encoder stochastic-depth rate.")
    p.add_argument("--freeze-encoder", dest="freeze_encoder", action="store_true", default=False)
    p.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false")
    p.add_argument("--devices", type=single_gpu_devices, default=1)
    p.add_argument("--lr-schedule", type=Schedule, choices=[Schedule.NONE, Schedule.COSINE], default=Schedule.COSINE)
    p.add_argument("--warmup-epochs", type=int, default=3,
                   help="Linear LR warmup length (epochs), applied at start and again "
                        "for the encoder at unfreeze. 0 disables warmup.")
    p.add_argument("--unfreeze-epoch", type=int, default=5,
                   help="Staged unfreeze: keep the encoder frozen for this many epochs "
                        "(decoder ignites on fixed features), then fine-tune it. "
                        "0 = train the encoder from the start (no staging).")
    p.add_argument("--lr-floor", type=float, default=0.001,
                   help="Cosine anneals to this fraction of the peak LR instead of 0, "
                        "so a late-igniting run still has LR left to converge.")
    p.add_argument("--aux-weight", type=float, default=0.0,
                   help="Weight on piece-type + colour aux heads (0 disables multi-task).")
    p.add_argument("--datasets", default="chessred,chesscog",
                   help="Comma-separated train datasets (e.g. 'chessred,chesscog'). "
                        "Multiple sources => joint training on their concatenated train/val splits.")
    p.add_argument("--augment", type=Augment, choices=list(Augment), default=Augment.GEOMETRIC_MILD,
                   help="Train-time augmentation. 'geometric'/'geometric_mild' add rotation/perspective/"
                        "scale jitter for viewpoint robustness (labels are viewpoint-invariant).")
    p.add_argument("--encoder", default=DEFAULT_ENCODER,
                   help="timm encoder name (e.g. vit_large_patch14_dinov2.lvd142m for a bigger backbone).")
    p.add_argument("--head-type", type=HeadType, choices=list(HeadType), default=HeadType.QUERY,
                   help="Output head: 'query' (square-query transformer decoder, default) or "
                        "'linear' (no-decoder ablation: 8x8 pool of the encoder grid + shared head).")
    p.add_argument("--init-checkpoint", type=Path, default=None,
                   help="Warm-start model weights from a prior LitChessQueriesModel checkpoint "
                        "(e.g. fine-tune V1 on a new dataset). Optimizer/schedule restart fresh.")
    p.add_argument("--name", required=True, help="Run tag; names checkpoints/<name> and the logger run.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed everything (incl. dataloader workers) for reproducible runs.")
    args = p.parse_args(argv)

    cfg = get_config()
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)
        print(f"Seeded everything with seed={args.seed}")
    sources = tuple(DatasetName(s.strip()) for s in args.datasets.split(","))
    dm = JointDataModule(sources, args.resolution, args.batch_size, args.workers, augment=args.augment)
    print(f"Training on: {[s.value for s in sources]}")
    model = LitChessQueriesModel(
        encoder_name=args.encoder,
        freeze_encoder=args.freeze_encoder,
        decoder_layers=args.decoder_layers,
        lr=args.lr,
        encoder_lr=args.encoder_lr,
        empty_weight=args.empty_weight,
        lr_schedule=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        unfreeze_epoch=args.unfreeze_epoch,
        lr_floor=args.lr_floor,
        aux_weight=args.aux_weight,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        drop_path_rate=args.drop_path,
        head_type=args.head_type,
    )
    if args.init_checkpoint:
        state = LitChessQueriesModel.load_from_checkpoint(args.init_checkpoint, map_location="cpu")
        model.model.load_state_dict(state.model.state_dict())
        print(f"Warm-started weights from {args.init_checkpoint}")

    out_dir = cfg.OUTPUTS_ROOT / "train"
    ckpt_cb = ModelCheckpoint(
        dirpath=cfg.CHECKPOINTS_ROOT / args.name,
        filename="{epoch:02d}-{val_board_acc:.4f}",
        monitor="val_board_acc",
        mode="max",
        save_top_k=2,
        save_last=True,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.devices,
        precision="bf16-mixed",
        logger=CSVLogger(str(out_dir), name=args.name),
        callbacks=[ckpt_cb, LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=20,
        gradient_clip_val=1.0,
    )
    trainer.fit(model, dm)
    print("\nBest val_board_acc:", ckpt_cb.best_model_score, "->", ckpt_cb.best_model_path)


if __name__ == "__main__":
    main()
