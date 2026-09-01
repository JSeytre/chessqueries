"""Download datasets and released model weights.

Usage:
    poetry run python scripts/download_data.py chessred
    poetry run python scripts/download_data.py chessred --checkpoint
    poetry run python scripts/download_data.py chesscog cvchess
    poetry run python scripts/download_data.py all
    poetry run python scripts/download_data.py --release-checkpoint
"""
import argparse

from chessqueries.data import download as dl
from chessqueries.data.base import DatasetName


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "datasets",
        nargs="*",
        # None, not []: with nargs="*" argparse validates a list default against
        # `choices` (fixed only in Python 3.13); an absent positional still
        # parses to [] either way.
        default=None,
        choices=[*(d.value for d in dl.DATASETS), "all"],
        help="Which dataset(s) to download. May be omitted when only fetching a checkpoint.",
    )
    p.add_argument(
        "--checkpoint",
        action="store_true",
        help="Also download the ChessReD ResNeXt baseline checkpoint.",
    )
    p.add_argument(
        "--release-checkpoint",
        action="store_true",
        help="Also download and SHA-256 verify the released ChessQueries safetensors weights.",
    )
    args = p.parse_args(argv)
    if not args.datasets and not (args.checkpoint or args.release_checkpoint):
        p.error("nothing to do: name dataset(s) and/or pass --checkpoint / --release-checkpoint")

    names = list(dl.DATASETS) if "all" in args.datasets else [DatasetName(d) for d in args.datasets]
    for name in names:
        dl.DATASETS[name]()

    if args.checkpoint:
        dl.download_chessred_checkpoint()
    if args.release_checkpoint:
        dl.download_release_checkpoint()

    print("\nDone.")


if __name__ == "__main__":
    main()
