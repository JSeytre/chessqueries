"""Image transforms: scale [0,255]->[0,1], square resize, normalize.

Two normalization regimes (see :class:`Normalization`): the project default
(ImageNet stats, used by the ViT models) and ChessReD's own preprocessing — a
``ToPILImage`` round-trip + their channel stats — kept bit-faithful so the
ResNeXt baseline trains/evaluates exactly as Masouris & van Gemert (VISAPP 2024)."""
from __future__ import annotations

from enum import Enum

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ChessReD's exact normalization statistics (computed on their 1024x1024 images).
CHESSRED_MEAN = [0.47225544, 0.51124555, 0.55296206]
CHESSRED_STD = [0.27787283, 0.27054584, 0.27802786]


class Normalization(str, Enum):
    """Preprocessing regime: ImageNet (ViT pipeline) or ChessReD (ResNeXt baseline)."""

    IMAGENET = "imagenet"
    CHESSRED = "chessred"


class Augment(str, Enum):
    """Train-time augmentation preset.

    Geometric augmentation teaches viewpoint robustness (the model has no
    orientation input, so it must read boards at any angle). Labels are board
    STATE — viewpoint-invariant — so they need no transform. `GEOMETRIC` is the
    aggressive preset, `GEOMETRIC_MILD` the gentler one; the transfer/in-domain
    trade-off between them is documented in the results log."""

    NONE = "none"  # no-augmentation ablation: train transform == eval transform
    PHOTOMETRIC = "photometric"
    GEOMETRIC = "geometric"
    GEOMETRIC_MILD = "geometric_mild"


_GEOMETRIC_PRESETS = {
    Augment.GEOMETRIC: dict(rotation=180, perspective=0.5, scale=(0.7, 1.1)),
    Augment.GEOMETRIC_MILD: dict(rotation=45, perspective=0.3, scale=(0.85, 1.1)),
}


def build_transform(
    resolution: int = 518,
    train: bool = False,
    augment: Augment | str = Augment.PHOTOMETRIC,
    normalization: Normalization = Normalization.IMAGENET,
):
    augment = Augment(augment)  # unknown presets raise instead of silently degrading
    if normalization is Normalization.CHESSRED:
        # ChessReD-faithful preprocessing: square resize + ToPILImage round-trip
        # (load-bearing — the exact op the released weights were trained with) +
        # their channel stats. Their recipe used NO augmentation, so train/val are
        # identical here; `train`/`augment` are deliberately ignored.
        return transforms.Compose([
            transforms.Resize((resolution, resolution), antialias=None),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(CHESSRED_MEAN, CHESSRED_STD),
        ])
    ops = [
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Resize((resolution, resolution), antialias=True),
    ]
    if train:
        if augment in _GEOMETRIC_PRESETS:
            g = _GEOMETRIC_PRESETS[augment]
            ops += [
                transforms.RandomApply([transforms.RandomRotation(g["rotation"], fill=0)], p=0.9),
                transforms.RandomPerspective(distortion_scale=g["perspective"], p=0.7, fill=0),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=g["scale"], fill=0),
            ]
        # Photometric jitter applies in all presets except NONE.
        if augment is not Augment.NONE:
            ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(ops)
