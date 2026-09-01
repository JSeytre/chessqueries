"""Shot layout descriptor: DINOv2 spatial-grid features compressed by a fitted PCA.

DINOv2 patch tokens pooled on a 3x3 grid keep *where* physical-board content sits —
even a small picture-in-picture board — so "players at a real table" separates from
"twin digital boards" (whole-frame pooling would let persistent broadcast graphics
dominate). A PCA fitted on the template exemplars compresses the 6912-d grid to a
256-d descriptor, keeping the registry a small single-file JSON. Matching/clustering
downstream are dim-agnostic; the thresholds in `templates` are calibrated for this
cosine space.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from chessqueries.annotate.templates import Shot
from chessqueries.annotate.video import FrameReader, VideoFile

MODEL_NAME = "vit_base_patch14_reg4_dinov2"
IMG_SIZE = 518  # 37x37 patch grid at patch14 — enough detail for small PiP boards
GRID = 3  # 3x3 spatial pool of the patch tokens (chosen over 2/4 on the labeled pairs)
DESCRIPTOR_DIM = 256  # PCA output; matching/clustering below this are dim-agnostic
BLANK_STD = 1.0  # a near-uniform frame (transition/black) -> zero descriptor -> skipped
PCA_RESOURCE = Path(__file__).parent / "resources" / "slcc_grid3_pca.npz"

# ImageNet normalization DINOv2 was trained with.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _l2(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-9)


@dataclass(frozen=True)
class Pca:
    """A fitted PCA projection: mean-center, project onto ``components``, L2-normalize.

    Fitted on local template-exemplar grid embeddings (:meth:`fit`) and retained as an
    annotation-production resource. It is not part of the public SLCC release."""

    mean: np.ndarray  # [D]
    components: np.ndarray  # [k, D] top-k right singular vectors

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.components.ndim != 2:
            raise ValueError("PCA needs a 1-D mean and 2-D components")
        if self.components.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"components dim {self.components.shape[1]} != mean dim {self.mean.shape[0]}"
            )

    @property
    def dim(self) -> int:
        return int(self.components.shape[0])

    def transform(self, x: np.ndarray) -> np.ndarray:
        """[n, D] (or [D]) grid embeddings -> [n, k] (or [k]) L2-normalized descriptors.

        A zero input row (a blank frame, see :data:`BLANK_STD`) stays zero: otherwise
        mean-centering would turn it into a spurious non-zero descriptor and break the
        blank-skip contract the registry relies on (``assign`` drops zero-norm rows)."""
        x = np.asarray(x, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        z = _l2((x - self.mean) @ self.components.T)
        z[np.linalg.norm(x, axis=1) < 1e-6] = 0.0
        return z[0] if single else z

    @classmethod
    def fit(cls, x: np.ndarray, dim: int) -> "Pca":
        """Fit on ``x`` ([n, D]): mean + the top-``dim`` right singular vectors."""
        x = np.asarray(x, dtype=np.float32)
        mean = x.mean(0)
        _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
        return cls(mean=mean, components=np.ascontiguousarray(vt[:dim]))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, components=self.components)

    @classmethod
    def load(cls, path: Path) -> "Pca":
        with np.load(path) as d:
            return cls(
                mean=d["mean"].astype(np.float32),
                components=d["components"].astype(np.float32),
            )


class ShotEmbedder:
    """DINOv2 grid3 -> PCA descriptor for a shot keyframe (a BGR frame).

    The timm model is loaded lazily on first embed (heavy import + pretrained weights),
    so importing this module — or building the descriptor cache from pre-embedded grids
    — costs nothing. Deterministic: eval mode, no grad, fixed preprocessing."""

    def __init__(
        self,
        *,
        device: str | None = None,
        pca_path: Path = PCA_RESOURCE,
        model_name: str = MODEL_NAME,
        img_size: int = IMG_SIZE,
    ) -> None:
        self.model_name = model_name
        self.img_size = img_size
        self.pca = Pca.load(pca_path)
        self._device = device
        self._model = None
        self._n_prefix = 0

    def _ensure_model(self):
        if self._model is None:
            import timm
            import torch

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = (
                timm.create_model(self.model_name, pretrained=True, num_classes=0)
                .eval()
                .to(self._device)
            )
            self._n_prefix = self._model.num_prefix_tokens  # cls + register tokens
        return self._model

    def _grid_embed(self, frames_bgr: list[np.ndarray], batch_size: int) -> np.ndarray:
        """[n, GRID*GRID*C] pre-PCA grid embeddings; blank frames become zero rows."""
        import torch

        model = self._ensure_model()
        mean = torch.tensor(_MEAN, device=self._device).view(1, 3, 1, 1)
        std = torch.tensor(_STD, device=self._device).view(1, 3, 1, 1)

        def l2t(t: "torch.Tensor") -> "torch.Tensor":
            return t / (t.norm(dim=-1, keepdim=True) + 1e-9)

        out: list[np.ndarray] = []
        blank: list[bool] = []
        with torch.no_grad():
            for i in range(0, len(frames_bgr), batch_size):
                batch = []
                for f in frames_bgr[i : i + batch_size]:
                    blank.append(float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).std()) < BLANK_STD)
                    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    rgb = cv2.resize(
                        rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA
                    )
                    batch.append(rgb)
                x = torch.from_numpy(np.stack(batch)).float().permute(0, 3, 1, 2)
                x = (x.to(self._device) / 255.0 - mean) / std
                tokens = model.forward_features(x)  # [B, n_prefix+P, C]
                patches = tokens[:, self._n_prefix :]  # [B, P, C]
                b, p, c = patches.shape
                g = int(round(p**0.5))
                grid = patches.transpose(1, 2).reshape(b, c, g, g)
                pooled = torch.nn.functional.adaptive_avg_pool2d(grid, GRID)  # [B, C, G, G]
                pooled = l2t(pooled.permute(0, 2, 3, 1))  # per-cell L2 over C
                vec = l2t(pooled.reshape(b, -1))  # concat + global L2
                out.append(vec.cpu().numpy().astype(np.float32))
        emb = np.concatenate(out) if out else np.zeros((0, GRID * GRID), dtype=np.float32)
        emb[np.asarray(blank, dtype=bool)] = 0.0
        return emb

    def embed_frames(self, frames_bgr: list[np.ndarray], *, batch_size: int = 32) -> np.ndarray:
        """[n, DESCRIPTOR_DIM] L2-normalized descriptors for a list of BGR frames."""
        if not frames_bgr:
            return np.zeros((0, self.pca.dim), dtype=np.float32)
        return self.pca.transform(self._grid_embed(frames_bgr, batch_size))

    def embed_shots(
        self,
        video: VideoFile,
        shots: list[Shot],
        *,
        show_progress: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Descriptor per shot keyframe, decoding + embedding in batches so a full
        broadcast's 1080p keyframes are never all held in RAM at once."""
        from tqdm import tqdm

        chunks: list[np.ndarray] = []
        buf: list[np.ndarray] = []
        with FrameReader(video) as reader:
            for s in tqdm(shots, desc="keyframes", disable=not show_progress):
                buf.append(reader.frame_at_index(s.keyframe_index))
                if len(buf) >= batch_size:
                    chunks.append(self.embed_frames(buf, batch_size=batch_size))
                    buf = []
            if buf:
                chunks.append(self.embed_frames(buf, batch_size=batch_size))
        if not chunks:
            return np.zeros((0, self.pca.dim), dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)


@lru_cache(maxsize=1)
def default_embedder() -> ShotEmbedder:
    """Process-wide embedder (model + PCA loaded once) for the produce/label passes."""
    return ShotEmbedder()


def shot_descriptors(
    video: VideoFile, shots: list[Shot], *, show_progress: bool = False
) -> np.ndarray:
    """Descriptor for each shot's keyframe (shape ``[len(shots), DESCRIPTOR_DIM]``)."""
    return default_embedder().embed_shots(video, shots, show_progress=show_progress)
