"""Run a direction-capable ONNX model through the existing predictor."""

from pathlib import Path

import numpy as np
import torch

from chessqueries.models.base import BoardRecognizer, Recognition


class PortableRecognizer(BoardRecognizer):
    def __init__(self, path: Path | str) -> None:
        super().__init__()
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "image" or inputs[0].type != "tensor(float)":
            raise ValueError("expected one float32 image input")
        shape = inputs[0].shape
        if (len(shape) != 4 or shape[:2] != [1, 3] or not isinstance(shape[2], int)
                or shape[2] <= 0 or shape[2] != shape[3]):
            raise ValueError("expected image shape [1,3,size,size]")
        if ([(o.name, o.type, o.shape) for o in outputs]
                != [("logits", "tensor(float)", [1, 64, 13]),
                    ("direction", "tensor(float)", [1, 2])]):
            raise ValueError("unsupported model outputs")
        self.resolution = shape[2]

    def recognize(self, x: torch.Tensor) -> Recognition:
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, self.resolution, self.resolution):
            raise ValueError("image dimensions disagree with the model")
        labels, directions = [], []
        for sample in x.detach().to(device="cpu", dtype=torch.float32).numpy():
            logits, direction = self.session.run(None, {"image": np.ascontiguousarray(sample[None])})
            if not np.isfinite(logits).all():
                raise ValueError("nonfinite model output")
            labels.append(torch.from_numpy(logits.argmax(-1)))
            directions.append(torch.from_numpy(direction))
        return Recognition(
            torch.cat(labels) if labels else torch.empty((0, 64), dtype=torch.int64),
            torch.cat(directions) if directions else torch.empty((0, 2)),
        )

    def predict_labels(self, x: torch.Tensor) -> torch.Tensor:
        return self.recognize(x).labels
