import numpy as np
import pytest
import torch

from chessqueries.core import Board, Color
from chessqueries.models.base import Recognition
from chessqueries.models.portable import PortableRecognizer
from chessqueries.models.predictor import Prediction, Predictor

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
TensorProto, helper, numpy_helper = onnx.TensorProto, onnx.helper, onnx.numpy_helper


@pytest.fixture
def artifact(tmp_path):
    board = Board.from_fen("4k3/8/8/8/8/8/8/4K3")
    values = {
        "values": np.eye(13, dtype=np.float32)[board.labels][None],
        "w": np.array([[1, 0]], dtype=np.float32),
        "b": np.array([[-1, 0]], dtype=np.float32),
        "zero": np.array(0, dtype=np.float32),
    }
    nodes = [helper.make_node("Identity", ["values"], ["logits"]),
             helper.make_node("ReduceMean", ["image"], ["mean"], keepdims=0),
             helper.make_node("Less", ["mean", "zero"], ["negative"]),
             helper.make_node("Where", ["negative", "b", "w"], ["direction"])]
    graph = helper.make_graph(nodes, "fixture", [
        helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 8, 8]),
    ], [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 64, 13]),
        helper.make_tensor_value_info("direction", TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(value, name) for name, value in values.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    return path


def test_batch_orientation_does_not_change_position(artifact, monkeypatch):
    predictor = Predictor.from_checkpoint(artifact, resolution=8)
    monkeypatch.setattr(predictor, "prepare_image", lambda p: torch.full((3, 8, 8), -1. if p.name == "b" else 1.))
    results = predictor.predict(["w", "b", "w"], batch_size=2)
    assert [p.orientation for p in results] == [Color.WHITE, Color.BLACK, Color.WHITE]
    assert len({p.fen for p in results}) == 1
    assert results[1].fen.endswith(" w - - 0 1")
    assert results[1].to_record()["orientation"] == "black"
    assert results[1].to_record()["viewing_direction"] == {"cos": -1., "sin": 0.}
    assert results[1].lichess_url.endswith("?color=black")
    assert "/analysis/" in results[1].lichess_url


def test_legacy_record_stays_compatible(tmp_path):
    result = Prediction(tmp_path / "photo.jpg", Board.empty())
    assert set(result.to_record()) == {"image", "fen", "lichess_url"}
    assert result.lichess_url == result.board.lichess_url()
    with pytest.raises(TypeError):
        Prediction(result.image_path, result.board, "black")


@pytest.mark.parametrize("angle,color",
    [(a, Color.WHITE) for a in (0, -90, 90, -95, 95, -100, 100)]
    + [(a, Color.BLACK) for a in (-100.01, 100.01, -110, 110, -180, 180)])
def test_white_preference(artifact, monkeypatch, angle, color):
    graph = onnx.load(artifact)
    direction = np.array([[np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]], dtype=np.float32)
    for value in graph.graph.initializer:
        if value.name in {"w", "b"}:
            value.CopyFrom(numpy_helper.from_array(direction, value.name))
    onnx.save(graph, artifact)
    predictor = Predictor.from_checkpoint(artifact, resolution=8)
    monkeypatch.setattr(predictor, "prepare_image", lambda _: torch.ones(3, 8, 8))
    result, = predictor.predict(["photo"])
    assert result.orientation is color
    assert result.lichess_url.endswith(f"?color={color.value}")
    assert result.fen == "4k3/8/8/8/8/8/8/4K3 w - - 0 1"


def test_empty_and_invalid_inputs(artifact):
    model = PortableRecognizer(artifact)
    assert model.predict_labels(torch.empty(0, 3, 8, 8)).shape == (0, 64)
    with pytest.raises(ValueError, match="dimensions"):
        model.recognize(torch.zeros(1, 3, 16, 16))
    with pytest.raises(ValueError, match="resolution"):
        Predictor.from_checkpoint(artifact, resolution=16)
    with pytest.raises(ValueError, match="cpu"):
        Predictor.from_checkpoint(artifact, resolution=8, device="cuda")
    with pytest.raises(ValueError, match="batch_size"):
        Predictor.from_checkpoint(artifact, resolution=8).predict([], batch_size=0)


@pytest.mark.parametrize("labels,directions", [
    (torch.zeros(1, 63, dtype=torch.int64), None),
    (torch.zeros(1, 64), None),
    (torch.full((1, 64), 13), None),
    (torch.zeros(1, 64, dtype=torch.int64), torch.tensor([[0., 0.]])),
    (torch.zeros(1, 64, dtype=torch.int64), torch.tensor([[float("nan"), 0.]])),
    (torch.zeros(1, 64, dtype=torch.int64), torch.tensor([[1., 0.]], dtype=torch.float16)),
    (torch.zeros(1, 64, dtype=torch.int64), torch.tensor([[1.00001, 0.]])),
])
def test_invalid_recognition_rejected(labels, directions):
    with pytest.raises(ValueError):
        Recognition(labels, directions)


def test_wrong_graph_contract_rejected(artifact):
    model = onnx.load(artifact)
    model.graph.output.pop()
    onnx.save(model, artifact)
    with pytest.raises(ValueError, match="outputs"):
        PortableRecognizer(artifact)
