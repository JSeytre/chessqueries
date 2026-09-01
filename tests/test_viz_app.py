"""Dataset-browser loading policy."""

from chessqueries.core import Split
from chessqueries.data import DatasetName
from chessqueries.viz import app


def test_visualizer_allows_a_partial_diagnostic_dataset(monkeypatch):
    expected = object()

    class Dataset:
        splits = (Split.TEST,)

        def load_samples(self, split, *, allow_partial=False):
            assert split is Split.TEST
            assert allow_partial is True
            return expected

    monkeypatch.setattr(app, "get_dataset", lambda dataset: Dataset())

    assert app._load(DatasetName.SLCC, "test") is expected
