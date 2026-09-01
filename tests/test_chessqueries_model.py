"""ChessQueries model head-type contract (query decoder vs. the A1 linear ablation)."""

import pytest
import torch

from chessqueries.models.chessqueries_model import ChessQueriesModel, HeadType

# 98 = 7*14, the smallest DINOv2 grid that still exercises the 8x8 pool.
_X = torch.randint(0, 256, (2, 3, 98, 98)).float() / 255.0


@pytest.mark.parametrize("head_type", list(HeadType))
def test_head_types_emit_board_logits(head_type):
    m = ChessQueriesModel(freeze_encoder=True, head_type=head_type).eval()
    with torch.no_grad():
        out = m(_X)
        labels = m.predict_labels(_X)
    assert out.shape == (2, 64, 13)
    assert labels.shape == (2, 64)
    assert labels.min() >= 0 and labels.max() < 13


def test_linear_head_drops_query_decoder():
    # The A1 ablation must actually remove the queries + transformer decoder,
    # otherwise it is not testing "no query decoder".
    m = ChessQueriesModel(freeze_encoder=True, head_type="linear")
    assert not hasattr(m, "decoder")
    assert not hasattr(m, "queries")
    q = ChessQueriesModel(freeze_encoder=True, head_type="query")
    assert hasattr(q, "decoder") and hasattr(q, "queries")


def test_invalid_head_type_rejected():
    with pytest.raises(ValueError):
        ChessQueriesModel(head_type="bogus")


def test_head_type_accepts_checkpoint_strings():
    """Old checkpoints store head_type as a bare string in hparams; construction
    must normalize it to the enum."""
    m = ChessQueriesModel(pretrained=False, freeze_encoder=True, head_type="linear")
    assert m.head_type is HeadType.LINEAR
