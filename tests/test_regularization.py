"""The optional regularization knobs wire through to the model and loss."""
import torch

from chessqueries.train.lit import LitChessQueriesModel


def test_drop_path_activates_stochastic_depth():
    off = LitChessQueriesModel(freeze_encoder=False, drop_path_rate=0.0)
    on = LitChessQueriesModel(freeze_encoder=False, drop_path_rate=0.1)

    def probs(model):
        return {
            round(module.drop_prob, 4)
            for module in model.model.encoder.modules()
            if type(module).__name__ == "DropPath"
        }

    assert probs(off) <= {0.0}
    assert max(probs(on)) > 0.0  # linear schedule up to 0.1


def test_label_smoothing_raises_loss_on_confident_correct_preds():
    logits = torch.zeros(1, 64, 13)
    logits[..., 0] = 20.0           # confident, correct
    labels = torch.zeros(1, 64, dtype=torch.long)
    base = LitChessQueriesModel(freeze_encoder=True, label_smoothing=0.0)._loss(logits, labels)
    smooth = LitChessQueriesModel(freeze_encoder=True, label_smoothing=0.1)._loss(logits, labels)
    assert smooth > base  # smoothing penalizes over-confidence -> nonzero floor
