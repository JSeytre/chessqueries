"""Model implementations and baselines from the literature."""
from chessqueries.models.adapt import AdaptMode, configure_trainable
from chessqueries.models.base import BoardRecognizer, EvalPredictions
from chessqueries.models.chessred_resnext import ChessReDResNeXt
from chessqueries.models.llm import (
    BoardReader, ClaudeBoardReader, Effort, ImagePrep, LLMName, LocalBoardReader,
    OpenAIBoardReader, Provider, get_reader,
)
from chessqueries.models.chessqueries_model import ChessQueriesModel

__all__ = ["AdaptMode", "BoardReader", "BoardRecognizer", "ChessQueriesModel",
           "ChessReDResNeXt", "ClaudeBoardReader", "Effort", "EvalPredictions",
           "ImagePrep", "LLMName", "LocalBoardReader",
           "OpenAIBoardReader", "Provider", "configure_trainable", "get_reader"]
