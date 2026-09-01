import pytest

from chessqueries.core import Board, Color, Piece, PieceType, Square


def test_square_index_roundtrip():
    for i in range(64):
        assert Square.from_index(i).index == i


def test_square_known_corners():
    assert Square.from_name("a8").index == 0
    assert Square.from_name("h8").index == 7
    assert Square.from_name("a1").index == 56
    assert Square.from_name("h1").index == 63
    assert Square.from_index(0).name == "a8"
    assert Square.from_index(63).name == "h1"


def test_square_color():
    assert Square.from_name("h1").is_light
    assert not Square.from_name("a1").is_light


def test_piece_semantics():
    assert Piece.EMPTY.is_empty
    assert Piece.WHITE_KNIGHT.symbol == "N"
    assert Piece.BLACK_QUEEN.symbol == "q"
    assert Piece.WHITE_KNIGHT.color is Color.WHITE
    assert Piece.BLACK_QUEEN.piece_type is PieceType.QUEEN
    assert Piece.of(Color.BLACK, PieceType.ROOK) is Piece.BLACK_ROOK
    assert Piece.from_symbol("k") is Piece.BLACK_KING
    assert int(Piece.EMPTY) == 0


def test_board_fen_roundtrip_startpos():
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    b = Board.from_fen(start)
    assert b.piece_at(Square.from_name("a8")) is Piece.BLACK_ROOK
    assert b.piece_at(Square.from_name("e1")) is Piece.WHITE_KING
    assert b.placement == start


def test_board_fen_roundtrip_random():
    fen = "r1bk3r/p2pBpNp/n4n2/1p1NP2P/6P1/3P4/P1P1K3/q5b1"
    assert Board.from_fen(fen).placement == fen


def test_board_labels_roundtrip():
    b = Board.from_fen("r1bk3r/p2pBpNp/n4n2/1p1NP2P/6P1/3P4/P1P1K3/q5b1")
    assert Board.from_labels(b.labels).placement == b.placement
    assert len(b.labels) == 64


def test_board_validation():
    with pytest.raises(ValueError):
        Board((Piece.EMPTY,) * 63)
    with pytest.raises(ValueError):
        Board.from_fen("8/8/8/8/8/8/8")  # 7 ranks


def test_board_fen_rank_width_validated():
    # 64 squares total but mis-shaped ranks (9 + 7): every square after the first
    # rank would land one file off. Must raise, not parse shifted.
    with pytest.raises(ValueError, match="rank"):
        Board.from_fen("ppppppppp/7/8/8/8/8/8/8")
    with pytest.raises(ValueError, match="rank"):
        Board.from_fen("rnbqkbnr/pppppppp/8/8/8/8/8/7")  # short last rank


def test_board_diff():
    a = Board.empty()
    b = Board.from_labels([0] * 63 + [int(Piece.WHITE_KING)])
    assert a.diff(b) == {Square.from_index(63)}


def test_color_fen_char():
    assert Color.WHITE.fen == "w"
    assert Color.BLACK.fen == "b"
