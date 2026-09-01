"""Identify game + ply from names/clocks against relay timelines (no network)."""

from chessqueries.annotate.identify import (
    Nameplates,
    candidate_plies,
    identify_position,
    match_board_to_relay,
    match_games_by_name,
    parse_clock_text,
    surname,
)
from chessqueries.annotate.relay import parse_round_pgn
from chessqueries.core import Board, Color, Piece

# Two games with deliberately distinct clocks so matching is unambiguous.
PGN = """[Event "T"]
[White "Niemann, Hans Moke"]
[Black "Vachier-Lagrave, Maxime"]
[Result "*"]
[TimeControl "25+10"]

1. e4 {[%clk 0:25:00]} e5 {[%clk 0:24:40]} 2. Nf3 {[%clk 0:24:30]} Nc6 {[%clk 0:24:20]} *

[Event "T"]
[White "Gukesh D"]
[Black "So, Wesley"]
[Result "*"]
[TimeControl "25+10"]

1. d4 {[%clk 0:20:00]} d5 {[%clk 0:19:00]} 2. c4 {[%clk 0:18:00]} e6 {[%clk 0:17:00]} *
"""


def test_surname_variants():
    assert surname("Niemann, Hans Moke") == "niemann"
    assert surname("Vachier-Lagrave, Maxime") == "vachier-lagrave"
    assert surname("Gukesh D") == "gukesh"  # drops single-letter initial


def test_parse_clock_text():
    assert parse_clock_text("25:17") == 25 * 60 + 17
    assert parse_clock_text("0:25:17") == 25 * 60 + 17
    assert parse_clock_text("1.23") == 83  # '.' misread for ':'
    assert parse_clock_text("garbage") is None


def test_name_matching_picks_right_game():
    games = parse_round_pgn(PGN)
    assert match_games_by_name(Nameplates(("NIEMANN",), ("VACHIER-LAGRAVE",)), games) == [0]
    assert match_games_by_name(Nameplates(("GUKESH",), ("SO",)), games) == [1]


def test_name_matching_tolerates_clipped_ocr_char():
    # The reported bug: a too-tight nameplate rect clipped the leading "V", so OCR read
    # "ACHIER-LAGRAVE". Exact substring missed the game; lenient matching recovers it.
    games = parse_round_pgn(PGN)
    assert match_games_by_name(Nameplates(("NIEMANN",), ("ACHIER-LAGRAVE",)), games) == [0]
    # Short surnames still demand an exact token (a loose substring would match anything).
    assert match_games_by_name(Nameplates(("GUKESH",), ("S",)), games) == []


def test_identify_returns_none_when_names_read_but_unmatched():
    # Names were read but match no game (the Gukesh/Fedoseev case: a clipped name knocks
    # the right game out). We must NOT fall back to clock-only across every game — that is
    # how a frame gets a confident label for the wrong players. Leave it unidentified.
    games = parse_round_pgn(PGN)
    # (1470, 1476) cleanly identifies game 0 ply 3 by clock alone, but the nameplate names
    # nobody in the round -> None, instead of silently grabbing game 0.
    assert identify_position(games, (1470, 1476), Nameplates(("Carlsen",), ("Nakamura",))) is None
    # No names at all (board+clock layout) -> clock may still pick the game.
    assert identify_position(games, (1470, 1476), None) is not None


def test_name_matching_disambiguates_reversed_colour_rematch():
    # The reported bug: the same pair meet twice with reversed colours (a blitz
    # double-round), so both games carry both surnames. The White nameplate naming
    # Fedoseev (game 1's White) must pick game 1, not the Niemann-White game 0.
    rematch = parse_round_pgn(
        """[Event "T"]
[White "Niemann, Hans Moke"]
[Black "Fedoseev, Vladimir"]
[Result "*"]
[TimeControl "3+2"]

1. e4 {[%clk 0:03:00]} e5 {[%clk 0:03:00]} *

[Event "T"]
[White "Fedoseev, Vladimir"]
[Black "Niemann, Hans Moke"]
[Result "*"]
[TimeControl "3+2"]

1. d4 {[%clk 0:03:00]} d5 {[%clk 0:03:00]} *
"""
    )
    # Orientation resolves to exactly the correctly-coloured game, not both.
    assert match_games_by_name(Nameplates(("FEDOSEEV",), ("NIEMANN",)), rematch) == [1]
    assert match_games_by_name(Nameplates(("NIEMANN",), ("FEDOSEEV",)), rematch) == [0]
    # When orientation can't decide (both names in one region), fall back to both.
    assert match_games_by_name(Nameplates(("NIEMANN", "FEDOSEEV"), ()), rematch) == [0, 1]


def test_identify_recovers_game_and_ply():
    games = parse_round_pgn(PGN)
    # Game 0, ply 3 (after Nf3): White just moved -> White static at 24:30=1470,
    # Black running, ticked down a bit from 24:40=1480.
    pos = games[0].position_at(3)
    assert pos.white_clk_s == 1470 and pos.turn == Color.BLACK
    ident = identify_position(games, (1470, 1476), Nameplates(("Niemann",), ("Vachier-Lagrave",)))
    assert ident is not None
    assert ident.game_index == 0 and ident.ply == 3
    assert ident.static_side == Color.WHITE and ident.confidence > 0.9


def test_identify_trusts_screen_clock_order():
    games = parse_round_pgn(PGN)
    # Screen order (White, Black) identifies ply 3; the swapped pair must NOT match it
    # (the overlay layout is fixed, so we trust order instead of trying both).
    correct = identify_position(games, (1470, 1476), None)
    assert correct.game_index == 0 and correct.ply == 3
    swapped = identify_position(games, (1476, 1470), None)
    assert swapped is None or swapped.ply != 3


def test_candidate_plies_brackets_running_clock():
    games = parse_round_pgn(PGN)
    # An impossible pair (both far above any recorded value) yields no candidates.
    assert candidate_plies(games[1], 9999, 9999) == []


# A scramble where Black's clock recurs at the same value across plies, but White's
# running window separates them — the real bug behind frame 166360 (ply 88 vs 114).
SCRAMBLE_PGN = """[Event "T"]
[White "Vachier-Lagrave, Maxime"]
[Black "Fedoseev, Vladimir"]
[Result "*"]
[TimeControl "25+10"]

1. e4 {[%clk 0:05:00]} e5 {[%clk 0:00:30]} 2. Nf3 {[%clk 0:04:50]} Nc6 {[%clk 0:00:28]} 3. Bc4 {[%clk 0:00:25]} Bc5 {[%clk 0:00:30]} *
"""


def test_running_lower_bound_rejects_impossible_early_ply():
    g = parse_round_pgn(SCRAMBLE_PGN)[0]
    # Black=30 (static) recurs at ply 2 (White then had 300s) and ply 6 (White in
    # scramble at 25s). Observed White=27 is impossible at ply 2 — White would need
    # ~280-300s — so the increment-aware running bound rejects ply 2 and picks ply 6.
    plies = [m.ply for m in candidate_plies(g, 27, 30)]
    assert 2 not in plies
    assert candidate_plies(g, 27, 30)[0].ply == 6


def test_increment_widens_running_lower_bound():
    g = parse_round_pgn(SCRAMBLE_PGN)[0]  # rapid: +10s/move
    assert g.increment_s == 10
    # At ply 3 Black is running (White static at 290). Black's next recorded clock is 28,
    # so just before the press Black's live clock is 28 - 10 = 18. A frame at Black=19
    # matches only because the bound floors at (next - increment), not at next (28).
    assert any(m.ply == 3 for m in candidate_plies(g, 290, 19))


def test_match_board_to_relay_exact_and_noisy():
    games = parse_round_pgn(PGN)
    # Exact placement of game 1, ply 3 -> diff 0 at the right (game, ply).
    target = games[1].position_at(3)
    exact = match_board_to_relay(games, target.placement)
    assert exact.game_index == 1 and exact.ply == 3 and exact.diff == 0

    # Noisy prediction: blank out two occupied squares -> still nearest is ply 3.
    labels = list(Board.from_fen(target.placement).labels)
    occupied = [i for i, p in enumerate(labels) if p != Piece.EMPTY][:2]
    for i in occupied:
        labels[i] = Piece.EMPTY
    noisy = Board.from_labels(labels).placement
    m = match_board_to_relay(games, noisy, max_diff=4)
    assert m.game_index == 1 and m.ply == 3 and m.diff == 2


def test_match_board_to_relay_rejects_unrelated():
    games = parse_round_pgn(PGN)
    # An empty board is far from every relay position -> no confident match.
    assert match_board_to_relay(games, Board.empty().placement, max_diff=4) is None
