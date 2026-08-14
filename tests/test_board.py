# Tests for env/board.py's coordinate transforms and BoardState.

import pytest

from env.board import (
    BASE_POSITION,
    ENTRY_OFFSETS,
    FINISH_POSITION,
    SAFE_SQUARES,
    BoardState,
    is_base,
    is_home,
    is_in_home_column,
    is_on_main_track,
    is_safe_square,
    relative_to_global,
)


# Checks the 4 entry squares are spaced 13 apart.
def test_entry_offsets_are_13_apart():
    assert ENTRY_OFFSETS == (0, 13, 26, 39)


# Checks relative_to_global against hand-calculated cases.
@pytest.mark.parametrize(
    "player_id, relative_position, expected_global",
    [
        (0, 0, 0),
        (0, 50, 50),
        (1, 0, 13),
        (1, 39, 0),
        (1, 50, 11),
        (2, 0, 26),
        (2, 30, 4),
        (3, 0, 39),
        (3, 20, 7),
    ],
)
def test_relative_to_global_hand_calculated(player_id, relative_position, expected_global):
    assert relative_to_global(player_id, relative_position) == expected_global


# Checks relative_to_global rejects positions not on the main track.
@pytest.mark.parametrize("relative_position", [-1, 51, 56, 57, 58])
def test_relative_to_global_rejects_off_main_track_positions(relative_position):
    with pytest.raises(ValueError):
        relative_to_global(0, relative_position)


# Checks the safe-square set matches entry squares plus 8.
def test_safe_squares_are_entries_plus_eight():
    expected = {0, 8, 13, 21, 26, 34, 39, 47}
    assert SAFE_SQUARES == expected
    for square in expected:
        assert is_safe_square(square)
    assert not is_safe_square(1)
    assert not is_safe_square(25)


# Checks the boundary values for every state-classification helper.
def test_state_classification_boundaries():
    assert is_base(BASE_POSITION)
    assert not is_base(0)

    assert is_on_main_track(0)
    assert is_on_main_track(50)
    assert not is_on_main_track(51)
    assert not is_on_main_track(BASE_POSITION)

    assert is_in_home_column(51)
    assert is_in_home_column(56)
    assert not is_in_home_column(50)
    assert not is_in_home_column(57)

    assert is_home(FINISH_POSITION)
    assert not is_home(56)


# Checks a freshly constructed board starts every token in Base.
def test_board_state_reset_puts_every_token_in_base():
    board = BoardState()
    assert all(board.get(p, t) == BASE_POSITION for p in range(4) for t in range(4))


# Checks set/get round-trips and leaves other tokens untouched.
def test_board_state_set_and_get_round_trip():
    board = BoardState()
    board.set(2, 3, 45)
    assert board.get(2, 3) == 45
    assert board.get(2, 0) == BASE_POSITION
    assert board.get(0, 0) == BASE_POSITION


# Checks set rejects out-of-range positions.
def test_board_state_rejects_invalid_positions():
    board = BoardState()
    with pytest.raises(ValueError):
        board.set(0, 0, 58)
    with pytest.raises(ValueError):
        board.set(0, 0, -2)


# Checks tokens_on_global_square finds every token on a shared square, across players.
def test_tokens_on_global_square_finds_all_occupants():
    board = BoardState()
    board.set(0, 0, 5)  # global 5
    board.set(1, 0, 44)  # relative 44 -> global (13+44)%52 = 5
    board.set(2, 0, 10)  # unrelated square
    assert sorted(board.tokens_on_global_square(5)) == [(0, 0), (1, 0)]
    assert board.tokens_on_global_square(6) == []


# Checks global_square returns None for Base, home column, and Home.
def test_global_square_none_off_main_track():
    board = BoardState()
    board.set(0, 0, BASE_POSITION)
    assert board.global_square(0, 0) is None
    board.set(0, 0, 53)
    assert board.global_square(0, 0) is None
    board.set(0, 0, FINISH_POSITION)
    assert board.global_square(0, 0) is None


# Checks global_square agrees with relative_to_global.
def test_global_square_matches_relative_to_global():
    board = BoardState()
    board.set(1, 0, 10)
    assert board.global_square(1, 0) == relative_to_global(1, 10) == 23


# Checks copy() produces an independent board.
def test_copy_is_independent():
    board = BoardState()
    board.set(0, 0, 5)
    clone = board.copy()
    clone.set(0, 0, 40)
    assert board.get(0, 0) == 5
    assert clone.get(0, 0) == 40


# Checks equality compares token positions, not identity.
def test_equality():
    a = BoardState()
    b = BoardState()
    assert a == b
    a.set(0, 0, 5)
    assert a != b
    b.set(0, 0, 5)
    assert a == b


# Checks a hand-verified board state matches expectations from every helper.
def test_construct_hand_verified_board_state():
    board = BoardState()
    board.set(0, 0, 0)
    board.set(0, 1, BASE_POSITION)
    board.set(1, 0, 8)
    board.set(2, 2, 56)
    board.set(3, 3, FINISH_POSITION)

    assert board.global_square(0, 0) == 0
    assert is_safe_square(board.global_square(0, 0))
    assert board.global_square(1, 0) == 21
    assert is_safe_square(board.global_square(1, 0))
    assert board.global_square(2, 2) is None
    assert is_home(board.get(3, 3))
