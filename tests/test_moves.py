# Tests for env/moves.py's move application and capture resolution.

from env.board import BASE_POSITION, FINISH_POSITION, BoardState
from env.moves import apply_move


# Checks exiting Base places the token on its entry square.
def test_apply_move_exits_base_to_entry_square():
    board = BoardState()
    outcome = apply_move(board, player_id=0, token_id=0, roll=6)
    assert board.get(0, 0) == 0
    assert outcome.exited_base is True
    assert outcome.captured_player is None


# Checks a normal advance moves the token by exactly the roll.
def test_apply_move_advances_by_roll():
    board = BoardState()
    board.set(0, 0, 10)
    outcome = apply_move(board, player_id=0, token_id=0, roll=4)
    assert board.get(0, 0) == 14
    assert outcome.exited_base is False


# Checks landing on a lone opponent token on a non-safe square captures it.
def test_apply_move_captures_lone_opponent_on_unsafe_square():
    board = BoardState()
    board.set(0, 0, 10)
    board.set(2, 0, 40)  # player 2 relative 40 -> global (26+40)%52 = 14, not a safe square
    outcome = apply_move(board, player_id=0, token_id=0, roll=4)
    assert board.get(0, 0) == 14
    assert board.get(2, 0) == BASE_POSITION
    assert outcome.captured_player == 2
    assert outcome.captured_token == 0


# Checks landing on an opponent token on a safe square does not capture it.
def test_apply_move_no_capture_on_safe_square():
    board = BoardState()
    board.set(0, 0, 4)
    board.set(1, 0, 47)  # player 1 relative 47 -> global (13+47)%52 = 8, a star safe square
    outcome = apply_move(board, player_id=0, token_id=0, roll=4)
    assert board.get(0, 0) == 8
    assert board.get(1, 0) == 47  # untouched
    assert outcome.captured_player is None


# Checks reaching the final square is flagged as reaching home.
def test_apply_move_reaching_finish_is_flagged():
    board = BoardState()
    board.set(0, 0, 55)
    outcome = apply_move(board, player_id=0, token_id=0, roll=2)
    assert board.get(0, 0) == FINISH_POSITION
    assert outcome.reached_home is True
