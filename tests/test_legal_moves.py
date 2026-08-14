# Tests for env/legal_moves.py's legal-move generation.

from env.board import BoardState, FINISH_POSITION
from env.legal_moves import get_action_mask, get_legal_tokens


# Checks a token in Base can only exit on a roll of 6.
def test_base_token_needs_a_six_to_exit():
    board = BoardState()
    assert get_legal_tokens(board, player_id=0, roll=5) == ()
    assert get_legal_tokens(board, player_id=0, roll=6) == (0, 1, 2, 3)


# Checks a base exit is blocked when an opponent blockade sits on the entry square.
def test_base_exit_blocked_by_opponent_blockade_on_entry_square():
    board = BoardState()
    board.set(1, 0, 39)  # player 1 relative 39 -> global (13+39)%52 = 0, player 0's entry
    board.set(1, 1, 39)
    assert get_legal_tokens(board, player_id=0, roll=6) == ()


# Checks a token advances by exactly the roll on the main track.
def test_main_track_token_advances_by_roll():
    board = BoardState()
    board.set(0, 0, 10)
    assert get_legal_tokens(board, player_id=0, roll=4) == (0,)


# Checks a move is illegal if any square it must pass over is opponent-blockaded.
def test_main_track_move_blocked_by_opponent_blockade_in_path():
    board = BoardState()
    board.set(0, 0, 10)
    board.set(2, 0, 38)  # player 2 relative 38 -> global (26+38)%52 = 12, in the path to 14
    board.set(2, 1, 38)
    assert get_legal_tokens(board, player_id=0, roll=4) == ()


# Checks passing over a single (non-blockade) opponent token is legal.
def test_main_track_move_not_blocked_by_a_single_opponent_token():
    board = BoardState()
    board.set(0, 0, 10)
    board.set(2, 0, 38)  # single player-2 token at global 12, no blockade
    assert get_legal_tokens(board, player_id=0, roll=4) == (0,)


# Checks a player's own blockade never blocks that player's own other tokens.
def test_own_blockade_does_not_block_own_tokens():
    board = BoardState()
    board.set(0, 0, 20)
    board.set(0, 1, 20)  # own blockade at relative 20
    board.set(0, 2, 15)  # would land on/pass its own blockade at 20
    assert get_legal_tokens(board, player_id=0, roll=5) == (0, 1, 2)


# Checks an exact roll onto the final home square is legal.
def test_exact_roll_onto_finish_is_legal():
    board = BoardState()
    board.set(0, 0, 55)
    assert get_legal_tokens(board, player_id=0, roll=2) == (0,)


# Checks a roll that overshoots the final home square leaves that token with no legal move.
def test_overshoot_past_finish_has_no_legal_move():
    board = BoardState()
    board.set(0, 0, 55)
    assert get_legal_tokens(board, player_id=0, roll=4) == ()


# Checks a home-column destination already occupied by the player's own token is illegal.
def test_home_column_no_stacking_on_own_token():
    board = BoardState()
    board.set(0, 0, 53)  # occupies the destination; itself overshoots this roll (53+5=58)
    board.set(0, 1, 48)  # would land on 53, blocked by its own token already there
    assert get_legal_tokens(board, player_id=0, roll=5) == ()


# Checks a finished token is never offered as a legal move.
def test_finished_token_has_no_legal_move():
    board = BoardState()
    board.set(0, 0, FINISH_POSITION)
    assert get_legal_tokens(board, player_id=0, roll=3) == ()


# Checks the action mask agrees with get_legal_tokens.
def test_action_mask_matches_legal_tokens():
    board = BoardState()
    board.set(0, 0, 10)
    board.set(0, 2, 55)
    mask = get_action_mask(board, player_id=0, roll=2)
    assert mask.tolist() == [True, False, True, False]
