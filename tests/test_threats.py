# Tests for env/threats.py's rule-accurate next-roll capture checks.

from env.board import BoardState
from env.threats import capture_distance, is_token_threatened, nearest_opponent_behind, target_distance


# Builds an empty board with only the given (player, token, relative position) placements.
def _board(*placements):
    board = BoardState()
    for player_id, token_id, position in placements:
        board.set(player_id, token_id, position)
    return board


# Checks an opponent 4 squares behind on the main track is a threat at distance 4.
def test_opponent_within_six_behind_is_a_threat():
    board = _board((0, 0, 10), (1, 0, 45))
    assert capture_distance(board, 0, 10) == 4
    assert is_token_threatened(board, 0, 0)


# Checks an opponent about to turn into its own home column cannot reach, unlike a raw distance check.
def test_opponent_turning_into_its_home_column_is_not_a_threat():
    board = _board((0, 0, 12), (1, 0, 48))
    assert capture_distance(board, 0, 12) is None
    assert not is_token_threatened(board, 0, 0)


# Checks a token on a safe square is never threatened.
def test_token_on_safe_square_is_not_threatened():
    board = _board((0, 0, 8), (1, 0, 45))
    assert not is_token_threatened(board, 0, 0)


# Checks a blockade in the attacker's path stops the threat.
def test_blockade_in_the_path_blocks_the_threat():
    board = _board((0, 0, 10), (1, 0, 45), (0, 1, 7), (0, 2, 7))
    assert not is_token_threatened(board, 0, 0)
    board.set(0, 2, 8)
    assert is_token_threatened(board, 0, 0)


# Checks two of our own tokens on one square form a blockade the attacker cannot land on.
def test_own_blockade_cannot_be_landed_on():
    board = _board((0, 0, 10), (0, 1, 10), (1, 0, 45))
    assert not is_token_threatened(board, 0, 0)


# Checks tokens in Base or the home column are never threatened.
def test_tokens_off_the_main_track_are_not_threatened():
    board = _board((0, 1, 53), (1, 0, 45))
    assert not is_token_threatened(board, 0, 0)
    assert not is_token_threatened(board, 0, 1)


# Checks target_distance finds the smallest roll that captures an opponent ahead.
def test_target_distance_finds_the_capturing_roll():
    board = _board((0, 0, 10), (2, 0, 38))
    assert target_distance(board, 0, 0) == 2


# Checks there is no capture target when the only opponent ahead sits on a safe square.
def test_target_on_a_safe_square_is_not_a_target():
    board = _board((0, 0, 10), (1, 0, 0))
    assert target_distance(board, 0, 0) is None


# Checks nearest_opponent_behind looks beyond one roll, up to the given horizon.
def test_nearest_opponent_behind_reaches_past_one_roll():
    board = _board((0, 0, 10), (1, 0, 40))
    assert nearest_opponent_behind(board, 0, 0, max_distance=12) == 9
    assert nearest_opponent_behind(board, 0, 0, max_distance=6) is None
    assert not is_token_threatened(board, 0, 0)
