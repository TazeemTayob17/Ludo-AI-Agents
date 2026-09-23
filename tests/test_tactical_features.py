# Tests for env/tactical_features.py's per-move and per-token threat features.

import numpy as np
import pytest

from env.board import BoardState
from env.tactical_features import (
    MOVE_FEATURE_NAMES,
    NUM_MOVE_FEATURES,
    NUM_THREAT_FEATURES,
    THREAT_FEATURE_NAMES,
    encode_move_features,
    encode_threat_features,
)


# Builds an empty board with only the given (player, token, relative position) placements.
def _board(*placements):
    board = BoardState()
    for player_id, token_id, position in placements:
        board.set(player_id, token_id, position)
    return board


# Returns one token's move features as a name -> value dict.
def _moves(board, roll, token_id):
    features = encode_move_features(board, 0, roll).reshape(4, len(MOVE_FEATURE_NAMES))
    return dict(zip(MOVE_FEATURE_NAMES, features[token_id]))


# Returns one token's threat features as a name -> value dict.
def _threats(board, token_id):
    features = encode_threat_features(board, 0).reshape(4, len(THREAT_FEATURE_NAMES))
    return dict(zip(THREAT_FEATURE_NAMES, features[token_id]))


# Checks both blocks have the documented sizes.
def test_feature_block_sizes():
    board = BoardState()
    assert encode_move_features(board, 0, 6).shape == (NUM_MOVE_FEATURES,) == (32,)
    assert encode_threat_features(board, 0).shape == (NUM_THREAT_FEATURES,) == (12,)


# Checks a terminal state with no pending roll gives all-zero move features.
def test_no_pending_roll_gives_zero_move_features():
    assert not encode_move_features(_board((0, 0, 10)), 0, None).any()


# Checks an illegal token's features are all zero while a legal one is flagged.
def test_illegal_tokens_are_all_zero():
    board = _board((0, 0, 10))
    assert _moves(board, 3, 0)["is_legal"] == 1.0
    assert not any(_moves(board, 3, 1).values())


# Checks a move that lands on a lone opponent is flagged as a capture.
def test_capture_is_flagged():
    board = _board((0, 0, 10), (2, 0, 38))
    assert _moves(board, 2, 0)["captures"] == 1.0


# Checks leaving Base on a 6 is flagged as exiting base and landing safe.
def test_exit_base_is_flagged_and_safe():
    features = _moves(BoardState(), 6, 0)
    assert features["exits_base"] == 1.0
    assert features["lands_safe"] == 1.0


# Checks an exact roll to Home is flagged.
def test_reaching_home_is_flagged():
    assert _moves(_board((0, 0, 55)), 2, 0)["reaches_home"] == 1.0


# Checks landing on another of our own tokens on the main track forms a blockade.
def test_forming_a_blockade_is_flagged():
    board = _board((0, 0, 10), (0, 1, 12))
    assert _moves(board, 2, 0)["forms_blockade"] == 1.0


# Checks moving a threatened token onto a safe square counts as escaping, not ending in danger.
def test_escaping_a_threat_is_flagged():
    board = _board((0, 0, 10), (1, 0, 45))
    features = _moves(board, 3, 0)
    assert features["escapes_threat"] == 1.0
    assert features["ends_in_danger"] == 0.0


# Checks moving a safe token to within an opponent's reach is flagged as ending in danger.
def test_moving_into_danger_is_flagged():
    board = _board((0, 0, 16), (1, 0, 5))
    features = _moves(board, 4, 0)
    assert features["ends_in_danger"] == 1.0
    assert features["escapes_threat"] == 0.0


# Checks the threat block's three values for a token with an attacker behind and a target ahead.
def test_threat_features_values():
    board = _board((0, 0, 10), (1, 0, 45), (2, 0, 38))
    features = _threats(board, 0)
    assert features["is_threatened"] == 1.0
    assert features["opponent_behind_closeness"] == pytest.approx(9 / 12)
    assert features["capture_opportunity"] == pytest.approx(5 / 6)


# Checks a token in Base carries no threat information.
def test_threat_features_are_zero_off_the_track():
    assert not np.any(list(_threats(BoardState(), 0).values()))
