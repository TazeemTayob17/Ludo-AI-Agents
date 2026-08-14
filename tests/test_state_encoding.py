# Tests for env/state_encoding.py's egocentric observation encoding.

import numpy as np

from env.board import BASE_POSITION, FINISH_POSITION, BoardState
from env.state_encoding import (
    NUM_FEATURES_PER_TOKEN,
    OBSERVATION_SIZE,
    encode_observation,
    seat_order,
)


# Checks seat_order rotates starting from the acting player.
def test_seat_order_rotates_from_acting_player():
    assert seat_order(0) == (0, 1, 2, 3)
    assert seat_order(1) == (1, 2, 3, 0)
    assert seat_order(3) == (3, 0, 1, 2)


# Checks the observation vector has the expected shape and dtype.
def test_observation_shape_and_dtype():
    board = BoardState()
    obs = encode_observation(board, acting_player=0)
    assert obs.shape == (OBSERVATION_SIZE,)
    assert obs.dtype == np.float32


# Checks every token is flagged base/safe when the board is freshly reset.
def test_empty_board_all_tokens_in_base():
    board = BoardState()
    obs = encode_observation(board, acting_player=0).reshape(4, 4, NUM_FEATURES_PER_TOKEN)
    expected_token = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    for seat_offset in range(4):
        for token_id in range(4):
            np.testing.assert_allclose(obs[seat_offset, token_id], expected_token)


# Checks the same board state reorders depending on which player is acting.
def test_egocentric_reordering_moves_acting_players_tokens_to_first_block():
    board = BoardState()
    board.set(0, 0, 20)
    board.set(1, 1, 30)

    obs_as_player0 = encode_observation(board, acting_player=0).reshape(4, 4, NUM_FEATURES_PER_TOKEN)
    obs_as_player1 = encode_observation(board, acting_player=1).reshape(4, 4, NUM_FEATURES_PER_TOKEN)

    expected_p0_token0_norm = np.float32((20 + 1) / 58)

    assert obs_as_player0[0, 0, 0] == expected_p0_token0_norm

    assert seat_order(1)[3] == 0
    assert obs_as_player1[3, 0, 0] == expected_p0_token0_norm

    expected_p1_token1_norm = np.float32((30 + 1) / 58)
    assert obs_as_player1[0, 1, 0] == expected_p1_token1_norm


# Checks a finished token is flagged home and safe.
def test_finished_token_is_flagged_home_and_safe():
    board = BoardState()
    board.set(0, 0, FINISH_POSITION)
    obs = encode_observation(board, acting_player=0).reshape(4, 4, NUM_FEATURES_PER_TOKEN)
    normalized_position, is_base_flag, is_home_flag, is_safe_flag = obs[0, 0]
    assert is_home_flag == 1.0
    assert is_safe_flag == 1.0
    assert is_base_flag == 0.0
    assert normalized_position == np.float32((FINISH_POSITION + 1) / 58)


# Checks a base token is flagged base and safe.
def test_base_token_is_flagged_base_and_safe():
    board = BoardState()
    obs = encode_observation(board, acting_player=2).reshape(4, 4, NUM_FEATURES_PER_TOKEN)
    normalized_position, is_base_flag, is_home_flag, is_safe_flag = obs[0, 0]
    assert is_base_flag == 1.0
    assert is_safe_flag == 1.0
    assert is_home_flag == 0.0
    assert normalized_position == np.float32((BASE_POSITION + 1) / 58)


# Checks a token on a non-safe main-track square is not flagged safe.
def test_token_on_unsafe_main_track_square_is_not_flagged_safe():
    board = BoardState()
    board.set(0, 0, 20)
    obs = encode_observation(board, acting_player=0).reshape(4, 4, NUM_FEATURES_PER_TOKEN)
    assert obs[0, 0, 3] == 0.0


# Checks a token on a safe main-track square is flagged safe.
def test_token_on_safe_main_track_square_is_flagged_safe():
    board = BoardState()
    board.set(1, 0, 8)
    obs = encode_observation(board, acting_player=1).reshape(4, 4, NUM_FEATURES_PER_TOKEN)
    assert obs[0, 0, 3] == 1.0
