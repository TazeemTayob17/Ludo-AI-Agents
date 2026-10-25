# Tests for the track board layout: sizes, square and channel placement, seat rotation, counts and unchanged defaults.

import numpy as np

from env.board import BoardState
from env.state_encoding import (
    OBSERVATION_SIZE,
    TRACK_ENCODING_SIZE,
    ObservationSpec,
    encode_observation,
    observation_size,
)

TRACK = ObservationSpec(board_encoding="track")
OWN_HOME_COLUMN_START = 208
BASE_COUNTS_START = 217
HOME_COUNTS_START = 221


# Returns the value on the track block for a seat channel (0 = acting player) and ego square.
def _track_value(obs: np.ndarray, channel: int, square: int) -> float:
    return float(obs[channel * 52 + square])


# Checks the track block's size and that the default spec still gives the original 64 values.
def test_sizes_and_default_unchanged():
    assert TRACK_ENCODING_SIZE == 241
    assert observation_size(TRACK) == 241
    assert observation_size(ObservationSpec(board_encoding="track", include_dice_roll=True)) == 247
    assert observation_size() == OBSERVATION_SIZE == 64
    assert encode_observation(BoardState(), 0).shape == (64,)
    assert encode_observation(BoardState(), 0, spec=TRACK).shape == (241,)


# Checks an own token lands on its relative square in channel 0.
def test_own_token_square():
    board = BoardState()
    board.set(2, 1, 5)
    obs = encode_observation(board, 2, spec=TRACK)
    assert _track_value(obs, 0, 5) == 0.5
    assert obs[:52].sum() == 0.5


# Checks an opponent token appears at its square measured from the acting player's start, in its seat channel.
def test_opponent_token_is_in_the_acting_players_frame():
    board = BoardState()
    board.set(1, 0, 0)
    obs = encode_observation(board, 0, spec=TRACK)
    assert _track_value(obs, 1, 13) == 0.5
    obs_from_seat_2 = encode_observation(board, 2, spec=TRACK)
    assert _track_value(obs_from_seat_2, 3, 39) == 0.5


# Checks a stack of three tokens is capped at the same value as two.
def test_stack_is_capped():
    board = BoardState()
    for token in range(3):
        board.set(0, token, 7)
    assert _track_value(encode_observation(board, 0, spec=TRACK), 0, 7) == 1.0


# Checks base, home and home-column counts and the opponent home-column block.
def test_base_home_and_home_column_counts():
    board = BoardState()
    board.set(0, 0, 57)
    board.set(0, 1, 52)
    board.set(1, 0, 54)
    board.set(1, 1, 57)
    obs = encode_observation(board, 0, spec=TRACK)
    assert obs[OWN_HOME_COLUMN_START + 1] == 1.0
    assert obs[214] == 0.25
    assert obs[BASE_COUNTS_START] == 0.5
    assert obs[BASE_COUNTS_START + 1] == 0.5
    assert obs[HOME_COUNTS_START] == 0.25
    assert obs[HOME_COUNTS_START + 1] == 0.25


# Checks that shifting every seat by one and re-encoding from the next seat gives the same vector.
def test_seat_rotation_invariance():
    rng = np.random.default_rng(0)
    for _ in range(20):
        board = BoardState()
        for player in range(4):
            for token in range(4):
                board.set(player, token, int(rng.integers(-1, 58)))
        rotated = BoardState()
        for player in range(4):
            rotated.positions[player] = board.positions[(player + 1) % 4]
        assert np.array_equal(encode_observation(board, 1, spec=TRACK), encode_observation(rotated, 0, spec=TRACK))
