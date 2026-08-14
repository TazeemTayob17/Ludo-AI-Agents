# Egocentric observation encoding for the learning agent.

from __future__ import annotations

import numpy as np

from env.board import (
    BoardState,
    FINISH_POSITION,
    NUM_PLAYERS,
    NUM_TOKENS_PER_PLAYER,
    is_base,
    is_home,
    is_on_main_track,
    is_safe_square,
)

FEATURE_NAMES = ("normalized_position", "is_base", "is_home", "is_safe")
NUM_FEATURES_PER_TOKEN = len(FEATURE_NAMES)
OBSERVATION_SIZE = NUM_PLAYERS * NUM_TOKENS_PER_PLAYER * NUM_FEATURES_PER_TOKEN

_NORMALIZATION_DENOMINATOR = FINISH_POSITION + 1


# Returns the seat order starting from acting_player, then clockwise turn order.
def seat_order(acting_player: int) -> tuple[int, ...]:
    return tuple((acting_player + offset) % NUM_PLAYERS for offset in range(NUM_PLAYERS))


# Builds the 4-feature vector for a single token.
def _encode_token(board: BoardState, player_id: int, token_id: int) -> tuple[float, float, float, float]:
    relative_position = board.get(player_id, token_id)

    normalized_position = (relative_position + 1) / _NORMALIZATION_DENOMINATOR
    base_flag = 1.0 if is_base(relative_position) else 0.0
    home_flag = 1.0 if is_home(relative_position) else 0.0

    if is_on_main_track(relative_position):
        global_square = board.global_square(player_id, token_id)
        assert global_square is not None
        safe_flag = 1.0 if is_safe_square(global_square) else 0.0
    else:
        safe_flag = 1.0

    return normalized_position, base_flag, home_flag, safe_flag


# Builds the full egocentric observation vector for the acting player.
def encode_observation(board: BoardState, acting_player: int) -> np.ndarray:
    features = np.empty(
        (NUM_PLAYERS, NUM_TOKENS_PER_PLAYER, NUM_FEATURES_PER_TOKEN), dtype=np.float32
    )
    for seat_offset, player_id in enumerate(seat_order(acting_player)):
        for token_id in range(NUM_TOKENS_PER_PLAYER):
            features[seat_offset, token_id] = _encode_token(board, player_id, token_id)
    return features.reshape(OBSERVATION_SIZE)
