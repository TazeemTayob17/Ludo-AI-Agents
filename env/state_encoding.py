# Egocentric observation encoding for the learning agent.

from __future__ import annotations

from dataclasses import dataclass

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
from env.tactical_features import NUM_MOVE_FEATURES, NUM_THREAT_FEATURES, encode_move_features, encode_threat_features

FEATURE_NAMES = ("normalized_position", "is_base", "is_home", "is_safe")
NUM_FEATURES_PER_TOKEN = len(FEATURE_NAMES)
OBSERVATION_SIZE = NUM_PLAYERS * NUM_TOKENS_PER_PLAYER * NUM_FEATURES_PER_TOKEN

NUM_ROLL_FEATURES = 6
OBSERVATION_SIZE_WITH_ROLL = OBSERVATION_SIZE + NUM_ROLL_FEATURES

_NORMALIZATION_DENOMINATOR = FINISH_POSITION + 1


# Which optional feature blocks follow the 64 board features; all off reproduces the original observation.
@dataclass(frozen=True)
class ObservationSpec:
    include_dice_roll: bool = False
    include_move_features: bool = False
    include_threat_features: bool = False


# Returns the observation vector length for a given feature spec.
def observation_size(spec: ObservationSpec | None = None) -> int:
    spec = spec or ObservationSpec()
    size = OBSERVATION_SIZE
    if spec.include_dice_roll:
        size += NUM_ROLL_FEATURES
    if spec.include_move_features:
        size += NUM_MOVE_FEATURES
    if spec.include_threat_features:
        size += NUM_THREAT_FEATURES
    return size


# Encodes a die value 1-6 as a one-hot vector; an absent roll (terminal state) is all zeros.
def encode_roll(roll: int | None) -> np.ndarray:
    one_hot = np.zeros(NUM_ROLL_FEATURES, dtype=np.float32)
    if roll is not None:
        one_hot[roll - 1] = 1.0
    return one_hot


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


# Builds the egocentric observation for the acting player, appending whichever optional blocks the spec enables.
def encode_observation(
    board: BoardState, acting_player: int, roll: int | None = None, spec: ObservationSpec | None = None
) -> np.ndarray:
    spec = spec or ObservationSpec()
    features = np.empty(
        (NUM_PLAYERS, NUM_TOKENS_PER_PLAYER, NUM_FEATURES_PER_TOKEN), dtype=np.float32
    )
    for seat_offset, player_id in enumerate(seat_order(acting_player)):
        for token_id in range(NUM_TOKENS_PER_PLAYER):
            features[seat_offset, token_id] = _encode_token(board, player_id, token_id)
    blocks = [features.reshape(OBSERVATION_SIZE)]
    if spec.include_dice_roll:
        blocks.append(encode_roll(roll))
    if spec.include_move_features:
        blocks.append(encode_move_features(board, acting_player, roll))
    if spec.include_threat_features:
        blocks.append(encode_threat_features(board, acting_player))
    return blocks[0] if len(blocks) == 1 else np.concatenate(blocks)
