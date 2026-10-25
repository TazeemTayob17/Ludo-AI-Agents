# Egocentric observation encoding for the learning agent.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from env.board import (
    ENTRY_OFFSETS,
    HOME_COLUMN_START,
    MAIN_TRACK_LENGTH,
    BoardState,
    FINISH_POSITION,
    NUM_PLAYERS,
    NUM_TOKENS_PER_PLAYER,
    is_base,
    is_home,
    is_in_home_column,
    is_on_main_track,
    is_safe_square,
    relative_to_global,
)
from env.tactical_features import (
    NUM_GRADED_FEATURES,
    NUM_MOVE_FEATURES,
    NUM_RACE_FEATURES,
    NUM_RISK_FEATURES,
    NUM_THREAT_FEATURES,
    encode_graded_features,
    encode_move_features,
    encode_race_features,
    encode_risk_features,
    encode_threat_features,
)

FEATURE_NAMES = ("normalized_position", "is_base", "is_home", "is_safe")
NUM_FEATURES_PER_TOKEN = len(FEATURE_NAMES)
OBSERVATION_SIZE = NUM_PLAYERS * NUM_TOKENS_PER_PLAYER * NUM_FEATURES_PER_TOKEN

NUM_ROLL_FEATURES = 6
OBSERVATION_SIZE_WITH_ROLL = OBSERVATION_SIZE + NUM_ROLL_FEATURES

_NORMALIZATION_DENOMINATOR = FINISH_POSITION + 1

BOARD_ENCODINGS = ("token", "track")
HOME_COLUMN_SQUARES = 6
TRACK_ENCODING_SIZE = (
    MAIN_TRACK_LENGTH * NUM_PLAYERS
    + HOME_COLUMN_SQUARES
    + (NUM_PLAYERS - 1)
    + 2 * NUM_PLAYERS
    + NUM_TOKENS_PER_PLAYER * NUM_FEATURES_PER_TOKEN
)


# Which optional feature blocks follow the 64 board features; all off reproduces the original observation.
@dataclass(frozen=True)
class ObservationSpec:
    include_dice_roll: bool = False
    include_move_features: bool = False
    include_threat_features: bool = False
    include_graded_features: bool = False
    include_risk_features: bool = False
    include_race_features: bool = False
    board_encoding: str = "token"
    canonical_token_order: bool = False

    # Rejects unknown encodings and the unsupported canonical-order plus graded-block combination.
    def __post_init__(self) -> None:
        if self.board_encoding not in BOARD_ENCODINGS:
            raise ValueError(f"unknown board_encoding: {self.board_encoding}")
        if self.canonical_token_order and self.include_graded_features:
            raise ValueError("canonical_token_order does not support the graded feature block")


# Returns the observation vector length for a given feature spec.
def observation_size(spec: ObservationSpec | None = None) -> int:
    spec = spec or ObservationSpec()
    size = TRACK_ENCODING_SIZE if spec.board_encoding == "track" else OBSERVATION_SIZE
    if spec.include_dice_roll:
        size += NUM_ROLL_FEATURES
    if spec.include_move_features:
        size += NUM_MOVE_FEATURES
    if spec.include_threat_features:
        size += NUM_THREAT_FEATURES
    if spec.include_graded_features:
        size += NUM_GRADED_FEATURES
    if spec.include_risk_features:
        size += NUM_RISK_FEATURES
    if spec.include_race_features:
        size += NUM_RACE_FEATURES
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


# Returns a player's token ids ordered by progress (furthest first, ties by id), so equal positions look identical to the network.
def canonical_order(board: BoardState, player_id: int) -> tuple[int, ...]:
    return tuple(sorted(range(NUM_TOKENS_PER_PLAYER), key=lambda t: (-board.get(player_id, t), t)))


# Builds the original per-token board block: 4 features for each of the 16 tokens, acting player first.
def _encode_token_board(board: BoardState, acting_player: int, canonical: bool) -> np.ndarray:
    features = np.empty((NUM_PLAYERS, NUM_TOKENS_PER_PLAYER, NUM_FEATURES_PER_TOKEN), dtype=np.float32)
    for seat_offset, player_id in enumerate(seat_order(acting_player)):
        order = canonical_order(board, player_id) if canonical else range(NUM_TOKENS_PER_PLAYER)
        for slot, token_id in enumerate(order):
            features[seat_offset, slot] = _encode_token(board, player_id, token_id)
    return features.reshape(OBSERVATION_SIZE)


# Builds the track board block: token counts per square from the acting player's viewpoint, plus home columns, base and home counts, and the own token list.
def _encode_track_board(board: BoardState, acting_player: int, canonical: bool) -> np.ndarray:
    track = np.zeros((NUM_PLAYERS, MAIN_TRACK_LENGTH), dtype=np.float32)
    own_home_column = np.zeros(HOME_COLUMN_SQUARES, dtype=np.float32)
    opponent_home_column = np.zeros(NUM_PLAYERS - 1, dtype=np.float32)
    base_counts = np.zeros(NUM_PLAYERS, dtype=np.float32)
    home_counts = np.zeros(NUM_PLAYERS, dtype=np.float32)
    for seat_offset, player_id in enumerate(seat_order(acting_player)):
        for token_id in range(NUM_TOKENS_PER_PLAYER):
            position = board.get(player_id, token_id)
            if is_base(position):
                base_counts[seat_offset] += 1
            elif is_home(position):
                home_counts[seat_offset] += 1
            elif is_in_home_column(position):
                if seat_offset == 0:
                    own_home_column[position - HOME_COLUMN_START] += 1
                else:
                    opponent_home_column[seat_offset - 1] += 1
            else:
                square = (relative_to_global(player_id, position) - ENTRY_OFFSETS[acting_player]) % MAIN_TRACK_LENGTH
                track[seat_offset, square] += 1
    own_order = canonical_order(board, acting_player) if canonical else range(NUM_TOKENS_PER_PLAYER)
    own_tokens = np.array([_encode_token(board, acting_player, t) for t in own_order], dtype=np.float32)
    return np.concatenate(
        [
            np.minimum(track, 2.0).reshape(-1) / 2.0,
            own_home_column,
            opponent_home_column / NUM_TOKENS_PER_PLAYER,
            base_counts / NUM_TOKENS_PER_PLAYER,
            home_counts / NUM_TOKENS_PER_PLAYER,
            own_tokens.reshape(-1),
        ]
    )


# Reorders a per-token block (one row per token id) into canonical slot order.
def _reorder_block(block: np.ndarray, order: tuple[int, ...]) -> np.ndarray:
    return block.reshape(NUM_TOKENS_PER_PLAYER, -1)[list(order)].reshape(-1)


# Builds the egocentric observation for the acting player, appending whichever optional blocks the spec enables.
def encode_observation(
    board: BoardState, acting_player: int, roll: int | None = None, spec: ObservationSpec | None = None
) -> np.ndarray:
    spec = spec or ObservationSpec()
    canonical = spec.canonical_token_order
    order = canonical_order(board, acting_player) if canonical else None
    if spec.board_encoding == "track":
        blocks = [_encode_track_board(board, acting_player, canonical)]
    else:
        blocks = [_encode_token_board(board, acting_player, canonical)]
    if spec.include_dice_roll:
        blocks.append(encode_roll(roll))
    if spec.include_move_features:
        move_block = encode_move_features(board, acting_player, roll)
        blocks.append(_reorder_block(move_block, order) if canonical else move_block)
    if spec.include_threat_features:
        threat_block = encode_threat_features(board, acting_player)
        blocks.append(_reorder_block(threat_block, order) if canonical else threat_block)
    if spec.include_graded_features:
        blocks.append(encode_graded_features(board, acting_player, roll))
    if spec.include_risk_features:
        risk_block = encode_risk_features(board, acting_player, roll)
        blocks.append(_reorder_block(risk_block, order) if canonical else risk_block)
    if spec.include_race_features:
        blocks.append(encode_race_features(board, acting_player))
    return blocks[0] if len(blocks) == 1 else np.concatenate(blocks)
