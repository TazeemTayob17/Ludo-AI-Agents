# Optional per-token features describing each legal move's effect and the danger around each token.

from __future__ import annotations

import numpy as np

from env.board import (
    NUM_TOKENS_PER_PLAYER,
    BoardState,
    is_home,
    is_in_home_column,
    is_on_main_track,
    is_safe_square,
    relative_to_global,
)
from env.legal_moves import get_legal_tokens
from env.moves import apply_move
from env.threats import MAX_ROLL, build_occupancy, is_token_threatened, nearest_opponent_behind, target_distance

MOVE_FEATURE_NAMES = (
    "is_legal",
    "captures",
    "exits_base",
    "reaches_home",
    "forms_blockade",
    "lands_safe",
    "escapes_threat",
    "ends_in_danger",
)
THREAT_FEATURE_NAMES = ("is_threatened", "opponent_behind_closeness", "capture_opportunity")

NUM_MOVE_FEATURES = NUM_TOKENS_PER_PLAYER * len(MOVE_FEATURE_NAMES)
NUM_THREAT_FEATURES = NUM_TOKENS_PER_PLAYER * len(THREAT_FEATURE_NAMES)
OPPONENT_BEHIND_HORIZON = 2 * MAX_ROLL


# Simulates one legal move on a copy of the board and returns its 8 effect flags.
def _move_effects(board: BoardState, player_id: int, token_id: int, roll: int, threatened_before: bool) -> list[float]:
    after = board.copy()
    outcome = apply_move(after, player_id, token_id, roll)
    new_position = after.get(player_id, token_id)

    forms_blockade = False
    lands_safe = is_in_home_column(new_position)
    if is_on_main_track(new_position):
        square = relative_to_global(player_id, new_position)
        lands_safe = is_safe_square(square)
        own_on_square = sum(1 for t in range(NUM_TOKENS_PER_PLAYER) if after.global_square(player_id, t) == square)
        forms_blockade = own_on_square >= 2

    threatened_after = is_token_threatened(after, player_id, token_id)
    return [
        1.0,
        float(outcome.captured_player is not None),
        float(outcome.exited_base),
        float(is_home(new_position)),
        float(forms_blockade),
        float(lands_safe),
        float(threatened_before and not threatened_after),
        float(threatened_after),
    ]


# Encodes what moving each of the acting player's 4 tokens would do with this roll (zeros if illegal).
def encode_move_features(board: BoardState, player_id: int, roll: int | None) -> np.ndarray:
    features = np.zeros((NUM_TOKENS_PER_PLAYER, len(MOVE_FEATURE_NAMES)), dtype=np.float32)
    if roll is None:
        return features.reshape(NUM_MOVE_FEATURES)
    occupancy = build_occupancy(board)
    for token_id in get_legal_tokens(board, player_id, roll):
        threatened_before = is_token_threatened(board, player_id, token_id, occupancy)
        features[token_id] = _move_effects(board, player_id, token_id, roll, threatened_before)
    return features.reshape(NUM_MOVE_FEATURES)


# Encodes each of the acting player's tokens' danger and attacking chances on the current board.
def encode_threat_features(board: BoardState, player_id: int) -> np.ndarray:
    features = np.zeros((NUM_TOKENS_PER_PLAYER, len(THREAT_FEATURE_NAMES)), dtype=np.float32)
    occupancy = build_occupancy(board)
    for token_id in range(NUM_TOKENS_PER_PLAYER):
        behind = nearest_opponent_behind(board, player_id, token_id, OPPONENT_BEHIND_HORIZON)
        target = target_distance(board, player_id, token_id, occupancy)
        features[token_id] = (
            float(is_token_threatened(board, player_id, token_id, occupancy)),
            0.0 if behind is None else (OPPONENT_BEHIND_HORIZON + 1 - behind) / OPPONENT_BEHIND_HORIZON,
            0.0 if target is None else (MAX_ROLL + 1 - target) / MAX_ROLL,
        )
    return features.reshape(NUM_THREAT_FEATURES)
