# Optional per-token features describing each legal move's effect and the danger around each token.

from __future__ import annotations

import numpy as np

from env.board import (
    FINISH_POSITION,
    NUM_PLAYERS,
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
from env.threats import (
    MAX_ROLL,
    build_occupancy,
    capturing_rolls,
    count_attackers,
    is_token_threatened,
    nearest_opponent_behind,
    target_distance,
)

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

GRADED_MOVE_FEATURE_NAMES = (
    "progress_gained",
    "distance_to_home_after",
    "capture_value",
    "attackers_after",
    "capture_risk_after",
    "breaks_own_blockade",
    "blocks_opponent",
)
NUM_RACE_FEATURES = 2 * NUM_PLAYERS
RISK_FEATURE_NAMES = ("moved_token_capture_chance", "expected_progress_lost")
NUM_RISK_FEATURES = NUM_TOKENS_PER_PLAYER * len(RISK_FEATURE_NAMES)
NUM_GRADED_FEATURES = NUM_TOKENS_PER_PLAYER * len(GRADED_MOVE_FEATURE_NAMES) + NUM_RACE_FEATURES
MAX_COUNTED_ATTACKERS = 3
_TRACK_SPAN = FINISH_POSITION + 1


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


# Counts this player's tokens sharing one main-track global square (0 if that square is None).
def _own_tokens_on(board: BoardState, player_id: int, square: int | None) -> int:
    if square is None:
        return 0
    return sum(1 for t in range(NUM_TOKENS_PER_PLAYER) if board.global_square(player_id, t) == square)


# Simulates one legal move on a copy of the board and returns its 7 graded effect values.
def _graded_move_effects(board: BoardState, player_id: int, token_id: int, roll: int) -> list[float]:
    old_position = board.get(player_id, token_id)
    breaks_blockade = _own_tokens_on(board, player_id, board.global_square(player_id, token_id)) >= 2

    after = board.copy()
    outcome = apply_move(after, player_id, token_id, roll)
    new_position = after.get(player_id, token_id)

    capture_value = 0.0
    if outcome.captured_player is not None:
        capture_value = (board.get(outcome.captured_player, outcome.captured_token) + 1) / _TRACK_SPAN

    attackers = 0
    risk = 0.0
    blocks_opponent = False
    if is_on_main_track(new_position):
        square = relative_to_global(player_id, new_position)
        occupancy = build_occupancy(after)
        attackers = count_attackers(after, player_id, square, occupancy)
        risk = len(capturing_rolls(after, player_id, square, occupancy)) / MAX_ROLL
        if _own_tokens_on(after, player_id, square) >= 2:
            blocks_opponent = nearest_opponent_behind(after, player_id, token_id, MAX_ROLL) is not None

    return [
        (new_position - old_position) / MAX_ROLL,
        (FINISH_POSITION - new_position) / _TRACK_SPAN,
        capture_value,
        min(attackers, MAX_COUNTED_ATTACKERS) / MAX_COUNTED_ATTACKERS,
        risk,
        float(breaks_blockade),
        float(blocks_opponent),
    ]


# Encodes each seat's total progress and tokens-in-base share, acting player first, so the agent can see who is ahead.
def _race_standing(board: BoardState, player_id: int) -> np.ndarray:
    progress = []
    in_base = []
    for offset in range(NUM_PLAYERS):
        seat = (player_id + offset) % NUM_PLAYERS
        positions = [board.get(seat, t) for t in range(NUM_TOKENS_PER_PLAYER)]
        progress.append(sum(p + 1 for p in positions) / (NUM_TOKENS_PER_PLAYER * _TRACK_SPAN))
        in_base.append(sum(1 for p in positions if p < 0) / NUM_TOKENS_PER_PLAYER)
    return np.array(progress + in_base, dtype=np.float32)


# Encodes graded per-move values (zeros if illegal) for the 4 tokens, then the 8 race-standing values.
def encode_graded_features(board: BoardState, player_id: int, roll: int | None) -> np.ndarray:
    per_token = np.zeros((NUM_TOKENS_PER_PLAYER, len(GRADED_MOVE_FEATURE_NAMES)), dtype=np.float32)
    if roll is not None:
        for token_id in get_legal_tokens(board, player_id, roll):
            per_token[token_id] = _graded_move_effects(board, player_id, token_id, roll)
    return np.concatenate([per_token.reshape(-1), _race_standing(board, player_id)])


# Encodes, per legal move, the chance the moved token is captured next turn and the expected share of own progress lost to captures afterwards (zeros if illegal).
def encode_risk_features(board: BoardState, player_id: int, roll: int | None) -> np.ndarray:
    features = np.zeros((NUM_TOKENS_PER_PLAYER, len(RISK_FEATURE_NAMES)), dtype=np.float32)
    if roll is None:
        return features.reshape(NUM_RISK_FEATURES)
    for token_id in get_legal_tokens(board, player_id, roll):
        after = board.copy()
        apply_move(after, player_id, token_id, roll)
        occupancy = build_occupancy(after)
        expected_loss = 0.0
        for other in range(NUM_TOKENS_PER_PLAYER):
            position = after.get(player_id, other)
            if not is_on_main_track(position):
                continue
            chance = len(capturing_rolls(after, player_id, relative_to_global(player_id, position), occupancy)) / MAX_ROLL
            if other == token_id:
                features[token_id, 0] = chance
            expected_loss += chance * (position + 1) / (NUM_TOKENS_PER_PLAYER * _TRACK_SPAN)
        features[token_id, 1] = expected_loss
    return features.reshape(NUM_RISK_FEATURES)


# Encodes each seat's total progress and tokens-in-base share, acting player first (the race-standing part of the graded block on its own).
def encode_race_features(board: BoardState, player_id: int) -> np.ndarray:
    return _race_standing(board, player_id)
