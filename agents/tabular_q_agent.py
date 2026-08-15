# The Tabular Q-learning agent: discretized state, a Q-table, and masked Q-learning updates.

from __future__ import annotations

import numpy as np

from env.board import (
    MAIN_TRACK_LENGTH,
    NUM_PLAYERS,
    NUM_TOKENS_PER_PLAYER,
    BoardState,
    is_base,
    is_home,
    is_in_home_column,
    is_on_main_track,
    is_safe_square,
    relative_to_global,
)

_ZONE_BASE = 0
_ZONE_EARLY = 1
_ZONE_MID = 2
_ZONE_LATE = 3
_ZONE_HOME_STRETCH = 4
_ZONE_HOME = 5
_NUM_ZONES = 6

# Buckets a single token's raw position into one of 6 coarse progress zones.
def _zone(position: int) -> int:
    if is_base(position):
        return _ZONE_BASE
    if is_home(position):
        return _ZONE_HOME
    if is_in_home_column(position):
        return _ZONE_HOME_STRETCH
    if position <= 16:
        return _ZONE_EARLY
    if position <= 33:
        return _ZONE_MID
    return _ZONE_LATE

# Checks whether any opponent token could capture this token with a single roll of 1-6.
def _is_threatened(board: BoardState, player_id: int, token_id: int) -> bool:
    position = board.get(player_id, token_id)
    if not is_on_main_track(position):
        return False
    square = relative_to_global(player_id, position)
    if is_safe_square(square):
        return False
    for other_player in range(NUM_PLAYERS):
        if other_player == player_id:
            continue
        for other_token in range(NUM_TOKENS_PER_PLAYER):
            other_position = board.get(other_player, other_token)
            if not is_on_main_track(other_position):
                continue
            other_square = relative_to_global(other_player, other_position)
            distance = (square - other_square) % MAIN_TRACK_LENGTH
            if 1 <= distance <= 6:
                return True
    return False

# Combines a token's zone and threatened flag into one small descriptor value.
def _token_descriptor(board: BoardState, player_id: int, token_id: int) -> int:
    position = board.get(player_id, token_id)
    threatened = 1 if _is_threatened(board, player_id, token_id) else 0
    return _zone(position) * 2 + threatened

DISCRETE_STATE_VALUES_PER_TOKEN = _NUM_ZONES * 2

# Discretizes the board into the acting player's 4-token state key for the Q-table.
def discretize_state(board: BoardState, player_id: int) -> tuple[int, ...]:
    return tuple(_token_descriptor(board, player_id, token_id) for token_id in range(NUM_TOKENS_PER_PLAYER))

# Holds the Q-table and implements epsilon-greedy action selection plus the masked Q-learning update.
class TabularQAgent:

    # Sets the learning rate, discount factor, exploration rate, and an empty table.
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self._rng = rng if rng is not None else np.random.default_rng()
        self.q_table: dict[tuple[int, ...], np.ndarray] = {}

    # Returns the Q-value row for a state, creating a zero-initialized one if unseen.
    def q_values(self, state: tuple[int, ...]) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(NUM_TOKENS_PER_PLAYER, dtype=np.float64)
        return self.q_table[state]

    # Matches the ChooseActionFn signature: epsilon-greedy over the legal tokens only.
    def __call__(self, board: BoardState, player_id: int, roll: int, legal_tokens: tuple[int, ...]) -> int:
        if self._rng.random() < self.epsilon:
            index = self._rng.integers(0, len(legal_tokens))
            return legal_tokens[index]
        state = discretize_state(board, player_id)
        values = self.q_values(state)
        return max(legal_tokens, key=lambda token_id: (values[token_id], -token_id))

    # Applies one masked Q-learning update: the next-state max is taken over legal actions only.
    def update(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...] | None,
        next_legal_tokens: tuple[int, ...],
        done: bool,
    ) -> None:
        values = self.q_values(state)
        if done or not next_legal_tokens:
            target = reward
        else:
            next_values = self.q_values(next_state)
            target = reward + self.gamma * max(next_values[token_id] for token_id in next_legal_tokens)
        values[action] += self.alpha * (target - values[action])
