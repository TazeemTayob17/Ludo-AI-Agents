# A simplified, single-agent, no-randomness Ludo environment for Step 9's DQN sanity check.

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.board import BASE_POSITION, FINISH_POSITION, BoardState
from env.legal_moves import get_action_mask, get_legal_tokens
from env.moves import apply_move
from env.rewards import compute_move_reward, truncation_reward, win_reward
from env.state_encoding import OBSERVATION_SIZE, encode_observation
from env.turns import has_player_won

PLAYER_ID = 0
DEFAULT_ROLL_SEQUENCE = (6, 3, 5, 2, 6, 4, 1, 6, 3, 5)
DEFAULT_MAX_STEPS = 200

# Mixed starting layout (Base/mid-track/near-home/already-home) so the mask is never trivially all-legal.
def _default_start_positions() -> tuple[int, int, int, int]:
    return (BASE_POSITION, 20, 50, FINISH_POSITION)

# A simplified single-player, no-dice-randomness Ludo environment for the toy DQN check.
class ToyLudoEnv(gym.Env):

    # Sets the fixed roll sequence, step cap, and Gym action/observation spaces.
    def __init__(
        self,
        roll_sequence: tuple[int, ...] = DEFAULT_ROLL_SEQUENCE,
        max_steps: int = DEFAULT_MAX_STEPS,
        start_positions: tuple[int, int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.roll_sequence = roll_sequence
        self.max_steps = max_steps
        self.start_positions = start_positions or _default_start_positions()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)

        self.board: BoardState | None = None
        self._roll_index = 0
        self._step_count = 0
        self._pending_roll: int | None = None
        self._pending_legal_tokens: tuple[int, ...] | None = None

    # Resets the board to the fixed starting layout and rolls the first scripted decision.
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board = BoardState()
        for token_id, position in enumerate(self.start_positions):
            self.board.set(PLAYER_ID, token_id, position)
        self._roll_index = 0
        self._step_count = 0
        self._pending_roll = None
        self._pending_legal_tokens = None

        self._advance_to_decision()
        return self._build_observation(), self._build_info()

    # Applies the chosen token's move for the pending scripted roll, then advances.
    def step(self, action: int):
        if self._pending_legal_tokens is None:
            raise RuntimeError("step() called with no pending decision; call reset() first")
        if action not in self._pending_legal_tokens:
            raise ValueError(f"action {action} is not legal; legal actions were {self._pending_legal_tokens}")

        outcome = apply_move(self.board, PLAYER_ID, action, self._pending_roll)
        reward = compute_move_reward(outcome, self.board, PLAYER_ID)
        self._pending_roll = None
        self._pending_legal_tokens = None
        self._step_count += 1

        terminated = has_player_won(self.board, PLAYER_ID)
        truncated = False
        if terminated:
            reward += win_reward()
        elif self._step_count >= self.max_steps:
            truncated = True
            reward += truncation_reward()
        else:
            self._advance_to_decision()

        return self._build_observation(), reward, terminated, truncated, self._build_info()

    # Steps through the scripted roll sequence until one actually has a legal move.
    def _advance_to_decision(self, max_attempts: int = 50) -> None:
        for _ in range(max_attempts):
            roll = self.roll_sequence[self._roll_index % len(self.roll_sequence)]
            self._roll_index += 1
            legal_tokens = get_legal_tokens(self.board, PLAYER_ID, roll)
            if legal_tokens:
                self._pending_roll = roll
                self._pending_legal_tokens = legal_tokens
                return
        raise RuntimeError("no legal move found after many scripted rolls")

    # Builds the egocentric observation vector for the single player.
    def _build_observation(self) -> np.ndarray:
        return encode_observation(self.board, PLAYER_ID)

    # Builds the info dict contract described in docs/env_api.md.
    def _build_info(self) -> dict:
        if self._pending_legal_tokens is not None:
            action_mask = get_action_mask(self.board, PLAYER_ID, self._pending_roll)
            dice_roll = self._pending_roll
        else:
            action_mask = np.zeros(4, dtype=bool)
            dice_roll = None
        return {"action_mask": action_mask, "dice_roll": dice_roll, "current_player": PLAYER_ID}
