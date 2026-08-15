# Gymnasium wrapper exposing the Ludo rules engine as a single-agent environment.

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.board import NUM_PLAYERS, NUM_TOKENS_PER_PLAYER
from env.dice import roll_die
from env.game_state import DEFAULT_MAX_TURNS, GameState
from env.legal_moves import get_action_mask, get_legal_tokens
from env.moves import apply_move
from env.rewards import captured_penalty, compute_move_reward, loss_reward, truncation_reward, win_reward
from env.state_encoding import OBSERVATION_SIZE, encode_observation
from env.turns import has_player_won

OpponentPolicy = Callable[[object, int, int, tuple[int, ...]], int]

# Single-agent Gymnasium environment: the agent plays one seat, opponents follow a fixed policy.
class LudoEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # Sets up the fixed agent seat, opponent policy, and Gym action/observation spaces.
    def __init__(
        self,
        opponent_policy: OpponentPolicy | None = None,
        agent_player_id: int = 0,
        max_turns: int = DEFAULT_MAX_TURNS,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.agent_player_id = agent_player_id
        self.opponent_policy = opponent_policy or self._default_random_opponent_policy
        self.max_turns_config = max_turns
        self._injected_rng = rng
        self._rng = None
        self.action_space = spaces.Discrete(NUM_TOKENS_PER_PLAYER)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)

        self.game: GameState | None = None
        self._turn_start_positions = None
        self._consecutive_sixes = 0
        self._pending_roll: int | None = None
        self._pending_legal_tokens: tuple[int, ...] | None = None
        self._pending_captured_penalty = 0.0

    # A uniformly random fallback opponent policy, used until real baseline agents exist.
    def _default_random_opponent_policy(self, board, player_id, roll, legal_tokens):
        index = self._rng.integers(0, len(legal_tokens))
        return legal_tokens[index]

    # Starts a new game and fast-forwards to the agent's first real decision.
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = self._injected_rng if self._injected_rng is not None else self.np_random

        self.game = GameState(max_turns=self.max_turns_config)
        self._turn_start_positions = None
        self._consecutive_sixes = 0
        self._pending_roll = None
        self._pending_legal_tokens = None
        self._pending_captured_penalty = 0.0

        self._run_until_agent_turn_or_done()
        return self._build_observation(), self._build_info()

    # Applies the agent's chosen token, then fast-forwards to its next decision or episode end.
    def step(self, action: int):
        if self.game is None or self._pending_legal_tokens is None:
            raise RuntimeError("step() called with no pending decision; call reset() first")
        if action not in self._pending_legal_tokens:
            raise ValueError(f"action {action} is not legal; legal actions were {self._pending_legal_tokens}")

        roll = self._pending_roll
        outcome = apply_move(self.game.board, self.agent_player_id, action, roll)
        reward = compute_move_reward(outcome, self.game.board, self.agent_player_id)
        reward += self._pending_captured_penalty
        self._pending_captured_penalty = 0.0
        self._pending_roll = None
        self._pending_legal_tokens = None

        if has_player_won(self.game.board, self.agent_player_id):
            self.game.terminated = True
            self.game.winner = self.agent_player_id
            self.game.turn_count += 1
            reward += win_reward()
        else:
            bonus = roll == 6 or outcome.captured_player is not None or outcome.reached_home
            if not bonus:
                self._end_agent_turn()
            if not (self.game.terminated or self.game.truncated):
                self._run_until_agent_turn_or_done()

            if self.game.terminated and self.game.winner != self.agent_player_id:
                reward += loss_reward()
            elif self.game.truncated:
                reward += truncation_reward()

            if self.game.terminated or self.game.truncated:
                reward += self._pending_captured_penalty
                self._pending_captured_penalty = 0.0

        return self._build_observation(), reward, self.game.terminated, self.game.truncated, self._build_info()

    # Prints and returns a simple text view of the board.
    def render(self) -> str:
        lines = [f"Turn {self.game.turn_count} - player {self.game.current_player}'s turn"]
        for player_id in range(NUM_PLAYERS):
            positions = [self.game.board.get(player_id, token_id) for token_id in range(NUM_TOKENS_PER_PLAYER)]
            marker = "*" if player_id == self.game.current_player else " "
            lines.append(f"{marker} player {player_id}: {positions}")
        if self.game.terminated:
            lines.append(f"GAME OVER - player {self.game.winner} wins")
        elif self.game.truncated:
            lines.append("GAME OVER - truncated (max turns reached)")
        text = "\n".join(lines)
        print(text)
        return text

    # Runs whole opponent turns and the agent's own auto-skipped rolls until a real
    # agent decision is needed or the episode has ended. A single flat loop, no
    # recursion, so it stays safe even when many turns in a row are auto-skipped.
    def _run_until_agent_turn_or_done(self):
        while not (self.game.terminated or self.game.truncated):
            if self.game.current_player != self.agent_player_id:
                result = self.game.step_turn(self.opponent_policy, self._rng)
                for move in result.moves:
                    if move.captured_player == self.agent_player_id:
                        self._pending_captured_penalty += captured_penalty()
                continue

            if self._turn_start_positions is None:
                self._turn_start_positions = self.game.board.positions.copy()
                self._consecutive_sixes = 0

            roll = roll_die(self._rng)
            self._consecutive_sixes = self._consecutive_sixes + 1 if roll == 6 else 0

            if self._consecutive_sixes == 3:
                self.game.board.positions = self._turn_start_positions.copy()
                self._end_agent_turn()
                continue

            legal_tokens = get_legal_tokens(self.game.board, self.agent_player_id, roll)
            if legal_tokens:
                self._pending_roll = roll
                self._pending_legal_tokens = legal_tokens
                return

            if roll == 6:
                continue

            self._end_agent_turn()

    # Ends the agent's current turn: updates the turn count/truncation, and rotates to the next player.
    def _end_agent_turn(self):
        self.game.turn_count += 1
        if self.game.turn_count >= self.game.max_turns:
            self.game.truncated = True
        self.game.current_player = (self.game.current_player + 1) % NUM_PLAYERS
        self._turn_start_positions = None

    # Builds the egocentric observation vector for the agent's current seat.
    def _build_observation(self) -> np.ndarray:
        return encode_observation(self.game.board, self.agent_player_id)

    # Builds the info dict contract described in docs/env_api.md.
    def _build_info(self) -> dict:
        if self._pending_legal_tokens is not None:
            action_mask = get_action_mask(self.game.board, self.agent_player_id, self._pending_roll)
            dice_roll = self._pending_roll
        else:
            action_mask = np.zeros(NUM_TOKENS_PER_PLAYER, dtype=bool)
            dice_roll = None
        info = {
            "action_mask": action_mask,
            "dice_roll": dice_roll,
            "current_player": self.game.current_player,
        }
        if self.game.terminated:
            info["winner"] = self.game.winner
        return info
