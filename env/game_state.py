# Owns whole-game turn progression: whose turn it is, turn counting, and episode end conditions.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from env.board import NUM_PLAYERS, BoardState
from env.turns import ChooseActionFn, TurnResult, play_turn

DEFAULT_MAX_TURNS = 1000

# Tracks the board, current player, turn count, and terminated/truncated status for one game.
@dataclass
class GameState:
    board: BoardState = field(default_factory=BoardState)
    current_player: int = 0
    turn_count: int = 0
    max_turns: int = DEFAULT_MAX_TURNS
    terminated: bool = False
    truncated: bool = False
    winner: int | None = None

    # Resolves the current player's full turn and advances play to the next player.
    def step_turn(self, choose_action_fn: ChooseActionFn, rng: np.random.Generator) -> TurnResult:
        result = play_turn(self.board, self.current_player, choose_action_fn, rng)
        self.turn_count += 1
        if result.winner:
            self.terminated = True
            self.winner = self.current_player
        elif self.turn_count >= self.max_turns:
            self.truncated = True
        self.current_player = (self.current_player + 1) % NUM_PLAYERS
        return result
