# Monte Carlo rollout player: scores each legal move by how often it wins when Heuristic players finish the game. Headroom estimate, no learning.

from __future__ import annotations

import numpy as np

from agents.heuristic_agent import HeuristicAgent
from env.board import NUM_PLAYERS, BoardState
from env.game_state import DEFAULT_MAX_TURNS, GameState
from env.moves import apply_move
from env.turns import ChooseActionFn, has_player_won


# Picks the legal move with the highest win fraction over simulated games played out by rollout_policy in every seat.
class RolloutAgent:

    # Stores the rollout count, the policy used in every seat of the simulations, and the seed for dice streams.
    def __init__(
        self,
        num_rollouts: int = 16,
        rollout_policy: ChooseActionFn | None = None,
        seed: int = 0,
        max_rollout_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        self.num_rollouts = num_rollouts
        self.rollout_policy = rollout_policy if rollout_policy is not None else HeuristicAgent()
        self.max_rollout_turns = max_rollout_turns
        self._rng = np.random.default_rng(seed)

    # Matches the ChooseActionFn signature so this can be used as a policy directly.
    def __call__(self, board: BoardState, player_id: int, roll: int, legal_tokens: tuple[int, ...]) -> int:
        if len(legal_tokens) == 1:
            return legal_tokens[0]
        base_seed = int(self._rng.integers(2**31 - 1))
        return max(
            legal_tokens,
            key=lambda token_id: (
                self._win_fraction(board, player_id, token_id, roll, base_seed),
                board.get(player_id, token_id),
            ),
        )

    # Applies the move on a copy and returns the fraction of rollouts player_id wins; every candidate reuses the same dice seeds.
    def _win_fraction(self, board: BoardState, player_id: int, token_id: int, roll: int, base_seed: int) -> float:
        after = board.copy()
        outcome = apply_move(after, player_id, token_id, roll)
        if has_player_won(after, player_id):
            return 1.0
        extra_roll = roll == 6 or outcome.captured_player is not None or outcome.reached_home
        next_player = player_id if extra_roll else (player_id + 1) % NUM_PLAYERS

        wins = 0
        for i in range(self.num_rollouts):
            game = GameState(board=after.copy(), current_player=next_player, max_turns=self.max_rollout_turns)
            rng = np.random.default_rng(base_seed + i)
            while not (game.terminated or game.truncated):
                game.step_turn(self.rollout_policy, rng)
            wins += game.winner == player_id
        return wins / self.num_rollouts
