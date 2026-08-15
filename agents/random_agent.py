# The Random baseline agent: samples uniformly from the legal action mask.

from __future__ import annotations

import numpy as np

from env.board import BoardState

# Picks uniformly at random among the legal tokens for this roll.
class RandomAgent:

    # Sets up the agent's own seeded random generator.
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng if rng is not None else np.random.default_rng()

    # Matches the ChooseActionFn signature so this can be used as a policy directly.
    def __call__(self, board: BoardState, player_id: int, roll: int, legal_tokens: tuple[int, ...]) -> int:
        index = self._rng.integers(0, len(legal_tokens))
        return legal_tokens[index]
