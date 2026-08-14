# Resolves one player's full turn, including bonus rolls and the three-sixes bust rule.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from env.board import FINISH_POSITION, NUM_TOKENS_PER_PLAYER, BoardState
from env.dice import roll_die
from env.legal_moves import get_legal_tokens
from env.moves import MoveOutcome, apply_move

DEFAULT_MAX_BONUS_ROLLS = 100

ChooseActionFn = Callable[[BoardState, int, int, tuple[int, ...]], int]

# Records everything that happened during one player's turn.
@dataclass
class TurnResult:
    player_id: int
    rolls: list[int]
    moves: list[MoveOutcome]
    busted: bool
    winner: bool

# Checks whether a player has moved all 4 of their tokens Home.
def has_player_won(board: BoardState, player_id: int) -> bool:
    return all(board.get(player_id, token_id) == FINISH_POSITION for token_id in range(NUM_TOKENS_PER_PLAYER))

# Resolves a player's full turn: rolls, decisions, bonus rolls, and the three-sixes bust.
def play_turn(
    board: BoardState,
    player_id: int,
    choose_action_fn: ChooseActionFn,
    rng: np.random.Generator,
    max_bonus_rolls: int = DEFAULT_MAX_BONUS_ROLLS,
) -> TurnResult:
    turn_start_positions = board.positions.copy()
    consecutive_sixes = 0
    rolls: list[int] = []
    moves: list[MoveOutcome] = []

    for _ in range(max_bonus_rolls):
        roll = roll_die(rng)
        rolls.append(roll)
        consecutive_sixes = consecutive_sixes + 1 if roll == 6 else 0

        if consecutive_sixes == 3:
            board.positions = turn_start_positions.copy()
            return TurnResult(player_id, rolls, [], busted=True, winner=False)

        bonus = roll == 6
        legal_tokens = get_legal_tokens(board, player_id, roll)
        if legal_tokens:
            token_id = choose_action_fn(board, player_id, roll, legal_tokens)
            if token_id not in legal_tokens:
                raise ValueError(f"chosen token {token_id} is not among legal tokens {legal_tokens}")
            outcome = apply_move(board, player_id, token_id, roll)
            moves.append(outcome)
            if outcome.captured_player is not None or outcome.reached_home:
                bonus = True

        if has_player_won(board, player_id):
            return TurnResult(player_id, rolls, moves, busted=False, winner=True)
        if not bonus:
            return TurnResult(player_id, rolls, moves, busted=False, winner=False)

    return TurnResult(player_id, rolls, moves, busted=False, winner=False)
