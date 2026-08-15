# The Heuristic baseline agent: a fixed priority list, no learning.

from __future__ import annotations

from env.board import (
    FINISH_POSITION,
    BoardState,
    is_base,
    is_on_main_track,
    is_safe_square,
    relative_to_global,
)

_PRIORITY = {"capture": 0, "exit_base": 1, "enter_home": 2, "blockade": 3, "advance": 4}

# Classifies what a legal move would do, per the priority list: capture > exit base >
# enter home > form/join a blockade > advance.
def _classify_token(board: BoardState, player_id: int, token_id: int, roll: int) -> str:
    position = board.get(player_id, token_id)
    if is_base(position):
        return "exit_base"

    new_position = position + roll
    if new_position == FINISH_POSITION:
        return "enter_home"

    if is_on_main_track(new_position):
        destination = relative_to_global(player_id, new_position)
        occupants = board.tokens_on_global_square(destination)
        if not is_safe_square(destination) and any(owner != player_id for owner, _ in occupants):
            return "capture"
        if any(owner == player_id for owner, _ in occupants):
            return "blockade"

    return "advance"


# Picks the highest-priority legal move, breaking ties by the most-advanced token.
class HeuristicAgent:

    # Matches the ChooseActionFn signature so this can be used as a policy directly.
    def __call__(self, board: BoardState, player_id: int, roll: int, legal_tokens: tuple[int, ...]) -> int:
        def sort_key(token_id: int) -> tuple[int, int]:
            category = _classify_token(board, player_id, token_id, roll)
            position = board.get(player_id, token_id)
            return (_PRIORITY[category], -position)

        return min(legal_tokens, key=sort_key)
