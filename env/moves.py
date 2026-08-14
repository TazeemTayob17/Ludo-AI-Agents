# Applies a chosen move to the board and resolves its immediate consequences.

from __future__ import annotations

from dataclasses import dataclass

from env.board import (
    BASE_POSITION,
    BoardState,
    is_base,
    is_home,
    is_on_main_track,
    is_safe_square,
    relative_to_global,
)

# Describes what happened when a single token was moved.
@dataclass
class MoveOutcome:
    token_id: int
    exited_base: bool
    captured_player: int | None
    captured_token: int | None
    reached_home: bool

# Moves one token by the given roll, resolving any capture, and returns what happened.
def apply_move(board: BoardState, player_id: int, token_id: int, roll: int) -> MoveOutcome:
    position = board.get(player_id, token_id)
    exited_base = is_base(position)
    new_position = 0 if exited_base else position + roll
    board.set(player_id, token_id, new_position)

    captured_player = None
    captured_token = None
    if is_on_main_track(new_position):
        global_square = relative_to_global(player_id, new_position)
        if not is_safe_square(global_square):
            for owner, occupant_token_id in board.tokens_on_global_square(global_square):
                if owner != player_id:
                    board.set(owner, occupant_token_id, BASE_POSITION)
                    captured_player = owner
                    captured_token = occupant_token_id
                    break

    return MoveOutcome(
        token_id=token_id,
        exited_base=exited_base,
        captured_player=captured_player,
        captured_token=captured_token,
        reached_home=is_home(new_position),
    )
