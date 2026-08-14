# Legal-move generation for the Ludo rules engine.

from __future__ import annotations

import numpy as np

from env.board import (
    FINISH_POSITION,
    MAIN_TRACK_RELATIVE_END,
    NUM_TOKENS_PER_PLAYER,
    BoardState,
    is_base,
    is_home,
    is_in_home_column,
    is_on_main_track,
    relative_to_global,
)

# Checks whether any opponent has a blockade (2+ tokens) on a given global square.
def _opponent_blockade_at(board: BoardState, player_id: int, global_square: int) -> bool:
    counts: dict[int, int] = {}
    for owner, _token_id in board.tokens_on_global_square(global_square):
        if owner != player_id:
            counts[owner] = counts.get(owner, 0) + 1
    return any(count >= 2 for count in counts.values())

# Checks whether any main-track square between two relative positions is opponent-blockaded.
def _path_blocked(board: BoardState, player_id: int, start_relative: int, end_relative: int) -> bool:
    for relative_position in range(start_relative, end_relative + 1):
        global_square = relative_to_global(player_id, relative_position)
        if _opponent_blockade_at(board, player_id, global_square):
            return True
    return False

# Checks whether another of the player's own tokens already sits on a home-column square.
def _own_token_at_home_column(board: BoardState, player_id: int, relative_position: int, exclude_token: int) -> bool:
    for token_id in range(NUM_TOKENS_PER_PLAYER):
        if token_id != exclude_token and board.get(player_id, token_id) == relative_position:
            return True
    return False

# Returns the token ids the player may legally move for a given roll.
def get_legal_tokens(board: BoardState, player_id: int, roll: int) -> tuple[int, ...]:
    legal: list[int] = []
    for token_id in range(NUM_TOKENS_PER_PLAYER):
        position = board.get(player_id, token_id)
        if is_home(position):
            continue
        if is_base(position):
            if roll != 6:
                continue
            entry_square = relative_to_global(player_id, 0)
            if _opponent_blockade_at(board, player_id, entry_square):
                continue
            legal.append(token_id)
            continue

        new_position = position + roll
        if new_position > FINISH_POSITION:
            continue
        if is_on_main_track(position):
            main_track_limit = min(new_position, MAIN_TRACK_RELATIVE_END)
            if _path_blocked(board, player_id, position + 1, main_track_limit):
                continue
        if new_position != FINISH_POSITION and is_in_home_column(new_position):
            if _own_token_at_home_column(board, player_id, new_position, exclude_token=token_id):
                continue
        legal.append(token_id)
    return tuple(legal)

# Returns a boolean mask over token ids 0-3 marking which tokens may legally move.
def get_action_mask(board: BoardState, player_id: int, roll: int) -> np.ndarray:
    mask = np.zeros(NUM_TOKENS_PER_PLAYER, dtype=bool)
    for token_id in get_legal_tokens(board, player_id, roll):
        mask[token_id] = True
    return mask
