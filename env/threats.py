# Who-can-capture-whom on the very next roll, following the real movement rules.

from __future__ import annotations

from env.board import (
    MAIN_TRACK_LENGTH,
    MAIN_TRACK_RELATIVE_END,
    NUM_PLAYERS,
    NUM_TOKENS_PER_PLAYER,
    BoardState,
    is_on_main_track,
    is_safe_square,
    relative_to_global,
)

MAX_ROLL = 6

Occupancy = dict[int, dict[int, int]]


# Counts each player's tokens on every occupied main-track square, so blockade checks are cheap.
def build_occupancy(board: BoardState) -> Occupancy:
    occupancy: Occupancy = {}
    for player_id in range(NUM_PLAYERS):
        for token_id in range(NUM_TOKENS_PER_PLAYER):
            square = board.global_square(player_id, token_id)
            if square is not None:
                owners = occupancy.setdefault(square, {})
                owners[player_id] = owners.get(player_id, 0) + 1
    return occupancy


# Checks whether a blockade (2+ tokens of one player other than mover) sits on a square.
def _blocked_for(occupancy: Occupancy, mover: int, square: int) -> bool:
    return any(owner != mover and count >= 2 for owner, count in occupancy.get(square, {}).items())


# Returns the smallest roll that lets any opponent legally land on this square next turn, or None.
def capture_distance(board: BoardState, player_id: int, global_square: int, occupancy: Occupancy | None = None) -> int | None:
    if is_safe_square(global_square):
        return None
    occupancy = occupancy if occupancy is not None else build_occupancy(board)
    best = None
    for opponent in range(NUM_PLAYERS):
        if opponent == player_id:
            continue
        for token_id in range(NUM_TOKENS_PER_PLAYER):
            position = board.get(opponent, token_id)
            if not is_on_main_track(position):
                continue
            for roll in range(1, MAX_ROLL + 1):
                if position + roll > MAIN_TRACK_RELATIVE_END:
                    break
                square = relative_to_global(opponent, position + roll)
                if _blocked_for(occupancy, opponent, square):
                    break
                if square == global_square:
                    if best is None or roll < best:
                        best = roll
                    break
    return best


# Checks whether one of player_id's tokens could be captured by some opponent's next roll.
def is_token_threatened(board: BoardState, player_id: int, token_id: int, occupancy: Occupancy | None = None) -> bool:
    square = board.global_square(player_id, token_id)
    if square is None:
        return False
    return capture_distance(board, player_id, square, occupancy) is not None


# Returns the smallest roll with which this token could legally capture an opponent next, or None.
def target_distance(board: BoardState, player_id: int, token_id: int, occupancy: Occupancy | None = None) -> int | None:
    position = board.get(player_id, token_id)
    if not is_on_main_track(position):
        return None
    occupancy = occupancy if occupancy is not None else build_occupancy(board)
    for roll in range(1, MAX_ROLL + 1):
        if position + roll > MAIN_TRACK_RELATIVE_END:
            return None
        square = relative_to_global(player_id, position + roll)
        if _blocked_for(occupancy, player_id, square):
            return None
        if not is_safe_square(square) and any(owner != player_id for owner in occupancy.get(square, {})):
            return roll
    return None


# Returns the distance to the nearest opponent behind this token whose route passes it, or None.
def nearest_opponent_behind(board: BoardState, player_id: int, token_id: int, max_distance: int) -> int | None:
    square = board.global_square(player_id, token_id)
    if square is None:
        return None
    best = None
    for opponent in range(NUM_PLAYERS):
        if opponent == player_id:
            continue
        for other_token in range(NUM_TOKENS_PER_PLAYER):
            position = board.get(opponent, other_token)
            if not is_on_main_track(position):
                continue
            distance = (square - relative_to_global(opponent, position)) % MAIN_TRACK_LENGTH
            if 1 <= distance <= max_distance and position + distance <= MAIN_TRACK_RELATIVE_END:
                if best is None or distance < best:
                    best = distance
    return best
