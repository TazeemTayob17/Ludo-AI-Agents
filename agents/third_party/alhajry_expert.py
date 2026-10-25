# Re-implementation of the Alhajry et al. (2012) rule-based expert, read from github.com/Nitinrajgor07/ludo-deep-reinforcement-learning at commit 13edaa1.

from __future__ import annotations

from env.board import (
    HOME_COLUMN_START,
    MAIN_TRACK_LENGTH,
    NUM_PLAYERS,
    NUM_TOKENS_PER_PLAYER,
    BoardState,
    is_base,
    is_on_main_track,
    is_safe_square,
    relative_to_global,
)

THREAT_RANGE = 6


# Checks whether a token on the shared track sits on an unsafe square with an opponent token 1-6 squares behind it.
# The source tests a single opponent (2 players); here every one of the three opponents counts.
def _is_vulnerable(board: BoardState, player_id: int, token_id: int) -> bool:
    position = board.get(player_id, token_id)
    if not is_on_main_track(position):
        return False
    square = relative_to_global(player_id, position)
    if is_safe_square(square):
        return False
    for opponent in range(NUM_PLAYERS):
        if opponent == player_id:
            continue
        for other in range(NUM_TOKENS_PER_PLAYER):
            other_square = board.global_square(opponent, other)
            if other_square is not None and 1 <= (square - other_square) % MAIN_TRACK_LENGTH <= THREAT_RANGE:
                return True
    return False


# Checks whether moving the token lands it on a safe square or in the home column while it is currently vulnerable.
def _moves_to_safety(board: BoardState, player_id: int, token_id: int, roll: int) -> bool:
    if not _is_vulnerable(board, player_id, token_id):
        return False
    new_position = board.get(player_id, token_id) + roll
    if new_position >= HOME_COLUMN_START:
        return True
    return is_safe_square(relative_to_global(player_id, new_position))


# Checks whether moving the token lands on an unsafe square holding an opponent token.
def _captures(board: BoardState, player_id: int, token_id: int, roll: int) -> bool:
    position = board.get(player_id, token_id)
    new_position = 0 if is_base(position) else position + roll
    if not is_on_main_track(new_position):
        return False
    destination = relative_to_global(player_id, new_position)
    if is_safe_square(destination):
        return False
    return any(owner != player_id for owner, _ in board.tokens_on_global_square(destination))


# Picks by the four-level priority list: release on a 6, move a vulnerable token to safety, capture, then advance the lead token.
class AlhajryExpert:

    # Matches the ChooseActionFn signature so this can be used as a policy directly.
    def __call__(self, board: BoardState, player_id: int, roll: int, legal_tokens: tuple[int, ...]) -> int:
        if len(legal_tokens) == 1:
            return legal_tokens[0]

        if roll == 6:
            releases = [t for t in legal_tokens if is_base(board.get(player_id, t))]
            if releases:
                return releases[0]

        defensive = [t for t in legal_tokens if _moves_to_safety(board, player_id, t, roll)]
        if defensive:
            return max(defensive, key=lambda t: board.get(player_id, t))

        capturing = [t for t in legal_tokens if _captures(board, player_id, t, roll)]
        if capturing:
            return max(capturing, key=lambda t: board.get(player_id, t))

        return max(legal_tokens, key=lambda t: board.get(player_id, t))
