# Board state representation and coordinate transforms for the Ludo environment.

from __future__ import annotations

import numpy as np

NUM_PLAYERS = 4
NUM_TOKENS_PER_PLAYER = 4

MAIN_TRACK_LENGTH = 52
ENTRY_OFFSETS = (0, 13, 26, 39)

MAIN_TRACK_RELATIVE_END = 50
HOME_COLUMN_START = 51
HOME_COLUMN_END = 56
FINISH_POSITION = 57

BASE_POSITION = -1

SAFE_SQUARES = frozenset(
    offset
    for entry in ENTRY_OFFSETS
    for offset in (entry, (entry + 8) % MAIN_TRACK_LENGTH)
)

# Converts a token's player-relative track position to its global main-track square.
def relative_to_global(player_id: int, relative_position: int) -> int:
    if not (0 <= relative_position <= MAIN_TRACK_RELATIVE_END):
        raise ValueError(
            f"relative_position {relative_position} is not on the shared main track "
            f"(valid range 0-{MAIN_TRACK_RELATIVE_END}); Base/home-column/Home have no global square"
        )
    return (ENTRY_OFFSETS[player_id] + relative_position) % MAIN_TRACK_LENGTH


# Checks whether a global square is one of the 8 safe squares.
def is_safe_square(global_square: int) -> bool:
    return global_square in SAFE_SQUARES


# Checks whether a relative position means the token is still in Base.
def is_base(relative_position: int) -> bool:
    return relative_position == BASE_POSITION


# Checks whether a relative position is on the shared main track.
def is_on_main_track(relative_position: int) -> bool:
    return 0 <= relative_position <= MAIN_TRACK_RELATIVE_END


# Checks whether a relative position is in the private home column.
def is_in_home_column(relative_position: int) -> bool:
    return HOME_COLUMN_START <= relative_position <= HOME_COLUMN_END


# Checks whether a relative position means the token has finished.
def is_home(relative_position: int) -> bool:
    return relative_position == FINISH_POSITION


# Stores the positions of all 16 tokens and provides get/set/copy access to them.
class BoardState:

    # Initializes every token to Base.
    def __init__(self) -> None:
        self.positions = np.full(
            (NUM_PLAYERS, NUM_TOKENS_PER_PLAYER), BASE_POSITION, dtype=np.int8
        )

    # Resets every token back to Base.
    def reset(self) -> None:
        self.positions.fill(BASE_POSITION)

    # Returns a single token's relative position.
    def get(self, player_id: int, token_id: int) -> int:
        return int(self.positions[player_id, token_id])

    # Sets a single token's relative position.
    def set(self, player_id: int, token_id: int, relative_position: int) -> None:
        if not (relative_position == BASE_POSITION or 0 <= relative_position <= FINISH_POSITION):
            raise ValueError(f"invalid relative_position: {relative_position}")
        self.positions[player_id, token_id] = relative_position

    # Returns a token's global main-track square, or None if it isn't on the main track.
    def global_square(self, player_id: int, token_id: int) -> int | None:
        relative_position = self.get(player_id, token_id)
        if not is_on_main_track(relative_position):
            return None
        return relative_to_global(player_id, relative_position)

    # Returns every (player_id, token_id) currently occupying a given global square.
    def tokens_on_global_square(self, global_square: int) -> list[tuple[int, int]]:
        occupants = []
        for player_id in range(NUM_PLAYERS):
            for token_id in range(NUM_TOKENS_PER_PLAYER):
                if self.global_square(player_id, token_id) == global_square:
                    occupants.append((player_id, token_id))
        return occupants

    # Returns an independent copy of this board state.
    def copy(self) -> "BoardState":
        clone = BoardState()
        clone.positions = self.positions.copy()
        return clone

    # Checks whether two board states have identical token positions.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoardState):
            return NotImplemented
        return bool(np.array_equal(self.positions, other.positions))
