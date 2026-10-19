# The Expert baseline: release on a 6, defend a threatened token, capture, else advance the lead token. No learning.

from __future__ import annotations

from agents.heuristic_agent import _classify_token
from env.board import BoardState, is_base
from env.moves import apply_move
from env.threats import build_occupancy, capture_distance, is_token_threatened


# Checks whether moving this token leaves it somewhere no opponent can capture it on their next roll.
def _ends_unthreatened(board: BoardState, player_id: int, token_id: int, roll: int) -> bool:
    after = board.copy()
    apply_move(after, player_id, token_id, roll)
    square = after.global_square(player_id, token_id)
    return square is None or capture_distance(after, player_id, square, build_occupancy(after)) is None


# Picks by strict priority: release a token from Base, rescue a threatened token, capture, advance the lead token.
class ExpertAgent:

    # Matches the ChooseActionFn signature so this can be used as a policy directly.
    def __call__(self, board: BoardState, player_id: int, roll: int, legal_tokens: tuple[int, ...]) -> int:
        def lead(candidates: list[int]) -> int:
            return max(candidates, key=lambda token_id: board.get(player_id, token_id))

        releases = [t for t in legal_tokens if is_base(board.get(player_id, t))]
        if releases:
            return releases[0]

        occupancy = build_occupancy(board)
        rescues = [
            t
            for t in legal_tokens
            if is_token_threatened(board, player_id, t, occupancy) and _ends_unthreatened(board, player_id, t, roll)
        ]
        if rescues:
            return lead(rescues)

        captures = [t for t in legal_tokens if _classify_token(board, player_id, t, roll) == "capture"]
        if captures:
            return lead(captures)

        return lead(list(legal_tokens))
