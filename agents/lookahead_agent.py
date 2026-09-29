# One-roll lookahead baseline: simulates each legal move with the real rules engine and
# scores the resulting position, weighting each own token by its risk of being captured
# on the opponents' next roll. No learning; a hand-built reference for "how good is good play".

from __future__ import annotations

from env.board import NUM_PLAYERS, NUM_TOKENS_PER_PLAYER, BoardState
from env.moves import apply_move
from env.threats import build_occupancy, capturing_rolls


# Scores a resulting position for player_id: own progress minus opponents' average progress,
# minus each own token's expected progress lost to a capture on the opponents' next roll.
def _position_value(board: BoardState, player_id: int) -> float:
    my_progress = sum(board.get(player_id, token_id) + 1 for token_id in range(NUM_TOKENS_PER_PLAYER))
    opponent_progress = sum(
        board.get(opponent, token_id) + 1
        for opponent in range(NUM_PLAYERS)
        for token_id in range(NUM_TOKENS_PER_PLAYER)
        if opponent != player_id
    ) / (NUM_PLAYERS - 1)

    occupancy = build_occupancy(board)
    expected_loss = 0.0
    for token_id in range(NUM_TOKENS_PER_PLAYER):
        square = board.global_square(player_id, token_id)
        if square is None:
            continue
        rolls = capturing_rolls(board, player_id, square, occupancy)
        expected_loss += (len(rolls) / 6) * (board.get(player_id, token_id) + 1)

    return my_progress - opponent_progress - expected_loss


# Picks the legal move whose resulting position scores highest, breaking ties by the most-advanced token.
class LookaheadAgent:

    # Matches the ChooseActionFn signature so this can be used as a policy directly.
    def __call__(self, board: BoardState, player_id: int, roll: int, legal_tokens: tuple[int, ...]) -> int:
        def score(token_id: int) -> tuple[float, int]:
            after = board.copy()
            apply_move(after, player_id, token_id, roll)
            return (_position_value(after, player_id), board.get(player_id, token_id))

        return max(legal_tokens, key=score)
