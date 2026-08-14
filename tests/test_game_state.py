# Tests for env/game_state.py's whole-game turn progression.

from env.board import FINISH_POSITION, NUM_PLAYERS
from env.game_state import GameState


# A scripted policy that always picks the first legal token offered.
def _first_legal(board, player_id, roll, legal_tokens):
    return legal_tokens[0]


# A fixed sequence of rolls fed in place of real dice, cycling if exhausted.
class _CyclingRng:
    def __init__(self, rolls):
        self._rolls = list(rolls)
        self._index = 0

    def integers(self, low, high):
        value = self._rolls[self._index % len(self._rolls)]
        self._index += 1
        return value


# Checks the current player rotates in fixed order after each turn.
def test_current_player_rotates_in_order():
    game = GameState()
    rng = _CyclingRng([3])  # a non-six with no legal move for anyone at the start
    seen = []
    for _ in range(NUM_PLAYERS * 2):
        seen.append(game.current_player)
        game.step_turn(_first_legal, rng)
    assert seen == [0, 1, 2, 3, 0, 1, 2, 3]


# Checks turn_count increments once per step_turn call.
def test_turn_count_increments():
    game = GameState()
    rng = _CyclingRng([3])
    game.step_turn(_first_legal, rng)
    game.step_turn(_first_legal, rng)
    assert game.turn_count == 2


# Checks the game is marked terminated, not truncated, when a player wins.
def test_terminated_when_a_player_wins():
    game = GameState()
    game.board.set(0, 0, FINISH_POSITION)
    game.board.set(0, 1, FINISH_POSITION)
    game.board.set(0, 2, FINISH_POSITION)
    game.board.set(0, 3, 55)
    rng = _CyclingRng([2])
    game.step_turn(_first_legal, rng)
    assert game.terminated is True
    assert game.truncated is False
    assert game.winner == 0


# Checks the game is marked truncated, not terminated, once max_turns is reached without a winner.
def test_truncated_when_max_turns_reached_without_a_winner():
    game = GameState(max_turns=3)
    rng = _CyclingRng([3])  # no legal moves for anyone, no winner possible
    for _ in range(3):
        game.step_turn(_first_legal, rng)
    assert game.truncated is True
    assert game.terminated is False
