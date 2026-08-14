# Tests for env/turns.py's per-turn resolution, bonus rolls, and the three-sixes bust.

from env.board import BoardState, FINISH_POSITION
from env.turns import has_player_won, play_turn


# A scripted policy that always picks the first legal token offered.
def _first_legal(board, player_id, roll, legal_tokens):
    return legal_tokens[0]


# A fixed sequence of rolls fed in place of real dice, for deterministic turn tests.
class _ScriptedRng:
    def __init__(self, rolls):
        self._rolls = list(rolls)

    def integers(self, low, high):
        return self._rolls.pop(0)


# Checks a non-six roll with a move ends the turn with no bonus.
def test_turn_ends_after_a_single_non_six_roll():
    board = BoardState()
    board.set(0, 0, 10)
    result = play_turn(board, player_id=0, choose_action_fn=_first_legal, rng=_ScriptedRng([3]))
    assert result.rolls == [3]
    assert len(result.moves) == 1
    assert board.get(0, 0) == 13


# Checks rolling a six grants a bonus roll.
def test_six_grants_a_bonus_roll():
    board = BoardState()
    board.set(0, 0, 10)
    result = play_turn(board, player_id=0, choose_action_fn=_first_legal, rng=_ScriptedRng([6, 2]))
    assert result.rolls == [6, 2]
    assert board.get(0, 0) == 18  # advanced by 6, then by the bonus roll of 2
    assert len(result.moves) == 2


# Checks capturing an opponent grants a bonus roll.
def test_capture_grants_a_bonus_roll():
    board = BoardState()
    board.set(0, 0, 10)
    board.set(2, 0, 40)  # player 2 relative 40 -> global 14, landed on by a roll of 4
    result = play_turn(board, player_id=0, choose_action_fn=_first_legal, rng=_ScriptedRng([4, 3]))
    assert result.rolls == [4, 3]
    assert len(result.moves) == 2


# Checks reaching Home grants a bonus roll.
def test_reaching_home_grants_a_bonus_roll():
    board = BoardState()
    board.set(0, 0, 55)
    result = play_turn(board, player_id=0, choose_action_fn=_first_legal, rng=_ScriptedRng([2, 3]))
    assert result.rolls == [2, 3]
    assert board.get(0, 0) == FINISH_POSITION


# Checks three consecutive sixes forfeits the whole turn and reverts the board.
def test_three_consecutive_sixes_busts_the_turn():
    board = BoardState()
    board.set(0, 0, 10)
    original = board.positions.copy()
    result = play_turn(board, player_id=0, choose_action_fn=_first_legal, rng=_ScriptedRng([6, 6, 6]))
    assert result.busted is True
    assert result.moves == []
    assert (board.positions == original).all()


# Checks a roll with zero legal moves auto-skips without asking for a decision.
def test_no_legal_move_auto_skips_without_a_decision():
    board = BoardState()  # all tokens in Base; a roll of 3 has no legal move
    called = []

    def _tracking_policy(board, player_id, roll, legal_tokens):
        called.append(roll)
        return legal_tokens[0]

    result = play_turn(board, player_id=0, choose_action_fn=_tracking_policy, rng=_ScriptedRng([3]))
    assert result.rolls == [3]
    assert result.moves == []
    assert called == []


# Checks a roll of six with no legal move still grants a bonus roll.
def test_dead_six_still_grants_a_bonus_roll():
    board = BoardState()
    board.set(1, 0, FINISH_POSITION)
    board.set(1, 1, FINISH_POSITION)
    board.set(1, 2, FINISH_POSITION)
    board.set(1, 3, 52)  # would overshoot with a roll of 6 (52+6=58); no tokens in Base
    result = play_turn(board, player_id=1, choose_action_fn=_first_legal, rng=_ScriptedRng([6, 3]))
    assert result.rolls == [6, 3]
    assert len(result.moves) == 1
    assert board.get(1, 3) == 55


# Checks the turn ends immediately once the player wins.
def test_turn_ends_immediately_on_winning():
    board = BoardState()
    board.set(0, 0, FINISH_POSITION)
    board.set(0, 1, FINISH_POSITION)
    board.set(0, 2, FINISH_POSITION)
    board.set(0, 3, 55)
    result = play_turn(board, player_id=0, choose_action_fn=_first_legal, rng=_ScriptedRng([2]))
    assert result.winner is True
    assert has_player_won(board, 0)


# Checks the internal safety cap stops the loop even while bonus keeps being granted.
def test_max_bonus_rolls_safety_cap_stops_the_loop():
    board = BoardState()
    board.set(0, 0, 10)
    result = play_turn(
        board, player_id=0, choose_action_fn=_first_legal, rng=_ScriptedRng([6, 6]), max_bonus_rolls=2
    )
    assert result.rolls == [6, 6]
    assert result.busted is False
