# Tests for agents/rollout_agent.py.

from agents.rollout_agent import RolloutAgent
from env.board import FINISH_POSITION, BoardState
from env.legal_moves import get_legal_tokens


# Checks the only legal move is returned without running any rollouts.
def test_returns_the_only_legal_move():
    board = BoardState()
    board.set(0, 0, 20)
    agent = RolloutAgent(num_rollouts=1)
    assert agent(board, player_id=0, roll=3, legal_tokens=(0,)) == 0


# Checks a move that completes the win scores exactly 1.0.
def test_winning_move_scores_one():
    board = BoardState()
    for token in (1, 2, 3):
        board.set(0, token, FINISH_POSITION)
    board.set(0, 0, 55)
    agent = RolloutAgent(num_rollouts=2)
    assert agent._win_fraction(board, 0, 0, 2, base_seed=1) == 1.0


# Checks the same seed gives the same choice, and the choice is legal.
def test_choice_is_legal_and_reproducible():
    board = BoardState()
    board.set(0, 0, 10)
    board.set(0, 1, 30)
    board.set(2, 0, 20)
    legal = get_legal_tokens(board, 0, 3)
    first = RolloutAgent(num_rollouts=3, seed=7)(board, 0, 3, legal)
    second = RolloutAgent(num_rollouts=3, seed=7)(board, 0, 3, legal)
    assert first in legal
    assert first == second


# Checks the rollout leaves the real board untouched.
def test_does_not_mutate_the_board():
    board = BoardState()
    board.set(0, 0, 10)
    board.set(0, 1, 30)
    before = board.copy()
    RolloutAgent(num_rollouts=2)(board, 0, 3, get_legal_tokens(board, 0, 3))
    assert board == before
