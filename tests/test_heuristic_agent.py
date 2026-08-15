# Tests for agents/heuristic_agent.py.

from agents.heuristic_agent import HeuristicAgent
from env.board import BoardState

agent = HeuristicAgent()


# Checks a capturing move is chosen over a merely advancing one.
def test_prefers_capture_over_advance():
    board = BoardState()
    board.set(0, 0, 10)  # would capture if moved
    board.set(0, 1, 20)  # would only advance
    board.set(2, 0, 40)  # player 2 relative 40 -> global 14, landed on by token 0's roll of 4
    action = agent(board, player_id=0, roll=4, legal_tokens=(0, 1))
    assert action == 0


# Checks exiting Base is chosen over a merely advancing move, when no capture is available.
def test_prefers_exit_base_over_advance():
    board = BoardState()
    board.set(0, 1, 20)  # would only advance
    action = agent(board, player_id=0, roll=6, legal_tokens=(0, 1))
    assert action == 0  # token 0 is in Base; roll of 6 makes exiting legal


# Checks reaching Home is chosen over a merely advancing move.
def test_prefers_enter_home_over_advance():
    board = BoardState()
    board.set(0, 0, 55)  # would reach Home exactly
    board.set(0, 1, 20)  # would only advance
    action = agent(board, player_id=0, roll=2, legal_tokens=(0, 1))
    assert action == 0


# Checks forming/joining a blockade is chosen over a merely advancing move.
def test_prefers_blockade_over_advance():
    board = BoardState()
    board.set(0, 0, 10)
    board.set(0, 2, 6)  # moving with roll=4 lands on token 0's square -> blockade
    board.set(0, 1, 20)  # would only advance
    action = agent(board, player_id=0, roll=4, legal_tokens=(1, 2))
    assert action == 2


# Checks a capture outranks a same-roll blockade opportunity.
def test_capture_outranks_blockade():
    board = BoardState()
    board.set(0, 0, 6)  # would join a blockade with token 2 if moved
    board.set(0, 1, 10)  # would capture if moved
    board.set(0, 2, 10)  # sits at the blockade-forming square for token 0
    board.set(2, 0, 40)  # player 2 relative 40 -> global 14, landed on by token 1's roll of 4
    action = agent(board, player_id=0, roll=4, legal_tokens=(0, 1))
    assert action == 1


# Checks ties within the same category are broken by the most-advanced token.
def test_ties_broken_by_most_advanced_token():
    board = BoardState()
    board.set(0, 0, 10)  # both only advance with this roll
    board.set(0, 1, 20)
    action = agent(board, player_id=0, roll=3, legal_tokens=(0, 1))
    assert action == 1  # token 1 (position 20) is further along than token 0 (position 10)
