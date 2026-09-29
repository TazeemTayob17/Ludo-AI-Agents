# Tests for agents/lookahead_agent.py.

from agents.lookahead_agent import LookaheadAgent
from env.board import BoardState

agent = LookaheadAgent()


# Checks a capturing move is chosen over a merely advancing one (it wins back the opponent's progress).
def test_prefers_capture_over_advance():
    board = BoardState()
    board.set(0, 0, 10)  # would capture if moved
    board.set(0, 1, 20)  # would only advance
    board.set(2, 0, 40)  # player 2 relative 40 -> global 14, landed on by token 0's roll of 4
    action = agent(board, player_id=0, roll=4, legal_tokens=(0, 1))
    assert action == 0


# Checks reaching Home is chosen over a merely advancing move.
def test_prefers_enter_home_over_advance():
    board = BoardState()
    board.set(0, 0, 55)  # would reach Home exactly
    board.set(0, 1, 20)  # would only advance
    action = agent(board, player_id=0, roll=2, legal_tokens=(0, 1))
    assert action == 0


# Checks moving a threatened token to safety is chosen over moving an unthreatened token into an
# opponent's reach, leaving the first token exposed.
def test_prefers_escaping_threat_over_moving_into_danger():
    board = BoardState()
    board.set(0, 0, 10)  # threatened token; roll of 3 lands it on a safe square (global 13)
    board.set(0, 1, 16)  # untouched token; roll of 3 would land it in a different opponent's reach (global 19)
    board.set(1, 0, 5)  # would capture on global 19 with roll 1, if token 1 moved there
    board.set(2, 0, 30)  # would capture on global 10 with roll 6, if token 0 stayed there
    action = agent(board, player_id=0, roll=3, legal_tokens=(0, 1))
    assert action == 0


# Checks the only legal move is returned even if not obviously good.
def test_returns_the_only_legal_move():
    board = BoardState()
    board.set(0, 0, 20)
    action = agent(board, player_id=0, roll=3, legal_tokens=(0,))
    assert action == 0
