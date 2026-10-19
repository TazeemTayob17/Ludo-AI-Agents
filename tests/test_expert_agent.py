# Tests for agents/expert_agent.py.

from agents.expert_agent import ExpertAgent
from env.board import BoardState
from env.legal_moves import get_legal_tokens

agent = ExpertAgent()


# Checks a token is released from Base on a 6 even when another token could advance.
def test_releases_from_base_first():
    board = BoardState()
    board.set(0, 1, 20)
    assert agent(board, player_id=0, roll=6, legal_tokens=(0, 1)) == 0


# Checks a threatened token that can reach safety is moved before a capture is taken.
def test_defends_before_capturing():
    board = BoardState()
    board.set(0, 0, 10)  # threatened by player 2's token; roll 3 lands on safe global square 13
    board.set(0, 1, 20)  # roll 3 captures player 1's token on global 23
    board.set(2, 0, 30)  # global 4, reaches global 10 with a 6
    board.set(1, 0, 10)  # global 23
    assert agent(board, player_id=0, roll=3, legal_tokens=(0, 1)) == 0


# Checks a capture is taken over a plain advance when nothing needs defending.
def test_captures_over_advancing():
    board = BoardState()
    board.set(0, 0, 10)  # would capture
    board.set(0, 1, 30)  # would only advance, and is further ahead than token 0
    board.set(2, 0, 40)  # global 14, out of reach of token 1
    assert agent(board, player_id=0, roll=4, legal_tokens=(0, 1)) == 0


# Checks that with nothing special the most advanced token moves.
def test_advances_lead_token():
    board = BoardState()
    board.set(0, 0, 5)
    board.set(0, 1, 30)
    assert agent(board, player_id=0, roll=2, legal_tokens=(0, 1)) == 1


# Checks the chosen token is always among the legal ones across many random positions.
def test_always_returns_a_legal_token():
    import numpy as np

    rng = np.random.default_rng(0)
    for _ in range(300):
        board = BoardState()
        for player in range(4):
            for token in range(4):
                board.set(player, token, int(rng.integers(-1, 58)))
        roll = int(rng.integers(1, 7))
        legal = get_legal_tokens(board, 0, roll)
        if legal:
            assert agent(board, 0, roll, legal) in legal
