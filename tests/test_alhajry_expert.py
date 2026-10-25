# Tests for agents/third_party/alhajry_expert.py: one test per priority level of the re-implemented source.

import numpy as np

from agents.third_party.alhajry_expert import AlhajryExpert
from env.board import BoardState
from env.legal_moves import get_legal_tokens

agent = AlhajryExpert()


# Checks a base token is released on a 6 even when a capture is available.
def test_releases_before_capturing():
    board = BoardState()
    board.set(0, 1, 10)  # roll 6 lands on global 16, which holds player 1's token
    board.set(1, 0, 3)  # global 16
    assert agent(board, player_id=0, roll=6, legal_tokens=(0, 1)) == 0


# Checks a vulnerable token moves to a safe square before a capture is taken.
def test_defends_before_capturing():
    board = BoardState()
    board.set(0, 0, 10)  # global 10, player 2's token on global 4 is 6 squares behind
    board.set(2, 0, 30)  # global 4
    board.set(0, 1, 20)  # roll 3 captures on global 23
    board.set(1, 0, 10)  # global 23
    assert agent(board, player_id=0, roll=3, legal_tokens=(0, 1)) == 0


# Checks a token already on a safe square is not treated as vulnerable, so the lead token advances instead.
def test_token_on_safe_square_is_not_defended():
    board = BoardState()
    board.set(0, 0, 8)  # global 8 is safe; roll 5 would reach safe global 13 if it were wrongly seen as vulnerable
    board.set(1, 0, 45)  # global 6, two squares behind token 0
    board.set(0, 1, 30)  # the lead token
    assert agent(board, player_id=0, roll=5, legal_tokens=(0, 1)) == 1


# Checks a threat from any of the three opponents makes a token vulnerable, not only the next seat.
def test_threat_from_any_opponent_counts():
    board = BoardState()
    board.set(0, 0, 10)  # global 10
    board.set(3, 0, 20)  # global 7, three squares behind token 0
    board.set(0, 1, 30)  # the lead token
    assert agent(board, player_id=0, roll=3, legal_tokens=(0, 1)) == 0  # token 0 reaches safe global 13


# Checks entering the home column counts as reaching safety.
def test_home_column_counts_as_safe():
    board = BoardState()
    board.set(0, 0, 48)  # global 48, not safe
    board.set(1, 0, 33)  # global 46, two squares behind token 0
    board.set(0, 1, 52)  # already in the home column and further along, so it wins a plain advance
    assert agent(board, player_id=0, roll=4, legal_tokens=(0, 1)) == 0  # 48 + 4 = 52 enters the home column


# Checks that among several defensive moves the most progressed token is chosen.
def test_most_progressed_defensive_move_wins():
    board = BoardState()
    board.set(0, 0, 10)  # global 10, vulnerable, roll 3 reaches safe 13
    board.set(0, 1, 18)  # global 18, vulnerable, roll 3 reaches 21 which is safe
    board.set(2, 0, 30)  # global 4: 6 behind token 0
    board.set(2, 1, 40)  # global 14: 4 behind token 1
    assert agent(board, player_id=0, roll=3, legal_tokens=(0, 1)) == 1


# Checks a capture is taken over a plain advance when nothing needs defending.
def test_captures_over_advancing():
    board = BoardState()
    board.set(0, 0, 10)  # roll 4 captures on global 14
    board.set(0, 1, 30)  # would only advance and is further along
    board.set(1, 0, 1)  # global 14
    assert agent(board, player_id=0, roll=4, legal_tokens=(0, 1)) == 0


# Checks landing on a safe square holding an opponent is not a capture, so the lead token advances.
def test_no_capture_on_safe_square():
    board = BoardState()
    board.set(0, 0, 9)  # roll 4 lands on global 13, which is safe
    board.set(0, 1, 30)
    board.set(1, 0, 0)  # global 13
    assert agent(board, player_id=0, roll=4, legal_tokens=(0, 1)) == 1


# Checks that with nothing special the most advanced token moves.
def test_advances_lead_token():
    board = BoardState()
    board.set(0, 0, 5)
    board.set(0, 1, 30)
    assert agent(board, player_id=0, roll=2, legal_tokens=(0, 1)) == 1


# Checks a single legal move is returned as is.
def test_single_legal_move_is_returned():
    board = BoardState()
    board.set(0, 2, 12)
    assert agent(board, player_id=0, roll=3, legal_tokens=(2,)) == 2


# Checks the chosen token is always among the legal ones across many random positions.
def test_always_returns_a_legal_token():
    rng = np.random.default_rng(0)
    for _ in range(300):
        board = BoardState()
        for player in range(4):
            for token in range(4):
                board.set(player, token, int(rng.choice([-1, *range(0, 58)])))
        player_id, roll = int(rng.integers(0, 4)), int(rng.integers(1, 7))
        legal = get_legal_tokens(board, player_id, roll)
        if legal:
            assert agent(board, player_id=player_id, roll=roll, legal_tokens=legal) in legal
