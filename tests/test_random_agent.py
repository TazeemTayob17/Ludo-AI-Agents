# Tests for agents/random_agent.py.

import numpy as np

from agents.random_agent import RandomAgent
from env.board import BoardState


# Checks the agent only ever returns a token from the legal set.
def test_random_agent_always_returns_a_legal_token():
    agent = RandomAgent(rng=np.random.default_rng(0))
    board = BoardState()
    legal_tokens = (0, 2)
    for _ in range(200):
        assert agent(board, 0, 6, legal_tokens) in legal_tokens


# Checks the choice distribution is roughly uniform over the legal tokens.
def test_random_agent_distribution_is_roughly_uniform():
    agent = RandomAgent(rng=np.random.default_rng(1))
    board = BoardState()
    legal_tokens = (0, 1, 2, 3)
    counts = {token_id: 0 for token_id in legal_tokens}
    num_trials = 40000
    for _ in range(num_trials):
        counts[agent(board, 0, 6, legal_tokens)] += 1
    expected = num_trials / len(legal_tokens)
    for count in counts.values():
        assert abs(count - expected) / expected < 0.05


# Checks the same seed produces the same sequence of choices.
def test_random_agent_is_deterministic_given_a_seed():
    board = BoardState()
    legal_tokens = (0, 1, 3)
    agent_a = RandomAgent(rng=np.random.default_rng(7))
    agent_b = RandomAgent(rng=np.random.default_rng(7))
    choices_a = [agent_a(board, 0, 6, legal_tokens) for _ in range(30)]
    choices_b = [agent_b(board, 0, 6, legal_tokens) for _ in range(30)]
    assert choices_a == choices_b
