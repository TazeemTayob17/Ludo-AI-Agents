# Tests for agents/tabular_q_agent.py.

import numpy as np
import pytest

from agents.tabular_q_agent import TabularQAgent, discretize_state
from env.board import BoardState


# Checks each progress zone shows up as a distinct descriptor value.
def test_discretize_state_reflects_progress_zone():
    board = BoardState()  # token 0 in Base
    base_descriptor = discretize_state(board, player_id=0)[0]

    board.set(0, 0, 5)  # early main track
    early_descriptor = discretize_state(board, player_id=0)[0]

    board.set(0, 0, 25)  # mid main track
    mid_descriptor = discretize_state(board, player_id=0)[0]

    board.set(0, 0, 45)  # late main track
    late_descriptor = discretize_state(board, player_id=0)[0]

    board.set(0, 0, 53)  # home column
    home_stretch_descriptor = discretize_state(board, player_id=0)[0]

    board.set(0, 0, 57)  # finished
    home_descriptor = discretize_state(board, player_id=0)[0]

    descriptors = [
        base_descriptor,
        early_descriptor,
        mid_descriptor,
        late_descriptor,
        home_stretch_descriptor,
        home_descriptor,
    ]
    assert len(set(descriptors)) == 6  # all distinct


# Checks a token on an unsafe square within 6 of an opponent is flagged threatened.
def test_discretize_state_flags_a_threatened_token():
    board = BoardState()
    board.set(0, 0, 20)  # global square 20, unsafe
    board.set(1, 0, 46)  # player 1 relative 46 -> global (13+46)%52 = 7; 20-7=13 away, not threatened yet

    safe_descriptor = discretize_state(board, player_id=0)[0]

    board.set(1, 0, 45)  # relative 45 -> global 6; 20-6=14 away, still not close enough
    board.set(2, 0, 40)  # player 2 relative 40 -> global (26+40)%52 = 14; 20-14=6, within range

    threatened_descriptor = discretize_state(board, player_id=0)[0]

    assert safe_descriptor != threatened_descriptor
    assert threatened_descriptor == safe_descriptor + 1  # same zone, threatened bit flips on


# Checks a token on a safe square is never flagged threatened, regardless of opponents nearby.
def test_discretize_state_safe_square_is_never_threatened():
    board = BoardState()
    board.set(0, 0, 8)  # global square 8, a star safe square
    board.set(1, 0, 45)  # player 1 relative 45 -> global 6; 2 squares behind, would threaten if unsafe
    descriptor = discretize_state(board, player_id=0)[0]
    assert descriptor % 2 == 0  # threatened bit is off


# Checks a freshly seen state gets a zero-initialized Q-value row.
def test_unseen_state_defaults_to_zero_q_values():
    agent = TabularQAgent()
    values = agent.q_values((0, 0, 0, 0))
    assert values.tolist() == [0.0, 0.0, 0.0, 0.0]


# Checks epsilon=0 always picks the legal token with the highest Q-value.
def test_greedy_action_selection_picks_highest_q_value():
    agent = TabularQAgent(epsilon=0.0, rng=np.random.default_rng(0))
    board = BoardState()
    state = discretize_state(board, player_id=0)
    agent.q_values(state)[:] = [0.1, 0.9, 0.2, 0.5]
    action = agent(board, player_id=0, roll=3, legal_tokens=(0, 2, 3))
    assert action == 3  # highest Q-value among the legal tokens (token 1 is excluded)


# Checks epsilon=1 always explores, still respecting the legal action set.
def test_fully_exploratory_agent_still_respects_legal_tokens():
    agent = TabularQAgent(epsilon=1.0, rng=np.random.default_rng(0))
    board = BoardState()
    legal_tokens = (1, 3)
    for _ in range(100):
        assert agent(board, player_id=0, roll=3, legal_tokens=legal_tokens) in legal_tokens


# Checks a terminal update ignores the next state entirely.
def test_update_with_done_ignores_next_state():
    agent = TabularQAgent(alpha=0.5, gamma=0.9)
    state = (0, 0, 0, 0)
    agent.update(state, action=1, reward=2.0, next_state=None, next_legal_tokens=(), done=True)
    assert agent.q_values(state)[1] == pytest.approx(0.5 * 2.0)  # alpha * (reward - 0)


# Checks a non-terminal update bootstraps off the masked max over next_legal_tokens only.
def test_update_masks_the_next_state_max():
    agent = TabularQAgent(alpha=0.5, gamma=1.0)
    next_state = (0, 0, 0, 0)
    agent.q_values(next_state)[:] = [10.0, 3.0, 7.0, 0.0]  # token 0 is the true max, but illegal next

    state = (1, 1, 1, 1)
    agent.update(state, action=2, reward=1.0, next_state=next_state, next_legal_tokens=(1, 2), done=False)

    expected_target = 1.0 + 1.0 * 7.0  # max over {token 1: 3.0, token 2: 7.0}, not token 0's 10.0
    assert agent.q_values(state)[2] == pytest.approx(0.5 * expected_target)
