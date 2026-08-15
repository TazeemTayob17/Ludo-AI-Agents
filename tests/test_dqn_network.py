# Tests for agents/dqn_network.py (Step 8.1).

import torch

from agents.dqn_network import QNetwork
from env.board import NUM_TOKENS_PER_PLAYER
from env.state_encoding import OBSERVATION_SIZE


# Checks a batch of dummy observations produces one Q-value row per token action.
def test_forward_pass_shape_for_a_batch():
    network = QNetwork()
    dummy_batch = torch.zeros((5, OBSERVATION_SIZE))
    output = network(dummy_batch)
    assert output.shape == (5, NUM_TOKENS_PER_PLAYER)


# Checks a batch of size 1 still produces a correctly-shaped output.
def test_forward_pass_shape_for_a_single_observation():
    network = QNetwork()
    dummy_batch = torch.zeros((1, OBSERVATION_SIZE))
    output = network(dummy_batch)
    assert output.shape == (1, NUM_TOKENS_PER_PLAYER)


# Checks a custom hidden size still produces the same input/output shape contract.
def test_custom_hidden_size_does_not_change_input_or_output_shape():
    network = QNetwork(hidden_size=16)
    dummy_batch = torch.zeros((3, OBSERVATION_SIZE))
    output = network(dummy_batch)
    assert output.shape == (3, NUM_TOKENS_PER_PLAYER)
