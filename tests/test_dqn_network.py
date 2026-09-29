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


# Tests for SharedTokenQNetwork: weight sharing, output shape, and context-only mode.

from agents.dqn_network import SharedTokenQNetwork

# Checks the output shape matches QNetwork's contract: (batch, 4) regardless of block sizes.
def test_shared_token_network_output_shape():
    network = SharedTokenQNetwork(context_size=64, move_feature_size=8, threat_feature_size=3)
    dummy_batch = torch.zeros((5, 64 + 4 * 8 + 4 * 3))
    output = network(dummy_batch)
    assert output.shape == (5, NUM_TOKENS_PER_PLAYER)


# Checks two tokens with identical own-feature slices get identical Q-values, the point of sharing one scorer.
def test_shared_weights_give_identical_output_for_identical_token_features():
    network = SharedTokenQNetwork(context_size=64, move_feature_size=8, threat_feature_size=3)
    obs = torch.zeros((1, 64 + 4 * 8 + 4 * 3))
    obs[0, :64] = torch.linspace(0, 1, 64)  # non-trivial shared context
    obs[0, 64:72] = torch.linspace(0, 1, 8)  # token 0's move features
    obs[0, 64 + 24:64 + 24 + 8] = torch.linspace(0, 1, 8)  # token 3's move features, copied
    output = network(obs)
    assert torch.allclose(output[0, 0], output[0, 3])


# Checks a network with both per-token blocks disabled (context-only) still runs and gives 4 outputs.
def test_shared_token_network_with_no_per_token_blocks():
    network = SharedTokenQNetwork(context_size=64, move_feature_size=0, threat_feature_size=0)
    output = network(torch.zeros((2, 64)))
    assert output.shape == (2, NUM_TOKENS_PER_PLAYER)
