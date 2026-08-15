# Tests for training/checkpoint.py.

import numpy as np
import torch

from agents.dqn_agent import DQNAgent
from agents.tabular_q_agent import TabularQAgent
from training.checkpoint import (
    load_dqn_checkpoint,
    load_tabular_checkpoint,
    save_dqn_checkpoint,
    save_tabular_checkpoint,
)


# Checks a DQN checkpoint round-trips its weights and carries the expected metadata.
def test_dqn_checkpoint_round_trip(tmp_path):
    agent = DQNAgent(rng=np.random.default_rng(0))
    with torch.no_grad():
        agent.online_network.fc3.bias.fill_(7.0)

    path = tmp_path / "checkpoint.pt"
    save_dqn_checkpoint(path, agent, episode=42, config_dict={"seed": 1}, eval_win_rate=0.55)

    fresh_agent = DQNAgent(rng=np.random.default_rng(1))
    metadata = load_dqn_checkpoint(path, fresh_agent)

    assert torch.allclose(fresh_agent.online_network.fc3.bias, agent.online_network.fc3.bias)
    assert torch.allclose(fresh_agent.target_network.fc3.bias, agent.target_network.fc3.bias)
    assert metadata["episode"] == 42
    assert metadata["config"] == {"seed": 1}
    assert metadata["eval_win_rate"] == 0.55
    assert "git_commit" in metadata


# Checks a Tabular Q checkpoint round-trips its Q-table and carries the expected metadata.
def test_tabular_checkpoint_round_trip(tmp_path):
    agent = TabularQAgent(rng=np.random.default_rng(0))
    agent.q_values((1, 2, 3, 4))[:] = [1.0, 2.0, 3.0, 4.0]

    path = tmp_path / "checkpoint.pkl"
    save_tabular_checkpoint(path, agent, episode=17, config_dict={"seed": 2}, eval_win_rate=0.3)

    fresh_agent = TabularQAgent(rng=np.random.default_rng(1))
    metadata = load_tabular_checkpoint(path, fresh_agent)

    assert fresh_agent.q_values((1, 2, 3, 4)).tolist() == [1.0, 2.0, 3.0, 4.0]
    assert metadata["episode"] == 17
    assert metadata["config"] == {"seed": 2}
    assert metadata["eval_win_rate"] == 0.3
    assert "git_commit" in metadata
