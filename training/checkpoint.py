# Saves and loads DQN/Double DQN and Tabular Q checkpoints, with traceable run metadata.

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch

from agents.dqn_agent import DQNAgent
from agents.tabular_q_agent import TabularQAgent
from training.config import get_git_commit_hash

# Saves a DQN/Double DQN checkpoint: both networks' weights plus traceable run metadata.
def save_dqn_checkpoint(path: Path, agent: DQNAgent, episode: int, config_dict: dict, eval_win_rate: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "online_state_dict": agent.online_network.state_dict(),
            "target_state_dict": agent.target_network.state_dict(),
            "episode": episode,
            "config": config_dict,
            "git_commit": get_git_commit_hash(),
            "eval_win_rate": eval_win_rate,
        },
        path,
    )

# Loads a DQN/Double DQN checkpoint's weights into an existing agent, and returns its metadata.
def load_dqn_checkpoint(path: Path, agent: DQNAgent) -> dict[str, Any]:
    checkpoint = torch.load(path, weights_only=False)
    agent.online_network.load_state_dict(checkpoint["online_state_dict"])
    agent.target_network.load_state_dict(checkpoint["target_state_dict"])
    return checkpoint

# Saves a Tabular Q checkpoint: the Q-table plus traceable run metadata.
def save_tabular_checkpoint(
    path: Path, agent: TabularQAgent, episode: int, config_dict: dict, eval_win_rate: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "q_table": agent.q_table,
                "episode": episode,
                "config": config_dict,
                "git_commit": get_git_commit_hash(),
                "eval_win_rate": eval_win_rate,
            },
            f,
        )

# Loads a Tabular Q checkpoint's Q-table into an existing agent, and returns its metadata.
def load_tabular_checkpoint(path: Path, agent: TabularQAgent) -> dict[str, Any]:
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    agent.q_table = checkpoint["q_table"]
    return checkpoint
