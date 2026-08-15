# Versioned training run configuration: hyperparameters, seed, and a reward-dict snapshot.

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from env.game_state import DEFAULT_MAX_TURNS
from env.rewards import REWARD_CONFIG

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# One training run's full configuration: agent choice, hyperparameters, seed, and episode budget.
@dataclass
class TrainingConfig:
    agent_type: str
    seed: int
    num_episodes: int
    run_name: str = "run"
    max_turns: int = DEFAULT_MAX_TURNS
    opponent_type: str = "heuristic"
    eval_episodes: int = 50
    checkpoint_every_episodes: int = 10000
    gamma: float = 0.95
    hidden_size: int = 128
    learning_rate: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 2000
    target_sync_every_steps: int = 500
    batch_size: int = 32
    replay_capacity: int = 50000
    min_buffer_size: int = 1000
    tabular_alpha: float = 0.1
    tabular_epsilon_start: float = 1.0
    tabular_epsilon_end: float = 0.05
    tabular_epsilon_decay_episodes: int = 2000

# Returns the current git commit hash, or None if it can't be determined.
def get_git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

# Writes the config, a reward-dict snapshot, and the git commit hash to a JSON file.
def save_config(config: TrainingConfig, path: Path) -> None:
    payload = {
        "config": asdict(config),
        "reward_config_snapshot": dict(REWARD_CONFIG),
        "git_commit": get_git_commit_hash(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

# Reads back a TrainingConfig from a JSON file written by save_config.
def load_config(path: Path) -> TrainingConfig:
    with open(path) as f:
        payload = json.load(f)
    return TrainingConfig(**payload["config"])
