# Tests for training/run_training.py's orchestration. Deliberately tiny (a few episodes) -
# this proves the pipeline is wired correctly end-to-end, it is not a real training run.

import csv

from training.config import TrainingConfig, load_config
from training.run_training import run_training

_COMMON_KWARGS = dict(
    num_episodes=3,
    checkpoint_every_episodes=2,
    eval_episodes=3,
    max_turns=300,
    opponent_type="random",
    min_buffer_size=5,
    batch_size=4,
    replay_capacity=200,
)


# Checks a tiny DQN run produces a config file, a full-length log, and both checkpoints.
def test_run_training_produces_expected_outputs_for_dqn(tmp_path):
    config = TrainingConfig(agent_type="dqn", seed=0, **_COMMON_KWARGS)
    output_dir = tmp_path / "dqn_run"
    run_training(config, output_dir)

    assert load_config(output_dir / "config.json") == config

    with open(output_dir / "log.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert [row["episode"] for row in rows] == ["1", "2", "3"]

    checkpoints = sorted(p.name for p in (output_dir / "checkpoints").iterdir())
    assert checkpoints == ["best.pt", "episode_2.pt", "episode_3.pt"]


# Checks a tiny Tabular Q run produces a config file, a full-length log, and both checkpoints.
def test_run_training_produces_expected_outputs_for_tabular_q(tmp_path):
    config = TrainingConfig(agent_type="tabular_q", seed=0, **_COMMON_KWARGS)
    output_dir = tmp_path / "tabular_run"
    run_training(config, output_dir)

    with open(output_dir / "log.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert all(row["avg_loss"] == "" for row in rows)  # tabular Q has no batch loss

    checkpoints = sorted(p.name for p in (output_dir / "checkpoints").iterdir())
    assert checkpoints == ["best.pkl", "episode_2.pkl", "episode_3.pkl"]
