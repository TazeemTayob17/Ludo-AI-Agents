# Tests for training/run_training.py's orchestration. Deliberately tiny (a few episodes) -
# this proves the pipeline is wired correctly end-to-end, it is not a real training run.

import csv

import pytest

from env.rewards import REWARD_CONFIG, SPARSE_REWARD_CONFIG
from training.config import TrainingConfig, load_config
from training.run_training import build_reward_config, run_training

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
    best_win_rate = run_training(config, output_dir)
    assert 0.0 <= best_win_rate <= 1.0

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


# Checks build_reward_config resolves "dense"/"sparse" to the right dict, and rejects anything else.
def test_build_reward_config_resolves_dense_and_sparse():
    dense_config = TrainingConfig(agent_type="dqn", seed=0, num_episodes=1, reward_mode="dense")
    sparse_config = TrainingConfig(agent_type="dqn", seed=0, num_episodes=1, reward_mode="sparse")
    assert build_reward_config(dense_config) is REWARD_CONFIG
    assert build_reward_config(sparse_config) is SPARSE_REWARD_CONFIG

    bad_config = TrainingConfig(agent_type="dqn", seed=0, num_episodes=1, reward_mode="bogus")
    with pytest.raises(ValueError):
        build_reward_config(bad_config)


# Checks build_env produces a reproducible dice stream from the config seed. Without an
# injected rng, Gymnasium seeds np_random from OS entropy and runs cannot be reproduced.
def test_build_env_dice_stream_is_reproducible_from_the_seed():
    from training.run_training import build_env

    def first_rolls(dice_seed):
        config = TrainingConfig(agent_type="dqn", seed=0, num_episodes=1)
        env = build_env(config, dice_seed=dice_seed)
        _obs, info = env.reset()
        rolls = [info["dice_roll"]]
        for _ in range(8):
            if env.game.terminated or env.game.truncated:
                break
            legal = tuple(int(t) for t in info["action_mask"].nonzero()[0])
            _obs, _r, _t, _tr, info = env.step(legal[0])
            rolls.append(info["dice_roll"])
        return rolls

    assert first_rolls(1234) == first_rolls(1234)
    assert first_rolls(1234) != first_rolls(5678)


# Checks training and evaluation draw from different dice streams, so the eval games are not
# a replay of the games just trained on.
def test_training_and_evaluation_dice_streams_differ():
    from training.run_training import EVAL_DICE_SEED_OFFSET, TRAINING_DICE_SEED_OFFSET

    assert TRAINING_DICE_SEED_OFFSET != EVAL_DICE_SEED_OFFSET


# Checks a DQN run with move and threat features trains end to end and records both switches.
def test_run_training_with_move_and_threat_features(tmp_path):
    config = TrainingConfig(
        agent_type="dqn", seed=0, include_move_features=True, include_threat_features=True, **_COMMON_KWARGS
    )
    run_training(config, tmp_path)
    loaded = load_config(tmp_path / "config.json")
    assert loaded.include_move_features and loaded.include_threat_features
    assert (tmp_path / "checkpoints" / "episode_3.pt").exists()
