# Tests for training/config.py.

from training.config import TrainingConfig, get_git_commit_hash, load_config, save_config


# Checks a config survives a save/load round trip unchanged.
def test_save_and_load_round_trip(tmp_path):
    config = TrainingConfig(agent_type="dqn", seed=3, num_episodes=100, learning_rate=5e-4)
    path = tmp_path / "config.json"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded == config


# Checks the saved file also embeds a reward-dict snapshot and a git commit field.
def test_saved_file_includes_reward_snapshot_and_git_commit(tmp_path):
    import json

    config = TrainingConfig(agent_type="tabular_q", seed=0, num_episodes=10)
    path = tmp_path / "config.json"
    save_config(config, path)

    with open(path) as f:
        payload = json.load(f)
    assert "reward_config_snapshot" in payload
    assert payload["reward_config_snapshot"]["win"] == 2.0
    assert "git_commit" in payload  # value may be None outside a git repo, but the key must exist


# Checks the git commit hash lookup returns a plausible value (a hash string or None).
def test_get_git_commit_hash_returns_a_string_or_none():
    result = get_git_commit_hash()
    assert result is None or (isinstance(result, str) and len(result) > 0)


# Checks reward_mode defaults to "dense" and survives a save/load round trip when set to "sparse".
def test_reward_mode_defaults_to_dense_and_round_trips_when_sparse(tmp_path):
    assert TrainingConfig(agent_type="dqn", seed=0, num_episodes=10).reward_mode == "dense"

    config = TrainingConfig(agent_type="dqn", seed=0, num_episodes=10, reward_mode="sparse")
    path = tmp_path / "config.json"
    save_config(config, path)
    assert load_config(path).reward_mode == "sparse"


# Checks the saved reward snapshot follows the config's reward_mode rather than always
# recording the dense dict - a sparse run's provenance record must say "sparse".
def test_reward_snapshot_follows_reward_mode(tmp_path):
    import json

    from env.rewards import REWARD_CONFIG, SPARSE_REWARD_CONFIG

    dense_path = tmp_path / "dense.json"
    save_config(TrainingConfig(agent_type="dqn", seed=0, num_episodes=1), dense_path)
    assert json.loads(dense_path.read_text())["reward_config_snapshot"] == REWARD_CONFIG

    sparse_path = tmp_path / "sparse.json"
    save_config(TrainingConfig(agent_type="dqn", seed=0, num_episodes=1, reward_mode="sparse"), sparse_path)
    assert json.loads(sparse_path.read_text())["reward_config_snapshot"] == SPARSE_REWARD_CONFIG


# Checks a config saved before the feature switches existed loads with every switch off.
def test_old_config_without_feature_switches_loads_with_them_off(tmp_path):
    import json

    path = tmp_path / "old.json"
    path.write_text(json.dumps({"config": {"agent_type": "dqn", "seed": 0, "num_episodes": 10}}))
    spec = load_config(path).observation_spec()
    assert not (spec.include_dice_roll or spec.include_move_features or spec.include_threat_features)
