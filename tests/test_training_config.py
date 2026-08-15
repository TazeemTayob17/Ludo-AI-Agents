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
