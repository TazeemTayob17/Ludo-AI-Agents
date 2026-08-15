# Tests for training/sweep.py (Step 10.4's hyperparameter sweep grid).

from training.config import TrainingConfig
from training.sweep import EPSILON_DECAY_EPISODES, LEARNING_RATES, TARGET_SYNC_EVERY_STEPS, generate_sweep_configs


# Checks the grid produces exactly the documented 12 combinations (2 x 2 x 3), all unique.
def test_generates_the_full_grid_with_unique_run_names():
    base = TrainingConfig(agent_type="dqn", seed=0, num_episodes=1000, run_name="base")
    configs = generate_sweep_configs(base)
    assert len(configs) == len(LEARNING_RATES) * len(TARGET_SYNC_EVERY_STEPS) * len(EPSILON_DECAY_EPISODES)
    assert len(configs) == 12
    assert len(set(c.run_name for c in configs)) == 12


# Checks every combination of the three swept hyperparameters actually appears.
def test_covers_every_combination_of_swept_hyperparameters():
    base = TrainingConfig(agent_type="dqn", seed=0, num_episodes=1000, run_name="base")
    configs = generate_sweep_configs(base)
    seen = {(c.learning_rate, c.target_sync_every_steps, c.epsilon_decay_episodes) for c in configs}
    expected = {
        (lr, sync, decay) for lr in LEARNING_RATES for sync in TARGET_SYNC_EVERY_STEPS for decay in EPSILON_DECAY_EPISODES
    }
    assert seen == expected


# Checks every non-swept field is copied unchanged from the base config.
def test_non_swept_fields_are_preserved():
    base = TrainingConfig(agent_type="dqn", seed=5, num_episodes=1234, gamma=0.8, run_name="base")
    configs = generate_sweep_configs(base)
    for config in configs:
        assert config.seed == 5
        assert config.num_episodes == 1234
        assert config.gamma == 0.8


# Checks generating the sweep does not mutate the base config.
def test_base_config_is_not_mutated():
    base = TrainingConfig(agent_type="dqn", seed=0, num_episodes=1000, run_name="base")
    original_learning_rate = base.learning_rate
    generate_sweep_configs(base)
    assert base.learning_rate == original_learning_rate
    assert base.run_name == "base"
