# Generates Step 10.4's hyperparameter sweep: a small, literature-justified grid, not a full search.

from __future__ import annotations

from dataclasses import replace

from training.config import TrainingConfig

LEARNING_RATES = (1e-3, 5e-4)
TARGET_SYNC_EVERY_STEPS = (250, 1000)
EPSILON_DECAY_EPISODES = (1000, 3000, 6000)

# Builds the sweep grid (2 learning rates x 2 sync frequencies x 3 decay schedules = 12 configs).
def generate_sweep_configs(base_config: TrainingConfig) -> list[TrainingConfig]:
    configs = []
    for learning_rate in LEARNING_RATES:
        for sync_every in TARGET_SYNC_EVERY_STEPS:
            for decay_episodes in EPSILON_DECAY_EPISODES:
                run_name = f"{base_config.run_name}_lr{learning_rate}_sync{sync_every}_decay{decay_episodes}"
                configs.append(
                    replace(
                        base_config,
                        run_name=run_name,
                        learning_rate=learning_rate,
                        target_sync_every_steps=sync_every,
                        epsilon_decay_episodes=decay_episodes,
                    )
                )
    return configs
