# Tests for training/run_full_training.py. Deliberately tiny (2 episodes per run) - proves
# the orchestration is wired correctly end-to-end, it is not a real training run.

import json

from training.run_full_training import run_full_training
from training.sweep import LEARNING_RATES, TARGET_SYNC_EVERY_STEPS, EPSILON_DECAY_EPISODES


# Checks the whole pipeline runs, picks a winner, and writes a complete summary.json.
def test_run_full_training_produces_a_complete_summary(tmp_path):
    output_root = tmp_path / "full"
    summary = run_full_training(output_root, sweep_episodes=2, final_episodes=2, num_seeds=1, eval_episodes=2)

    assert len(summary["sweep_results"]) == len(LEARNING_RATES) * len(TARGET_SYNC_EVERY_STEPS) * len(EPSILON_DECAY_EPISODES)
    win_rates = [r["eval_win_rate"] for r in summary["sweep_results"]]
    assert win_rates == sorted(win_rates, reverse=True)  # best result listed first

    winner = summary["winning_hyperparameters"]
    assert winner["learning_rate"] in LEARNING_RATES
    assert winner["target_sync_every_steps"] in TARGET_SYNC_EVERY_STEPS
    assert winner["epsilon_decay_episodes"] in EPSILON_DECAY_EPISODES

    assert len(summary["dqn_final_win_rates"]) == 1
    assert len(summary["double_dqn_final_win_rates"]) == 1
    assert len(summary["tabular_q_final_win_rates"]) == 1

    saved_summary = json.loads((output_root / "summary.json").read_text())
    assert saved_summary == summary


# Checks every sweep run and every final run actually produced its own output directory.
def test_run_full_training_produces_output_dirs_for_every_run(tmp_path):
    output_root = tmp_path / "full"
    run_full_training(output_root, sweep_episodes=2, final_episodes=2, num_seeds=1, eval_episodes=2)

    sweep_dirs = list((output_root / "sweep").iterdir())
    assert len(sweep_dirs) == 12

    final_dirs = sorted(p.name for p in (output_root / "final").iterdir())
    assert final_dirs == ["double_dqn_seed0", "dqn_seed0", "tabular_q_seed0"]
