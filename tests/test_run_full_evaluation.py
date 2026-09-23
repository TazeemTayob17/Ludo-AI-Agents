# Tests for training/run_full_evaluation.py. Builds tiny fake "Step 10 output" (a few
# episodes per run, not real training) so the Step 11 pipeline can be exercised end to
# end without depending on a real multi-hour training run's output.

import csv
import json

from training.config import TrainingConfig
from training.run_full_evaluation import (
    AGENT_TYPES,
    NUM_TRAINED_SEEDS,
    load_trained_policy,
    run_full_evaluation,
)
from training.run_training import run_training

_TRAIN_KWARGS = dict(
    num_episodes=3,
    checkpoint_every_episodes=2,
    eval_episodes=2,
    max_turns=100,
    opponent_type="random",
    min_buffer_size=2,
    batch_size=2,
    replay_capacity=50,
)


# Builds a fake runs/full/final/ tree (every agent_type x seed combo) plus a matching
# summary.json, mirroring what run_full_training.py would have produced for real.
def _build_fake_final_root(tmp_path):
    final_root = tmp_path / "full" / "final"
    for agent_type in AGENT_TYPES:
        for seed in range(NUM_TRAINED_SEEDS):
            config = TrainingConfig(agent_type=agent_type, seed=seed, **_TRAIN_KWARGS)
            run_training(config, final_root / f"{agent_type}_seed{seed}")

    summary = {"winning_hyperparameters": {"learning_rate": 1e-3, "target_sync_every_steps": 100, "epsilon_decay_episodes": 50}}
    (final_root.parent / "summary.json").write_text(json.dumps(summary))
    return final_root


# Checks a trained run's saved checkpoint can be reloaded as a working greedy policy.
def test_load_trained_policy_produces_a_callable_policy(tmp_path):
    final_root = _build_fake_final_root(tmp_path)
    policy = load_trained_policy(final_root / "dqn_seed0")

    from agents.heuristic_agent import HeuristicAgent
    from env.ludo_env import LudoEnv

    env = LudoEnv(opponent_policy=HeuristicAgent(), max_turns=100)
    obs, info = env.reset(seed=0)
    action = policy(env, obs, info)
    assert info["action_mask"][action]


# Checks the full Step 11 pipeline (tournament + ablation) runs end to end and writes both CSVs.
def test_run_full_evaluation_writes_both_csv_files(tmp_path):
    final_root = _build_fake_final_root(tmp_path)
    output_dir = tmp_path / "full" / "evaluation"

    result = run_full_evaluation(
        final_root,
        output_dir,
        tournament_episodes=2,
        run_reward_ablation=True,
        ablation_episodes=2,
        ablation_seeds=1,
        ablation_eval_episodes=2,
    )

    # Every trained agent/seed combo, plus the two reference policies, plus the ablation seed.
    expected_labels = {f"{a}_seed{s}" for a in AGENT_TYPES for s in range(NUM_TRAINED_SEEDS)}
    expected_labels |= {"random_baseline", "heuristic_mirror", "dqn_sparse_seed0"}
    assert set(result["tournament_results"].keys()) == expected_labels

    # One summary row per agent type plus the ablation's own "dqn_sparse" row.
    summary_agent_types = {s["agent_type"] for s in result["summaries"]}
    assert summary_agent_types == set(AGENT_TYPES) | {"dqn_sparse"}

    with open(output_dir / "tournament_results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(expected_labels)

    with open(output_dir / "summary_stats.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(AGENT_TYPES) + 1


# Checks skipping the reward ablation leaves the tournament results untouched by it.
def test_skipping_reward_ablation_omits_the_sparse_dqn_run(tmp_path):
    final_root = _build_fake_final_root(tmp_path)
    output_dir = tmp_path / "full" / "evaluation"

    result = run_full_evaluation(final_root, output_dir, tournament_episodes=2, run_reward_ablation=False)

    assert not any(label.startswith("dqn_sparse") for label in result["tournament_results"])
    assert not (output_dir / "reward_ablation_runs").exists()


# Checks trained policies load from the final checkpoint, so a missing best checkpoint doesn't matter.
def test_load_trained_policy_uses_the_final_checkpoint(tmp_path):
    from training.run_training import run_training as train

    run_dir = tmp_path / "dqn_seed0"
    train(TrainingConfig(agent_type="dqn", seed=0, **_TRAIN_KWARGS), run_dir)
    (run_dir / "checkpoints" / "best.pt").unlink()
    assert callable(load_trained_policy(run_dir))
