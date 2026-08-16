# Orchestrates Step 10's full scope: the hyperparameter sweep, then multi-seed final
# training for DQN, Double DQN, and Tabular Q, using the sweep's winning hyperparameters.

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from training.config import TrainingConfig
from training.run_training import run_training
from training.sweep import generate_sweep_configs

DEFAULT_SWEEP_EPISODES = 3000
DEFAULT_FINAL_EPISODES = 10000
DEFAULT_NUM_SEEDS = 5
DEFAULT_EVAL_EPISODES = 50
SWEEP_SEED = 0


# Runs every sweep config (plain DQN only - Double DQN reuses the same winning hyperparameters,
# since the two share the same hyperparameter surface and differ only in the target computation
# itself) and returns the winning config plus every config's result, best first.
def run_sweep_phase(
    output_root: Path, sweep_episodes: int, eval_episodes: int
) -> tuple[TrainingConfig, list[tuple[TrainingConfig, float]]]:
    base_config = TrainingConfig(
        agent_type="dqn", seed=SWEEP_SEED, num_episodes=sweep_episodes, eval_episodes=eval_episodes, run_name="sweep"
    )
    results = []
    for config in generate_sweep_configs(base_config):
        win_rate = run_training(config, output_root / "sweep" / config.run_name)
        results.append((config, win_rate))
    results.sort(key=lambda pair: pair[1], reverse=True)
    return results[0][0], results


# Runs num_seeds final training runs for one DQN-family agent type, using the swept hyperparameters.
def run_dqn_family_final_phase(
    agent_type: str,
    winning_config: TrainingConfig,
    output_root: Path,
    final_episodes: int,
    num_seeds: int,
    eval_episodes: int,
) -> list[float]:
    win_rates = []
    for seed in range(num_seeds):
        config = replace(
            winning_config,
            agent_type=agent_type,
            seed=seed,
            num_episodes=final_episodes,
            eval_episodes=eval_episodes,
            run_name=f"{agent_type}_final",
        )
        win_rates.append(run_training(config, output_root / "final" / f"{agent_type}_seed{seed}"))
    return win_rates


# Runs Tabular Q's final multi-seed training with its own (non-swept) default hyperparameters.
def run_tabular_q_final_phase(output_root: Path, final_episodes: int, num_seeds: int, eval_episodes: int) -> list[float]:
    win_rates = []
    for seed in range(num_seeds):
        config = TrainingConfig(
            agent_type="tabular_q",
            seed=seed,
            num_episodes=final_episodes,
            eval_episodes=eval_episodes,
            run_name="tabular_q_final",
        )
        win_rates.append(run_training(config, output_root / "final" / f"tabular_q_seed{seed}"))
    return win_rates


# Runs the whole Step 10 scope end to end and writes a summary.json tying it all together.
def run_full_training(
    output_root: Path,
    sweep_episodes: int = DEFAULT_SWEEP_EPISODES,
    final_episodes: int = DEFAULT_FINAL_EPISODES,
    num_seeds: int = DEFAULT_NUM_SEEDS,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
) -> dict:
    winning_config, sweep_results = run_sweep_phase(output_root, sweep_episodes, eval_episodes)

    summary = {
        "sweep_results": [
            {
                "run_name": config.run_name,
                "learning_rate": config.learning_rate,
                "target_sync_every_steps": config.target_sync_every_steps,
                "epsilon_decay_episodes": config.epsilon_decay_episodes,
                "eval_win_rate": win_rate,
            }
            for config, win_rate in sweep_results
        ],
        "winning_hyperparameters": {
            "learning_rate": winning_config.learning_rate,
            "target_sync_every_steps": winning_config.target_sync_every_steps,
            "epsilon_decay_episodes": winning_config.epsilon_decay_episodes,
        },
        "dqn_final_win_rates": run_dqn_family_final_phase(
            "dqn", winning_config, output_root, final_episodes, num_seeds, eval_episodes
        ),
        "double_dqn_final_win_rates": run_dqn_family_final_phase(
            "double_dqn", winning_config, output_root, final_episodes, num_seeds, eval_episodes
        ),
        "tabular_q_final_win_rates": run_tabular_q_final_phase(output_root, final_episodes, num_seeds, eval_episodes),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# Parses flags and runs the full Step 10 training scope from the command line.
def _main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 10's full scope: sweep, then multi-seed final training.")
    parser.add_argument("--output-root", default="runs/full", help="root directory for all sweep/final run output")
    parser.add_argument("--sweep-episodes", type=int, default=DEFAULT_SWEEP_EPISODES)
    parser.add_argument("--final-episodes", type=int, default=DEFAULT_FINAL_EPISODES)
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    args = parser.parse_args()
    run_full_training(
        Path(args.output_root), args.sweep_episodes, args.final_episodes, args.num_seeds, args.eval_episodes
    )


if __name__ == "__main__":
    _main()
