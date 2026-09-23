# Convenience CLI: builds a TrainingConfig from flags and runs it, no config.json needed first.

from __future__ import annotations

import argparse
from pathlib import Path

from training.config import TrainingConfig
from training.run_training import run_training


# Parses common flags, builds a TrainingConfig from them, and runs it.
def _main() -> None:
    parser = argparse.ArgumentParser(description="Run one Ludo training run directly from command-line flags.")
    parser.add_argument("--agent-type", required=True, choices=["dqn", "double_dqn", "tabular_q"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--run-name", default="run")
    parser.add_argument("--opponent-type", default="heuristic", choices=["random", "heuristic"])
    parser.add_argument("--reward-mode", default="dense", choices=["dense", "sparse"])
    parser.add_argument("--gamma", type=float, default=TrainingConfig.gamma)
    parser.add_argument(
        "--include-dice-roll", action="store_true", help="append the pending roll to the observation"
    )
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--target-sync-every-steps", type=int, default=TrainingConfig.target_sync_every_steps)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=TrainingConfig.epsilon_decay_episodes)
    parser.add_argument("--eval-episodes", type=int, default=TrainingConfig.eval_episodes)
    parser.add_argument(
        "--checkpoint-every-episodes", type=int, default=TrainingConfig.checkpoint_every_episodes
    )
    parser.add_argument("--output-dir", default=None, help="defaults to runs/<run-name>_seed<seed>")
    args = parser.parse_args()

    config = TrainingConfig(
        agent_type=args.agent_type,
        seed=args.seed,
        num_episodes=args.num_episodes,
        run_name=args.run_name,
        opponent_type=args.opponent_type,
        reward_mode=args.reward_mode,
        gamma=args.gamma,
        include_dice_roll=args.include_dice_roll,
        learning_rate=args.learning_rate,
        target_sync_every_steps=args.target_sync_every_steps,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        eval_episodes=args.eval_episodes,
        checkpoint_every_episodes=args.checkpoint_every_episodes,
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path("runs") / f"{args.run_name}_seed{args.seed}"
    run_training(config, output_dir)


if __name__ == "__main__":
    _main()
