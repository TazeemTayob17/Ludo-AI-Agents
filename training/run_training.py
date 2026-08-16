# Builds an agent/trainer/environment from a config and runs the full training loop.

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from agents.dqn_agent import DQNAgent
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from agents.replay_buffer import ReplayBuffer
from agents.tabular_q_agent import TabularQAgent
from env.ludo_env import LudoEnv
from training.checkpoint import save_dqn_checkpoint, save_tabular_checkpoint
from training.config import TrainingConfig, load_config, save_config
from training.csv_logger import CsvLogger
from training.dqn_trainer import DQNTrainer
from training.evaluate import evaluate
from training.schedules import linear_epsilon
from training.tabular_q_trainer import TabularQTrainer

LOG_FIELDNAMES = ["episode", "total_reward", "won", "episode_length", "epsilon", "avg_loss"]

# Builds the (agent, trainer) pair matching the config's agent_type.
def build_agent_and_trainer(config: TrainingConfig):
    rng = np.random.default_rng(config.seed)
    if config.agent_type in ("dqn", "double_dqn"):
        agent = DQNAgent(
            double_dqn=(config.agent_type == "double_dqn"),
            hidden_size=config.hidden_size,
            gamma=config.gamma,
            learning_rate=config.learning_rate,
            epsilon=config.epsilon_start,
            rng=rng,
        )
        buffer = ReplayBuffer(capacity=config.replay_capacity, rng=rng)
        trainer = DQNTrainer(
            agent,
            buffer,
            batch_size=config.batch_size,
            min_buffer_size=config.min_buffer_size,
            sync_every_steps=config.target_sync_every_steps,
        )
        return agent, trainer
    if config.agent_type == "tabular_q":
        agent = TabularQAgent(alpha=config.tabular_alpha, gamma=config.gamma, epsilon=config.tabular_epsilon_start, rng=rng)
        return agent, TabularQTrainer(agent)
    raise ValueError(f"unknown agent_type: {config.agent_type}")

# Builds the fixed opponent policy every non-agent seat uses.
def build_opponent_policy(config: TrainingConfig, rng: np.random.Generator):
    if config.opponent_type == "random":
        return RandomAgent(rng=rng)
    if config.opponent_type == "heuristic":
        return HeuristicAgent()
    raise ValueError(f"unknown opponent_type: {config.opponent_type}")

# Returns the epsilon schedule this episode should use, per the config's agent type.
def _epsilon_for_episode(config: TrainingConfig, episode: int) -> float:
    if config.agent_type == "tabular_q":
        return linear_epsilon(
            episode, config.tabular_epsilon_start, config.tabular_epsilon_end, config.tabular_epsilon_decay_episodes
        )
    return linear_epsilon(episode, config.epsilon_start, config.epsilon_end, config.epsilon_decay_episodes)

# Saves a checkpoint matching the config's agent type.
def _save_checkpoint(config: TrainingConfig, agent, path: Path, episode: int, eval_win_rate: float) -> None:
    if config.agent_type in ("dqn", "double_dqn"):
        save_dqn_checkpoint(path, agent, episode, asdict(config), eval_win_rate)
    else:
        save_tabular_checkpoint(path, agent, episode, asdict(config), eval_win_rate)

# Runs the full training loop for one config: logs every episode, checkpoints periodically,
# and keeps a separate best-so-far checkpoint by evaluation win rate. Returns that best win rate.
def run_training(config: TrainingConfig, output_dir: Path) -> float:
    output_dir = Path(output_dir)
    save_config(config, output_dir / "config.json")

    agent, trainer = build_agent_and_trainer(config)
    opponent_rng = np.random.default_rng(config.seed + 1)
    env = LudoEnv(opponent_policy=build_opponent_policy(config, opponent_rng), max_turns=config.max_turns)

    checkpoints_dir = output_dir / "checkpoints"
    extension = "pt" if config.agent_type in ("dqn", "double_dqn") else "pkl"
    logger = CsvLogger(output_dir / "log.csv", LOG_FIELDNAMES)
    best_win_rate = -1.0

    try:
        for episode in range(1, config.num_episodes + 1):
            epsilon = _epsilon_for_episode(config, episode)
            stats = trainer.run_episode(env, epsilon)
            logger.log(
                {
                    "episode": episode,
                    "total_reward": stats.total_reward,
                    "won": int(stats.won),
                    "episode_length": stats.episode_length,
                    "epsilon": epsilon,
                    "avg_loss": stats.avg_loss if stats.avg_loss is not None else "",
                }
            )

            is_checkpoint_episode = episode % config.checkpoint_every_episodes == 0
            if is_checkpoint_episode or episode == config.num_episodes:
                eval_win_rate = evaluate(trainer, env, config.eval_episodes)
                _save_checkpoint(config, agent, checkpoints_dir / f"episode_{episode}.{extension}", episode, eval_win_rate)
                if eval_win_rate > best_win_rate:
                    best_win_rate = eval_win_rate
                    _save_checkpoint(config, agent, checkpoints_dir / f"best.{extension}", episode, eval_win_rate)
    finally:
        logger.close()

    return best_win_rate

# Parses --config/--output-dir and runs one training run from the command line.
def _main() -> None:
    parser = argparse.ArgumentParser(description="Run one Ludo training run from a config file.")
    parser.add_argument("--config", required=True, help="path to a saved TrainingConfig JSON file")
    parser.add_argument("--output-dir", required=True, help="directory to write logs/checkpoints to")
    args = parser.parse_args()
    run_training(load_config(Path(args.config)), Path(args.output_dir))

if __name__ == "__main__":
    _main()
