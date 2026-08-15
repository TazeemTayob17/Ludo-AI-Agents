# Tests for training/tabular_q_trainer.py. Deliberately tiny - checks the wiring works,
# not a claim about training quality.

import numpy as np

from agents.random_agent import RandomAgent
from agents.tabular_q_agent import TabularQAgent
from env.ludo_env import LudoEnv
from training.tabular_q_trainer import TabularQTrainer


def _make_trainer():
    agent = TabularQAgent(rng=np.random.default_rng(0))
    return TabularQTrainer(agent)


# Checks run_episode returns sane, correctly-typed stats.
def test_run_episode_returns_valid_stats():
    trainer = _make_trainer()
    env = LudoEnv(opponent_policy=RandomAgent(rng=np.random.default_rng(1)), max_turns=300)
    stats = trainer.run_episode(env, epsilon=0.5)
    assert isinstance(stats.total_reward, float)
    assert isinstance(stats.won, bool)
    assert stats.episode_length > 0
    assert stats.avg_loss is None  # tabular Q has no batch loss to report


# Checks an episode actually updates the Q-table.
def test_episode_populates_the_q_table():
    trainer = _make_trainer()
    env = LudoEnv(opponent_policy=RandomAgent(rng=np.random.default_rng(1)), max_turns=300)
    assert len(trainer.agent.q_table) == 0
    trainer.run_episode(env, epsilon=0.5)
    assert len(trainer.agent.q_table) > 0


# Checks select_greedy_action returns a legal action and restores epsilon afterward.
def test_select_greedy_action_is_legal_and_restores_epsilon():
    trainer = _make_trainer()
    trainer.agent.epsilon = 0.7
    env = LudoEnv(opponent_policy=RandomAgent(rng=np.random.default_rng(1)), max_turns=300)
    obs, info = env.reset(seed=0)
    action = trainer.select_greedy_action(env, obs, info)
    assert info["action_mask"][action]
    assert trainer.agent.epsilon == 0.7
