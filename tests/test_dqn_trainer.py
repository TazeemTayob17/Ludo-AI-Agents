# Tests for training/dqn_trainer.py. Deliberately tiny (a couple of episodes) - this checks
# the wiring works, it is not a claim about training quality (that's Step 10's real runs).

import numpy as np

from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
from agents.replay_buffer import ReplayBuffer
from env.ludo_env import LudoEnv
from training.dqn_trainer import DQNTrainer


def _make_trainer():
    agent = DQNAgent(rng=np.random.default_rng(0))
    buffer = ReplayBuffer(capacity=500, rng=np.random.default_rng(0))
    return DQNTrainer(agent, buffer, batch_size=8, min_buffer_size=20, sync_every_steps=10)


# Checks run_episode returns sane, correctly-typed stats.
def test_run_episode_returns_valid_stats():
    trainer = _make_trainer()
    env = LudoEnv(opponent_policy=RandomAgent(rng=np.random.default_rng(1)), max_turns=300)
    stats = trainer.run_episode(env, epsilon=0.5)
    assert isinstance(stats.total_reward, float)
    assert isinstance(stats.won, bool)
    assert stats.episode_length > 0
    assert stats.epsilon == 0.5


# Checks a few episodes actually populate the replay buffer and produce training losses.
def test_a_few_episodes_populate_the_buffer_and_train():
    trainer = _make_trainer()
    env = LudoEnv(opponent_policy=RandomAgent(rng=np.random.default_rng(1)), max_turns=300)
    stats = None
    for _ in range(3):
        stats = trainer.run_episode(env, epsilon=0.5)
    assert len(trainer.replay_buffer) > 0
    assert stats.avg_loss is not None  # buffer should have exceeded min_buffer_size by episode 3


# Checks select_greedy_action returns a legal action and restores epsilon afterward.
def test_select_greedy_action_is_legal_and_restores_epsilon():
    trainer = _make_trainer()
    trainer.agent.epsilon = 0.7
    env = LudoEnv(opponent_policy=RandomAgent(rng=np.random.default_rng(1)), max_turns=300)
    obs, info = env.reset(seed=0)
    action = trainer.select_greedy_action(env, obs, info)
    assert info["action_mask"][action]
    assert trainer.agent.epsilon == 0.7
