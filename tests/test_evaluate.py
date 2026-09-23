# Tests for training/evaluate.py.

import numpy as np

from agents.random_agent import RandomAgent
from agents.tabular_q_agent import TabularQAgent
from env.ludo_env import LudoEnv
from training.evaluate import evaluate
from training.tabular_q_trainer import TabularQTrainer


# Checks evaluate returns a win rate in [0, 1] over a small number of episodes.
def test_evaluate_returns_a_fraction_between_zero_and_one():
    trainer = TabularQTrainer(TabularQAgent(rng=np.random.default_rng(0)))
    env = LudoEnv(opponent_policy=RandomAgent(rng=np.random.default_rng(1)), max_turns=300)
    win_rate = evaluate(trainer, env, num_episodes=5)
    assert 0.0 <= win_rate <= 1.0


# Checks evaluate does not permanently change the trainer's exploration rate.
def test_evaluate_does_not_leave_epsilon_changed():
    trainer = TabularQTrainer(TabularQAgent(epsilon=0.3, rng=np.random.default_rng(0)))
    env = LudoEnv(opponent_policy=RandomAgent(rng=np.random.default_rng(1)), max_turns=300)
    evaluate(trainer, env, num_episodes=3)
    assert trainer.agent.epsilon == 0.3


# Rolls a non-six every time, so the agent's tokens can never leave Base.
class _NeverSixRng:
    def integers(self, low, high):
        return 3


# Regression test: evaluate() must not crash if reset() leaves an already-truncated
# episode - the agent's tokens never rolling a 6 for long enough that max_turns is
# reached before its first real decision. It used to call step() anyway and crash with
# "no pending decision" instead of correctly counting it as a zero-length loss.
def test_evaluate_handles_an_already_truncated_episode_without_crashing():
    trainer = TabularQTrainer(TabularQAgent(rng=np.random.default_rng(0)))
    env = LudoEnv(opponent_policy=RandomAgent(rng=np.random.default_rng(1)), max_turns=6, rng=_NeverSixRng())
    win_rate = evaluate(trainer, env, num_episodes=3)
    assert win_rate == 0.0
