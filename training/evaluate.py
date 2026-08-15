# Runs greedy (no-exploration) episodes to measure a trainer's current win rate.

from __future__ import annotations

from env.ludo_env import LudoEnv

# Plays num_episodes greedily and returns the fraction that ended in a win.
def evaluate(trainer, env: LudoEnv, num_episodes: int) -> float:
    wins = 0
    for _ in range(num_episodes):
        obs, info = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = trainer.select_greedy_action(env, obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            wins += 1
    return wins / num_episodes
