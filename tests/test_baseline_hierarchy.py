# Automated regression test (7.4): the baseline hierarchy must hold, not just look right by eye.

from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from env.ludo_env import LudoEnv

NUM_GAMES = 300
MAX_TURNS = 1000
MIN_HEURISTIC_WIN_RATE = 0.40


# Checks Heuristic clearly beats Random over many games, as a numeric, rerunnable threshold.
def test_heuristic_beats_random_win_rate_threshold():
    heuristic = HeuristicAgent()
    wins = 0

    for episode in range(NUM_GAMES):
        env = LudoEnv(opponent_policy=RandomAgent(), max_turns=MAX_TURNS)
        obs, info = env.reset(seed=episode)
        terminated = truncated = False
        while not (terminated or truncated):
            action = heuristic(env.game.board, env.agent_player_id, info["dice_roll"], tuple(info["action_mask"].nonzero()[0]))
            obs, reward, terminated, truncated, info = env.step(action)
        if terminated and info["winner"] == env.agent_player_id:
            wins += 1

    win_rate = wins / NUM_GAMES
    assert win_rate > MIN_HEURISTIC_WIN_RATE, f"Heuristic win rate {win_rate:.2%} did not clear {MIN_HEURISTIC_WIN_RATE:.0%}"
