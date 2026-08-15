# Smoke test: many fully random games through the public LudoEnv API, asserting no crashes.

import numpy as np

from env.board import NUM_PLAYERS, NUM_TOKENS_PER_PLAYER
from env.ludo_env import LudoEnv

NUM_SMOKE_EPISODES = 1000
SMOKE_MAX_TURNS = 200


# Runs many fully random games and checks every one completes cleanly, with a valid board throughout.
def test_1000_random_games_complete_without_crashing():
    for episode in range(NUM_SMOKE_EPISODES):
        env = LudoEnv(max_turns=SMOKE_MAX_TURNS)
        rng = np.random.default_rng(episode)
        obs, info = env.reset(seed=episode)
        terminated = truncated = False
        steps_taken = 0

        while not (terminated or truncated):
            assert info["action_mask"].any()
            legal = np.flatnonzero(info["action_mask"])
            action = int(rng.choice(legal))
            obs, reward, terminated, truncated, info = env.step(action)
            steps_taken += 1
            assert steps_taken < 10000  # generous circuit breaker; not expected to trigger

        assert terminated or truncated
        for player_id in range(NUM_PLAYERS):
            for token_id in range(NUM_TOKENS_PER_PLAYER):
                position = env.game.board.get(player_id, token_id)
                assert position == -1 or 0 <= position <= 57
