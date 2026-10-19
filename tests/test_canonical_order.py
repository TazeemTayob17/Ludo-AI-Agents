# Tests for canonical token order: order function, permutation invariance, action-slot mapping in the env and agent.

import numpy as np

from agents.dqn_agent import DQNAgent
from env.board import BoardState
from env.legal_moves import get_legal_tokens
from env.ludo_env import LudoEnv
from env.state_encoding import ObservationSpec, canonical_order, encode_observation

SPEC = ObservationSpec(include_move_features=True, include_threat_features=True, canonical_token_order=True)
TRACK_SPEC = ObservationSpec(board_encoding="track", include_dice_roll=True, canonical_token_order=True)


# Builds a random board for all four players.
def _random_board(rng) -> BoardState:
    board = BoardState()
    for player in range(4):
        for token in range(4):
            board.set(player, token, int(rng.integers(-1, 58)))
    return board


# Returns a copy of the board with one player's token ids permuted.
def _permuted(board: BoardState, player: int, perm) -> BoardState:
    clone = board.copy()
    clone.positions[player] = board.positions[player][list(perm)]
    return clone


# Checks the order puts the furthest token first, base last, and breaks ties by id.
def test_canonical_order_sorts_by_progress():
    board = BoardState()
    board.set(0, 0, 10)
    board.set(0, 1, 30)
    board.set(0, 2, 10)
    assert canonical_order(board, 0) == (1, 0, 2, 3)


# Checks relabelling any player's token ids leaves the observation unchanged, for both encodings.
def test_observation_is_invariant_to_token_labels():
    rng = np.random.default_rng(1)
    for spec in (SPEC, TRACK_SPEC):
        for _ in range(15):
            board = _random_board(rng)
            roll = int(rng.integers(1, 7))
            base = encode_observation(board, 0, roll, spec)
            for player in (0, 1, 3):
                perm = rng.permutation(4)
                assert np.array_equal(base, encode_observation(_permuted(board, player, perm), 0, roll, spec))


# Checks the env's action slot k moves the k-th canonical token, and the mask is in slot space.
def test_env_slots_map_to_canonical_tokens():
    env = LudoEnv(observation_spec=SPEC, rng=np.random.default_rng(3))
    obs, info = env.reset()
    for _ in range(40):
        if env.game.terminated or env.game.truncated:
            break
        order = canonical_order(env.game.board, 0)
        legal_tokens = get_legal_tokens(env.game.board, 0, info["dice_roll"])
        legal_slots = [slot for slot in range(4) if info["action_mask"][slot]]
        assert sorted(order[s] for s in legal_slots) == sorted(legal_tokens)
        slot = legal_slots[0]
        before = env.game.board.get(0, order[slot])
        obs, reward, terminated, truncated, info = env.step(slot)
        assert terminated or truncated or env.game.board.get(0, order[slot]) != before


# Checks the default env keeps token ids as actions.
def test_default_env_is_identity():
    env = LudoEnv(rng=np.random.default_rng(0))
    env.reset()
    assert [env.token_for_action(a) for a in range(4)] == [0, 1, 2, 3]


# Checks the agent returns a legal token id and the same physical move after relabelling its tokens.
def test_agent_picks_same_physical_move_after_relabelling():
    rng = np.random.default_rng(5)
    agent = DQNAgent(observation_spec=SPEC, epsilon=0.0, rng=np.random.default_rng(0))
    checked = 0
    while checked < 20:
        board = _random_board(rng)
        roll = int(rng.integers(1, 7))
        legal = get_legal_tokens(board, 0, roll)
        if len(legal) < 2:
            continue
        perm = rng.permutation(4)
        other = _permuted(board, 0, perm)
        legal_other = get_legal_tokens(other, 0, roll)
        pick = agent(board, 0, roll, legal)
        pick_other = agent(other, 0, roll, legal_other)
        assert pick in legal and pick_other in legal_other
        assert board.get(0, pick) == other.get(0, pick_other)
        checked += 1
