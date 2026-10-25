# Tests for the risk and race feature blocks: hand-checked values, sizes, default-off behaviour and token-label invariance.

import numpy as np

from env.board import BoardState
from env.state_encoding import ObservationSpec, encode_observation, observation_size
from env.tactical_features import NUM_RACE_FEATURES, NUM_RISK_FEATURES, encode_race_features, encode_risk_features

RISK_SPEC = ObservationSpec(include_move_features=True, include_risk_features=True, include_race_features=True, canonical_token_order=True)


# Checks the moved token's capture chance and the expected progress lost on a position worked out by hand.
def test_risk_features_hand_checked():
    board = BoardState()
    board.set(0, 0, 19)
    board.set(1, 0, 3)
    features = encode_risk_features(board, 0, 1).reshape(4, 2)
    assert abs(features[0, 0] - 1 / 6) < 1e-6
    assert abs(features[0, 1] - (21 / (4 * 58)) * (1 / 6)) < 1e-6
    assert not features[1:].any()


# Checks a move onto a safe square has zero risk.
def test_risk_is_zero_on_a_safe_square():
    board = BoardState()
    board.set(0, 0, 20)
    board.set(1, 0, 3)
    assert encode_risk_features(board, 0, 1).reshape(4, 2)[0, 0] == 0.0


# Checks the race block is zero-centred at the start and rises with the acting player's progress.
def test_race_features_track_progress():
    board = BoardState()
    start = encode_race_features(board, 0)
    assert start.shape == (NUM_RACE_FEATURES,) and start[:4].sum() == 0.0 and start[4:].tolist() == [1.0] * 4
    board.set(0, 0, 30)
    assert encode_race_features(board, 0)[0] > 0.0


# Checks sizes, that both blocks are off by default, and that relabelling tokens leaves the observation unchanged.
def test_sizes_default_off_and_label_invariance():
    assert observation_size(RISK_SPEC) == 64 + 32 + NUM_RISK_FEATURES + NUM_RACE_FEATURES
    assert observation_size() == 64
    rng = np.random.default_rng(2)
    for _ in range(15):
        board = BoardState()
        for player in range(4):
            for token in range(4):
                board.set(player, token, int(rng.integers(-1, 58)))
        roll = int(rng.integers(1, 7))
        base = encode_observation(board, 0, roll, RISK_SPEC)
        shuffled = board.copy()
        shuffled.positions[0] = board.positions[0][rng.permutation(4)]
        assert np.array_equal(base, encode_observation(shuffled, 0, roll, RISK_SPEC))
