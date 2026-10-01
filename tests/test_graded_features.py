# Tests for the Phase 3 graded feature block: per-move graded values, race standing, and its wiring.

import numpy as np
import pytest

from agents.dqn_agent import DQNAgent
from env.board import BoardState
from env.state_encoding import ObservationSpec, encode_observation, observation_size
from env.tactical_features import (
    GRADED_MOVE_FEATURE_NAMES,
    NUM_GRADED_FEATURES,
    encode_graded_features,
)
from env.threats import count_attackers
from training.config import TrainingConfig

NUM_PER_TOKEN = len(GRADED_MOVE_FEATURE_NAMES)


# Builds a board with only the given (player, token, relative position) placements.
def _board(*placements):
    board = BoardState()
    for player_id, token_id, position in placements:
        board.set(player_id, token_id, position)
    return board


# Returns one token's graded move values for player 0 as a name -> value dict.
def _graded(board, roll, token_id):
    features = encode_graded_features(board, 0, roll)[: 4 * NUM_PER_TOKEN].reshape(4, NUM_PER_TOKEN)
    return dict(zip(GRADED_MOVE_FEATURE_NAMES, features[token_id]))


# Checks the block has 4 tokens x 7 values plus 8 race values.
def test_block_size():
    assert NUM_GRADED_FEATURES == 36
    assert encode_graded_features(BoardState(), 0, 6).shape == (36,)


# Checks with no pending roll the per-token part is zero.
def test_no_roll_gives_zero_move_values():
    assert not encode_graded_features(_board((0, 0, 10)), 0, None)[:28].any()


# Checks an illegal token is all zeros while a legal one has progress.
def test_illegal_token_is_zero():
    board = _board((0, 0, 10))
    assert _graded(board, 3, 0)["progress_gained"] > 0
    assert not any(_graded(board, 3, 1).values())


# Checks progress gained and distance-to-home are scaled from the real move.
def test_progress_and_distance_to_home():
    values = _graded(_board((0, 0, 10)), 3, 0)
    assert values["progress_gained"] == pytest.approx(3 / 6)
    assert values["distance_to_home_after"] == pytest.approx((57 - 13) / 58)


# Checks exiting base counts as one square of progress.
def test_exit_base_progress():
    assert _graded(BoardState(), 6, 0)["progress_gained"] == pytest.approx(1 / 6)


# Checks capture value grows with how far the captured token had travelled.
def test_capture_value_scales_with_victim_progress():
    far = _graded(_board((0, 0, 10), (2, 0, 38)), 2, 0)["capture_value"]
    assert far == pytest.approx(39 / 58)
    assert _graded(_board((0, 0, 10)), 2, 0)["capture_value"] == 0.0


# Checks attacker count and capture risk after a move that lands in front of two attackers with distinct rolls.
def test_attackers_and_risk_after_move():
    board = _board((0, 0, 10), (3, 0, 23), (2, 0, 34))
    values = _graded(board, 2, 0)
    assert values["attackers_after"] == pytest.approx(2 / 3)
    assert values["capture_risk_after"] == pytest.approx(2 / 6)


# Checks landing on a safe square carries no danger even with an attacker in range.
def test_safe_square_has_no_danger():
    board = _board((0, 0, 5), (3, 0, 19))
    values = _graded(board, 3, 0)
    assert values["attackers_after"] == 0.0
    assert values["capture_risk_after"] == 0.0


# Checks count_attackers counts distinct attacking tokens.
def test_count_attackers():
    board = _board((3, 0, 23), (2, 0, 34))
    assert count_attackers(board, 0, 12) == 2
    assert count_attackers(board, 0, 8) == 0


# Checks moving one of two stacked tokens is flagged as breaking a blockade.
def test_breaks_own_blockade():
    stacked = _board((0, 0, 10), (0, 1, 10))
    assert _graded(stacked, 3, 0)["breaks_own_blockade"] == 1.0
    assert _graded(_board((0, 0, 10), (0, 1, 20)), 3, 0)["breaks_own_blockade"] == 0.0


# Checks forming a blockade in front of an opponent is flagged, and not when nobody is behind.
def test_blocks_opponent():
    with_chaser = _board((0, 0, 8), (0, 1, 12), (3, 0, 23))
    assert _graded(with_chaser, 4, 0)["blocks_opponent"] == 1.0
    alone = _board((0, 0, 8), (0, 1, 12))
    assert _graded(alone, 4, 0)["blocks_opponent"] == 0.0


# Checks race standing lists progress then tokens-in-base, acting player's seat first.
def test_race_standing_is_egocentric():
    board = _board((1, 0, 29))
    from_seat_0 = encode_graded_features(board, 0, None)[-8:]
    from_seat_1 = encode_graded_features(board, 1, None)[-8:]
    assert from_seat_0[1] == pytest.approx(30 / (4 * 58))
    assert from_seat_1[0] == pytest.approx(30 / (4 * 58))
    assert from_seat_1[4] == pytest.approx(3 / 4)
    assert from_seat_0[4] == 1.0


# Checks the graded block is appended last and the spec size adds up.
def test_observation_wiring():
    spec = ObservationSpec(
        include_dice_roll=True, include_move_features=True, include_threat_features=True, include_graded_features=True
    )
    board = _board((0, 0, 10), (3, 0, 23))
    assert observation_size(spec) == 64 + 6 + 32 + 12 + 36
    observation = encode_observation(board, 0, 2, spec)
    assert observation.shape == (150,)
    assert np.array_equal(observation[-36:], encode_graded_features(board, 0, 2))
    assert observation_size(ObservationSpec(include_graded_features=True)) == 64 + 36


# Checks the config flag defaults off and reaches the observation spec.
def test_config_flag():
    assert not TrainingConfig(agent_type="dqn", seed=0, num_episodes=1).include_graded_features
    config = TrainingConfig(agent_type="dqn", seed=0, num_episodes=1, include_graded_features=True)
    assert config.observation_spec().include_graded_features


# Checks a DQN with the graded block builds and only picks legal moves.
def test_dqn_with_graded_features_acts_legally():
    spec = ObservationSpec(include_move_features=True, include_threat_features=True, include_graded_features=True)
    agent = DQNAgent(observation_spec=spec, rng=np.random.default_rng(0))
    board = _board((0, 0, 10), (0, 1, 20))
    assert agent(board, 0, 3, (0, 1)) in (0, 1)


# Checks the shared-token network refuses the graded block instead of mis-slicing the observation.
def test_shared_token_rejects_graded_features():
    with pytest.raises(ValueError):
        DQNAgent(observation_spec=ObservationSpec(include_graded_features=True), network_type="shared_token")
