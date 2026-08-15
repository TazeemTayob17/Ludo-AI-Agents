# Tests for agents/replay_buffer.py (Step 8.2), independent of the environment.

import numpy as np

from agents.replay_buffer import ReplayBuffer


def _dummy_transition(value):
    state = np.full(4, value, dtype=np.float32)
    next_state = np.full(4, value + 1, dtype=np.float32)
    mask = np.array([True, False, True, False])
    next_mask = np.array([False, True, False, True])
    return state, value % 4, float(value), next_state, False, mask, next_mask


# Checks length starts at zero and grows with each push.
def test_len_reflects_number_of_pushed_transitions():
    buffer = ReplayBuffer(capacity=10)
    assert len(buffer) == 0
    buffer.push(*_dummy_transition(0))
    buffer.push(*_dummy_transition(1))
    assert len(buffer) == 2


# Checks the buffer discards the oldest transitions once it exceeds capacity.
def test_buffer_respects_its_capacity():
    buffer = ReplayBuffer(capacity=3)
    for i in range(5):
        buffer.push(*_dummy_transition(i))
    assert len(buffer) == 3


# Checks sample returns correctly shaped and typed arrays for every field, including
# the action mask for both s and s'.
def test_sample_returns_correctly_shaped_batches():
    buffer = ReplayBuffer(capacity=10, rng=np.random.default_rng(0))
    for i in range(10):
        buffer.push(*_dummy_transition(i))

    states, actions, rewards, next_states, dones, masks, next_masks = buffer.sample(4)
    assert states.shape == (4, 4)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
    assert next_states.shape == (4, 4)
    assert dones.shape == (4,)
    assert masks.shape == (4, 4)
    assert next_masks.shape == (4, 4)
    assert states.dtype == np.float32
    assert actions.dtype == np.int64
    assert masks.dtype == bool


# Checks sampling is reproducible given a seeded generator.
def test_sample_is_deterministic_given_a_seed():
    buffer_a = ReplayBuffer(capacity=10, rng=np.random.default_rng(7))
    buffer_b = ReplayBuffer(capacity=10, rng=np.random.default_rng(7))
    for i in range(10):
        buffer_a.push(*_dummy_transition(i))
        buffer_b.push(*_dummy_transition(i))

    batch_a = buffer_a.sample(5)
    batch_b = buffer_b.sample(5)
    for field_a, field_b in zip(batch_a, batch_b):
        assert np.array_equal(field_a, field_b)
