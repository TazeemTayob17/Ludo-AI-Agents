# Tests for agents/dqn_agent.py: sync (8.3), masked action selection (8.4), the masked
# Bellman target (8.5), loss/training convergence (8.6), and Double DQN (8.7).

import numpy as np
import pytest
import torch

from agents.dqn_agent import DQNAgent
from env.state_encoding import OBSERVATION_SIZE

_OBS_SIZE = OBSERVATION_SIZE


# Forces a network to output a fixed vector for any input, for hand-constructed transitions.
def _set_constant_output(network, output_vector):
    with torch.no_grad():
        network.fc1.weight.zero_()
        network.fc1.bias.zero_()
        network.fc2.weight.zero_()
        network.fc2.bias.zero_()
        network.fc3.weight.zero_()
        network.fc3.bias.copy_(torch.tensor(output_vector, dtype=torch.float32))


# Checks the target network only updates when sync_target_network is called.
def test_sync_target_network_copies_online_weights():
    agent = DQNAgent()
    _set_constant_output(agent.online_network, [1.0, 2.0, 3.0, 4.0])
    online_before = agent.online_network(torch.zeros((1, _OBS_SIZE)))
    target_before = agent.target_network(torch.zeros((1, _OBS_SIZE)))
    assert not torch.allclose(online_before, target_before)

    agent.sync_target_network()
    target_after = agent.target_network(torch.zeros((1, _OBS_SIZE)))
    assert torch.allclose(online_before, target_after)


# Checks epsilon=0 always picks the highest-value LEGAL action, even when a masked-out
# action has a higher raw value.
def test_greedy_selection_never_picks_a_masked_out_action():
    agent = DQNAgent(epsilon=0.0, rng=np.random.default_rng(0))
    _set_constant_output(agent.online_network, [1.0, 100.0, 2.0, 9.0])
    observation = np.zeros(_OBS_SIZE, dtype=np.float32)
    mask = np.array([True, False, True, True])
    action = agent.select_action(observation, mask)
    assert action == 3  # highest among the legal set {0, 2, 3}; token 1 is illegal despite its 100.0


# Checks epsilon=1 always explores, still respecting the legal action set.
def test_fully_exploratory_selection_still_respects_the_mask():
    agent = DQNAgent(epsilon=1.0, rng=np.random.default_rng(0))
    observation = np.zeros(_OBS_SIZE, dtype=np.float32)
    mask = np.array([False, True, False, True])
    for _ in range(100):
        assert agent.select_action(observation, mask) in (1, 3)


# Checks the plain-DQN target masks the next state's max over legal actions only.
def test_plain_dqn_target_masks_next_state_max():
    agent = DQNAgent(double_dqn=False, gamma=0.9)
    _set_constant_output(agent.target_network, [3.0, 0.5, 4.0, 0.0])
    next_mask = np.array([[True, True, True, False]])  # token 3 illegal

    target = agent.compute_targets(
        rewards=np.array([1.0], dtype=np.float32),
        next_states=np.zeros((1, _OBS_SIZE), dtype=np.float32),
        next_action_masks=next_mask,
        dones=np.array([0.0], dtype=np.float32),
    )
    assert target.item() == pytest.approx(1.0 + 0.9 * 4.0)  # max of the legal {3.0, 0.5, 4.0}


# Checks a terminal transition's target is the reward alone, ignoring the next state entirely.
def test_target_for_a_done_transition_ignores_next_state():
    agent = DQNAgent(double_dqn=False, gamma=0.9)
    _set_constant_output(agent.target_network, [999.0, 999.0, 999.0, 999.0])  # would dominate if used
    next_mask = np.array([[True, True, True, True]])

    target = agent.compute_targets(
        rewards=np.array([2.0], dtype=np.float32),
        next_states=np.zeros((1, _OBS_SIZE), dtype=np.float32),
        next_action_masks=next_mask,
        dones=np.array([1.0], dtype=np.float32),
    )
    assert target.item() == pytest.approx(2.0)


# Checks a next state with zero legal actions degrades to zero future value, not NaN/inf.
def test_target_with_no_legal_next_action_is_treated_as_zero_future_value():
    agent = DQNAgent(double_dqn=False, gamma=0.9)
    _set_constant_output(agent.target_network, [5.0, 5.0, 5.0, 5.0])
    next_mask = np.array([[False, False, False, False]])

    target = agent.compute_targets(
        rewards=np.array([1.5], dtype=np.float32),
        next_states=np.zeros((1, _OBS_SIZE), dtype=np.float32),
        next_action_masks=next_mask,
        dones=np.array([0.0], dtype=np.float32),
    )
    assert target.item() == pytest.approx(1.5)


# Checks the Double DQN target (8.7) diverges from the plain-DQN target on a hand-constructed
# transition where the online network's argmax differs from the target network's argmax.
def test_double_dqn_target_diverges_from_plain_dqn_target():
    agent = DQNAgent(double_dqn=True, gamma=1.0)
    _set_constant_output(agent.online_network, [1.0, 5.0, 2.0, 0.0])  # argmax among legal: token 1
    _set_constant_output(agent.target_network, [3.0, 0.5, 4.0, 0.0])  # argmax among legal: token 2
    next_mask = np.array([[True, True, True, False]])
    rewards = np.array([0.0], dtype=np.float32)
    dones = np.array([0.0], dtype=np.float32)
    next_states = np.zeros((1, _OBS_SIZE), dtype=np.float32)

    double_target = agent.compute_targets(rewards, next_states, next_mask, dones)
    agent.double_dqn = False
    plain_target = agent.compute_targets(rewards, next_states, next_mask, dones)

    assert plain_target.item() == pytest.approx(4.0)  # target network's own best legal value
    assert double_target.item() == pytest.approx(0.5)  # target network's value AT the online network's pick
    assert double_target.item() != pytest.approx(plain_target.item())


# Checks training actually drives Q-values toward known-correct values on a tiny, fully
# deterministic 2-action MDP (Step 8.6's numerical toy-MDP check).
def test_training_converges_on_a_toy_two_action_mdp():
    agent = DQNAgent(gamma=0.0, learning_rate=0.05, epsilon=0.0, rng=np.random.default_rng(0))
    state = np.zeros(_OBS_SIZE, dtype=np.float32)
    mask = np.array([True, True, False, False])

    for _ in range(500):
        batch = (
            np.stack([state, state]),
            np.array([0, 1]),
            np.array([1.0, -1.0], dtype=np.float32),  # action 0 always earns 1.0, action 1 always earns -1.0
            np.stack([state, state]),
            np.array([1.0, 1.0], dtype=np.float32),  # episode ends immediately either way
            np.stack([mask, mask]),
            np.stack([mask, mask]),
        )
        agent.train_on_batch(batch)

    values = agent.q_values(state)
    assert values[0] == pytest.approx(1.0, abs=0.1)
    assert values[1] == pytest.approx(-1.0, abs=0.1)
