"""Test configurations for the cybersecurity environment testing."""

import torch

from free_range_zoo.envs.cybersecurity.env.structures.configuration import (
    CybersecurityConfiguration,
    AttackerConfiguration,
    DefenderConfiguration,
    NetworkConfiguration,
    RewardConfiguration,
    StochasticConfiguration,
)


def non_stochastic():
    """Create a non-stochastic configuration for the cybersecurity environment."""
    attacker_config = AttackerConfiguration(
        initial_presence=torch.tensor([True, True], dtype=torch.bool),
        threat=torch.tensor([1.0, 1.0], dtype=torch.float32),
        persist_probs=torch.tensor([1.0, 1.0], dtype=torch.float32),
        return_probs=torch.tensor([1.0, 1.0], dtype=torch.float32),
    )

    defender_config = DefenderConfiguration(
        initial_location=torch.tensor([0, 1], dtype=torch.int32),
        initial_presence=torch.tensor([True, True], dtype=torch.bool),
        mitigation=torch.tensor([1.0, 1.0], dtype=torch.float32),
        persist_probs=torch.tensor([1.0, 1.0], dtype=torch.float32),
        return_probs=torch.tensor([1.0, 1.0], dtype=torch.float32),
    )

    network_config = NetworkConfiguration(
        patched_states=1,
        vulnerable_states=1,
        exploited_states=3,
        temperature=1.0,
        initial_state=torch.tensor([0, 0, 0], dtype=torch.int32),
        adj_matrix=torch.tensor([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=torch.bool),
    )

    reward_config = RewardConfiguration(
        bad_action_penalty=-100.0,
        patch_reward=0.0,
        network_state_rewards=torch.tensor([4.0, 0.0, -2.0, -4.0, -8.0], dtype=torch.float32),
    )

    stochastic_config = StochasticConfiguration(network_state=False)

    configuration = CybersecurityConfiguration(
        attacker_config=attacker_config,
        defender_config=defender_config,
        network_config=network_config,
        reward_config=reward_config,
        stochastic_config=stochastic_config,
    )

    return configuration
