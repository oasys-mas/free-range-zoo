import torch
import numpy as np

from free_range_zoo.free_range_zoo.envs.cybersecurity.env.structures.configuration import (CybersecurityConfiguration,
                                                                                           AttackerConfiguration,
                                                                                           DefenderConfiguration,
                                                                                           NetworkConfiguration)


def non_stochastic():
    """Create a non-stochastic configuration for the cybersecurity environment."""
    attacker_config = AttackerConfiguration(initial_location=torch.tensor([0, 1], dtype=torch.int32),
                                            threat=torch.tensor([1.0, 1.0], dtype=torch.float32),
                                            persist=torch.tensor([1.0, 1.0], dtype=torch.float32),
                                            returns=torch.tensor([1.0, 1.0], dtype=torch.float32))

    defender_config = DefenderConfiguration(initial_location=torch.tensor([0, 1], dtype=torch.int32),
                                            initial_presence=torch.tensor([True, True], dtype=torch.bool),
                                            mitigation=torch.tensor([1.0, 1.0], dtype=torch.float32),
                                            sigma_away=0.1,
                                            sigma_at=0.1)

    network_config = NetworkConfiguration(patched_states=1,
                                          vulnerable_states=2,
                                          exploited_states=3,
                                          danger_chi=1.0,
                                          latency=torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32),
                                          subnetworks=torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.bool))

    configuration = CybersecurityConfiguration(attacker_config=attacker_config,
                                               defender_config=defender_config,
                                               network_config=network_config)

    return configuration
