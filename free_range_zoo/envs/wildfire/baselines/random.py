"""
Random action baseline agent for wildfire environments.

This module provides a baseline agent that samples actions uniformly at random from the available action space
in the wildfire environment. The agent serves as a stochastic benchmark for evaluating the effectiveness of more
sophisticated wildfire suppression strategies.
"""
import free_range_rust
import torch

from free_range_zoo.utils.agent import Agent


class RandomBaseline(Agent):
    """
    Select actions uniformly at random in the wildfire environment.

    This baseline agent samples actions from the available action space for each parallel environment using a uniform
    distribution. Each action is chosen independently. Useful for benchmarking or as a stochastic control policy.
    """

    def act(self, action_space: free_range_rust.Space) -> torch.IntTensor:
        """
        Sample and return a list of random actions for each parallel environment.

        Args:
            action_space: free_range_rust.Space - The current action space available to the agent.
        Returns:
            torch.IntTensor: Tensor of actions, one for each parallel environment.
        """
        seed = torch.randint(0, (2**63) - 2, (1, ), generator=self.generator, dtype=torch.uint64).item()
        return torch.tensor(action_space.sample_nested_with_seed(seed), dtype=torch.int32)
