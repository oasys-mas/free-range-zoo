"""
Random action baseline agent for cybersecurity environments.

This module provides a baseline agent that samples actions uniformly at random from the available action space,
serving as a stochastic benchmark for agent performance in cybersecurity tasks.
"""
import free_range_rust
import torch

from free_range_zoo.utils.agent import Agent


class RandomBaseline(Agent):
    """
    Agent that samples actions available to it in a uniform distribution.

    This agent samples actions uniformly at random from the available action space at each step. It serves as a
    stochastic baseline for evaluating the effectiveness of more sophisticated cybersecurity agents.
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
