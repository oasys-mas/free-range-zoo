"""Utility for converting between different environment types."""
from typing import Dict, Tuple, Any, List, Union

from pettingzoo.utils.conversions import aec_to_parallel_wrapper
from pettingzoo.utils.env import ParallelEnv, AgentID, ObsType, ActionType
from pettingzoo.utils.wrappers import OrderEnforcingWrapper
import torch
from tensordict import TensorDict

from free_range_zoo.utils.env import BatchedAECEnv


def batched_aec_to_batched_parallel(aec_env: BatchedAECEnv) -> ParallelEnv[AgentID, ObsType, ActionType]:
    """
    Convert a batched AEC environment to a batched parallel environment.

    In the case of an existing batched parallel environment wrapped using a `parallel_to_aec_wrapper`,
    this function will return the original environment Otherwise, it will apply the `aec_to_paralel_wrapper`
    to convert the environment

    Args:
        aec_env: AECEnv - The environment to convert
    Returns:
        ParallelEnv: The parallel environment
    """
    if isinstance(aec_env, OrderEnforcingWrapper) and isinstance(aec_env.env, batched_aec_to_batched_parallel_wrapper):
        return aec_env.env.env
    else:
        return batched_aec_to_batched_parallel_wrapper(aec_env)


class batched_aec_to_batched_parallel_wrapper(aec_to_parallel_wrapper):
    """Convert a single agent environment to a parallel environment."""

    def reset_batches(self, *args, **kwargs):
        """Reset specific batches of the environment."""
        self.aec_env.reset_batches(*args, **kwargs)

    def reset(self,
              seed: Union[int, List[int]] = None,
              options: Dict[str, Any] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Reset the environment and returns the initial observations.

        Args:
            seed: Union[int, List[int]] -  The seed for the environment
            options: Dict[str, Any] - The options for the environment
        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]] - The initial observations and infos
        """
        self.aec_env.reset(seed=seed, options=options)
        self.agents = self.aec_env.agents

        observations = {agent: self.aec_env.observe(agent) for agent in self.agents}

        infos = self.aec_env.infos
        return observations, infos

    def step(
        self, actions: Dict[str, ActionType]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str,
                                                                                                                        Dict]]:
        """
        Modify step function to handle parallel environments.

        Args:
            actions: dict - The actions for each agent
        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor],
                  Dict[str, Dict]] - The observations, rewards, terminations, truncations, and infos
        """
        terminations = {
            agent: torch.zeros(self.aec_env.parallel_envs, dtype=torch.bool, device=self.aec_env.device)
            for agent in self.aec_env.agents
        }
        truncations = {
            agent: torch.zeros(self.aec_env.parallel_envs, dtype=torch.bool, device=self.aec_env.device)
            for agent in self.aec_env.agents
        }
        rewards = {
            agent: torch.zeros(self.aec_env.parallel_envs, dtype=torch.float32, device=self.aec_env.device)
            for agent in self.aec_env.agents
        }
        observations = {}
        infos = {}

        for agent in self.aec_env.agents:
            self.aec_env.step(actions[agent])
            for agent in self.aec_env.agents:
                rewards[agent] += self.aec_env.rewards[agent]

        terminations = self.aec_env.terminations
        truncations = self.aec_env.truncations
        infos = self.aec_env.infos
        observations = {agent: self.aec_env.observe(agent) for agent in self.aec_env.agents}

        self.agents = self.aec_env.agents

        return observations, rewards, terminations, truncations, infos

    def observe(self) -> Dict[str, TensorDict]:
        """
        Observe the environment.

        Returns:
            Dict[str, TensorDict] - The observations for each agent
        """
        return {agent: self.aec_env.observe(agent) for agent in self.aec_env.agents}

    @property
    def finished(self) -> torch.Tensor:
        """
        Return a boolean tensor indicating which environments have finished.

        Returns:
            torch.Tensor - The tensor indicating which environments have finished
        """
        return self.aec_env.finished
