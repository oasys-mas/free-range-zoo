"""Wrapper utilities for applying wrapper classes for environment output modification."""
from typing import List, Union, Dict, Any, Tuple

import gymnasium
from pettingzoo.utils import BaseParallelWrapper
from pettingzoo.utils.wrappers import OrderEnforcingWrapper as BaseWrapper
from pettingzoo.utils.env import ActionType
from supersuit.utils.wrapper_chooser import WrapperChooser

from free_range_zoo.utils.env import BatchedAECEnv

import torch


class shared_wrapper_aec(BaseWrapper):
    """Wrapper application utility for AEC environments."""

    def __init__(self, env: BatchedAECEnv, modifier_class: BaseWrapper, **kwargs):
        """
        Initialize the wrapper.

        Args:
            env: BatchedAECEnv - The environment to wrap
            modifier_class: BaseWrapper - The modifier class to use for modifying the environment output
        """
        super().__init__(env)

        self.modifier_class = modifier_class
        self.modifiers = {}
        self._cur_seed = None
        self._cur_options = None

        self.modifier_args = kwargs

        self.reset()

        if hasattr(self.env, "possible_agents"):
            self.add_modifiers(self.env.possible_agents)

    def observation_space(self, agent: str) -> gymnasium.Space:
        """
        Return the modified observation space for the agent.

        Args:
            agent: str - the agent for which the observation space is required
        Returns:
            gymnasium.Space: the modified observation space for the agent
        """
        return self.modifiers[agent].modify_obs_space(self.env.observation_space(agent))

    def action_space(self, agent: str) -> gymnasium.Space:
        """
        Return the modified action space for the agent.

        Args:
            agent: str - the agent for which the action space is required
        Returns:
            gymnasium.Space: the modified action space for the agent
        """
        return self.modifiers[agent].modify_action_space(self.env.action_space(agent))

    def add_modifiers(self, agents_list: List[str]) -> None:
        """
        Add the modifiers for the agents.

        Args:
            agents_list: List[str] - the list of agents for which the modifiers need to be added
        """
        for agent in agents_list:
            if agent not in self.modifiers:
                args = []
                if hasattr(self.modifier_class, 'env') and self.modifier_class.env:
                    args.append(self.env)
                if hasattr(self.modifier_class, 'subject_agent') and self.modifier_class.subject_agent:
                    args.append(agent)

                self.modifiers[agent] = self.modifier_class(*args, **self.modifier_args)

                self.observation_space(agent)
                self.action_space(agent)
                self.modifiers[agent].modify_obs(self.env.observe(agent))
                self.modifiers[agent].reset(seed=self._cur_seed, options=self._cur_options)

                # Modifiers for each agent has a different seed
                if self._cur_seed is not None:
                    self._cur_seed += 1

    def reset(self, seed: Union[List[int], int] = None, options: Dict[str, Any] = None) -> None:
        """
        Reset the environment.

        Args:
            seed: Union[List[int], int] - the seed for the environment
            options: Dict[str, Any] - the options for the environment
        """
        self._cur_seed = seed
        self._cur_options = options

        super().reset(seed=seed, options=options)

        for mod in self.modifiers.values():
            mod.reset(seed=seed, options=options)

        self.add_modifiers(self.agents)

        for agent in self.agents:
            self.modifiers[agent].modify_obs(self.env.observe(agent))

    def step(self, action: ActionType) -> None:
        """
        Step the environment for each individual agent.

        Args:
            action: ActionType - the action for the agent
        """
        mod = self.modifiers[self.agent_selection]
        action = mod.modify_action(action)

        super().step(action)
        self.add_modifiers(self.agents)

        # Step takes the last agent and makes it the first, so when we look for that last agent that really is the first
        if self.env.agent_selector.is_first():
            for agent in self.agents:
                self.modifiers[agent].modify_obs(self.env.observe(agent))

    def observe(self, agent: str) -> torch.Tensor:
        """
        Return the last observation for the agent.

        Args:
            agent: str - the agent for which the observation is required
        Returns:
            torch.Tensor: the last observation for the agent
        """
        return self.modifiers[agent].get_last_obs()


class shared_wrapper_parr(BaseParallelWrapper):
    """Wrapper application utility for parallel environments."""

    def __init__(self, env: BatchedAECEnv, modifier_class: BaseWrapper, **kwargs):
        """
        Initialize the wrapper.

        Args:
            env: BatchedAECEnv - The environment to wrap
            modifier_class: BaseWrapper - The modifier class to use for modifying the environment output
        """
        super().__init__(env)
        self.env.reset()

        self.modifier_class = modifier_class
        self.modifiers = {}
        self._cur_seed = None
        self._cur_options = None

        self.modifier_args = kwargs

        if hasattr(self.env, "possible_agents"):
            self.add_modifiers(self.env.possible_agents)

    def observation_space(self, agent: str) -> Dict[str, List[gymnasium.Space]]:
        """
        Return the modified observation space for the agent.

        Args:
            agent: str - the agent for which the observation space is required
        Returns:
            Dict[str, List[gymnasium.Space]]: the modified observation space for the agent
        """
        return self.modifiers[agent].modify_obs_space(self.env.observation_space(agent))

    def action_space(self, agent: str) -> List[gymnasium.Space]:
        """
        Return the modified action space for the agent.

        Args:
            agent: str - the agent for which the action space is required
        Returns:
            List[gymnasium.Space]: the modified action space for the agent
        """
        return self.modifiers[agent].modify_action_space(self.env.action_space(agent))

    def add_modifiers(self, agents_list: List[str]) -> None:
        """
        Add the modifiers for the agents.

        Args:
            agents_list: List[str] - the list of agents for which the modifiers need to be added
        """
        for agent in agents_list:
            if agent not in self.modifiers:
                args = []
                if hasattr(self.modifier_class, 'env') and self.modifier_class.env:
                    args.append(self.env)
                if hasattr(self.modifier_class, 'subject_agent') and self.modifier_class.subject_agent:
                    args.append(agent)

                self.modifiers[agent] = self.modifier_class(*args, **self.modifier_args)

                self.observation_space(agent)
                self.action_space(agent)

                self.modifiers[agent].reset(seed=self._cur_seed, options=self._cur_options)

                # Modifiers for each agent has a different seed
                if self._cur_seed is not None:
                    self._cur_seed += 1

    def reset(self,
              seed: Union[int, List[int]] = None,
              options: Dict[str, Any] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Reset the environment and returns the initial observations.

        Args:
            seed: Union[int, List[int]] - The seed for the environment
            options: Dict[str, Any] - The options for the environment
        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]] - The initial observations and infos
        """
        self._cur_seed = seed
        self._cur_options = options

        observations, infos = super().reset(seed=seed, options=options)
        self.add_modifiers(self.agents)

        observations = {agent: self.modifiers[agent].modify_obs(obs) for agent, obs in observations.items()}

        for agent, mod in self.modifiers.items():
            mod.reset(seed=seed, options=options)

        return observations, infos

    def step(
        self, actions: Dict[str, ActionType]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str,
                                                                                                                        Any]]:
        """
        Step the environment for all agents.

        Args:
            actions: Dict[str, ActionType] - The actions for each agent
        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor],
                  Dict[str, Dict]] - The observations, rewards, terminations, truncations, and infos
        """
        actions = {agent: self.modifiers[agent].modify_action(action) for agent, action in actions.items()}

        observations, rewards, terminations, truncations, infos = super().step(actions)
        self.add_modifiers(self.agents)

        observations = {agent: self.modifiers[agent].modify_obs(obs) for agent, obs in observations.items()}

        return observations, rewards, terminations, truncations, infos


class shared_wrapper_gym(gymnasium.Wrapper):
    """Wrapper application utility for gym environments."""

    def __init__(self, env: gymnasium.Env, modifier_class: BaseWrapper):
        """
        Initialize the wrapper.

        Args:
            env: gymnasium.Env - The environment to wrap
            modifier_class: BaseWrapper - The modifier class to use for modifying the environment output
        """
        super().__init__(env)

        self.modifier = modifier_class()

        self.observation_space = self.modifier.modify_obs_space(self.observation_space)
        self.action_space = self.modifier.modify_action_space(self.action_space)

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Reset the environment and returns the initial observations.

        Args:
            seed: int - The seed for the environment
            options: Dict[str, Any] - The options for the environment
        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]] - The initial observations, infos
        """
        self.modifier.reset(seed=seed, options=options)
        obs, info = super().reset(seed=seed, options=options)
        obs = self.modifier.modify_obs(obs)
        return obs, info

    def step(
        self, action: Dict[str, ActionType]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str,
                                                                                                                        Any]]:
        """
        Step the environment.

        Args:
            action: Dict[str, ActionType] - The action for the agent
        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor],
                  Dict[str, Any]] - The observations, rewards, terminations, truncations, and infos
        """
        obs, rew, term, trunc, info = super().step(self.modifier.modify_action(action))
        obs = self.modifier.modify_obs(obs)
        return obs, rew, term, trunc, info


shared_wrapper = WrapperChooser(
    aec_wrapper=shared_wrapper_aec,
    gym_wrapper=shared_wrapper_gym,
    parallel_wrapper=shared_wrapper_parr,
)
