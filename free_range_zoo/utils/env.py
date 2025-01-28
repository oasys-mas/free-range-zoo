"""BatchedAECEnv class for batched environments."""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any, Optional

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import gymnasium
import torch
from tensordict import TensorDict

from free_range_zoo.utils.configuration import Configuration
from free_range_zoo.utils.state import State
from free_range_zoo.utils.random_generator import RandomGenerator


class BatchedAECEnv(ABC, AECEnv):
    """Pettingzoo environment for adapter for batched environments."""

    def __init__(self,
                 *args,
                 configuration: Configuration = None,
                 max_steps: int = 1,
                 parallel_envs: int = 1,
                 device: torch.DeviceObjType = torch.device('cpu'),
                 render_mode: str | None = None,
                 log_dir: str = None,
                 single_seeding: bool = False,
                 buffer_size: int = 0,
                 **kwargs):
        """
        Initialize the environment.

        Args:
            configuration: Configuration - the configuration for the environment
            max_steps: int - the maximum number of steps to take in the environment
            parallel_envs: int - the number of parallel environments to run
            device: torch.DeviceObjType - the device to run the environment on
            render_mode: str | None - the mode to render the environment in
            log: bool - whether to log the environment
            log_dir: str - the directory to log the environment to
            single_seeding: bool - whether to seed all parallel environments with the same seed
            buffer_size: int - the size of the buffer for random number generation
        """
        super().__init__(*args, **kwargs)
        self.parallel_envs = parallel_envs
        self.max_steps = max_steps
        self.device = device
        self.render_mode = render_mode
        self.log_dir = log_dir
        self.is_logging = log_dir is not None
        self.single_seeding = single_seeding
        self.is_new_environment = True

        if configuration is not None:
            self.config = configuration.to(device)

            for key, value in configuration.__dict__.items():
                if isinstance(value, Configuration):
                    setattr(self, key, value)

        # Checks if any environments reset for logging purposes
        self._any_reset = None

        # Default logging param
        self.constant_observations = []
        self.log_exclusions = []

        self.generator = RandomGenerator(
            parallel_envs=parallel_envs,
            buffer_size=buffer_size,
            single_seeding=single_seeding,
            device=device,
        )

    @torch.no_grad()
    def reset(self, seed: Optional[List[int]] = None, options: Optional[Dict[str, Any]] = None) -> None:
        """
        Reset the environment.

        Args:
            seed: Union[List[int], None] - the seed to use
            options: Dict[str, Any] - the options for the reset
        """
        # Allow for a custom horizon to be set for the environment
        if options is not None and options.get('max_steps') is not None:
            self.max_steps = options['max_steps']

        if options is not None and options.get('log_label') is not None:
            self._log_label = options['log_label']
        else:
            self._log_label = None

        # Set seeding if given (prepares for the next random number generation i.e. self._make_randoms())
        self.seeds = torch.zeros((self.parallel_envs), dtype=torch.int32, device=self.device)

        # Make sure that generator has been initialized if we're calling skip seeding
        if options and options.get('skip_seeding'):
            if not hasattr(self.generator, 'seeds') or not hasattr(self.generator, 'generator_states'):
                raise ValueError("Seed must be set before skipping seeding is possible")

        # Seed the environment if we aren't skipping seeding
        if not options or not options.get('skip_seeding'):
            self.generator.seed(seed, partial_seeding=None)

        # Initial environment AEC attributes
        self.agents = self.possible_agents
        self.rewards = {agent: torch.zeros(self.parallel_envs, dtype=torch.float32, device=self.device) for agent in self.agents}
        self._cumulative_rewards = {agent: torch.zeros(self.parallel_envs, device=self.device) for agent in self.agents}
        self.terminations = {
            agent: torch.zeros(self.parallel_envs, dtype=torch.bool, device=self.device)
            for agent in self.agents
        }
        self.truncations = {agent: torch.zeros(self.parallel_envs, dtype=torch.bool, device=self.device) for agent in self.agents}

        # Task-action-index-map identifies the global <-> local task indices for environments
        # This is used when the availability of actions/tasks is not uniform across agents & environments for logging
        self.infos = {agent: {'task-action-index-map': [None for _ in range(self.parallel_envs)]} for agent in self.agents}

        # Dictionary storing actions for each agent
        self.actions = {
            agent: torch.empty((self.parallel_envs, 2), dtype=torch.int32, device=self.device)
            for agent in self.agents
        }

        self.num_moves = torch.zeros(self.parallel_envs, dtype=torch.int32, device=self.device)

        # Initialize AEC agent selection
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.reset()

        # Intialize the mapping of the tasks "in" the environment, used to map actions
        self.environment_task_count = torch.empty((self.parallel_envs, ), dtype=torch.int32, device=self.device)
        self.agent_task_count = torch.empty((self.num_agents, self.parallel_envs), dtype=torch.int32, device=self.device)

        self._any_reset = True

    @torch.no_grad()
    def reset_batches(self,
                      batch_indices: List[int],
                      seed: Optional[List[int]] = None,
                      options: Optional[Dict[str, Any]] = None) -> None:
        """
        Reset a batch of environments.

        Args:
            batch_indices: List[int] - The batch indices to reset
            seed: Optional[List[int]] - the seeds to use for the reset
            options: Optional[Dict[str, Any]] - the options for the reset
        """
        self.generator.seed(seed, partial_seeding=batch_indices)

        if options is not None and options.get('log_label') is not None:
            self._log_label = options['log_label']

        for agent in self.agents:
            self.rewards[agent][batch_indices] = 0
            self._cumulative_rewards[agent][batch_indices] = 0
            self.terminations[agent][batch_indices] = False
            self.truncations[agent][batch_indices] = False
            self.actions[agent][batch_indices] = torch.empty(2, dtype=torch.int32, device=self.device)

        self.num_moves[batch_indices] = 0

        self._any_reset = batch_indices

    @abstractmethod
    @torch.no_grad()
    def step_environment(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Dict[str, bool]]]:
        """Simulatenous step of the entire environment."""
        raise NotImplementedError('This method should be implemented in the subclass')

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> None:
        """
        Take a step in the environment.

        Args:
            actions: torch.Tensor - The actions to take in the environment
            log_label: Optional[str] - A additional label added as a column to the log file if logging is enabled
        """
        # Handle stepping an agent which is completely dead
        if torch.all(self.terminations[self.agent_selection]) or torch.all(self.truncations[self.agent_selection]):
            return

        # Reset logging, logs if any batches reset
        if self._any_reset and self.is_logging:
            self._state.log(path=self.log_dir,
                            new_episode=True,
                            constant_observations=self.constant_observations,
                            initial=self.is_new_environment,
                            label=self._log_label,
                            partial_log=self._any_reset,
                            actions=self.agents,
                            log_exclusions=self.log_exclusions,
                            rewards=self.rewards,
                            infos=self.infos)

            # Flip tags
            self._any_reset = None
            self.is_new_environment = False

        self._clear_rewards()
        agent = self.agent_selection
        self.actions[agent] = actions

        is_last = self.agent_selector.is_last()

        # Step the environment after all agents have taken actions
        if is_last:
            # Handle the stepping of the environment itself and update the AEC attributes
            rewards, terminations, infos = self.step_environment()
            self.rewards = rewards
            self.terminations = terminations
            self.infos = infos

            # Increment the number of steps taken in each batched environment
            self.num_moves += 1

            if self.max_steps is not None:
                is_truncated = self.num_moves >= self.max_steps
                for agent in self.agents:
                    self.truncations[agent] = is_truncated

            self._accumulate_rewards()
            self.update_observations()
            self.update_actions()

            # Log the new state of the environment
            if self.is_logging and is_last:
                self._state.log(path=self.log_dir,
                                new_episode=False,
                                constant_observations=self.constant_observations,
                                initial=False,
                                label=self._log_label,
                                actions=self.actions,
                                rewards=self.rewards,
                                log_exclusions=self.log_exclusions,
                                infos=self.infos)

        self.agent_selection = self.agent_selector.next()

    @torch.no_grad()
    def _accumulate_rewards(self) -> None:
        """Accumulate environmental rewards while taking into account parallel environments."""
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]

    @torch.no_grad()
    def _clear_rewards(self) -> None:
        """Clear environmental rewards while taking into account parallel environments."""
        for agent in self.rewards:
            self.rewards[agent] = torch.zeros(self.parallel_envs, dtype=torch.float32, device=self.device)

    @abstractmethod
    def update_actions(self) -> None:
        """Update tasks in the environment for the next step and renew agent action-task mappings."""
        raise NotImplementedError('This method should be implemented in the subclass')

    @abstractmethod
    def update_observations(self) -> None:
        """Update observations for the next step and update observation space."""
        raise NotImplementedError('This method should be implemented in the subclass')

    @torch.no_grad()
    def observe(self, agent: str) -> TensorDict:
        """
        Return the current observations for this agent.

        Args:
            agent (str): the name of the agent to retrieve the observations for
        Returns:
            TensorDict: the observations for the given agent
        """
        return self.observations[agent]

    @abstractmethod
    @torch.no_grad()
    def action_space(self, agent: str) -> List[gymnasium.Space]:
        """
        Return the action space for the given agent.

        Args:
            agent (str): the name of the agent to retrieve the action space for
        Returns:
            List[gymnasium.Space]: the action space for the given agent
        """
        raise NotImplementedError('This method should be implemented in the subclass')

    @abstractmethod
    @torch.no_grad()
    def observation_space(self, agent: str) -> List[gymnasium.Space]:
        """
        Return the observation space for the given agent.

        Args:
            agent (str): the name of the agent to retrieve the observation space for
        Returns:
            List[gymnasium.Space]: the observation space for the given agent
        """
        raise NotImplementedError('This method should be implemented in the subclass')

    @torch.no_grad()
    def state(self) -> State:
        """
        Return the current state of the environment.

        Returns:
            WildfireState: the current state of the environment
        """
        return self._state

    @property
    def finished(self) -> torch.Tensor:
        """
        Return a boolean tensor indicating which environments have finished.

        Returns:
            torch.Tensor - The tensor indicating which environments have finished
        """
        return torch.logical_or(self.terminated, self.truncated)

    @property
    def terminated(self) -> torch.Tensor:
        """
        Return a boolean tensor indicating which environments have terminated.

        Returns:
            torch.Tensor - The tensor indicating which environments have terminated
        """
        return torch.all(torch.stack([self.terminations[agent] for agent in self.agents]), dim=0)

    @property
    def truncated(self) -> torch.Tensor:
        """
        Return a boolean tensor indicating which environments have been truncated.

        Returns:
            torch.Tensor - The tensor indicating which environments have been truncated
        """
        return torch.all(torch.stack([self.truncations[agent] for agent in self.agents]), dim=0)
