"""BatchedAECEnv class for batched environments."""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any, Optional
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import gymnasium
import torch
from tensordict import TensorDict
import pandas as pd
import os

from free_range_zoo.utils.configuration import Configuration
from free_range_zoo.utils.state import State
from free_range_zoo.utils.random_generator import RandomGenerator


class BatchedAECEnv(ABC, AECEnv):
    """Pettingzoo environment for adapter for batched environments."""

    def __init__(
        self,
        *args,
        configuration: Configuration = None,
        max_steps: int = 1,
        parallel_envs: int = 1,
        device: torch.DeviceObjType = torch.device('cpu'),
        render_mode: str | None = None,
        log_directory: str = None,
        single_seeding: bool = False,
        buffer_size: int = 0,
        override_initialization_check: bool = False,
        **kwargs,
    ):
        """
        Initialize the environment.

        Args:
            configuration: Configuration - the configuration for the environment
            max_steps: int - the maximum number of steps to take in the environment
            parallel_envs: int - the number of parallel environments to run
            device: torch.DeviceObjType - the device to run the environment on
            render_mode: str | None - the mode to render the environment in
            log_directory: str - the directory to log the environment to
            single_seeding: bool - whether to seed all parallel environments with the same seed
            buffer_size: int - the size of the buffer for random number generation
            override_initialization_check: bool - whether to override throwing logs being rewritten on init
        """
        super().__init__(*args, **kwargs)
        self.parallel_envs = parallel_envs
        self.max_steps = max_steps
        self.device = device
        self.render_mode = render_mode
        self.log_directory = log_directory
        self.single_seeding = single_seeding
        self.log_description = None

        if configuration is not None:
            self.config = configuration.to(device)

            for key, value in configuration.__dict__.items():
                if isinstance(value, Configuration):
                    setattr(self, key, value)

        self._are_logs_initialized = False
        if not override_initialization_check and self.log_directory is not None and os.path.exists(self.log_directory):
            if os.listdir(self.log_directory):
                raise FileExistsError("The logging output directory already exists. Set override_initialization_check or rename.")
        if self.log_directory is not None and not os.path.exists(self.log_directory):
            os.mkdir(self.log_directory)

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

        # Reset the log label if one has been provided, None otherwise
        self.log_description = None
        if options and options.get('log_description'):
            self.log_description = options.get('log_description')

        # Initial environment AEC attributes
        self.agents = self.possible_agents
        self.rewards = {agent: torch.zeros(self.parallel_envs, dtype=torch.float32, device=self.device) for agent in self.agents}
        self._cumulative_rewards = {agent: torch.zeros(self.parallel_envs, device=self.device) for agent in self.agents}
        self.terminations = {
            agent: torch.zeros(self.parallel_envs, dtype=torch.bool, device=self.device)
            for agent in self.agents
        }
        self.truncations = {agent: torch.zeros(self.parallel_envs, dtype=torch.bool, device=self.device) for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

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

    @torch.no_grad()
    def reset_batches(
        self,
        batch_indices: List[int],
        seed: Optional[List[int]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
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

    @torch.no_grad()
    def _post_reset_hook(self):
        """Actions to take after each environment has globally handled all reset functionality."""
        if self.log_directory is not None:
            self._log_environment(reset=True)

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
        """
        # Handle stepping an agent which is completely dead
        if torch.all(self.terminations[self.agent_selection]) or torch.all(self.truncations[self.agent_selection]):
            return

        self._clear_rewards()
        agent = self.agent_selection
        self.actions[agent] = actions

        # Step the environment after all agents have taken actions
        if self.agent_selector.is_last():
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

            if self.log_directory is not None:
                self._log_environment()

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

    def _log_environment(self, reset: bool = False, extra: pd.DataFrame = None) -> None:
        """
        Log all portions of the environment.

        Args:
            reset: bool - Indicates whether the source of the logging is due to a reset. N/A flag on step-specific items.
            extra: pd.DataFrame = Additional data to append to the environment logs.
        """
        if extra is not None and len(extra) != self.parallel_envs:
            raise ValueError('The number of elements in extras must match the number of parallel environments.')

        df = self._state.to_dataframe()
        if reset:
            for agent in self.possible_agents:
                df[[f'{agent}_action', f'{agent}_rewards']] = None
                df[f'{agent}_action_map'] = [str(mapping.tolist()) for mapping in self.agent_action_mapping[agent]]
                df[f'{agent}_observation_map'] = [str(mapping.tolist()) for mapping in self.agent_observation_mapping[agent]]

            df['step'] = -1
            df['complete'] = None
        else:
            for agent in self.possible_agents:
                df[f'{agent}_action'] = [str(action) for action in self.actions[agent].cpu().tolist()]
                df[f'{agent}_rewards'] = self.rewards[agent].cpu()
                df[f'{agent}_action_map'] = [str(mapping.tolist()) for mapping in self.agent_action_mapping[agent]]
                df[f'{agent}_observation_map'] = [str(mapping.tolist()) for mapping in self.agent_observation_mapping[agent]]

            df['step'] = self.num_moves.cpu()
            df['complete'] = self.finished.cpu()

        if extra is not None:
            df = pd.concat([df, extra], axis=1)

        df['description'] = self.log_description

        for i in range(self.parallel_envs):
            df.iloc[[i]].to_csv(
                os.path.join(self.log_directory, f'{i}.csv'),
                mode='w' if not self._are_logs_initialized else 'a',
                header=not self._are_logs_initialized,
                index=False,
                na_rep="NULL",
            )
        self._are_logs_initialized = True

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
