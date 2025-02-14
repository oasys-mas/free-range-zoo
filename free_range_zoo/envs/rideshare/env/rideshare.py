"""
# Rideshare
<hr>

| Import             | `from freerangezoo.otask import rideshare_v0` |
|--------------------|------------------------------------|
| Actions            | Discrete and perfect                            |
| Observations | Discrete and fully observed with private observations |
| Parallel API       | Yes                                |
| Manual Control     | No                                 |
| Agent Names             | [$driver$_0, ..., $driver$_n] |
| #Agents             |    $n$                                  |
| Action Shape       | (envs, 2)                 |
| Action Values      | [-1, $\|tasks\|$], [-1,2]\*                    |
| Observation Shape | TensorDict: { <br> &emsp; **Agent's self obs**, <ins>'self'</ins>: 5 `<agent index, ypos, xpos, #accepted passengers, #riding passengers>`, <br> &emsp; **Other agent obs**, <ins>'others'</ins>: ($\|Ag\| \times 5$) `<agent index, ypos, xpos, #accepted passengers, #riding passengers>`, <br> &emsp; **Fire/Task obs**, <ins>'tasks'</ins>: ($\|X\| \times 5$) `<task index, ystart, xstart, yend, xend, acceptedBy, ridingWith, fare, time entered>` <br> **batch_size: `num_envs`** <br>}|
| Observation Values   | <ins>self</ins> <br> **agent index**: [0,n), <br> **ypos**: [0,grid_height], <br> **xpos**: [0, grid_width], <br> **number accepted passengers**: [0, $\infty$), <br> **number riding passengers**: [0,$\infty$) <br> <br> <ins>others</ins> <br> **agent index**: [0,$n$), <br> **ypos**: [0,grid_height], <br> **xpos**: [0, grid_width], <br> **number accepted passengers**: [0, $\infty$), <br> **number riding passengers**: [0,$\infty$)  <br> <br> <ins>tasks</ins> <br> **task index**: [0, $\infty$), <br> **ystart**: [0,grid_height], <br> **xstart**: [0, grid_width], <br> **yend**: [0, grid_height], <br> **xend**: [0,grid_width] <br> **accepted by**: [0,$n$) <br> **riding with**: [0,$n$), <br> **fare**: (0, $\infty$), <br> **time entered**: [0,max steps] |
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import torch
from tensordict.tensordict import TensorDict
import gymnasium
from pettingzoo.utils.wrappers import OrderEnforcingWrapper

from free_range_zoo.utils.env import BatchedAECEnv
from free_range_zoo.utils.conversions import batched_aec_to_batched_parallel
from free_range_zoo.envs.rideshare.env.structures.state import RideshareState
from free_range_zoo.envs.rideshare.env.spaces.observations import build_observation_space
from free_range_zoo.envs.rideshare.env.spaces.actions import build_action_space
import free_range_rust


def parallel_env(wrappers: List[Callable] = [], **kwargs) -> BatchedAECEnv:
    """
    Paralellized version of the rideshare environment.

    Args:
        wrappers: List[Callable[[BatchedAECEnv], BatchedAECEnv]] - the wrappers to apply to the environment
    Returns:
        BatchedAECEnv: the parallelized rideshare environment
    """
    env = raw_env(**kwargs)
    env = OrderEnforcingWrapper(env)

    for wrapper in wrappers:
        env = wrapper(env)

    env = batched_aec_to_batched_parallel(env)
    return env


def env(wrappers: List[Callable] = [], **kwargs) -> BatchedAECEnv:
    """
    AEC wrapped version of the rideshare environment.

    Args:
        wrappers: List[Callable[[BatchedAECEnv], BatchedAECEnv]] - the wrappers to apply to the environment
    Returns:
        BatchedAECEnv: the rideshare environment
    """
    env = raw_env(**kwargs)
    env = OrderEnforcingWrapper(env)

    for wrapper in wrappers:
        env = wrapper(env)

    return env


class raw_env(BatchedAECEnv):
    """Implementation of the dynamic rideshare environment."""
    metadata = {"render.modes": ["human", "rgb_array"], "name": "rideshare_v0", "is_parallelizable": True, "render_fps": 2}

    @torch.no_grad()
    def __init__(self, *args, **kwargs):
        """Initialize the simulation."""
        super().__init__(*args, **kwargs)

        self.possible_agents = tuple(f'driver_{i}' for i in range(1, self.agent_config.num_agents + 1))
        self.agent_name_mapping = {agent: idx for idx, agent in enumerate(self.possible_agents)}

        self.max_x = self.config.grid_width
        self.max_y = self.config.grid_height

        self.agent_observation_bounds = tuple([
            self.max_y,
            self.max_x,
            self.agent_config.pool_limit,
            self.agent_config.pool_limit,
        ])
        self.passenger_observation_bounds = tuple([
            self.max_y,
            self.max_x,
            self.max_y,
            self.max_x,
            self.agent_config.num_agents,
            self.agent_config.num_agents,
            self.config.max_fare,
            self.max_steps,
        ])

        self.environment_range = torch.arange(0, self.parallel_envs, dtype=torch.int32, device=self.device)

        # Create the agent mapping for observation ordering
        agent_ids = torch.arange(0, self.agent_config.num_agents, device=self.device)
        self.observation_ordering = {}
        for agent in self.possible_agents:
            other_agents = agent_ids[agent_ids != self.agent_name_mapping[agent]]
            self.observation_ordering[agent] = other_agents

        self.movement_transition = self.config.movement_transition(self.parallel_envs).to(self.device)
        self.passenger_entry_transition = self.config.passenger_entry_transition(self.parallel_envs).to(self.device)
        self.passenger_exit_transition = self.config.passenger_exit_transition(self.parallel_envs).to(self.device)
        self.passenger_state_transition = self.config.passenger_state_transition(self.parallel_envs).to(self.device)

    @torch.no_grad()
    def reset(self, seed: Optional[List[int]] = None, options: Optional[Dict[str, Any]] = None):
        """
        Reset the environment.

        Args:
            seed: Union[List[int], int] - the seed to use
            options: Dict[str, Any] - the options for the reset
        """
        super().reset(seed=seed, options=options)

        self._state = RideshareState(
            agents=torch.empty((self.parallel_envs, self.num_agents, 2), dtype=torch.int32, device=self.device),
            passengers=None,
        )

        if options is not None and options.get('initial_state') is not None:
            initial_state = options['initial_state']
            if len(initial_state) != self.parallel_envs:
                raise ValueError("Initial state must have the same number of environments as the parallel environments")
            self._state = initial_state
        else:
            self._state.agents[:] = self.agent_config.start_positions

        self._state = self.passenger_entry_transition(self._state, self.num_moves)

        # Save the initial state of the environment
        self._state.save_initial()

        # Intialize the mapping of the tasks "in" the environment, used to map actions
        self.agent_action_mapping = {agent: None for agent in self.agents}
        self.agent_observation_mapping = {agent: None for agent in self.agents}
        self.agent_bad_actions = {agent: None for agent in self.agents}

        # Set the observations and action space
        if not options or not options.get('skip_actions', False):
            self.update_actions()
        if not options or not options.get('skip_observations', False):
            self.update_observations()

        self._post_reset_hook()

    @torch.no_grad()
    def reset_batches(
        self,
        batch_indices: torch.IntTensor,
        seed: Optional[List[int]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Partial reset of the environment for the given batch indices.

        Args:
            batch_indices: torch.IntTensor - the batch indices to reset
            seed: Optional[List[int]] - the seed to use
            options: Optional[Dict[str, Any]] - the options for the reset
        """
        super().reset_batches(batch_indices, seed, options)
        raise NotImplementedError("Reset batches not implemented yet.")

    @torch.no_grad()
    def step_environment(self) -> Tuple[Dict[str, torch.Tensor] | Dict[str, Dict[str, bool]]]:
        # Initialize storages
        rewards = {agent: torch.zeros(self.parallel_envs, dtype=torch.float32, device=self.device) for agent in self.agents}
        terminations = {agent: torch.zeros(self.parallel_envs, dtype=torch.bool, device=self.device) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Initialize action groupings for the various action types
        noop = torch.empty((self.parallel_envs, self.num_agents), dtype=torch.bool, device=self.device)
        accept = torch.empty((self.parallel_envs, self.num_agents), dtype=torch.bool, device=self.device)
        pick = torch.empty((self.parallel_envs, self.num_agents), dtype=torch.bool, device=self.device)
        drop = torch.empty((self.parallel_envs, self.num_agents), dtype=torch.bool, device=self.device)

        # Initialize additional storages to make processing transitions easier
        task_targets = torch.ones((self.parallel_envs, self.num_agents), dtype=torch.int32, device=self.device) * -100
        task_vectors = torch.ones((self.parallel_envs, self.num_agents, 4), dtype=torch.int32, device=self.device) * -100

        # Calculate the offsets from an individual batch space to task store
        index_shifts = torch.roll(torch.cumsum(self.environment_task_count, dim=0, dtype=torch.int32), 1, dims=0)
        index_shifts[0] = 0
        index_shifts = index_shifts[self.environment_range.unsqueeze(1)]

        for agent_name, agent_actions in self.actions.items():
            agent_index = self.agent_name_mapping[agent_name]

            noop[:, agent_index] = agent_actions[:, 1] == -1
            accept[:, agent_index] = agent_actions[:, 1] == 0
            pick[:, agent_index] = agent_actions[:, 1] == 1
            drop[:, agent_index] = agent_actions[:, 1] == 2

            targets = torch.cat([self.environment_range.unsqueeze(1), agent_actions[:, [0]]], dim=1)

            # Figure out the indices of the targeted tasks within the global environment context
            if self.agent_task_count[agent_index].sum() > 0:
                mask = ~noop[:, agent_index]
                values = self.agent_action_mapping[agent_name].to_padded_tensor(-100)[targets[mask].split(
                    1, dim=1)].flatten().int() + index_shifts[mask].flatten()

                task_targets.index_put_((mask, torch.tensor([agent_index], device=targets.device)), values)

        # Calculate point vectors for agent movement and distance calculations
        task_vectors.index_put_(
            (accept, ),
            torch.cat([self._state.agents[accept], self._state.passengers[task_targets[accept]][:, 1:3]], dim=1),
        )
        task_vectors.index_put_(
            (pick, ),
            torch.cat([self._state.agents[pick], self._state.passengers[task_targets[pick]][:, 1:3]], dim=1),
        )
        task_vectors.index_put_(
            (drop, ),
            torch.cat([self._state.agents[drop], self._state.passengers[task_targets[drop]][:, 3:5]], dim=1),
        )

        self._state, distance = self.movement_transition(self._state, task_vectors)

        # NOTE: Passenger transitions must remain in the order of state, exit, entry due to have stable mutability
        self._state = self.passenger_state_transition(self._state, accept, pick, task_targets, task_vectors, self.num_moves)
        self._state, fares = self.passenger_exit_transition(self._state, drop, task_targets, task_vectors, self.num_moves)
        self._state = self.passenger_entry_transition(self._state, self.num_moves + 1)

        # Handle reward distributions for all agents
        global_rewards = torch.zeros((self.parallel_envs, ), dtype=torch.float32, device=self.device)
        if self.reward_config.use_waiting_costs:
            await_accept = self._state.passengers[:, 6] == 0
            await_pick = self._state.passengers[:, 6] == 1
            await_drop = self._state.passengers[:, 6] == 2

            # Calculate the number of steps since the last action
            last_action = torch.empty((self._state.passengers.size(0), ), dtype=torch.int32, device=self.device)
            last_action[await_accept] = self._state.passengers[:, 8][await_accept]
            last_action[await_pick] = self._state.passengers[:, 9][await_pick]
            last_action[await_drop] = self._state.passengers[:, 10][await_drop]
            last_action = self.num_moves[self._state.passengers[:, 0]] - last_action

            # Apply waiting penalities to the global rewards
            global_rewards[self._state.passengers[await_accept][:, 0]] += (
                last_action[await_accept] >= self.reward_config.wait_limit[0]).float() * self.reward_config.general_wait_cost
            global_rewards[self._state.passengers[await_pick][:, 0]] += (
                last_action[await_pick] >= self.reward_config.wait_limit[1]).float() * self.reward_config.general_wait_cost
            global_rewards[self._state.passengers[await_drop][:, 0]] += (
                last_action[await_drop] >= self.reward_config.wait_limit[2]).float() * self.reward_config.general_wait_cost

            # Apply long waiting penalities to the global rewards
            global_rewards[self._state.passengers[await_accept][:, 0]] += (
                last_action[await_accept] >= self.reward_config.long_wait_time).float() * self.reward_config.long_wait_cost

            # Apply unserved costs
            slots = self.agent_config.num_agents * self.agent_config.pool_limit
            task_count = self._state.passengers[:, 0].bincount(minlength=self.parallel_envs)
            unaccepted = self._state.passengers[await_accept][:, 0].bincount(minlength=self.parallel_envs)
            global_rewards += (unaccepted >= (slots - task_count)).int() * -0.5 * (slots - task_count)

        for agent_name, agent_index in self.agent_name_mapping.items():
            # Handle penalties for hitting the pooling limit
            mask = self._state.passengers[:, 7] == agent_index
            accepted = self._state.passengers[mask][:, 0].bincount(minlength=self.parallel_envs)
            rewards[agent_name] += torch.where(accepted > self.agent_config.pool_limit, self.reward_config.pool_limit_cost, 0)

            # Distribute noop penalty
            rewards[agent_name] += noop[:, agent_index].int() * self.reward_config.noop_cost

            # Distribute accept costs
            rewards[agent_name] += accept[:, agent_index].int() * self.reward_config.accept_cost

            # Distribute fare rewards and drop costs
            rewards[agent_name] += torch.where(fares[:, agent_index] > 0, fares[:, agent_index] - self.reward_config.drop_cost, 0)

            # Distribute movement penalties
            distance_rewards = distance[:, agent_index] * self.reward_config.move_cost
            if self.reward_config.use_variable_move_cost:
                distance_rewards /= accepted + 1
            rewards[agent_name] += distance_rewards

            # Distribute the global rewards
            rewards[agent_name] += global_rewards

        return rewards, terminations, infos

    @torch.no_grad()
    def update_actions(self) -> None:
        """Update the action space for all agents."""
        self.environment_task_count = torch.bincount(self._state.passengers[:, 0], minlength=self.parallel_envs)

        index_shifts = torch.roll(torch.cumsum(self.environment_task_count, dim=0), 1, 0)
        index_shifts[0] = 0

        task_range = torch.arange(0, self.environment_task_count.sum(), device=self.device)
        task_indices = (task_range - index_shifts[self._state.passengers[:, 0]].flatten())

        for agent_name, agent_index in self.agent_name_mapping.items():
            general_tasks = self._state.passengers[:, 6] == 0
            exclusive_tasks = self._state.passengers[:, 7] == agent_index
            agent_tasks = general_tasks | exclusive_tasks

            self.agent_task_count[agent_index] = torch.bincount(
                self._state.passengers[:, 0][agent_tasks],
                minlength=self.parallel_envs,
            )

            indices_nested = torch.nested.as_nested_tensor(task_indices[agent_tasks].split(
                self.agent_task_count[agent_index].tolist(),
                dim=0,
            ))

            self.agent_observation_mapping[agent_name] = indices_nested
            self.agent_action_mapping[agent_name] = indices_nested.clone()

    @torch.no_grad()
    def update_observations(self) -> None:
        """
        Update the observations for the agents.

        Observations consist of the following:
            - Agent observation format: (batch, 1, (y, x, num_accepted, num_riding))
            - Others observation format: (batch, agents - 1, (y, x, num_accepted, num_riding))
            - Passenger observation format: (batch, tasks, (y, x, y_dest, x_dest, accepted_by, riding_by, fare, entered_step))
        """
        self.environment_task_count = self._state.passengers[:, 0].bincount(minlength=self.parallel_envs)

        # Aggregate all of the information for the task observations
        task_positional_info = self._state.passengers[:, 1:5]
        entered_info = self._state.passengers[:, [8]]

        accepted = self._state.passengers[:, [6]] == 1
        accepted_info = torch.where(accepted, self._state.passengers[:, [7]], -100)
        riding = self._state.passengers[:, [6]] == 2
        riding_info = torch.where(riding, self._state.passengers[:, [7]], -100)

        fare_info = self._state.passengers[:, [5]]

        task_observations = torch.cat([task_positional_info, accepted_info, riding_info, fare_info, entered_info], dim=1)
        self.task_store = torch.nested.as_nested_tensor(task_observations.split(self.environment_task_count.tolist()))

        # Aggregate all of the informations for the agent observations
        agent_observations = torch.empty(
            (self.parallel_envs, self.agent_config.num_agents, 4),
            dtype=torch.int32,
            device=self.device,
        )
        agent_observations[:, :, :2] = self._state.agents[:, :, :2]
        for agent_name, agent_index in self.agent_name_mapping.items():
            exclusive_tasks = self._state.passengers[:, [7]] == agent_index
            accepted_info = self._state.passengers[:, [0]][accepted & exclusive_tasks].bincount(minlength=self.parallel_envs)
            riding_info = self._state.passengers[:, [0]][riding & exclusive_tasks].bincount(minlength=self.parallel_envs)

            agent_observations[:, agent_index, 2] = accepted_info
            agent_observations[:, agent_index, 3] = riding_info

        # Build the action observation space itself
        self.observations = {}
        for agent_name, agent_index in self.agent_name_mapping.items():
            general_tasks = self._state.passengers[:, 6] == 0
            exclusive_tasks = self._state.passengers[:, 7] == agent_index
            agent_tasks = general_tasks | exclusive_tasks

            self.agent_task_count[agent_index] = self._state.passengers[agent_tasks][:, 0].bincount(minlength=self.parallel_envs)

            nested_observation = torch.nested.as_nested_tensor(task_observations[agent_tasks].split(
                self.agent_task_count[agent_index].tolist()))

            agent_mask = torch.ones(self.agent_config.num_agents, dtype=torch.bool, device=self.device)
            agent_mask[agent_index] = False

            self.observations[agent_name] = TensorDict(
                {
                    'self': agent_observations[:, agent_index].clone(),
                    'others': agent_observations[:, agent_mask].clone(),
                    'tasks': nested_observation.clone(),
                },
                batch_size=[self.parallel_envs],
                device=self.device,
            )

    @torch.no_grad()
    def action_space(self, agent: str) -> List[gymnasium.Space]:
        """
        Create action space for the agent.

        Args:
            agent: str - the agent to create the action space for

        Returns:
            List[gymnasium.Space] - the action spaces for the agent batchwise listed
        """
        agent_index = self.agent_name_mapping[agent]

        general_tasks = self._state.passengers[:, 6] == 0
        exclusive_tasks = self._state.passengers[:, 7] == agent_index
        agent_tasks = general_tasks | exclusive_tasks

        actions = self._state.passengers[agent_tasks][:, [0, 6]]
        return build_action_space(actions, self.agent_task_count[agent_index])

    @torch.no_grad()
    def observation_space(self, agent: str) -> free_range_rust.Space:
        """
        Return the observation space for the given agent.

        Args:
            agent: str - the name of the agent to retrieve the observation space for
        Returns:
            free_range_rust.Space: the observation space for the given agent
        """
        return build_observation_space(
            self.agent_task_count[self.agent_name_mapping[agent]],
            self.agent_config.num_agents,
            self.agent_observation_bounds,
            self.passenger_observation_bounds,
        )
