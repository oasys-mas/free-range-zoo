"""

# Wildfire

| Import             | `from free_range_zoo.envs import wildfire_v0` |
|--------------------|------------------------------------|
| Actions            | Discrete & Stochastic                            |
| Observations | Discrete and fully Observed with private observations [^1]
| Parallel API       | Yes                                |
| Manual Control     | No                                 
|
| Agent Names             | [$firefighter$_0, ..., $firefighter$_n] |
| #Agents             |    $[0,n]$                                  |
| 
Action Shape       | (envs, 2)              |
| Action Values      |  [-1, $\|tasks\|$], [0] [^2]             
|
| Observation Shape | TensorDict: { <br> &emsp; **Agent's self obs**, <ins>'self'</ins>: 4 `<ypos, xpos, fire power, suppressant>`, <br> &emsp; **Other agent obs**, <ins>'others'</ins>: ($\|Ag\| \times 4$) `<ypos,xpos,fire power, suppressant>`, <br> &emsp; **Fire/Task obs**, <ins>'tasks'</ins>: ($\|X\| \times 4$) `<y, x, fire level, intensity>` <br> **batch_size: `num_envs`** <br>}|
| Observation Values   | <ins>Self</ins> <br> **ypos**: [0,grid_height], <br> **xpos**: [0, grid_width], <br> *fire_reduction_power*: [0, initial_fire_power_reduction], <br> **suppressant**: [0,suppressant_states) <br> <br> <ins>Other Agents</ins> <br> **ypos**: [0,grid_height], <br> **xpos**: [0, grid_width], <br> *fire_reduction_power*: [0, initial_fire_power_reduction], <br> **suppressant**: [0,suppressant_states)  <br> <br> <ins>Task</ins> <br> **ypos**: [0,grid_height], <br> **xpos**: [0, grid_width], <br> **fire level**: [initial_fire_level] <br> **intensity**: [0,num_fire_states) |

## Description

A cooperative **agent open** and **task open** domain where agents coordinate to extinguish fires before they burn out. 
Agents do not move, and they choose to either *suppress* (0) a fire they can reach, or *NOOP* (-1) to refill their suppressant. 

Task openness is present as fires which ignite and spread according to a realistic wildfire spreading model used in prior implementations of this environment <cite wildfire papers>. Agents are not present in the environment when out of suppressant as they can only NOOP.

"""



from typing import Tuple, Dict, Any, Union, List, Optional

import torch
from tensordict.tensordict import TensorDict
import gymnasium

from pettingzoo.utils import wrappers

from free_range_zoo.utils.env import BatchedAECEnv
from free_range_zoo.wrappers.planning import planning_wrapper_v0
from free_range_zoo.utils.conversions import batched_aec_to_batched_parallel

from free_range_zoo.envs.wildfire.env import transitions
from free_range_zoo.envs.wildfire.env.utils import in_range_check, random_generator
from free_range_zoo.envs.wildfire.env.spaces import actions, observations
from free_range_zoo.envs.wildfire.env.structures.state import WildfireState


def parallel_env(planning: bool = False, **kwargs):
    """
    Paralellized version of the wildfire environment.

    Args:
        planning: bool - whether to use the planning wrapper
    """
    env = raw_env(**kwargs)
    env = wrappers.OrderEnforcingWrapper(env)

    if planning:
        env = planning_wrapper_v0(env)

    env = batched_aec_to_batched_parallel(env)
    return env


def env(planning: bool = False, **kwargs):
    """
    AEC wrapped version of the wildfire environment.

    Args:
        planning: bool - whether to use the planning wrapper
    """
    env = raw_env(**kwargs)
    env = wrappers.OrderEnforcingWrapper(env)

    if planning:
        env = planning_wrapper_v0(env)

    return env


class raw_env(BatchedAECEnv):
    """Environment definition for the wildfire environment."""

    metadata = {"render.modes": ["human", "rgb_array"], "name": "wildfire_v0", "is_parallelizable": True, "render_fps": 2}

    @torch.no_grad()
    def __init__(self,
                 *args,
                 observe_other_suppressant: bool = True,
                 observe_other_power: bool = True,
                 show_bad_actions: bool = True,
                 **kwargs) -> None:
        """
        Initialize the Wildfire environment.

        Args:
            observe_others_suppressant: bool - whether to observe the suppressant of other agents
            observe_other_power: bool - whether to observe the power of other agents
            show_bad_actions: bool  - whether to show bad actions
        """
        super().__init__(*args, **kwargs)

        self.constant_observations = ['agents']

        self.observe_other_suppressant = observe_other_suppressant
        self.observe_other_power = observe_other_power
        self.show_bad_actions = show_bad_actions

        self.possible_agents = tuple(f"firefighter_{i}" for i in range(1, self.agent_config.num_agents + 1))
        self.agent_name_mapping = dict(zip(self.possible_agents, torch.arange(0, len(self.possible_agents), device=self.device)))
        self.agent_position_mapping = dict(zip(self.possible_agents, self.agent_config.agents))

        self.ignition_temp = self.fire_config.ignition_temp
        self.max_x = self.config.grid_width
        self.max_y = self.config.grid_height

        # Set the transition filter for the fire spread
        self.fire_spread_weights = self.config.fire_spread_weights.to(self.device)

        # Pre-create range indices tensors for use in later operations
        self.fire_index_ranges = torch.arange(self.parallel_envs * self.max_y * self.max_x, device=self.device)
        self.parallel_ranges = torch.arange(self.parallel_envs, device=self.device)

        # Create the agent mapping for observation ordering
        agent_ids = torch.arange(0, self.agent_config.num_agents, device=self.device)
        self.observation_ordering = {}
        for agent in self.possible_agents:
            agent_idx = self.agent_name_mapping[agent]
            other_agents = agent_ids[agent_ids != agent_idx]
            self.observation_ordering[agent] = other_agents

        self.agent_observation_bounds = tuple([
            self.max_y,
            self.max_x,
            self.agent_config.max_fire_reduction_power,
            self.agent_config.suppressant_states,
        ])
        self.fire_observation_bounds = tuple([
            self.max_y,
            self.max_x,
            self.fire_config.max_fire_type,
            self.fire_config.num_fire_states,
        ])

        self.observation_mask = torch.ones(4, dtype=torch.bool, device=self.device)
        self.observation_mask[2] = self.observe_other_power
        self.observation_mask[3] = self.observe_other_suppressant

        # Initialize all of the transition layers based on the environment configurations
        self.capacity_transition = transitions.CapacityTransition(
            agent_shape=(self.parallel_envs, self.agent_config.num_agents),
            stochastic_switch=self.stochastic_config.tank_switch,
            tank_switch_probability=self.agent_config.tank_switch_probability,
            possible_capacities=self.agent_config.possible_capacities,
            capacity_probabilities=self.agent_config.capacity_probabilities,
        ).to(self.device)

        self.equipment_transition = transitions.EquipmentTransition(
            equipment_states=self.agent_config.equipment_states,
            stochastic_repair=self.stochastic_config.repair,
            repair_probability=self.agent_config.repair_probability,
            stochastic_degrade=self.stochastic_config.degrade,
            degrade_probability=self.agent_config.degrade_probability,
            critical_error=self.stochastic_config.critical_error,
            critical_error_probability=self.agent_config.critical_error_probability,
        ).to(self.device)

        self.fire_decrease_transition = transitions.FireDecreaseTransition(
            fire_shape=(self.parallel_envs, self.max_y, self.max_x),
            stochastic_decrease=self.stochastic_config.fire_decrease,
            decrease_probability=self.fire_config.intensity_decrease_probability,
            extra_power_decrease_bonus=self.fire_config.extra_power_decrease_bonus,
        ).to(self.device)

        self.fire_increase_transition = transitions.FireIncreaseTransition(
            fire_shape=(self.parallel_envs, self.max_y, self.max_x),
            fire_states=self.fire_config.num_fire_states,
            stochastic_increase=self.stochastic_config.fire_increase,
            intensity_increase_probability=self.fire_config.intensity_increase_probability,
            stochastic_burnouts=self.stochastic_config.special_burnout_probability,
            burnout_probability=self.fire_config.burnout_probability,
        ).to(self.device)

        self.fire_spread_transition = transitions.FireSpreadTransition(
            fire_spread_weights=self.fire_spread_weights,
            ignition_temperatures=self.ignition_temp,
            use_fire_fuel=self.stochastic_config.fire_fuel,
        ).to(self.device)

        self.suppressant_decrease_transition = transitions.SuppressantDecreaseTransition(
            agent_shape=(self.parallel_envs, self.agent_config.num_agents),
            stochastic_decrease=self.stochastic_config.suppressant_decrease,
            decrease_probability=self.agent_config.suppressant_decrease_probability,
        ).to(self.device)

        self.suppressant_refill_transition = transitions.SuppressantRefillTransition(
            agent_shape=(self.parallel_envs, self.agent_config.num_agents),
            stochastic_refill=self.stochastic_config.suppressant_refill,
            refill_probability=self.agent_config.suppressant_refill_probability,
            equipment_bonuses=self.agent_config.equipment_states[:, 0],
        ).to(self.device)

    @torch.no_grad()
    def reset(self, seed: Union[List[int], int] = None, options: Dict[str, Any] = None) -> None:
        """
        Reset the environment.

        Args:
            seed: Union[List[int], int] - the seed to use
            options: Dict[str, Any] - the options for the reset
        """
        super().reset(seed=seed, options=options)

        # Initialize the agent action to task mapping
        self.agent_action_to_task_mapping = {agent: {} for agent in self.agents}
        self.fire_reduction_power = self.agent_config.fire_reduction_power
        self.suppressant_states = self.agent_config.suppressant_states

        # Initialize the state
        self._state = WildfireState(
            fires=torch.zeros(
                (self.parallel_envs, self.max_y, self.max_x),
                dtype=torch.int32,
                device=self.device,
            ),
            intensity=torch.zeros(
                (self.parallel_envs, self.max_y, self.max_x),
                dtype=torch.int32,
                device=self.device,
            ),
            fuel=torch.zeros(
                (self.parallel_envs, self.max_y, self.max_x),
                dtype=torch.int32,
                device=self.device,
            ),
            agents=self.agent_config.agents,
            capacity=torch.ones(
                (self.parallel_envs, self.agent_config.num_agents),
                dtype=torch.float32,
                device=self.device,
            ),
            suppressants=torch.ones(
                (self.parallel_envs, self.agent_config.num_agents),
                dtype=torch.float32,
                device=self.device,
            ),
            equipment=torch.ones(
                (self.parallel_envs, self.agent_config.num_agents),
                dtype=torch.int32,
                device=self.device,
            ),
        )

        if options is not None and options.get('initial_state') is not None:
            initial_state = options['initial_state']
            if len(initial_state) != self.parallel_envs:
                raise ValueError("Initial state must have the same number of environments as the parallel environments")
            self._state = initial_state
        else:
            self._state.fires[:, self.fire_config.lit] = self.fire_config.fire_types[self.fire_config.lit]
            self._state.fires[:, ~self.fire_config.lit] = -1 * self.fire_config.fire_types[~self.fire_config.lit]
            self._state.intensity[:, self.fire_config.lit] = self.ignition_temp[self.fire_config.lit]
            self._state.fuel[self._state.fires != 0] = self.fire_config.initial_fuel

            self._state.suppressants[:, :] = self.agent_config.initial_suppressant
            self._state.capacity[:, :] = self.agent_config.initial_capacity
            self._state.equipment[:, :] = self.agent_config.initial_equipment_state

        self._state.save_initial()

        # Initialize the rewards for all environments
        self.fire_rewards = self.reward_config.fire_rewards.unsqueeze(0).expand(self.parallel_envs, -1, -1)
        self.num_burnouts = torch.zeros(self.parallel_envs, dtype=torch.int32, device=self.device)

        # Intialize the mapping of the tasks "in" the environment, used to map actions
        self.environment_task_indices = torch.nested.nested_tensor([torch.tensor([]) for _ in range(self.parallel_envs)],
                                                                   dtype=torch.int32)
        self.agent_task_indices = {
            agent: torch.nested.nested_tensor([torch.tensor([]) for _ in range(self.parallel_envs)], dtype=torch.int32)
            for agent in self.agents
        }

        # Set the observations and action space
        if not options or not options.get('skip_observations', False):
            self.update_observations()
        if not options or not options.get('skip_actions', False):
            self.update_actions()

    @torch.no_grad()
    def reset_batches(self,
                      batch_indices: torch.Tensor,
                      seed: Optional[List[int]] = None,
                      options: Optional[Dict[str, Any]] = None) -> None:
        """
        Partial reset of the environment for the given batch indices.

        Args:
            batch_indices: torch.Tensor - the batch indices to reset
            seed: Optional[List[int]] - the seed to use
            options: Optional[Dict[str, Any]] - the options for the reset
        """
        super().reset_batches(batch_indices, seed, options)

        # Reset the state
        self._state.restore_initial(batch_indices)

        self.num_burnouts[batch_indices] = 0

        # Reset the observation updates
        self.update_observations()
        self.update_actions()

    @torch.no_grad()
    def step_environment(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Dict[str, bool]]]:
        """
        The actual simultaneous action wildfire environment step
        """
        # Initialize storages
        rewards = {agent: torch.zeros(self.parallel_envs, dtype=torch.float32, device=self.device) for agent in self.agents}
        terminations = {agent: torch.zeros(self.parallel_envs, dtype=torch.bool, device=self.device) for agent in self.agents}
        infos = {agent: {'task-action-index-map':[None for _ in range(self.parallel_envs)]} for agent in self.agents}

        # For simplification purposes, one randomness generation is done per step, then taken piecewise
        field_randomness, self.generator_states = random_generator.generate_field_randomness(
            generator_states=self.generator_states,
            parallel_envs=self.parallel_envs,
            events=3,
            max_y=self.max_y,
            max_x=self.max_x,
            device=self.device,
        )

        agent_randomness, self.generator_states = random_generator.generate_agent_randomness(
            generator_states=self.generator_states,
            parallel_envs=self.parallel_envs,
            events=5,
            num_agents=self.agent_config.num_agents,
            device=self.device,
        )

        shape = (self.agent_config.num_agents, self.parallel_envs)
        refills = torch.zeros(shape, dtype=torch.bool, device=self.device)
        users = torch.zeros(shape, dtype=torch.bool, device=self.device)
        attack_powers = torch.zeros_like(self._state.fires, dtype=torch.float32, device=self.device)

        env_task_indices_pad = self.environment_task_indices.to_padded_tensor(padding=-100)

        # Loop over each agent
        for agent_name, agent_actions in self.actions.items():
            agent_index = self.agent_name_mapping[agent_name]

            # Determine in which environments this agent is refilling suppressant
            refills[agent_index] = agent_actions[:, 1] == -1

            if self.agent_task_count[agent_index].sum() == 0:
                continue

            # Gather information from the environment
            task_indices = torch.hstack([self.parallel_ranges.unsqueeze(1), agent_actions[:, 0].unsqueeze(1)])
            fire_task_indices = task_indices[~refills[agent_index]]

            agent_task_indices_pad = self.agent_task_indices[agent_name].to_padded_tensor(padding=-100)

            # Get the coordinates for attacking agents
            if self.show_bad_actions:
                fire_coords = env_task_indices_pad[fire_task_indices.split(1, dim=1)]
            else:
                fire_coords = agent_task_indices_pad[fire_task_indices.split(1, dim=1)]
            fire_coords = fire_coords.squeeze(1)

            full_coords = torch.cat([fire_task_indices[:, 0].unsqueeze(1), fire_coords], dim=1)

            reduction_powers = self.fire_reduction_power[agent_index].expand(self.parallel_envs)
            equipment_bonuses = self.agent_config.equipment_states[self._state.equipment[:, agent_index].unsqueeze(0)][:, :, 1]
            full_powers = (reduction_powers + equipment_bonuses).squeeze(0)

            # Create a fight tensor that we will update to filter bad actions
            good_fight = torch.ones(self.parallel_envs, dtype=torch.bool, device=self.device)
            good_fight[refills[agent_index]] = False

            if self.show_bad_actions:
                repeated_attack = fire_coords.unsqueeze(1).expand_as(agent_task_indices_pad[~refills[agent_index]])
                good_actions = (repeated_attack == agent_task_indices_pad[~refills[agent_index]]).all(dim=2).any(dim=1)
                good_fight[~refills[agent_index]] = good_actions
                attack_powers[full_coords[good_actions].split(1, dim=1)] += full_powers[good_fight].unsqueeze(1)
            else:
                attack_powers[full_coords.split(1, dim=1)] += full_powers[good_fight].unsqueeze(1)

            # Aggregate the filtered information
            users[agent_index] = good_fight
            bad_users = torch.logical_not(torch.logical_or(users[agent_index], refills[agent_index]))

            # Give out rewards for bad actions
            rewards[agent_name][bad_users] = self.reward_config.bad_attack_penalty

        refills = refills.T
        users = users.T

        # Handle agent suppressant decrease
        self._state = self.suppressant_decrease_transition(
            state=self._state,
            used_suppressants=users,
            randomness_source=agent_randomness[0],
        )

        # Handle agent equipment transitions
        self._state = self.equipment_transition(
            state=self._state,
            randomness_source=agent_randomness[1],
        )

        # Handle agent suppressant transitions
        self._state, who_increased_suppressants = self.suppressant_refill_transition(
            state=self._state,
            refilled_suppressants=refills,
            randomness_source=agent_randomness[2],
            return_increased=True,
        )

        # Handle the agent capacity transitions
        self._state = self.capacity_transition(
            state=self._state,
            targets=who_increased_suppressants,
            randomness_source=agent_randomness[3:5],
        )

        # Handle fire intensity transitions
        self._state, just_burned_out = self.fire_increase_transition(
            state=self._state,
            attack_counts=attack_powers,
            randomness_source=field_randomness[0],
            return_burned_out=True,
        )
        self._state, just_put_out = self.fire_decrease_transition(
            state=self._state,
            attack_counts=attack_powers,
            randomness_source=field_randomness[1],
            return_put_out=True,
        )
        self._state = self.fire_spread_transition(state=self._state, randomness_source=field_randomness[2])

        fire_rewards = torch.zeros_like(self._state.fires, device=self.device, dtype=torch.float)
        fire_rewards[just_put_out] = self.fire_rewards[just_put_out]
        fire_rewards[just_burned_out] = self.reward_config.burnout_penalty
        fire_rewards_per_batch = fire_rewards.sum(dim=(1, 2))

        self.num_burnouts += just_burned_out.int().sum(dim=(1, 2))

        # Assign rewards
        for agent in self.agents:
            rewards[agent] += fire_rewards_per_batch

        # Determine environment terminations due to no more fires
        fires_are_out = self._state.fires.flatten(start_dim=1).max(dim=1)[0] <= 0
        if self.stochastic_config.fire_fuel:
            # If all fires are out and all fuel is depleted the episode is over
            depleted_fuel = self._state.fuel.sum(dim=(1, 2)) <= 0
            batch_is_dead = torch.logical_and(depleted_fuel, fires_are_out)
        else:
            # If all fires are out then the episode is over
            batch_is_dead = fires_are_out

        newly_terminated = torch.logical_xor(self.terminated, batch_is_dead)
        termination_reward = self.reward_config.termination_reward / (self.num_burnouts + 1)
        for agent in self.agents:
            rewards[agent][newly_terminated] += termination_reward[newly_terminated]

            terminations[agent] = batch_is_dead

        return rewards, terminations, infos

    @torch.no_grad()
    def update_actions(self) -> None:
        """Update the action space for all agents."""
        # Gather all the tasks in the environment
        lit_fires = torch.where(self._state.fires > 0, 1, 0)
        lit_fire_indices = lit_fires.nonzero(as_tuple=False)

        num_tasks = lit_fires.sum()

        task_positions = lit_fire_indices.expand(self.agent_config.num_agents, -1, -1)
        task_positions = task_positions.flatten(end_dim=1)

        agent_positions = self._state.agents.unsqueeze(1).expand(-1, num_tasks, -1)
        agent_positions = agent_positions.flatten(end_dim=1)

        # Get the indices of all agents within the environment
        agent_indices = torch.arange(0, self.agent_config.num_agents, device=self.device).unsqueeze(1)
        agent_indices = agent_indices.expand(-1, num_tasks).flatten()
        agent_indices = torch.cat([task_positions[:, 0].unsqueeze(1), agent_indices.unsqueeze(1)], dim=1)

        # Gather the ranges for the respective agents
        agent_ranges = self.agent_config.attack_range[agent_indices[:, 1]].flatten()

        # Get the respective equipment states and range bonuses
        equipment_states = self._state.equipment[agent_indices.split(1, dim=1)].squeeze(1)
        range_bonuses = self.agent_config.equipment_states[equipment_states.unsqueeze(0)][:, :, 2].squeeze(0)

        # Calculate the agent range after bonuses have been applied
        true_range = (agent_ranges + range_bonuses).flatten()

        in_range = in_range_check.chebyshev(
            agent_position=agent_positions,
            task_position=task_positions[:, 1:],
            attack_range=true_range,
        )
        in_range = in_range.reshape(self.agent_config.num_agents, num_tasks)

        # Detemine which agents have suppressants
        has_suppressants = self._state.suppressants[agent_indices.split(1, dim=1)].squeeze(1) > 0
        has_suppressants = has_suppressants.reshape(self.agent_config.num_agents, num_tasks)

        # Combine the two checks to determine which agents can fight which fires
        checks = torch.logical_and(in_range, has_suppressants)

        # Aggregate the indices of all tasks to agent mappingk
        agent_tasks = {}
        for agent, agent_number in self.agent_name_mapping.items():
            #  Get all of the valid tasks for this agent
            tasks = lit_fire_indices[checks[agent_number]]

            # Count the number of tasks in each batch and aggregate the indices
            task_count = torch.bincount(tasks[:, 0], minlength=self.parallel_envs)
            batchwise_indices = torch.nested.as_nested_tensor(tasks[:, 1:].split(task_count.tolist(), dim=0))

            self.agent_task_count[agent_number] = task_count
            agent_tasks[agent] = batchwise_indices

        # Update the agent action to task mapping
        self.agent_task_indices = agent_tasks

        # Aggregate the indices of all tasks in each environment
        task_count = lit_fires.sum(dim=(1, 2))
        batchwise_indices = torch.nested.as_nested_tensor(lit_fire_indices[:, 1:].split(task_count.tolist(), dim=0))

        # Set the indices for all tasks in each environment
        self.environment_task_count = task_count
        self.environment_task_indices = batchwise_indices

    @torch.no_grad()
    def update_observations(self) -> None:
        """
        Update the observations for the agents.

        Observations consist of the following:
            - Agent observation format: (batch, 1, (y, x, power, suppressant))
            - Others observation format: (batch, agents - 1, (y, x, power, suppressant))
            - Fire observation format: (batch, fires, (y, x, heat, intensity))
        """
        # Build the agent observations
        agent_positions = self._state.agents.expand(self.parallel_envs, -1, -1)
        fire_reduction_power = self.fire_reduction_power.unsqueeze(-1).expand(self.parallel_envs, -1, -1)
        suppressants = self._state.suppressants.unsqueeze(-1)
        agent_observations = torch.cat((agent_positions, fire_reduction_power, suppressants), dim=2)

        # Build the fire observations
        lit_fires = torch.where(self._state.fires > 0, 1, 0)
        lit_fire_indices = lit_fires.nonzero(as_tuple=False)

        intensities = self._state.intensity[lit_fire_indices.split(1, dim=1)]
        fires = self._state.fires[lit_fire_indices.split(1, dim=1)]

        task_count = lit_fires.sum(dim=(1, 2))
        fire_observations = torch.cat([lit_fire_indices[:, 1:], fires, intensities], dim=1)
        fire_observations = torch.nested.as_nested_tensor(fire_observations.split(task_count.tolist(), dim=0))

        # Aggregate the full observation space
        self.observations = {}
        for agent in self.agents:
            agent_index = self.agent_name_mapping[agent]
            agent_mask = torch.ones(self.agent_config.num_agents, dtype=torch.bool, device=self.device)
            agent_mask[agent_index] = False

            self.observations[agent] = TensorDict(
                {
                    'self': agent_observations[:, agent_index],
                    'others': agent_observations[:, agent_mask][:, :, self.observation_mask],
                    'tasks': fire_observations
                },
                batch_size=[self.parallel_envs],
                device=self.device,
            )

    @torch.no_grad()
    def action_space(self, agent: str) -> List[gymnasium.Space]:
        """
        Return the action space for the given agent.

        Args:
            agent: str - the name of the agent to retrieve the action space for
        Returns:
            List[gymnasium.Space]: the action space for the given agent
        """
        if self.show_bad_actions:
            num_tasks_in_environment = self.environment_task_count
        else:
            num_tasks_in_environment = self.agent_task_count[self.agent_name_mapping[agent]]

        return actions.build_action_space(environment_task_counts=num_tasks_in_environment)

    @torch.no_grad()
    def observation_space(self, agent: str) -> List[gymnasium.Space]:
        """
        Return the observation space for the given agent.

        Args:
            agent: str - the name of the agent to retrieve the observation space for
        Returns:
            List[gymnasium.Space]: the observation space for the given agent
        """
        return observations.build_observation_space(environment_task_counts=self.environment_task_count,
                                                    num_agents=self.agent_config.num_agents,
                                                    agent_high=self.agent_observation_bounds,
                                                    fire_high=self.fire_observation_bounds,
                                                    include_suppressant=self.observe_other_suppressant,
                                                    include_power=self.observe_other_power)
