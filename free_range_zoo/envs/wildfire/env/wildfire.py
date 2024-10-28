from typing import Tuple, Dict, Any, Union, List, Optional

import torch
from tensordict.tensordict import TensorDict
import gymnasium

from pettingzoo.utils import wrappers

from free_range_zoo.free_range_zoo.utils.env import BatchedAECEnv
from free_range_zoo.free_range_zoo.wrappers.planning import planning_wrapper_v0
from free_range_zoo.free_range_zoo.utils.conversions import batched_aec_to_batched_parallel

from free_range_zoo.free_range_zoo.envs.wildfire.env.utils import in_range_check, random_generator
from free_range_zoo.free_range_zoo.envs.wildfire.env.spaces import actions, observations
from free_range_zoo.free_range_zoo.envs.wildfire.env.structures.state import WildfireState
from free_range_zoo.free_range_zoo.envs.wildfire.env.transitions import (suppressant_refill,
                                                                         suppressant_decrease,
                                                                         capacity,
                                                                         equipment,
                                                                         fire_spreads,
                                                                         fire_decrease,
                                                                         fire_increase)


def parallel_env(planning: bool = False, **kwargs):
    env = raw_env(**kwargs)
    env = wrappers.OrderEnforcingWrapper(env)

    if planning:
        env = planning_wrapper_v0(env)

    env = batched_aec_to_batched_parallel(env)
    return env


def env(planning: bool = False, **kwargs):
    env = raw_env(**kwargs)
    env = wrappers.OrderEnforcingWrapper(env)

    if planning:
        env = planning_wrapper_v0(env)

    return env


class raw_env(BatchedAECEnv):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "name": "wildfire_v0",
        "is_parallelizable": True,
        "render_fps": 2
    }

    BAD_ATTACK_PENALTY = -100
    BURNED_OUT_PENALTY = -1

    @torch.no_grad()
    def __init__(self,
                 *args,
                 observe_other_suppressant: bool = True,
                 observe_other_power: bool = True,
                 show_bad_actions: bool = True, **kwargs) -> None:
        """
        Initializes the Wildfire environment

        Args:
            observe_others_suppressant: bool - whether to observe the suppressant of other agents
            observe_other_power: bool - whether to observe the power of other agents
            show_bad_actions: bool  - whether to show bad actions
        """
        super().__init__(*args, **kwargs)

        #for logging
        self.constant_observations = ['agents']

        #environment config
        self.observe_other_suppressant = observe_other_suppressant
        self.observe_other_power = observe_other_power
        self.show_bad_actions = show_bad_actions

        self.possible_agents = tuple(f"firefighter_{i}" for i in range(1, self.agent_config.num_agents + 1))
        self.agent_name_mapping = dict(zip(self.possible_agents,
                                           torch.arange(0, len(self.possible_agents) + 1, device=self.device)))
        self.agent_position_mapping = dict(zip(self.possible_agents, self.agent_config.agents))

        self.ignition_temp = self.fire_config.ignition_temp
        self.max_x = self.config.grid_width
        self.max_y = self.config.grid_height

        # Set the transition filter for the fire spread
        self.fire_spread_weights = self.config.fire_spread_weights.to(self.device)

        # Create the agent mapping for observation ordering
        agent_ids = torch.arange(0, self.agent_config.num_agents, device=self.device)
        self.observation_ordering = {}
        for agent in self.possible_agents:
            agent_idx = self.agent_name_mapping[agent]
            other_agents = agent_ids[agent_ids != agent_idx]
            self.observation_ordering[agent] = other_agents

        self.agent_observation_bounds = tuple([self.max_y, self.max_x,
                                               self.agent_config.max_fire_reduction_power,
                                               self.agent_config.suppressant_states])
        self.fire_observation_bounds = tuple([self.max_y, self.max_x,
                                              self.fire_config.max_fire_type,
                                              self.fire_config.num_fire_states])

        self.observation_mask = torch.ones(4, dtype=torch.bool, device=self.device)
        self.observation_mask[2] = self.observe_other_power
        self.observation_mask[3] = self.observe_other_suppressant

    @torch.no_grad()
    def reset(self, seed: Union[List[int], int] = None, options: Dict[str, Any] = None) -> None:
        """
        Resets the environment

        Args:
            seed: Union[List[int], int] - the seed to use
            options: Dict[str, Any] - the options for the reset
        """
        super().reset(seed=seed, options=options)

        # Dictionary storing actions for each agent
        self.actions = {agent: torch.empty(self.parallel_envs, 2) for agent in self.agents}

        # Initialize the agent action to task mapping
        self.agent_action_to_task_mapping = {agent: {} for agent in self.agents}
        self.fire_reduction_power = self.agent_config.fire_reduction_power
        self.suppressant_states = self.agent_config.suppressant_states

        # Initialize the state
        self._state = WildfireState(
            fires=torch.zeros((self.parallel_envs, self.max_y, self.max_x), dtype=torch.int32),
            intensity=torch.zeros((self.parallel_envs, self.max_y, self.max_x), dtype=torch.int32),
            fuel=torch.zeros((self.parallel_envs, self.max_y, self.max_x), dtype=torch.int32),

            agents=torch.tensor(self.agent_config.agents, dtype=torch.int32),
            capacity=torch.ones((self.parallel_envs, self.agent_config.num_agents), dtype=torch.int32),
            suppressants=torch.ones((self.parallel_envs, self.agent_config.num_agents), dtype=torch.float32),
            equipment=torch.ones((self.parallel_envs, self.agent_config.num_agents), dtype=torch.int32),
        )
        self._state.to(self.device)

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
        self.fire_rewards = self.fire_config.fire_rewards.unsqueeze(0).repeat(self.parallel_envs, 1, 1)

        # Intialize the mapping of the tasks "in" the environment, used to map actions
        self.environment_task_indices = torch.nested.nested_tensor(
            [torch.tensor([]) for _ in range(self.parallel_envs)], dtype=torch.int32)
        self.agent_task_indices = {agent: torch.nested.nested_tensor(
            [torch.tensor([]) for _ in range(self.parallel_envs)], dtype=torch.int32) for agent in self.agents}

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
        Partially resets the environment for the given batch indices

        Args:
        """
        super().reset_batches(batch_indices, seed, options)

        # Reset the state
        self._state.restore_initial(batch_indices)

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
        infos = {agent: {} for agent in self.agents}

        # For simplification purposes, one randomness generation is done per step, then taken piecewise
        field_randomness, self.generator_states = random_generator.generate_field_randomness(
            generator_states=self.generator_states,
            events=3,
            max_y=self.max_y,
            max_x=self.max_x,
            device=self.device)

        agent_randomness, self.generator_states = random_generator.generate_agent_randomness(
            generator_states=self.generator_states,
            events=5,
            num_agents=self.agent_config.num_agents,
            device=self.device)

        bad_actions = {'agent_name': [], 'agent_index': []}
        good_actions = {'agent_name': [], 'agent_index': [], 'fire_position': []}

        suppressant_refills = []

        # Loop over each agent
        for agent_name in self.actions:
            agent_index = self.agent_name_mapping[agent_name]

            # Process each agent's actions in all environments
            for index, action in enumerate(self.actions[agent_name]):
                if self.show_bad_actions:
                    noop_action_index = self.environment_task_count[index]
                else:
                    noop_action_index = self.agent_task_count[agent_index, index]

                index = torch.tensor(index, device=self.device)
                if action[0] == noop_action_index:
                    suppressant_refills.append(torch.cat([index.unsqueeze(0), agent_index.unsqueeze(0)]))
                else:
                    try:
                        if self.show_bad_actions:
                            fire_position = self.environment_task_indices[index, action[0]]

                            if not (self.agent_task_indices[agent_name][index] == fire_position).all(dim=1).any():
                                bad_actions['agent_name'].append(agent_name)
                                bad_actions['agent_index'].append(torch.tensor([index, agent_index], device=self.device))
                            else:
                                good_actions['agent_name'].append(agent_name)
                                good_actions['agent_index'].append(torch.cat([index.unsqueeze(0), agent_index.unsqueeze(0)]))
                                good_actions['fire_position'].append(torch.cat([index.unsqueeze(0), fire_position]))
                        else:
                            fire_position = self.agent_task_indices[agent_name][index, action[0]]
                            good_actions['agent_name'].append(agent_name)
                            good_actions['agent_index'].append(torch.cat([index.unsqueeze(0), agent_index.unsqueeze(0)]))
                            good_actions['fire_position'].append(torch.cat([index.unsqueeze(0), fire_position]))
                    except IndexError as e:
                        print(f'Agent: {agent_name}, Index: {index}, Action: {action[0]}')
                        raise ValueError(f'ERROR: Task ID not found in environment - {e}')

        if len(suppressant_refills) > 0:
            suppressant_refills = torch.stack(suppressant_refills, dim=0)

        if len(good_actions['agent_index']) > 0:
            good_actions['agent_index'] = torch.stack(good_actions['agent_index'], dim=0)
            good_actions['fire_position'] = torch.stack(good_actions['fire_position'], dim=0)

        if len(bad_actions['agent_index']) > 0:
            bad_actions['agent_index'] = torch.stack(bad_actions['agent_index'], dim=0)

        # Determine total attack power for all squares
        attack_powers = torch.zeros_like(self._state.fires, device=self.device, dtype=torch.float32)
        if len(good_actions['agent_index']) > 0:
            reduction_power = self.fire_reduction_power[good_actions['agent_index'][:, 1]].unsqueeze(-1)
            equipment_states = self._state.equipment[good_actions['agent_index'].split(1, dim=1)]
            equipment_bonuses = self.agent_config.equipment_states[equipment_states][:, :, 1]

            attack_powers[good_actions['fire_position'].split(1, dim=1)] += reduction_power
            attack_powers[good_actions['fire_position'].split(1, dim=1)] += equipment_bonuses

        # Handle suppressant refills
        if len(suppressant_refills) > 0:
            refill_mask = suppressant_refill.calculate_mask(
                targets=suppressant_refills,
                refill_probability=self.agent_config.suppressant_refill_probability,
                stochastic_transition=self.stochastic_config.suppressant_refill,
                randomness_source=agent_randomness[:, 0],
                device=self.device)

            capacity_mask, new_capacities = capacity.calculate_modified(
                targets=refill_mask.nonzero(),
                stochastic_transition=self.stochastic_config.tank_switch,
                tank_switch_probability=self.agent_config.tank_switch_probability,
                possible_capacities=self.agent_config.possible_capacities,
                capacity_probabilities=self.agent_config.capacity_probabilities,
                randomness_source=agent_randomness[:, 1:3],
                device=self.device)

            equipment_states = self._state.equipment.flatten().unsqueeze(1)
            equipment_bonuses = self.agent_config.equipment_states[:, 0][equipment_states].reshape(self.parallel_envs, -1)

            self._state.capacity[capacity_mask] = new_capacities[capacity_mask]
            self._state.suppressants[refill_mask] = self._state.capacity[refill_mask].float()
            self._state.suppressants[refill_mask] += equipment_bonuses[refill_mask]

        # Handle agent suppressant decrease
        if len(good_actions['agent_index']) > 0:
            decrease_mask = suppressant_decrease.calculate_mask(
                targets=good_actions['agent_index'],
                stochastic_transition=self.stochastic_config.suppressant_decrease,
                decrease_probability=self.agent_config.suppressant_decrease_probability,
                randomness_source=agent_randomness[:, 3],
                device=self.device)

            self._state.suppressants[decrease_mask] -= 1
            self._state.suppressants[self._state.suppressants < 0] = 0

        # Handle agent equipment transitions
        self._state.equipment = equipment.calculate_modified(
            equipment_conditions=self._state.equipment,
            num_equipment_states=self.agent_config.num_equipment_states,
            repair_probability=self.agent_config.repair_probability,
            degrade_probability=self.agent_config.degrade_probability,
            critical_error_probability=self.agent_config.critical_error_probability,
            stochastic_repair=self.stochastic_config.repair,
            stochastic_degrade=self.stochastic_config.degrade,
            critical_error=self.stochastic_config.critical_error,
            randomness_source=agent_randomness[:, 4])

        # Handle fire intensity transitions
        fire_spread_probabilities = fire_spreads.calculate_fire_spread_probabilities(
            fire_spread_weights=self.fire_spread_weights,
            fires=self._state.fires,
            intensities=self._state.intensity,
            fuel=self._state.fuel,
            use_fire_fuel=self.stochastic_config.fire_fuel)
        fire_decrease_probabilities = fire_decrease.calculate_fire_decrease_probabilities(
            fires=self._state.fires,
            intensity=self._state.intensity,
            attack_counts=attack_powers,
            base_fire_reduction=self.fire_config.intensity_decrease_probability,
            fire_reduction_per_extra_agent=self.agent_config.fire_reduction_power_per_extra_agent,
            device=self.device)
        fire_increase_probabilities = fire_increase.calculate_fire_increase_probabilities(
            fires=self._state.fires,
            intensity=self._state.intensity,
            attack_counts=attack_powers,
            use_almost_burned_out=self.stochastic_config.special_burnout_probability,
            almost_burned_out=self.fire_config.almost_burned_out,
            intensity_increase_probability=self.fire_config.intensity_increase_probability,
            burnout_probability=self.fire_config.burnout_probability,
            device=self.device)

        fire_spread_mask = field_randomness[:, 0] < fire_spread_probabilities
        fire_decrease_mask = field_randomness[:, 1] < fire_decrease_probabilities
        fire_increase_mask = field_randomness[:, 2] < fire_increase_probabilities

        # Apply the fire intensity masks
        self._state.intensity[fire_decrease_mask] -= 1
        self._state.intensity[fire_increase_mask] += 1
        self._state.fires[fire_spread_mask] *= -1
        self._state.intensity[fire_spread_mask] = self.ignition_temp.repeat(self.parallel_envs, 1, 1)[fire_spread_mask]

        # Apply post-intensity change transitions
        just_put_out = torch.logical_and(fire_decrease_mask, self._state.intensity <= 0)
        just_burned_out = torch.logical_and(fire_increase_mask, self._state.intensity >= self.fire_config.burned_out)
        self._state.fires[just_put_out] *= -1
        self._state.fuel[just_put_out] -= 1
        self._state.fires[just_burned_out] *= -1
        self._state.fuel[just_burned_out] = 0

        # Aggregate rewards
        if len(bad_actions['agent_index']) > 0:
            for agent_name, agent_position in zip(bad_actions['agent_name'], bad_actions['agent_index']):
                rewards[agent_name][agent_position[0]] = self.BAD_ATTACK_PENALTY
                infos[agent_name]['bad_attack'] = True

        fire_rewards = torch.zeros_like(self._state.fires, device=self.device, dtype=torch.float)
        fire_rewards[just_put_out] = self.fire_rewards[just_put_out]
        fire_rewards[just_burned_out] = self.BURNED_OUT_PENALTY
        fire_rewards_per_batch = fire_rewards.sum(dim=(1, 2))

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

        for agent in self.agents:
            terminations[agent] = batch_is_dead

        return rewards, terminations, infos

    @torch.no_grad()
    def update_actions(self) -> None:
        """
        Updates the action space for all agents
        """
        # Gather all the tasks in the environment
        lit_fires = torch.where(self._state.fires > 0, 1, 0)
        lit_fire_indices = lit_fires.nonzero().type(torch.int32)
        num_tasks = lit_fire_indices.shape[0]

        task_positions = lit_fire_indices.repeat(self.agent_config.num_agents, 1, 1)
        task_positions = task_positions.flatten(end_dim=1)

        agent_positions = self._state.agents.unsqueeze(1).repeat(1, num_tasks, 1)
        agent_positions = agent_positions.flatten(end_dim=1)

        # Get the indices of all agents within the environment
        agent_indices = torch.arange(0, self.agent_config.num_agents, device=self.device).unsqueeze(1)
        agent_indices = agent_indices.repeat(1, num_tasks).flatten()
        agent_indices = torch.cat([task_positions[:, 0].unsqueeze(1), agent_indices.unsqueeze(1)], dim=1)

        # Gather the ranges for the respective agents
        agent_ranges = self.agent_config.attack_range[agent_indices[:, 1]].flatten()

        # Get the respective equipment states and range bonuses
        equipment_states = self._state.equipment[agent_indices.split(1, dim=1)].squeeze(1)
        range_bonuses = self.agent_config.equipment_states[equipment_states.unsqueeze(0)][:, :, 2].squeeze(0)

        # Calculate the agent range after bonuses have been applied
        true_range = (agent_ranges + range_bonuses).flatten()

        in_range = in_range_check.chebyshev(agent_position=agent_positions,
                                            task_position=task_positions[:, 1:],
                                            attack_range=true_range)
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
        Updates the observations for the agents

        Observations consist of the following:
            - Agent observation format: (batch, 1, (y, x, power, suppressant))
            - Others observation format: (batch, agents - 1, (y, x, power, suppressant))
            - Fire observation format: (batch, fires, (y, x, heat, intensity))
        """
        # Build the agent observations
        agent_positions = self._state.agents.repeat(self.parallel_envs, 1, 1)
        fire_reduction_power = self.fire_reduction_power.unsqueeze(-1).repeat(self.parallel_envs, 1, 1)
        suppressants = self._state.suppressants.unsqueeze(-1)
        agent_observations = torch.cat((agent_positions, fire_reduction_power, suppressants), dim=2)

        # Build the fire observations
        lit_fires = torch.where(self._state.fires > 0, 1, 0)
        lit_fire_indices = lit_fires.nonzero().type(torch.int32)

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

            self.observations[agent] = TensorDict({'self': agent_observations[:, agent_index],
                                                   'others': agent_observations[:, agent_mask][:, :, self.observation_mask],
                                                   'fire': fire_observations}, batch_size=[self.parallel_envs], device=self.device)

    @torch.no_grad()
    def action_space(self, agent: str) -> List[gymnasium.Space]:
        """
        Returns the action space for the given agent

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
        Returns the observation space for the given agent

        Args:
            agent: str - the name of the agent to retrieve the observation space for
        Returns:
            List[gymnasium.Space]: the observation space for the given agent
        """
        return observations.build_observation_space(
            environment_task_counts=self.environment_task_count,
            num_agents=self.agent_config.num_agents,
            agent_high=self.agent_observation_bounds,
            fire_high=self.fire_observation_bounds,
            include_suppressant=self.observe_other_suppressant,
            include_power=self.observe_other_power)
