"""
# Wildfire
## Description

The wildfire domain simulates a grid-based environment where fires can spread and agents are tasked with extinguishing
them by applying suppressant. The environment is dynamic and partially observable, with fires that can spread across
adjacent tiles and vary in intensity. Fires can also burn out once they reach a certain intensity threshold.

<u>**Environment Dynamics**</u><br>
- Fire Spread: Fires start at designated locations and spread to neighboring tiles, increasing in intensity over
  time. The intensity of the fire influences how much suppressant is needed to extinguish it. Fires will continue
  to spread until they either burn out or are controlled by agents.
- Fire Intensity and Burnout: As fires spread, their intensity increases, making them harder to fight. Once a
  fire reaches a critical intensity, it may burn out naturally, stopping its spread and extinguishing itself.
  However, this is unpredictable, and timely intervention is often necessary to prevent further damage.
- Suppression Mechanism: Agents apply suppressant to the fire to reduce its intensity. However, suppressant is a
  finite resource. When an agent runs out of suppressant, they must leave the environment to refill at a designated
  station before returning to continue fighting fires.

<u>**Environment Openness**</u><br>
- **agent openness**: Environments where agents can dynamically enter and leave, enabling dynamic cooperation and
  multi-agent scenarios with evolving participants.
    - `wildfire`: Agents can run out of suppressant and leave the environment, removing their contributions
      to existing fires. Agents must reason about their collaborators leaving, or new collaborators entering.
- **task openness**: Tasks can be introduced or removed from the environment, allowing for flexbile goal setting
  and adaptable planning models
    - `wildfire`: Fires can spread beyond their original starting point, requiring agents to reason about new
      tasks possibly entering the environment as well as a changing action space: Fires can spread beyond
      their original starting point, requiring agents to reason about new tasks possibly entering the
      environment as well as a changing action space.
- **frame / type openness**: Different frames (e.g. agent abilities or skills) can be added, removed, or modified,
  expending the environmental complexity and requiring agents to infer their neighbors changing abilities.
    - `wildfire`: Agents can damage their equipment over time, and have their capabilities slowly degrade. On
      the other hand, agents might also recieve different equipment upon leaving the environment to resupply.

# Specification

---

| Import             | `from free_range_zoo.envs import wildfire_v0`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Actions            | Discrete & Stochastic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Observations       | Discrete and fully Observed with private observations                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Parallel API       | Yes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Manual Control     | No                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Agent Names        | [$firefighter$_0, ..., $firefighter$_n]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| # Agents           | [0, $n_{firefighters}$]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Action Shape       | ($envs$, 2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Action Values      | [$fight_0$, ..., $fight_{tasks}$, $noop$ (-1)]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Observation Shape  | TensorDict: { <br>&emsp;**self**: $<ypos, xpos, fire power, suppressant>$<br>&emsp;**others**: $<ypos,xpos,fire power, suppressant>$<br>&emsp;**tasks**: $<y, x, fire level, intensity>$ <br> **batch_size**: $num\_envs$ }                                                                                                                                                                                                                                                                                                                                                                      |
| Observation Values | <u>**self**</u>:<br>&emsp;$ypos$: $[0, max_y]$<br>&emsp;$xpos$: $[0, max_x]$<br>&emsp;$fire\_power\_reduction$: $[0, max_{fire\_power\_reduction}]$<br>&emsp;$suppressant$: $[0, max_{suppressant}]$<br><u>**others**</u>:<br>&emsp;$ypos$: $[0, max_y]$<br>&emsp;$xpos$: $[0, max_x]$<br>&emsp;$fire\_power\_reduction$: $[0, max_{fire\_power\_reduction}]$<br>&emsp;$suppressant$: $[0, max_{suppressant}]$<br> <u>**tasks**</u><br>&emsp;$ypos$: $[0, max_y]$<br>&emsp;$xpos$: $[0, max_x]$<br>&emsp;$fire\_level$: $[0, max_{fire\_level}]$<br>&emsp;$intensity$: $[0, num_{fire\_states}]$ |

---
"""

from typing import Tuple, Dict, Any, Union, List, Optional, Callable

import torch
from tensordict.tensordict import TensorDict
import gymnasium
import pandas as pd
from pettingzoo.utils.wrappers import OrderEnforcingWrapper

from free_range_zoo.utils.env import BatchedAECEnv
from free_range_zoo.utils.conversions import batched_aec_to_batched_parallel
from free_range_zoo.envs.wildfire.env import transitions
from free_range_zoo.envs.wildfire.env.utils import in_range_check
from free_range_zoo.envs.wildfire.env.spaces import actions, observations
from free_range_zoo.envs.wildfire.env.structures.state import WildfireState


def parallel_env(wrappers: List[Callable] = [], **kwargs) -> BatchedAECEnv:
    """
    Paralellized version of the wildfire environment.

    Args:
        wrappers: List[Callable[[BatchedAECEnv], BatchedAECEnv]] - the wrappers to apply to the environment
    Returns:
        BatchedAECEnv: the parallelized wildfire environment
    """
    env = raw_env(**kwargs)
    env = OrderEnforcingWrapper(env)

    for wrapper in wrappers:
        env = wrapper(env)

    env = batched_aec_to_batched_parallel(env)
    return env


def env(wrappers: List[Callable], **kwargs) -> BatchedAECEnv:
    """
    AEC wrapped version of the wildfire environment.

    Args:
        wrappers: List[Callable[[BatchedAECEnv], BatchedAECEnv]] - the wrappers to apply to the environment
    Returns:
        BatchedAECEnv: the AEC wrapped wildfire environment
    """
    env = raw_env(**kwargs)
    env = OrderEnforcingWrapper(env)

    for wrapper in wrappers:
        env = wrapper(env)

    return env


class raw_env(BatchedAECEnv):
    """Environment definition for the wildfire environment."""

    metadata = {"render.modes": ["human", "rgb_array"], "name": "wildfire_v0", "is_parallelizable": True, "render_fps": 2}

    @torch.no_grad()
    def __init__(self,
                 *args,
                 observe_other_suppressant: bool = False,
                 observe_other_power: bool = False,
                 show_bad_actions: bool = False,
                 **kwargs) -> None:
        """
        Initialize the Wildfire environment.

        Args:
            observe_others_suppressant: bool - whether to observe the suppressant of other agents
            observe_other_power: bool - whether to observe the power of other agents
            show_bad_actions: bool  - whether to show bad actions
        """
        super().__init__(*args, **kwargs)

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

        observation_mask = torch.ones(4, dtype=torch.bool, device=self.device)
        observation_mask[2] = self.observe_other_power
        observation_mask[3] = self.observe_other_suppressant
        self.agent_observation_mask = lambda agent_name: observation_mask

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
            fire_random_spread_weight=self.config.fire_random_spread_weight,
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
        self.agent_action_mapping = {agent: None for agent in self.agents}
        self.agent_observation_mapping = {agent: None for agent in self.agents}
        self.agent_bad_actions = {agent: None for agent in self.agents}

        # Set the observations and action space
        if not options or not options.get('skip_observations', False):
            self.update_observations()
        if not options or not options.get('skip_actions', False):
            self.update_actions()

        self._post_reset_hook()

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
        infos = {agent: {} for agent in self.agents}

        # For simplification purposes, one randomness generation is done per step, then taken piecewise
        field_randomness = self.generator.generate(self.parallel_envs, 3, (self.max_y, self.max_x), key='field')
        agent_randomness = self.generator.generate(self.parallel_envs, 5, (self.agent_config.num_agents, ), key='agent')

        shape = (self.agent_config.num_agents, self.parallel_envs)
        refills = torch.zeros(shape, dtype=torch.bool, device=self.device)
        users = torch.zeros(shape, dtype=torch.bool, device=self.device)
        attack_powers = torch.zeros_like(self._state.fires, dtype=torch.float32, device=self.device)

        fire_positions = (self._state.fires > 0).nonzero(as_tuple=False)

        index_shifts = torch.roll(torch.cumsum(self.environment_task_count, dim=0), 1, dims=0)
        index_shifts[0] = 0
        index_shifts = index_shifts[self.parallel_ranges.unsqueeze(1)]

        # Loop over each agent
        for agent_name, agent_actions in self.actions.items():
            agent_index = self.agent_name_mapping[agent_name]

            # Determine in which environments this agent is refilling suppressant
            refills[agent_index] = agent_actions[:, 1] == -1

            if self.agent_task_count[agent_index].sum() == 0:
                continue

            agent_action_mapping_pad = self.agent_action_mapping[agent_name].to_padded_tensor(padding=-100)

            # Gather the indexes of all of the agent actions
            task_indices = torch.hstack([self.parallel_ranges.unsqueeze(1), agent_actions[:, 0].unsqueeze(1)])
            task_indices = agent_action_mapping_pad[task_indices[~refills[agent_index]].split(1, dim=1)]

            global_task_indices = (agent_actions[:, 0] + index_shifts.squeeze(1))[~refills[agent_index]]

            fire_coords = fire_positions[global_task_indices]

            reduction_powers = self.fire_reduction_power[agent_index].expand(self.parallel_envs)
            equipment_bonuses = self.agent_config.equipment_states[self._state.equipment[:, agent_index].unsqueeze(0)][:, :, 1]
            full_powers = (reduction_powers + equipment_bonuses).squeeze(0)

            # Create a fight tensor that we will update to filter bad actions
            good_fight = torch.ones(self.parallel_envs, dtype=torch.bool, device=self.device)
            good_fight[refills[agent_index]] = False

            if self.show_bad_actions and self.agent_bad_actions[agent_name].numel() > 0:
                bad_actions = self.agent_bad_actions[agent_name].to_padded_tensor(-100)
                attacks = agent_actions[:, 0].unsqueeze(1).expand_as(bad_actions)

                bad_actions = (bad_actions == attacks).any(dim=1)

                good_fight = good_fight & (~bad_actions & ~refills[agent_index])

            attack_powers[fire_coords[good_fight[~refills[agent_index]]].split(1, dim=1)] += full_powers[good_fight].unsqueeze(1)

            # Aggregate the filtered information
            users[agent_index] = good_fight
            bad_users = ~(users[agent_index] | refills[agent_index])

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
        self._state = self.fire_spread_transition(
            state=self._state,
            randomness_source=field_randomness[2],
        )

        fire_rewards = torch.zeros_like(self._state.fires, device=self.device, dtype=torch.float)
        fire_rewards[just_put_out] = self.fire_rewards[just_put_out]
        fire_rewards[just_burned_out] = self.reward_config.burnout_penalty
        fire_rewards_per_batch = fire_rewards.sum(dim=(1, 2))

        self.num_burnouts += just_burned_out.int().sum(dim=(1, 2))

        infos['burnouts'] = just_burned_out.int().sum(dim=(1, 2))
        infos['putouts'] = just_put_out.int().sum(dim=(1, 2))

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

        # Remove all fires from dead batches to enforce termination
        self._state.fires[batch_is_dead].fill_(0)

        # Assign rewards for terminated environments
        newly_terminated = ~self.terminated & batch_is_dead
        termination_penalty = self.reward_config.termination_kappa * torch.log(self.num_burnouts + 1.0)
        termination_reward = self.reward_config.termination_reward - termination_penalty
        termination_reward = torch.clamp(termination_reward, min=0)

        for agent in self.agents:
            rewards[agent][newly_terminated] += termination_reward[newly_terminated]
            self.terminations[agent] |= batch_is_dead

        self.num_burnouts += just_burned_out.int().sum(dim=(1, 2))
        infos['burnouts'] = just_burned_out.int().sum(dim=(1, 2))
        infos['putouts'] = just_put_out.int().sum(dim=(1, 2))

        return rewards, self.terminations, infos

    @torch.no_grad()
    def update_actions(self) -> None:
        """Update the action space for all agents."""
        # Gather all the tasks in the environment
        fires = self._state.fires > 0
        num_tasks_per_environment = fires.sum(dim=(1, 2))
        num_tasks = num_tasks_per_environment.sum()

        # Gather the positions of all of the fires
        fire_positions = fires.nonzero(as_tuple=False)
        fire_positions_expanded = fire_positions.expand(self.agent_config.num_agents, -1, -1)
        fire_positions_expanded = fire_positions_expanded.flatten(end_dim=1)

        # Match up all of the agents with all of the tasks
        agent_indices = torch.arange(0, self.agent_config.num_agents, device=self.device).unsqueeze(1)
        agent_indices_expanded = agent_indices.expand(-1, num_tasks).flatten().unsqueeze(1)
        agent_indices_expanded = torch.cat([fire_positions_expanded[:, 0].unsqueeze(1), agent_indices_expanded], dim=1)
        agent_positions = self._state.agents.unsqueeze(1).expand(-1, num_tasks, -1).flatten(end_dim=1)

        # Calculate the true range for each of the agents in each environment
        agent_ranges = self.agent_config.attack_range[agent_indices_expanded[:, 1]].flatten()
        equipment_states = self._state.equipment[agent_indices_expanded.split(1, dim=1)].squeeze(1)
        range_bonuses = self.agent_config.equipment_states[equipment_states.unsqueeze(0)][:, :, 2].squeeze(0)
        true_range = (agent_ranges + range_bonuses).flatten()

        # Check which agents are in range of which tasks
        in_range = in_range_check.chebyshev(
            agent_position=agent_positions,
            task_position=fire_positions_expanded[:, 1:],
            attack_range=true_range,
        ).reshape(self.agent_config.num_agents, num_tasks)

        # Check which agents have suppressants
        has_suppressants = self._state.suppressants[agent_indices_expanded.split(1, dim=1)].squeeze(1) > 0
        has_suppressants = has_suppressants.reshape(self.agent_config.num_agents, num_tasks)

        # Combine the two checks to determine which agents can fight which fires
        checks = torch.logical_and(in_range, has_suppressants)

        index_shifts = torch.roll(torch.cumsum(num_tasks_per_environment, dim=0), 1, 0)
        index_shifts[0] = 0

        task_range = torch.arange(0, num_tasks, device=self.device)
        task_indices = (task_range - index_shifts[fire_positions[:, 0]].flatten())
        task_indices_nested = torch.nested.as_nested_tensor(
            task_indices.split(num_tasks_per_environment.tolist(), dim=0),
            device=self.device,
            layout=torch.jagged,
        )

        # Aggregate the indices of all tasks to agent mapping
        for agent, agent_number in self.agent_name_mapping.items():
            # Count the number of tasks in each batch and aggregate the indices
            task_count = torch.bincount(fire_positions[checks[agent_number]][:, 0], minlength=self.parallel_envs)
            bad_task_count = num_tasks_per_environment - task_count

            tasks = task_indices[checks[agent_number]]
            batchwise_indices = torch.nested.as_nested_tensor(
                tasks.split(task_count.tolist(), dim=0),
                device=self.device,
                layout=torch.jagged,
            )

            bad_tasks = task_indices[~checks[agent_number]]
            bad_batchwise_indices = torch.nested.as_nested_tensor(
                bad_tasks.split(bad_task_count.tolist(), dim=0),
                device=self.device,
                layout=torch.jagged,
            )

            self.agent_task_count[agent_number] = task_count

            if self.show_bad_actions:
                self.agent_action_mapping[agent] = task_indices_nested
                self.agent_bad_actions[agent] = bad_batchwise_indices
            else:
                self.agent_action_mapping[agent] = batchwise_indices

            self.agent_observation_mapping[agent] = task_indices_nested

        self.environment_task_count = num_tasks_per_environment

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
        fire_observations = torch.nested.as_nested_tensor(
            fire_observations.split(task_count.tolist(), dim=0),
            device=self.device,
            layout=torch.jagged,
        )

        self.task_store = fire_observations

        # Aggregate the full observation space
        self.observations = {}
        for agent in self.agents:
            observation_mask = self.agent_observation_mask(agent)
            agent_index = self.agent_name_mapping[agent]
            agent_mask = torch.ones(self.agent_config.num_agents, dtype=torch.bool, device=self.device)
            agent_mask[agent_index] = False

            self.observations[agent] = TensorDict(
                {
                    'self': agent_observations[:, agent_index],
                    'others': agent_observations[:, agent_mask][:, :, observation_mask],
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
        return observations.build_observation_space(
            environment_task_counts=self.environment_task_count,
            num_agents=self.agent_config.num_agents,
            agent_high=self.agent_observation_bounds,
            fire_high=self.fire_observation_bounds,
            include_suppressant=self.observe_other_suppressant,
            include_power=self.observe_other_power,
        )

    def _log_environment(self, *args, **kwargs) -> None:
        """Log the environment state to a csv."""
        df = pd.DataFrame({
            'burnouts': self.infos.get('burnouts', [None] * self.parallel_envs),
            'putouts': self.infos.get('putouts', [None] * self.parallel_envs),
        })

        super()._log_environment(*args, **kwargs, extra=df)
