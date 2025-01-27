"""Configurations classes for the wildfire environments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import functools
import torch
import numpy as np

from free_range_zoo.utils.configuration import Configuration


@dataclass
class RewardConfiguration(Configuration):
    """
    Settings for configuring the reward function.

    Attributes:
        fire_rewards: torch.FloatTensor - Reward for extinguishing a fire
        bad_attack_penalty: float - Penalty for attacking a tile that is not on fire
        burnout_penalty: float - Penalty for attacking a burned out fire
        termination_reward: float - Reward for terminating the environment
    """

    fire_rewards: torch.FloatTensor
    bad_attack_penalty: float
    burnout_penalty: float
    termination_reward: float

    def validate(self) -> bool:
        """
        Validate the configuration to ensure logical consistency.

        Returns:
            bool - True if the configuration is valid
        """
        if len(self.fire_rewards.shape) != 2:
            raise ValueError('fire_rewards should be a 2D tensor')

        return True


@dataclass
class FireConfiguration(Configuration):
    """
    Setting for configuring fire properties in the environment.

    Attributes:
        fire_types: torch.IntTensor - Required attack power in order to extinguish the fire
        num_fire_states: int - Number of fire states
        lit: torch.IntTensor - Tensor representing the initially lit tiles
        intensity_increase_probability: float - Probability of fire intensity increase
        intensity_decrease_probability: float - Probability of fire intensity decrease
        extra_power_decrease_bonus: float - Additional decrease bonus per extra power
        burnout_probability: float - Probability of fire burnout
        base_spread_rate: float - Base spread rate of the fire
        max_spread_rate: float - Maximum spread rate of the fire
        random_ignition_probability: float - Probability of random ignition
        cell_size: float - Size of each cell
        wind_direction: float - Direction of the wind (radians)
        ignition_temp: torch.IntTensor - Initial intensity of each fire once ignited
        initial_fuel: int - Initial fuel value of each cell in the grid, controls the number of re-ignitions
    """

    fire_types: torch.IntTensor
    num_fire_states: int
    lit: torch.Tensor
    intensity_increase_probability: float
    intensity_decrease_probability: float
    extra_power_decrease_bonus: float
    burnout_probability: float

    base_spread_rate: float
    max_spread_rate: float
    random_ignition_probability: float
    cell_size: float
    wind_direction: float
    ignition_temp: torch.IntTensor
    initial_fuel: int

    @functools.cached_property
    def realistic_burnout_probability(self) -> float:
        """Return the burnout probability with realistic spread rates."""
        return 4 * 0.167 * self.grid_conf.max_spread_rate / self.grid_conf.cell_size

    @functools.cached_property
    def burned_out(self) -> int:
        """Return the burned out fire state."""
        return self.num_fire_states - 1

    @functools.cached_property
    def almost_burned_out(self) -> int:
        """Return the alomst burned out fire state."""
        return self.num_fire_states - 2

    @functools.cached_property
    def max_fire_type(self) -> int:
        """Return the maximum fire type."""
        return self.fire_types.max().item()

    @functools.cached_property
    def realistic_spread_rates(self) -> List[float]:
        """Return the spread rates in each direction calculated with the wind direction."""
        cell_spread_factor = self.base_spread_rate / self.cell_size
        max_spread_factor = 1 - self.base_spread_rate / self.max_spread_rate

        cos_terms = [
            np.cos(0 - self.wind_direction),  # North
            np.cos(0.5 * np.pi - self.wind_direction),  # East
            np.cos(np.pi - self.wind_direction),  # South
            np.cos(1.5 * np.pi - self.wind_direction)  # West
        ]

        spread_rates = [cell_spread_factor / (1 - cos_term * max_spread_factor) for cos_term in cos_terms]
        return spread_rates

    def validate(self) -> bool:
        """
        Validate the configuration to ensure logical consistency.

        Returns:
            bool - True if the configuration is valid
        """
        if len(self.fire_types.shape) != 2:
            raise ValueError('fires should be a 2D tensor')
        if self.num_fire_states < 4:
            raise ValueError('num_fire_states should be greater than 4')
        if len(self.lit.shape) != 2:
            raise ValueError('lit should be a 2D tensor')
        if self.intensity_increase_probability > 1 or self.intensity_increase_probability < 0:
            raise ValueError('intensity_increase_probability should be between 0 and 1')
        if self.intensity_decrease_probability > 1 or self.intensity_decrease_probability < 0:
            raise ValueError('intensity_decrease_probability should be between 0 and 1')
        if self.burnout_probability > 1 or self.burnout_probability < 0:
            raise ValueError('burnout_probability should be between 0 and 1')

        if self.random_ignition_probability > 1 or self.random_ignition_probability < 0:
            raise ValueError('random_ignition_probability should be between 0 and 1')
        if not (0.0 <= self.wind_direction <= 2 * np.pi):
            raise ValueError("Wind direction must be between 0 and 2 * pi")

        if not (self.lit.shape == self.fire_types.shape == self.ignition_temp.shape):
            raise ValueError("lit, fire_types, and ignition_temp must have the same shape")

        return True


@dataclass
class AgentConfiguration(Configuration):
    """
    Setting for configuring agent properties in the environment.

    Attributes:
        agents: torch.IntTensor - Tensor representing the location of each agent
        fire_reduction_power: torch.FloatTensor - Power of each agent to reduce the fire intensity
        attack_range: torch.Tensor - Range of attack for each agent
        suppressant_states: int - Number of suppressant states
        initial_suppressant: int - Initial suppressant value for each agent
        suppressant_decrease_probability: float - Probability of suppressant decrease
        suppressant_refill_probability: float - Probability of suppressant refill
        intial_equipment_state: int - Initial equipment state for each agent
        equipment_states: torch.FloatTensor - Definition of equipment states modifiers in the form of (capacity, power, range)
        repair_probability: float - Probability that an agent get their repaired equipment once fully damaged
        degrade_probability: float - Probability that an agent's tank will degrade
        critical_error_probability: float - Probability that an agent at full will suffer a critical error
        tank_switch_probability: float - Probability that an agent will be supplied with a different tank on refill
        possible_capacities: torch.Tensor - Possible maximum suppressant values
        capacity_probabilities: torch.Tensor - Probability that each suppressant maximum is chosen
    """

    agents: torch.IntTensor
    fire_reduction_power: torch.FloatTensor
    attack_range: torch.Tensor

    suppressant_states: int
    initial_suppressant: int
    suppressant_decrease_probability: float
    suppressant_refill_probability: float

    initial_equipment_state: int
    equipment_states: torch.FlaotTensor
    repair_probability: float
    degrade_probability: float
    critical_error_probability: float

    initial_capacity: int
    tank_switch_probability: float
    possible_capacities: torch.Tensor
    capacity_probabilities: torch.Tensor

    @functools.cached_property
    def num_agents(self) -> int:
        """Return the number of agents."""
        return self.agents.shape[0]

    @functools.cached_property
    def max_fire_reduction_power(self) -> float:
        """Return the maximum fire reduction power of the agents."""
        return self.fire_reduction_power.max().item()

    @functools.cached_property
    def num_equipment_states(self) -> int:
        """Return the number of equipment states."""
        return self.equipment_states.shape[0]

    def validate(self) -> bool:
        if len(self.agents.shape) != 2:
            raise ValueError('agents should be a 2D tensor')
        if len(self.fire_reduction_power.shape) != 1:
            raise ValueError('fire_reduction_power should be a 1D tensor')
        if len(self.attack_range.shape) != 1:
            raise ValueError('attack_range should be a 1D tensor')
        if self.agents.shape[0] != self.agents.shape[0] or self.agents.shape[0] != self.fire_reduction_power.shape[0]:
            raise ValueError('agents, fire_reduction_power, and attack_range should have the same length')

        if self.suppressant_states < 2:
            raise ValueError('suppressant_states should be greater than 2')
        if self.initial_suppressant > self.suppressant_states:
            raise ValueError('init_suppressant should be less than suppressant_states')
        if self.suppressant_decrease_probability > 1 or self.suppressant_decrease_probability < 0:
            raise ValueError('suppressant_use_probability should be between 0 and 1')
        if self.suppressant_refill_probability > 1 or self.suppressant_refill_probability < 0:
            raise ValueError('suppressant_refill_probability should be between 0 and 1')

        if self.initial_equipment_state > self.equipment_states.shape[0]:
            raise ValueError('initial_equipment_state should be less than the number of equipment states')
        if len(self.equipment_states.shape) != 2:
            raise ValueError('equipment_states should be a 2D tensor')
        if self.equipment_states.shape[1] != 3:
            raise ValueError('equipment_states should have 3 modifers: suppressant maximum, power, range')
        if self.repair_probability > 1 or self.repair_probability < 0:
            raise ValueError('repair_probability should be between 0 and 1')
        if self.degrade_probability > 1 or self.degrade_probability < 0:
            raise ValueError('degrade_probability should be between 0 and 1')
        if self.critical_error_probability > 1 or self.critical_error_probability < 0:
            raise ValueError('critical_error_probability should be between 0 and 1')
        if self.degrade_probability + self.critical_error_probability > 1:
            raise ValueError('degrade_probability + critical_error_probability should be less than or equal to 1')

        if self.tank_switch_probability > 1 or self.tank_switch_probability < 0:
            raise ValueError('tank_switch_probability should be between 0 and 1')
        if len(self.possible_capacities.shape) != 1:
            raise ValueError('possible_suppressant_maximums should be a 1D tensor')
        if len(self.capacity_probabilities.shape) != 1:
            raise ValueError('suppressant_maximum_probabilities should be a 1D tensor')

        if self.possible_capacities.shape[0] != self.capacity_probabilities.shape[0]:
            raise ValueError('possible_suppressant_maximums and suppressant_maximum_probabilities should have the same length')
        if self.possible_capacities.min() < 1:
            raise ValueError('possible_suppressant_maximums should be greater than 1')
        if self.capacity_probabilities.sum().item() != 1:
            raise ValueError('suppressant_maximum_probabilities should sum to 1')


@dataclass
class StochasticConfiguration(Configuration):
    """
    Configuration for the stochastic elements of the environment.

    Attributes:
        special_burnout_probability: bool - Whether to use special burnout probabilities
        suppressant_refill: bool - Whether suppressants refill stochastically
        suppressant_decrease: bool - Whether suppressants decrease stochastically
        tank_switch: bool - Whether to use stochastic tank switching
        critical_error: bool - Whether equipment state can have a critical error
        degrade: bool - Whether equipment state stochastically degrades
        repair: bool - Whether equipment state stochastically repairs
        fire_decrease: bool - Whether fires decrease stochastically
        fire_increase: bool - Whether fires increase stochastically
        fire_spread: bool - Whether fires spread
        realistic_fire_spread: bool - Whether fires spread realistically
        random_fire_ignition: bool - Whether fires can ignite randomly
        fire_fuel: bool - Whether fires consume fuel and have limited ignitions
    """

    special_burnout_probability: bool

    suppressant_refill: bool
    suppressant_decrease: bool

    tank_switch: bool
    critical_error: bool
    degrade: bool
    repair: bool

    fire_increase: bool
    fire_decrease: bool
    fire_spread: bool
    realistic_fire_spread: bool
    random_fire_ignition: bool
    fire_fuel: bool

    def validate(self) -> bool:
        """
        Validate the configuration to ensure logical consistency.

        Returns:
            bool - True if the configuration is valid
        """
        if not self.fire_spread and self.realistic_fire_spread:
            raise ValueError('Cannot use realistic fire spread without fire spread')

        if self.critical_error and not self.degrade:
            raise ValueError('Cannot have critical errors without equipment degradation')

        return True


@dataclass
class WildfireConfiguration(Configuration):
    """
    Configuration for the wildfire environment.

    Attributes:
        grid_width: int - Width of the grid
        grid_height: int - Height of the grid
        fire_configuration: FireConfiguration - Configuration for the fire properties
        agent_configuration: AgentConfiguration - Configuration for the agent properties
        reward_configuration: RewardConfiguration - Configuration for the environment rewards
        stochastic_configuration: StochasticConf - Configuration for the stochastic elements
    """

    grid_width: int
    grid_height: int

    fire_config: FireConfiguration
    agent_config: AgentConfiguration
    reward_config: RewardConfiguration
    stochastic_config: StochasticConfiguration

    @functools.cached_property
    def fire_spread_weights(self) -> torch.Tensor:
        """Return the fire spread weights with the current fire and stochastic configuration."""
        if not self.stochastic_config.fire_spread:
            return torch.zeros((1, 1, 3, 3), dtype=torch.float32)

        if self.stochastic_config.realistic_fire_spread:
            Ns, Es, Ss, Ws = self.fire_config.realistic_spread_rates
        else:
            Ns, Es, Ss, Ws = [self.fire_config.base_spread_rate] * 4

        fire_filter = torch.tensor([
            [0.0, Ns, 0.0],
            [Ws, 0.0, Es],
            [0.0, Ss, 0.0],
        ], dtype=torch.float32)

        if self.stochastic_config.random_fire_ignition:
            fire_filter[1, 1] = fire_filter[1, 1] + self.fire_config.random_ignition_probability

        return fire_filter.unsqueeze(0).unsqueeze(0).to(torch.float32)

    def validate(self) -> bool:
        """
        Validate the configuration to ensure logical consistency.

        Returns:
            bool - True if the configuration is valid
        """
        super().validate()

        if self.grid_width < 1:
            raise ValueError('grid_width should be greater than 0')
        if self.grid_height < 1:
            raise ValueError('grid_height should be greater than 0')

        if not (self.fire_config.lit.shape == self.reward_config.fire_rewards.shape):
            raise ValueError('lit and fire_rewards should have the same shape')

        return True
