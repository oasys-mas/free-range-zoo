"""Configurations from the AAAI-2025 paper."""
import torch
import numpy as np

from free_range_zoo.envs.wildfire.env.structures.configuration import (
    WildfireConfiguration,
    FireConfiguration,
    AgentConfiguration,
    StochasticConfiguration,
    RewardConfiguration,
)

openness_levels = {1: 25.0, 2: 50.0, 3: 75.0}


def aaai_2025_ol_config(openness_level: int) -> WildfireConfiguration:
    """
    Create the openness level 1, stochastic configuration A from AAAI-2025 paper.

    Args:
        openness_level: int - The openness level of the environment.
    Returns:
        WildfireConfiguration: The configuration with specified openness level.
    """
    if openness_level not in openness_levels:
        raise ValueError('Openness level must be one of 1, 2, or 3.')

    base_spread = openness_levels[openness_level]

    reward_configuration = RewardConfiguration(
        fire_rewards=torch.tensor([[0, 0, 0], [20.0, 50.0, 20.0]], dtype=torch.float32),
        burnout_penalty=-1.0,
        bad_attack_penalty=-100.0,
        termination_reward=0.0,
    )

    fire_configuration = FireConfiguration(
        fire_types=torch.tensor([[0, 0, 0], [1, 2, 1]], dtype=torch.int32),
        num_fire_states=5,
        lit=torch.tensor([[0, 0, 0], [0, 1, 0]], dtype=torch.bool),
        intensity_increase_probability=1.0,
        intensity_decrease_probability=0.8,
        extra_power_decrease_bonus=0.12,
        burnout_probability=4 * 0.167 * 67.0 / 200.0,
        base_spread_rate=base_spread,
        max_spread_rate=67.0,
        random_ignition_probability=0.0,
        cell_size=200.0,
        wind_direction=0.25 * np.pi,
        ignition_temp=torch.tensor([[2, 2, 2], [2, 2, 2]], dtype=torch.int32),
        initial_fuel=2,
    )

    agent_configuration = AgentConfiguration(
        agents=torch.tensor([[0, 0], [0, 2]], dtype=torch.int32),
        fire_reduction_power=torch.tensor([1, 1], dtype=torch.int32),
        attack_range=torch.tensor([1, 1], dtype=torch.int32),
        suppressant_states=3,
        initial_suppressant=2,
        suppressant_decrease_probability=1.0 / 3,
        suppressant_refill_probability=1.0 / 3,
        equipment_states=torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=torch.float32),
        initial_equipment_state=2,
        repair_probability=1.0,
        degrade_probability=1.0,
        critical_error_probability=0.0,
        initial_capacity=2,
        tank_switch_probability=1.0,
        possible_capacities=torch.tensor([1, 2, 3], dtype=torch.float32),
        capacity_probabilities=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
    )

    stochastic_configuration = StochasticConfiguration(
        special_burnout_probability=True,
        suppressant_refill=True,
        suppressant_decrease=True,
        tank_switch=False,
        critical_error=False,
        degrade=False,
        repair=False,
        fire_spread=True,
        realistic_fire_spread=True,
        random_fire_ignition=False,
        fire_fuel=False,
        fire_increase=True,
        fire_decrease=True,
    )

    return WildfireConfiguration(
        grid_width=3,
        grid_height=2,
        fire_config=fire_configuration,
        agent_config=agent_configuration,
        reward_config=reward_configuration,
        stochastic_config=stochastic_configuration,
    )
