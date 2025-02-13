import torch
import numpy as np

from free_range_zoo.envs.wildfire.env.structures.configuration import (WildfireConfiguration, FireConfiguration,
                                                                       AgentConfiguration, StochasticConfiguration,
                                                                       RewardConfiguration)


def non_stochastic() -> WildfireConfiguration:
    """
    Create a non-stochastic configuration for the wildfire environment.

    Returns:
        WildfireConfiguration: The configuration.
    """
    reward_configuration = RewardConfiguration(
        fire_rewards=torch.tensor([[0, 0, 0], [20.0, 50.0, 20.0]], dtype=torch.float32),
        bad_attack_penalty=-100.0,
        burnout_penalty=-1.0,
        termination_reward=0.0,
    )

    fire_configuration = FireConfiguration(
        fire_types=torch.tensor([[0, 0, 0], [1, 2, 1]], dtype=torch.int32),
        num_fire_states=5,
        lit=torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.bool),
        intensity_increase_probability=1.0,
        intensity_decrease_probability=1.0,
        extra_power_decrease_bonus=0.0,
        burnout_probability=1.0,
        base_spread_rate=3.0,
        max_spread_rate=67.0,
        random_ignition_probability=0.0,
        cell_size=200.0,
        wind_direction=0.0 * np.pi,
        ignition_temp=torch.tensor([[2, 2, 2], [2, 2, 2]], dtype=torch.int32),
        initial_fuel=2,
    )

    agent_configuration = AgentConfiguration(
        agents=torch.tensor([[0, 0], [0, 1], [0, 2]], dtype=torch.int32),
        fire_reduction_power=torch.tensor([1, 1, 1], dtype=torch.int32),
        attack_range=torch.tensor([1, 1, 1], dtype=torch.int32),
        suppressant_states=3,
        initial_suppressant=2,
        suppressant_decrease_probability=1.0,
        suppressant_refill_probability=1.0,
        equipment_states=torch.tensor(
            [
                # [-1.0, -0.5, -0.5],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                # [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32),
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
        special_burnout_probability=False,
        suppressant_refill=False,
        suppressant_decrease=False,
        tank_switch=False,
        critical_error=False,
        degrade=False,
        repair=False,
        fire_decrease=False,
        fire_increase=False,
        fire_spread=False,
        realistic_fire_spread=False,
        random_fire_ignition=False,
        fire_fuel=False,
    )

    return WildfireConfiguration(
        grid_width=3,
        grid_height=2,
        fire_config=fire_configuration,
        agent_config=agent_configuration,
        reward_config=reward_configuration,
        stochastic_config=stochastic_configuration,
    )
