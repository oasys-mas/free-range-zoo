import torch
import numpy as np

from free_range_zoo.envs.wildfire.env.structures.configuration import (WildfireConfiguration,
                                                                                      FireConfiguration,
                                                                                      AgentConfiguration,
                                                                                      StochasticConfiguration)



openness_levels = {
    1: 25.0,
    2: 50.0,
    3: 75.0
}


def aaai_2025_ol_config(openness_level:int) -> WildfireConfiguration:
    """
    creates the Openness level 1, stochastic configuration A from AAAI-2025 paper
    """

    assert openness_level in openness_levels, f"Openness level {openness_level} not found in {openness_levels.keys()}"

    #?changed across OLs
    base_spread = openness_levels[openness_level]
    
    max_spread_rate = 67.0
    cell_size = 200.0
    burnout_probability = 4 * 0.167 * max_spread_rate / cell_size

    intensity_decrease_probability = 0.8

    fire_configuration = FireConfiguration(
        fire_types=torch.tensor([
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.int32),
        num_fire_states=5,
        fire_rewards=torch.tensor([
            [0, 0, 0],
            [20.0, 50.0, 20.0]
        ], dtype=torch.float32),
        lit=torch.tensor([
            [0, 0, 0],
            [0, 1, 0]
        ], dtype=torch.bool),

        #stochastics
        intensity_increase_probability=1.0, #int^+
        intensity_decrease_probability=intensity_decrease_probability, #int^-
        burnout_probability=burnout_probability, #int^+_burn

        base_spread_rate=base_spread,
        max_spread_rate=max_spread_rate,
        random_ignition_probability=0.0,

        cell_size=cell_size,
        wind_direction=0.25 * np.pi,

        ignition_temp=torch.tensor([
            [2, 2, 2],
            [2, 2, 2]
        ], dtype=torch.int32),

        initial_fuel=2,
    )


    agent_configuration = AgentConfiguration(
        agents = torch.tensor([
            [0,0],[0,2]
        ], dtype=torch.int32),

        fire_reduction_power=torch.tensor([1, 1], dtype=torch.int32),

        fire_reduction_power_per_extra_agent= intensity_decrease_probability * 0.15,
        
        attack_range=torch.tensor([1, 1], dtype=torch.int32),

        suppressant_states=3,

        initial_suppressant=2,

        suppressant_decrease_probability=1.0 / 3,

        suppressant_refill_probability=1.0 / 3,

        #?equipment states, agent type default

        equipment_states=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32
        ),
        initial_equipment_state=2,
        repair_probability=1.0,
        degrade_probability=1.0,
        critical_error_probability=0.0,

        initial_capacity=2,
        tank_switch_probability=1.0,
        possible_capacities=torch.tensor([1, 2, 3], dtype=torch.int32),
        capacity_probabilities=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
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
    )

    return WildfireConfiguration(
        grid_width=3,
        grid_height=2,

        fire_config=fire_configuration,
        agent_config=agent_configuration,
        stochastic_config=stochastic_configuration
    )


