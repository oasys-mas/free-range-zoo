from typing import Tuple
import torch
import numpy as np

from free_range_zoo.free_range_zoo.envs.wildfire.env.structures.configuration import GridConf, StochasticConf


def non_stochastic() -> Tuple[GridConf, StochasticConf]:
    """
    Create a non-stochastic configuration for the wildfire environment.

    Returns:
        Tuple[GridConf, StochasticConf]: The grid configuration and the stochastic configuration.
    """
    grid_conf = GridConf(
        grid_width=3,
        grid_height=2,
        lit=torch.tensor([
            [0, 0, 0],
            [1, 1, 1]
        ], dtype=torch.bool),
        fire_types=torch.tensor([
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.int32),
        fire_rewards=torch.tensor([
            [0, 0, 0],
            [20.0, 50.0, 20.0]
        ], dtype=torch.float32),
        ignition_temp=torch.tensor([
            [2, 2, 2],
            [2, 2, 2]
        ], dtype=torch.int32),
        burnable_tiles=((1, 0), (1, 1), (1, 2)),
        num_fire_states=5,
        initial_fuel=2,
        base_spread=3.0,
        max_spread_rate=67.0,
        cell_size=200.0,
        wind_direction=0.0*np.pi,

        agents=torch.tensor([
            [1, 3, 2],
            [0, 0, 0]
        ], dtype=torch.int32),
        attack_range=torch.tensor([1, 1, 1], dtype=torch.int32),
        fire_reduction_power=torch.tensor([1, 1, 1], dtype=torch.int32),
        init_suppressant=2,
        suppressant_states=3,
    )

    stochastic_conf = StochasticConf(
        suppressant_decrease=False,
        suppressant_refill=False,
        BASE_FIRE_REDUCTION=1.0,
        FIRE_REDUCTION_PER_EXTRA_AGENT=0.0,

        use_fire_spread=False,
        use_random_fire_ignition=False,
        use_almost_burned_out=False,
        use_fire_fuel=False,
        realistic_fire_spread=False,
        FIRE_INTENSITY_INCREASE_PROBABILITY=1.0
    )

    return grid_conf, stochastic_conf
