import torch

from free_range_zoo.envs.rideshare.env.structures.configuration import RewardConfiguration,\
    PassengerConfiguration, AgentConfiguration, RideshareConfiguration


def non_stochastic():
    """Creates a basic non-stochastic (though rideshare atm is deterministic) configuration for the rideshare environment."""

    agent_conf = AgentConfiguration(
        start_positions=torch.tensor([[0, 0], [9, 9], [0, 9], [9, 0]]),
        pool_limit=4,
        use_fast_travel=False,
        use_diagonal_travel=False,
    )

    reward_conf = RewardConfiguration(
        pick_cost=-0.1,
        move_cost=-0.8,
        drop_cost=0.0,
        noop_cost=-1,
        accept_cost=0.0,
        pool_limit_cost=-2.0,
        use_pooling_rewards=False,
        use_variable_move_cost=True,
        use_waiting_costs=False,
        wait_limit=torch.tensor([1, 2, 3]),
        long_wait_time=10,
        general_wait_cost=-.1,
        long_wait_cost=-.2,
    )

    #simple batch independent schedule
    schedule = torch.tensor([
        [0, -1, 1, 1, 1, 1, 1],
        [1, -1, 1, 1, 1, 1, 2],
        [2, 1, 1, 1, 1, 1, 3],
    ], dtype=torch.int)

    passenger_conf = PassengerConfiguration(schedule=schedule)

    configuration = RideshareConfiguration(grid_height=10,
                                           grid_width=10,
                                           agent_config=agent_conf,
                                           reward_config=reward_conf,
                                           passenger_config=passenger_conf)

    return configuration
