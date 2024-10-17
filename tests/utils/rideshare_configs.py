import torch

from free_range_zoo.free_range_zoo.envs.rideshare.env.structures.configuration import RewardConfiguration,\
    PassengerConfiguration, AgentConfiguration, GridConfiguration, RideShareConfiguration



def non_stochastic():
    """Creates a basic non-stochastic (though rideshare atm is deterministic) configuration for the rideshare environment."""

    grid_conf = GridConfiguration(
        grid_height= 10,
        grid_width= 10,
        fast_travel= False,
        allow_diagonal= False,
    )

    agent_conf = AgentConfiguration(
        num_agents= 2,
        pool_limit= 4,
    )

    reward_conf = RewardConfiguration(
        pick_cost= -0.1,
        move_cost= -0.8,
        drop_cost= 0.0,
        noop_cost= -1,
        accept_cost= 0.0,
        pool_limit_cost= -2.0, 

        use_no_pass_cost= False,
        use_variable_move_cost= False,
        use_variable_pick_cost=True,
        use_waiting_costs= False
    )

    #simple batch independent schedule
    schedule = {
        0:torch.tensor( [[1,1, 8,8, 13]]),
        1:torch.tensor( [[7,3, 5,3, 10]]),
        3:torch.tensor( [[6,0, 7,2, 12]]),
        4:torch.tensor( [[1,4, 5,1, 10]]),
        10:torch.tensor( [[1,5, 4,3, 15]]),
    }

    passenger_conf = PassengerConfiguration(schedule=schedule)

    configuration = RideShareConfiguration(
        grid_conf=grid_conf,
        agent_conf=agent_conf,
        reward_conf=reward_conf,
        passenger_conf=passenger_conf
    )

    return configuration