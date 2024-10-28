from typing import Self, Tuple, Optional, Union, Dict, List
from dataclasses import dataclass

from free_range_zoo.utils.state import State
from free_range_zoo.utils.configuration import Configuration

import torch
from tensordict.tensordict import TensorDict


@dataclass
class RewardConfiguration(Configuration):
    """
    Reward settings for rideshare
    
    Costs:
        pick_cost: torch.FloatTensor - Cost of picking up a passenger
        move_cost: torch.FloatTensor - Cost of moving to a new location
        drop_cost: torch.FloatTensor - Cost of dropping off a passenger
        noop_cost: torch.FloatTensor - Cost of taking no action
        accept_cost: torch.FloatTensor - Cost of accepting a passenger
        pool_limit_cost: torch.FloatTensor - Cost of exceeding the pool limit
        
    Flags:
        use_no_pass_cost: torch.BoolTensor - Whether to use the "no passenger" cost
        use_variable_move_cost: torch.BoolTensor - Whether to use the variable move cost
        use_variable_pick_cost: torch.BoolTensor - Whether to use the variable pick cost
        use_waiting_costs: torch.BoolTensor - Whether to use waiting costs
    
    waiting:
        wait_limit: List[int] - List of wait limits for each state of the passenger [unaccepted, accepted, riding]
        long_wait_time: int - Time after which a passenger is considered to be waiting for a long time (default maximum of wait_limit)
        general_wait_cost: torch.FloatTensor - Cost of waiting for a passenger
        long_wait_cost: torch.FloatTensor - Cost of waiting for a passenger for a long time (added to wait cost)
    
    
    """
    pick_cost: torch.FloatTensor
    move_cost: torch.FloatTensor
    drop_cost: torch.FloatTensor
    noop_cost: torch.FloatTensor
    accept_cost: torch.FloatTensor
    pool_limit_cost: torch.FloatTensor

    use_no_pass_cost: torch.BoolTensor
    use_variable_move_cost: torch.BoolTensor
    use_variable_pick_cost: torch.BoolTensor
    use_waiting_costs: torch.BoolTensor

    wait_limit: Tuple[int] = (5, 5, 10)
    long_wait_time: int = 10
    general_wait_cost: torch.FloatTensor = -.1
    long_wait_cost: torch.FloatTensor = -.2

    def validate(self):
        assert len(self.wait_limit) == 3, "Wait limit should have 3 elements"
        assert all([wait_limit > 0 for wait_limit in self.wait_limit]), "All wait limits should be greater than 0"
        assert self.long_wait_time > 0, "Long wait time should be greater than 0"

        costs_to_check = [
            self.pick_cost, self.move_cost, self.drop_cost, self.noop_cost, self.accept_cost, self.pool_limit_cost,
            self.general_wait_cost, self.long_wait_cost
        ]
        assert all([
            cost.shape == torch.Size([1]) if not isinstance(cost, Union[float, int]) else True for cost in costs_to_check
        ]), "All costs should be tensors of shape [1] or []"

        bools_to_check = [self.use_no_pass_cost, self.use_variable_move_cost, self.use_variable_pick_cost, self.use_waiting_costs]
        assert all([flag.shape == torch.Size([1]) if not isinstance(flag, bool) else True
                    for flag in bools_to_check]), "All bools should be tensors of shape [1] or []"


@dataclass()
class GridConfiguration(Configuration):
    """
    Grid settings for rideshare
    
    Attributes:
        grid_height: int - Height of the grid
        grid_width: int - Width of the grid
        allow_diagonal: bool - Whether diagonal movement is allowed
        fast_travel: bool - Whether time shifts differently for different agents (instantly move to destinations, but incur the same costs)
    """
    grid_height: int
    grid_width: int
    allow_diagonal: bool
    fast_travel: bool

    def validate(self):
        assert self.grid_height > 0, "Grid height must be greater than 0"
        assert self.grid_width > 0, "Grid width must be greater than 0"

        assert isinstance(self.allow_diagonal, bool), "Allow diagonal must be a boolean"
        assert isinstance(self.fast_travel, bool), "Fast travel must be a boolean"


@dataclass
class PassengerConfiguration(Configuration):
    """
    Task settings for rideshare

    schedule: TensorDict - tensor dictionary keyed by either
        (batch_index, timestep) of shape <#passengers, 5>
        (timestep) of shape <#passengers, 5> #?where the schedule is duplicated across batches 
    """
    schedule: Dict[Union[int, Tuple[int, int]], torch.FloatTensor]

    def validate(self):
        assert all([len(schedule.shape) == 2
                    for schedule in self.schedule.values()]), "All instances within a schedule should be 2D tensors"
        assert all([schedule.shape[1] == 5 for schedule in self.schedule.values()
                    ]), "All instances within a schedule should have 5 columns <start_x, start_y, end_x, end_y, fare>"


@dataclass()
class AgentConfiguration(Configuration):
    """
    Agent settings for rideshare
    
    num_agents: int - Number of agents
    pool_limit: int - Maximum number of passengers that can be in a car
    start_positions: torch.IntTensor - Starting positions of the agents (if not specified then random)
    driving_algorithm: str - Algorithm to use for driving (direct or A*)
    """

    num_agents: int
    pool_limit: int
    start_positions: Optional[torch.IntTensor] = None
    driving_algorithm: Optional[str] = 'direct'
    use_relative_distance: Optional[bool] = False

    def validate(self):
        assert self.num_agents > 0, "Number of agents must be greater than 0"
        assert self.pool_limit > 0, "Pool limit must be greater than 0"
        assert self.driving_algorithm in ['direct', 'A*'], "Driving algorithm must be either 'direct' or 'A*'"
        assert isinstance(self.use_relative_distance, bool), "Use relative distance must be a boolean"

        if self.start_positions is not None:
            assert len(
                self.start_positions) == self.num_agents, "Number of start positions should be equal to the number of agents"
            assert all([pos.shape == torch.Size([2])
                        for pos in self.start_positions]), "All start positions should be tensors of shape [2]"


@dataclass()
class RideShareConfig(Configuration):
    grid_conf: GridConfiguration
    passenger_conf: PassengerConfiguration
    agent_conf: AgentConfiguration
    reward_conf: RewardConfiguration

    def validate(self):
        self.grid_conf.validate()
        self.passenger_conf.validate()
        self.agent_conf.validate()
        self.reward_conf.validate()
