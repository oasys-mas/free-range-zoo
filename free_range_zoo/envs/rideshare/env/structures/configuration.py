"""Configuration classes for the rideshare domain."""
from typing import Tuple, Union, Dict
from dataclasses import dataclass
import functools
import torch

from free_range_zoo.utils.configuration import Configuration
from free_range_zoo.envs.rideshare.env.transitions.passenger_entry import PassengerEntryTransition
from free_range_zoo.envs.rideshare.env.transitions.passenger_exit import PassengerExitTransition
from free_range_zoo.envs.rideshare.env.transitions.passenger_state import PassengerStateTransition
from free_range_zoo.envs.rideshare.env.transitions.movement import MovementTransition


@dataclass
class RewardConfiguration(Configuration):
    """
    Reward settings for rideshare.

    Attributes:
        pick_cost: torch.FloatTensor - Cost of picking up a passenger
        move_cost: torch.FloatTensor - Cost of moving to a new location
        drop_cost: torch.FloatTensor - Cost of dropping off a passenger
        noop_cost: torch.FloatTensor - Cost of taking no action
        accept_cost: torch.FloatTensor - Cost of accepting a passenger
        pool_limit_cost: torch.FloatTensor - Cost of exceeding the pool limit

        use_variable_move_cost: torch.BoolTensor - Whether to use the variable move cost
        use_variable_pick_cost: torch.BoolTensor - Whether to use the variable pick cost
        use_waiting_costs: torch.BoolTensor - Whether to use waiting costs

        wait_limit: List[int] - List of wait limits for each state of the passenger [unaccepted, accepted, riding]
        long_wait_time: int - Time after which a passenger is considered to be waiting for a long time (default maximum of wait_limit)
        general_wait_cost: torch.FloatTensor - Cost of waiting for a passenger
        long_wait_cost: torch.FloatTensor - Cost of waiting for a passenger for a long time (added to wait cost)
    """

    pick_cost: float
    move_cost: float
    drop_cost: float
    noop_cost: float
    accept_cost: float
    pool_limit_cost: float

    use_pooling_rewards: bool
    use_variable_move_cost: bool
    use_waiting_costs: bool

    wait_limit: torch.IntTensor = torch.tensor([1, 2, 3])
    long_wait_time: int = 10
    general_wait_cost: float = -.1
    long_wait_cost: float = -.2

    def validate(self):
        """Validate the configuration."""
        if len(self.wait_limit) != 3:
            raise ValueError('Wait limit should have three elements.')
        if not self.wait_limit.min() > 0:
            raise ValueError('Wait limit elements should all be greater than 0.')
        if not self.long_wait_time > 0:
            raise ValueError('Long wait time should be greater than 0.')


@dataclass
class PassengerConfiguration(Configuration):
    """
    Task settings for rideshare.

    Attributes:
        schedule: TensorDict - tensor dictionary keyed by either
            (batch_index, timestep) of shape <#passengers, 5>
            (timestep) of shape <#passengers, 5> #?where the schedule is duplicated across batches
    """

    schedule: Dict[Union[int, Tuple[int, int]], torch.FloatTensor]

    def validate(self):
        """Validate the configuration."""
        if len(self.schedule.shape) != 2:
            raise ValueError("Schedule should be a 2D tensor")
        if self.schedule.shape[-1] != 7:
            raise ValueError("Schedule should have 7 elements in the last dimesion.")


@dataclass()
class AgentConfiguration(Configuration):
    """
    Agent settings for rideshare.

    Attributes:
        num_agents: int - Number of agents
        pool_limit: int - Maximum number of passengers that can be in a car
        start_positions: torch.IntTensor - Starting positions of the agents
        driving_algorithm: str - Algorithm to use for driving (direct or A*)
    """

    start_positions: torch.IntTensor
    pool_limit: int
    use_diagonal_travel: bool
    use_fast_travel: bool

    @functools.cached_property
    def num_agents(self) -> int:
        """Return the number of agents within the configuration."""
        return self.start_positions.shape[0]

    def validate(self) -> bool:
        """Validate the configuration."""
        if self.pool_limit <= 0:
            raise ValueError("Pool limit must be greater than 0")

        return True


@dataclass()
class RideshareConfiguration(Configuration):
    """
    Configuration settings for rideshare environment.

    Attributes:
        grid_height: int - grid height for the rideshare environment space.
        grid_width: int - grid width for the rideshare environment space.

        passenger_config: PassengerConfiguration - Passenger settings for the rideshare environment.
        agent_config: AgentConfiguration - Agent settings for the rideshare environment.
        reward_config: RewardConfiguration - Reward configuration for the rideshare environment.
    """
    grid_height: int
    grid_width: int

    agent_config: AgentConfiguration
    passenger_config: PassengerConfiguration
    reward_config: RewardConfiguration

    def passenger_entry_transition(self, parallel_envs: int) -> PassengerEntryTransition:
        return PassengerEntryTransition(self.passenger_config.schedule, parallel_envs)

    def passenger_exit_transition(self, parallel_envs: int) -> PassengerExitTransition:
        return PassengerExitTransition(parallel_envs)

    def passenger_state_transition(self, parallel_envs: int) -> PassengerStateTransition:
        return PassengerStateTransition(self.agent_config.num_agents, parallel_envs)

    def movement_transition(self, parallel_envs: int) -> PassengerStateTransition:
        return MovementTransition(
            self.agent_config.num_agents,
            parallel_envs,
            self.agent_config.use_diagonal_travel,
            self.agent_config.use_fast_travel,
        )

    def validate(self) -> bool:
        """Validate the configuration."""
        super().validate()

        if self.grid_width < 1:
            raise ValueError('grid_width should be greater than 0')
        if self.grid_height < 1:
            raise ValueError('grid_height should be greater than 0')

        return True
