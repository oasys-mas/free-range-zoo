"""Configuration classes for the rideshare domain."""
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

    wait_limit: torch.IntTensor
    long_wait_time: int
    general_wait_cost: float
    long_wait_cost: float

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
        schedule: torch.IntTensor: tensor in the shape of <tasks, (timestep, batch, y, x, y_dest, x_dest, fare)>
            where batch can be set to -1 to indicate a wildcard for all batches
    """

    schedule: torch.IntTensor

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
        start_positions: torch.IntTensor - Starting positions of the agents
        pool_limit: int - Maximum number of passengers that can be in a car
        use_diagonal_travel: bool - whether to enable diagonal travel for agents
        use_fast_travel: bool - whether to enable fast travel for agents
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

        agent_config: AgentConfiguration - Agent settings for the rideshare environment.
        passenger_config: PassengerConfiguration - Passenger settings for the rideshare environment.
        reward_config: RewardConfiguration - Reward configuration for the rideshare environment.
    """

    grid_height: int
    grid_width: int

    agent_config: AgentConfiguration
    passenger_config: PassengerConfiguration
    reward_config: RewardConfiguration

    def passenger_entry_transition(self, parallel_envs: int) -> PassengerEntryTransition:
        """Get the passenger entry transition configured for the specific environment."""
        return PassengerEntryTransition(self.passenger_config.schedule, parallel_envs)

    def passenger_exit_transition(self, parallel_envs: int) -> PassengerExitTransition:
        """Get the passenger exit transition configured for the specific environment."""
        return PassengerExitTransition(parallel_envs)

    def passenger_state_transition(self, parallel_envs: int) -> PassengerStateTransition:
        """Get the passenger state transition configured for the specific environment."""
        return PassengerStateTransition(self.agent_config.num_agents, parallel_envs)

    def movement_transition(self, parallel_envs: int) -> PassengerStateTransition:
        """Get the movement transition configured for the specific environment."""
        return MovementTransition(
            self.agent_config.num_agents,
            parallel_envs,
            self.agent_config.use_diagonal_travel,
            self.agent_config.use_fast_travel,
        )

    @functools.cached_property
    def max_fare(self) -> int:
        """Get the maximum fare out of passengers."""
        return self.passenger_config.schedule[:, 6].max().item()

    def validate(self) -> bool:
        """Validate the configuration."""
        super().validate()

        if self.grid_width < 1:
            raise ValueError('grid_width should be greater than 0')
        if self.grid_height < 1:
            raise ValueError('grid_height should be greater than 0')

        return True
