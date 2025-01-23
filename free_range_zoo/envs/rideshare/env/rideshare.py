"""
# Rideshare

---

| Import             | `from freerangezoo.otask import rideshare_v0` |
|--------------------|------------------------------------|
| Actions            | Discrete and perfect                            |
| Observations | Discrete and fully observed with private observations |
| Parallel API       | Yes                                |
| Manual Control     | No                                 |
| Agent Names             | [$driver$_0, ..., $driver$_n] |
| #Agents             |    $n$                                  |
| Action Shape       | (envs, 2)                 |
| Action Values      | [-1, $\|tasks\|$], [-1,2]\*                    |
| Observation Shape | TensorDict: { <br> &emsp; **Agent's self obs**, <ins>'self'</ins>: 5 `<agent index, ypos, xpos, #accepted passengers, #riding passengers>`, <br> &emsp; **Other agent obs**, <ins>'others'</ins>: ($\|Ag\| \times 5$) `<agent index, ypos, xpos, #accepted passengers, #riding passengers>`, <br> &emsp; **Fire/Task obs**, <ins>'tasks'</ins>: ($\|X\| \times 5$) `<task index, ystart, xstart, yend, xend, acceptedBy, ridingWith, fare, time entered>` <br> **batch_size: `num_envs`** <br>}|
| Observation Values   | <ins>self</ins> <br> **agent index**: [0,n), <br> **ypos**: [0,grid_height], <br> **xpos**: [0, grid_width], <br> **number accepted passengers**: [0, $\infty$), <br> **number riding passengers**: [0,$\infty$) <br> <br> <ins>others</ins> <br> **agent index**: [0,$n$), <br> **ypos**: [0,grid_height], <br> **xpos**: [0, grid_width], <br> **number accepted passengers**: [0, $\infty$), <br> **number riding passengers**: [0,$\infty$)  <br> <br> <ins>tasks</ins> <br> **task index**: [0, $\infty$), <br> **ystart**: [0,grid_height], <br> **xstart**: [0, grid_width], <br> **yend**: [0, grid_height], <br> **xend**: [0,grid_width] <br> **accepted by**: [0,$n$) <br> **riding with**: [0,$n$), <br> **fare**: (0, $\infty$), <br> **time entered**: [0,max steps] |

<hr>
"""

from typing import Dict, Any, Union, List, Optional, Tuple, Callable
from collections import defaultdict
import warnings

import numpy as np

import torch
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
import torch_geometric as tg
import gymnasium

import numpy as np

from pettingzoo.utils.wrappers import OrderEnforcingWrapper

from free_range_zoo.utils.env import BatchedAECEnv
from free_range_zoo.utils.conversions import batched_aec_to_batched_parallel
from free_range_zoo.envs.rideshare.env.structures.state import RideshareState
from free_range_zoo.envs.rideshare.env.utils.schedule import create_random_schedule
from free_range_zoo.envs.rideshare.env.utils.direct_distance import DirectPath
from free_range_zoo.envs.rideshare.env.utils.action_space_modifier import action_space_OneOf_adjuster
from free_range_rust import Space

# ?spaces used in action_space(agent)
accept = Space.Discrete(1, start=0)
noop = Space.Discrete(1, start=-1)
pickup = Space.Discrete(1, start=1)
dropoff = Space.Discrete(1, start=2)
action_choice = {-1: noop, 0: accept, 1: pickup, 2: dropoff}


def parallel_env(wrappers: List[Callable] = [], **kwargs) -> BatchedAECEnv:
    """
    Paralellized version of the rideshare environment.

    Args:
        wrappers: List[Callable[[BatchedAECEnv], BatchedAECEnv]] - the wrappers to apply to the environment
    Returns:
        BatchedAECEnv: the parallelized rideshare environment
    """
    env = raw_env(**kwargs)
    env = OrderEnforcingWrapper(env)

    for wrapper in wrappers:
        env = wrapper(env)

    env = batched_aec_to_batched_parallel(env)
    return env


def env(wrappers: List[Callable] = [], **kwargs) -> BatchedAECEnv:
    """
    AEC wrapped version of the rideshare environment.

    Args:
        wrappers: List[Callable[[BatchedAECEnv], BatchedAECEnv]] - the wrappers to apply to the environment
    Returns:
        BatchedAECEnv: the rideshare environment
    """
    env = raw_env(**kwargs)
    env = OrderEnforcingWrapper(env)

    for wrapper in wrappers:
        env = wrapper(env)

    return env


class raw_env(BatchedAECEnv):
    metadata = {"render.modes": ["human", "rgb_array"], "name": "rideshare_v0", "is_parallelizable": True, "render_fps": 2}

    @torch.no_grad()
    def __init__(
            self,
            *args,
            per_batch_buffer_allocation_size: int = 20,
            offer_hyperedges: bool = False,
            expand_as_neeeded: bool = False,  #expand tensor beyond PER_BATCH_PASSANGER_BUFFER_ALLOCATION_SIZE if needed       
            **kwargs):
        super().__init__(*args, **kwargs)

        #?do we support hyperedges
        self.provide_hyperedges = True
        self.offer_hyperedges = offer_hyperedges

        #set grid parameters
        self.grid_height, self.grid_width = self.grid_conf.grid_height, self.grid_conf.grid_width
        self.fast_travel = self.grid_conf.fast_travel

        #set agent parameters
        self.possible_agents: List[str] = [f'driver_{i}' for i in range(self.agent_conf.num_agents)]
        self.fixed_start_positions: Dict[str, torch.Tensor] = self.agent_conf.start_positions

        #set the driving algorithm
        if self.agent_conf.driving_algorithm == "direct":
            assert self.grid_height == self.grid_width, "DirectPath only supports square grids"  #?can be made to work with non-square grids just for now

            self.driving_algorithm = DirectPath(grid_size=self.grid_height,
                                                allow_diagonal=self.grid_conf.allow_diagonal,
                                                fast_travel=self.fast_travel,
                                                device=self.device)
        elif self.agent_conf.driving_algorithm == "astar":
            raise NotImplementedError("A* driving algorithm not implemented yet")
        else:
            raise ValueError("Invalid driving algorithm, expected 'direct' or 'astar'")

        #TODO this can be updated to a list for frame openness
        self.pool_limit: int = self.agent_conf.pool_limit

        #decides if observed distances are relative to the agent or absolute.
        self.use_relative_distance = self.agent_conf.use_relative_distance
        assert not self.use_relative_distance, "Relative distance directional encoding not implemented yet"

        #set reward parameters
        self.pick_cost: float = torch.tensor(self.reward_conf.pick_cost, dtype=torch.float32, device=self.device)
        self.move_cost: float = torch.tensor(self.reward_conf.move_cost, dtype=torch.float32, device=self.device)
        self.drop_cost: float = torch.tensor(self.reward_conf.drop_cost, dtype=torch.float32, device=self.device)
        self.noop_cost: float = torch.tensor(self.reward_conf.noop_cost, dtype=torch.float32, device=self.device)
        self.accept_cost: float = torch.tensor(self.reward_conf.accept_cost, dtype=torch.float32, device=self.device)
        self.pool_limit_cost: float = torch.tensor(self.reward_conf.pool_limit_cost, dtype=torch.float32, device=self.device)
        self.general_wait_cost: float = torch.tensor(self.reward_conf.general_wait_cost, dtype=torch.float32, device=self.device)
        self.long_wait_cost: float = torch.tensor(self.reward_conf.long_wait_cost, dtype=torch.float32, device=self.device)

        self.wait_limit: Tuple[int] = self.reward_conf.wait_limit
        self.long_wait_time: int = self.reward_conf.long_wait_time

        self.expand_as_neeeded = expand_as_neeeded
        self.per_batch_buffer_allocation_size: int = per_batch_buffer_allocation_size

        self.use_no_pass_cost: bool = self.reward_conf.use_no_pass_cost
        if self.use_no_pass_cost:
            warnings.warn("No passenger cost not implemented yet")

        self.use_variable_move_cost: bool = self.reward_conf.use_variable_move_cost
        if self.use_variable_move_cost:
            warnings.warn(
                "Variable move cost here refers to  move_cost = move_cost/(num riding passengers) not the 'use_variable_move_cost' which was unused in original imp"
            )

        self.use_variable_pick_cost: bool = self.reward_conf.use_variable_pick_cost
        if self.use_variable_pick_cost:
            warnings.warn("Variable pick cost not implemented yet. Not used in original implementation")

        self.use_waiting_costs = self.reward_conf.use_waiting_costs

        #passenger scheduling parameters (migrate to device)
        if isinstance(self.passenger_conf.schedule, str):
            assert self.passenger_conf.schedule == "random", "Only random scheduling or directly passed schedules are supported for now"

            #will randomly generate a schedule at each reset
            if self.passenger_conf.schedule == "random":
                self.passenger_schedule: Dict[Union[int, Tuple[int, int]], torch.FloatTensor] = create_random_schedule(
                    gridLength=self.grid_height,
                    gridWidth=self.grid_width,
                    reward_config=self.reward_conf,
                    num_agents=self.agent_conf.num_agents,
                    num_envs=self.parallel_envs,
                    passenger_config=self.passenger_conf,
                    max_timesteps=self.max_steps,
                )

            else:
                raise ValueError("The only autogenerated schedule is 'random'")
        else:
            self.passenger_schedule: Dict[Union[int, Tuple[int, int]], torch.FloatTensor] = {
                k: v.to(self.device)
                for k, v in self.passenger_conf.schedule.items()
            }

        #?find the last scheduled task arrival
        __passenger_scheduled_steps = list(self.passenger_schedule.keys())

        if isinstance(__passenger_scheduled_steps[0], tuple) or isinstance(__passenger_scheduled_steps[0], list):
            self._last_scheduled_arrival: int = max([step[1] for step in __passenger_scheduled_steps])
        else:
            self._last_scheduled_arrival: int = max(__passenger_scheduled_steps)

        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}

        #TODO this must be changed if made agent open
        self.observation_ordering: Dict[str, torch.IntTensor] = {
            self.agent_name_mapping[this_agent]:
            torch.tensor([self.agent_name_mapping[ag] for ag in self.possible_agents if ag != this_agent])
            for this_agent in self.possible_agents
        }

    def _apply_schedule(self, batch_indices: List[int]) -> None:
        """
        Queries schedule for new passengers to use at this timestep
        
        Args:
            batch_indices: List[int] - the batch indices to apply the schedule to.
        
        Reminder* the schedule is a tensor of shape <batch, timesteps, num_passengers, 3> or 
            <timesteps,num_passengers, 3> where the columns are <start, dest, fare>
        """

        try:
            sample_key = list(self.passenger_schedule.keys())[0]
        except Exception as e:
            print(e)
            raise ValueError("Invalid Schedule must have at least one entry")

        #check if schedule is batched
        if isinstance(sample_key, tuple):
            new_passengers = []
            for batch_index in batch_indices:
                key = (batch_index, self._step_count)

                #?If this batch at this timestep exists add it otherwise don't
                if key in self.passenger_schedule.keys():
                    found_passengers = self.passenger_schedule[key]
                    new_passengers.append(
                        torch.cat([
                            torch.ones(found_passengers.shape[0], device=self.device).unsqueeze(1) * batch_index, found_passengers
                        ],
                                  dim=-1))

            #return if no passengers found
            if len(new_passengers) == 0:
                return

            #?concatenate all the new passengers <#passengers, 6>
            new_passengers = torch.cat(new_passengers, dim=0)

        #not batched thus dupe
        else:
            if self._step_count in self.passenger_schedule.keys():
                found_passengers = self.passenger_schedule[self._step_count]

                #?Cat the batch index to the front of the tensor and duplicate for each batch  <#passengers, 6>
                new_passengers = torch.cat([
                    torch.cat(
                        [torch.ones(found_passengers.shape[0], device=self.device).unsqueeze(1) * batch_idx, found_passengers],
                        dim=-1) for batch_idx in batch_indices
                ],
                                           dim=0)

            #no new passengers
            else:
                return

        #find the locations and destinations
        locations = new_passengers[:, [0, 1, 2]]
        destinations = new_passengers[:, [0, 3, 4]]

        #set the accepted by, riding with params to NULL
        associations = new_passengers[:, [0, 5, 5, 5]]
        associations[:, [1, 2]] = torch.nan

        #set timings
        timings = new_passengers[:, [0, 5, 5, 5, 5]]
        timings[:, 1] = self._step_count
        timings[:, 2:] = -42

        # ?Apply to state

        # Check if space is available
        needed_space = new_passengers.shape[0]
        free_space = ~self._state.used_space.clone()

        # ?Expand statespace if needed
        # if self._state.used_space.sum()+needed_space > 3 * self.per_batch_buffer_allocation_size / 4:
        #     if self.expand_as_neeeded:
        #         self._expand_state()
        #         free_space = ~self._state.used_space.clone()
        #     else:
        #         raise RuntimeError("Not enough space in state to add new passengers")

        # Get unused space
        masked_loc, masked_dest, masked_assoc, masked_timing = \
            self._state.locations[free_space],\
            self._state.destinations[free_space],\
            self._state.associations[free_space],\
            self._state.timing[free_space]

        # Apply to masked region
        masked_loc[:needed_space] = locations
        masked_dest[:needed_space] = destinations
        masked_assoc[:needed_space] = associations

        masked_timing[:needed_space] = timings
        masked_timing[:needed_space, [2, 3, 4]] = torch.nan

        # Update state
        self._state.locations[free_space] = masked_loc
        self._state.destinations[free_space] = masked_dest
        self._state.associations[free_space] = masked_assoc
        self._state.timing[free_space] = masked_timing

        # Mark space as used
        free_space[:needed_space + torch.sum(self._state.used_space)] = False
        self._state.used_space = torch.logical_not(free_space)

    @torch.no_grad()
    def reset(self, seed: Optional[List[int]] = None, options: Optional[Dict[str, Any]] = None):
        """
        Resets the environment

        Args:
            seed: Union[List[int], int] - the seed to use
            options: Dict[str, Any] - the options for the reset
        """
        super().reset(seed=seed, options=options)

        self._step_count = 0

        # Clear dictionary storing actions for each agent
        self.actions = {agent: torch.empty(self.parallel_envs, 2) for agent in self.agents}

        self.agent_task_indices: Dict[str, torch.nested.nested_tensor] = {}
        self.environment_task_indices: torch.nested.nested_tensor = None
        self.agent_action_indices: Dict[str, List[torch.IntTensor]] = {}

        self._state = RideshareState(
            agents=torch.empty(
                (self.parallel_envs * self.num_agents, 4),
                device=self.device,
            ),
            locations=torch.empty(
                (self.parallel_envs * self.per_batch_buffer_allocation_size, 3),
                device=self.device,
            ),
            destinations=torch.empty(
                (self.parallel_envs * self.per_batch_buffer_allocation_size, 3),
                device=self.device,
            ),
            associations=torch.empty(
                (self.parallel_envs * self.per_batch_buffer_allocation_size, 4),
                device=self.device,
            ),
            timing=torch.empty(
                (self.parallel_envs * self.per_batch_buffer_allocation_size, 5),
                device=self.device,
                dtype=torch.float,
            ),
            used_space=torch.zeros(
                self.parallel_envs * self.per_batch_buffer_allocation_size,
                device=self.device,
                dtype=torch.bool,
            ),
        )

        # Define agent batch indices
        self._state.agents[:, 0] = torch.arange(0, self.parallel_envs, device=self.device).repeat_interleave(self.num_agents)
        self._state.agents[:, 1] = torch.arange(0, self.num_agents, device=self.device).repeat(self.parallel_envs)

        #build passengers
        #recompute schedule if random
        if isinstance(self.passenger_conf.schedule, str) and self.passenger_conf.schedule == "random":
            self.passenger_schedule = create_random_schedule(
                gridLength=self.grid_height,
                gridWidth=self.grid_width,
                reward_config=self.reward_conf,
                num_agents=self.num_agents,
                num_envs=self.parallel_envs,
                passenger_config=self.passenger_conf,
                max_timesteps=self.max_steps,
            )

        self._apply_schedule(list(range(self.parallel_envs)))

        # Set agent positions
        if self.fixed_start_positions is None:
            self._state.agents[:, 2] = torch.randint(
                0,
                self.grid_height,
                (self.parallel_envs * self.num_agents, ),
                device=self.device,
            )
            self._state.agents[:, 3] = torch.randint(
                0,
                self.grid_width,
                (self.parallel_envs * self.num_agents, ),
                device=self.device,
            )

        else:
            self._state.agents[:, 2] = self.fixed_start_positions[:, 0].repeat_interleave(self.parallel_envs)
            self._state.agents[:, 3] = self.fixed_start_positions[:, 1].repeat_interleave(self.parallel_envs)

        # Set the observations and action space
        if not options or not options.get('skip_observations', False):
            self.update_observations()
        if not options or not options.get('skip_actions', False):
            self.update_actions()

        self._post_reset_hook()

    @torch.no_grad()
    def reset_batches(self,
                      batch_indices: List[int],
                      seed: List[int] | None = None,
                      options: Dict[str, Any] | None = None) -> None:
        """
        
        """

        raise NotImplementedError("Reset batches not implemented yet. self._time_count is not batch specific yet.")

        self._step_count = 0

        agent_reset_mask = self._state.agents[:, 0].isin(batch_indices)
        passenger_reset_mask = self._state.locations[:, 0].isin(batch_indices)

        #?set the agent positions
        if self.fixed_start_positions is None:
            self._state.agents[agent_reset_mask, 2] = torch.randint(0,
                                                                    self.grid_height, (self.parallel_envs * self.num_agents, ),
                                                                    device=self.device)[agent_reset_mask]
            self._state.agents[agent_reset_mask, 3] = torch.randint(0,
                                                                    self.grid_width, (self.parallel_envs * self.num_agents, ),
                                                                    device=self.device)[agent_reset_mask]

        else:
            self._state.agents[agent_reset_mask,
                               2] = self.fixed_start_positions[:, 1].repeat_interleave(self.parallel_envs)[agent_reset_mask]
            self._state.agents[agent_reset_mask,
                               3] = self.fixed_start_positions[:, 2].repeat_interleave(self.parallel_envs)[agent_reset_mask]

        #?mark passenger data as stale
        self._state.used_space[passenger_reset_mask] = False

        #?adding new passengers for this batch
        self._apply_schedule(batch_indices)

        # Reset the observation updates
        self.update_observations()
        self.update_actions()

    @torch.no_grad()
    def _expand_state(self) -> None:
        """
        Increases the size of the state buffer

        Expands by  #envs * #per_batch_buffer_allocation_size // 4
        """
        self._state.locations = torch.cat(
            (self._state.locations,
             torch.empty((self.parallel_envs * self.per_batch_buffer_allocation_size // 4, 3), device=self.device)),
            dim=0)

        self._state.destinations = torch.cat(
            (self._state.destinations,
             torch.empty((self.parallel_envs * self.per_batch_buffer_allocation_size // 4, 3), device=self.device)),
            dim=0)

        self._state.associations = torch.cat(
            (self._state.associations,
             torch.empty((self.parallel_envs * self.per_batch_buffer_allocation_size // 4, 4), device=self.device)),
            dim=0)

        self._state.used_space = torch.cat(
            (self._state.used_space,
             torch.zeros(self.parallel_envs * self.per_batch_buffer_allocation_size // 4, device=self.device, dtype=torch.bool)),
            dim=0)

    @torch.no_grad()
    def _drive_car(self, agent: int, batch: int, destination: torch.IntTensor) -> None:
        """
        Moves the agent (and all riding passengers) in the specified direction

        Args:
            agent: int - the agent to move
            batch: int - the batch index
            destination: torch.IntTensor - the destination to move to
        """

        #update this agent's position
        agent_mask = torch.logical_and(self._state.agents[:, 0] == batch, self._state.agents[:, 1] == agent)

        assert self._state.agents[agent_mask].shape[
            0] == 1, f"Agent not found or too many found? {self._state.agents[agent_mask.shape[0]]}"

        self._state.agents[agent_mask, 2:] = destination

        #update riding passenger positions
        passenger_mask = torch.logical_and(self._state.associations[:, 0] == batch, self._state.associations[:, 2] == agent)

        #must exclude dead memory
        passenger_mask = torch.logical_and(self._state.used_space, passenger_mask)

        if torch.any(passenger_mask):
            self._state.locations[passenger_mask, 1:] = destination

    @torch.no_grad()
    def step_environment(self) -> Tuple[Dict[str, torch.Tensor] | Dict[str, Dict[str, bool]]]:

        #increment step count for term
        self._step_count += 1

        # Initialize storages
        rewards = {agent: torch.zeros(self.parallel_envs, dtype=torch.float32, device=self.device) for agent in self.agents}
        terminations = {agent: torch.zeros(self.parallel_envs, dtype=torch.bool, device=self.device) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        #?collect accept actions to process later
        accepts = [defaultdict(lambda: []) for _ in range(self.parallel_envs)]

        #are we penalizing all agents for waiting passengers?
        if self.use_waiting_costs:
            #find waiting passengers
            wait_accept = torch.isnan(self._state.timing[self._state.used_space][:, 2])
            wait_pickup = torch.logical_and(~wait_accept, torch.isnan(self._state.timing[self._state.used_space][:, 3]))
            wait_dropoff = torch.logical_and(~wait_pickup, torch.isnan(self._state.timing[self._state.used_space][:, 4]))

            #adjust masks by wait limits
            wait_accept = torch.logical_and(
                wait_accept, self._state.timing[self._state.used_space][:, 1] + self.wait_limit[0] < self._step_count)
            wait_pickup = torch.logical_and(
                wait_pickup, self._state.timing[self._state.used_space][:, 1] + self.wait_limit[1] < self._step_count)
            wait_dropoff = torch.logical_and(
                wait_dropoff, self._state.timing[self._state.used_space][:, 1] + self.wait_limit[2] < self._step_count)

            #TODO maybe have action specific costs later
            waiting_passengers = torch.logical_or(torch.logical_or(wait_accept, wait_pickup), wait_dropoff)

            long_waiting_passengers = torch.logical_and(
                waiting_passengers, self._state.timing[self._state.used_space][:, 1] + self.long_wait_time < self._step_count)

            #?construct joint penalties
            waiting_penalties = torch.stack([
                torch.sum(waiting_passengers[self._state.associations[self._state.used_space][:, 0] == batch_index] *
                          self.general_wait_cost +
                          long_waiting_passengers[self._state.associations[self._state.used_space][:, 0] == batch_index] *
                          self.long_wait_cost) for batch_index in range(self.parallel_envs)
            ],
                                            dim=0)

            #apply penalties
            rewards = {agent: rewards[agent] + waiting_penalties for agent in self.agents}

        #see if new passengers need to be added
        self._apply_schedule(list(range(self.parallel_envs)))

        for agent_name in self.actions:
            agent = self.agent_name_mapping[agent_name]

            for batch_index, action in enumerate(self.actions[agent_name]):

                #switch to global indices
                #! this handles cases where we accidentally duplicate actions across batches (as ref rather than copies)
                action = action.clone().to(torch.int)
                action[0] = self.agent_task_indices[agent_name][batch_index][action[0]][0]

                #accept #?(handle conflicts later)
                if action[1] == 0:

                    #check for pool_limit
                    try:
                        accepted_passenger_count = torch.sum(
                            torch.logical_and(self._state.associations[self._state.used_space][:, 0] == batch_index,
                                              self._state.associations[self._state.used_space][:, 1] == agent))
                    except Exception as e:
                        print(e)
                        print("Likely a problem with the used_space mask")

                    if accepted_passenger_count >= self.pool_limit:
                        rewards[agent_name][batch_index] += self.pool_limit_cost

                    else:
                        accepts[batch_index][action[0]].append(agent)

                #pickup
                elif action[1] == 1:

                    assert self._state.associations[action[0],
                                                    1] == agent, f"Passenger {action[0]} not accepted by agent {agent_name}"

                    agent_pos = self._state.agents[self._state.agents[:, 0] == batch_index][agent, 2:]

                    #?doesn't need to use "used_space" because that is already accounted for in action index conversion
                    passenger_pos = self._state.locations[action[0], 1:]

                    #i'm not at the passenger let's go get them
                    if not torch.all(agent_pos == passenger_pos):

                        destination, distance = self.driving_algorithm(start=agent_pos, goal=passenger_pos)

                        self._drive_car(agent=agent, batch=batch_index, destination=destination)

                        if self.use_variable_move_cost:
                            #check for riding passengers
                            accepted_passenger_count = torch.sum(
                                torch.logical_and(self._state.associations[self._state.used_space][:, 0] == batch_index,
                                                  self._state.associations[self._state.used_space][:, 2] == agent)).unsqueeze(0)
                            #TODO in prior implementation includes num_accepted_passengers in this calculation. That seems wrong.

                            if accepted_passenger_count > 1:
                                rewards[agent_name][[batch_index]] += distance * self.move_cost / accepted_passenger_count
                            else:
                                rewards[agent_name][[batch_index]] += distance * self.move_cost
                        else:
                            rewards[agent_name][[batch_index]] += distance * self.move_cost

                    #get the potentially updated agent position
                    agent_pos = self._state.agents[self._state.agents[:, 0] == batch_index][agent, 2:]

                    #i'm at the passenger now
                    if torch.all(agent_pos == passenger_pos):

                        #mark pickup
                        self._state.associations[action[0], 2] = agent
                        self._state.timing[action[0]][3] = self._step_count

                        rewards[agent_name][batch_index] += self.pick_cost

                #dropoff
                elif action[1] == 2:

                    #?confirm passenger in car
                    assert self._state.associations[action[0],
                                                    2] == agent, f"Passenger {action[0]} not riding with agent {agent_name}"

                    agent_pos = self._state.agents[self._state.agents[:, 0] == batch_index][agent, 2:]
                    passenger_dest = self._state.destinations[action[0], 1:]

                    #i'm not at the destination let's go there
                    if not torch.all(agent_pos == passenger_dest):
                        destination, distance = self.driving_algorithm(start=agent_pos, goal=passenger_dest)

                        self._drive_car(agent=agent, batch=batch_index, destination=destination)

                        rewards[agent_name][[batch_index]] += self.move_cost * distance

                    #get the potentially updated agent position
                    agent_pos = self._state.agents[self._state.agents[:, 0] == batch_index][agent, 2:]

                    #now I'm at the destination
                    if torch.all(agent_pos == passenger_dest):
                        #fare + dropoff cost
                        rewards[agent_name][batch_index] += self.drop_cost + self._state.associations[action[0], 3]
                        #TODO there may be a extra "bonus" reward here for dropping off multiple passengers
                        #remove the passenger from the game
                        self._state.used_space[action[0]] = False
                        self._state.timing[action[0], 4] = self._step_count

                #noop
                elif action[1] == -1:
                    rewards[agent_name][batch_index] += self.noop_cost

                else:
                    raise ValueError("Invalid action, {} is not a valid action".format(action[1]))

        #?process accepts (passenger here is action[0])
        for batch_index, batch in enumerate(accepts):
            for passenger, drivers in batch.items():
                if len(drivers) > 1:

                    #find the accept distance between drivers and the passenger, the best is accepted
                    driver_locations = self._state.agents[self._state.agents[:, 0] == batch_index][drivers, 2:]

                    #find the passenger location
                    passenger_location = self._state.locations[passenger, [1, 2]]

                    #find the distance between the drivers and the passenger
                    distances = torch.sqrt(torch.sum((driver_locations - passenger_location)**2, dim=1))

                    #find the best driver
                    driver = drivers[torch.argmin(distances)]

                else:
                    driver = drivers[0]

                self._state.associations[passenger, 1] = driver
                self._state.timing[passenger, 2] = self._step_count

                rewards[self.agents[driver]][batch_index] += self.accept_cost

        try:
            assert all([torch.any(~torch.isnan(reward)) for reward in rewards.values()])

            #are there no passengers right now?
            if self._state.locations[self._state.used_space].shape[0] == 0:
                #check if there are any passengers scheduled
                if self._step_count > self._last_scheduled_arrival:
                    terminations = {
                        agent: torch.ones(self.parallel_envs, dtype=torch.bool, device=self.device)
                        for agent in self.agents
                    }

            return rewards, terminations, infos
        except AssertionError as e:
            print(e)
            print("Found nan in rewards")
            print(rewards)
            raise e

    @torch.no_grad()
    def update_actions(self) -> None:
        """
        Update the available actions for all agents, based on the current state

        Must be executed prior to action_space(agent)
        """

        # masked_batch_index = self._state.associations[self._state.used_space][:, 0]
        self.agent_task_indices = {}

        index_array = torch.arange(self._state.used_space.shape[0], device=self.device)

        #find global indices for all present passengers irrespective of agent
        environment_task_indices = []
        for batch_index in range(self.parallel_envs):
            batch_mask = torch.logical_and(self._state.associations[:, 0] == batch_index, self._state.used_space)
            environment_task_indices.append(index_array[batch_mask].unsqueeze(1))

        self.environment_task_indices = torch.nested.nested_tensor(environment_task_indices)

        for agent, agent_idx in self.agent_name_mapping.items():

            #TODO this will be switched to a nested_tensor when I update step_environment to be vectorized. For now having it as a list doesn't change anything.
            self.agent_task_indices[agent] = []
            self.agent_action_indices[agent] = []

            #driver related passengers
            accept_mask = self._state.associations[:, 1] == agent_idx
            ride_mask = self._state.associations[:, 2] == agent_idx

            try:
                driver_related_mask = torch.logical_or(accept_mask, ride_mask)
            except Exception as e:
                raise e

            #hide dead memory
            driver_related_mask = driver_related_mask[self._state.used_space]

            #unaccepted passengers
            unaccepted_passengers = self._state.associations[:, 1][self._state.used_space].isnan()

            #find available action for each passenger
            action_tensor = torch.ones_like(driver_related_mask, dtype=torch.float)
            action_tensor[action_tensor.clone().to(torch.bool)] = torch.nan
            action_tensor[torch.logical_not(accept_mask[self._state.used_space])] = 0.0
            action_tensor[torch.logical_and(accept_mask[self._state.used_space],
                                            torch.logical_not(ride_mask[self._state.used_space]))] = 1.0
            action_tensor[ride_mask[self._state.used_space]] = 2.0

            assert torch.all(~torch.isnan(action_tensor)), "Invalid action tensor, check for a valid state"

            action_tensor = action_tensor.to(torch.int)

            #?find global indices (batch, global index, action type, masked_index [used only in logging])
            _index_array = torch.cat([
                index_array.unsqueeze(1)[self._state.used_space], self._state.locations[:, [0]][self._state.used_space],
                action_tensor.unsqueeze(1),
                torch.arange(index_array[self._state.used_space].shape[0], device=self.device).unsqueeze(1)
            ],
                                     dim=1).to(torch.int)

            #construct task lists
            agent_task_indices = []
            for batch_index in range(self.parallel_envs):

                #handling global<--> local loging indices
                batch_piece = _index_array[_index_array[:, 1] == batch_index]
                batch_piece[:, 3] = torch.arange(batch_piece.shape[0], device=self.device)
                _index_array[_index_array[:, 1] == batch_index] = batch_piece

                #construct global index <--> task index mapping
                dropoff_passenger = _index_array[ride_mask[self._state.used_space]
                                                 & (_index_array[:, 1] == batch_index)][:, [0, 2, 3]]

                #pickup passenger
                pickup_passenger = _index_array[accept_mask[self._state.used_space] & (~ride_mask[self._state.used_space]) &
                                                (_index_array[:, 1] == batch_index)][:, [0, 2, 3]]

                riding_or_accepted_passengers = torch.cat([dropoff_passenger, pickup_passenger], dim=0)

                unaccepted_batch_mask = torch.logical_and(unaccepted_passengers, _index_array[:, 1] == batch_index)
                unaccepted_passengers_batch = _index_array[unaccepted_batch_mask][:, [0, 2, 3]]

                passenger_list = torch.cat([
                    torch.tensor([-1, -1, -1], device=self.device).unsqueeze(0), riding_or_accepted_passengers,
                    unaccepted_passengers_batch
                ],
                                           dim=0)

                agent_task_indices.append(passenger_list[:, [0]])  #use global index
                self.agent_action_indices[agent].append(passenger_list[:, [1]])  #just show actions

            #convert agent_task_indices to nested tensor
            self.agent_task_indices[agent] = torch.nested.nested_tensor(agent_task_indices)

    @torch.no_grad()
    def view_state(self):
        var = self._state.associations[self._state.used_space].clone()
        return var

    @torch.no_grad()
    def update_observations(self) -> None:
        """
        Updates self.observations with the current state of the environment
        """

        #?get in environment passengers
        locations_masked = self._state.locations[self._state.used_space]
        destinations_masked = self._state.destinations[self._state.used_space]
        associations_masked = self._state.associations[self._state.used_space]
        timings_masked = self._state.timing[self._state.used_space]

        #get unaccepted passenger indices
        unac_pass_ind = torch.isnan(associations_masked[:, 1])

        #?get global indices for each passenger
        index_array = torch.arange(self._state.used_space.shape[0], device=self.device)
        index_array = torch.cat([index_array.unsqueeze(1), self._state.locations[:, [0]]], dim=1).to(torch.int)
        index_array = index_array[self._state.used_space]

        #construct unaccepted passenger observations
        unaccepted_passengers = []

        for batch_index in range(self.parallel_envs):
            batch_mask = associations_masked[unac_pass_ind][:, 0] == batch_index

            #?calculate relative distances for observations
            if self.use_relative_distance:

                #TODO rename this to something else
                batch_map = self._state.agents[:, 0] == batch_index
                batch_passengers = {}

                for agent in self.agent_name_mapping.values():
                    agent_batch_map = torch.logical_and(batch_map, self._state.agents[:, 1] == agent)

                    batch_passengers[agent] = torch.cat(
                        [
                            #!dep prior displayed global index as observation
                            index_array[unac_pass_ind][batch_mask][:, [0]],
                            #TODO confirm this simple heuristic is acceptable
                            torch.abs(locations_masked[unac_pass_ind][batch_mask][:, [1, 2]] -
                                      self._state.agents[agent_batch_map][:, [2, 3]]),
                            torch.abs(destinations_masked[unac_pass_ind][batch_mask][:, [1, 2]] -
                                      self._state.agents[agent_batch_map][:, [2, 3]]),
                            associations_masked[unac_pass_ind][batch_mask][:, [1, 2, 3]],
                            timings_masked[unac_pass_ind][batch_mask][:, [1]]  #time of arrival
                        ],
                        dim=1)
                unaccepted_passengers.append(batch_passengers)

            else:
                unaccepted_passengers.append(
                    torch.cat(
                        [
                            #!dep prior displayed global index as observation
                            index_array[unac_pass_ind][batch_mask][:, [0]],
                            locations_masked[unac_pass_ind][batch_mask][:, [1, 2]],
                            destinations_masked[unac_pass_ind][batch_mask][:, [1, 2]],
                            associations_masked[unac_pass_ind][batch_mask][:, [1, 2, 3]],
                            timings_masked[unac_pass_ind][batch_mask][:, [1]]  #time of arrival 
                        ],
                        dim=1))

        accept_counts: Dict[int, torch.IntTensor] = {}
        riding_counts: Dict[int, torch.IntTensor] = {}

        passenger_observations: List[torch.IntTensor] = []
        agent_obs = {}

        #?loop over all agents then assemble by batch
        for agent in self.possible_agents:
            agent_idx = self.agent_name_mapping[agent]

            #accepted counts (recall all ridding passengers are accepted, so we only use the accepted indices for later grouping)
            accpt_pass_mask = associations_masked[:, 1] == agent_idx
            accpt_pass_ind = associations_masked[accpt_pass_mask]
            accpt_batches, accpt_counts = torch.unique(accpt_pass_ind[:, 0].to(torch.int), return_counts=True)
            accpt_counts_padded = torch.zeros(self.parallel_envs, device=self.device, dtype=torch.int)
            if accpt_counts.shape[0] > 0:
                accpt_counts_padded[accpt_batches] = accpt_counts.to(torch.int)
            accept_counts[agent_idx] = accpt_counts_padded

            #riding counts
            ridn_pass_ind = associations_masked[associations_masked[:, 2] == agent_idx]
            ridn_batches, ridn_counts = torch.unique(ridn_pass_ind[:, 0], return_counts=True)
            ridn_counts_padded = torch.zeros(self.parallel_envs, device=self.device, dtype=torch.int)
            if ridn_counts.shape[0] > 0:
                ridn_counts_padded[ridn_batches.to(torch.int)] = ridn_counts.to(torch.int)
            riding_counts[agent_idx] = ridn_counts_padded

            #construct accepted passenger observations
            accepted_passengers = []

            for batch_index in range(self.parallel_envs):

                batch_mask = associations_masked[accpt_pass_mask][:, 0] == batch_index
                batch_rider_mask = ~torch.isnan(associations_masked[accpt_pass_mask][batch_mask][:, 2])

                if self.use_relative_distance:

                    agent_batch_map = torch.logical_and(batch_map, self._state.agents[:, 1] == agent_idx)

                    dstart = locations_masked[accpt_pass_mask][batch_mask][:,
                                                                           [1, 2]] - self._state.agents[agent_batch_map][:,
                                                                                                                         [2, 3]]
                    dgoal = destinations_masked[accpt_pass_mask][batch_mask][:,
                                                                             [1, 2]] - self._state.agents[agent_batch_map][:,
                                                                                                                           [2, 3]]

                    #add riders then pickups
                    accepted_passengers.append(
                        torch.cat(
                            [
                                torch.cat(
                                    [
                                        #!DEP prior displayed global index as observation
                                        index_array[accpt_pass_mask][batch_mask][:, [0]][batch_rider_mask],

                                        #manhattan distance (location)
                                        torch.sum(torch.abs(dstart), dim=1).unsqueeze(1)[batch_rider_mask],

                                        #manhattan distance (location)
                                        torch.sum(torch.abs(dgoal), dim=1).unsqueeze(1)[batch_rider_mask],
                                        associations_masked[accpt_pass_mask][batch_mask][:, [1, 2, 3]][batch_rider_mask],
                                        timings_masked[accpt_pass_mask][batch_mask][:, [1]][batch_rider_mask]  #time of arrival 
                                    ],
                                    dim=1),
                                torch.cat(
                                    #!DEP prior displayed global index as observation
                                    index_array[accpt_pass_mask][batch_mask][:, [0]][~batch_rider_mask],

                                    #manhattan distance (location)
                                    torch.sum(torch.abs(dstart), dim=1).unsqueeze(1)[~batch_rider_mask],

                                    #manhattan distance (location)
                                    torch.sum(torch.abs(dgoal), dim=1).unsqueeze(1)[~batch_rider_mask],
                                    associations_masked[accpt_pass_mask][batch_mask][:, [1, 2, 3]][~batch_rider_mask],
                                    timings_masked[accpt_pass_mask][batch_mask][:, [1]][~batch_rider_mask]  #time of arrival 
                                )
                            ],
                            dim=0))

                else:
                    accepted_passengers.append(
                        torch.cat([
                            torch.cat(
                                [
                                    #!DEP prior displayed global index as observation
                                    index_array[accpt_pass_mask][batch_mask][:, [0]][batch_rider_mask],
                                    locations_masked[accpt_pass_mask][batch_mask][:, [1, 2]][batch_rider_mask],
                                    destinations_masked[accpt_pass_mask][batch_mask][:, [1, 2]][batch_rider_mask],
                                    associations_masked[accpt_pass_mask][batch_mask][:, [1, 2, 3]][batch_rider_mask],
                                    timings_masked[accpt_pass_mask][batch_mask][:, [1]][batch_rider_mask]  #time of arrival 
                                ],
                                dim=1),
                            torch.cat(
                                [
                                    #!DEP prior displayed global index as observation
                                    index_array[accpt_pass_mask][batch_mask][:, [0]][~batch_rider_mask],
                                    locations_masked[accpt_pass_mask][batch_mask][:, [1, 2]][~batch_rider_mask],
                                    destinations_masked[accpt_pass_mask][batch_mask][:, [1, 2]][~batch_rider_mask],
                                    associations_masked[accpt_pass_mask][batch_mask][:, [1, 2, 3]][~batch_rider_mask],
                                    timings_masked[accpt_pass_mask][batch_mask][:, [1]][~batch_rider_mask]  #time of arrival 
                                ],
                                dim=1)
                        ]))

            passenger_observations.append(accepted_passengers)

            #this agent obs
            agent_obs[agent_idx] = torch.cat(
                [
                    torch.ones((self.parallel_envs, 1), device=self.device, dtype=torch.int) * agent_idx,  #agent index
                    self._state.agents[self._state.agents[:, 1] == agent_idx][:, [2, 3]],  #agent position
                    accept_counts[agent_idx].unsqueeze(1),  #num passengers accepted
                    riding_counts[agent_idx].unsqueeze(1)  #num passengers riding
                ],
                dim=1)

        #?construct full observation
        observations = {}

        for i, agent in enumerate(self.possible_agents):

            observations[agent] = {
                'self': agent_obs[i],
                'others':torch.stack([obs for ag, obs in agent_obs.items() if ag != i], dim=1) if len(self.possible_agents)>1 else torch.ones((self.parallel_envs,1),dtype=torch.float)*torch.nan,
                'tasks': torch.nested.nested_tensor(
                    [torch.cat([passenger_observations[i][b][:,1:], unaccepted_passengers[b][:,1:]\
                        if not self.use_relative_distance else unaccepted_passengers[b][i][:,1:]],dim=0)
                    for b in range(self.parallel_envs)],
                )
            }

            if self.provide_hyperedges:
                observations[agent]['hyperedges'] = []

                for b in range(self.parallel_envs):

                    all_passenger_observations = torch.cat([
                        passenger_observations[i][b],
                        unaccepted_passengers[b] if not self.use_relative_distance else unaccepted_passengers[b][i],
                    ],
                                                           dim=0)

                    #accepted passengers
                    accepted_passenger_mask = all_passenger_observations[:, 5] == i

                    #mask of passengers riding with this agent  (dropoff)
                    riding_passenger_mask = all_passenger_observations[:, 6] == i

                    #(pickup) passengers
                    pickup_passenger_mask = torch.logical_and(accepted_passenger_mask, ~riding_passenger_mask)

                    #(accept) passengers
                    to_accept_passenger_mask = ~accepted_passenger_mask

                    #?global index here is used to create the critic graph
                    #<LOCAL INDEX, GLOBAL INDEX> will later be labeled to <ACTION index, LOCAL TASK INDEX, GLOBAL TASK INDEX>
                    accepts = torch.cat([
                        to_accept_passenger_mask.nonzero().reshape(-1, 1),
                        all_passenger_observations[to_accept_passenger_mask][:, [0]]
                    ],
                                        dim=-1)
                    pickups = torch.cat([
                        pickup_passenger_mask.nonzero().reshape(-1, 1), all_passenger_observations[pickup_passenger_mask][:, [0]]
                    ],
                                        dim=-1)
                    dropoffs = torch.cat([
                        riding_passenger_mask.nonzero().reshape(-1, 1), all_passenger_observations[riding_passenger_mask][:, [0]]
                    ],
                                         dim=-1)

                    non_empty_cats = []

                    try:
                        if dropoffs.shape[0] > 0: non_empty_cats.append(dropoffs)
                        if pickups.shape[0] > 0: non_empty_cats.append(pickups)
                        if accepts.shape[0] > 0: non_empty_cats.append(accepts)

                    except Exception as e:
                        raise e

                    #?no passengers
                    if len(non_empty_cats) == 0:
                        graph = tg.data.Data(x=torch.tensor([[-1, -1]], device=self.device),
                                             edge_index=torch.tensor([[0, 0]], device=self.device),
                                             edge_attr=torch.tensor([-1], device=self.device))

                        observations[agent]['hyperedges'].append(graph)
                    else:

                        nodes = torch.cat(non_empty_cats, dim=0)

                        accepts = accepts[:, 0]
                        pickups = pickups[:, 0]
                        dropoffs = dropoffs[:, 0]

                        #add label to accepts
                        accepts = torch.stack([torch.zeros(accepts.shape[0], device=self.device), accepts], dim=1)

                        riders = torch.cat([pickups, dropoffs], dim=0)

                        #?cartesian product for effects of tasks
                        try:
                            if pickups.shape[0] > 0:
                                src_pickup_task, pickup_effected_task = torch.meshgrid(pickups, riders, indexing='ij')

                                #label action
                                src_pickup_task = torch.stack([
                                    torch.ones_like(src_pickup_task, device=self.device, dtype=torch.int),
                                    src_pickup_task,
                                ],
                                                              dim=-1)
                                #label target nan
                                pickup_effected_task = torch.stack([
                                    torch.ones_like(pickup_effected_task, device=self.device, dtype=torch.float).fill_(torch.nan),
                                    pickup_effected_task
                                ],
                                                                   dim=-1)

                                #expand across possibilities
                                pickup_task_edges = torch.flatten(torch.stack([src_pickup_task, pickup_effected_task], dim=2),
                                                                  start_dim=0,
                                                                  end_dim=1)

                            else:
                                pickup_task_edges = torch.empty((0, 2, 2), device=self.device)

                            if dropoffs.shape[0] > 0:
                                src_dropoff_task, dropoff_effected_task = torch.meshgrid(dropoffs, riders, indexing='ij')

                                #label action
                                src_dropoff_task = torch.stack([
                                    torch.ones_like(src_dropoff_task, device=self.device, dtype=torch.float).fill_(2),
                                    src_dropoff_task
                                ],
                                                               dim=-1)

                                #label target nan
                                dropoff_effected_task = torch.stack([
                                    torch.ones_like(dropoff_effected_task, device=self.device, dtype=torch.float).fill_(
                                        torch.nan), dropoff_effected_task
                                ],
                                                                    dim=-1)

                                #expand across possibilities
                                dropoff_task_edges = torch.flatten(torch.stack([src_dropoff_task, dropoff_effected_task], dim=2),
                                                                   start_dim=0,
                                                                   end_dim=1)

                            else:
                                dropoff_task_edges = torch.empty((0, 2, 2), device=self.device)

                        except Exception as e:
                            raise e

                        try:
                            #dims: <number of edges, 2 src dest, 2 action, task>
                            edges = torch.cat([dropoff_task_edges, pickup_task_edges,
                                               torch.stack([accepts, accepts], dim=1)],
                                              dim=0)
                        except Exception as e:
                            raise e

                        #edge src->dest
                        edge_index = edges[:, [0, 1], 1]

                        #action index
                        edge_attr = edges[:, 0, 0]

                        #?append NOOP
                        nodes = torch.cat([nodes, torch.tensor([-1, -1], device=self.device).unsqueeze(0)], dim=0)
                        edge_index = torch.cat(
                            [edge_index, torch.tensor([[nodes.shape[0] - 1, nodes.shape[0] - 1]], device=self.device)], dim=0)
                        edge_attr = torch.cat([edge_attr, torch.tensor([-1], device=self.device)])

                        #construct graph
                        graph = tg.data.Data(
                            x=nodes,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                        )

                        observations[agent]['hyperedges'].append(graph)

        #TODO keep this as a tensor dict even with tg attributes
        self.observations = TensorDict(observations, batch_size=[self.parallel_envs
                                                                 ]) if not self.provide_hyperedges else observations

    @torch.no_grad()
    def action_space(self, agent: str, use_fast_action_space: bool = False) -> List[gymnasium.Space]:
        """
        Creates action space for the agent

        agent: str - the agent to create the action space for

        Returns:
            List[gymnasium.Space] - the action spaces for the agent batchwise listed
        """

        agent_task_actions = self.agent_action_indices[agent]
        spaces = []

        for batch in range(self.parallel_envs):
            space = Space.OneOf([action_choice[task_action[0]] for task_action in agent_task_actions[batch].tolist()])
            spaces.append(space)

        return Space.Vector(spaces)

    @torch.no_grad()
    def observation_space(self, agent: str) -> List[gymnasium.Space]:
        """
        Creates observation space for the agent

        agent: str - the agent to create the observation space for

        Returns:
            gymnasium.Space - the observation space for the agent
            self - <agent index, y position, x position, num_passenders accepted, num_passengers riding>
            others - same as self
            passengers - <y position, x position, y destination, x destination, accepted by, riding by, time entered environment>
        """

        ag = self.agent_name_mapping[agent]

        spaces = []

        #TODO optimize this like crazy later
        for batch_index in range(self.parallel_envs):

            number_of_present_tasks = torch.sum(self._state.used_space[self._state.associations[:, 0] == batch_index])

            space_dictionary = {
                    'self': gymnasium.spaces.Box(
                        low=np.array([ag, 0, 0, 0, 0]),
                        high=np.array([ag, self.grid_height, self.grid_width, self.pool_limit, self.pool_limit]),
                    ),
                    'others': gymnasium.spaces.Tuple(
                        [
                        gymnasium.spaces.Box(
                                low=np.array([0, 0, 0, 0, 0]),
                                high=np.array([self.num_agents, self.grid_height, self.grid_width, self.pool_limit, self.pool_limit]),
                            )
                        for _ in range(self.num_agents - 1)
                        ]
                    ),
                    #passenger details
                    'tasks': gymnasium.spaces.Tuple(
                        [
                            gymnasium.spaces.Box(
                                low=np.array([0, 0, 0, 0, -1, -1, 0]),
                                high=np.array([self.grid_height, self.grid_width, self.grid_height, self.grid_width,\
                                    len(self.possible_agents), len(self.possible_agents), self.max_steps]
                                )
                            )
                        for _ in range(number_of_present_tasks)
                        ]
                    )
                }

            if self.offer_hyperedges:
                #?where the nodes here represent the task-actions and the edges represent which task actions are grouped together
                # nodes: <task index (in the obs)>
                # edges: <action index (hardcoded here)> [from: task of the task action, to: task effected by TA]
                space_dictionary['hyperedges'] = gymnasium.spaces.Graph(
                    node_space=gymnasium.spaces.Discrete(number_of_present_tasks, shape=(1, )),
                    edge_space=gymnasium.spaces.Discrete(4, start=-1, shape=(1, )),
                )

            space = gymnasium.spaces.Dict(space_dictionary)

            spaces.append(space)

        return spaces
