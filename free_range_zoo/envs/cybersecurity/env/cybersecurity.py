"""
# Cybersecurity
## Description
The cybersecurity domain simulates a network environment where nodes are either attacked or patched by agents, with
the goal of protecting or exploiting the system. The environment is partially observable, with defenders needing to
respond to exploited nodes, while attackers aim to increase the exploited state of nodes. The dynamic interaction
between attackers and defenders creates an evolving cybersecurity landscape where agents must adapt to the changing
system state.

<u>**Dynamics**</u><br>
- Nodes: The network consists of multiple nodes, each of which can be in one of several states, ranging from
  unexploited to fully exploited. Exploited nodes represent compromised parts of the system that attackers have
  successfully infiltrated, while unexploited nodes are safe and intact.
- Exploited State: Nodes can be attacked by malicious agents to increase their exploited state, making them
  vulnerable to further exploitation. As nodes become more exploited, they pose a greater risk to the overall
  system.
- Patching and Exploiting: Nodes can be patched by defenders to reduce their exploited state, while attackers
  attempt to exploit unpatched or partially patched nodes to further their objectives. The environment is
  partially observable, meaning that defenders do not always know the state of all nodes, requiring them to
  take actions based on limited information.

<u>**Environment Openness**</u><br>
- **agent openness**: Environments where agents can dynamically enter and leave, enabling dynamic cooperation and
  multi-agent scenarios with evolving participants.
    - `cybersecurity`: Agents can lose access to the network, disallowing them from taking actions within the
      environment for a period of time. Agents must reason about how many collaborators are within the
      environment with them, and whether they are able to sufficiently fight opposing agents.

# Specification

---

| Import             | `from free_range_zoo.envs import cybersecurity_v0`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Actions            | Discrete & Stochastic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Observations       | Discrete and Partially Observed with Private Observations                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Parallel API       | Yes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Manual Control     | No                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Agent Names        | [$attacker_0$, ..., $attacker_n$, $defender_0$, ..., $defender_n$]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| # Agents           | [0, $n_{attackers}$ + $n_{defenders}$]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Action Shape       | ($envs$, 2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Action Values      | Attackers: [$attack_0$, ..., $attack_{tasks}$, $noop$ (-1)]<br>Defenders: [$move_0$, ..., $move_{tasks}$, $noop$ (-1), $patch$ (-2), $monitor$ (-3)]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Observation Shape  | Attackers: TensorDict { <br>&emsp;**self**: $<power, presence>$ <br>&emsp;**others**: $<power, presence>$ <br>&emsp;**tasks**: $<state>$ <br> **batch_size**: $num\_envs$ } <br> Defenders: TensorDict { <br>&emsp;**self**: $<power, presence, location>$ <br>&emsp;**others**: $<power, presence, location>$ <br>&emsp;**tasks**: $<state>$<br> **batch_size**: $num\_envs$}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Observation Values | Attackers: <br>&emsp;<u>**self**</u><br>&emsp;&emsp;$power$: [$0$, $max\_power_{attacker}$]<br>&emsp;&emsp;$presence$: [$0$, $1$]<br>&emsp;<u>**others**</u><br>&emsp;&emsp;$power$: [$0$, $max\_power_{attacker}$]<br>&emsp;&emsp;$presence$: [$0$, $1$]<br>&emsp;<u>**tasks**</u><br>&emsp;&emsp;$state$: [$0$, $n_{network\_states}$] <br><br> Defenders: <br>&emsp;<u>**self**</u><br>&emsp;&emsp;$power$: [$0$, $max\_power_{defender}$]<br>&emsp;&emsp;$presence$: [$0$, $1$]<br>&emsp;&emsp;$location$: [$0$, $n_{subnetworks}$]<br>&emsp;<u>**others**</u><br>&emsp;&emsp;$power$: [$0$, $max\_power_{defender}$]<br>&emsp;&emsp;$presence$: [$0$, $1$]<br>&emsp;&emsp;$location$: [$0$, $n_{subnetworks}$]</u><br>&emsp;<u>**tasks**</u><br>&emsp;&emsp;$state$: [$0$, $n_{network\_states}$] |

---
"""

from typing import Tuple, Dict, Any, Union, List, Optional, Callable

import functools
import torch
from tensordict.tensordict import TensorDict
import gymnasium
from pettingzoo.utils.wrappers import OrderEnforcingWrapper
import pandas as pd

from free_range_zoo.utils.env import BatchedAECEnv
from free_range_zoo.utils.conversions import batched_aec_to_batched_parallel
from free_range_zoo.envs.cybersecurity.env.spaces import actions, observations
from free_range_zoo.envs.cybersecurity.env.structures.state import CybersecurityState
from free_range_zoo.envs.cybersecurity.env.utils import masking


def parallel_env(wrappers: List[Callable] = [], **kwargs) -> BatchedAECEnv:
    """
    Paralellized version of the cybersecurity environment.

    Args:
        wrappers: List[Callable[[BatchedAECEnv], BatchedAECEnv]] - the wrappers to apply to the environment
    Returns:
        BatchedAECEnv: the parallelized cybersecurity environment
    """
    env = raw_env(**kwargs)
    env = OrderEnforcingWrapper(env)

    for wrapper in wrappers:
        env = wrapper(env)

    env = batched_aec_to_batched_parallel(env)
    return env


def env(wrappers: List[Callable] = [], **kwargs) -> BatchedAECEnv:
    """
    AEC wrapped version of the cybersecurity environment.

    Args:
        wrappers: List[Callable[[BatchedAECEnv], BatchedAECEnv]] - the wrappers to apply to the environment
    Returns:
        BatchedAECEnv: the cybersecurity environment
    """
    env = raw_env(**kwargs)
    env = OrderEnforcingWrapper(env)

    for wrapper in wrappers:
        env = wrapper(env)

    return env


class raw_env(BatchedAECEnv):
    """Environment definition for the cybersecurity environment."""

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "name": "cybersecurity_v0",
        "is_parallelizable": True,
        "render_fps": 2,
        "null_value": -100
    }

    @torch.no_grad()
    def __init__(
        self,
        *args,
        observe_other_location: bool = False,
        observe_other_presence: bool = False,
        observe_other_power: bool = True,
        partially_observable: bool = True,
        show_bad_actions: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the cybersecurity environment.

        Args:
            observe_other_location: bool - whether to observe the location of other agents
            observe_other_presence: bool - whether to observe the presence of other agents
            observe_other_power: bool - whether to observe the power of other agents
            partially_observable: bool - whether observations of subnetwork states should only be returned on monitor
            show_bad_actions: bool - whether to show bad actions (patch at home node)
        """
        super().__init__(*args, **kwargs)

        self.observe_other_power = observe_other_power
        self.observe_other_location = observe_other_location
        self.observe_other_presence = observe_other_presence
        self.partially_obserable = partially_observable
        self.show_bad_actions = show_bad_actions

        attacker_names = tuple(f"attacker_{i}" for i in range(1, self.config.attacker_config.num_attackers + 1))
        self.attacker_name_mapping = dict(zip(attacker_names, torch.arange(0, len(attacker_names) + 1, device=self.device)))
        defender_names = tuple(f"defender_{i}" for i in range(1, self.config.defender_config.num_defenders + 1))
        self.defender_name_mapping = dict(zip(defender_names, torch.arange(0, len(defender_names) + 1, device=self.device)))
        self.possible_agents = attacker_names + defender_names

        self.agent_name_mapping = {}
        self.agent_name_mapping.update(self.attacker_name_mapping)
        self.agent_name_mapping.update(self.defender_name_mapping)
        self.offset_agent_name_mapping = dict(
            zip(self.possible_agents, torch.arange(0, len(self.possible_agents) + 1, device=self.device)))

        self.observation_ordering = {}
        agent_ids = torch.arange(0, self.attacker_config.num_attackers, device=self.device)
        for attacker_name, agent_idx in self.attacker_name_mapping.items():
            other_agents = agent_ids[agent_ids != agent_idx]
            self.observation_ordering[attacker_name] = other_agents
        for defender_name, agent_idx in self.defender_name_mapping.items():
            other_agents = agent_ids[agent_ids != agent_idx]
            self.observation_ordering[defender_name] = other_agents + self.attacker_config.num_attackers

        self.agent_observation_mask = lambda agent_name: masking.mask_observation(
            agent_name=agent_name,
            observe_other_power=observe_other_power,
            observe_other_presence=observe_other_presence,
            observe_other_location=observe_other_location,
        )

        self.subnetwork_transition = self.config.subnetwork_transition.to(self.device)
        self.presence_transition = self.config.presence_transition.to(self.device)
        self.movement_transition = self.config.movement_transition.to(self.device)

    @torch.no_grad()
    def reset(self, seed: Union[List[int], int] = None, options: Dict[str, Any] = None) -> None:
        """
        Reset the environment.

        Args:
            seed: Union[List[int], int] - the seed to use
            options: Dict[str, Any] - the options for the reset
        """
        super().reset(seed=seed, options=options)

        self.actions = {
            agent: torch.ones((self.parallel_envs, 2), dtype=torch.int32, device=self.device) * -2
            for agent in self.agents
        }

        self.agent_action_to_task_mapping = {agent: {} for agent in self.agents}

        # Initialize the state
        if options is not None and options.get('initial_state') is not None:
            initial_state = options['initial_state']
            if len(initial_state) != self.parallel_envs:
                raise ValueError("Initial state must have the same number of environments as the parallel environments")
            self._state = initial_state
        else:
            self._state = CybersecurityState(
                network_state=self.network_config.initial_state.unsqueeze(0).repeat(self.parallel_envs, 1),
                location=self.defender_config.initial_location.repeat(self.parallel_envs, 1),
                presence=self.config.initial_presence.repeat(self.parallel_envs, 1),
            )

        self._state.save_initial()

        # Initialize range for network indexing
        self.network_range = torch.arange(0, self.network_config.num_nodes, dtype=torch.int32, device=self.device)

        # Intialize the mapping of the tasks "in" the environment, used to map actions
        self.agent_action_mapping = {agent: None for agent in self.agents}
        self.agent_observation_mapping = {agent: None for agent in self.agents}
        self.agent_bad_actions = {agent: None for agent in self.agents}

        self.environment_range = torch.arange(0, self.parallel_envs, dtype=torch.int32, device=self.device)

        # Set the observations and action space
        if not options or not options.get('skip_observations', False):
            self.update_observations()
        if not options or not options.get('skip_actions', False):
            self.update_actions()

        self._post_reset_hook()

    @torch.no_grad()
    def reset_batches(self,
                      batch_indices: torch.Tensor,
                      seed: Optional[List[int]] = None,
                      options: Optional[Dict[str, Any]] = None) -> None:
        """
        Partial reset of the environment for the given batch indices.

        Args:
            batch_indices: torch.Tensor - the batch indices to reset
            seed: Optional[List[int]] - the seed to use
            options: Optional[Dict[str, Any]] - the options for the reset
        """
        super().reset_batches(batch_indices, seed, options)

        # Reset the state
        self._state.restore_initial(batch_indices)

        # Reset the observation updates
        self.update_observations()
        self.update_actions()

    @torch.no_grad()
    def step_environment(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Dict[str, bool]]]:
        """Step the environment forward based on agent actions."""
        # Initialize storages
        rewards = {agent: torch.zeros(self.parallel_envs, dtype=torch.float32, device=self.device) for agent in self.agents}
        terminations = {agent: torch.zeros(self.parallel_envs, dtype=torch.bool, device=self.device) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Generate randomness
        network_randomness = self.generator.generate(
            self.parallel_envs,
            events=1,
            shape=(self.network_config.num_nodes, ),
            key='network',
        )
        agent_randomness = self.generator.generate(
            self.parallel_envs,
            events=1,
            shape=(self.config.num_agents, ),
            key='agent',
        )

        patches = torch.zeros((self.parallel_envs, self.network_config.num_nodes), dtype=torch.float32, device=self.device)
        attacks = torch.zeros((self.parallel_envs, self.network_config.num_nodes), dtype=torch.float32, device=self.device)
        movement_targets = torch.empty(
            (self.parallel_envs, self.defender_config.num_defenders),
            dtype=torch.int32,
            device=self.device,
        )
        movement_mask = torch.zeros(
            (self.parallel_envs, self.defender_config.num_defenders),
            dtype=torch.bool,
            device=self.device,
        )
        for agent_name, agent_actions in self.actions.items():
            agent_index = self.agent_name_mapping[agent_name]
            offset_agent_index = self.offset_agent_name_mapping[agent_name]
            presence = self._state.presence[:, offset_agent_index]

            agent_type = agent_name.split('_')[0]
            match agent_type:
                case 'attacker':
                    attack = agent_actions[:, 1] == 0

                    # Check that all attack targets are valid
                    attack_nodes = agent_actions[attack][:, 0]
                    if not ((attack_nodes >= 0) & (attack_nodes < self.network_config.num_nodes)).all():
                        raise ValueError('Invalid attack target')

                    # Check that agents are not taking actions while not present in the environment
                    if not self.show_bad_actions and torch.any(agent_actions[:, 0][~presence] != -1):
                        raise ValueError('Invalid action for non-present agent')

                    agent_threat = self.attacker_config.threat[agent_index]
                    attack_targets = torch.cat([self.environment_range.unsqueeze(1), agent_actions], dim=1)[attack][:, :2]
                    attacks[attack_targets.split(1, dim=1)] += agent_threat

                case 'defender':
                    move = agent_actions[:, 1] == 0
                    patch = torch.logical_and(agent_actions[:, 1] == -2, self._state.location[:, agent_index] != -1)

                    # Check that all movement targets are valid
                    move_nodes = agent_actions[move][:, 0]
                    if not ((move_nodes >= 0) & (move_nodes < self.network_config.num_nodes)).all():
                        raise ValueError('Invalid movement target')

                    # Check that agents are not taking actions while not present in the environment
                    if not self.show_bad_actions and torch.any(agent_actions[:, 0][~presence] != -1):
                        raise ValueError('Invalid action for non-present agent')

                    # Process agent movements
                    movement_mask[:, agent_index] = move
                    movement_targets[:, agent_index][move] = agent_actions[:, 0][move]

                    # Process agent patches
                    agent_mitigation = self.defender_config.mitigation[agent_index]
                    patch_targets = torch.cat(
                        [self.environment_range.unsqueeze(1), self._state.location[:, agent_index].unsqueeze(1)],
                        dim=1,
                    )[patch]
                    patches[patch_targets.split(1, dim=1)] += agent_mitigation

                    rewards[agent_name][patch] += self.reward_config.patch_reward

                    # If an agent is at the home node (-1), apply a negative reward for patching home node
                    if self.show_bad_actions:
                        bad_patch = torch.logical_and(patch, self._state.location[:, agent_index] == -1)
                        rewards[agent_name][bad_patch] += self.reward_config.bad_action_penalty
                case _:
                    raise ValueError(f'Invalid agent type: {agent_type}')

        self._state = self.movement_transition(state=self._state, movement_targets=movement_targets, movement_mask=movement_mask)
        self._state = self.presence_transition(state=self._state, randomness_source=agent_randomness[0])
        self._state = self.subnetwork_transition(
            state=self._state,
            patches=patches,
            attacks=attacks,
            randomness_source=network_randomness[0],
        )

        # Assign rewards
        network_states = self._state.network_state.flatten().unsqueeze(1)
        state_rewards = self.reward_config.network_state_rewards
        network_rewards = state_rewards[network_states].reshape_as(self._state.network_state)
        network_rewards = torch.matmul(network_rewards, self.network_config.criticality.float())

        for agent_name in self.agents:
            agent_type = agent_name.split('_')[0]
            match agent_type:
                case 'attacker':
                    rewards[agent_name] += network_rewards * -1
                case 'defender':
                    rewards[agent_name] += network_rewards
                case _:
                    raise ValueError(f'Invalid agent type: {agent_type}')

        return rewards, terminations, infos

    @torch.no_grad()
    def update_actions(self) -> None:
        """
        Update the action space for all agents.

        The action space consists of the following encoding:
            - Attackers: [attack_1..n, noop]
            - Defenders: [move_1..n, patch, monitor, noop]
        """
        self.environment_task_count.fill_(self.network_config.num_nodes)
        self.agent_task_count.fill_(0)

        # Agents that are present in the environment have access to all of their tasks
        presence_state = self._state.presence.T
        self.agent_task_count[presence_state] = self.network_config.num_nodes

        # The only action that non-present agents can take is noop
        for agent in self.agents:
            agent_number = self.offset_agent_name_mapping[agent]
            presence_state = self._state.presence[:, agent_number].unsqueeze(1)
            presence_state = presence_state.expand(-1, self.network_config.num_nodes)

            tasks = self.network_range.unsqueeze(0).expand(self.parallel_envs, -1)
            tasks = tasks[presence_state].flatten(end_dim=0)

            task_counts = self.agent_task_count[agent_number]

            self.agent_observation_mapping[agent] = torch.nested.as_nested_tensor(tasks.split(task_counts.tolist(), dim=0))
            self.agent_action_mapping[agent] = torch.nested.as_nested_tensor(tasks.split(task_counts.tolist(), dim=0))

    @torch.no_grad()
    def update_observations(self) -> None:
        """
        Update the observations for the agents. Attackers and defenders have slightly different observations.

        Observations for defenders consist of the following:
            - Self: (batch, 1, (mitigation, presence, location))
            - Others: (batch, num_defenders - 1, (mitigation, presence, location))
            - Subnetworks: (batch, num_subnetworks, (state))

        Observations for attackers consist of the following:
            - Self: (batch, 1, (threat, presence))
            - Others: (batch, nun_attackers - 1, (threat, presence))
            - Subnetworks: (batch, num_subnetworks, (state))
        """
        # Build the defender observations
        defender_mitigation = self.defender_config.mitigation.unsqueeze(0).expand(self.parallel_envs, -1).unsqueeze(2)
        defender_presence = self._state.presence[:, self.attacker_config.num_attackers:].unsqueeze(2)
        defender_locations = self._state.location.unsqueeze(2)
        defender_observation = torch.cat([defender_mitigation, defender_presence, defender_locations], dim=2)

        # Build the attacker observations
        attacker_threat = self.attacker_config.threat.unsqueeze(0).expand(self.parallel_envs, -1).unsqueeze(2)
        attacker_presence = self._state.presence[:, :self.attacker_config.num_attackers].unsqueeze(2)
        attacker_observation = torch.cat([attacker_threat, attacker_presence], dim=2)

        # Aggregate information needed for the network observations
        network_state = self._state.network_state.unsqueeze(2)
        criticality = self.network_config.criticality.unsqueeze(1).unsqueeze(0).expand(self.parallel_envs, -1, -1)
        self.task_store = torch.cat([network_state, criticality], dim=2)

        self.observations = {}
        for agent in self.agents:
            agent_index = self.agent_name_mapping[agent]
            observation_mask = self.agent_observation_mask(agent)

            match agent.split('_')[0]:
                case 'defender':
                    not_monitor = self.actions[agent][:, 1] != -3
                    agent_mask = torch.ones(self.defender_config.num_defenders, dtype=torch.bool, device=self.device)
                    agent_mask[agent_index] = False
                    observation = TensorDict(
                        {
                            'self': defender_observation[:, agent_index],
                            'others': defender_observation[:, agent_mask][:, :, observation_mask],
                            'tasks': self.task_store.clone(),
                        },
                        batch_size=[self.parallel_envs],
                        device=self.device,
                    )

                    if self.partially_obserable:
                        observation['tasks'][not_monitor] = observation['tasks'][not_monitor].fill_(-100)

                case 'attacker':
                    agent_mask = torch.ones(self.attacker_config.num_attackers, dtype=torch.bool, device=self.device)
                    agent_mask[agent_index] = False
                    observation = TensorDict(
                        {
                            'self': attacker_observation[:, agent_index],
                            'others': attacker_observation[:, agent_mask][:, :, observation_mask],
                            'tasks': self.task_store.clone(),
                        },
                        batch_size=[self.parallel_envs],
                        device=self.device,
                    )

            self.observations[agent] = observation

    @torch.no_grad()
    def action_space(self, agent: str) -> List[gymnasium.Space]:
        """
        Return the action space for the given agent.

        Args:
            agent: str - the name of the agent to retrieve the action space for
        Returns:
            List[gymnasium.Space]: the action space for the given agent
        """
        if self.show_bad_actions:
            num_tasks_in_environment = self.environment_task_count
        else:
            num_tasks_in_environment = self.agent_task_count[self.agent_name_mapping[agent]]

        agent_type = agent.split('_')[0]
        agent_index = self.agent_name_mapping[agent]

        return actions.build_action_space(
            agent_type=agent_type,
            show_bad_actions=self.show_bad_actions,
            environment_task_counts=num_tasks_in_environment,
            current_location=self._state.location[:, agent_index],
        )

    @torch.no_grad()
    @functools.lru_cache(maxsize=100)  # Observation space never changes size
    def observation_space(self, agent: str) -> List[gymnasium.Space]:
        """
        Return the observation space for the given agent.

        Args:
            agent: str - the name of the agent to retrieve the observation space for
        Returns:
            List[gymnasium.Space]: the observation space for the given agent
        """
        agent_type = agent.split('_')[0]

        return observations.build_observation_space(
            agent_type=agent_type,
            num_nodes=self.network_config.num_nodes,
            parallel_envs=self.parallel_envs,
            num_attackers=self.attacker_config.num_attackers,
            num_defenders=self.defender_config.num_defenders,
            attacker_high=self.config.attacker_observation_bounds,
            defender_high=self.config.defender_observation_bounds,
            network_high=self.config.network_observation_bounds,
            include_power=self.observe_other_power,
            include_location=self.observe_other_location,
            include_presence=self.observe_other_presence,
        )

    def _log_environment(self, *args, **kwargs) -> None:
        """Log the environment state to a csv."""
        adj_matrix = str(self.network_config.adj_matrix.int().tolist())
        df = pd.DataFrame({'adj_matrix': [adj_matrix] * self.parallel_envs})
        super()._log_environment(*args, **kwargs, extra=df)
