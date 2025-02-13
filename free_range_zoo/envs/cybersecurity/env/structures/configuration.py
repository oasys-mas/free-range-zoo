"""Configuration classes for the cybersecurity environment."""

from __future__ import annotations

from typing import Tuple
from dataclasses import dataclass
import functools

import torch

from free_range_zoo.utils.configuration import Configuration
from free_range_zoo.envs.cybersecurity.env.transitions import (MovementTransition, PresenceTransition, SubnetworkTransition)


@dataclass
class CybersecurityConfiguration(Configuration):
    """
    Configuration for the cybersecurity environment.

    Attributes:
        attacker_config: AttackerConfiguration - Configuration for the attacker agent properties
        defender_config: DefenderConfiguration - Configuration for the defender agent properties
        network_config: NetworkConfiguration - Configuration for the network nodes
        reward_config: RewardConfiguration - Configuration for the environment rewards
        stochastic_config: StochasticConfiguration - Configuration for the stochastic components of the environment
    """

    attacker_config: AttackerConfiguration
    defender_config: DefenderConfiguration
    network_config: NetworkConfiguration
    reward_config: RewardConfiguration
    stochastic_config: StochasticConfiguration

    @functools.cached_property
    def movement_transition(self) -> MovementTransition:
        """Get the movement transition function for the environment."""
        return MovementTransition()

    @functools.cached_property
    def presence_transition(self) -> PresenceTransition:
        """Get the presence transition function for the environment."""
        return PresenceTransition(self.persist_probs, self.return_probs, self.attacker_config.num_attackers)

    @functools.cached_property
    def subnetwork_transition(self) -> SubnetworkTransition:
        """Get the subnetwork transition function for the environment."""
        return SubnetworkTransition(self.network_config.patched_states, self.network_config.vulnerable_states,
                                    self.network_config.exploited_states, self.network_config.temperature,
                                    self.stochastic_config.network_state)

    @functools.cached_property
    def attacker_observation_bounds(self) -> torch.Tensor:
        """Get the observation bounds for the agent (threat, presence)."""
        return tuple([self.attacker_config.highest_threat, 1])

    @functools.cached_property
    def defender_observation_bounds(self) -> torch.Tensor:
        """Get the observation bounds for the agent (mitigation, presence, location)."""
        return tuple([self.defender_config.highest_mitigation, 1, self.network_config.num_nodes - 1])

    @functools.cached_property
    def network_observation_bounds(self) -> Tuple[int, int]:
        """Get the observation bounds for the subnetwork (state)."""
        return tuple([self.network_config.num_states])

    @functools.cached_property
    def num_agents(self) -> int:
        """Get the number of agents of all types in the environment."""
        return self.attacker_config.num_attackers + self.defender_config.num_defenders

    @functools.cached_property
    def persist_probs(self) -> torch.FloatTensor:
        """Get the persist probabilities for all agents."""
        return torch.cat([self.attacker_config.persist_probs, self.defender_config.persist_probs])

    @functools.cached_property
    def return_probs(self) -> torch.FloatTensor:
        """Get the return probabilities for all agents."""
        return torch.cat([self.attacker_config.return_probs, self.defender_config.return_probs])

    @functools.cached_property
    def initial_presence(self) -> torch.BoolTensor:
        """Get the initial presence of all agents."""
        return torch.cat([self.attacker_config.initial_presence, self.defender_config.initial_presence])

    def validate(self) -> bool:
        """
        Validate the configuration.

        Returns:
            bool - True if the configuration is valid, nothing otherwise
        """
        if self.reward_config.network_state_rewards.size(0) != self.network_config.num_states:
            raise ValueError("The number of network state rewards must match the number of network states.")

        return True


@dataclass
class AttackerConfiguration(Configuration):
    """
    Configuration for the attacker in the cybersecurity environment.

    Attributes:
        initial_presence: torch.BoolTensor - Initial presence of each attacking agent

        threat: torch.FloatTensor - Threat values for each attacking agent
        persist_probs: torch.FloatTensor - Probability for each attacking agent to leave the environment
        return_probs: torch.FloatTensor - Probability for each attacking agent to return to the environment
    """

    initial_presence: torch.BoolTensor

    threat: torch.FloatTensor
    persist_probs: torch.FloatTensor
    return_probs: torch.FloatTensor

    @functools.cached_property
    def num_attackers(self) -> int:
        """Get the number of attackers."""
        return self.threat.size(0)

    @functools.cached_property
    def highest_threat(self) -> float:
        """Get the highest threat value of all attackers."""
        return self.threat.max().item()

    def validate(self) -> bool:
        """Validate the configuration."""
        if self.persist_probs.min() < 0 or self.persist_probs.max() > 1:
            raise ValueError('Persist probabilities must be between 0 and 1.')
        if self.return_probs.min() < 0 or self.return_probs.max() > 1:
            raise ValueError('Return probabilities must be between 0 and 1.')
        if self.threat.size(0) != self.persist_probs.size(0) or self.threat.size(0) != self.return_probs.size(0):
            raise ValueError("The size of threats must match the size of persist and return probabilities.")
        if self.threat.size(0) != self.initial_presence.size(0):
            raise ValueError("The size of threats must match the size of initial presence values.")

        return True


@dataclass
class DefenderConfiguration(Configuration):
    """
    Configuration for the defender in the cybersecurity environment.

    Attributes:
        initial_location: torch.IntTensor - Initial location of each defending agent
        initial_presence: torch.BoolTensor - Initial presence of each defending agent

        mitigation: torch.FloatTensor - mitigation values for each defending agent
        persist_probs: torch.FloatTensor - Probability for each defending agent to leave the environment
        return_probs: torch.FloatTensor - Probability for each defending agent to return to the environment
    """

    initial_location: torch.IntTensor
    initial_presence: torch.BoolTensor

    mitigation: torch.FloatTensor
    persist_probs: torch.FloatTensor
    return_probs: torch.FloatTensor

    @functools.cached_property
    def num_defenders(self) -> int:
        """Get the number of defenders."""
        return self.mitigation.size(0)

    @functools.cached_property
    def highest_mitigation(self) -> float:
        """Get the highest mitigation value of all defenders."""
        return self.mitigation.max().item()

    def validate(self) -> bool:
        """Validate the configuration."""
        if self.persist_probs.min() < 0 or self.persist_probs.max() > 1:
            raise ValueError('Persist probabilities must be between 0 and 1.')
        if self.return_probs.min() < 0 or self.return_probs.max() > 1:
            raise ValueError('Return probabilities must be between 0 and 1.')
        if self.mitigation.size(0) != self.persist_probs.size(0) or self.mitigation.size(0) != self.return_probs.size(0):
            raise ValueError("The size of mitigations must match the size of persist and return probabilities.")
        if self.mitigation.size(0) != self.initial_location.size(0) or self.mitigation.size(0) != self.initial_presence.size(0):
            raise ValueError("The size of mitigations must match the size of initial location and presence values.")

        return True


@dataclass
class NetworkConfiguration(Configuration):
    """
    Configuration for the network components of the cybersecurity simulation.

    The home node for the simulation is automatically defined as node -1.

    Attributes:
        patched_states: int - Number of patched states in the network
        vulnerable_states: int - Number of vulnerable states in the network
        exploited_states: int - Number of exploited states in the network

        temperature: float - Temperature for the softmax function for the danger score

        initial_state: torch.IntTensor - Subnetwork-parallel array representing the exploitment state of each subnetwork
        adj_matrix: torch.BoolTensor - 2D array representing adjacency matrix for all subnetwork connections
    """

    patched_states: int
    vulnerable_states: int
    exploited_states: int

    temperature: float

    initial_state: torch.IntTensor
    adj_matrix: torch.BoolTensor

    @functools.cached_property
    def criticality(self) -> torch.FloatTensor:
        """Get the criticality of each node. Based on the number of outward connections."""
        return self.adj_matrix.sum(dim=1)

    @functools.cached_property
    def num_nodes(self) -> int:
        """Get the number of nodes in the network."""
        return self.adj_matrix.size(0)

    @functools.cached_property
    def num_states(self) -> int:
        """Get the number of states in the network."""
        return self.patched_states + self.vulnerable_states + self.exploited_states

    def validate(self) -> bool:
        """Validate the configuration."""
        if self.initial_state.size(0) != self.num_nodes:
            raise ValueError("The size of initial state must match the number of nodes.")
        if self.adj_matrix.size(0) != self.adj_matrix.size(1):
            raise ValueError("The adjacency matrix must be square.")

        return True


@dataclass
class StochasticConfiguration(Configuration):
    """
    Configuration for the stochastic components of the cybersecurity simulation.

    Attributes:
        network_state: bool - Whether the subnetwork states degrade / repair stochastically
    """

    network_state: bool

    def validate(self) -> bool:
        """Validate the configuration."""
        return True


@dataclass
class RewardConfiguration(Configuration):
    """
    Configuration for the rewards in the cybersecurity environment.

    Attributes:
        bad_action_penalty: float - Penalty for committing a bad action (patching while at the home node)
        patch_reward: float - Reward (or penalty) for patching a node
        network_state_rewards: torch.FloatTensor - Subnetwork-parallel array representing the rewards for each
    """

    bad_action_penalty: float
    patch_reward: float
    network_state_rewards: torch.FloatTensor

    def validate(self) -> bool:
        """Validate the configuration."""
        return True
