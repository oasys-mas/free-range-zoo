"""Transition function for subnetwork state."""

import torch
from torch import nn

from free_range_zoo.envs.cybersecurity.env.structures.state import CybersecurityState


class SubnetworkTransition(nn.Module):
    """
    Transition function for subnetwork state.

    From the attacks and mitigations of the defenders, the difference is taken, and then `tanh` is determined
    scaled by `temperature`. Positive values represent the probability of state degradation (increase in state
    value), and negative values represent the probability of state improvement (decrease in state value).
    """

    def __init__(self, patched_states, vulnerable_states: int, exploited_states: int, temperature: float,
                 stochastic_state: bool) -> None:
        """
        Initialize the transition function.

        Args:
            patched_states: int - the number of states that have been patched
            vulnerable_states: int - the number of states that are vulnerable
            exploited_states: int - the number of states that are exploited
            temperature: float - the temperature for the transition
            stochastic_state: bool - whether to use stochastic state transitions
        """
        super().__init__()

        self.register_buffer('temperature', torch.tensor(temperature, dtype=torch.float32))
        self.register_buffer('stochastic_state', torch.tensor(stochastic_state, dtype=torch.bool))

        self.register_buffer('patched_states', torch.tensor(patched_states, dtype=torch.int32))
        self.register_buffer('vulnerable_states', torch.tensor(vulnerable_states, dtype=torch.int32))
        self.register_buffer('exploited_states', torch.tensor(exploited_states, dtype=torch.int32))

    @torch.no_grad()
    def forward(self, state: CybersecurityState, attacks: torch.FloatTensor, patches: torch.FloatTensor,
                randomness_source: torch.FloatTensor) -> CybersecurityState:
        """
        Calculate the next subnetwork states for all subnetworks.

        Args:
            state: CybersecurityState - the current state of the environment
            attacks: torch.FloatTensor - the attacks on each subnetwork
            patches: torch.FloatTensor - the patches on each subnetwork
            randomness_source: torch.FloatTensor - the source of randomness for the transition
        Returns:
            CybersecurityState - the next state of the environment with the subnetwork states transformed
        """
        attack_difference = patches - attacks
        danger_score = torch.tanh(attack_difference / self.temperature)

        if self.stochastic_state:
            abs_danger_score = torch.abs(danger_score)
            better = torch.logical_and(danger_score > 0, abs_danger_score <= randomness_source)
            worse = torch.logical_and(danger_score < 0, abs_danger_score <= randomness_source)
        else:
            better = danger_score > 0
            worse = danger_score < 0

        state.network_state[better] -= 1
        state.network_state[worse] += 1

        state.network_state = torch.clamp(
            state.network_state,
            0,
            self.patched_states + self.vulnerable_states + self.exploited_states - 1,
        )
        return state
