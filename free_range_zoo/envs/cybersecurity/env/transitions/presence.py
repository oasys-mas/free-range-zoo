"""transition function for agent presence."""
import torch
from torch import nn

from free_range_zoo.envs.cybersecurity.env.structures.state import CybersecurityState


class PresenceTransition(nn.Module):
    """
    Transition function for agent presence.

    Agents that are present have a probability to leave determined by `1 - persist_probs`.
    Agents that are not present have a return probability determined by `return_probs`.

    Defenders which return are automatically placed at the home node.
    """

    def __init__(self, persist_probs: torch.FloatTensor, return_probs: torch.FloatTensor, num_attackers: int) -> None:
        """
        Initialize the transition function.

        Args:
            persist_probs: torch.FloatTensor - the probability for each agent to leave the environment
            return_probs: torch.FloatTensor - the probability for each agent to return to the environment
            num_attackers: int - the number of attackers in the environment
        """
        super().__init__()

        self.register_buffer('persist_probs', persist_probs)
        self.register_buffer('return_probs', return_probs)

        self.register_buffer('num_attackers', torch.tensor(num_attackers, dtype=torch.int32))

    @torch.no_grad()
    def forward(self, state: CybersecurityState, randomness_source: torch.FloatTensor) -> CybersecurityState:
        """
        Calculate the next presence states for all agents.

        Args:
            state: CybersecurityState - the current state of the environment
            randomness_source: torch.FloatTensor - the source of randomness for the transition
        Returns:
            CybersecurityState - the next state of the environment with the presence states transformed
        """
        # Calculate which agents return to the environment
        return_mask = randomness_source < self.return_probs
        return_mask = torch.logical_and(torch.logical_not(state.presence), return_mask)

        # Calculate which agents leave the environment
        leave_mask = randomness_source >= self.persist_probs
        leave_mask = torch.logical_and(state.presence, leave_mask)

        # Apply the masks
        state.presence[return_mask] = True
        state.presence[leave_mask] = False

        # Defenders that return are placed at the home node
        state.location[return_mask[:, self.num_attackers:]] = -1

        return state
