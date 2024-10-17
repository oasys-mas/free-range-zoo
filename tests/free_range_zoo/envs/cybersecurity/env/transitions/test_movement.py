import unittest

import torch

from free_range_zoo.envs.cybersecurity.env.transitions.movement import MovementTransition
from free_range_zoo.envs.cybersecurity.env.structures.state import CybersecurityState


class TestMovementTransitionForward(unittest.TestCase):

    def setUp(self):
        self.state = CybersecurityState(
            network_state=torch.arange(0, 12, dtype=torch.int32).reshape((4, 3)),
            location=torch.arange(0, 4, dtype=torch.int32).unsqueeze(0).repeat(4, 1),
            presence=torch.zeros((4, 8), dtype=torch.bool),
        )

        self.transition = MovementTransition()

    def test_no_movement(self) -> None:
        movement_mask = torch.zeros_like(self.state.location, dtype=torch.bool)
        movement_targets = torch.empty_like(self.state.location, dtype=torch.int32)

        state = self.transition.forward(self.state, movement_targets, movement_mask)

        self.assertTrue(torch.equal(state.network_state, self.state.network_state))
        self.assertTrue(torch.equal(state.location, self.state.location))
        self.assertTrue(torch.equal(state.presence, self.state.presence))

    def test_normal_forward(self) -> None:
        movement_mask = torch.ones_like(self.state.location, dtype=torch.bool)
        movement_targets = torch.tensor([0, 1, 2, 1], dtype=torch.int32).unsqueeze(0).repeat(4, 1)

        state = self.transition.forward(self.state.clone(), movement_targets, movement_mask)

        self.assertTrue(
            torch.equal(state.network_state, self.state.network_state), f"""
            \rNetwork state should not change after movement.
                \rExpected: {self.state.network_state}
                \rActual: {state.network_state}""")

        self.assertTrue(
            torch.equal(state.presence, self.state.presence), f"""
                \rPresence should not change after movement.
                    \rExpected: {self.state.presence}
                    \rActual: {state.presence}""")

        self.assertTrue(
            torch.equal(state.location, movement_targets), f"""
                \rLocation should change to movement_targets.
                    \rExpected: {movement_targets}
                    \rActual: {state.location}""")
