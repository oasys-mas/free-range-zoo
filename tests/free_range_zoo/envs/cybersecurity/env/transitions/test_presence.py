import unittest

import torch

from free_range_zoo.envs.cybersecurity.env.transitions.presence import PresenceTransition
from free_range_zoo.envs.cybersecurity.env.structures.state import CybersecurityState


class TestPresenceTransitionForward(unittest.TestCase):

    def setUp(self):
        self.state = CybersecurityState(
            network_state=torch.arange(0, 12, dtype=torch.int32).reshape((4, 3)),
            location=torch.arange(0, 4, dtype=torch.int32).unsqueeze(0).repeat(4, 1),
            presence=torch.zeros((4, 8), dtype=torch.bool),
        )

        self.transition = PresenceTransition(
            persist_probs=torch.zeros_like(self.state.presence, dtype=torch.float32),
            return_probs=torch.zeros_like(self.state.presence, dtype=torch.float32),
            num_attackers=4,
        )

        self.randomness_source = torch.empty((4, 8), dtype=torch.float32).fill_(0.5)

    def test_normal_forward(self) -> None:
        self.transition.persist_probs.fill_(1.0)
        self.transition.return_probs.fill_(1.0)

        self.state.presence = torch.tensor([[True, False, True, False, True, False, True, False]], dtype=torch.bool).repeat(
            (4, 1))

        result = self.transition.forward(self.state.clone(), self.randomness_source)

        expected = torch.tensor([[True, True, True, True, True, True, True, True]], dtype=torch.bool).repeat((4, 1))
        expected = {
            'presence': torch.ones_like(self.state.presence, dtype=torch.bool),
            'location': torch.tensor([[0, -1, 2, -1]], dtype=torch.int32).repeat((4, 1)),
        }

        self.assertTrue(
            torch.equal(result.presence, expected['presence']), f"""
                \rPresence match expected.
                    \rExpected: {expected['presence']}
                    \rActual: {result.presence}""")

        self.assertTrue(
            torch.equal(result.location, expected['location']), f"""
                \rLocation should match expected.
                    \rExpected: {expected['location']}
                    \rActual: {result.location}""")

        self.assertTrue(
            torch.equal(result.network_state, self.state.network_state), f"""
                \rNetwork state should not change with presence transition.
                    \rExpected: {self.state.network_state}
                    \rActual: {result.network_state}""")

    def test_persist_probabilty_zero(self) -> None:
        self.state.presence.fill_(True)

        result = self.transition.forward(self.state.clone(), self.randomness_source)

        self.assertTrue(
            torch.equal(result.presence, torch.zeros_like(self.state.presence)), f"""
            \rPresence should all change when persist probability is zero.
                \rExpected: {self.state.presence}
                \rActual: {result.presence}""")

        self.assertTrue(
            torch.equal(result.location, self.state.location), f"""
            \rLocation should not change when persist probability is zero.
                \rExpected: {self.state.location}
                \rActual: {result.location}""")

        self.assertTrue(
            torch.equal(result.network_state, self.state.network_state), f"""
                \rNetwork state should not change with presence transition
                    \rExpected: {self.state.network_state}
                    \rActual: {result.network_state}""")

    def test_return_probability_zero(self) -> None:
        result = self.transition.forward(self.state.clone(), self.randomness_source)

        self.assertTrue(
            torch.equal(result.presence, self.state.presence), f"""
            \rPresence should not change when return probability is zero.
                \rExpected: {self.state.presence}
                \rActual: {result.presence}""")

        self.assertTrue(
            torch.equal(result.location, self.state.location), f"""
            \rLocation should not change when return probability is zero.
                \rExpected: {self.state.location}
                \rActual: {result.location}""")

        self.assertTrue(
            torch.equal(result.network_state, self.state.network_state), f"""
            \rNetwork state should not change with presence transition
                \rExpected: {self.state.network_state}
                \rActual: {result.network_state}""")
