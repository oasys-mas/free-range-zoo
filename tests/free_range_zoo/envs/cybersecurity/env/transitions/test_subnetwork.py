import unittest

import torch

from free_range_zoo.envs.cybersecurity.env.transitions.subnetwork import SubnetworkTransition
from free_range_zoo.envs.cybersecurity.env.structures.state import CybersecurityState


class TestTransitionForward(unittest.TestCase):

    def setUp(self) -> None:
        self.state = CybersecurityState(
            network_state=torch.arange(0, 3, dtype=torch.int32).unsqueeze(0).repeat(4, 1),
            location=torch.arange(0, 4, dtype=torch.int32).unsqueeze(0).repeat(4, 1),
            presence=torch.zeros((4, 8), dtype=torch.bool),
        )

        self.transition = SubnetworkTransition(
            patched_states=1,
            vulnerable_states=2,
            exploited_states=2,
            temperature=1.0,
            stochastic_state=False,
        )

        self.randomness_source = torch.empty((4, 3), dtype=torch.float32).fill_(0.5)

    def test_deterministic_transition(self) -> None:
        self.transition.stochastic_state.fill_(False)

        result = self.transition.forward(
            self.state.clone(),
            torch.tensor([[1, 0, 1]]).repeat((4, 1)),
            torch.tensor([[0, 1, 0]]).repeat((4, 1)),
            self.randomness_source,
        )

        expected = torch.tensor([[1, 0, 3]]).repeat(4, 1)

        self.assertTrue(
            torch.equal(result.network_state, expected), f"""
                \rNetwork state should change to expected.
                    \rExpected:\n{expected}
                    \rActual:\n{result.network_state}""")

        self.assertTrue(
            torch.equal(result.location, self.state.location), f"""
                \rLocation should not change.
                    \rExpected:\n{self.state.location}
                    \rActual:\n{result.location}""")

        self.assertTrue(
            torch.equal(result.presence, self.state.presence), f"""
                \rPresence should not change.
                    \rExpected:\n{self.state.presence}
                    \rActual:\n{result.presence}""")

    def test_stochastic_transition(self) -> None:
        self.transition.stochastic_state.fill_(True)
        self.transition.temperature.fill_(1.0)

        result = self.transition.forward(
            self.state.clone(),
            torch.tensor([[1, 0, 2]]).repeat((4, 1)),
            torch.tensor([[0, 1, 0]]).repeat((4, 1)),
            self.randomness_source,
        )

        expected = torch.tensor([[0, 1, 2]]).repeat(4, 1)

        self.assertTrue(
            torch.equal(result.network_state, expected), f"""
                \rNetwork state should change to expected.
                    \rExpected:\n{expected}
                    \rActual:\n{result.network_state}""")

        self.assertTrue(
            torch.equal(result.location, self.state.location), f"""
                \rLocation should not change.
                    \rExpected:\n{self.state.location}
                    \rActual:\n{result.location}""")

        self.assertTrue(
            torch.equal(result.presence, self.state.presence), f"""
                \rPresence should not change.
                    \rExpected:\n{self.state.presence}
                    \rActual:\n{result.presence}""")

    def test_clamping(self) -> None:
        self.state.network_state.fill_(0)

        result = self.transition.forward(
            self.state.clone(),
            torch.zeros_like(self.randomness_source),
            torch.ones_like(self.randomness_source),
            self.randomness_source,
        )

        self.assertTrue(
            torch.equal(result.network_state, self.state.network_state), f"""
                \rNetwork state should be clamped to zero.
                    \rExpected:\n{self.state.network_state}
                    \rActual:\nn{result.network_state}""")

        self.state.network_state.fill_(4)

        result = self.transition.forward(
            self.state.clone(),
            torch.ones_like(self.randomness_source),
            torch.zeros_like(self.randomness_source),
            self.randomness_source,
        )

        self.assertTrue(
            torch.equal(result.network_state, self.state.network_state), f"""
                \rNetwork state should be clamped to maximum.
                    \rExpected:\n{self.state.network_state}
                    \rActual:\n{result.network_state}""")
