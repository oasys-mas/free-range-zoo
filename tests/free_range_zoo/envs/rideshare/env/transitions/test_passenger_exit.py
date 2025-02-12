import unittest

import torch

from free_range_zoo.envs.rideshare.env.transitions.passenger_exit import PassengerExitTransition
from free_range_zoo.envs.rideshare.env.structures.state import RideshareState


class TestPassengerExitTransition(unittest.TestCase):

    def setUp(self):
        self.state = RideshareState(
            agents=torch.tensor([[[1, 1]]], dtype=torch.int32),
            passengers=torch.tensor([[0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 0]], dtype=torch.int32),
        )

        self.transition = PassengerExitTransition(1)

    def test_passengers_do_not_exit_even_at_location_if_not_dropped(self) -> None:
        result, fares = self.transition(
            state=self.state,
            drops=torch.tensor([[0]], dtype=torch.bool),
            targets=torch.tensor([[0]], dtype=torch.int32),
            vectors=torch.tensor([[[1, 1, 1, 1]]], dtype=torch.int32),
            timesteps=torch.tensor([1], dtype=torch.int32),
        )

        self.assertEqual(result.passengers.size(0), 1, 'Passenger should not be dropped if drop action at destination.')

    def test_passengers_do_not_exit_if_dropped_at_not_destination(self) -> None:
        self.state.agents = torch.tensor([[[0, 0]]], dtype=torch.int32)
        result, fares = self.transition(
            state=self.state,
            drops=torch.tensor([[0]], dtype=torch.bool),
            targets=torch.tensor([[0]], dtype=torch.int32),
            vectors=torch.tensor([[[0, 0, 1, 1]]], dtype=torch.int32),
            timesteps=torch.tensor([1], dtype=torch.int32),
        )

        self.assertEqual(result.passengers.size(0), 1, 'Passenger should not be removed from environment if not at destination')

    def test_passengers_exit_at_location_if_dropped(self) -> None:
        result, fares = self.transition(
            state=self.state,
            drops=torch.tensor([[1]], dtype=torch.bool),
            targets=torch.tensor([[0]], dtype=torch.int32),
            vectors=torch.tensor([[[1, 1, 1, 1]]], dtype=torch.int32),
            timesteps=torch.tensor([1], dtype=torch.int32),
        )

        self.assertEqual(result.passengers.size(0), 0, 'Passenger should be removed from environment if dropped at destination.')

    def test_fares_are_correctly_allocated_to_agents_after_completion(self) -> None:
        self.state.agents = torch.tensor([[[1, 1], [0, 0]]])
        result, fares = self.transition(
            state=self.state,
            drops=torch.tensor([[1, -100]], dtype=torch.bool),
            targets=torch.tensor([[0, -100]], dtype=torch.int32),
            vectors=torch.tensor([[[1, 1, 1, 1], [-100, -100, -100, -100]]], dtype=torch.int32),
            timesteps=torch.tensor([1], dtype=torch.int32),
        )

        self.assertTrue(
            torch.equal(torch.tensor([[2, 0]], dtype=torch.float32), fares), f'''Fares should match expected
                        \rExpected:\n{torch.tensor([[2, 0]], dtype=torch.float32)}
                        \rActual:\n{fares}''')
