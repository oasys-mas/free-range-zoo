import unittest

import torch

from free_range_zoo.envs.rideshare.env.transitions.passenger_state import PassengerStateTransition
from free_range_zoo.envs.rideshare.env.structures.state import RideshareState


class TestPassengerStateTransition(unittest.TestCase):

    def setUp(self):
        self.state = RideshareState(
            agents=torch.tensor([[[0, 0], [1, 1]]], dtype=torch.int32),
            passengers=torch.tensor(
                [
                    [0, 1, 1, 1, 1, 5, 0, -1, 0, -1, -1],
                    [0, 1, 1, 1, 1, 5, 1, 0, 0, 1, -1],
                    [0, 1, 1, 1, 1, 5, 0, 0, 0, 1, -1],
                ],
                dtype=torch.int32,
            ),
        )

        self.transition = PassengerStateTransition(2, 1)

    def test_accept_targets_are_correctly_rewarded_to_closest_agents(self) -> None:
        result = self.transition(
            state=self.state,
            accepts=torch.tensor([[1, 1]], dtype=torch.bool),
            picks=torch.tensor([[0, 0]], dtype=torch.bool),
            targets=torch.tensor([[0, 0]], dtype=torch.int32),
            vectors=torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]], dtype=torch.int32),
            timesteps=torch.tensor([0], dtype=torch.int32),
        )

        self.assertEqual(result.passengers[0, 7], 1, 'Accept should be granted to the closest agent.')

    def test_accepted_passengers_are_updated_to_accepted_with_timestep_and_association(self) -> None:
        result = self.transition(
            state=self.state,
            accepts=torch.tensor([[1, 1]], dtype=torch.bool),
            picks=torch.tensor([[0, 0]], dtype=torch.bool),
            targets=torch.tensor([[0, 0]], dtype=torch.int32),
            vectors=torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]], dtype=torch.int32),
            timesteps=torch.tensor([10], dtype=torch.int32),
        )

        self.assertEqual(result.passengers[0, 6], 1, 'Accepted task should have task state updated.')
        self.assertEqual(result.passengers[0, 7], 1, 'Accept should have association property properly set.')
        self.assertEqual(result.passengers[0, 9], 10, 'Accept timestep should be properly set.')

    def test_pick_passengers_are_correctly_updated_to_pick_with_timestep(self) -> None:
        result = self.transition(
            state=self.state,
            accepts=torch.tensor([[0, 0]], dtype=torch.bool),
            picks=torch.tensor([[0, 1]], dtype=torch.bool),
            targets=torch.tensor([[0, 1]], dtype=torch.int32),
            vectors=torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]], dtype=torch.int32),
            timesteps=torch.tensor([10], dtype=torch.int32),
        )

        self.assertEqual(result.passengers[1, 6], 2, 'Picked passenger should have task state updated.')
        self.assertEqual(result.passengers[1, 10], 10, 'Picked timestep should be properly set.')

    def test_pick_passengers_far_from_agent_are_not_updated(self) -> None:
        expected_state = self.state.clone()

        result = self.transition(
            state=self.state,
            accepts=torch.tensor([[0, 0]], dtype=torch.bool),
            picks=torch.tensor([[1, 0]], dtype=torch.bool),
            targets=torch.tensor([[2, -100]], dtype=torch.int32),
            vectors=torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]], dtype=torch.int32),
            timesteps=torch.tensor([10], dtype=torch.int32),
        )

        self.assertTrue(
            torch.equal(expected_state.passengers, result.passengers), f'''
            \rPassengers should not be updated if far from agent.
                \rExpected:\n{expected_state.passengers}
                \rActual:\n{result.passengers}''')
