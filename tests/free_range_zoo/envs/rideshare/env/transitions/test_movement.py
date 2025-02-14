import unittest

import torch
import math

from free_range_zoo.envs.rideshare.env.transitions.movement import MovementTransition
from free_range_zoo.envs.rideshare.env.structures.state import RideshareState


class TestPassengerEntryTransition(unittest.TestCase):

    def setUp(self):
        self.state = RideshareState(
            agents=torch.tensor([[[0, 0], [1, 1]]], dtype=torch.int32),
            passengers=torch.tensor(
                [
                    [0, 0, 0, 1, 1, 5, 2, 0, 0, 1, 1],
                    [0, 1, 1, 1, 1, 5, 2, 1, 0, 1, 1],
                    [0, 1, 1, 1, 1, 5, 0, -1, 0, 1, 1],
                ],
                dtype=torch.int32,
            ),
        )

        self.transition = MovementTransition(1, 2, False, False)

    def test_all_movement_happens_in_the_correct_direction(self) -> None:
        result, distances = self.transition(
            state=self.state,
            vectors=torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]], dtype=torch.int32),
        )

        self.assertTrue(
            torch.equal(torch.tensor([0, 1], dtype=torch.int32), result.agents[0, 0]),
            f'''First agent should move down or left to the present position.
                \rExpected:\n{torch.tensor([0, 1], dtype=torch.int32)}
                \rActual:\n{result.agents[0, 0]}''')

        self.assertTrue(
            torch.equal(torch.tensor([1, 1], dtype=torch.int32), result.agents[0, 1]),
            f'''Second agent should remain in current position since it is already in optimal position.
                \rExpected:\n{torch.tensor([1, 1], dtype=torch.int32)}
                \rActual:\n{result.agents[0, 0]}''')

    def test_all_movement_happens_in_the_correct_direction_with_eight_dir(self) -> None:
        self.transition = MovementTransition(1, 2, False, True)

        result, distances = self.transition(
            state=self.state,
            vectors=torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]], dtype=torch.int32),
        )

        self.assertTrue(
            torch.equal(torch.tensor([1, 1], dtype=torch.int32), result.agents[0, 0]),
            f'''First agent should diagonally down with eight directional movement.
                \rExpected:\n{torch.tensor([1, 1], dtype=torch.int32)}
                \rActual:\n{result.agents[0, 0]}''')

        self.assertTrue(
            torch.equal(torch.tensor([1, 1], dtype=torch.int32), result.agents[0, 1]),
            f'''Second agent should remain in current position since it is already in optimal position.
                \rExpected:\n{torch.tensor([1, 1], dtype=torch.int32)}
                \rActual:\n{result.agents[0, 1]}''')

    def test_distances_are_correctly_calculated_for_movement(self) -> None:
        result, distances = self.transition(
            state=self.state,
            vectors=torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]], dtype=torch.int32),
        )

        self.assertTrue(
            torch.equal(torch.tensor([[1, 0]], dtype=torch.float32), distances),
            f'''Distances should be properly representative of the distance an agent moved.
                \rExpected:\n{torch.tensor([[1, 0]], dtype=torch.float32)}
                \rActual:\n{distances}''')

    def test_that_distance_is_still_correctly_calculated_with_eight_dir_movement(self) -> None:
        self.transition = MovementTransition(1, 2, False, True)

        result, distances = self.transition(
            state=self.state,
            vectors=torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]], dtype=torch.int32),
        )

        self.assertTrue(
            torch.equal(torch.tensor([[math.sqrt(2), 0]], dtype=torch.float32), distances),
            f'''Distances should be properly representative of the distance an agent moved.
                \rExpected:\n{torch.tensor([[math.sqrt(2), 0]], dtype=torch.float32)}
                \rActual:\n{distances}''')

    def test_tasks_of_agent_move_with_them(self) -> None:
        before = self.state.passengers.clone()
        result, distances = self.transition(
            state=self.state,
            vectors=torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]], dtype=torch.int32),
        )

        self.assertTrue(
            torch.equal(result.passengers[0, 1:3], result.agents[0, 0]), f'''
            \rTasks should move with their respective agents
                \rExpected:\n{result.agents[0, 0]}
                \rActual:\n{result.passengers[0, 1:3]}''')

        self.assertTrue(
            torch.equal(result.passengers[1, 1:3], result.agents[0, 1]), f'''
            \rTasks should move with their respective agents
                \rExpected:\n{result.agents[0, 1]}
                \rActual:\n{result.passengers[1, 1:3]}''')

        self.assertTrue(
            torch.equal(result.passengers[2, 1:3], before[2, 1:3]), f'''
            \rTasks which do not belong to any agent should not be moved.
                \rExpected:\n{result.agents[0, 1]}
                \rActual:\n{before[2, 1:3]}''')
