import unittest
import torch
from abc import ABC

from free_range_zoo.envs.wildfire.env.utils.in_range_check import chebyshev, euclidean


class TestDistanceFunction(ABC):

    def distance_function(self, agent_position, task_position, attack_range):
        raise NotImplementedError('Subclasses must implement this method')

    def test_implicit_differences(self) -> None:
        raise NotImplementedError('Subclasses must implement this method')

    def test_within_attack_range(self) -> None:
        agent_position = torch.tensor([(0, 0)])
        task_position = torch.tensor([(1, 0)])
        attack_range = torch.tensor([1])
        self.assertTrue(self.distance_function(agent_position, task_position, attack_range))

    def test_outside_attack_range(self) -> None:
        agent_position = torch.tensor([(0, 0)])
        task_position = torch.tensor([(1, 1)])
        attack_range = torch.tensor([0])
        self.assertFalse(self.distance_function(agent_position, task_position, attack_range))

    def test_multiple_agents_and_tasks_within_range(self) -> None:
        agent_position = torch.tensor([(0, 0), (1, 1)])
        task_position = torch.tensor([(1, 1), (2, 2)])
        attack_range = torch.tensor([2, 2])
        self.assertTrue(torch.all(self.distance_function(agent_position, task_position, attack_range)))

    def test_multiple_agents_and_tasks_outside_range(self) -> None:
        agent_position = torch.tensor([(0, 0), (1, 1)])
        task_position = torch.tensor([(3, 3), (4, 4)])
        attack_range = torch.tensor([2, 2])
        self.assertFalse(torch.any(self.distance_function(agent_position, task_position, attack_range)))

    def test_varying_ranges(self) -> None:
        agent_position = torch.tensor([(0, 0), (1, 1)])
        task_position = torch.tensor([(1, 1), (3, 3)])
        attack_range = torch.tensor([2, 1])
        expected_result = torch.tensor([True, False])
        self.assertTrue(torch.equal(self.distance_function(agent_position, task_position, attack_range), expected_result))


class TestChebyshev(TestDistanceFunction, unittest.TestCase):

    def distance_function(self, agent_position, task_position, attack_range):
        return chebyshev(agent_position, task_position, attack_range)

    def test_implicit_differences(self) -> None:
        agent_position = torch.tensor([(0, 0)])
        task_position = torch.tensor([(2, 1)])
        attack_range = torch.tensor([1])
        self.assertFalse(self.distance_function(agent_position, task_position, attack_range),
                         "Chebyshev should fail when diagonal distance exceeds range")

        task_position = torch.tensor([(1, 1)])
        attack_range = torch.tensor([1])
        self.assertTrue(self.distance_function(agent_position, task_position, attack_range),
                        "Chebyshev should succeed when within the maximum of x or y distance")


class TestEuclidean(TestDistanceFunction, unittest.TestCase):

    def distance_function(self, agent_position, task_position, attack_range):
        return euclidean(agent_position, task_position, attack_range)

    def test_implicit_differences(self) -> None:
        agent_position = torch.tensor([(0, 0)])
        task_position = torch.tensor([(2, 2)])
        attack_range = torch.tensor([2])
        self.assertFalse(self.distance_function(agent_position, task_position, attack_range),
                         "Euclidean should not succeed if diagonal distance is not within range")

        task_position = torch.tensor([(2, 1)])
        attack_range = torch.tensor([1])
        self.assertFalse(self.distance_function(agent_position, task_position, attack_range),
                         "Euclidean should fail when straight-line distance exceeds range")


if __name__ == '__main__':
    unittest.main()
