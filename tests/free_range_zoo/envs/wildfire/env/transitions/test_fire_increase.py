import unittest
from copy import deepcopy

import torch

from free_range_zoo.free_range_zoo.envs.wildfire.env.structures.state import WildfireState
from free_range_zoo.free_range_zoo.envs.wildfire.env.transitions.fire_increase import FireIncreaseTransition


class TestFireIncreaseTransition(unittest.TestCase):
    def setUp(self) -> None:
        self.parallel_envs = 2
        self.max_x = 4
        self.max_y = 4
        self.num_agents = 4

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.state = WildfireState(
            fires=torch.ones((self.parallel_envs, self.max_y, self.max_x), dtype=torch.int32, device=self.device),
            intensity=torch.ones((self.parallel_envs, self.max_y, self.max_x), dtype=torch.int32, device=self.device),
            fuel=torch.zeros((self.parallel_envs, self.max_y, self.max_x), dtype=torch.int32, device=self.device),
            agents=torch.randint(0, self.max_y, (self.num_agents, 2), dtype=torch.int32, device=self.device),
            capacity=torch.ones((self.parallel_envs, self.num_agents), dtype=torch.float32, device=self.device),
            suppressants=torch.ones((self.parallel_envs, self.num_agents), dtype=torch.float32, device=self.device),
            equipment=torch.ones((self.parallel_envs, self.num_agents), dtype=torch.int32, device=self.device)
        )

        self.fire_increase_transition = FireIncreaseTransition(
            fire_shape=(self.parallel_envs, self.max_y, self.max_x),
            fire_states=5,
            stochastic_increase=False,
            intensity_increase_probability=0.5,
            stochastic_burnouts=False,
            burnout_probability=0.2).to(self.device)

        self.randomness_source = torch.tensor([
            [0.1, 0.6, 0.1, 0.6],
            [0.1, 0.6, 0.3, 0.8],
            [0.3, 0.8, 0.2, 0.7],
            [0.3, 0.8, 0.2, 0.7]
        ], dtype=torch.float32, device=self.device).repeat((self.parallel_envs, 1, 1))

        self.attack_counts = torch.zeros((self.parallel_envs, self.max_y, self.max_x), dtype=torch.int32, device=self.device)

    def test_stochastic_increase(self) -> None:
        self.fire_increase_transition.stochastic_increase.fill_(True)

        result = self.fire_increase_transition(self.state, self.attack_counts, self.randomness_source)

        expected_fires = torch.tensor([
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        ], device=self.device, dtype=torch.int32)
        expected_intensity = torch.tensor([
            [[2, 1, 2, 1], [2, 1, 2, 1], [2, 1, 2, 1], [2, 1, 2, 1]],
            [[2, 1, 2, 1], [2, 1, 2, 1], [2, 1, 2, 1], [2, 1, 2, 1]]
        ], device=self.device, dtype=torch.int32)

        self.assertTrue(torch.allclose(result.fires, expected_fires), f"""
            \rFires should match expected
                \rExpected:\n{expected_fires}
                \rResult:\n{result.fires}""")

        self.assertTrue(torch.allclose(result.intensity, expected_intensity), f"""
            \rIntensity should match expected
                \rExpected:\n{expected_intensity}
                \rResult:\n{result.intensity}""")

    def test_deterministic_increase(self) -> None:
        result = self.fire_increase_transition(self.state, self.attack_counts, self.randomness_source)

        expected_fires = torch.tensor([
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        ], device=self.device, dtype=torch.int32)
        expected_intensity = torch.tensor([
            [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]],
            [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]
        ], device=self.device, dtype=torch.int32)

        self.assertTrue(torch.allclose(result.fires, expected_fires), f"""
            \rFires should match expected
                \rExpected:\n{expected_fires}
                \rResult:\n{result.fires}""")

        self.assertTrue(torch.allclose(result.intensity, expected_intensity), f"""
            \rIntensity should match expected
                \rExpected:\n{expected_intensity}
                \rResult:\n{result.intensity}""")

    def test_stochastic_burnouts(self) -> None:
        self.fire_increase_transition.stochastic_burnouts.fill_(True)
        self.state.intensity.fill_(3)

        result = self.fire_increase_transition(self.state, self.attack_counts, self.randomness_source)

        expected_fires = torch.tensor([
            [[-1, 1, -1, 1], [-1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            [[-1, 1, -1, 1], [-1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        ], device=self.device, dtype=torch.int32)
        expected_intensity = torch.tensor([
            [[4, 3, 4, 3], [4, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]],
            [[4, 3, 4, 3], [4, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]]
        ], device=self.device, dtype=torch.int32)

        self.assertTrue(torch.allclose(result.fires, expected_fires), f"""
            \rFires should match expected
                \rExpected:\n{expected_fires}
                \rResult:\n{result.fires}""")

        self.assertTrue(torch.allclose(result.intensity, expected_intensity), f"""
            \rIntensity should match expected
                \rExpected:\n{expected_intensity}
                \rResult:\n{result.intensity}""")

    def test_stochastic_increase_and_burnouts(self) -> None:
        self.fire_increase_transition.stochastic_increase.fill_(True)
        self.fire_increase_transition.stochastic_burnouts.fill_(True)

        self.state.intensity = torch.tensor([
            [[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]],
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        ], dtype=torch.int32, device=self.device)

        result = self.fire_increase_transition(self.state, self.attack_counts, self.randomness_source)

        expected_fires = torch.tensor([
            [[-1, 1, -1, 1], [-1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        ], device=self.device, dtype=torch.int32)
        expected_intensity = torch.tensor([
            [[4, 3, 4, 3], [4, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]],
            [[2, 1, 2, 1], [2, 1, 2, 1], [2, 1, 2, 1], [2, 1, 2, 1]]
        ], device=self.device, dtype=torch.int32)

        self.assertTrue(torch.allclose(result.fires, expected_fires), f"""
            \rFires should match expected
                \rExpected:\n{expected_fires}
                \rResult:\n{result.fires}""")

        self.assertTrue(torch.allclose(result.intensity, expected_intensity), f"""
            \rIntensity should match expected
                \rExpected:\n{expected_intensity}
                \rResult:\n{result.intensity}""")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cpu_gpu_compatibility(self) -> None:
        transition_gpu = deepcopy(self.fire_increase_transition).cuda()
        transition_cpu = deepcopy(self.fire_increase_transition).cpu()

        cpu_result = transition_cpu(self.state.clone().to('cpu'), self.attack_counts.cpu(), self.randomness_source.cpu())
        gpu_result = transition_gpu(self.state.clone().to('cuda'), self.attack_counts.cuda(), self.randomness_source.cuda())

        for key in cpu_result.__annotations__:
            self.assertTrue(torch.allclose(getattr(cpu_result, key), getattr(gpu_result, key).cpu()), f"""
                \rResult should be the same on CPU and GPU
                    \rCPU:\n{getattr(cpu_result, key)}
                    \rGPU:\n{getattr(gpu_result, key).cpu()}""")


if __name__ == '__main__':
    unittest.main()
