import unittest
from copy import deepcopy

import torch

from free_range_zoo.envs.wildfire.env.structures.state import WildfireState
from free_range_zoo.envs.wildfire.env.transitions.fire_decrease import FireDecreaseTransition


class TestTransitionForward(unittest.TestCase):

    def setUp(self) -> None:
        self.parallel_envs = 2
        self.max_x = 4
        self.max_y = 4
        self.num_agents = 4

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.state = WildfireState(
            fires=torch.ones(
                (self.parallel_envs, self.max_y, self.max_x),
                dtype=torch.int32,
                device=self.device,
            ),
            intensity=torch.ones(
                (self.parallel_envs, self.max_y, self.max_x),
                dtype=torch.int32,
                device=self.device,
            ),
            fuel=torch.zeros(
                (self.parallel_envs, self.max_y, self.max_x),
                dtype=torch.int32,
                device=self.device,
            ),
            agents=torch.randint(
                0,
                self.max_y,
                (self.num_agents, 2),
                dtype=torch.int32,
                device=self.device,
            ),
            capacity=torch.ones(
                (self.parallel_envs, self.num_agents),
                dtype=torch.float32,
                device=self.device,
            ),
            suppressants=torch.ones(
                (self.parallel_envs, self.num_agents),
                dtype=torch.float32,
                device=self.device,
            ),
            equipment=torch.ones(
                (self.parallel_envs, self.num_agents),
                dtype=torch.int32,
                device=self.device,
            ),
        )

        self.fire_decrease_transition = FireDecreaseTransition(
            fire_shape=(self.parallel_envs, self.max_y, self.max_x),
            stochastic_decrease=False,
            decrease_probability=0.5,
            extra_power_decrease_bonus=0.1,
        ).to(self.device)

        self.randomness_source = torch.tensor(
            [[0.1, 0.6, 0.1, 0.6], [0.1, 0.6, 0.3, 0.8], [0.3, 0.8, 0.2, 0.7], [0.3, 0.8, 0.2, 0.7]],
            dtype=torch.float32,
            device=self.device).expand(self.parallel_envs, -1, -1)

        self.attack_counts = torch.ones((self.parallel_envs, self.max_y, self.max_x), dtype=torch.int32, device=self.device)

    def test_stochastic_decrease(self) -> None:
        self.fire_decrease_transition.stochastic_decrease.fill_(True)

        result = self.fire_decrease_transition.forward(self.state, self.attack_counts, self.randomness_source)

        fires_expected = torch.tensor([[[-1, 1, -1, 1], [-1, 1, -1, 1], [-1, 1, -1, 1], [-1, 1, -1, 1]],
                                       [[-1, 1, -1, 1], [-1, 1, -1, 1], [-1, 1, -1, 1], [-1, 1, -1, 1]]],
                                      dtype=torch.int32,
                                      device=self.device)
        intensity_expected = torch.tensor(
            [[[0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]], [[0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]]],
            device=self.device,
            dtype=torch.int32)

        self.assertTrue(
            torch.allclose(result.fires, fires_expected), f"""
            \rFires should match expected
                \rExpected:\n{fires_expected}
                \rResult:\n{result.fires}""")
        self.assertTrue(
            torch.allclose(result.intensity, intensity_expected), f"""
            \rIntensity should match expected
                \rExpected:\n{intensity_expected}
                \rResult:\n{result.intensity}""")

    def test_deterministic_decrease(self) -> None:
        result = self.fire_decrease_transition.forward(self.state, self.attack_counts, self.randomness_source)

        fires_expected = torch.tensor([[[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                                       [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]],
                                      device=self.device,
                                      dtype=torch.int32)
        intensity_expected = torch.tensor(
            [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]],
            device=self.device,
            dtype=torch.int32)

        self.assertTrue(
            torch.allclose(result.fires, fires_expected), f"""
            \rFires should match expected
                \rExpected:\n{fires_expected}
                \rResult:\n{result.fires}""")
        self.assertTrue(
            torch.allclose(result.intensity, intensity_expected), f"""
            \rIntensity should match expected
                \rExpected:\n{intensity_expected}
                \rResult:\n{result.intensity}""")

    def test_bonus_decrease_probability(self) -> None:
        self.fire_decrease_transition.stochastic_decrease.fill_(True)
        self.attack_counts.fill_(3)

        result = self.fire_decrease_transition.forward(self.state, self.attack_counts, self.randomness_source)

        fires_expected = torch.tensor([[[-1, -1, -1, -1], [-1, -1, -1, 1], [-1, 1, -1, 1], [-1, 1, -1, 1]],
                                       [[-1, -1, -1, -1], [-1, -1, -1, 1], [-1, 1, -1, 1], [-1, 1, -1, 1]]],
                                      device=self.device,
                                      dtype=torch.int32)

        intensity_expected = torch.tensor(
            [[[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]], [[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]]],
            device=self.device,
            dtype=torch.int32)

        self.assertTrue(
            torch.allclose(result.fires, fires_expected), f"""
            \rFires should match expected
                \rExpected:\n{fires_expected}
                \rResult:\n{result.fires}""")
        self.assertTrue(
            torch.allclose(result.intensity, intensity_expected), f"""
            \rIntensity should match expected
                \rExpected:\n{intensity_expected}
                \rResult:\n{result.intensity}""")

    def test_returning_put_out(self) -> None:
        _, result = self.fire_decrease_transition.forward(self.state,
                                                          self.attack_counts,
                                                          self.randomness_source,
                                                          return_put_out=True)

        expected = torch.tensor(
            [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]],
            dtype=torch.bool,
            device=self.device)

        self.assertTrue(
            torch.allclose(result, expected), f"""
            \rPut out mask should match expected
                \rExpected:\n{expected}
                \rResult:\n{result}""")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cpu_gpu_compatibility(self) -> None:
        transition_gpu = deepcopy(self.fire_decrease_transition).cuda()
        transition_cpu = deepcopy(self.fire_decrease_transition).cpu()

        cpu_result = transition_cpu(self.state.clone().to('cpu'), self.attack_counts.cpu(), self.randomness_source.cpu())
        gpu_result = transition_gpu(self.state.clone().to('cuda'), self.attack_counts.cuda(), self.randomness_source.cuda())

        for key in cpu_result.__annotations__:
            self.assertTrue(
                torch.allclose(getattr(cpu_result, key),
                               getattr(gpu_result, key).cpu()), f"""
                \rResult should be the same on CPU and GPU
                    \rCPU:\n{getattr(cpu_result, key)}
                    \rGPU:\n{getattr(gpu_result, key).cpu()}""")


if __name__ == '__main__':
    unittest.main()
