import unittest
from copy import deepcopy

import torch

from free_range_zoo.envs.wildfire.env.structures.state import WildfireState
from free_range_zoo.envs.wildfire.env.transitions.fire_spreads import FireSpreadTransition


class TestTransitionForward(unittest.TestCase):

    def setUp(self) -> None:
        self.parallel_envs = 2
        self.max_x = 4
        self.max_y = 4
        self.num_agents = 4

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.state = WildfireState(fires=torch.ones(
            (self.parallel_envs, self.max_y, self.max_x), dtype=torch.int32, device=self.device) * -1,
                                   intensity=torch.zeros((self.parallel_envs, self.max_y, self.max_x),
                                                         dtype=torch.int32,
                                                         device=self.device),
                                   fuel=torch.zeros((self.parallel_envs, self.max_y, self.max_x),
                                                    dtype=torch.int32,
                                                    device=self.device),
                                   agents=torch.randint(0,
                                                        self.max_y, (self.num_agents, 2),
                                                        dtype=torch.int32,
                                                        device=self.device),
                                   capacity=torch.ones((self.parallel_envs, self.num_agents),
                                                       dtype=torch.float32,
                                                       device=self.device),
                                   suppressants=torch.ones((self.parallel_envs, self.num_agents),
                                                           dtype=torch.float32,
                                                           device=self.device),
                                   equipment=torch.ones((self.parallel_envs, self.num_agents),
                                                        dtype=torch.int32,
                                                        device=self.device))
        self.state.fires[0, 1, 1] = 1
        self.state.intensity[0, 1, 1] = 1

        self.fire_spread_transition = FireSpreadTransition(
            fire_spread_weights=torch.tensor([[[[0, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0]]]],
                                             dtype=torch.float32,
                                             device=self.device),
            ignition_temperatures=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.int32),
            use_fire_fuel=False).to(self.device)

        self.randomness_source = torch.tensor(
            [[0.1, 0.6, 0.1, 0.6], [0.1, 0.6, 0.3, 0.8], [0.3, 0.8, 0.2, 0.7], [0.3, 0.8, 0.2, 0.7]],
            dtype=torch.float32,
            device=self.device).expand(self.parallel_envs, -1, -1)

    def test_fire_spreads_weight_application(self):
        result = self.fire_spread_transition(self.state, self.randomness_source)

        expected = torch.tensor([[[-1, -1, -1, -1], [1, 1, 1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                                 [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]],
                                dtype=torch.int32,
                                device=self.device)

        self.assertTrue(
            torch.allclose(result.fires, expected), f"""
            \rFires should match expected
                \rExpected:\n{expected}
                \rResult:\n{result.fires}""")

    def test_using_fire_fuel(self):
        self.fire_spread_transition.use_fire_fuel.fill_(True)
        self.state.fuel[0, 1, 0] = 1

        result = self.fire_spread_transition(self.state, self.randomness_source)

        expected = torch.tensor([[[-1, -1, -1, -1], [1, 1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                                 [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]],
                                dtype=torch.int32,
                                device=self.device)

        self.assertTrue(
            torch.allclose(result.fires, expected), f"""
            \rFires should match expected
                \rExpected:\n{expected}
                \rResult:\n{result.fires}""")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cpu_gpu_compatibility(self) -> None:
        transition_gpu = deepcopy(self.fire_spread_transition).cuda()
        transition_cpu = deepcopy(self.fire_spread_transition).cpu()

        cpu_result = transition_cpu(self.state.clone().to('cpu'), self.randomness_source.cpu())
        gpu_result = transition_gpu(self.state.clone().to('cuda'), self.randomness_source.cuda())

        for key in cpu_result.__annotations__:
            self.assertTrue(
                torch.allclose(getattr(cpu_result, key),
                               getattr(gpu_result, key).cpu()), f"""
                \rResult should be the same on CPU and GPU
                    \rCPU:\n{getattr(cpu_result, key)}
                    \rGPU:\n{getattr(gpu_result, key).cpu()}""")


if __name__ == '__main__':
    unittest.main()
