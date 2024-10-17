import unittest
from copy import deepcopy

import torch

from free_range_zoo.envs.wildfire.env.structures.state import WildfireState
from free_range_zoo.envs.wildfire.env.transitions.suppressant_decrease import SuppressantDecreaseTransition


class TestTransitionForward(unittest.TestCase):

    def setUp(self) -> None:
        self.parallel_envs = 2
        self.max_x = 4
        self.max_y = 4
        self.num_agents = 4

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.state = WildfireState(fires=torch.zeros(
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

        self.suppressant_decrease_transition = SuppressantDecreaseTransition(agent_shape=(self.parallel_envs, self.num_agents),
                                                                             stochastic_decrease=False,
                                                                             decrease_probability=0.5).to(self.device)

        self.randomness_source = torch.tensor([[0.1, 0.6, 0.1, 0.6], [0.1, 0.6, 0.3, 0.8]],
                                              dtype=torch.float32,
                                              device=self.device)

        self.used_suppressants = torch.tensor([[0, 1, 1, 1], [0, 1, 0, 1]], dtype=torch.bool, device=self.device)

    def test_stochastic_decrease(self):
        self.suppressant_decrease_transition.stochastic_decrease.fill_(True)

        result = self.suppressant_decrease_transition(self.state, self.used_suppressants, self.randomness_source)

        expected = torch.tensor([[1, 1, 0, 1], [1, 1, 1, 1]], dtype=torch.float32, device=self.device)

        self.assertTrue(
            torch.allclose(result.suppressants, expected), f"""
            \rSuppressants should match expected
                \rExpected:\n{expected}
                \rResult:\n{result.suppressants}""")

    def test_deterministic_decrease(self):
        result = self.suppressant_decrease_transition(self.state, self.used_suppressants, self.randomness_source)

        expected = torch.tensor([[1, 0, 0, 0], [1, 0, 1, 0]], dtype=torch.float32, device=self.device)

        self.assertTrue(
            torch.allclose(result.suppressants, expected), f"""
            \rSuppressants should match expected
                \rExpected:\n{expected}
                \rResult:\n{result.suppressants}""")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cpu_gpu_compatibility(self) -> None:
        transition_gpu = deepcopy(self.suppressant_decrease_transition).cuda()
        transition_cpu = deepcopy(self.suppressant_decrease_transition).cpu()

        cpu_result = transition_cpu(self.state.clone().to('cpu'), self.used_suppressants.cpu(), self.randomness_source.cpu())
        gpu_result = transition_gpu(self.state.clone().to('cuda'), self.used_suppressants.cuda(), self.randomness_source.cuda())

        for key in cpu_result.__annotations__:
            self.assertTrue(
                torch.allclose(getattr(cpu_result, key),
                               getattr(gpu_result, key).cpu()), f"""
                \rResult should be the same on CPU and GPU
                    \rCPU:\n{getattr(cpu_result, key)}
                    \rGPU:\n{getattr(gpu_result, key).cpu()}""")


if __name__ == '__main__':
    unittest.main()
