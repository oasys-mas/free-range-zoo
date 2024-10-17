import unittest
from copy import deepcopy

import torch

from free_range_zoo.envs.wildfire.env.structures.state import WildfireState
from free_range_zoo.envs.wildfire.env.transitions.suppressant_refill import SuppressantRefillTransition


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
                                   suppressants=torch.zeros((self.parallel_envs, self.num_agents),
                                                            dtype=torch.float32,
                                                            device=self.device),
                                   equipment=torch.ones((self.parallel_envs, self.num_agents),
                                                        dtype=torch.int32,
                                                        device=self.device))

        self.suppressant_refill_transition = SuppressantRefillTransition(agent_shape=(self.parallel_envs, self.num_agents),
                                                                         stochastic_refill=False,
                                                                         refill_probability=0.5,
                                                                         equipment_bonuses=torch.tensor([0, 0, 0],
                                                                                                        dtype=torch.float32,
                                                                                                        device=self.device)).to(
                                                                                                            self.device)

        self.randomness_source = torch.tensor([[0.1, 0.6, 0.1, 0.6], [0.1, 0.6, 0.3, 0.8]],
                                              dtype=torch.float32,
                                              device=self.device)

        self.refills = torch.tensor([[0, 1, 1, 1], [0, 1, 0, 1]], dtype=torch.bool, device=self.device)

    def test_stochastic_refill(self):
        self.suppressant_refill_transition.stochastic_refill.fill_(True)

        result = self.suppressant_refill_transition(self.state, self.refills, self.randomness_source)

        expected = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 0]], dtype=torch.float32, device=self.device)

        self.assertTrue(
            torch.allclose(result.suppressants, expected), f"""
            \rSuppressants should match expected
                \rExpected:\n{expected}
                \rResult:\n{result.suppressants}""")

    def test_deterministic_refill(self):
        result = self.suppressant_refill_transition(self.state, self.refills, self.randomness_source)

        expected = torch.tensor([[0, 1, 1, 1], [0, 1, 0, 1]], dtype=torch.float32, device=self.device)

        self.assertTrue(
            torch.allclose(result.suppressants, expected), f"""
            \rSuppressants should match expected
                \rExpected:\n{expected}
                \rResult:\n{result.suppressants}""")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cpu_gpu_compatibility(self) -> None:
        transition_gpu = deepcopy(self.suppressant_refill_transition).cuda()
        transition_cpu = deepcopy(self.suppressant_refill_transition).cpu()

        cpu_result = transition_cpu(self.state.clone().to('cpu'), self.refills.cpu(), self.randomness_source.cpu())
        gpu_result = transition_gpu(self.state.clone().to('cuda'), self.refills.cuda(), self.randomness_source.cuda())

        for key in cpu_result.__annotations__:
            self.assertTrue(
                torch.allclose(getattr(cpu_result, key),
                               getattr(gpu_result, key).cpu()), f"""
                \rResult should be the same on CPU and GPU
                    \rCPU:\n{getattr(cpu_result, key)}
                    \rGPU:\n{getattr(gpu_result, key).cpu()}""")


if __name__ == '__main__':
    unittest.main()
