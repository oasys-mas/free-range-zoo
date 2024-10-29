import unittest
from copy import deepcopy

import torch

from free_range_zoo.envs.wildfire.env.structures.state import WildfireState
from free_range_zoo.envs.wildfire.env.transitions.capacity import CapacityTransition


class TestTransitionForward(unittest.TestCase):

    def setUp(self) -> None:
        self.parallel_envs = 2
        self.max_x = 4
        self.max_y = 4
        self.num_agents = 4

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.state = WildfireState(
            fires=torch.zeros(
                (self.parallel_envs, self.max_y, self.max_x),
                dtype=torch.int32,
                device=self.device,
            ),
            intensity=torch.zeros(
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

        self.capacity_transition = CapacityTransition(
            agent_shape=(self.parallel_envs, self.num_agents),
            stochastic_switch=False,
            tank_switch_probability=0.5,
            possible_capacities=torch.tensor([10, 20, 30], dtype=torch.float32),
            capacity_probabilities=torch.tensor([0.2, 0.5, 0.3], dtype=torch.float32),
        ).to(self.device)

        self.randomness_source = torch.tensor(
            [
                [[0.1, 0.6, 0.1, 0.6], [0.1, 0.6, 0.1, 0.6]],
                [[0.1, 0.6, 0.3, 0.8], [0.3, 0.8, 0.2, 0.7]],
            ],
            device=self.device,
        )
        self.targets = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.bool, device=self.device)

    def test_stochastic_tank_switch(self) -> None:
        self.capacity_transition.stochastic_switch.fill_(True)

        result_state = self.capacity_transition(self.state, self.targets, self.randomness_source)
        expected_capacities = torch.tensor([[10, 1, 10, 1], [1, 1, 1, 1]], dtype=torch.float32, device=self.device)
        expected_suppressants = torch.tensor([[10, 1, 10, 1], [1, 1, 1, 1]], dtype=torch.float32, device=self.device)

        self.assertTrue(
            torch.allclose(result_state.capacity, expected_capacities), f"""
            \rCapacity should match expected
                \rExpected:\n{expected_capacities}
                \rResult:\n{result_state.capacity}""")
        self.assertTrue(
            torch.allclose(result_state.suppressants, expected_suppressants), f"""
            \rSuppressants should match expected
                \rExpected:\n{expected_suppressants}
                \rResult:\n{result_state.suppressants}""")

    def test_deterministic_tank_switch(self) -> None:
        self.capacity_transition.stochastic_switch.fill_(False)

        result_state = self.capacity_transition(self.state, self.targets, self.randomness_source)
        expected_capacities = torch.tensor([[10, 1, 10, 1], [1, 20, 1, 20]], dtype=torch.float32, device=self.device)
        expected_suppressants = torch.tensor([[10, 1, 10, 1], [1, 20, 1, 20]], dtype=torch.float32, device=self.device)

        self.assertTrue(
            torch.allclose(result_state.capacity, expected_capacities), f"""
                \rCapacity should match expected
                    \rExpected:\n{expected_capacities}
                    \rResult:\n{result_state.capacity}""")
        self.assertTrue(
            torch.allclose(result_state.suppressants, expected_suppressants), f"""
                \rSuppressants should match expected
                    \rExpected:\n{expected_suppressants}
                    \rResult:\n{result_state.suppressants}""")

    def test_no_targets(self) -> None:
        targets = torch.zeros((self.parallel_envs, self.num_agents), dtype=torch.bool, device=self.device)

        state_before = self.state.clone()
        state_after = self.capacity_transition(self.state, targets, self.randomness_source)

        for key in state_before.__annotations__:
            self.assertTrue(
                torch.allclose(getattr(state_before, key), getattr(state_after, key)), f"""
                \rResult should be the same when no targets
                    \rExpected:\n{getattr(state_before, key)}
                    \rResult:\n{getattr(state_after, key)}""")

    def test_bonuses_from_refill_are_maintained_after_suppressant_increase(self) -> None:
        self.state.suppressants += 0.5
        targets = torch.ones((self.parallel_envs, self.num_agents), dtype=torch.bool, device=self.device)

        state_before = self.state.clone()
        new_state = self.capacity_transition(self.state, targets, self.randomness_source)

        expected_bonuses = state_before.suppressants - state_before.capacity
        received_bonuses = new_state.suppressants - new_state.capacity

        self.assertTrue(
            torch.allclose(expected_bonuses, received_bonuses), f"""
            \rBonuses should be after capacity shift
                \rExpected:\n{expected_bonuses}
                \rResult:\n{received_bonuses}""")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cpu_gpu_compatibility(self) -> None:
        targets = torch.ones((self.parallel_envs, self.num_agents), dtype=torch.bool)

        transition_cpu = deepcopy(self.capacity_transition).to('cpu')
        transition_gpu = deepcopy(self.capacity_transition).to('cuda')

        cpu_result = transition_cpu(self.state.clone().to('cpu'), targets.cpu(), self.randomness_source.cpu())
        gpu_result = transition_gpu(self.state.clone().to('cuda'), targets.cuda(), self.randomness_source.cuda())

        for key in cpu_result.__annotations__:
            self.assertTrue(
                torch.allclose(getattr(cpu_result, key),
                               getattr(gpu_result, key).cpu()), f"""
                \rResult should be the same on CPU and GPU
                    \rCPU:\n{getattr(cpu_result, key)}
                    \rGPU:\n{getattr(gpu_result, key)}""")

    def test_reset_buffers(self) -> None:
        self.capacity_transition.tank_switches[:, :] = True
        self.capacity_transition._reset_buffers()

        self.assertTrue(torch.all(~self.capacity_transition.tank_switches), 'Tank switches should be reset')


if __name__ == '__main__':
    unittest.main()
