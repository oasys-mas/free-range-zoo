import unittest
from copy import deepcopy

import torch

from free_range_zoo.envs.wildfire.env.structures.state import WildfireState
from free_range_zoo.envs.wildfire.env.transitions.equipment import EquipmentTransition


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

        self.equipment_transition = EquipmentTransition(
            equipment_states=torch.tensor([0, 1, 2], dtype=torch.int32),
            stochastic_repair=False,
            repair_probability=0.5,
            stochastic_degrade=False,
            degrade_probability=0.5,
            critical_error=False,
            critical_error_probability=0.2,
        )

        self.randomness_source = torch.tensor(
            [[0.1, 0.6, 0.1, 0.6], [0.1, 0.6, 0.3, 0.8]],
            dtype=torch.float32,
            device=self.device,
        )

    def test_stochastic_repair(self) -> None:
        self.equipment_transition.stochastic_repair.fill_(True)
        self.state.equipment.fill_(0)

        result = self.equipment_transition(self.state, self.randomness_source)
        expected = torch.tensor([[2, 0, 2, 0], [2, 0, 2, 0]], dtype=torch.int32, device=self.device)

        self.assertTrue(
            torch.allclose(result.equipment, expected), f"""
            \rEquipment should match expected
                \rExpected:\n{expected}
                \rResult:\n{result.equipment}""")

    def test_deterministic_repair(self) -> None:
        self.state.equipment.fill_(0)

        result = self.equipment_transition(self.state, self.randomness_source)
        expected = torch.tensor([[2, 2, 2, 2], [2, 2, 2, 2]], dtype=torch.int32, device=self.device)

        self.assertTrue(
            torch.allclose(result.equipment, expected), f"""
            \rEquipment should match expected
                \rExpected:\n{expected}
                \rResult:\n{result.equipment}""")

    def test_stochastic_degrade(self) -> None:
        self.equipment_transition.stochastic_degrade.fill_(True)
        self.state.equipment.fill_(1)

        result = self.equipment_transition(self.state, self.randomness_source)
        expected = torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=torch.int32, device=self.device)

        self.assertTrue(
            torch.allclose(result.equipment, expected), f"""
            \rEquipment should match expected
                \rExpected:\n{expected}
                \rResult:\n{result.equipment}""")

    def test_deteministic_degrade(self) -> None:
        self.state.equipment.fill_(1)

        result = self.equipment_transition(self.state, self.randomness_source)
        expected = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int32, device=self.device)

        self.assertTrue(
            torch.allclose(result.equipment, expected), f"""
            \rEquipment should match expected
                \rExpected:\n{expected}
                \rResult:\n{result.equipment}""")

    def test_critical_error(self) -> None:
        self.equipment_transition.critical_error.fill_(True)
        self.state.equipment.fill_(2)

        result = self.equipment_transition(self.state, self.randomness_source)
        expected = torch.tensor([[0, 1, 0, 1], [0, 1, 1, 1]], dtype=torch.int32, device=self.device)

        self.assertTrue(
            torch.allclose(result.equipment, expected), f"""
            \rEquipment should match expected
                \rExpected:\n{expected}
                \rResult:\n{result.equipment}""")

    def test_combined_repair_and_degrade(self) -> None:
        self.equipment_transition.stochastic_repair.fill_(True)
        self.equipment_transition.stochastic_degrade.fill_(True)

        self.state.equipment = torch.tensor([[0, 0, 0, 0], [2, 2, 2, 2]], dtype=torch.int32, device=self.device)

        result = self.equipment_transition(self.state, self.randomness_source)
        expected = torch.tensor([[2, 0, 2, 0], [1, 2, 1, 2]], dtype=torch.int32, device=self.device)

        self.assertTrue(
            torch.allclose(result.equipment, expected), f"""
            \rEquipment should match expected
                \rExpected:\n{expected}
                \rResult:\n{result.equipment}""")

    def test_combined_degrade_repair_and_critical_error(self) -> None:
        self.equipment_transition.stochastic_repair.fill_(True)
        self.equipment_transition.stochastic_degrade.fill_(True)
        self.equipment_transition.critical_error.fill_(True)

        self.state.equipment = torch.tensor([[0, 0, 0, 1], [2, 2, 2, 2]], dtype=torch.int32, device=self.device)

        result = self.equipment_transition(self.state, self.randomness_source)
        expected = torch.tensor([[2, 0, 2, 1], [0, 2, 1, 2]], dtype=torch.int32, device=self.device)

        self.assertTrue(
            torch.allclose(result.equipment, expected), f"""
            \rEquipment should match expected
                \rExpected:\n{expected}
                \rResult:\n{result.equipment}""")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cpu_gpu_compatibility(self) -> None:
        transition_gpu = deepcopy(self.equipment_transition).cuda()
        transition_cpu = deepcopy(self.equipment_transition).cpu()

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
