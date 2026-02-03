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


class TestConditionedOnActions(unittest.TestCase):

    def setUp(self) -> None:
        self.parallel_envs = 2
        self.num_agents = 4

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.state = WildfireState(
            fires=torch.zeros(
                (self.parallel_envs, 4, 4),
                dtype=torch.int32,
                device=self.device,
            ),
            intensity=torch.zeros(
                (self.parallel_envs, 4, 4),
                dtype=torch.int32,
                device=self.device,
            ),
            fuel=torch.zeros(
                (self.parallel_envs, 4, 4),
                dtype=torch.int32,
                device=self.device,
            ),
            agents=torch.randint(
                0,
                4,
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
            stochastic_repair=True,
            repair_probability=0.5,
            stochastic_degrade=True,
            degrade_probability=0.5,
            critical_error=False,
            critical_error_probability=0.2,
            condition_on_actions=True,
        )

        self.randomness_source = torch.tensor(
            [[0.1, 0.6, 0.1, 0.6], [0.1, 0.6, 0.3, 0.8]],
            dtype=torch.float32,
            device=self.device,
        )

    def test_degrade_only_on_fight(self) -> None:
        """Equipment should only degrade when agent fought."""
        # Start with pristine equipment (2)
        self.state.equipment.fill_(2)

        # fought = True for all agents
        fought = torch.ones((self.parallel_envs, self.num_agents), dtype=torch.bool, device=self.device)

        result = self.equipment_transition(self.state, self.randomness_source, fought=fought)

        # All should degrade (stochastic with prob 0.5, and randomness < 0.5 for some)
        # Agents with randomness < 0.5 should degrade: [0, 2] in batch 0, [0, 2] in batch 1
        # Starting from 2 (pristine), degrading goes to 1 (intermediate), not 0 (damaged)
        expected = torch.tensor([[1, 2, 1, 2], [1, 2, 1, 2]], dtype=torch.int32, device=self.device)

        self.assertTrue(
            torch.allclose(result.equipment, expected), f"""
            \rEquipment should degrade when fought
                \rExpected:\n{expected}
                \rResult:\n{result.equipment}""")

    def test_no_degrade_without_fight(self) -> None:
        """Equipment should not degrade when agent did not fight (noop)."""
        # Start with pristine equipment (2)
        self.state.equipment.fill_(2)

        # fought = False for all agents (all nooped)
        fought = torch.zeros((self.parallel_envs, self.num_agents), dtype=torch.bool, device=self.device)

        result = self.equipment_transition(self.state, self.randomness_source, fought=fought)

        # None should degrade since no one fought
        expected = torch.tensor([[2, 2, 2, 2], [2, 2, 2, 2]], dtype=torch.int32, device=self.device)

        self.assertTrue(
            torch.allclose(result.equipment, expected), f"""
            \rEquipment should not degrade without fight
                \rExpected:\n{expected}
                \rResult:\n{result.equipment}""")

    def test_repair_only_on_noop_at_damaged(self) -> None:
        """Equipment should only repair when agent noops at fully damaged."""
        # Start with damaged equipment (0)
        self.state.equipment.fill_(0)

        # fought = False for all agents (all nooped)
        fought = torch.zeros((self.parallel_envs, self.num_agents), dtype=torch.bool, device=self.device)

        result = self.equipment_transition(self.state, self.randomness_source, fought=fought)

        # Stochastic repair with prob 0.5
        expected = torch.tensor([[2, 0, 2, 0], [2, 0, 2, 0]], dtype=torch.int32, device=self.device)

        self.assertTrue(
            torch.allclose(result.equipment, expected), f"""
            \rEquipment should repair when nooping at damaged
                \rExpected:\n{expected}
                \rResult:\n{result.equipment}""")

    def test_no_repair_after_fight_at_damaged(self) -> None:
        """Equipment should not repair when agent fought at fully damaged."""
        # Start with damaged equipment (0)
        self.state.equipment.fill_(0)

        # fought = True for all agents (all fought, even though damaged)
        fought = torch.ones((self.parallel_envs, self.num_agents), dtype=torch.bool, device=self.device)

        result = self.equipment_transition(self.state, self.randomness_source, fought=fought)

        # None should repair since all fought
        expected = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int32, device=self.device)

        self.assertTrue(
            torch.allclose(result.equipment, expected), f"""
            \rEquipment should not repair after fight at damaged
                \rExpected:\n{expected}
                \rResult:\n{result.equipment}""")

    def test_mixed_fight_noop(self) -> None:
        """Test with mixed fight and noop agents."""
        # Start with pristine equipment (2)
        self.state.equipment.fill_(2)

        # fought = True for agents 0,2 and False for 1,3
        fought = torch.tensor([[True, False, True, False], [True, False, True, False]], dtype=torch.bool, device=self.device)

        result = self.equipment_transition(self.state, self.randomness_source, fought=fought)

        # Agents 0,2 fought and should degrade if randomness < 0.5
        # Agents 1,3 nooped and should not degrade
        # Starting from 2 (pristine), degrading goes to 1 (intermediate)
        expected = torch.tensor([[1, 2, 1, 2], [1, 2, 1, 2]], dtype=torch.int32, device=self.device)

        self.assertTrue(
            torch.allclose(result.equipment, expected), f"""
            \rEquipment should respect mixed fight/noop
                \rExpected:\n{expected}
                \rResult:\n{result.equipment}""")

    def test_intermediate_state_degrades_on_fight(self) -> None:
        """Intermediate equipment state should degrade on fight."""
        # Start with intermediate equipment (1)
        self.state.equipment.fill_(1)

        # fought = True for all agents
        fought = torch.ones((self.parallel_envs, self.num_agents), dtype=torch.bool, device=self.device)

        result = self.equipment_transition(self.state, self.randomness_source, fought=fought)

        # All should degrade (stochastic with prob 0.5)
        expected = torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=torch.int32, device=self.device)

        self.assertTrue(
            torch.allclose(result.equipment, expected), f"""
            \rIntermediate equipment should degrade on fight
                \rExpected:\n{expected}
                \rResult:\n{result.equipment}""")

    def test_intermediate_no_change_without_fight(self) -> None:
        """Intermediate equipment state should not change without fight."""
        # Start with intermediate equipment (1)
        self.state.equipment.fill_(1)

        # fought = False for all agents (all nooped)
        fought = torch.zeros((self.parallel_envs, self.num_agents), dtype=torch.bool, device=self.device)

        result = self.equipment_transition(self.state, self.randomness_source, fought=fought)

        # None should change since no one fought
        expected = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.int32, device=self.device)

        self.assertTrue(
            torch.allclose(result.equipment, expected), f"""
            \rIntermediate equipment should not change without fight
                \rExpected:\n{expected}
                \rResult:\n{result.equipment}""")

    def test_conditioned_critical_error(self) -> None:
        """Critical errors should only occur when agent fought at pristine."""
        self.equipment_transition.critical_error.fill_(True)

        # Start with pristine equipment (2)
        self.state.equipment.fill_(2)

        # fought = True for agents 0,1 and False for 2,3
        fought = torch.tensor([[True, True, False, False], [True, True, False, False]], dtype=torch.bool, device=self.device)

        result = self.equipment_transition(self.state, self.randomness_source, fought=fought)

        # Agents 0,1 fought and pristine: critical error if randomness < 0.2
        # - Agent 0: 0.1 < 0.2 -> critical error -> 0
        # - Agent 1: 0.6 >= 0.2 -> no critical error, 0.6 >= 0.5 -> no degrade -> 2
        # Agents 2,3 nooped: no critical error, no degrade -> 2
        # Batch 1: Agent 2 randomness 0.3 < 0.2? No -> 2, Agent 3 0.8 >= 0.5 -> no degrade -> 2
        expected = torch.tensor([[0, 2, 2, 2], [0, 2, 2, 2]], dtype=torch.int32, device=self.device)

        self.assertTrue(
            torch.allclose(result.equipment, expected), f"""
            \rCritical errors should only occur when fought at pristine
                \rExpected:\n{expected}
                \rResult:\n{result.equipment}""")


if __name__ == '__main__':
    unittest.main()
