import unittest
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import torch

from free_range_zoo.utils.env import BatchedAECEnv
from free_range_zoo.utils.random_generator import RandomGenerator


class MockedBatchedAECEnv(BatchedAECEnv):

    def step_environment(self) -> None:
        pass

    def update_actions(self) -> None:
        pass

    def update_observations(self) -> None:
        pass

    def action_space(self) -> None:
        pass

    def observation_space(self) -> None:
        pass


class TestInitialization(unittest.TestCase):

    def setUp(self) -> None:
        self.mock_config = MagicMock()
        self.mock_config.to.return_value = self.mock_config
        self.mock_config.value = 123

        self.env = MockedBatchedAECEnv(configuration=self.mock_config)

    def test_initialization_with_no_config(self) -> None:
        env = MockedBatchedAECEnv()

        self.assertFalse(hasattr(env, 'config'), 'Config should not be set if not provided')

    def test_initialization_with_config(self) -> None:
        env = MockedBatchedAECEnv(configuration=self.mock_config)

        self.assertEqual(env.config, self.mock_config, 'Config should be set to the provided value')


class TestReset(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.parallel_envs = 2
        self.env = MockedBatchedAECEnv(parallel_envs=self.parallel_envs, device=self.device, render_mode=None)

        self.env.possible_agents = ['agent_1', 'agent_2']

    def test_reset_seeds_environment(self) -> None:
        with patch.object(RandomGenerator, 'seed') as mock_reset:
            self.env.reset()

        mock_reset.assert_called_once()

    def test_initializes_aec_attributes(self) -> None:
        self.env.reset()

        expected_rewards = {'agent_1': torch.zeros(2, device=self.device), 'agent_2': torch.zeros(2, device=self.device)}

        self.assertEqual(self.env.agents, self.env.possible_agents, 'Agents should be set to possible_agents')
        for agent in self.env.agents:
            self.assertTrue(
                torch.equal(expected_rewards[agent], self.env.rewards[agent]), f"""
                \rRewards for {agent} should be initialized to zeros
                    \rExpected:\n{expected_rewards[agent]}
                    \rActual:\n{self.env.rewards[agent]}""")

            self.assertTrue(
                torch.equal(expected_rewards[agent], self.env._cumulative_rewards[agent]), f"""
                \rCumulative rewards for {agent} should be initialized to zeros
                    \rExpected:\n{expected_rewards[agent]}
                    \rActual:\n{self.env._cumulative_rewards[agent]}""")

            self.assertTrue(
                torch.equal(torch.zeros(self.parallel_envs, device=self.device), self.env.terminations[agent]), f"""
                \rTerminations for {agent} should be initialized to False
                    \rExpected:\n{expected_rewards[agent]}
                    \rActual:\n{self.env.terminations[agent]}""")

            self.assertTrue(
                torch.equal(torch.zeros(self.parallel_envs, device=self.device), self.env.truncations[agent]), f"""
                \rTruncations for {agent} should be initialized to False
                    \rExpected:\n{expected_rewards[agent]}
                    \rActual:\n{self.env.truncations[agent]}""")

            self.assertTrue({} == self.env.infos[agent]
                            or "task-action-index-map" in self.env.infos[agent].keys() and len(self.env.infos[agent]), f"""
                \rInfos for {agent} should be initialized to an empty dictionary or a dict with a empty task-action-indices key
                    \rActual:\n{self.env.infos[agent]}""")

        self.assertTrue(
            torch.equal(torch.zeros(self.parallel_envs, device=self.device), self.env.num_moves), f"""
            \rNum moves should be initialized to zeros
                \rActual:\n{self.env.num_moves}""")

    def test_initializes_agent_selector(self) -> None:
        self.env.reset()

        self.assertEqual(
            self.env.agent_selector.agent_order, self.env.agents, f"""
            \rAgent selector should be initialized with the agents
                \rExpected:\n{self.env.agents}
                \rActual:\n{self.env.agent_selector.agent_order}""")

        self.assertEqual(
            self.env.agent_selection, self.env.agent_selector.reset(), f"""
            \rAgent selection should be reset to the initial
                \rExpected:\n{self.env.agent_selector.reset()}
                \rActual:\n{self.env.agent_selection}""")

    def test_task_counters(self) -> None:
        self.env.reset()

        self.assertEqual(
            self.env.environment_task_count.shape, (self.parallel_envs, ), f"""
            \rEnvironment task count should be initialized to empty of shape (parallel_envs)
                \rActual:\n{self.env.environment_task_count.shape}""")

        self.assertEqual(
            self.env.agent_task_count.shape, (len(self.env.agents), self.parallel_envs), f"""
            \rAgent task count should be initialized to empty of shape (agents, self.parallel_envs)
                \rActual:\n{self.env.agent_task_count.shape}""")


class TestResetBatches(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.env = MockedBatchedAECEnv(
            parallel_envs=4,
            device=self.device,
            render_mode=None,
        )
        self.env.possible_agents = ['agent_1', 'agent_2']

        self.env.reset()
        for agent in self.env.possible_agents:
            self.env.rewards[agent].fill_(1)
            self.env._cumulative_rewards[agent].fill_(1)
            self.env.terminations[agent].fill_(True)
            self.env.truncations[agent].fill_(True)
            self.env.num_moves.fill_(1)

    def test_environment_properties_are_properly_reset(self) -> None:
        self.env.reset_batches(batch_indices=torch.tensor([1, 3], dtype=torch.int32),
                               seed=torch.tensor([12345, 67890], dtype=torch.int32, device=self.device))

        expected = {
            'rewards': torch.tensor([1, 0, 1, 0], dtype=torch.float32, device=self.device),
            'cumulative_rewards': torch.tensor([1, 0, 1, 0], dtype=torch.float32, device=self.device),
            'terminations': torch.tensor([1, 0, 1, 0], dtype=torch.bool, device=self.device),
            'truncations': torch.tensor([1, 0, 1, 0], dtype=torch.bool, device=self.device),
            'num_moves': torch.tensor([1, 0, 1, 0], dtype=torch.int32, device=self.device),
        }

        for agent in self.env.possible_agents:
            self.assertTrue(
                torch.equal(expected['rewards'], self.env.rewards[agent]), f"""
                \rRewards for {agent} should be reset to zeros
                    \rActual:\n{self.env.rewards[agent]}""")
            self.assertTrue(
                torch.equal(expected['cumulative_rewards'], self.env._cumulative_rewards[agent]), f"""
                \rCumulative rewards for {agent} should be reset to zeros
                    \rActual:\n{self.env._cumulative_rewards[agent]}""")
            self.assertTrue(
                torch.equal(expected['terminations'], self.env.terminations[agent]), f"""
                \rTerminations for {agent} should be reset to False
                    \rActual:\n{self.env.terminations[agent]}""")
            self.assertTrue(
                torch.equal(expected['truncations'], self.env.truncations[agent]), f"""
                \rTruncations for {agent} should be reset to False
                    \rActual:\n{self.env.truncations[agent]}""")
            self.assertTrue(
                torch.equal(expected['num_moves'], self.env.num_moves), f"""
                \rNum moves should be reset to zeros
                    \rActual:\n{self.env.num_moves}""")


class TestStep(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.env = MockedBatchedAECEnv(parallel_envs=3, device=self.device, render_mode=None)
        self.env.possible_agents = ['agent_1', 'agent_2']

        self.env.reset()
        for agent in self.env.possible_agents:
            self.env.rewards[agent].fill_(1)
            self.env.terminations[agent].fill_(False)
            self.env.truncations[agent].fill_(False)
            self.env.num_moves.fill_(1)

        while not self.env.agent_selector.is_last():
            self.env.agent_selection = self.env.agent_selector.next()

        self.env.step_environment = Mock(
            return_value=[{
                agent: torch.tensor([1, 2, 3], dtype=torch.int32, device=self.device)
                for agent in self.env.possible_agents
            }, {
                agent: torch.ones(self.env.parallel_envs, dtype=torch.bool, device=self.device)
                for agent in self.env.possible_agents
            }, {
                agent: torch.ones(self.env.parallel_envs, dtype=torch.bool, device=self.device)
            }])

    def test_step(self) -> None:
        expected = {
            'num_moves': torch.tensor([2, 2, 2], dtype=torch.int32, device=self.device),
            'rewards': torch.tensor([2, 2, 2], dtype=torch.int32, device=self.device),
            'terminations': torch.ones(self.env.parallel_envs, dtype=torch.bool, device=self.device),
            'infos': {
                'thing': True
            },
        }

        self.env.step_environment = Mock(return_value=[
            {
                agent: expected['rewards']
                for agent in self.env.possible_agents
            },
            {
                agent: expected['terminations']
                for agent in self.env.possible_agents
            },
            {
                agent: expected['infos']
                for agent in self.env.possible_agents
            },
        ])

        actions = {agent: torch.tensor([[0, 1], [2, 3], [4, 5]], device=self.device) for agent in self.env.agents}
        self.env.step(actions)

        self.env.step_environment.assert_called_once()

        self.assertTrue(
            torch.equal(self.env.num_moves, expected['num_moves']), f"""
            \rNum moves should be updated
                \rExpected:\n{expected['num_moves']}
                \rActual:\n{self.env.num_moves}""")

        for agent in self.env.agents:
            self.assertTrue(
                torch.equal(self.env._cumulative_rewards[agent], expected['rewards']), f"""
                \rRewards for {agent} should be updated
                    \rExpected:\n{expected['rewards']}
                    \rActual:\n{self.env.rewards[agent]}""")
            self.assertTrue(
                torch.equal(self.env.terminations[agent], expected['terminations']), f"""
                \rTerminations for {agent} should be updated
                    \rExpected:\n{expected['terminations']}
                    \rActual:\n{self.env.terminations[agent]}""")
            self.assertEqual(
                self.env.infos[agent], expected['infos'], f"""
                \rInfos for {agent} should be updated
                    \rExpected:\n{expected['infos']}
                    \rActual:\n{self.env.infos[agent]}""")
            self.assertTrue(
                torch.equal(expected['rewards'], self.env.rewards[agent]), f"""
                \rRewards should not be cleared after updating
                    \rActual:\n{self.env.rewards}""")

        self.assertEqual(
            self.env.agent_selection, self.env.agent_selector.reset(), f"""
            \rAgent selection should be reset to the initial
                \rExpected:\n{self.env.agent_selector.reset()}
                \rActual:\n{self.env.agent_selection}""")


class TestRewardAccumulation(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.env = MockedBatchedAECEnv()
        self.env.parallel_envs = 3
        self.env.device = self.device
        self.env.agents = ['agent_1', 'agent_2']
        self.env.rewards = {
            'agent_1': torch.tensor([1, 2, 3]),
            'agent_2': torch.tensor([4, 5, 6]),
        }
        self.env._cumulative_rewards = {
            'agent_1': torch.zeros(3) + 20,
            'agent_2': torch.zeros(3) + 20,
        }

    def test_reward_accumulation(self) -> None:
        expected = {
            'agent_1': torch.tensor([21, 22, 23]),
            'agent_2': torch.tensor([24, 25, 26]),
        }

        self.env._accumulate_rewards()
        for agent in self.env.agents:
            actual = self.env._cumulative_rewards[agent]
            self.assertTrue(
                torch.equal(actual, expected[agent]), f"""
            \rRewards for {agent} are not as expected
                \rExpected:\n{expected}
                \rActual:\n{actual}""")


class TestRewardClearing(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.env = MockedBatchedAECEnv()
        self.env.parallel_envs = 3
        self.env.device = self.device
        self.env.agents = ['agent_1', 'agent_2']
        self.env.rewards = {
            'agent_1': torch.tensor([1, 2, 3]),
            'agent_2': torch.tensor([4, 5, 6]),
        }

    def test_clear_rewards(self) -> None:
        self.env._clear_rewards()
        for agent in self.env.agents:
            actual = self.env.rewards[agent]
            expected = torch.zeros_like(actual)
            self.assertTrue(
                torch.equal(actual, expected), f"""
            \rRewards for {agent} are not as expected
                \rExpected:\n{expected}
                \rActual:\n{actual}""")


class TestObserve(unittest.TestCase):

    def setUp(self) -> None:
        self.env = MockedBatchedAECEnv()
        self.env.agents = ['agent_1', 'agent_2']
        self.env.observations = {
            'agent_1': torch.tensor([1, 2, 3]),
            'agent_2': torch.tensor([4, 5, 6]),
        }

    def test_observe_returns_correct_observation(self) -> None:
        expected = {
            'agent_1': torch.tensor([1, 2, 3]),
            'agent_2': torch.tensor([4, 5, 6]),
        }

        for agent in self.env.agents:
            actual = self.env.observe(agent)
            self.assertTrue(
                torch.equal(actual, expected[agent]), f"""
                \rObservation for {agent} is not as expected
                    \rExpected:\n{expected}
                    \rActual:\n{actual}""")

    def test_observe_agent_not_found(self) -> None:
        with self.assertRaises(KeyError):
            self.env.observe('agent_3')


class TestState(unittest.TestCase):

    def setUp(self) -> None:
        self.env = MockedBatchedAECEnv()
        self.env._state = torch.tensor([1, 2, 3])  # Placeholder value for actual state

    def test_state_returns_correct_state(self) -> None:
        expected = torch.tensor([1, 2, 3])
        actual = self.env.state()

        self.assertTrue(
            torch.equal(actual, expected), f"""
            \rState is not as expected
                \rExpected:\n{expected}
                \rActual:\n{actual}""")


class TestAbstractMethods(unittest.TestCase):

    def setUp(self) -> None:
        self.env = BatchedAECEnv

    def _test_abstract_method(self, func_name: str) -> None:
        self.assertTrue(hasattr(self.env, func_name), f'{self.env.__name__} does not include {func_name} method')
        self.assertTrue(callable(getattr(self.env, func_name)), f'{self.env.__name__} {func_name} method is not callable')
        self.assertTrue(
            getattr(self.env, func_name).__isabstractmethod__, f'{self.env.__name__} {func_name} method should be abstract')

    def test_step_environment(self) -> None:
        self._test_abstract_method('step_environment')

    def test_update_actions(self) -> None:
        self._test_abstract_method('update_actions')

    def test_update_observations(self) -> None:
        self._test_abstract_method('update_observations')

    def test_action_space(self) -> None:
        self._test_abstract_method('action_space')

    def test_observation_space(self) -> None:
        self._test_abstract_method('observation_space')


class TestCalculatedProperties(unittest.TestCase):

    def setUp(self) -> None:
        self.env = MockedBatchedAECEnv()
        self.env.agents = ['agent_1', 'agent_2']
        self.env.terminations = {
            'agent_1': torch.tensor([False, True, True], dtype=torch.bool),
            'agent_2': torch.tensor([False, True, False], dtype=torch.bool),
        }
        self.env.truncations = {
            'agent_1': torch.tensor([False, True, True], dtype=torch.bool),
            'agent_2': torch.tensor([False, True, False], dtype=torch.bool),
        }

    def _test_calculated_property(self, prop_name: str) -> None:
        self.assertTrue(hasattr(self.env, prop_name), f'{self.env.__class__.__name__} does not include {prop_name} property')

    def test_finished(self) -> None:
        self._test_calculated_property('finished')

        expected = torch.tensor([False, True, False], dtype=torch.bool)
        self.assertTrue(
            torch.equal(self.env.finished, expected), f"""
            \rFinished property is not as expected
                \rExpected:\n{expected}
                \rActual:\n{self.env.finished}""")

    def test_terminated(self) -> None:
        self._test_calculated_property('terminated')

        expected = torch.tensor([False, True, False], dtype=torch.bool)
        self.assertTrue(
            torch.equal(self.env.terminated, expected), f"""
            \rTerminated property is not as expected
                \rExpected:\n{expected}
                \rActual:\n{self.env.terminated}""")

    def test_truncated(self) -> None:
        self._test_calculated_property('truncated')

        expected = torch.tensor([False, True, False], dtype=torch.bool)
        self.assertTrue(
            torch.equal(self.env.truncated, expected), f"""
            \rTruncated property is not as expected
                \rExpected:\n{expected}
                \rActual:\n{self.env.truncated}""")


if __name__ == '__main__':
    unittest.main()
