import unittest
from unittest.mock import MagicMock, call

import torch

from free_range_zoo.utils.conversions import batched_aec_to_batched_parallel_wrapper


class TestReset(unittest.TestCase):

    def setUp(self) -> None:
        self.env = MagicMock()
        self.wrapped_env = batched_aec_to_batched_parallel_wrapper(self.env)

        self.wrapped_env.reset()

    def test_reset_calls_inner_env_reset(self) -> None:
        self.env.reset.assert_called_once()

    def test_reset_initializes_agents_from_environment(self) -> None:
        self.assertEqual(self.env.agents, self.wrapped_env.agents)

    def test_reset_initializes_observations_from_environment(self) -> None:
        self.env.assert_has_calls([call(agent) for agent in self.env.agents])

    def test_reset_initialzies_infos_from_environment(self) -> None:
        _, infos = self.wrapped_env.reset()
        self.assertEqual(infos, self.env.infos)


class TestResetBatches(unittest.TestCase):

    def setUp(self) -> None:
        self.env = MagicMock()
        self.wrapped_env = batched_aec_to_batched_parallel_wrapper(self.env)

    def test_wrapped_environment_passes_down_reset_batches(self) -> None:
        self.wrapped_env.reset_batches()
        self.env.reset_batches.assert_called_once()


class TestStep(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.env = MagicMock()
        self.env.device = self.device
        self.env.parallel_envs = 2
        self.env.agents = ['agent1', 'agent2']

        self.wrapped_env = batched_aec_to_batched_parallel_wrapper(self.env)

    def test_step_updates_agents(self) -> None:
        self.wrapped_env.step({'agent1': 'action1', 'agent2': 'action2'})
        self.assertEqual(self.env.agents, self.wrapped_env.agents)

    def test_step_is_called_with_each_agent(self) -> None:
        expected = {agent: 'action' for agent in self.env.agents}

        self.wrapped_env.step(expected)
        self.env.step.assert_has_calls([call(expected[agent]) for agent in self.env.agents])

    def test_returns(self) -> None:
        o, r, te, tr, i = self.wrapped_env.step({'agent1': 'action1', 'agent2': 'action2'})

        self.env.observe.assert_has_calls([call(agent) for agent in self.env.agents])

        self.assertTrue(
            all([agent in r for agent in self.env.agents]), f"""
            \rExpected all agents to have rewards
                \rExpected: {self.env.agents}
                \rActual: {r}""")

        self.assertEqual(
            te, self.env.terminations, f"""
            \rTerminations do not match
                \rExpected:\n{self.env.terminations}
                \rActual:\n{te}""")
        self.assertEqual(
            tr, self.env.truncations, f"""
            \rTruncations do not match
                \rExpected:\n{self.env.truncations}
                \rActual:\n{tr}""")
        self.assertEqual(
            i, self.env.infos, f"""
            \rInfos do not match
                \rExpected:\n{self.env.infos}
                \rActual:\n{i}""")


class TestCalcuatedProperties(unittest.TestCase):

    def setUp(self) -> None:
        self.env = MagicMock()
        self.wrapped_env = batched_aec_to_batched_parallel_wrapper(self.env)

    def test_finished_is_accessed_from_batched(self) -> None:
        result = self.wrapped_env.finished
        self.assertEqual(self.env.finished, result)
