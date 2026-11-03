"""Tests the indexing for v0 / v1 compatibility in rideshare."""
import unittest

import torch

from free_range_zoo.envs.rideshare.baselines import (
    FirstInFirstOutTfocusBaseline,
    FirstInFirstOutTglobalBaseline,
    GreedyTaskFocus,
    GreedyTaskGlobal,
)
from free_range_zoo.envs.rideshare.env.v0.structures.configuration import AgentConfiguration


class TestRideshareBaselineObservationIndexing(unittest.TestCase):

    def setUp(self) -> None:
        self.parallel_envs = 2
        self.agent_configuration = AgentConfiguration(
            start_positions=torch.tensor([[0, 0], [0, 0]], dtype=torch.int32),
            pool_limit=2,
            use_diagonal_travel=False,
            use_fast_travel=True,
        )

        self.agent_mapping = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        self.self_observation = torch.tensor(
            [
                [0, 0, 0, 1],
                [9, 9, 0, 1],
            ],
            dtype=torch.float32,
        )

    def _build_baselines(self):
        return {
            "fifo_focus":
            FirstInFirstOutTfocusBaseline(
                "driver_1",
                parallel_envs=self.parallel_envs,
                seed=0,
            ),
            "fifo_global":
            FirstInFirstOutTglobalBaseline(
                "driver_1",
                parallel_envs=self.parallel_envs,
                seed=0,
            ),
            "greedy_focus":
            GreedyTaskFocus(
                "driver_1",
                parallel_envs=self.parallel_envs,
                agent_configuration=self.agent_configuration,
                seed=0,
            ),
            "greedy_global":
            GreedyTaskGlobal(
                "driver_1",
                parallel_envs=self.parallel_envs,
                agent_configuration=self.agent_configuration,
                seed=0,
            ),
        }

    def _assert_baseline_actions(self, task_observation):
        observation = {
            "self": self.self_observation,
            "tasks": task_observation,
        }
        wrapped_observation = (
            observation,
            {
                "agent_action_mapping": self.agent_mapping
            },
        )

        for baseline_name, baseline in self._build_baselines().items():
            baseline.observe(wrapped_observation)
            actions = baseline.act(None)

            self.assertEqual(
                actions[:, 0].tolist(),
                [0, 1],
                msg=f"{baseline_name} selected unexpected task index order",
            )
            self.assertEqual(
                actions[:, 1].tolist(),
                [1, 1],
                msg=f"{baseline_name} treated accepted task as wrong action type",
            )

    def test_baselines_are_backward_compatible_with_v0_layout(self):
        batch_zero_tasks = torch.tensor(
            [
                [0, 1, 0, 2, 1, -100, 9, 1],
                [8, 8, 9, 9, -100, -100, 3, 5],
            ],
            dtype=torch.float32,
        )
        batch_one_tasks = torch.tensor(
            [
                [0, 0, 1, 1, -100, -100, 8, 5],
                [9, 8, 9, 7, 1, -100, 4, 1],
            ],
            dtype=torch.float32,
        )

        v0_style_tasks = torch.nested.as_nested_tensor(
            [batch_zero_tasks, batch_one_tasks],
            layout=torch.jagged,
        )

        self._assert_baseline_actions(v0_style_tasks)


if __name__ == "__main__":
    unittest.main()
