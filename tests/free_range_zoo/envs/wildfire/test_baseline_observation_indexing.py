"""Tests observation indexing and action selection for the wildfire baselines."""
import unittest

import torch

from free_range_zoo.envs.wildfire.baselines import (
    NoopBaseline,
    StrongestBaseline,
    WeakestBaseline,
)

# Task (fire) feature columns.
FIRE_SIZE = 2
FIRE_INTENSITY = 3
# Self/other agent feature columns.
AGENT_SUPPRESSANT = 3


class TestWildfireBaselineObservationIndexing(unittest.TestCase):

    def setUp(self) -> None:
        self.parallel_envs = 2
        self.self_observation = torch.tensor(
            [
                [0, 0, 2, 1, 5, 3, 1],
                [0, 0, 2, 0, 5, 3, 1],
            ],
            dtype=torch.float32,
        )
        self.others_observation = torch.zeros((self.parallel_envs, 1, 7), dtype=torch.float32)
        self.agent_mapping = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)

        self.v1_tasks = torch.tensor(
            [
                [[1, 1, 2, 3], [2, 2, 4, 1]],
                [[1, 1, 2, 5], [2, 2, 4, 2]],
            ],
            dtype=torch.float32,
        )

    def _actions(self, baseline, tasks, mapping=None, self_obs=None, others=None):
        """Feed an observation to ``baseline`` and return its action tensor."""
        observation = (
            {
                "self": self.self_observation if self_obs is None else self_obs,
                "others": self.others_observation if others is None else others,
                "tasks": tasks,
            },
            {
                "agent_action_mapping": self.agent_mapping if mapping is None else mapping,
            },
        )
        baseline.observe(observation)
        return baseline.act(None)

    def _baseline(self, baseline_cls, parallel_envs=None, **kwargs):
        return baseline_cls(
            "firefighter_0",
            parallel_envs=self.parallel_envs if parallel_envs is None else parallel_envs,
            seed=0,
            **kwargs,
        )

    def test_targeting_baselines_force_noop_without_suppressant(self) -> None:
        """Every targeting baseline no-ops the action type for suppressant-depleted agents."""
        for baseline_cls in (StrongestBaseline, WeakestBaseline):
            actions = self._actions(self._baseline(baseline_cls), self.v1_tasks)
            self.assertEqual(
                actions[:, 1].tolist(),
                [0, -1],
                msg=f"{baseline_cls.__name__} mishandled suppressant masking",
            )

    def test_empty_action_mapping_produces_noop(self) -> None:
        """With no fires mapped to the agent, every slot becomes a full no-op (-1, -1)."""
        empty_mapping = torch.zeros((self.parallel_envs, 0), dtype=torch.int32)
        for baseline_cls in (StrongestBaseline, WeakestBaseline):
            actions = self._actions(self._baseline(baseline_cls), self.v1_tasks, mapping=empty_mapping)
            self.assertEqual(
                actions.tolist(),
                [[-1, -1], [-1, -1]],
                msg=f"{baseline_cls.__name__} did not no-op on an empty action mapping",
            )

    def test_targeting_baselines_pick_among_tied_candidates(self) -> None:
        """With equal candidates resolve to one of the tied indices (tie-break is random by design)."""
        tie_tasks = torch.tensor(
            [
                [[1, 1, 2, 3], [2, 2, 2, 3]],
                [[1, 1, 2, 3], [2, 2, 2, 3]],
            ],
            dtype=torch.float32,
        )

        strongest = self._actions(self._baseline(StrongestBaseline), tie_tasks)
        weakest = self._actions(self._baseline(WeakestBaseline), tie_tasks)

        tied_indices = {0, 1}
        for choice in strongest[:, 0].tolist():
            self.assertIn(choice, tied_indices, msg="StrongestBaseline picked an untied index")
        for choice in weakest[:, 0].tolist():
            self.assertIn(choice, tied_indices, msg="WeakestBaseline picked an untied index")

    def test_noop_baseline_always_noops(self) -> None:
        baseline = NoopBaseline("firefighter_0", parallel_envs=self.parallel_envs, seed=0)
        actions = baseline.act(None)

        self.assertEqual(actions[:, 0].tolist(), [0, 0])
        self.assertEqual(actions[:, 1].tolist(), [-1, -1])


if __name__ == "__main__":
    unittest.main()
