import sys

sys.path.append('.')

import unittest

import torch
import cProfile

from free_range_zoo.free_range_zoo.envs import cybersecurity_v0
from tests.utils.cybersecurity_configs import non_stochastic


class TestCybersecurityEnvironmentRuntime(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device('cuda')
        self.configuration = non_stochastic()
        self.env = cybersecurity_v0.parallel_env(
            parallel_envs=100,
            max_steps=15,
            configuration=self.configuration,
            device=self.device,
        )

    def test_environment_runtime(self) -> None:
        self.env.reset()

        current_step = 1
        while not torch.all(self.env.finished):
            action = {}

            for agent in self.env.agents:
                self.env.observation_space(agent)
                actions = []
                for action_space in self.env.action_space(agent):
                    actions.append(action_space.sample())
                actions = torch.tensor(actions, device=self.device)
                action[agent] = actions

            observation, reward, term, trunc, info = self.env.step(action)
            current_step += 1


if __name__ == '__main__':
    unittest.main()
