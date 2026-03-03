import sys

sys.path.append('.')

import unittest

import torch

from free_range_zoo.envs import wildfire_v0
from tests.utils.wildfire_configs import non_stochastic


class TestWildfireEnvironmentRuntime(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.configuration = non_stochastic()
        self.env = wildfire_v0.parallel_env(
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
                actions = self.env.action_space(agent).sample_nested()
                actions = torch.tensor(actions, device=self.device, dtype=torch.int32)
                action[agent] = actions

            observation, reward, term, trunc, info = self.env.step(action)
            current_step += 1


if __name__ == '__main__':
    unittest.main()
