import unittest
import warnings

import shutil
import os
import torch

from free_range_zoo.envs import wildfire_v0
from free_range_zoo.envs.wildfire.configs.aaai_2024 import aaai_2025_ol_config


class TestConfiguration(unittest.TestCase):

    def setUp(self) -> None:
        self.paper_configuration = aaai_2025_ol_config(2)

        self.env = wildfire_v0.parallel_env(
            parallel_envs=3,
            max_steps=3,
            configuration=self.paper_configuration,
            show_bad_actions=False,
            observe_other_suppressant=False,
            observe_other_power=True,
            device=torch.device('cpu'),
            log_dir="unittest_logs",
        )

    def test_log_created(self) -> None:
        if os.path.exists('unittest_logs'):
            warnings.warn("Logs directory already exists. Do not use unittest_logs for experiments.")
            shutil.rmtree('unittest_logs')

        obs, info = self.env.reset()

        actions = {agent: torch.tensor(self.env.action_space(agent).sample_nested()) for agent in obs.keys()}

        self.env.step(actions)

        self.assertTrue(os.path.exists("unittest_logs"), "Log directory not created")

        # Confirm batch specific log files are created
        for i in range(2):
            self.assertTrue(os.path.exists(os.path.join("unittest_logs", f"{i}.csv")), "Log files missing")

        self.env.reset()

        actions = {agent: torch.tensor(self.env.action_space(agent).sample_nested()) for agent in obs.keys()}

        self.env.step(actions)

        # Confirm that reset logging is working
        with open(os.path.join('unittest_logs', '0.csv'), 'r') as f:
            lines = f.readlines()[1:]

            # Silly but trying to avoid using a eval here
            reset_lines = [bool(line.split(",")[-1]) for line in lines if "True" in line]

        self.assertEqual(sum(reset_lines), 2)

    def tearDown(self) -> None:
        if os.path.exists('unittest_logs'):
            shutil.rmtree('unittest_logs')


if __name__ == '__main__':
    unittest.main()
