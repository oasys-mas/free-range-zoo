import unittest
import shutil
import os
import torch
import pandas as pd

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
            log_directory="unittest_logs",
        )

    def test_log_created(self) -> None:
        observations, infos = self.env.reset()
        actions = {agent: torch.tensor(self.env.action_space(agent).sample_nested()) for agent in observations.keys()}

        self.env.step(actions)

        # Confirm batch specific log files are created
        for i in range(2):
            path = os.path.join("unittest_logs", f"{i}.csv")
            self.assertTrue(os.path.exists(path), "The log files were not properly created")

        self.env.reset()

        for _ in range(2):
            actions = {agent: torch.tensor(self.env.action_space(agent).sample_nested()) for agent in observations.keys()}
            self.env.step(actions)

        # Confirm that reset logging is working
        for i in range(self.env.aec_env.parallel_envs):
            df = pd.read_csv(os.path.join('unittest_logs', f'{i}.csv'))
            self.assertEqual(
                len(df[(df.step == -1)]), 2, f'''Environment logs must include a row indicating a reset
                    \rExpected:\n\t{2}
                    \rActual:\n\t{len(df[(df.step == -1)])}''')
            self.assertEqual(
                len(df) - len(df[(df.step == -1)]), 3, f'''Environment logs must include a row indicating actions that were taken
                    \rExpected:\n\t{3}
                    \rActual:\n\t{len(df) - len(df[(df.step == -1)])}''')

    def tearDown(self) -> None:
        if os.path.exists('unittest_logs'):
            shutil.rmtree('unittest_logs')


if __name__ == '__main__':
    unittest.main()
