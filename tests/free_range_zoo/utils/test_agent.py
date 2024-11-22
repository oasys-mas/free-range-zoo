import unittest

from free_range_zoo.utils.agent import Agent


class TestAgent(unittest.TestCase):

    def test_agent_includes_act(self) -> None:
        self.assertTrue(hasattr(Agent, 'act'), 'Agent does not include act method')
        self.assertTrue(callable(Agent.act), 'Agent act method is not callable')

    def test_agent_includes_observe(self) -> None:
        self.assertTrue(hasattr(Agent, 'observe'), 'Agent does not include observe method')
        self.assertTrue(callable(Agent.observe), 'Agent observe method is not callable')


if __name__ == '__main__':
    unittest.main()
