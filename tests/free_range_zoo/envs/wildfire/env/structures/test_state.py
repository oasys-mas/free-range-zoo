import unittest

from free_range_zoo.envs.wildfire.env.structures.state import WildfireState


class TestWildfireState(unittest.TestCase):

    def test_state_includes_to(self) -> None:
        self.assertTrue(hasattr(WildfireState, 'to'), 'WildfireState does not include to method')
        self.assertTrue(callable(WildfireState.to), 'WildfireState to method is not callable')

    def test_state_includes_save_initial(self) -> None:
        self.assertTrue(hasattr(WildfireState, 'save_initial'), 'WildfireState does not include save_initial method')
        self.assertTrue(callable(WildfireState.save_initial), 'WildfireState save_initial method is not callable')

    def test_state_includes_restore_initial(self) -> None:
        self.assertTrue(hasattr(WildfireState, 'restore_initial'), 'WildfireState does not include restore_initial method')
        self.assertTrue(callable(WildfireState.restore_initial), 'WildfireState restore_initial method is not callable')

    def test_state_includes_save_checkpoint(self) -> None:
        self.assertTrue(hasattr(WildfireState, 'save_checkpoint'), 'WildfireState does not include save_checkpoint method')
        self.assertTrue(callable(WildfireState.save_checkpoint), 'WildfireState save_checkpoint method is not callable')

    def test_state_includes_restore_from_checkpoint(self) -> None:
        self.assertTrue(hasattr(WildfireState, 'restore_from_checkpoint'),
                        'WildfireState does not include restore_from_checkpoint method')
        self.assertTrue(callable(WildfireState.restore_from_checkpoint),
                        'WildfireState restore_from_checkpoint method is not callable')

    def test_state_includes_clone(self) -> None:
        self.assertTrue(hasattr(WildfireState, 'clone'), 'WildfireState does not include clone method')
        self.assertTrue(callable(WildfireState.clone), 'WildfireState clone method is not callable')


if __name__ == '__main__':
    unittest.main()
