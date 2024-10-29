import unittest

from free_range_zoo.envs.cybersecurity.env.structures.state import CybersecurityState


class TestCybersecurityState(unittest.TestCase):

    def test_state_includes_to(self) -> None:
        self.assertTrue(hasattr(CybersecurityState, 'to'), 'CybersecurityState does not include to method')
        self.assertTrue(callable(CybersecurityState.to), 'CybersecurityState to method is not callable')

    def test_state_includes_save_initial(self) -> None:
        self.assertTrue(hasattr(CybersecurityState, 'save_initial'), 'CybersecurityState does not include save_initial method')
        self.assertTrue(callable(CybersecurityState.save_initial), 'CybersecurityState save_initial method is not callable')

    def test_state_includes_restore_initial(self) -> None:
        self.assertTrue(hasattr(CybersecurityState, 'restore_initial'),
                        'CybersecurityState does not include restore_initial method')
        self.assertTrue(callable(CybersecurityState.restore_initial), 'CybersecurityState restore_initial method is not callable')

    def test_state_includes_save_checkpoint(self) -> None:
        self.assertTrue(hasattr(CybersecurityState, 'save_checkpoint'),
                        'CybersecurityState does not include save_checkpoint method')
        self.assertTrue(callable(CybersecurityState.save_checkpoint), 'CybersecurityState save_checkpoint method is not callable')

    def test_state_includes_restore_from_checkpoint(self) -> None:
        self.assertTrue(hasattr(CybersecurityState, 'restore_from_checkpoint'),
                        'CybersecurityState does not include restore_from_checkpoint method')
        self.assertTrue(callable(CybersecurityState.restore_from_checkpoint),
                        'CybersecurityState restore_from_checkpoint method is not callable')

    def test_state_includes_clone(self) -> None:
        self.assertTrue(hasattr(CybersecurityState, 'clone'), 'CybersecurityState does not include clone method')
        self.assertTrue(callable(CybersecurityState.clone), 'CybersecurityState clone method is not callable')


if __name__ == '__main__':
    unittest.main()
