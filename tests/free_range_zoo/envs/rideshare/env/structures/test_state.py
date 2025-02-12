import unittest

from free_range_zoo.envs.rideshare.env.structures.state import RideshareState


class TestRideshareState(unittest.TestCase):

    def test_state_includes_to(self) -> None:
        self.assertTrue(hasattr(RideshareState, 'to'), 'RideshareState does not include to method')
        self.assertTrue(callable(RideshareState.to), 'RideshareState to method is not callable')

    def test_state_includes_save_initial(self) -> None:
        self.assertTrue(hasattr(RideshareState, 'save_initial'), 'RideshareState does not include save_initial method')
        self.assertTrue(callable(RideshareState.save_initial), 'RideshareState save_initial method is not callable')

    def test_state_includes_restore_initial(self) -> None:
        self.assertTrue(hasattr(RideshareState, 'restore_initial'), 'RideshareState does not include restore_initial method')
        self.assertTrue(callable(RideshareState.restore_initial), 'RideshareState restore_initial method is not callable')

    def test_state_includes_save_checkpoint(self) -> None:
        self.assertTrue(hasattr(RideshareState, 'save_checkpoint'), 'RideshareState does not include save_checkpoint method')
        self.assertTrue(callable(RideshareState.save_checkpoint), 'RideshareState save_checkpoint method is not callable')

    def test_state_includes_restore_from_checkpoint(self) -> None:
        self.assertTrue(hasattr(RideshareState, 'restore_from_checkpoint'),
                        'RideshareState does not include restore_from_checkpoint method')
        self.assertTrue(callable(RideshareState.restore_from_checkpoint),
                        'RideshareState restore_from_checkpoint method is not callable')

    def test_state_includes_clone(self) -> None:
        self.assertTrue(hasattr(RideshareState, 'clone'), 'RideshareState does not include clone method')
        self.assertTrue(callable(RideshareState.clone), 'RideshareState clone method is not callable')


if __name__ == '__main__':
    unittest.main()
