import unittest

from free_range_zoo.utils.state import State


class TestState(unittest.TestCase):

    def test_state_includes_to(self) -> None:
        self.assertTrue(hasattr(State, 'to'), 'State does not include to method')
        self.assertTrue(callable(State.to), 'State to method is not callable')

    def test_state_includes_save_initial(self) -> None:
        self.assertTrue(hasattr(State, 'save_initial'), 'State does not include save_initial method')
        self.assertTrue(callable(State.save_initial), 'State save_initial method is not callable')

    def test_state_includes_restore_initial(self) -> None:
        self.assertTrue(hasattr(State, 'restore_initial'), 'State does not include restore_initial method')
        self.assertTrue(callable(State.restore_initial), 'State restore_initial method is not callable')

    def test_state_includes_save_checkpoint(self) -> None:
        self.assertTrue(hasattr(State, 'save_checkpoint'), 'State does not include save_checkpoint method')
        self.assertTrue(callable(State.save_checkpoint), 'State save_checkpoint method is not callable')

    def test_state_includes_restore_from_checkpoint(self) -> None:
        self.assertTrue(hasattr(State, 'restore_from_checkpoint'), 'State does not include restore_from_checkpoint method')
        self.assertTrue(callable(State.restore_from_checkpoint), 'State restore_from_checkpoint method is not callable')

    def test_state_includes_clone(self) -> None:
        self.assertTrue(hasattr(State, 'clone'), 'State does not include clone method')
        self.assertTrue(callable(State.clone), 'State clone method is not callable')


if __name__ == '__main__':
    unittest.main()
