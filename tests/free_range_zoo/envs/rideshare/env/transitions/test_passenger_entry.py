import unittest

import torch

from free_range_zoo.envs.rideshare.env.transitions.passenger_entry import PassengerEntryTransition
from free_range_zoo.envs.rideshare.env.structures.state import RideshareState


class TestPassengerEntryTransition(unittest.TestCase):

    def setUp(self):
        self.state = RideshareState(agents=torch.tensor([[[1, 1]], [[1, 1]]]), passengers=None)

        self.transition = PassengerEntryTransition(
            torch.tensor([
                [0, -1, 1, 1, 1, 1, 1],
                [1, -1, 1, 1, 1, 1, 2],
                [2, 1, 1, 1, 1, 1, 3],
            ]),
            2,
        )

    def test_initial_passenger_entry(self) -> None:
        result = self.transition(self.state, torch.tensor([0, 0], dtype=torch.int32))

        self.assertIsNotNone(
            result.passengers, f'''
            \rPassenger entries should not be None after entry.
                \rActual: {result.passengers}''')

        self.assertEqual(result.passengers.size(0), 2, 'Passenger entries should successfully populate.')

    def test_batch_agnostic_passenger_entry(self) -> None:
        result = self.transition(self.state, torch.tensor([1, 1], dtype=torch.int32))

        self.assertEqual(result.passengers.size(0), 2, 'Passengers should successfully be introduced for all environments.')
        self.assertEqual(result.passengers[0, 0], 0, 'First entry should be in the first parallel environment.')
        self.assertEqual(result.passengers[0, 1], 1, 'Second entry should be in the second parallel environment.')

    def test_batch_specific_passenger_entry(self) -> None:
        result = self.transition(self.state, torch.tensor([2, 2], dtype=torch.int32))

        self.assertEqual(result.passengers.size(0), 1,
                         'Passengers should successfully be introduced for only target environment.')
        self.assertEqual(result.passengers[0, 0], 1, 'First entry should be in the first parallel environment.')

    def test_no_passengers_enter_if_no_schedule_allows(self) -> None:
        result = self.transition(self.state, torch.tensor([1, 3], dtype=torch.int32))

        self.assertEqual(result.passengers.size(0), 1, 'No passengers should be introduced if not defined in config.')

        self.assertTrue(
            torch.equal(result.passengers[0], torch.tensor([0, 1, 1, 1, 1, 2, 0, -1, 1, -1, -1])), f'''
            \rPassenger entry should match expected entry
                \rExpected:\n{torch.tensor([0, 1, 1, 1, 1, 2, 0, -1, 1, -1, -1])}
                \rActual:\n{result.passengers[0]}''')

    def test_passenger_entries_are_sorted_after_insertion(self) -> None:
        timestep = 0

        result = self.state
        for i in range(3):
            result = self.transition(result, torch.tensor([timestep, timestep], dtype=torch.int32))
            timestep += 1

        self.assertEqual(result.passengers.size(0), 5, 'Passengers should be correctly introduced in order')

        for index, batch in enumerate([0, 0, 1, 1, 1]):
            self.assertEqual(result.passengers[index, 0], batch, 'Batch should be in sorted orders after transition.')
