import unittest
import torch

from free_range_zoo.utils.random_generator import RandomGenerator


class TestGeneratorSeeding(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.generator = RandomGenerator(parallel_envs=3, device=self.device)
        self.generator.seed()

    def test_seeding_with_provided_seed(self) -> None:
        seed = torch.tensor([12345, 67890, 54321], dtype=torch.int32, device=self.device)
        self.generator.seed(seed=seed)

        self.assertTrue(torch.equal(self.generator.seeds, seed), "Seeds should be set to the provided values")

        for index, gen_state in enumerate(self.generator.generator_states):
            self.assertIsNotNone(gen_state, f"Generator state for environment {index} should not be None")

    def test_seeding_with_partial_seeding(self) -> None:
        seed = torch.tensor([12345, 67890], dtype=torch.int32, device=self.device)
        partial_seeding = torch.tensor([0, 2], dtype=torch.int32, device=self.device)

        self.generator.seed(seed=seed, partial_seeding=partial_seeding)

        expected_seeds = torch.tensor([12345, 0, 67890], dtype=torch.int64, device=self.device)
        expected_seeds[1] = self.generator.seeds[1]

        self.assertTrue(
            torch.equal(self.generator.seeds, expected_seeds), f"""
            \rSeeds should be set to the provided values
                \rExpected:\n{expected_seeds}
                \rActual:\n{self.generator.seeds}""")

    def test_seeding_with_random_seed(self) -> None:
        self.generator.seed()

        self.assertFalse(
            torch.equal(self.generator.seeds, torch.zeros_like(self.generator.seeds)), f"""
            \rSeeds should be set to random values
                \rActual:\n{self.generator.seeds} """)


class TestGeneratorRandomGeneration(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.generator = RandomGenerator(parallel_envs=3, device=self.device)
        self.generator.seed()

        self.parallel_envs = 3
        self.events = 5
        self.shape = (3, 4)

    def test_random_generation(self) -> None:
        random_values = self.generator.generate(self.parallel_envs, self.events, self.shape)

        expected_shape = torch.Size((self.events, self.parallel_envs, *self.shape))

        self.assertEqual(
            random_values.shape, expected_shape, f"""
            \rRandom values should the correct shape
                \rExpected:\n{expected_shape}
                \rActual:\n{random_values.shape}""")

    def test_buffered_generator_returns_correct_shape(self) -> None:
        self.generator.buffer_size = 10

        random_values = self.generator.generate(self.parallel_envs, self.events, self.shape, key="test")

        expected_shape = torch.Size((self.events, self.parallel_envs, *self.shape))

        self.assertEqual(
            random_values.shape, expected_shape, f"""
            \rRandom values should the correct shape
                \rExpected:\n{expected_shape}
                \rActual:\n{random_values.shape}""")

    def test_buffered_generator_creates_buffer(self) -> None:
        self.generator.buffer_size = 10
        self.generator.generate(self.parallel_envs, self.events, self.shape, key="test")

        expected_shape = torch.Size((10, self.events, self.parallel_envs, *self.shape))

        buffer_length = len(list(self.generator.buffers.values()))
        buffer_shape = list(self.generator.buffers.values())[0].shape
        buffer_index = list(self.generator.buffer_count.values())[0]

        self.assertEqual(
            buffer_length, 1, f"""
            \rBuffer store should have a single entry
                \rActual:\n{buffer_length}""")

        self.assertEqual(
            buffer_shape, expected_shape, f"""
            \rBuffer shape should be correct
                \rExpected:\n{expected_shape}
                \rActual:\n{buffer_shape}""")

        self.assertEqual(
            buffer_index, 1, f"""
            \rBuffer index should be initialized to 1 after buffer creation (creation and first return)
                \rActual:\n{buffer_index}""")

    def test_multiple_buffers_maintained(self) -> None:
        self.generator.buffer_size = 10
        self.generator.generate(self.parallel_envs, self.events, self.shape, key="test")
        self.generator.generate(self.parallel_envs, self.events, self.shape, key="test2")

        expected_shape = torch.Size((10, self.events, self.parallel_envs, *self.shape))

        for buffer in self.generator.buffers.values():
            self.assertEqual(
                buffer.shape, expected_shape, f"""
                \rBuffer shape should be correct
                    \rExpected:\n{expected_shape}
                    \rActual:\n{buffer.shape}""")
