"""Random generators for free-range-zoo stochastic generation."""
from typing import Tuple
import torch


class RandomGenerator:
    """Random number generator for the environment."""

    def __init__(self,
                 parallel_envs: int,
                 buffer_size: int = 0,
                 single_seeding: bool = False,
                 device: torch.DeviceObjType = torch.device('cpu')):
        """
        Initialize the random generator.

        Args:
            buffer_size: int - the size of the buffer for random number generation
            single_seeding: bool - whether to seed the generator once for all environments
            device: torch.DeviceObjType - the device to use for random number generation
        """
        self.parallel_envs = parallel_envs
        self.buffer_size = buffer_size
        self.single_seeding = single_seeding
        self.device = device

        match str(self.device):
            case device if device.startswith('cuda'):
                state_size = 16
            case 'cpu':
                state_size = 5056
            case _:
                raise ValueError(f"Device {self.device} not supported")

        self.generator = torch.Generator(device=self.device)

        if self.single_seeding:
            self.seeds = torch.empty((1, ), dtype=torch.int32, device=self.device)
            self.generator_states = torch.empty((1, state_size), dtype=torch.uint8, device=torch.device('cpu'))
        else:
            self.seeds = torch.empty((self.parallel_envs), dtype=torch.int32, device=self.device)
            self.generator_states = torch.empty((self.parallel_envs, state_size), dtype=torch.uint8, device=torch.device('cpu'))

        self.buffer_count = {}
        self.buffers = {}

        self.has_been_seeded = False

    @torch.no_grad()
    def seed(self, seed: torch.Tensor = None, partial_seeding: torch.Tensor = None) -> None:
        """
        Seeds the environment, or randomly generates a seed if none is provided.

        Args:
            seed: torch.Tensor - the seed to use for the environment
            partial_seeding: torch.Tensor - the environment indices to seed
        """
        if seed is None:
            seed_shape = self.seeds.shape if partial_seeding is None else partial_seeding.shape
            seed = torch.randint(100000000, seed_shape, device=self.device)

        if self.single_seeding:
            self.seeds[:] = seed
            generator = torch.Generator(device=self.device)
            self.generator_states[0] = generator.get_state()

            self.has_been_seeded = True
            return

        match partial_seeding:
            case None:
                self.seeds[:] = seed
            case _:
                self.seeds[partial_seeding] = seed

        generator = torch.Generator(device=self.device)
        for index, seed in enumerate(self.seeds):
            if partial_seeding is not None and not torch.isin(index, partial_seeding):
                continue

            generator.manual_seed(seed.item())
            self.generator_states[index] = generator.get_state()

        self.has_been_seeded = True

    @torch.no_grad()
    def generate(self, parallel_envs: int, events: int, shape: Tuple[int], key: str = None) -> torch.FloatTensor:
        """
        Generate random tensors for the environment.

        Args:
            parallel_envs: int - the number of parallel environments
            events: int - the number of events that require randomness
            shape: Tuple[int] - the shape of the random tensors
            key: str - the key to use for the buffer
        """
        if not self.has_been_seeded:
            raise ValueError("The environment must be seeded before generating randomness")

        # If we are choosing not to use a buffered generation, then just return the raw tensor
        if key is None or self.buffer_size == 0:
            if self.single_seeding:
                self.generator.set_state(self.generator_states[0])
                output = torch.rand((parallel_envs, events, *shape), device=self.device, generator=self.generator)
                self.generator_states[0] = self.generator.get_state()
            else:
                output = torch.empty((parallel_envs, events, *shape), device=self.device)
                for index in range(parallel_envs):
                    self.generator.set_state(self.generator_states[index])
                    output[index] = torch.rand((events, *shape), device=self.device, generator=self.generator)
                    self.generator_states[index] = self.generator.get_state()

            output = output.transpose(1, 0)
            return output

        # If the buffer does not exist or the buffer is exhausted, create / refresh it
        buffer_key = (key, (parallel_envs, events, *shape))
        if buffer_key not in self.buffers or self.buffer_count[buffer_key] >= self.buffer_size:
            if self.single_seeding:
                self.generator.set_state(self.generator_states[0])
                output = torch.rand(
                    (self.buffer_size, parallel_envs, events, *shape),
                    device=self.device,
                    generator=self.generator,
                )
                self.generator_states[0] = self.generator.get_state()
            else:
                output = torch.empty(
                    (self.generator_states.shape[0], self.buffer_size, events, *shape),
                    device=self.device,
                )
                for index in range(self.generator_states.shape[0]):
                    self.generator.set_state(self.generator_states[index])
                    output[index] = torch.rand((self.buffer_size, events, *shape), device=self.device, generator=self.generator)
                    self.generator_states[index] = self.generator.get_state()
                output = output.transpose(0, 1)

            output = output.transpose(1, 2)

            self.buffers[buffer_key] = output
            self.buffer_count[buffer_key] = 0

        # Get the buffer and increment
        output = self.buffers[buffer_key][self.buffer_count[buffer_key]]
        self.buffer_count[buffer_key] += 1
        return output

    def state_dict(self) -> dict:
        """Save the state of the random generator."""
        return {
            'parallel_envs': self.parallel_envs,
            'buffer_size': self.buffer_size,
            'single_seeding': self.single_seeding,
            'device': str(self.device),
            'generator_state': self.generator.get_state(),
            'seeds': self.seeds.cpu() if self.seeds.is_cuda else self.seeds.clone(),
            'generator_states': self.generator_states.clone(),
            'buffer_count': self.buffer_count,
            'buffers': self.buffers,
            'has_been_seeded': self.has_been_seeded
        }

    def load_state_dict(self, state: dict) -> None:
        """Load the state of the random generator."""
        self.parallel_envs = state['parallel_envs']
        self.buffer_size = state['buffer_size']
        self.single_seeding = state['single_seeding']
        self.device = torch.device(state['device'])

        self.generator.set_state(state['generator_state'])

        self.seeds = state['seeds'].to(self.device)
        self.generator_states = state['generator_states'].clone()
        self.buffer_count = state['buffer_count']
        self.buffers = state['buffers']
        self.has_been_seeded = state['has_been_seeded']
