"""Hashing functions for caching and memoization."""
from typing import Union
import xxhash
import torch
from tensordict import TensorDict


def hash_observation(observation: TensorDict) -> int:
    """Hash an observation dictionary through hashing its elements."""
    observation_elements = []
    for key, value in observation.items():
        observation_elements.append(optimized_convert_hashable(value))

    return hash(tuple(observation_elements))


def optimized_convert_hashable(data: torch.Tensor) -> Union[int, float]:
    """
    Convert a tensor to a hashable value using one of a few hashing functions.

    Hashing functions:
    - Uses a xxhash to encode a tensor into a hashable uint64.

    Args:
        data: torch.Tensor - The data to convert
    Returns:
        int | float - The hashable value
    """
    # if torch.cuda.is_available() and data.is_cuda:
    #     return positional_encoding_hash(data)

    return convert_using_xxhash(data)


def positional_encoding_hash(data: torch.Tensor, batched: bool = False) -> int:
    """
    Convert a tensor to a hashable using a positional encoding hash function.

    Args:
        data: torch.Tensor - The data to convert
        batched:torch.Tensor - The data to convert
    Returns:
        int - The hash of the data
    """
    data = data.flatten()

    # Scale each value by the index of each element
    indices = torch.arange(data.shape[0], device=data.device, dtype=torch.float64)
    data = indices * data

    # Calculate the positional encoding scaling for each element
    angles = indices / torch.pow(5000, (2 * indices) / data.shape[0])
    angles[0::2] = torch.sin(angles[0::2])
    angles[1::2] = torch.cos(angles[1::2])

    # Combine both encodings
    return torch.sum(data * angles).item()


def convert_using_xxhash(data: torch.Tensor) -> int:
    """
    Convert a tensor to a hash using xxhash.

    Args:
        data: torch.Tensor - The data to convert
    Returns:
        int - The hash of the data
    """
    data = data.flatten().cpu()
    data_bytes = data.numpy().tobytes()

    return xxhash.xxh64(data_bytes).intdigest()


def convert_using_tuple(data: torch.Tensor) -> int:
    """
    Convert a tensor to a tuple.

    Args:
        data: torch.Tensor - The data to convert
    Returns:
        int - The hash of the data
    """
    data = data.flatten().cpu().numpy()

    return hash(tuple(data))
