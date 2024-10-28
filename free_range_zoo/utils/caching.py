import xxhash
import torch
from tensordict import TensorDict


def hash_observation(observation: TensorDict) -> int:
    observation_elements = []
    for key, value in observation.items():
        observation_elements.append(convert_using_xxhash(value))

    return hash(tuple(observation_elements))


def optimized_convert_hashable(data: torch.Tensor) -> int:
    """
        Convert a tensor to a hashable value using either xxhash or a tuple.
        If the size of the tensor is greater than 10000, xxhash is used, otherwise a tuple is used.

        Args:
            data: torch.Tensor - The data to convert
        Returns:
            int | Tuple[int] - The hashable value
    """
    if data.shape[0] > 10000:
        hashable = convert_using_xxhash(data)
    else:
        hashable = convert_using_tuple(data)

    return hashable


def convert_using_xxhash(data: torch.Tensor) -> int:
    """
    Convert a tensor to a hash using xxhash

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
    Convert a tensor to a tuple

    Args:
        data: torch.Tensor - The data to convert
    Returns:
        int - The hash of the data
    """
    data = data.flatten().cpu().numpy()

    return hash(tuple(data))
