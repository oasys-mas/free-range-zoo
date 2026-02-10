"""Memory Restriction Utilities for FreeRangeZoo.

This module provides functions for limiting memory usage during environment
execution. This is useful for preventing memory overflow in long-running
simulations or when running in constrained environments.

Note:
    Memory restriction via setrlimit() only works on Unix-like systems
    (Linux, macOS) and not on Windows.

Functions:
    limit_memory: Restrict process memory usage to a specified limit.

Example Usage:
    >>> from free_range_zoo.utils.mem_restrict import limit_memory
    >>> limit_memory(0.89)  # Limit to 89% of available memory
"""

import psutil
import resource

import logging

logger = logging.getLogger('free_range_zoo')


def limit_memory(memory_limit: float) -> None:
    """
    Limit memory usage to the given limit. Does not function on Windows.

    Args:
        memory_limit (float): Percentage of memory to be used
    """
    virtual_memory = psutil.virtual_memory()
    available_memory = virtual_memory.available
    memory_limit = int(available_memory * memory_limit)

    logger.info(f'{memory_limit} memory limit, available: {available_memory}')

    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))


def main() -> None:
    limit_memory(0.89)


if __name__ == '__main__':
    main()
