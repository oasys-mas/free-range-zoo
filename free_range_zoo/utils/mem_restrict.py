import psutil
import resource

import logging

logger = logging.getLogger('free_range_zoo')


def limit_memory(memory_limit: float) -> None:
    """
    Limit memory usage to the given limit.
    Does not function on Windows.

    Args:
        memory_limit: float - percentage of memory to be used
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
