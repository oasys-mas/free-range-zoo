from free_range_zoo.wrappers.wrapper_util import shared_wrapper_aec, shared_wrapper_gym, shared_wrapper_parr


def list_wrappers(env):
    """
    Return an ordered list of all wrappers applied to a PettingZoo environment.
    Ordered from outermost wrapper to innermost.

    Args:
        env: a free-range-zoo wrapped environment
    """
    wrappers = []
    current = env

    # Walk down the chain of wrappers
    while hasattr(current, "env") or hasattr(current, "aec_env"):
        wrappers.append(type(current).__name__)
        if isinstance(current, shared_wrapper_parr) or \
            isinstance(current, shared_wrapper_gym) or \
            isinstance(current, shared_wrapper_aec):
            wrappers[-1] = wrappers[-1] + f":{current.modifier_class}"

        current = current.env if hasattr(current, "env") else current.aec_env

    # current is now the base environment, not a wrapper
    return wrappers
