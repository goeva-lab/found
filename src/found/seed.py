# @TODO: should this be behind a multiprocessing.Lock (?)
__RAND_SEED = None


def set_seed(seed: int | None):
    """
    conveninence function used to set a singular random seed for all methods utilized in this module, with the aim of guaranteeing reproducibility.

    :param seed: integer seed, must be within [0, 4294967295], set to None to remove fixed seeding
    """
    global __RAND_SEED
    if seed is not None and (seed < 0 or seed > 4294967295):
        raise ValueError(f"provided seed {seed} outside of allowed range [0, 4294967295]")
    __RAND_SEED = seed


def get_seed() -> int | None:
    return __RAND_SEED
