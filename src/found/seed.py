from collections.abc import Callable


def mk() -> tuple[Callable[[int | None], None], Callable[[], int | None]]:
    # @TODO: should this be behind a multiprocessing.Lock (?)
    __RAND_SEED = None

    def set_seed(seed: int | None):
        """
        convenience function used to set a singular random seed for all methods utilized in this module, with the aim of guaranteeing reproducibility.
        see :py:func:`~found.seed.get_seed` for seed access.

        :param seed: integer seed, must be within [0, 4294967295], set to None to remove fixed seeding
        """
        if (seed is not None) and ((seed < 0) or (seed > 4294967295)):
            raise ValueError(f"provided seed {seed} outside of allowed range [0, 4294967295]")

        nonlocal __RAND_SEED
        __RAND_SEED = seed

    def get_seed() -> int | None:
        """
        convenience function used to access a singular random seed for all methods utilized in this module, with the aim of guaranteeing reproducibility.
        see :py:func:`~found.seed.set_seed` for seed setting.

        :return: integer seed, within [0, 4294967295], or None if no seed has been fixed
        """
        nonlocal __RAND_SEED
        return __RAND_SEED

    return set_seed, get_seed


set_seed, get_seed = mk()
