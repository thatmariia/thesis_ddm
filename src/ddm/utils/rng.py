from __future__ import annotations

from contextlib import contextmanager
import numpy as np


def make_rng(seed: int | None = None) -> np.random.Generator:
    """
    Create a numpy Generator.
    """
    return np.random.default_rng(seed)


@contextmanager
def temp_numpy_seed(seed: int):
    """
    Temporarily set the *global* numpy RNG seed and restore it afterwards.
    """
    state = np.random.get_state()
    np.random.seed(int(seed) % (2**32 - 1))
    try:
        yield
    finally:
        np.random.set_state(state)
