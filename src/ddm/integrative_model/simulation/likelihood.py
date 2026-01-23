from __future__ import annotations

import numpy as np

from ...utils import temp_numpy_seed
from .core import batch_simulator
from .priors import IntegrativeParams


def likelihood(
    params: IntegrativeParams,
    n_obs: int,
    *,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """
    BayesFlow-compatible likelihood simulator.
    """
    params_arr = params.as_array()

    if seed is None:
        choicert, z = batch_simulator(params_arr, int(n_obs))
        return {"choicert": choicert, "z": z}

    with temp_numpy_seed(int(seed)):
        choicert, z = batch_simulator(params_arr, int(n_obs))
        return {"choicert": choicert, "z": z}