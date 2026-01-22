from __future__ import annotations

import numpy as np

from ddm.utils import temp_numpy_seed
from .core import batch_simulator
from .priors import IntegrativeParams


def likelihood(
    alpha: float,
    tau: float,
    beta: float,
    mu_delta: float,
    eta_delta: float,
    gamma: float,
    sigma: float,
    n_obs: int,
    *,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """
    BayesFlow-compatible likelihood simulator.
    """
    params = IntegrativeParams(
        alpha, tau, beta, mu_delta, eta_delta, gamma, sigma
    ).as_array()

    if seed is None:
        choicert, z = batch_simulator(params, int(n_obs))
        return {"choicert": choicert, "z": z}

    with temp_numpy_seed(int(seed)):
        choicert, z = batch_simulator(params, int(n_obs))
        return {"choicert": choicert, "z": z}
