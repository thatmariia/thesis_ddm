from __future__ import annotations

import numpy as np
from numba import njit


@njit
def simulate_ddm(
    alpha: float,
    tau: float,
    delta: float,
    beta: float,
    dt: float = 0.001,
    dc: float = 1.0,
) -> float:
    """
    Simulate one DDM trial; return signed choice-RT.
    Positive => upper boundary hit, negative => lower boundary hit.
    """
    evidence = alpha * beta
    n_steps = 0.0

    while 0.0 < evidence < alpha:
        evidence += delta * dt + np.sqrt(dt) * dc * np.random.normal()
        n_steps += 1.0

    rt = n_steps * dt
    return (tau + rt) if (evidence >= alpha) else (-tau - rt)


@njit
def simulate_trial(params_arr: np.ndarray) -> tuple[float, float]:
    """
    Simulate one trial (choicert, z).
    params_arr order: [alpha, tau, beta, mu_delta, eta_delta, gamma, sigma]
    """
    alpha, tau, beta, mu_delta, eta_delta, gamma, sigma = params_arr

    delta = np.random.normal(mu_delta, eta_delta)
    choicert = simulate_ddm(alpha, tau, delta, beta)
    z = np.random.normal(gamma * delta, sigma)

    return choicert, z


@njit
def batch_simulator(
    params_arr: np.ndarray, n_obs: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate n_obs trials. Returns float32 arrays (same as your original behavior).
    """
    sim_choicert = np.empty(n_obs, dtype=np.float32)
    sim_z = np.empty(n_obs, dtype=np.float32)

    for i in range(n_obs):
        cr, zz = simulate_trial(params_arr)
        sim_choicert[i] = cr
        sim_z[i] = zz

    return sim_choicert, sim_z
