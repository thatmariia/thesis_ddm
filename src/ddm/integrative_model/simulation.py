from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.stats import truncnorm


@dataclass(frozen=True)
class IntegrativeParams:
    """
    Parameters for the integrative DDM + P300 model.

    alpha      : boundary separation
    tau        : non-decision time
    beta       : starting point bias (0..1)
    mu_delta   : mean drift
    eta_delta  : sd drift (trial-to-trial)
    gamma      : P300 coupling to drift
    sigma      : P300 noise sd
    """

    alpha: float
    tau: float
    beta: float
    mu_delta: float
    eta_delta: float
    gamma: float
    sigma: float

    def as_array(self, dtype=np.float64) -> np.ndarray:
        return np.array(
            [
                self.alpha,
                self.tau,
                self.beta,
                self.mu_delta,
                self.eta_delta,
                self.gamma,
                self.sigma,
            ],
            dtype=dtype,
        )


def truncated_normal(mean: float, std: float, low: float, high: float, size=None):
    """Sample from a truncated normal using scipy.stats.truncnorm."""
    a, b = (low - mean) / std, (high - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)


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
    Simulate one DDM trial and return signed choice-RT (positive = upper boundary, negative = lower boundary).

    Notes:
    - evidence starts at alpha * beta
    - while loop until boundary hit
    - choicert = tau + rt if hit upper else -tau - rt
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
    Simulate one trial: (choicert, z).
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
    Simulate n_obs trials given a single parameter vector.
    Returns float32 arrays (same as before).
    """
    sim_choicert = np.empty(n_obs, dtype=np.float32)
    sim_z = np.empty(n_obs, dtype=np.float32)

    for i in range(n_obs):
        cr, zz = simulate_trial(params_arr)
        sim_choicert[i] = cr
        sim_z[i] = zz

    return sim_choicert, sim_z


def prior() -> dict[str, float]:
    """
    Prior distribution for the integrative DDM + P300 model.
    Keeps your original priors (same ranges).
    """
    alpha = float(np.random.uniform(0.5, 2.0))
    tau = float(np.random.uniform(0.1, 1.0))
    beta = float(truncated_normal(0.5, 0.25, 0.001, 0.99))
    mu_delta = float(np.random.normal(0.0, 1.0))
    eta_delta = float(np.random.uniform(0.0, 2.0))
    gamma = float(np.random.uniform(-3.0, 3.0))
    sigma = float(np.random.uniform(0.0, 2.0))

    return dict(
        alpha=alpha,
        tau=tau,
        beta=beta,
        mu_delta=mu_delta,
        eta_delta=eta_delta,
        gamma=gamma,
        sigma=sigma,
    )


def likelihood(
    alpha: float,
    tau: float,
    beta: float,
    mu_delta: float,
    eta_delta: float,
    gamma: float,
    sigma: float,
    n_obs: int,
) -> dict[str, np.ndarray]:
    """
    BayesFlow-compatible likelihood simulator.
    Returns dict with keys: 'choicert', 'z'.
    """
    params = IntegrativeParams(
        alpha, tau, beta, mu_delta, eta_delta, gamma, sigma
    ).as_array()
    choicert, z = batch_simulator(params, int(n_obs))
    return dict(choicert=choicert, z=z)
