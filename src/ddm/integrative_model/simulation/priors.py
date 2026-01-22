from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import truncnorm


@dataclass(frozen=True)
class IntegrativeParams:
    """
    Integrative DDM + P300 parameters.
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


def truncated_normal(
    mean: float,
    std: float,
    low: float,
    high: float,
    size=None,
    rng: np.random.Generator | None = None,
):
    """
    Truncated normal sampling via scipy.stats.truncnorm.
    """
    a, b = (low - mean) / std, (high - mean) / std
    if rng is None:
        return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size, random_state=rng)


def prior(rng: np.random.Generator | None = None) -> dict[str, float]:
    """
    Prior distribution.
    """
    rng = rng or np.random.default_rng()

    alpha = float(rng.uniform(0.5, 2.0))
    tau = float(rng.uniform(0.1, 1.0))
    beta = float(truncated_normal(0.5, 0.25, 0.001, 0.99, rng=rng))
    mu_delta = float(rng.normal(0.0, 1.0))
    eta_delta = float(rng.uniform(0.0, 2.0))
    gamma = float(rng.uniform(-3.0, 3.0))
    sigma = float(rng.uniform(0.0, 2.0))

    return {
        "alpha": alpha,
        "tau": tau,
        "beta": beta,
        "mu_delta": mu_delta,
        "eta_delta": eta_delta,
        "gamma": gamma,
        "sigma": sigma,
    }
