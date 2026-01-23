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

    def as_dict(self) -> dict[str, float]:
        return {
            "alpha": float(self.alpha),
            "tau": float(self.tau),
            "beta": float(self.beta),
            "mu_delta": float(self.mu_delta),
            "eta_delta": float(self.eta_delta),
            "gamma": float(self.gamma),
            "sigma": float(self.sigma),
        }


@dataclass(frozen=True)
class PriorConfig:
    """
    Prior hyperparameters for the integrative model.
    """

    alpha_low: float = 0.5
    alpha_high: float = 2.0

    tau_low: float = 0.1
    tau_high: float = 1.0

    beta_mean: float = 0.5
    beta_std: float = 0.25
    beta_low: float = 0.001
    beta_high: float = 0.99

    mu_delta_mean: float = 0.0
    mu_delta_std: float = 1.0

    eta_delta_low: float = 0.0
    eta_delta_high: float = 2.0

    gamma_low: float = -3.0
    gamma_high: float = 3.0

    sigma_low: float = 0.0
    sigma_high: float = 2.0


def truncated_normal(
    mean: float,
    std: float,
    low: float,
    high: float,
    *,
    size: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    a, b = (low - mean) / std, (high - mean) / std
    if rng is None:
        return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size, random_state=rng)


def prior(
    *,
    cfg: PriorConfig = PriorConfig(),
    rng: np.random.Generator | None = None,
) -> IntegrativeParams:
    """
    One typed prior draw.
    """
    rng = rng or np.random.default_rng()

    alpha = float(rng.uniform(cfg.alpha_low, cfg.alpha_high))
    tau = float(rng.uniform(cfg.tau_low, cfg.tau_high))
    beta = float(
        truncated_normal(
            cfg.beta_mean,
            cfg.beta_std,
            cfg.beta_low,
            cfg.beta_high,
            size=None,
            rng=rng,
        )
    )

    mu_delta = float(rng.normal(cfg.mu_delta_mean, cfg.mu_delta_std))
    eta_delta = float(rng.uniform(cfg.eta_delta_low, cfg.eta_delta_high))
    gamma = float(rng.uniform(cfg.gamma_low, cfg.gamma_high))
    sigma = float(rng.uniform(cfg.sigma_low, cfg.sigma_high))

    return IntegrativeParams(
        alpha=alpha,
        tau=tau,
        beta=beta,
        mu_delta=mu_delta,
        eta_delta=eta_delta,
        gamma=gamma,
        sigma=sigma,
    )


def sample_prior_params(
    n_sims: int,
    *,
    cfg: PriorConfig = PriorConfig(),
    rng: np.random.Generator | None = None,
) -> list[IntegrativeParams]:
    """
    Many typed prior draws.
    """
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")

    rng = rng or np.random.default_rng()
    return [prior(cfg=cfg, rng=rng) for _ in range(int(n_sims))]


def sample_prior_dict(
    *,
    cfg: PriorConfig = PriorConfig(),
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """
    BayesFlow-friendly: one draw as a dict of floats.
    """
    return prior(cfg=cfg, rng=rng).as_dict()
