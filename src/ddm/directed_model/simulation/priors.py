from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Literal
import numpy as np
from scipy.stats import truncnorm

SigmaZPrior = Literal["abs_normal", "trunc_low"]
MuZMode = Literal["random", "zero"]


@dataclass(frozen=True)
class PriorConfig:
    sigma_z_prior: SigmaZPrior = "abs_normal"
    mu_z_mode: MuZMode = "random"
    # parameters for trunc_low
    trunc_mean: float = 0.1
    trunc_std: float = 0.05
    trunc_low: float = 0.01
    trunc_high: float = 2.0


def _truncated_normal(
    mean: float,
    std: float,
    low: float,
    high: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    a, b = (low - mean) / std, (high - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size, random_state=rng)


def sample_prior_params(
    n_parts: int,
    *,
    cfg: PriorConfig = PriorConfig(),
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """
    Sample participant-level parameters from priors.

    Returns a dict with arrays of shape (n_parts,).
    """
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")

    rng = rng or np.random.default_rng()

    alpha = rng.uniform(0.8, 2.0, size=n_parts)
    tau = rng.uniform(0.15, 0.6, size=n_parts)
    beta = rng.uniform(0.3, 0.7, size=n_parts)

    if cfg.mu_z_mode == "random":
        mu_z = rng.normal(0.0, 1.0, size=n_parts)
    elif cfg.mu_z_mode == "zero":
        mu_z = np.zeros(n_parts, dtype=float)
    else:
        raise ValueError(f"Unknown mu_z_mode: {cfg.mu_z_mode}")

    if cfg.sigma_z_prior == "abs_normal":
        sigma_z = np.abs(rng.normal(0.5, 0.5, size=n_parts))
    elif cfg.sigma_z_prior == "trunc_low":
        sigma_z = _truncated_normal(
            mean=cfg.trunc_mean,
            std=cfg.trunc_std,
            low=cfg.trunc_low,
            high=cfg.trunc_high,
            size=n_parts,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown sigma_z_prior: {cfg.sigma_z_prior}")

    b = rng.normal(0.0, 1.0, size=n_parts)
    eta = rng.uniform(0.0, 1.0, size=n_parts)
    lambda_param = rng.uniform(-3.0, 3.0, size=n_parts)

    return {
        "alpha": alpha,
        "tau": tau,
        "beta": beta,
        "mu_z": mu_z,
        "sigma_z": sigma_z,
        "b": b,
        "eta": eta,
        "lambda_param": lambda_param,
    }
