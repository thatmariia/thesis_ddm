from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Literal
import numpy as np
from scipy.stats import truncnorm

SigmaZPrior = Literal["abs_normal", "trunc_low"]
MuZMode = Literal["random", "zero"]


@dataclass(frozen=True)
class DirectedParams:
    """
    Directed DDM parameters (participant-level).
    """

    alpha: float
    tau: float
    beta: float
    mu_z: float
    sigma_z: float
    b: float
    eta: float
    lambda_param: float

    def as_dict(self) -> dict[str, float]:
        return {
            "alpha": float(self.alpha),
            "tau": float(self.tau),
            "beta": float(self.beta),
            "mu_z": float(self.mu_z),
            "sigma_z": float(self.sigma_z),
            "b": float(self.b),
            "eta": float(self.eta),
            "lambda_param": float(self.lambda_param),
        }

    def as_array(self, dtype=np.float64) -> np.ndarray:
        """
        Array representation in a fixed, documented order.

        Order:
        [alpha, tau, beta, mu_z, sigma_z, b, eta, lambda_param]
        """
        return np.array(
            [
                self.alpha,
                self.tau,
                self.beta,
                self.mu_z,
                self.sigma_z,
                self.b,
                self.eta,
                self.lambda_param,
            ],
            dtype=dtype,
        )

    def param_names() -> list[str]:
        """
        Parameter names in the order of as_array().
        """
        return [
            "alpha",
            "tau",
            "beta",
            "mu_z",
            "sigma_z",
            "b",
            "eta",
            "lambda_param",
        ]


@dataclass(frozen=True)
class PriorConfig:
    """
    Prior hyperparameters for directed model.
    Mirrors your current hard-coded choices, but configurable.
    """

    alpha_low: float = 0.8
    alpha_high: float = 2.0

    tau_low: float = 0.15
    tau_high: float = 0.6

    beta_low: float = 0.3
    beta_high: float = 0.7

    mu_z_mode: MuZMode = "random"
    mu_z_mean: float = 0.0
    mu_z_std: float = 1.0

    sigma_z_prior: SigmaZPrior = "abs_normal"
    sigma_z_abs_mean: float = 0.5
    sigma_z_abs_std: float = 0.5

    # trunc_low option for sigma_z
    trunc_mean: float = 0.1
    trunc_std: float = 0.05
    trunc_low: float = 0.01
    trunc_high: float = 2.0

    b_mean: float = 0.0
    b_std: float = 1.0

    eta_low: float = 0.0
    eta_high: float = 1.0

    lambda_low: float = -3.0
    lambda_high: float = 3.0


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
) -> DirectedParams:
    """
    One typed prior draw (one participant).
    """
    rng = rng or np.random.default_rng()

    alpha = float(rng.uniform(cfg.alpha_low, cfg.alpha_high))
    tau = float(rng.uniform(cfg.tau_low, cfg.tau_high))
    beta = float(rng.uniform(cfg.beta_low, cfg.beta_high))

    if cfg.mu_z_mode == "random":
        mu_z = float(rng.normal(cfg.mu_z_mean, cfg.mu_z_std))
    elif cfg.mu_z_mode == "zero":
        mu_z = 0.0
    else:
        raise ValueError(f"Unknown mu_z_mode: {cfg.mu_z_mode}")

    if cfg.sigma_z_prior == "abs_normal":
        sigma_z = float(np.abs(rng.normal(cfg.sigma_z_abs_mean, cfg.sigma_z_abs_std)))
    elif cfg.sigma_z_prior == "trunc_low":
        sigma_z = float(
            truncated_normal(
                cfg.trunc_mean,
                cfg.trunc_std,
                cfg.trunc_low,
                cfg.trunc_high,
                size=None,
                rng=rng,
            )
        )
    else:
        raise ValueError(f"Unknown sigma_z_prior: {cfg.sigma_z_prior}")

    b = float(rng.normal(cfg.b_mean, cfg.b_std))
    eta = float(rng.uniform(cfg.eta_low, cfg.eta_high))
    lambda_param = float(rng.uniform(cfg.lambda_low, cfg.lambda_high))

    return DirectedParams(
        alpha=alpha,
        tau=tau,
        beta=beta,
        mu_z=mu_z,
        sigma_z=sigma_z,
        b=b,
        eta=eta,
        lambda_param=lambda_param,
    )


def sample_prior_params(
    n_parts: int,
    *,
    cfg: PriorConfig = PriorConfig(),
    rng: np.random.Generator | None = None,
) -> list[DirectedParams]:
    """
    Many typed draws (one per participant).
    """
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")

    rng = rng or np.random.default_rng()
    return [prior(cfg=cfg, rng=rng) for _ in range(int(n_parts))]


def sample_prior_arrays(
    n_parts: int,
    *,
    cfg: PriorConfig = PriorConfig(),
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """
    Bridge for existing directed code: dict of arrays shape (n_parts,).
    Uses DirectedParams.as_array() as the single source of truth for ordering.
    """
    draws = sample_prior_params(n_parts, cfg=cfg, rng=rng)

    mat = np.stack([d.as_array(dtype=float) for d in draws], axis=0)
    names = DirectedParams.param_names()

    return {name: mat[:, i] for i, name in enumerate(names)}


def sample_prior_dict(
    *,
    cfg: PriorConfig = PriorConfig(),
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """
    One participant draw as plain dict (symmetry with integrative).
    """
    return prior(cfg=cfg, rng=rng).as_dict()
