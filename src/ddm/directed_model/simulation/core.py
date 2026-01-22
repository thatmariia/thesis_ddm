from __future__ import annotations

from typing import Literal, Optional
import numpy as np

NoiseDist = Literal["gaussian", "laplace", "uniform", "base"]


def _sample_latent_z(
    n_trials: int,
    mu_z: float,
    sigma_z: float,
    noise_distribution: NoiseDist,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample latent z with mean mu_z and sd sigma_z under a chosen distribution."""
    dist = "gaussian" if noise_distribution == "base" else noise_distribution

    if dist == "gaussian":
        return rng.normal(mu_z, sigma_z, size=n_trials)

    if dist == "laplace":
        # Laplace var = 2*b^2 => b = sigma/sqrt(2)
        b_laplace = sigma_z / np.sqrt(2.0)
        return rng.laplace(mu_z, b_laplace, size=n_trials)

    if dist == "uniform":
        # Uniform var = (b-a)^2/12 => half-range = sqrt(3)*sigma
        a = mu_z - np.sqrt(3.0) * sigma_z
        b = mu_z + np.sqrt(3.0) * sigma_z
        return rng.uniform(a, b, size=n_trials)

    raise ValueError(f"Unknown noise_distribution: {noise_distribution}")


def simul_directed_ddm(
    *,
    n_trials: int = 100,
    alpha: float = 1.0,
    tau: float = 0.4,
    beta: float = 0.5,
    eta: float = 0.3,
    varsigma: float = 1.0,
    mu_z: float = 0.0,
    sigma_z: float = 1.0,
    lambda_param: float = 0.7,
    b: float = 0.5,
    noise_distribution: NoiseDist = "gaussian",
    n_steps: int = 10_000,
    step_length: float = 0.001,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate signed response times for a directed DDM where drift depends on latent z.

    Returns
    -------
    signed_rt : (n_trials,) float
        Signed RT (RT * choice). Positive = upper bound, negative = lower bound.
        If no bound hit within n_steps, choice is NaN -> signed_rt becomes NaN.
    random_walks : (n_steps, n_trials) float
        Random-walk paths.
    z_value : (n_trials,) float
        Latent z samples.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if n_steps <= 1:
        raise ValueError("n_steps must be > 1")
    if step_length <= 0:
        raise ValueError("step_length must be positive")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0, 1]")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    rng = rng or np.random.default_rng()

    rts = np.zeros(n_trials, dtype=float)
    choice = np.zeros(n_trials, dtype=float)

    z_value = _sample_latent_z(
        n_trials=n_trials,
        mu_z=mu_z,
        sigma_z=sigma_z,
        noise_distribution=noise_distribution,
        rng=rng,
    )

    # Trial-specific drift rates
    drift_rates = rng.normal(loc=lambda_param * z_value + b, scale=eta, size=n_trials)

    random_walks = np.zeros((n_steps, n_trials), dtype=float)

    for n in range(n_trials):
        drift = float(drift_rates[n])
        rw = np.zeros(n_steps, dtype=float)
        rw[0] = beta * alpha

        hit = False
        for s in range(1, n_steps):
            rw[s] = rw[s - 1] + rng.normal(
                loc=drift * step_length,
                scale=varsigma * np.sqrt(step_length),
            )

            if rw[s] >= alpha:
                rw[s:] = alpha
                rts[n] = s * step_length + tau
                choice[n] = 1.0
                hit = True
                break

            if rw[s] <= 0.0:
                rw[s:] = 0.0
                rts[n] = s * step_length + tau
                choice[n] = -1.0
                hit = True
                break

        if not hit:
            # Keep original behavior: no-decision -> NaN choice
            rts[n] = (n_steps - 1) * step_length + tau
            choice[n] = np.nan

        random_walks[:, n] = rw

    signed_rt = rts * choice
    return signed_rt, random_walks, z_value
