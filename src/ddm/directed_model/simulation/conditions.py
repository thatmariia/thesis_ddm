from __future__ import annotations

from typing import Literal, Optional
import numpy as np

Coupling = Literal["low", "high", "base", "zero"]
SNRLevel = Literal["low", "high", "base", "no_noise"]
SNRTransform = Literal["add", "mul"]


def adjust_sigma_z(
    base_sigma_z: float,
    *,
    snr_level: SNRLevel,
    transform: SNRTransform = "add",
    low_add: float = 0.5,
    low_mul: float = 5.0,
) -> float:
    """
    Adjust sigma_z according to an SNR level and a transform policy.

    - base/high: unchanged
    - no_noise: sigma_z = 0
    - low + add: sigma_z = base + low_add
    - low + mul: sigma_z = base * low_mul
    """
    if snr_level in ("high", "base"):
        return float(base_sigma_z)
    if snr_level == "no_noise":
        return 0.0
    if snr_level == "low":
        if transform == "add":
            return float(base_sigma_z + low_add)
        if transform == "mul":
            return float(base_sigma_z * low_mul)
        raise ValueError(f"Unknown transform: {transform}")
    raise ValueError(f"Unknown snr_level: {snr_level}")


def sample_lambda_param(
    n_parts: int,
    coupling: Coupling,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample lambda_param according to coupling condition."""
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")

    rng = rng or np.random.default_rng()

    if coupling == "base":
        return rng.uniform(-3.0, 3.0, size=n_parts)

    if coupling == "low":
        return rng.uniform(-0.2, 0.2, size=n_parts)

    if coupling == "high":
        signs = rng.choice([-1.0, 1.0], size=n_parts)
        magnitudes = rng.uniform(2.0, 3.0, size=n_parts)
        return signs * magnitudes

    if coupling == "zero":
        return np.zeros(n_parts, dtype=float)

    raise ValueError(f"Unknown coupling: {coupling}")
