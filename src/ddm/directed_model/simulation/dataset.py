from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import simul_directed_ddm, NoiseDist
from .priors import PriorConfig, sample_prior_params
from .conditions import (
    SNRLevel,
    SNRTransform,
    Coupling,
    adjust_sigma_z,
    sample_lambda_param,
)


@dataclass(frozen=True)
class DirectedDDMDataset:
    y: np.ndarray
    rt: np.ndarray
    acc: np.ndarray
    z: np.ndarray
    participant: np.ndarray  # 1..n_participants
    min_rt: np.ndarray  # per participant
    n_participants: int
    n_trials: int
    condition: str


def generate_directed_ddm_data(
    *,
    n_trials: int = 100,
    n_participants: int = 100,
    snr: SNRLevel = "base",
    snr_transform: SNRTransform = "add",
    coupling: Coupling = "base",
    dist: NoiseDist = "base",
    prior_cfg: PriorConfig = PriorConfig(),
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, np.ndarray], DirectedDDMDataset]:
    """
    Generate a full dataset for the directed DDM.

    Returns
    -------
    params : dict[str, np.ndarray]
        Participant-level parameters (arrays of shape (n_participants,))
    dataset : DirectedDDMDataset
        Trial-level simulated data
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if n_participants <= 0:
        raise ValueError("n_participants must be positive")

    rng = rng or np.random.default_rng()

    params = sample_prior_params(n_participants, cfg=prior_cfg, rng=rng)

    # Override lambda_param according to coupling condition
    params["lambda_param"] = sample_lambda_param(n_participants, coupling, rng=rng)

    condition_key = f"SNR_{snr}_{snr_transform}_COUP_{coupling}_DIST_{dist}"

    N = n_trials * n_participants
    y = np.zeros(N, dtype=float)
    rt = np.zeros(N, dtype=float)
    acc = np.zeros(N, dtype=float)
    participant = np.zeros(N, dtype=int)
    z_all = np.zeros(N, dtype=float)

    idx = 0
    for p in range(n_participants):
        sigma_z_p = adjust_sigma_z(
            float(params["sigma_z"][p]),
            snr_level=snr,
            transform=snr_transform,
        )

        signed_rt, _, z_sim = simul_directed_ddm(
            n_trials=n_trials,
            alpha=float(params["alpha"][p]),
            tau=float(params["tau"][p]),
            beta=float(params["beta"][p]),
            eta=float(params["eta"][p]),
            mu_z=float(params["mu_z"][p]),
            sigma_z=float(sigma_z_p),
            noise_distribution=dist,
            lambda_param=float(params["lambda_param"][p]),
            b=float(params["b"][p]),
            rng=rng,
        )

        accuracy_sign = np.sign(signed_rt)
        response_time = np.abs(signed_rt)

        start, end = idx, idx + n_trials
        y[start:end] = accuracy_sign * response_time
        rt[start:end] = response_time
        acc[start:end] = (accuracy_sign + 1.0) / 2.0
        participant[start:end] = p + 1  # Stan-friendly indexing
        z_all[start:end] = z_sim
        idx = end

    min_rt = np.full(n_participants, np.nan, dtype=float)
    for p in range(n_participants):
        rts_p = rt[participant == (p + 1)]
        valid = np.isfinite(rts_p)
        if np.any(valid):
            min_rt[p] = float(np.min(rts_p[valid]))

    dataset = DirectedDDMDataset(
        y=y,
        rt=rt,
        acc=acc,
        z=z_all,
        participant=participant,
        min_rt=min_rt,
        n_participants=n_participants,
        n_trials=n_trials,
        condition=condition_key,
    )
    return params, dataset
