from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import scipy.io as sio

from ..simulation import DirectedDDMDataset


def to_stan_mat_dict(
    dataset: DirectedDDMDataset, params: dict[str, np.ndarray]
) -> dict[str, Any]:
    params_out = dict(params)

    # naming compatibility: expose both lambda_param and lambda
    if "lambda_param" in params_out and "lambda" not in params_out:
        params_out["lambda"] = params_out["lambda_param"]
    if "lambda" in params_out and "lambda_param" not in params_out:
        params_out["lambda_param"] = params_out["lambda"]

    return {
        **params_out,
        "rt": dataset.rt,
        "acc": dataset.acc,
        "y": dataset.y,
        "participant": dataset.participant,
        "nparts": dataset.n_participants,
        "ntrials": dataset.n_trials,
        "N": int(dataset.y.size),
        "minRT": dataset.min_rt,
        "z": dataset.z,
        "condition": dataset.condition,
    }


def save_dataset_mat(
    dataset: DirectedDDMDataset, params: dict[str, np.ndarray], filepath: Path
) -> Path:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    sio.savemat(filepath, to_stan_mat_dict(dataset, params))
    return filepath


def load_mat(filepath: Path) -> dict[str, Any]:
    """Raw load for backwards compatibility."""
    return sio.loadmat(filepath)
