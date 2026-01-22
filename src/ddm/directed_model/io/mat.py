from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import scipy.io as sio

from ..simulation import DirectedDDMDataset


def to_stan_mat_dict(
    dataset: DirectedDDMDataset, params: dict[str, np.ndarray]
) -> dict[str, Any]:
    """
    Create a MATLAB/Stan-compatible dict with field names used there.
    Keeps naming differences isolated to IO.
    """
    return {
        **params,
        "rt": dataset.rt,
        "acc": dataset.acc,
        "y": dataset.y,
        "participant": dataset.participant,
        "nparts": dataset.n_parts,
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
