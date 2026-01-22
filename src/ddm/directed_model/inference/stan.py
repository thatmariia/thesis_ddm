from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional
import numpy as np
import cmdstanpy
import scipy.io as sio

StanModelName = Literal["directed_ddm", "directed_ddm_cross"]


def load_mat_for_stan(mat_file_path: Path) -> dict:
    genparam = sio.loadmat(mat_file_path)

    y = np.squeeze(genparam["y"]).astype(float)
    z = np.squeeze(genparam["z"]).astype(float)
    participant = np.squeeze(genparam["participant"]).astype(int)
    min_rt = np.squeeze(genparam["minRT"]).astype(float)
    n_parts = int(np.squeeze(genparam["nparts"]).item())

    valid = np.isfinite(y) & ~np.isnan(y)
    y = y[valid]
    z = z[valid]
    participant = participant[valid]

    return {
        "N": int(y.size),
        "nparts": n_parts,
        "y": y,
        "participant": participant,
        "minRT": min_rt,
        "z": z,
    }


def fit_directed_ddm(
    mat_file_path: Path,
    *,
    model: StanModelName = "directed_ddm",
    stan_dir: Optional[Path] = None,
    chains: int = 4,
    parallel_chains: int = 4,
    iter_sampling: int = 1000,
    iter_warmup: int = 500,
    seed: int = 2025,
    show_console: bool = True,
) -> cmdstanpy.CmdStanMCMC:
    mat_file_path = Path(mat_file_path)

    if stan_dir is None:
        stan_dir = Path(__file__).resolve().parent / "stan_models"

    stan_file = stan_dir / f"{model}.stan"
    if not stan_file.exists():
        raise FileNotFoundError(f"Stan file not found: {stan_file}")

    model_obj = cmdstanpy.CmdStanModel(stan_file=str(stan_file))
    data_dict = load_mat_for_stan(mat_file_path)

    return model_obj.sample(
        data=data_dict,
        chains=chains,
        parallel_chains=parallel_chains,
        iter_sampling=iter_sampling,
        iter_warmup=iter_warmup,
        seed=seed,
        show_console=show_console,
    )
