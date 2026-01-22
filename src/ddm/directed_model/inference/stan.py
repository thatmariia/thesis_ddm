from __future__ import annotations

from pathlib import Path
from collections.abc import Literal, TypedDict
import numpy as np
import cmdstanpy
import scipy.io as sio

StanModelName = Literal["directed_ddm", "directed_ddm_cross"]


class StanData(TypedDict):
    N: int
    nparts: int
    y: np.ndarray
    participant: np.ndarray
    minRT: np.ndarray
    z: np.ndarray


def load_mat_for_stan(mat_file_path: Path) -> StanData:
    genparam = sio.loadmat(mat_file_path)

    y = np.squeeze(genparam["y"]).astype(float)
    z = np.squeeze(genparam["z"]).astype(float)
    participant = np.squeeze(genparam["participant"]).astype(int)
    min_rt = np.squeeze(genparam["minRT"]).astype(float)
    n_participants = int(np.squeeze(genparam["nparts"]).item())

    valid = np.isfinite(y) & ~np.isnan(y)
    y = y[valid]
    z = z[valid]
    participant = participant[valid]

    # basic sanity checks
    if y.shape[0] != z.shape[0] or y.shape[0] != participant.shape[0]:
        raise ValueError("After filtering, y/z/participant lengths do not match")

    if min_rt.shape[0] != n_participants:
        raise ValueError(f"minRT length {min_rt.shape[0]} != nparts {n_participants}")

    if participant.min(initial=1) < 1:
        raise ValueError("participant indices must be 1..nparts (found < 1)")
    if participant.max(initial=0) > n_participants:
        raise ValueError("participant indices must be 1..nparts (found > nparts)")

    return {
        "N": int(y.size),
        "nparts": n_participants,
        "y": y,
        "participant": participant,
        "minRT": min_rt,
        "z": z,
    }


def fit_directed_ddm(
    mat_file_path: Path,
    *,
    model: StanModelName = "directed_ddm",
    stan_dir: Path | None = None,
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
