from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io.mat import save_dataset_mat
from .simulation import DirectedDDMDataset, generate_directed_ddm_data
from .inference.stan import fit_directed_ddm, StanModelName


@dataclass(frozen=True)
class WorkflowConfig:
    """
    Directed model workflow configuration.

    This is intentionally "workflow-ish", not "model-ish":
    - defaults for dataset generation
    - defaults for fitting
    """

    n_trials: int = 100
    n_parts: int = 50
    seed: int | None = 2025

    model: StanModelName = "directed_ddm"
    chains: int = 4
    parallel_chains: int = 4
    iter_sampling: int = 1000
    iter_warmup: int = 500
    show_console: bool = True


@dataclass
class DirectedWorkflow:
    """
    Small orchestration object to mirror the integrative model's "workflow" idea.

    It does NOT try to mimic BayesFlow internals. It just gives you:
      - simulate()
      - simulate_to_mat()
      - fit_from_mat()
      - draws_df()
    """

    cfg: WorkflowConfig = field(default_factory=WorkflowConfig)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.cfg.seed)

    def simulate(
        self,
        *,
        n_trials: int | None = None,
        n_parts: int | None = None,
        **sim_kwargs: Any,
    ) -> tuple[dict[str, np.ndarray], DirectedDDMDataset]:
        """
        Simulate a directed dataset (and participant-level params arrays).
        """
        return generate_directed_ddm_data(
            n_trials=int(n_trials if n_trials is not None else self.cfg.n_trials),
            n_parts=int(n_parts if n_parts is not None else self.cfg.n_parts),
            rng=self.rng,
            **sim_kwargs,
        )

    def simulate_to_mat(
        self,
        out_mat: str | Path,
        *,
        n_trials: int | None = None,
        n_parts: int | None = None,
        **sim_kwargs: Any,
    ) -> tuple[dict[str, np.ndarray], DirectedDDMDataset, Path]:
        """
        Simulate and write a .mat file for Stan.
        """
        params, dataset = self.simulate(
            n_trials=n_trials,
            n_parts=n_parts,
            **sim_kwargs,
        )
        out_path = save_dataset_mat(dataset, params, Path(out_mat))
        return params, dataset, out_path

    def fit_from_mat(
        self,
        mat_file: str | Path,
        *,
        model: StanModelName | None = None,
        stan_dir: str | Path | None = None,
        seed: int | None = None,
        **fit_overrides: Any,
    ):
        """
        Fit CmdStan model using defaults from cfg, with per-call overrides.

        Note: seed here is for Stan sampling; dataset reproducibility is controlled by cfg.seed.
        """
        return fit_directed_ddm(
            Path(mat_file),
            model=model if model is not None else self.cfg.model,
            stan_dir=Path(stan_dir) if stan_dir is not None else None,
            chains=int(fit_overrides.pop("chains", self.cfg.chains)),
            parallel_chains=int(
                fit_overrides.pop("parallel_chains", self.cfg.parallel_chains)
            ),
            iter_sampling=int(
                fit_overrides.pop("iter_sampling", self.cfg.iter_sampling)
            ),
            iter_warmup=int(fit_overrides.pop("iter_warmup", self.cfg.iter_warmup)),
            seed=int(seed if seed is not None else (self.cfg.seed or 2025)),
            show_console=bool(fit_overrides.pop("show_console", self.cfg.show_console)),
            **fit_overrides,
        )

    @staticmethod
    def draws_df(fit) -> pd.DataFrame:
        """
        CmdStanMCMC -> draws DataFrame.
        """
        return fit.draws_pd()


def create_workflow(
    cfg: WorkflowConfig = WorkflowConfig(),
) -> DirectedWorkflow:
    """
    Mirror integrative_model.create_workflow() naming:
    returns an object you can use for the end-to-end workflow.
    """
    return DirectedWorkflow(cfg=cfg)
