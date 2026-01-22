from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import bayesflow as bf
from bayesflow.adapters import Adapter
from bayesflow.simulators import make_simulator

from .simulation import prior, likelihood


@dataclass(frozen=True)
class WorkflowConfig:
    n_obs_min: int = 30
    n_obs_max: int = 1000  # inclusive
    summary_dim: int = 8


def default_meta_fn(cfg: WorkflowConfig) -> dict[str, int]:
    # np.random.randint high is exclusive -> use max+1 to make inclusive
    n_obs = int(np.random.randint(cfg.n_obs_min, cfg.n_obs_max + 1))
    return {"n_obs": n_obs}


def create_workflow(
    cfg: WorkflowConfig = WorkflowConfig(),
    meta_fn: Callable[[], dict[str, int]] | None = None,
) -> tuple[bf.BasicWorkflow, bf.simulators.Simulator, Adapter]:
    """
    Creates and configures the BayesFlow workflow for the integrative DDM.
    Returns (workflow, simulator, adapter).
    """
    if cfg.n_obs_min < 1 or cfg.n_obs_max < cfg.n_obs_min:
        raise ValueError(f"Invalid n_obs range: [{cfg.n_obs_min}, {cfg.n_obs_max}]")

    if meta_fn is None:
        meta_fn = lambda: default_meta_fn(cfg)

    simulator = make_simulator([prior, likelihood], meta_fn=meta_fn)

    # Networks
    summary_network = bf.networks.SetTransformer(summary_dim=cfg.summary_dim)
    inference_network = bf.networks.CouplingFlow()

    # Adapter pipeline
    adapter = (
        Adapter()
        .broadcast("n_obs", to="choicert")
        .as_set(["choicert", "z"])
        .standardize(exclude=["n_obs"])
        .convert_dtype("float64", "float32")
        .concatenate(
            ["alpha", "tau", "beta", "mu_delta", "eta_delta", "gamma", "sigma"],
            into="inference_variables",
        )
        .concatenate(["choicert", "z"], into="summary_variables")
        .rename("n_obs", "inference_conditions")
    )

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=inference_network,
        summary_network=summary_network,
    )

    return workflow, simulator, adapter
