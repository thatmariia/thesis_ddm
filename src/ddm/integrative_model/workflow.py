from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import bayesflow as bf
from bayesflow.adapters import Adapter
from bayesflow.simulators import make_simulator

from ..utils import make_rng
# from .simulation import prior as base_prior
# from .simulation import likelihood as base_likelihood

from .simulation.priors import (
    sample_prior_dict,
    IntegrativeParams,
    PriorConfig as IntegrativePriorConfig,
)
from .simulation.likelihood import likelihood as likelihood_from_params


@dataclass(frozen=True)
class WorkflowConfig:
    n_obs_min: int = 30
    n_obs_max: int = 1000  # inclusive
    summary_dim: int = 8
    prior: IntegrativePriorConfig = IntegrativePriorConfig()


def create_workflow(
    cfg: WorkflowConfig = WorkflowConfig(),
    *,
    seed: int | None = None,
    meta_fn: Callable[[], dict[str, int]] | None = None,
) -> tuple[bf.BasicWorkflow, bf.simulators.Simulator, Adapter]:
    """
    Creates and configures the BayesFlow workflow for the integrative DDM.

    Reproducibility policy:
    - If seed is provided, all generated datasets (prior + likelihood + variable n_obs) are reproducible.
    - We do NOT add seed to the data; it's only used internally to control Numba RNG safely.
    """
    if cfg.n_obs_min < 1 or cfg.n_obs_max < cfg.n_obs_min:
        raise ValueError(f"Invalid n_obs range: [{cfg.n_obs_min}, {cfg.n_obs_max}]")

    rng = make_rng(seed)

    # meta: reproducible n_obs if seed provided
    if meta_fn is None:
        def meta_fn() -> dict[str, int]:
            n_obs = int(rng.integers(cfg.n_obs_min, cfg.n_obs_max + 1))
            return {"n_obs": n_obs}

    def prior_wrapped() -> dict[str, float]:
        # BayesFlow wants dicts, so return dict here
        return sample_prior_dict(cfg=cfg.prior, rng=rng)

    # Wrap likelihood to:
    # - draw a deterministic seed per dataset from rng
    # - use it to isolate Numba randomness
    def likelihood_wrapped(
        alpha: float,
        tau: float,
        beta: float,
        mu_delta: float,
        eta_delta: float,
        gamma: float,
        sigma: float,
        n_obs: int,
    ) -> dict[str, np.ndarray]:
        sim_seed = int(rng.integers(0, 2**32 - 1))
        params = IntegrativeParams(
            alpha=alpha,
            tau=tau,
            beta=beta,
            mu_delta=mu_delta,
            eta_delta=eta_delta,
            gamma=gamma,
            sigma=sigma,
        )
        return likelihood_from_params(params=params, n_obs=int(n_obs), seed=sim_seed)

    simulator = make_simulator([prior_wrapped, likelihood_wrapped], meta_fn=meta_fn)

    # Networks
    summary_network = bf.networks.SetTransformer(summary_dim=cfg.summary_dim)
    inference_network = bf.networks.CouplingFlow()

    # Adapter pipeline (unchanged)
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
