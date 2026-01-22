from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..simulation import simul_directed_ddm
from ...utils import optional_import


@dataclass(frozen=True)
class PPCCondition:
    """One condition for PPC."""

    name: str
    y: np.ndarray
    z: np.ndarray
    participants: np.ndarray


@dataclass(frozen=True)
class PPCResult:
    """Observed vs predicted per condition."""

    name: str
    observed_y: np.ndarray
    observed_z: np.ndarray
    predicted_y: np.ndarray
    predicted_z: np.ndarray


def _require_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise KeyError(
            f"Missing posterior column '{col}'. Check Stan naming and participant IDs."
        )


def canonicalize_posterior_columns(posterior_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize posterior column naming to Python-side conventions.
    - Stan uses lambda[...] but Python prefers lambda_param[...]
    """
    # avoid copying unless needed
    if any(c.startswith("lambda[") for c in posterior_df.columns):
        rename_map = {
            c: c.replace("lambda[", "lambda_param[")
            for c in posterior_df.columns
            if c.startswith("lambda[")
        }
        posterior_df = posterior_df.rename(columns=rename_map)
    return posterior_df


def _parse_indexed_param_columns(df: pd.DataFrame, base: str) -> dict[int, str]:
    """
    Map participant id -> column name for columns like base[1], base[2], ...
    """
    out: dict[int, str] = {}
    prefix = f"{base}["
    for c in df.columns:
        if not c.startswith(prefix):
            continue
        # fast parse: base[123]
        try:
            pid = int(c[len(prefix) : c.index("]")])
        except Exception as e:
            raise ValueError(f"Bad indexed column format: {c}") from e
        out[pid] = c
    return out


def posterior_means_by_participant(
    posterior_df: pd.DataFrame,
    params: Sequence[str],
) -> dict[str, dict[int, float]]:
    """
    Compute posterior means for per-participant params.
    Returns: means[param][pid] = float
    """
    means: dict[str, dict[int, float]] = {}
    for p in params:
        colmap = _parse_indexed_param_columns(posterior_df, p)
        if not colmap:
            raise KeyError(f"No posterior columns found for '{p}[i]'")
        means[p] = {pid: float(posterior_df[col].mean()) for pid, col in colmap.items()}
    return means


def predict_trials_plugin_means(
    posterior_df: pd.DataFrame,
    participants: Sequence[int | str],
    n_trials: int,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict y,z by simulating one synthetic dataset using plug-in posterior means.

    Faster version:
    - precomputes participant-level posterior means once
    - simulates in batches per participant
    - returns predictions in the original trial order
    """
    if n_trials != len(participants):
        raise ValueError("n_trials must equal len(participants)")

    posterior_df = canonicalize_posterior_columns(posterior_df)

    # force participant ids to int for dict indexing
    pids = np.asarray(participants, dtype=int)

    needed = ["alpha", "tau", "beta", "eta", "mu_z", "sigma_z", "lambda_param", "b"]
    means = posterior_means_by_participant(posterior_df, needed)

    y_pred = np.empty(n_trials, dtype=float)
    z_pred = np.empty(n_trials, dtype=float)

    rng = np.random.default_rng(seed) if seed is not None else None

    # simulate per participant in one call
    unique_pids = np.unique(pids)
    for pid in unique_pids:
        idx = np.where(pids == pid)[0]
        n_p = int(idx.size)
        if n_p == 0:
            continue

        sim_y, _, sim_z = simul_directed_ddm(
            n_trials=n_p,
            alpha=means["alpha"][pid],
            tau=means["tau"][pid],
            beta=means["beta"][pid],
            eta=means["eta"][pid],
            mu_z=means["mu_z"][pid],
            sigma_z=means["sigma_z"][pid],
            lambda_param=means["lambda_param"][pid],
            b=means["b"][pid],
            rng=rng,
        )

        y_pred[idx] = sim_y
        z_pred[idx] = sim_z

    return y_pred, z_pred


def run_ppc(
    posterior_df: pd.DataFrame,
    conditions: Sequence[PPCCondition],
    *,
    seed: int | None = None,
) -> list[PPCResult]:
    """
    Run plug-in-mean PPC for multiple conditions.
    """
    results: list[PPCResult] = []
    for cond in conditions:
        y_pred, z_pred = predict_trials_plugin_means(
            posterior_df=posterior_df,
            participants=cond.participants,
            n_trials=len(cond.z),
            seed=seed,
        )
        results.append(
            PPCResult(
                name=cond.name,
                observed_y=np.asarray(cond.y),
                observed_z=np.asarray(cond.z),
                predicted_y=y_pred,
                predicted_z=z_pred,
            )
        )
    return results


def ppc_summary_metrics(result: PPCResult) -> dict[str, float]:
    """
    Replicates your old printed summary metrics (conditional RT by sign + z mean/var).
    """
    y_obs = result.observed_y
    y_pred = result.predicted_y

    rt_obs_neg = np.abs(y_obs[y_obs < 0])
    rt_obs_pos = y_obs[y_obs > 0]
    rt_pred_neg = np.abs(y_pred[y_pred < 0])
    rt_pred_pos = y_pred[y_pred > 0]

    def _mean(x: np.ndarray) -> float:
        return float(np.mean(x)) if x.size else float("nan")

    def _var(x: np.ndarray) -> float:
        return float(np.var(x)) if x.size else float("nan")

    return {
        "mean_rt_obs_neg": _mean(rt_obs_neg),
        "var_rt_obs_neg": _var(rt_obs_neg),
        "mean_rt_pred_neg": _mean(rt_pred_neg),
        "var_rt_pred_neg": _var(rt_pred_neg),
        "mean_rt_obs_pos": _mean(rt_obs_pos),
        "var_rt_obs_pos": _var(rt_obs_pos),
        "mean_rt_pred_pos": _mean(rt_pred_pos),
        "var_rt_pred_pos": _var(rt_pred_pos),
        "mean_z_obs": float(np.mean(result.observed_z)),
        "var_z_obs": float(np.var(result.observed_z)),
        "mean_z_pred": float(np.mean(result.predicted_z)),
        "var_z_pred": float(np.var(result.predicted_z)),
    }


def plot_ppc(
    results: Sequence[PPCResult],
    bins: int = 40,
    use_seaborn: bool = True,
) -> plt.Figure:
    """
    Plot observed vs predicted distributions for y and z.

    If seaborn is installed and use_seaborn=True, will mimic your original hist+kde style.
    Otherwise uses matplotlib hist only (no seaborn dependency).
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns = optional_import("seaborn") if use_seaborn else None

    for res in results:
        if sns is not None:
            sns.histplot(
                res.observed_y,
                stat="density",
                alpha=0.6,
                ax=axes[0],
                label=f"Obs {res.name}",
            )
            sns.kdeplot(res.predicted_y, ax=axes[0], label=f"Pred {res.name}")
            sns.histplot(
                res.observed_z,
                stat="density",
                alpha=0.6,
                ax=axes[1],
                label=f"Obs {res.name}",
            )
            sns.kdeplot(res.predicted_z, ax=axes[1], label=f"Pred {res.name}")
        else:
            axes[0].hist(
                res.observed_y,
                bins=bins,
                density=True,
                alpha=0.35,
                label=f"Obs: {res.name}",
            )
            axes[0].hist(
                res.predicted_y,
                bins=bins,
                density=True,
                alpha=0.35,
                label=f"Pred: {res.name}",
            )
            axes[1].hist(
                res.observed_z,
                bins=bins,
                density=True,
                alpha=0.35,
                label=f"Obs: {res.name}",
            )
            axes[1].hist(
                res.predicted_z,
                bins=bins,
                density=True,
                alpha=0.35,
                label=f"Pred: {res.name}",
            )

    axes[0].set_title("Posterior predictive: y (signed RT)")
    axes[0].set_xlabel("Signed RT")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=9)

    axes[1].set_title("Posterior predictive: z (latent)")
    axes[1].set_xlabel("z")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    return fig


def posterior_predictive_check_baseline(
    posterior_df: pd.DataFrame,
    participants: np.ndarray,
    true_y: np.ndarray,
    true_z: np.ndarray,
    conditions_data: list[dict[str, Any]] | None = None,
    use_seaborn: bool = True,
    *,
    seed: int | None = None,
) -> plt.Figure:
    """
    Replicates your baseline PPC function:
    - If conditions_data is None/empty -> in-sample baseline only
    - Else loops through condition dicts and overlays.

    Each condition dict expects keys:
      - condition_y
      - condition_z
      - condition_participants
      - condition_name
    """
    if conditions_data is None or len(conditions_data) == 0:
        conditions_data = [
            {
                "condition_y": true_y,
                "condition_z": true_z,
                "condition_participants": participants,
                "condition_name": "In-Sample (Baseline)",
            }
        ]

    conditions: list[PPCCondition] = []
    for c in conditions_data:
        conditions.append(
            PPCCondition(
                name=str(c["condition_name"]),
                y=np.asarray(c["condition_y"]),
                z=np.asarray(c["condition_z"]),
                participants=np.asarray(c["condition_participants"]),
            )
        )

    results = run_ppc(posterior_df=posterior_df, conditions=conditions, seed=seed)
    fig = plot_ppc(results, use_seaborn=use_seaborn)

    # Replicate print summary block
    print(f"\n{'=' * 60}\nPOSTERIOR PREDICTIVE CHECK SUMMARY\n{'=' * 60}")
    for res in results:
        metrics = ppc_summary_metrics(res)
        print(f"\n{res.name} - PPC Metrics:")
        print("  Conditional RTs:")
        print(
            f"  Neg choices    mean RT obs = {metrics['mean_rt_obs_neg']:.2f}, pred = {metrics['mean_rt_pred_neg']:.2f}"
        )
        print(
            f"                 var RT obs = {metrics['var_rt_obs_neg']:.2f}, var RT pred = {metrics['var_rt_pred_neg']:.2f}"
        )
        print(
            f"  Pos choices    mean RT obs = {metrics['mean_rt_obs_pos']:.2f}, pred = {metrics['mean_rt_pred_pos']:.2f}"
        )
        print(
            f"                 var RT obs = {metrics['var_rt_obs_pos']:.2f}, var RT pred = {metrics['var_rt_pred_pos']:.2f}"
        )
        print(
            f"  z (Latent):    mean obs = {metrics['mean_z_obs']:.2f}, mean pred = {metrics['mean_z_pred']:.2f}"
        )
        print(
            f"                 var obs = {metrics['var_z_obs']:.2f}, var pred = {metrics['var_z_pred']:.2f}"
        )

    return fig


def posterior_predictive_check_comprehensive(
    posterior_df: pd.DataFrame,
    participants: np.ndarray,
    true_y: np.ndarray,
    true_z: np.ndarray,
    true_params: dict[str, np.ndarray],
    nparts: int,
    oos_seed: int = 5202,
    use_seaborn: bool = True,
) -> plt.Figure:
    train_y = np.asarray(true_y)
    train_z = np.asarray(true_z)
    train_participants = np.asarray(participants)

    print("Generating new out-of-sample data using true parameters...")

    rng = np.random.default_rng(oos_seed)

    test_y_list: list[float] = []
    test_z_list: list[float] = []
    test_p_list: list[int] = []

    lambda_key = "lambda_param" if "lambda_param" in true_params else "lambda"

    for p in range(nparts):
        n_trials_p = int(np.sum(train_participants == (p + 1)))
        sim_y, _, sim_z = simul_directed_ddm(
            n_trials=n_trials_p,
            alpha=float(true_params["alpha"][p]),
            tau=float(true_params["tau"][p]),
            beta=float(true_params["beta"][p]),
            eta=float(true_params["eta"][p]),
            lambda_param=float(true_params[lambda_key][p]),
            mu_z=float(true_params["mu_z"][p]),
            sigma_z=float(true_params["sigma_z"][p]),
            b=float(true_params["b"][p]),
            rng=rng,
        )

        test_y_list.extend([float(v) for v in sim_y])
        test_z_list.extend([float(v) for v in sim_z])
        test_p_list.extend([p + 1] * n_trials_p)

    test_y = np.asarray(test_y_list, dtype=float)
    test_z = np.asarray(test_z_list, dtype=float)
    test_participants = np.asarray(test_p_list, dtype=int)

    print(f"Generated {len(test_y)} new out-of-sample trials")

    # --- Predictions using posterior means ---
    pred_y_train, pred_z_train = predict_trials_plugin_means(
        posterior_df=posterior_df,
        participants=train_participants,
        n_trials=len(train_z),
        seed=oos_seed + 1,
    )
    pred_y_test, pred_z_test = predict_trials_plugin_means(
        posterior_df=posterior_df,
        participants=test_participants,
        n_trials=len(test_z),
        seed=oos_seed + 2,
    )

    # Plot: 2x2
    sns = optional_import("seaborn") if use_seaborn else None

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    if sns is not None:
        sns.histplot(train_y, stat="density", ax=axes[0, 0], label="Observed")
        sns.kdeplot(pred_y_train, ax=axes[0, 0], label="Predicted")
        sns.histplot(train_z, stat="density", ax=axes[0, 1], label="Observed")
        sns.kdeplot(pred_z_train, ax=axes[0, 1], label="Predicted")

        sns.histplot(test_y, stat="density", ax=axes[1, 0], label="Observed")
        sns.kdeplot(pred_y_test, ax=axes[1, 0], label="Predicted")
        sns.histplot(test_z, stat="density", ax=axes[1, 1], label="Observed")
        sns.kdeplot(pred_z_test, ax=axes[1, 1], label="Predicted")
    else:
        axes[0, 0].hist(train_y, bins=40, density=True, alpha=0.5, label="Observed")
        axes[0, 0].hist(
            pred_y_train, bins=40, density=True, alpha=0.5, label="Predicted"
        )
        axes[0, 1].hist(train_z, bins=40, density=True, alpha=0.5, label="Observed")
        axes[0, 1].hist(
            pred_z_train, bins=40, density=True, alpha=0.5, label="Predicted"
        )

        axes[1, 0].hist(test_y, bins=40, density=True, alpha=0.5, label="Observed")
        axes[1, 0].hist(
            pred_y_test, bins=40, density=True, alpha=0.5, label="Predicted"
        )
        axes[1, 1].hist(test_z, bins=40, density=True, alpha=0.5, label="Observed")
        axes[1, 1].hist(
            pred_z_test, bins=40, density=True, alpha=0.5, label="Predicted"
        )

    axes[0, 0].set_title("In-Sample Choice/RT (y)")
    axes[0, 1].set_title("In-Sample Latent Variable (z)")
    axes[1, 0].set_title("Out-of-Sample Choice/RT (y)")
    axes[1, 1].set_title("Out-of-Sample Latent Variable (z)")

    for ax in axes.flatten():
        ax.grid(False)
        ax.legend()

    fig.tight_layout()

    summary = {
        "in_sample": {
            "mean_y_obs": float(np.mean(train_y)),
            "mean_y_pred": float(np.mean(pred_y_train)),
            "var_y_obs": float(np.var(train_y)),
            "var_y_pred": float(np.var(pred_y_train)),
            "mean_z_obs": float(np.mean(train_z)),
            "mean_z_pred": float(np.mean(pred_z_train)),
            "var_z_obs": float(np.var(train_z)),
            "var_z_pred": float(np.var(pred_z_train)),
        },
        "out_of_sample": {
            "mean_y_obs": float(np.mean(test_y)),
            "mean_y_pred": float(np.mean(pred_y_test)),
            "var_y_obs": float(np.var(test_y)),
            "var_y_pred": float(np.var(pred_y_test)),
            "mean_z_obs": float(np.mean(test_z)),
            "mean_z_pred": float(np.mean(pred_z_test)),
            "var_z_obs": float(np.var(test_z)),
            "var_z_pred": float(np.var(pred_z_test)),
        },
    }

    print("In-sample metrics:", summary["in_sample"])
    print("Out-of-sample metrics:", summary["out_of_sample"])

    return fig
