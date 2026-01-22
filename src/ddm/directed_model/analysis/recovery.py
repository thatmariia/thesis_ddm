from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_parameter_samples(
    df: pd.DataFrame, param_name: str, n_participants: int
) -> np.ndarray:
    """
    Extract and reshape posterior samples for a participant-indexed parameter.

    Matches columns like:
        alpha[1], alpha[2], ... alpha[n]

    Returns array shaped:
        (n_participants, n_draws)

    Notes
    -----
    This replicates the original behavior: columns are sorted by participant index.
    """
    cols = [c for c in df.columns if c.startswith(f"{param_name}[")]
    if not cols:
        raise KeyError(
            f"No columns found for parameter '{param_name}[' in posterior DataFrame"
        )

    def _idx(col: str) -> int:
        m = re.search(r"\[(\d+)\]", col)
        if not m:
            raise ValueError(f"Column '{col}' does not look like '{param_name}[i]'")
        return int(m.group(1))

    cols.sort(key=_idx)

    # df[cols] -> (n_draws, n_participants), transpose -> (n_participants, n_draws)
    samples = df[cols].to_numpy().T

    if samples.shape[0] != n_participants:
        # Don't hard-fail; but warn loudly because this indicates mismatch in participant indexing.
        raise ValueError(
            f"Expected {n_participants} participant columns for '{param_name}', "
            f"but found {samples.shape[0]}. Check nparts and Stan indexing."
        )

    return samples


def plot_recovery(
    true_vals: np.ndarray | float,
    estimated_vals: np.ndarray | float,
    param_name: str,
    ci_lower: np.ndarray | None = None,
    ci_upper: np.ndarray | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (7, 7),
) -> plt.Figure:
    true_vals = np.atleast_1d(true_vals).astype(float)
    est_vals = np.atleast_1d(estimated_vals).astype(float)

    if true_vals.shape != est_vals.shape:
        raise ValueError(
            f"true_vals shape {true_vals.shape} != estimated_vals shape {est_vals.shape}"
        )

    corr = (
        float(np.corrcoef(true_vals, est_vals)[0, 1])
        if true_vals.size > 1
        else float("nan")
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if ci_lower is not None and ci_upper is not None:
        ci_lower = np.atleast_1d(ci_lower).astype(float)
        ci_upper = np.atleast_1d(ci_upper).astype(float)
        if ci_lower.shape != est_vals.shape or ci_upper.shape != est_vals.shape:
            raise ValueError("CI arrays must match estimated_vals shape")

        xerr = np.vstack([est_vals - ci_lower, ci_upper - est_vals])
        ax.errorbar(est_vals, true_vals, xerr=xerr, fmt="o", alpha=0.8)
    else:
        ax.scatter(true_vals, est_vals, alpha=0.8)
        if true_vals.size > 1:
            m, b = np.polyfit(true_vals, est_vals, 1)
            xs = np.linspace(true_vals.min(), true_vals.max(), 100)
            ax.plot(xs, m * xs + b, linestyle="--")

    ax.set_xlabel(f"True {param_name}")
    ax.set_ylabel(f"Estimated {param_name}")
    ax.set_title(f"Parameter Recovery: {param_name} (r = {corr:.2f})")
    ax.grid(False)
    fig.tight_layout()
    return fig