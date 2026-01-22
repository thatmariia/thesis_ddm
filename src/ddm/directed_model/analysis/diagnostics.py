from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_PARAM_RE = re.compile(r"([a-zA-Z_]+)\[(\d+)\]")


def check_convergence(
    summary_df: pd.DataFrame,
    rhat_thresh: float = 1.01,
    ess_thresh: float = 400.0,
) -> dict[str, pd.DataFrame]:
    """
    Check CmdStan summary output for convergence red flags.

    Parameters
    ----------
    summary_df:
        Typically `fit.summary()` from CmdStanPy.
    rhat_thresh:
        Threshold above which R_hat is flagged.
    ess_thresh:
        Threshold below which ESS_bulk or ESS_tail is flagged.

    Returns
    -------
    dict with keys:
        - "rhat_issues": rows with R_hat > rhat_thresh
        - "ess_issues": rows with ESS_bulk or ESS_tail < ess_thresh
    """
    if "R_hat" not in summary_df.columns:
        raise KeyError(
            "summary_df is missing column 'R_hat' (is this CmdStanPy summary output?)"
        )
    if "ESS_bulk" not in summary_df.columns or "ESS_tail" not in summary_df.columns:
        raise KeyError(
            "summary_df is missing ESS columns (expected 'ESS_bulk' and 'ESS_tail')."
        )

    rhat_issues = summary_df[summary_df["R_hat"] > rhat_thresh]
    ess_issues = summary_df[
        (summary_df["ESS_bulk"] < ess_thresh) | (summary_df["ESS_tail"] < ess_thresh)
    ]

    # Keep old behavior: print useful info
    if not rhat_issues.empty:
        print(f"\nParameters with R-hat > {rhat_thresh}:")
        print(rhat_issues[["R_hat"]])
    else:
        print(f"\nAll parameters passed R-hat < {rhat_thresh}")

    if not ess_issues.empty:
        print(f"\nParameters with ESS < {ess_thresh}:")
        print(ess_issues[["ESS_bulk", "ESS_tail"]])
    else:
        print(f"\nAll parameters passed ESS > {ess_thresh}")

    return {"rhat_issues": rhat_issues, "ess_issues": ess_issues}


def plot_trace_grids(
    posterior_df: pd.DataFrame,
    fit: Any,
    params_of_interest: Iterable[str] = (
        "alpha",
        "tau",
        "beta",
        "eta",
        "mu_z",
        "sigma_z",
        "lambda",
        "lambda_param",
        "b",
    ),
    grid_cols: int = 5,
) -> dict[str, plt.Figure]:
    """
    Plot trace plots arranged in grids for each parameter group, e.g. alpha[1], alpha[2], ...

    Parameters
    ----------
    posterior_df:
        DataFrame of posterior draws (CmdStanPy's `fit.draws_pd()`).
    fit:
        CmdStanMCMC object (used to get chain count via `fit.chains`).
    params_of_interest:
        Base parameter names to include.
    grid_cols:
        Number of columns in each subplot grid.

    Returns
    -------
    dict mapping base param name -> matplotlib Figure.
    """
    if grid_cols < 1:
        raise ValueError("grid_cols must be >= 1")

    num_chains = int(getattr(fit, "chains", 1))
    param_cols = posterior_df.columns.tolist()

    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)

    for col in param_cols:
        m = _PARAM_RE.match(col)
        if not m:
            continue
        base, idx = m.groups()
        if base in set(params_of_interest):
            grouped[base].append((int(idx), col))

    figures: dict[str, plt.Figure] = {}

    for base, items in grouped.items():
        items.sort(key=lambda t: t[0])
        cols = [c for _, c in items]
        n = len(cols)
        if n == 0:
            continue

        grid_rows = math.ceil(n / grid_cols)
        fig, axes = plt.subplots(
            grid_rows,
            grid_cols,
            figsize=(grid_cols * 4, grid_rows * 2),
            sharex=True,
        )
        axes_arr = np.atleast_1d(axes).flatten()

        for i, col in enumerate(cols):
            ax = axes_arr[i]
            vals = posterior_df[col].values

            # Expect CmdStanPy draws_pd layout: all draws stacked; reshape into chains if possible
            if vals.size % num_chains != 0:
                # Fallback: just plot raw series
                ax.plot(vals, alpha=0.6)
            else:
                vals2 = vals.reshape(num_chains, -1)
                for chain in range(num_chains):
                    ax.plot(vals2[chain], alpha=0.6, label=f"Chain {chain + 1}")

            ax.axhline(
                float(posterior_df[col].mean()),
                color="black",
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
            )
            ax.set_title(col, fontsize=9)
            ax.tick_params(labelsize=6)

        for j in range(n, len(axes_arr)):
            axes_arr[j].axis("off")

        fig.text(0.5, 0.04, "Iteration", ha="center")
        fig.text(0.04, 0.5, "Parameter Value", va="center", rotation="vertical")
        fig.suptitle(f"Trace Plots for '{base}'", fontsize=14)
        fig.tight_layout(rect=[0.04, 0.04, 1, 0.96])

        handles, labels = axes_arr[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")

        figures[base] = fig

    return figures
