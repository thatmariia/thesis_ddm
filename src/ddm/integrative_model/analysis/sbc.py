from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

from bayesflow.utils import prepare_plot_data, add_titles_and_labels, prettify_subplots

from ...utils import optional_import


def calibration_histogram(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] | None = None,
    variable_names: Sequence[str] | None = None,
    figsize: Sequence[float] | None = None,
    num_bins: int = 10,
    binomial_interval: float = 0.99,
    label_fontsize: int = 14,
    title_fontsize: int = 18,
    tick_fontsize: int = 12,
    color: str = "teal",
    num_col: int | None = None,
    num_row: int | None = None,
    skip_indices: set[int] | None = None,
) -> plt.Figure:
    """
    SBC rank histograms.

    Use `skip_indices={7}` if you really want to skip the 8th plot.
    """
    if skip_indices is None:
        skip_indices = set()

    plot_data = prepare_plot_data(
        estimates=estimates,
        targets=targets,
        variable_keys=variable_keys,
        variable_names=variable_names,
        num_col=num_col,
        num_row=num_row,
        figsize=figsize,
    )

    estimates_arr = plot_data.pop("estimates")
    targets_arr = plot_data.pop("targets")

    num_sims = estimates_arr.shape[0]
    num_draws = estimates_arr.shape[1]
    ratio = int(num_sims / num_draws) if num_draws else 0

    if num_bins is None:
        num_bins = int(ratio / 2) if ratio else 10
        if num_bins == 1:
            num_bins = 4

    ranks = np.sum(estimates_arr < targets_arr[:, np.newaxis, :], axis=1)

    num_trials = int(targets_arr.shape[0])
    endpoints = binom.interval(binomial_interval, num_trials, 1 / num_bins)
    mean = num_trials / num_bins

    sns = optional_import("seaborn")
    use_sns = sns is not None

    for j, ax in enumerate(plot_data["axes"].flat):
        if j in skip_indices or j >= ranks.shape[1]:
            ax.set_visible(False)
            continue

        ax.axhspan(endpoints[0], endpoints[1], facecolor="gray", alpha=0.3)
        ax.axhline(mean, color="gray", zorder=0, alpha=0.9)

        if use_sns:
            sns.histplot(
                ranks[:, j], kde=False, ax=ax, color=color, bins=num_bins, alpha=0.95
            )
        else:
            ax.hist(ranks[:, j], bins=num_bins, color=color, alpha=0.95)

        ax.set_ylabel("")
        ax.get_yaxis().set_ticks([])
        ax.grid(False)

    prettify_subplots(plot_data["axes"], tick_fontsize)

    add_titles_and_labels(
        axes=plot_data["axes"],
        num_row=plot_data["num_row"],
        num_col=plot_data["num_col"],
        title=plot_data["variable_names"],
        xlabel="Rank Statistic",
        ylabel="Number of Simulations",
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
    )

    plot_data["fig"].tight_layout()
    return plot_data["fig"]
