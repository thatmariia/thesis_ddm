# =====================================================================================
# Code for recovery plots copied from Michael Nunez. 
# Adapted for our purposes.
# Changes:
# - 

# =====================================================================================
# Import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# =====================================================================================
# Compute credible interval coverage
def compute_credible_interval_coverage(post_samples, true_values, level=0.95):
    lower_bound = (1.0 - level) / 2.0
    upper_bound = 1.0 - lower_bound
    coverage = []

    for i in range(post_samples.shape[0]):  # participant level
        ci_lower = np.quantile(post_samples[i, :], lower_bound)
        ci_upper = np.quantile(post_samples[i, :], upper_bound)
        coverage.append(ci_lower <= true_values[i] <= ci_upper)

    return np.mean(coverage)

# =====================================================================================
def recovery_plot(estimates, targets, fig_width=15, fig_height=9, parameter_display_titles=None, ax=None, is_all_coupling=False):
    """
    Parameter recovery plots: true vs. estimated.
    Includes median, mean, 95% and 99% credible intervals.

    Parameters
    ----------
    estimates : dict[str, np.ndarray]
        Posterior samples per parameter: shape (n_participants, n_samples)
    targets : dict[str, np.ndarray]
        True parameter values per parameter
    fig_width : int, default=15
        Figure width in inches
    fig_height : int, default=9
        Figure height in inches
    parameter_display_titles : dict[str, str], optional
        Mapping from parameter names to display titles. If None, uses default mapping.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates a new figure.
    is_all_coupling : bool, default=False
        If True, every parameter is a coupling parameter.
    """
    
    # Default parameter display title mapping
    if parameter_display_titles is None:
        parameter_display_titles = {
            'alpha': r'$\alpha$: Boundary separation',
            'tau': r'$\tau$: Non-decision time',
            'beta': r'$\beta$: Starting point/bias',
            'mu_z': r'$\mu_z$: Mean of latent variable',
            'eta': r'$\eta$: Variance of drift rate',
            'mu_delta': r'$\mu_{\delta}$: Mean of drift rate',
            'eta_delta': r'$\eta_{\delta}$: Variance of drift rate',
            'sigma': r'$\sigma$: S.d. of latent signal',
            'sigma_z': r'$\sigma_z$: S.d. of latent signal',
            'lambda': r'$\lambda$: Coupling strength',
            'gamma': r'$\gamma$: Coupling strength',
            'b': r'$b$: Baseline drift rate'
        }
    
    # Identify parameters: use order from parameter_display_titles if provided
    if parameter_display_titles is not None:
        params = [param for param in parameter_display_titles.keys() if param in estimates]
    else:
        params = list(estimates.keys())
    
    # Create figure
    if ax is None:
        fig = plt.figure(figsize=(fig_width, fig_height), tight_layout=True)
        columns = 3
        rows = int(np.ceil(len(params) / columns))
        axes_list = [fig.add_subplot(rows, columns, i + 1) for i in range(len(params))]
    else:
        # If a single axis is passed, wrap it in a list to match the expected format
        if isinstance(ax, plt.Axes):
            axes_list = [ax]
        else:
            axes_list = ax # assume iterable of axes
        fig = axes_list[0].figure

    # Plot properties
    LineWidths = np.array([2, 5])
    teal = np.array([0, .7, .7])
    blue = np.array([0, 0, 1])
    orange = np.array([1, .3, 0])
    Colors = [teal, blue]

    # Plot
    for i, param in enumerate(params):
        ax_to_use = axes_list[i] # Use corresponding axis 

        # Store handles for legend
        h_95, h_99, h_median, h_mean, h_line = None, None, None, None, None

        # Pre-compute posterior means and true values to identify NaN indices
        posterior_means_all = np.mean(estimates[param], axis=1).flatten()
        true_vals_all = targets[param].flatten()
        
        # Identify valid (non-NaN) indices
        nan_mask = np.isnan(true_vals_all) | np.isnan(posterior_means_all)
        valid_indices = np.where(~nan_mask)[0]
        n_nan = np.sum(nan_mask)
        n_total = len(true_vals_all)
        
        # Display name (needed early for logging)
        display_name = parameter_display_titles.get(param, param)
        
        if n_nan > 0:
            print(f"Note: {display_name} has {n_nan}/{n_total} NaN value(s), removed from analysis (using {n_total - n_nan} valid values)")

        # Track if we've set legend handles (for first valid point)
        first_valid = True

        for v in range(estimates[param].shape[0]):
            # Skip NaN participants
            if nan_mask[v]:
                continue
                
            bounds = stats.scoreatpercentile(estimates[param][v, :], (0.5, 2.5, 97.5, 99.5))

            # 95% and 99% CI
            for b in range(2):
                # Plot credible intervals
                credint = np.ones(100) * targets[param][v]
                y = np.linspace(bounds[b], bounds[-1 - b], 100)
                line = ax_to_use.plot(credint, y, color=Colors[b], linewidth=LineWidths[b])
                if first_valid:
                    if b == 0:
                        h_95 = line[0]
                    else:
                        h_99 = line[0]

            # Mark median
            mmedian = ax_to_use.plot(targets[param][v], np.median(estimates[param][v, :]), 'o', color=[0, 0, 0], markersize=10)
            if first_valid:
                h_median = mmedian[0]

            # Mark mean
            mmean = ax_to_use.plot(targets[param][v], np.mean(estimates[param][v, :]), '*', color=teal, markersize=10)
            if first_valid:
                h_mean = mmean[0]
                first_valid = False
        
        # Get valid values for metrics and y=x line
        true_vals_valid = true_vals_all[valid_indices]
        posterior_means_valid = posterior_means_all[valid_indices]
        
        # Line y = x (using valid values only)
        if len(true_vals_valid) > 0:
            tempx = np.linspace(np.min(true_vals_valid), np.max(true_vals_valid), num=100)
            recoverline = ax_to_use.plot(tempx, tempx, color=orange, linewidth=3)
            h_line = recoverline[0]

        # Compute correlation on valid values
        if len(true_vals_valid) > 1:
            r, _ = stats.pearsonr(true_vals_valid, posterior_means_valid)
            r_squared = r ** 2
        else:
            r_squared = np.nan

        ax_to_use.set_xlabel('True')
        ax_to_use.set_ylabel('Posterior')

        # Check if coupling parameters
        is_coupling = (param in ["gamma", "lambda"]) or is_all_coupling
        if is_coupling:
            # For coupling parameters: show coverage of zero and R²
            # Filter estimates for valid indices only
            estimates_valid = estimates[param][valid_indices, :]
            if len(valid_indices) > 0:
                coverage_zero = float(compute_interval_coverage(estimates_valid, target="low"))
                title_text = f"{display_name}\nCover 0 = {coverage_zero:.2f}, R² = {r_squared:.2f}"
                if n_nan > 0:
                    title_text += f" (n={len(valid_indices)})"
            else:
                title_text = f"{display_name}\nNo valid data"
        else:
            if len(true_vals_valid) > 1:
                # For non-coupling parameters: show R² and NRMSE
                mse = mean_squared_error(true_vals_valid, posterior_means_valid)
                rmse = np.sqrt(mse)
                range_true = np.max(true_vals_valid) - np.min(true_vals_valid)
                nrmse = rmse / range_true if range_true > 0 else np.nan
                title_text = f"{display_name}\nR² = {r_squared:.2f}, NRMSE = {nrmse:.3f}"
                if n_nan > 0:
                    title_text += f" (n={len(valid_indices)})"
            else:
                print(f"Warning: {display_name} has insufficient valid values after NaN removal")
                title_text = f"{display_name}\nR² = NaN, NRMSE = NaN"

        ax_to_use.set_title(title_text)

        # Add legend 
        if i == 0:
            ax_to_use.legend([h_95, h_99, h_median, h_mean, h_line], ['95% CI', '99% CI', 'Median', 'Mean', 'y = x'], loc='upper left')

    return fig

# =====================================================================================
# Compute interval coverage of zero for coupling parameters
def compute_interval_coverage(post_samples, low_range=None, high_range=None, target="low"):
    """
    Computes proportion of 95% credible intervals that cover zero.

    Parameters
    ----------
    post_samples : np.ndarray
        Posterior samples per participant: shape (n_participants, n_samples)
    low_range : tuple, optional
        Unused. Kept for backward compatibility.
    high_range : list of tuples, optional
        Unused. Kept for backward compatibility.
    target : str, default 'low'
        Only 'low' is supported. Other values will raise an error.

    Returns
    -------
    float
        Proportion of credible intervals that cover zero.
    """
    n_participants = post_samples.shape[0]
    lower_bounds = np.percentile(post_samples, 2.5, axis=1)
    upper_bounds = np.percentile(post_samples, 97.5, axis=1)

    if target == "low":
        covered = (lower_bounds <= 0) & (upper_bounds >= 0)
    else:
        raise ValueError("Only target='low' is supported. High-range coverage has been removed.")

    return float(np.mean(covered))

# =====================================================================================
# Compute recovery metrics
def compute_recovery_metrics(post_draws, val_sims):
    print("{:<10} {:>10} {:>10} {:>10} {:>10}".format(
        "Param", "R^2", "NRMSE", "95% Cov.", "99% Cov."
    ))

    for param_name, true_values in val_sims.items():
        samples = post_draws[param_name].squeeze()
        true_values = true_values.squeeze()
        
        if true_values is None or samples is None:
            continue
        
        posterior_means = samples.mean(axis=1)

        if np.any(np.isnan(posterior_means)) or np.any(np.isnan(true_values)):
            print(f"Skipping {param_name} because of NaN values")
            continue

        # R²
        r, _ = pearsonr(true_values, posterior_means)
        r2 = r ** 2

        # NRMSE
        mse = mean_squared_error(true_values, posterior_means)
        rmse = np.sqrt(mse)
        range_true = np.max(true_values) - np.min(true_values)
        nrmse = rmse / range_true if range_true > 0 else np.nan

        # Coverage
        coverage_95 = compute_credible_interval_coverage(samples, true_values, level=0.95)
        coverage_99 = compute_credible_interval_coverage(samples, true_values, level=0.99)

        # Coupling-specific interval coverage (only coverage of zero retained)
        low_cover = np.nan
        if "gamma" in param_name or "lambda" in param_name:
            low_cover = compute_interval_coverage(samples, target="low")

        # Unified print line
        if np.isnan(low_cover):
            print("{:<10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}".format(
                param_name, r2, nrmse, coverage_95, coverage_99
            ))
        else:
            print("{:<10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}   (Cover 0: {:>5.3f})".format(
                param_name, r2, nrmse, coverage_95, coverage_99, low_cover
            ))