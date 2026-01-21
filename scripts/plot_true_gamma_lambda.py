#!/usr/bin/env python3
"""
Plot true lambda (from directed model) vs true gamma (from integrative model) for all conditions
using the recovery plot function to visualize the relationship between these parameters.
Each condition is plotted in a separate subplot.
"""

# Import modules
import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# PROJECT_ROOT = scripts/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Add project root to Python path so 'shared' is found
sys.path.append(str(PROJECT_ROOT))

# =====================================================================================
# Helper function to create subplots with grouped axis sharing
def create_grouped_subplots(n_subplots, cols=3, figsize=(15, 9), share_groups=None):
    """
    Create a figure with subplots where only specified groups share axes.
    
    Parameters
    ----------
    n_subplots : int
        Total number of subplots
    cols : int, default=3
        Number of columns
    figsize : tuple, default=(15, 9)
        Figure size
    share_groups : list of lists, optional
        Each inner list contains indices of subplots that should share axes.
        e.g., [[0, 1, 2], [6, 7, 8]] means subplots 0-2 share axes, 6-8 share axes.
        Subplots not in any group have independent axes.
    
    Returns
    -------
    fig : Figure
    axes : array of Axes
    sync_axes : callable
        Call this after plotting to synchronize axes within groups.
    """
    rows = int(np.ceil(n_subplots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=False, sharey=False)
    axes = axes.flatten()
    
    def sync_axes():
        """Synchronize axis limits within each share group after plotting."""
        if share_groups is None:
            return
        for group in share_groups:
            indices = [i for i in group if i < len(axes)]
            if len(indices) < 2:
                continue
            
            # Find the combined limits for this group
            xlims = [axes[i].get_xlim() for i in indices]
            ylims = [axes[i].get_ylim() for i in indices]
            
            xmin = min(lim[0] for lim in xlims)
            xmax = max(lim[1] for lim in xlims)
            ymin = min(lim[0] for lim in ylims)
            ymax = max(lim[1] for lim in ylims)
            
            # Apply to all axes in the group
            for i in indices:
                axes[i].set_xlim(xmin, xmax)
                axes[i].set_ylim(ymin, ymax)
    
    return fig, axes, sync_axes

# =====================================================================================
# Setup paths
DIRECTED_DATA_DIR = PROJECT_ROOT / 'directed_model' / 'data_new_sigma_z_cross'
INTEGRATIVE_DATA_DIR = PROJECT_ROOT / 'integrative_model' / 'data_new_sigma_new_conditions'
OUTPUT_DIR = PROJECT_ROOT / 'shared' / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

# =====================================================================================
print("Loading directed and integrative model data for lambda vs gamma plotting...")

# Find all directed model files
directed_files = sorted(DIRECTED_DATA_DIR.glob("ddmdata_*.mat"))
if not directed_files:
    print("No directed model files found!")
    sys.exit(1)

print(f"Found {len(directed_files)} directed model files")

# =====================================================================================
# Define condition display titles
condition_display_titles = {
    'SNR_low_COUP_low_DIST_gaussian': r'Gaussian, Low SNR, Low Coupling',
    'SNR_low_COUP_low_DIST_laplace': r'Laplace, Low SNR, Low Coupling',
    'SNR_low_COUP_low_DIST_uniform': r'Uniform, Low SNR, Low Coupling',

    'SNR_high_COUP_low_DIST_gaussian': r'Gaussian, High SNR, Low Coupling',
    'SNR_high_COUP_low_DIST_laplace': r'Laplace, High SNR, Low Coupling',
    'SNR_high_COUP_low_DIST_uniform': r'Uniform, High SNR, Low Coupling',

    'SNR_no_noise_COUP_low_DIST_gaussian': r'Gaussian, No Noise, Low Coupling',
    'SNR_no_noise_COUP_low_DIST_laplace': r'Laplace, No Noise, Low Coupling',
    'SNR_no_noise_COUP_low_DIST_uniform': r'Uniform, No Noise, Low Coupling',

    'SNR_low_COUP_high_DIST_gaussian': r'Gaussian, Low SNR, High Coupling',
    'SNR_low_COUP_high_DIST_laplace': r'Laplace, Low SNR, High Coupling',
    'SNR_low_COUP_high_DIST_uniform': r'Uniform, Low SNR, High Coupling',

    'SNR_high_COUP_high_DIST_gaussian': r'Gaussian, High SNR, High Coupling',
    'SNR_high_COUP_high_DIST_laplace': r'Laplace, High SNR, High Coupling',
    'SNR_high_COUP_high_DIST_uniform': r'Uniform, High SNR, High Coupling',
    
    'SNR_no_noise_COUP_high_DIST_gaussian': r'Gaussian, No Noise, High Coupling',
    'SNR_no_noise_COUP_high_DIST_laplace': r'Laplace, No Noise, High Coupling',
    'SNR_no_noise_COUP_high_DIST_uniform': r'Uniform, No Noise, High Coupling',
}

# =====================================================================================
# Define LOW coupling conditions (Figure 1) - 9 conditions
low_coupling_order = [
    'SNR_low_COUP_low_DIST_gaussian', 
    'SNR_low_COUP_low_DIST_laplace', 
    'SNR_low_COUP_low_DIST_uniform', 
    'SNR_high_COUP_low_DIST_gaussian', 
    'SNR_high_COUP_low_DIST_laplace', 
    'SNR_high_COUP_low_DIST_uniform',
    'SNR_no_noise_COUP_low_DIST_gaussian', 
    'SNR_no_noise_COUP_low_DIST_laplace', 
    'SNR_no_noise_COUP_low_DIST_uniform',
]

# Define HIGH coupling conditions (Figure 2) - 9 conditions
high_coupling_order = [
    'SNR_low_COUP_high_DIST_gaussian', 
    'SNR_low_COUP_high_DIST_laplace', 
    'SNR_low_COUP_high_DIST_uniform', 
    'SNR_high_COUP_high_DIST_gaussian', 
    'SNR_high_COUP_high_DIST_laplace', 
    'SNR_high_COUP_high_DIST_uniform',
    'SNR_no_noise_COUP_high_DIST_gaussian', 
    'SNR_no_noise_COUP_high_DIST_laplace', 
    'SNR_no_noise_COUP_high_DIST_uniform',
]

# Helper function to get files for a given order
def get_ordered_files(condition_order):
    ordered = []
    for condition in condition_order:
        for file in directed_files:
            if condition in file.stem:
                ordered.append(file)
                break
    return ordered

# =====================================================================================
# Create two figures: one for low coupling, one for high coupling
figure_configs = [
    {
        'name': 'Low Coupling',
        'files': get_ordered_files(low_coupling_order),
        'share_groups': None,  # Independent axes for low coupling
        'fig_num': 1,
    },
    {
        'name': 'High Coupling', 
        'files': get_ordered_files(high_coupling_order),
        'share_groups': [list(range(9))],  # All 9 subplots share axes
        'fig_num': 2,
    },
]

for config in figure_configs:
    file_subset = config['files']
    n_subplots = len(file_subset)
    
    if n_subplots == 0:
        print(f"No files found for {config['name']}, skipping...")
        continue
    
    print(f"\nProcessing {config['name']} ({n_subplots} conditions)...")
    
    # Create figure with appropriate axis sharing
    fig, axes, sync_axes = create_grouped_subplots(
        n_subplots, cols=3, figsize=(15, 9), share_groups=config['share_groups']
    )

    # Loop over all files for this figure
    for idx, directed_file in enumerate(file_subset):
        condition = directed_file.stem.replace("ddmdata_", "")
        ax = axes[idx]

        # Corresponding integrative model file
        integrative_file = INTEGRATIVE_DATA_DIR / f"cross_integrative_ddm_data_{condition}.mat"
        if not integrative_file.exists():
            print(f"Skipping {condition}: integrative file not found")
            continue
        
        # Get lambda and gamma values
        lambda_values = np.ravel(sio.loadmat(directed_file)['lambda_param'])
        gamma_values = np.ravel(sio.loadmat(integrative_file)['gamma'])
        
        # Verify same number of participants
        if len(lambda_values) != len(gamma_values):
            print(f"Skipping {condition}: participant count mismatch")
            continue
        
        print(f"  Loaded {condition}: {len(lambda_values)} participants")

        # Scatter plot lambda vs gamma
        ax.scatter(lambda_values, gamma_values, color='teal', s=40, alpha=0.7)

        # Calculate correlation between lambda and gamma
        r, _ = pearsonr(lambda_values, gamma_values)

        # Update axis labels and title
        ax.set_xlabel(r'Expected $\lambda$')
        ax.set_ylabel(r'Expected $\gamma$')
        ax.set_title(f'{condition_display_titles[condition]}\n$r$ = {r:.2f}')
    
    # Synchronize axes for groups that should share (only affects high coupling figure)
    sync_axes()

    # Remove any legends
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    
    # Save figure
    plt.tight_layout()
    output_file = OUTPUT_DIR / f'lambda_vs_gamma_expected_recovery_plot_fig{config["fig_num"]}.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")

print("\nPlot generation complete!")
