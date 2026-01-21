# Import modules
import scipy.io as sio
import numpy as np
from pathlib import Path
import sys

# Set project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Functions include empirical correlation between delta and z as well as the theoretical correlation between delta and z.
# Latter used for recovery plots.

# =====================================================================================
# Function to convert directed data to integrative data for recovery plotting and analysis
def directed_to_integrative_ddm(directed_data):
    """
    Reshape directed data to integrative ddm data.

    FROM: 'alpha', 'tau', 'beta', 'mu_z', 'sigma_z', 'lambda_param', 'b', 'eta'
    TO: 'alpha', 'tau', 'beta', 'mu_delta', 'eta_delta', 'gamma', 'sigma'
    """
    
    # Extract parameters from directed data
    lambda_param = np.ravel(directed_data['lambda_param'])
    b = np.ravel(directed_data['b'])
    eta = np.ravel(directed_data['eta'])
    sigma_z = np.ravel(directed_data['sigma_z'])

    # Theoretical correlation between delta and z (integrative model)
    corr_delta_z_theoretical = lambda_param * sigma_z / np.sqrt(lambda_param**2 * sigma_z**2 + eta**2)

    # Compute the transformation parameters
    # mu_delta = b (because mu_z = 0)
    mu_delta = b

    # eta_delta = sqrt(lambda_param^2 * sigma_z^2 + eta^2)
    eta_delta = np.sqrt(lambda_param**2 * sigma_z**2 + eta**2)

    # gamma = corr_delta_z * (sigma_z / sqrt(lambda^2 * sigma_z^2 + eta^2))
    gamma = corr_delta_z_theoretical * (sigma_z / np.sqrt(lambda_param**2 * sigma_z**2 + eta**2))

    # sigma = sqrt(sigma_z^2 * (1 - corr_delta_z^2))
    sigma = sigma_z * (np.sqrt(1 - corr_delta_z_theoretical**2))

    new_integrative_data = {
        # Core parameters stay the same
        'alpha': directed_data['alpha'],
        'beta': directed_data['beta'],
        'tau': directed_data['tau'],

        # Transformed parameters
        'mu_delta': mu_delta,
        'eta_delta': eta_delta,
        'gamma': gamma,
        'sigma': sigma,

        # Other parameters
        'rt': directed_data['rt'],
        'acc': directed_data['acc'],
        'choicert': directed_data['y'],
        'z': directed_data['z'],
        'participant': directed_data['participant'],
        'nparts': directed_data['nparts'],
        'ntrials': directed_data['ntrials'],
        'N': directed_data['N'],
        'minRT': directed_data['minRT'],
        'condition': directed_data['condition'],
        
        # Add the computed correlation between delta and z
        'cor_delta_z_theoretical': corr_delta_z_theoretical,
    }

    # Save the new integrative data
    condition = np.squeeze(new_integrative_data['condition'])
    DATA_DIR = PROJECT_ROOT / 'integrative_model' / 'data_new_sigma_new_conditions'
    file_path = DATA_DIR / f"cross_integrative_ddm_data_{condition}.mat"
    sio.savemat(file_path, new_integrative_data)
    print(f"Saved: {file_path}")

# =====================================================================================
# Function to convert integrative data to directed data for recovery plotting and analysis
def integrative_to_directed_ddm(integrative_data):
    """
     Reshape integrative data to directed data.

    FROM: 'alpha', 'tau', 'beta', 'mu_delta', 'eta_delta', 'gamma', 'sigma'
    TO: 'alpha', 'tau', 'beta', 'mu_z', 'sigma_z', 'lambda_param', 'b', 'eta'
     """

    # Extract parameters
    mu_delta = np.ravel(integrative_data['mu_delta'])
    eta_delta = np.ravel(integrative_data['eta_delta'])
    gamma = np.ravel(integrative_data['gamma'])
    sigma = np.ravel(integrative_data['sigma'])
    nparts = int(np.squeeze(integrative_data['nparts']))

    # Compute the theoretical correlation per trial between delta and z (directed model)
    corr_delta_z_theoretical = (gamma * eta_delta / np.sqrt(sigma**2 + gamma**2 * eta_delta**2))

    # mu_z = 0 (assumption in directed model)
    mu_z = np.zeros((1, nparts))

    # sigma_z = sqrt(gamma^2 * eta_delta^2 + sigma^2)
    sigma_z = np.sqrt(sigma**2 + gamma**2 * eta_delta**2)

    # lambda_param = eta_delta / sqrt(gamma^2 * eta_delta^2 + sigma^2)
    lambda_param = corr_delta_z_theoretical * (eta_delta / np.sqrt(gamma**2 * eta_delta**2 + sigma**2))

    # eta = eta_delta * sqrt(1 - corr_delta_z_theoretical^2)
    eta = eta_delta * np.sqrt(1 - corr_delta_z_theoretical**2)

    # b = mu_delta * (1 - lambda * gamma)
    b_value = mu_delta * (1 - lambda_param * gamma)

    new_directed_data = {
        # Core parameters stay the same
        'alpha': integrative_data['alpha'],
        'beta': integrative_data['beta'],
        'tau': integrative_data['tau'],

        # Transformed parameters
        'mu_z': mu_z,
        'sigma_z': sigma_z,
        'lambda_param': lambda_param,
        'b': b_value,
        'eta': eta,

        # Other parameters
        'rt': integrative_data['rt'],
        'acc': integrative_data['acc'],
        'y': integrative_data['choicert'],
        'z': integrative_data['z'],
        'participant': integrative_data['participant'],
        'nparts': integrative_data['nparts'],
        'ntrials': integrative_data['ntrials'],
        'N': integrative_data['N'],
        'minRT': integrative_data['minRT'],
        'condition': integrative_data['condition'],
        
        # Add the computed correlation
        'cor_delta_z_theoretical': corr_delta_z_theoretical,
    }

    # Save the new directed data
    condition = np.squeeze(new_directed_data['condition'])
    DATA_DIR = PROJECT_ROOT / 'directed_model' / 'data_new_sigma_z_cross'
    file_path = DATA_DIR / f"cross_directed_ddm_data_{condition}.mat"
    sio.savemat(file_path, new_directed_data)
    print(f"Saved: {file_path}")
