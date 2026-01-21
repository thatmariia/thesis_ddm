#!/usr/bin/env bash
#SBATCH --job-name=ddm_train
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/ddm_train-%j.out
#SBATCH --error=logs/ddm_train-%j.err

set -euf -o pipefail

# Load CUDA modules
echo "Loading CUDA modules..."
module purge  # purge needs to go before module load (otherwise it purges the loaded modules)
module load 2023

# UV version
UV_VERSION="0.9.5"

# Cache directory
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv_cache"
export UV_PROJECT_ENVIRONMENT="${TMPDIR:-/tmp}/thesis-ddm-venv"

# Set JAX as the Keras backend for BayesFlow
export KERAS_BACKEND=jax

# JAX GPU settings
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_PLATFORMS=cuda,cpu

# Print job information
echo "=========================================="
echo "Job ID: ${SLURM_JOBID}"
echo "Job name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "Backend: JAX"
echo "=========================================="

# Check GPU
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Download uv if needed (if there is no uv executable in the submit directory)
if [ ! -x ${SLURM_SUBMIT_DIR}/uv ]; then
    echo "Downloading uv ${UV_VERSION}..."
    cd ${SLURM_SUBMIT_DIR}
    wget -q https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-musl.tar.gz -O - | tar xz --strip-components=1 -C . uv-x86_64-unknown-linux-musl/uv
    chmod +x uv
fi

cd ${SLURM_SUBMIT_DIR}

# Install GPU-specific dependencies (jax with cuda)
./uv sync --frozen --group gpu

# Verify JAX GPU support
echo "=========================================="
echo "Checking JAX GPU support..."
./uv run python -c "import jax; print('JAX version:', jax.__version__); print('JAX devices:', jax.devices())"
echo "=========================================="

# Run training
echo "Starting training script..."
./uv run python scripts/integrative_ddm_train.py

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
