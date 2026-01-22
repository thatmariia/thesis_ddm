from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from ...utils import optional_import


def simulated_data_check(
    sim_data: dict, rt_path: str | Path, p300_path: str | Path
) -> tuple[str, str]:
    """
    Plot simulated choicert and z distributions from BayesFlow simulator output.
    """
    rt_path = Path(rt_path)
    p300_path = Path(p300_path)
    rt_path.parent.mkdir(parents=True, exist_ok=True)
    p300_path.parent.mkdir(parents=True, exist_ok=True)

    choicert = np.asarray(sim_data["choicert"]).flatten()
    z = np.asarray(sim_data["z"]).flatten()

    sns = optional_import("seaborn")
    use_sns = sns is not None

    # RTs
    plt.figure(figsize=(8, 6))
    if use_sns:
        sns.kdeplot(choicert, fill=True)
    else:
        plt.hist(choicert, bins=50, density=True, alpha=0.6)
    plt.title("Simulated Choice RTs")
    plt.xlabel("Choice RT")
    plt.ylabel("Density")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(rt_path, dpi=300)
    plt.close()

    # P300
    plt.figure(figsize=(8, 6))
    if use_sns:
        sns.kdeplot(z, fill=True)
    else:
        plt.hist(z, bins=50, density=True, alpha=0.6)
    plt.title("Simulated P300 (z) Distribution")
    plt.xlabel("P300 Amplitude")
    plt.ylabel("Density")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(p300_path, dpi=300)
    plt.close()

    print(f"RT plot saved to {rt_path}")
    print(f"P300 plot saved to {p300_path}")

    return str(rt_path), str(p300_path)
