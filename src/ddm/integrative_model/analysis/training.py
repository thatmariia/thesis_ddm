from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import bayesflow as bf


@dataclass
class MockHistory:
    """Mock history object for bayesflow plotting compatibility."""

    history: dict


def plot_training_history(
    history_dict: dict, figures_path: str | Path, window: int = 10
) -> str:
    """
    Plot training history including loss curves (train + val) with moving averages.
    """
    figures_path = Path(figures_path)
    figures_path.parent.mkdir(parents=True, exist_ok=True)

    train_loss = np.asarray(history_dict["loss"], dtype=float)
    val_loss = np.asarray(history_dict["val_loss"], dtype=float)

    train_ma = pd.Series(train_loss).rolling(window, min_periods=1).mean().to_numpy()
    val_ma = pd.Series(val_loss).rolling(window, min_periods=1).mean().to_numpy()

    plt.figure(figsize=(15, 5))
    plt.plot(train_loss, label="Train Loss", alpha=0.4)
    plt.plot(train_ma, label=f"Train MA ({window})", linewidth=2)
    plt.plot(val_loss, label="Val Loss", alpha=0.4)
    plt.plot(val_ma, label=f"Val MA ({window})", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs. Validation Loss")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(figures_path, dpi=300)
    plt.close()

    return str(figures_path)


def analyze_training_performance(
    history_dict: dict, verbose: bool = True, last_n: int = 20
) -> dict[str, float | int | None | bool]:
    """
    Analyze training performance and detect potential overfitting trend.
    """
    train_loss = np.asarray(history_dict["loss"], dtype=float)
    val_loss = np.asarray(history_dict["val_loss"], dtype=float)

    N = min(last_n, len(train_loss), len(val_loss))
    results: dict[str, float | int | None | bool] = {
        "train_loss_variance_last_N": float(np.var(train_loss[-N:])),
        "val_loss_variance_last_N": float(np.var(val_loss[-N:])),
        "final_train_loss": float(train_loss[-1]),
        "final_val_loss": float(val_loss[-1]),
        "best_val_loss": float(np.min(val_loss)),
        "best_val_epoch": int(np.argmin(val_loss) + 1),
        "N_epochs_analyzed": int(N),
    }

    if len(val_loss) > 10:
        half = len(val_loss) // 2
        slope = float(np.polyfit(np.arange(half, len(val_loss)), val_loss[half:], 1)[0])
        results["overfitting_detected"] = slope > 0
        results["val_loss_trend"] = slope
    else:
        results["overfitting_detected"] = None
        results["val_loss_trend"] = None

    if verbose:
        print(
            f"Train loss variance (last {N} epochs): {results['train_loss_variance_last_N']:.6f}"
        )
        print(
            f"Val   loss variance (last {N} epochs): {results['val_loss_variance_last_N']:.6f}"
        )
        print(f"Final train loss: {results['final_train_loss']:.4f}")
        print(f"Final validation loss: {results['final_val_loss']:.4f}")
        print(
            f"Best validation loss: {results['best_val_loss']:.4f} at epoch {results['best_val_epoch']}"
        )
        if results["overfitting_detected"] is True:
            print(
                "WARNING: Validation loss is trending upward (potential overfitting)."
            )

    return results


def plot_bayesflow_loss(history_dict: dict, figures_path: str | Path) -> str:
    """
    Create BayesFlow-style loss plot via bf.diagnostics.plots.loss.
    """
    figures_path = Path(figures_path)
    figures_path.parent.mkdir(parents=True, exist_ok=True)

    mock_history = MockHistory(history=history_dict)
    fig = bf.diagnostics.plots.loss(history=mock_history)
    fig.savefig(figures_path)

    return str(figures_path)
