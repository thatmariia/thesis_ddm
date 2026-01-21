import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------------------------------------------------------------------
# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load data from both models
MODELS = {
    "integrative": {
        "data_dir": PROJECT_ROOT / "integrative_model" / "data_new_sigma",
        "param_keys": ["mu_delta", "eta_delta", "gamma", "sigma"]
    },
    "directed": {
        "data_dir": PROJECT_ROOT / "directed_model" / "data_new_sigma_z",
        "param_keys": ["b", "lambda_param", "eta", "sigma_z"]
    }
}

# ------------------------------------------------------------------------------------------------
# Load and analyze one model
def analyze_model(model_name, config):
    data_dir = config["data_dir"]
    param_keys = config["param_keys"]

    rows = []
    mat_files = sorted(data_dir.glob("cross_*.mat"))
    if not mat_files:
        print(f"[Warning] No .mat files found for {model_name} model in {data_dir}")
        return pd.DataFrame()

    for mat_file in mat_files:
        data = sio.loadmat(mat_file, squeeze_me=True)
        condition = str(mat_file.stem)

        # Extract theoretical correlation
        corr = np.mean(np.ravel(data["cor_delta_z_theoretical"])) if "cor_delta_z_theoretical" in data else np.nan

        # Compute variances for each parameter
        variances = {}
        for key in param_keys:
            variances[f"var_{key}"] = np.var(np.ravel(data[key])) if key in data else np.nan

        # Extract SNR and COUP condition from filename
        SNR = "high" if "SNR_high" in condition else "low" if "SNR_low" in condition else "unknown"
        COUP = "high" if "COUP_high" in condition else "low" if "COUP_low" in condition else "unknown"

        rows.append({
            "model": model_name,
            "condition": condition,
            "SNR": SNR,
            "COUP": COUP,
            "corr_delta_z": corr,
            **variances
        })

    df = pd.DataFrame(rows)

    # Print summary correlations per SNR/COUP condition
    print(f"\n=== {model_name.upper()} MODEL ===")
    print(f"Loaded {len(df)} files from {data_dir}")

    for snr in ["high", "low"]:
        for coup in ["high", "low"]:
            subset = df[(df["SNR"] == snr) & (df["COUP"] == coup)]
            if subset.empty:
                continue
            print(f"\nSNR {snr.upper()} | COUP {coup.upper()} ({len(subset)} files):")
            for key in param_keys:
                r = subset["corr_delta_z"].corr(subset[f"var_{key}"])
                print(f"{key:12s}: corr(var, corr_delta_z_theoretical) = {r:.3f}")

    # --- NEW: Coupling-only analysis (aggregating across SNR) ---
    for coup in ["high", "low"]:
        subset = df[df["COUP"] == coup]
        if subset.empty:
            continue
        print(f"\nCOUP {coup.upper()} (all SNRs, {len(subset)} files):")
        for key in param_keys:
            r = subset["corr_delta_z"].corr(subset[f"var_{key}"])
            print(f"{key:12s}: corr(var, corr_delta_z_theoretical) = {r:.3f}")

    return df

# ------------------------------------------------------------------------------------------------
# Run both models
all_dfs = []
for model_name, config in MODELS.items():
    df_model = analyze_model(model_name, config)
    if not df_model.empty:
        all_dfs.append(df_model)

if not all_dfs:
    raise RuntimeError("No data loaded for either model.")

df_all = pd.concat(all_dfs, ignore_index=True)

# ------------------------------------------------------------------------------------------------
# Visualization setup
plt.style.use("seaborn-v0_8-whitegrid")

# --- Scatterplots: variance vs correlation, colored by COUP, faceted by SNR ---
for model_name, config in MODELS.items():
    df_model = df_all[df_all["model"] == model_name]
    for key in config["param_keys"]:
        g = sns.FacetGrid(df_model, col="SNR", hue="COUP", height=4, aspect=1.2)
        g.map_dataframe(sns.scatterplot, x="corr_delta_z", y=f"var_{key}", alpha=0.7)
        g.map_dataframe(sns.regplot, x="corr_delta_z", y=f"var_{key}", scatter=False)
        g.add_legend()
        g.set_axis_labels("Theoretical Correlation (delta - z)", f"Variance of {key}")
        g.fig.suptitle(f"{model_name.capitalize()} Model — {key}: Variance vs Correlation by SNR and COUP", y=1.03)
        plt.tight_layout()
        plt.show()

# ------------------------------------------------------------------------------------------------
# Save results
out_path = PROJECT_ROOT / "analysis" / "variance_correlation_summary_SNR_COUP.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df_all.to_csv(out_path, index=False)
print(f"\nSaved detailed summary table to: {out_path}")
