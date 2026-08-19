import os
import glob
import re
import pandas as pd
import numpy as np


ROOT = "/data3/zhangwanying/PWM/results"

rows = []

files = glob.glob(
    os.path.join(ROOT, "**", "*JSAE*results.csv"),
    recursive=True,
)

for f in files:
    # 从路径/文件名提取 latent dim 和 seed
    text = f.replace("_", "-")

    dim_match = re.search(r"(?:JSAE-|latent-?)(\d+)D", text, re.I)
    seed_match = re.search(r"seed-?(\d+)", text, re.I)

    if dim_match is None or seed_match is None:
        continue

    dim = int(dim_match.group(1))
    seed = int(seed_match.group(1))

    if dim not in [4, 6, 8, 16]:
        continue

    df = pd.read_csv(f)

    if "jsae_recon_mse" not in df.columns:
        print("Missing recon MSE:", f)
        continue

    mse = df["jsae_recon_mse"].dropna()
    mae = (
        df["jsae_recon_mae"].dropna()
        if "jsae_recon_mae" in df.columns
        else pd.Series(dtype=float)
    )

    if len(mse) == 0:
        continue

    # 整个训练过程平均 reconstruction error
    mse_mean = mse.mean()

    # 最后 1000 iterations
    if "iteration" in df.columns:
        last = df[df["iteration"] >= 9000]
        final_mse = last["jsae_recon_mse"].dropna().mean()

        if "jsae_recon_mae" in last.columns:
            final_mae = last["jsae_recon_mae"].dropna().mean()
        else:
            final_mae = np.nan
    else:
        final_mse = np.nan
        final_mae = np.nan

    rows.append({
        "latent_dim": dim,
        "seed": seed,
        "mse_mean": mse_mean,
        "final1000_mse": final_mse,
        "mae_mean": mae.mean() if len(mae) else np.nan,
        "final1000_mae": final_mae,
        "file": f,
    })


runs = pd.DataFrame(rows).sort_values(
    ["latent_dim", "seed"]
)

print("\n=== PER-SEED RECONSTRUCTION ===")
print(
    runs[
        [
            "latent_dim",
            "seed",
            "mse_mean",
            "final1000_mse",
            "mae_mean",
        ]
    ].to_string(index=False)
)

summary = (
    runs.groupby("latent_dim")
    .agg(
        mse_mean=("mse_mean", "mean"),
        mse_std=("mse_mean", "std"),

        final_mse_mean=("final1000_mse", "mean"),
        final_mse_std=("final1000_mse", "std"),

        mae_mean=("mae_mean", "mean"),
        mae_std=("mae_mean", "std"),

        seeds=("seed", "count"),
    )
    .reset_index()
)

print("\n=== MEAN ± STD ACROSS SEEDS ===")
print(summary.to_string(index=False))

summary.to_csv(
    "/data3/zhangwanying/PWM/results/reconstruction_summary.csv",
    index=False,
)

print(
    "\nSaved to:",
    "/data3/zhangwanying/PWM/results/reconstruction_summary.csv"
)