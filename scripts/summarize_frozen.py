import os
import glob
import pandas as pd
import numpy as np


RESULT_ROOT = "/data3/zhangwanying/PWM/results"

PATTERN = (
    RESULT_ROOT +
    "/frozen_test_*_seed*/logs/"
    "cheetah-run-backwards_results.csv"
)


def calc_auc(df):
    """
    reward curve area
    """
    df = df.dropna(subset=["episode_reward"])

    if len(df) < 2:
        return np.nan

    return np.trapz(
        df["episode_reward"].values,
        df["iteration"].values
    )


records = []


files = sorted(glob.glob(PATTERN))

print("Found files:")
for f in files:
    print(f)


for file in files:

    df = pd.read_csv(file)

    # metadata
    latent_dim = int(df["latent_action_dim"].dropna().iloc[0])
    seed = int(df["seed"].dropna().iloc[0])

    method = "JSAE_frozen"


    # reward only exists at evaluation points
    reward_df = df.dropna(
        subset=["episode_reward"]
    )


    mean_reward = (
        reward_df["episode_reward"]
        .mean()
    )


    # final 1000 training steps
    max_iter = reward_df["iteration"].max()

    final_df = reward_df[
        reward_df["iteration"]
        >= max_iter - 1000
    ]


    final1000_reward = (
        final_df["episode_reward"]
        .mean()
    )


    auc = calc_auc(reward_df)


    recon_mse = (
        df["jsae_recon_mse"]
        .dropna()
        .mean()
    )

    recon_mae = (
        df["jsae_recon_mae"]
        .dropna()
        .mean()
    )


    frozen_flag = (
        df["jsae_frozen"]
        .dropna()
        .iloc[-1]
    )


    max_param_change = (
        df["jsae_max_param_change"]
        .dropna()
        .max()
    )


    records.append({

        "method": method,
        "latent_dim": latent_dim,
        "seed": seed,

        "mean_reward": mean_reward,
        "final1000_reward": final1000_reward,
        "auc": auc,

        "recon_mse": recon_mse,
        "recon_mae": recon_mae,

        "jsae_frozen": frozen_flag,
        "max_param_change": max_param_change,
    })


result = pd.DataFrame(records)


# per seed
print("\n=== PER SEED ===")
print(result)


# mean std
summary = (
    result
    .groupby(
        ["method", "latent_dim"]
    )
    .agg(
        mean_reward_mean=(
            "mean_reward",
            "mean"
        ),
        mean_reward_std=(
            "mean_reward",
            "std"
        ),

        final1000_mean=(
            "final1000_reward",
            "mean"
        ),
        final1000_std=(
            "final1000_reward",
            "std"
        ),

        auc_mean=(
            "auc",
            "mean"
        ),
        auc_std=(
            "auc",
            "std"
        ),

        recon_mse_mean=(
            "recon_mse",
            "mean"
        ),

        max_param_change=(
            "max_param_change",
            "max"
        )
    )
    .reset_index()
)


print("\n=== SUMMARY ===")
print(summary)


out1 = (
    RESULT_ROOT +
    "/frozen_summary.csv"
)

out2 = (
    RESULT_ROOT +
    "/frozen_summary_mean_std.csv"
)


result.to_csv(
    out1,
    index=False
)

summary.to_csv(
    out2,
    index=False
)


print("\nSaved:")
print(out1)
print(out2)