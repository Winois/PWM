import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 用来绘制 latent action stability 的图表
# 看 latent action 的标准差、decoded action 的标准差、decoded action 的漂移，以及 episode reward 的变化情况。
CSV_PATH = (
    "/data3/zhangwanying/PWM/"
    "results/joint_frozen_test_6D_seed42_actor_lr1e-4/logs/"
    "cheetah-run-backwards_results.csv"
)


OUTPUT_DIR = (
    "/data3/zhangwanying/PWM/"
    "r_plots/figures/"
    "latent_stability"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

TASK_NAME = "cheetah-run-backwards"
LATENT_DIM = 6
SEED = 42


def smooth(y, window=5):

    return (
        pd.Series(y)
        .rolling(
            window,
            min_periods=1
        )
        .mean()
        .values
    )


def plot_metric(ax, df, col, title):

    data = (
        df[
            [
                "iteration",
                col
            ]
        ]
        .dropna()
    )

    x = data["iteration"]

    y = data[col].astype(float)


    ax.plot(
        x,
        y,
        alpha=0.35,
        label="Raw"
    )

    ax.plot(
        x,
        smooth(y),
        linewidth=2,
        label="Smoothed"
    )


    ax.set_title(title)

    ax.set_xlabel(
        "Training Iterations"
    )

    ax.grid(
        linestyle="--",
        alpha=0.4
    )

    ax.legend()



df = pd.read_csv(
    CSV_PATH
)


fig, axes = plt.subplots(
    4,
    1,
    figsize=(8,12),
    dpi=200
)

fig.suptitle(
    f"{TASK_NAME}: Latent Action Stability\n"
    f"Joint Training | Latent Action Dim: {LATENT_DIM} | Seed: {SEED}",
    fontsize=14,
    fontweight="bold"
)
plt.tight_layout(rect=[0,0,1,0.95])


plot_metric(
    axes[0],
    df,
    "latent_action_std",
    "Latent Action Std"
)


plot_metric(
    axes[1],
    df,
    "decoded_action_std",
    "Decoded Action Std"
)


plot_metric(
    axes[2],
    df,
    "decoded_action_drift",
    "Decoded Action Drift"
)


plot_metric(
    axes[3],
    df,
    "episode_reward",
    "Episode Reward"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "latent_action_stability.png"
    ),
    dpi=300
)


plt.close()
print(
    "Saved latent_action_stability.png"
)