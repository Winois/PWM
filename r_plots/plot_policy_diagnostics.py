import os
import pandas as pd
import matplotlib.pyplot as plt

# 用来画 policy optimization 的诊断曲线图，包含：
# 1. actor_loss
# 2. value_loss
# 3. actor_grad_norm
# 4. critic_grad_norm
# 5. episode_reward
# 目的是验证reward降低是不是因为policy optimization不稳定导致的
# ======================
# Config
# ======================
CSV_PATH = (
   "/data3/zhangwanying/PWM/"
    "results/joint_test_6D_seed42/logs/"
    "cheetah-run-backwards_results.csv"
)


OUTPUT_DIR = (
    "/data3/zhangwanying/PWM/"
    "r_plots/figures/"
    "policy_diagnostics"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


TASK_NAME = "cheetah-run-backwards"
LATENT_DIM = 6
SEED = 42
# ======================
# smooth
# ======================
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


# ======================
# plot function
# ======================

def plot_curve(ax, df, column, title):

    data = (
        df[
            [
                "iteration",
                column
            ]
        ]
        .dropna()
    )


    x = data["iteration"]

    y = data[column].astype(float)


    ax.plot(
        x,
        y,
        alpha=0.3,
        label="Raw"
    )


    ax.plot(
        x,
        smooth(y),
        linewidth=2,
        label="Smoothed"
    )


    ax.set_title(
        title,
        fontsize=11
    )


    ax.set_xlabel(
        "Training Iterations"
    )


    ax.grid(
        linestyle="--",
        alpha=0.4
    )


    ax.legend(
        fontsize=8
    )



# ======================
# main
# ======================


df = pd.read_csv(
    CSV_PATH
)


fig, axes = plt.subplots(
    4,
    1,
    figsize=(8,12),
    dpi=200
)


plot_curve(
    axes[0],
    df,
    "actor_loss",
    "Actor Loss"
)


plot_curve(
    axes[1],
    df,
    "value_loss",
    "Value Loss"
)


# gradient norm
if (
    "actor_grad_norm" in df.columns
    and
    "critic_grad_norm" in df.columns
):

    for col,label in [
        (
            "actor_grad_norm",
            "Actor Grad Norm"
        ),
        (
            "critic_grad_norm",
            "Critic Grad Norm"
        )
    ]:

        data=df[
            [
                "iteration",
                col
            ]
        ].dropna()


        axes[2].plot(
            data["iteration"],
            smooth(
                data[col].astype(float)
            ),
            linewidth=2,
            label=label
        )


    axes[2].set_title(
        "Gradient Norm"
    )

    axes[2].legend()

    axes[2].grid(
        linestyle="--",
        alpha=0.4
    )


plot_curve(
    axes[3],
    df,
    "episode_reward",
    "Episode Reward"
)



fig.suptitle(
    f"{TASK_NAME}: Policy Optimization Diagnostics\n"
    f"Joint Training | Latent Action Dim: {LATENT_DIM} | Seed: {SEED}",
    fontsize=14,
    fontweight="bold"
)


plt.tight_layout(
    rect=[0,0,1,0.94]
)


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "policy_diagnostics.png"
    ),
    dpi=300
)


plt.close()
print(
    "Saved policy_diagnostics.png"
)