import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 用来比较 joint training 和 frozen training 的曲线图，包含：
# 1. episode reward
# 2. actor loss
# 3. JSAE reconstruction MSE
# 4. decoded action drift
# 5. dynamics loss
# 6. reward loss
# 7. world model loss
# 验证Joint training 失败，到底是不是因为 JSAE 在训练过程中不断变化，
# 导致 latent action space（latent → physical action 映射）不稳定，从而让 actor 学习困难？

# ======================================================
# Config
# ======================================================

JOINT_CSV = (
    "/data3/zhangwanying/PWM/results/"
    "joint_test_6D_seed42/logs/"
    "cheetah-run-backwards_results.csv"
)

FROZEN_CSV = (
    "/data3/zhangwanying/PWM/results/"
    "joint_frozen_test_6D_seed42/logs/"
    "cheetah-run-backwards_results.csv"
)


OUTPUT_DIR = (
    "/data3/zhangwanying/PWM/r_plots/figures/"
    "joint_frozen_compare"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


TASK_NAME = "cheetah-run-backwards"
METHOD = "JointJSAE"
LATENT_DIM = 6
SEED = 42
ACTOR_LR = "1e-4"


filename = (
    f"{TASK_NAME}_"
    f"{METHOD}_"
    f"{LATENT_DIM}D_"
    f"seed{SEED}_"
    f"lr{ACTOR_LR}_"
    "latent_action_stability.png"
)



# ======================================================
# Utils
# ======================================================

def load_csv(path):

    df = pd.read_csv(path)

    # remove empty reward rows
    df = df.dropna(
        subset=["iteration"]
    )

    return df



def smooth(y, window=10):

    return (
        pd.Series(y)
        .rolling(
            window,
            min_periods=1
        )
        .mean()
    )



# ======================================================
# Plot
# ======================================================

def plot_compare(
        joint,
        frozen,
        metric,
        ylabel,
        filename,
        smooth_window=10
):

    plt.figure(
        figsize=(8,5),
        dpi=300
    )


    for df, name in [
        (joint, "Joint JSAE"),
        (frozen, "Frozen JSAE")
    ]:

        temp = df.dropna(
            subset=[metric]
        )

        x = temp["iteration"]
        y = temp[metric].astype(float)


        plt.plot(
            x,
            smooth(
                y,
                smooth_window
            ),
            linewidth=2,
            label=name
        )


    plt.xlabel(
        "Training Iterations"
    )

    plt.ylabel(
        ylabel
    )


    plt.title(
        f"{TASK_NAME} | "
        f"Latent Dim={LATENT_DIM} | "
        f"Seed={SEED}\n"
        f"{ylabel}",
        fontsize=12
    )


    plt.grid(
        linestyle="--",
        alpha=0.5
    )


    plt.legend(
        loc="best"
    )


    plt.tight_layout()


    save_path = os.path.join(
        OUTPUT_DIR,
        filename
    ),

    print(save_path)
    print(type(save_path))
    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "Saved:",
        save_path
    )



# ======================================================
# Main
# ======================================================
def main():

    joint = load_csv(
        JOINT_CSV
    )

    frozen = load_csv(
        FROZEN_CSV
    )


    print(
        "Joint rows:",
        len(joint)
    )

    print(
        "Frozen rows:",
        len(frozen)
    )


    # 1 Reward
    plot_compare(
        joint,
        frozen,
        "episode_reward",
        "Episode Reward",
        "reward_compare.png"
    )


    # 2 Actor loss
    plot_compare(
        joint,
        frozen,
        "actor_loss",
        "Actor Loss",
        "actor_loss_compare.png"
    )


    # 3 JSAE reconstruction
    plot_compare(
        joint,
        frozen,
        "jsae_recon_mse",
        "JSAE Reconstruction MSE",
        "jsae_recon_mse_compare.png"
    )


    # 4 Decoder drift
    plot_compare(
        joint,
        frozen,
        "decoded_action_drift",
        "Decoded Action Drift",
        "decoded_action_drift_compare.png"
    )


    # 5 World model dynamics
    plot_compare(
        joint,
        frozen,
        "dynamics_loss",
        "Dynamics Loss",
        "dynamics_loss_compare.png"
    )


    # 6 Reward model
    plot_compare(
        joint,
        frozen,
        "reward_loss",
        "Reward Loss",
        "reward_loss_compare.png"
    )


    # 7 WM loss
    plot_compare(
        joint,
        frozen,
        "wm_loss",
        "World Model Loss",
        "wm_loss_compare.png"
    )



if __name__ == "__main__":
    main()