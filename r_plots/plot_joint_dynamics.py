import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 画 joint training dynamics 的四联图，包含：
# 1. JSAE Parameter Drift
# 2. JSAE Reconstruction MSE
# 3. Actor Loss
# 4. Episode Reward
# ============================================================
# 配置
# ============================================================

CSV_PATH = (
    "/data3/zhangwanying/PWM/results/"
    "joint_test_6D_seed42/logs/"
    "cheetah-run-backwards_results.csv"
)


OUTPUT_DIR = (
    "/data3/zhangwanying/PWM/r_plots/"
    "figures/joint"
) # 四联图，用来看 joint training dynamics

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


TASK_NAME = "cheetah-run-backwards"

LATENT_DIM = 6

SEED = 42


# ============================================================
# 参数
# ============================================================

SMOOTH = True

SMOOTH_WINDOW = 5



# ============================================================
# 论文绘图风格
# ============================================================

plt.rcParams.update({

    "font.family": "serif",

    "font.size": 10,

    "axes.labelsize": 11,

    "axes.titlesize": 12,

    "xtick.labelsize": 9,

    "ytick.labelsize": 9,

    "legend.fontsize": 9,

    "axes.linewidth": 0.8
})



# ============================================================
# 工具函数
# ============================================================

def smooth(y, window=5):

    return (
        pd.Series(y)
        .rolling(
            window,
            min_periods=1
        )
        .mean()
        .to_numpy()
    )


def load_metric(df, column):

    if column not in df.columns:

        print(
            f"⚠️ Missing {column}"
        )

        return None, None


    data = (
        df[
            [
                "iteration",
                column
            ]
        ]
        .dropna()
        .sort_values(
            "iteration"
        )
    )


    if len(data)==0:

        return None, None


    x = data[
        "iteration"
    ].to_numpy(
        dtype=float
    )


    y = data[
        column
    ].to_numpy(
        dtype=float
    )


    return x,y



# ============================================================
# 主函数
# ============================================================

def main():


    df = pd.read_csv(
        CSV_PATH
    )


    print(
        "CSV columns:"
    )

    print(
        list(df.columns)
    )


    # ========================================================
    # 加载指标
    # ========================================================

    metrics = {

        "JSAE Parameter Drift":
            "jsae_param_drift",

        "JSAE Reconstruction MSE":
            "jsae_recon_mse",

        "Actor Loss":
            "actor_loss",

        "Episode Reward":
            "episode_reward"

    }


    data = {}


    for name,column in metrics.items():

        x,y = load_metric(
            df,
            column
        )


        if x is not None:

            data[name] = (
                x,
                y
            )


            print(
                f"{name}: "
                f"{len(y)} points"
            )


    # ========================================================
    # 创建四联图
    # ========================================================


    fig, axes = plt.subplots(

        4,
        1,

        figsize=(8,10),

        dpi=150,

        sharex=True

    )


    fig.subplots_adjust(

        top=0.88,

        bottom=0.08,

        left=0.13,

        right=0.95,

        hspace=0.35

    )



    # ========================================================
    # 绘制
    # ========================================================


    for ax,(title,(x,y)) in zip(
        axes,
        data.items()
    ):


        # 原始曲线

        ax.plot(

            x,

            y,

            linewidth=0.8,

            alpha=0.35

        )


        if SMOOTH:

            y_smooth = smooth(
                y,
                SMOOTH_WINDOW
            )

            ax.plot(

                x,

                y_smooth,

                linewidth=1.8,

                label="Smoothed"

            )

        ax.set_ylabel(
            title
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.4
        )

    axes[-1].set_xlabel(
        "Training Iterations"
    )


    # ========================================================
    # 标题
    # ========================================================
    fig.suptitle(
        f"{TASK_NAME}: Joint Training Dynamics",
        fontsize=14,
        fontweight="bold",
        y=0.96
    )

    fig.text(
        0.5,
        0.925,
        (
            f"Joint Training"
            f" | Latent Action Dim: {LATENT_DIM}"
            f" | Seed: {SEED}"
        ),
        ha="center",
        fontsize=10
    )

    # ========================================================
    # 保存
    # ========================================================
    save_path = os.path.join(
        OUTPUT_DIR,
        (
            f"{TASK_NAME}_"
            f"joint_dynamics_"
            f"latent{LATENT_DIM}D_"
            f"seed{SEED}.png"
        )

    )
    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"

    )
    plt.close()
    print(
        "\n✅ Saved:"
    )

    print(
        save_path
    )

if __name__ == "__main__":

    main()