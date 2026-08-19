import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 配置区
# 要修改csv文件名、任务名、算法名、seed、latent_action_dim
# ============================================================

# CSV 文件
CSV_PATH = (
    # "/data3/zhangwanying/PWM/results/"
    # "latent_ablation_20260817_091657/PWM_seed_44/"
    # "logs/cheetah-run-backwards_PWM_seed_44_results.csv"
    "/data3/zhangwanying/PWM/results"
    "/latent_ablation_20260817_091657/JSAE_6D_seed_42/logs"
    "/cheetah-run-backwards_JSAE_6D_seed_42_results.csv"
    
)

# 任务名称
TASK_NAME = "cheetah-run-backwards"

# 算法名称
ALGORITHM_NAME = "JSAE"

# Seed
SEED = 42

# Latent action dimension
# 原始 PWM 没有这个参数 -> None
# 如果是带 latent_action_dim 的实验，例如 4 -> 填 4
LATENT_ACTION_DIM = 6

# ------------------------------------------------------------
# 图片保存文件夹
# ------------------------------------------------------------
OUTPUT_DIR = "/data3/zhangwanying/PWM/r_plots/latent_ablation_figures" 

# 自动创建文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 根据实验类型自动生成图片名字
# ------------------------------------------------------------
if LATENT_ACTION_DIM is None:
    SAVE_IMAGE_NAME = (
        f"{TASK_NAME}_"
        f"{ALGORITHM_NAME}_"
        f"seed{SEED}.png"
    )

else:
    SAVE_IMAGE_NAME = (
        f"{TASK_NAME}_"
        f"{ALGORITHM_NAME}_"
        f"latent{LATENT_ACTION_DIM}D_"
        f"seed{SEED}.png"
    )


SAVE_IMAGE_PATH = os.path.join(
    OUTPUT_DIR,
    SAVE_IMAGE_NAME
)
# ============================================================

def plot_reward():
    # --------------------------------------------------------
    # 1. 检查文件
    # --------------------------------------------------------
    if not os.path.exists(CSV_PATH):
        print(
            f"错误：没找到 CSV 文件，请检查路径：\n"
            f"{CSV_PATH}"
        )
        return

    # --------------------------------------------------------
    # 2. 读取数据
    # --------------------------------------------------------
    df = pd.read_csv(CSV_PATH)

    required_columns = ["iteration", "episode_reward"]

    for col in required_columns:
        if col not in df.columns:
            print(
                f"错误：CSV 中不存在列 '{col}'。\n"
                f"当前列名为：{list(df.columns)}"
            )
            return

    # 删除没有 reward 的行
    reward_data = (
        df[["iteration", "episode_reward"]]
        .dropna()
        .sort_values("iteration")
    )

    if len(reward_data) == 0:
        print(
            "警告：CSV 中没有有效的 episode_reward 数据。"
        )
        return

    # --------------------------------------------------------
    # 3. 提取 X / Y
    # --------------------------------------------------------
    x = reward_data["iteration"].to_numpy()
    y = reward_data["episode_reward"].to_numpy()

    print(
        f"成功提取到 {len(y)} 个有效奖励记录点。"
    )

    # --------------------------------------------------------
    # 4. 计算指标
    # --------------------------------------------------------

    # 平均奖励
    mean_reward = np.mean(y)

    # 训练过程中的最大奖励
    best_reward = np.max(y)

    # Final-1000 Reward
    # 最后 1000 个 training iterations 内的平均 reward
    final_window_start = x.max() - 1000

    final_1000_mask = x > final_window_start
    final_1000_rewards = y[final_1000_mask]

    if len(final_1000_rewards) > 0:
        final_1000_reward = np.mean(final_1000_rewards)
    else:
        final_1000_reward = np.nan

    # 原始 AUC
    # 兼容不同 NumPy 版本
    if hasattr(np, "trapezoid"):
        auc = np.trapezoid(y, x)
    else:
        auc = np.trapz(y, x)

    # Normalized AUC
    iteration_range = x.max() - x.min()

    if iteration_range > 0:
        normalized_auc = auc / iteration_range
    else:
        normalized_auc = np.nan

    # 找到 best reward 所在 iteration
    best_index = np.argmax(y)
    best_iteration = x[best_index]

    print("-" * 45)
    print(f"Experiment       : {ALGORITHM_NAME}")
    print(f"Latent_action_dim: {LATENT_ACTION_DIM}")
    print(f"Seed             : {SEED}")
    print(f"Average Reward   : {mean_reward:.2f}")
    print(f"Best Reward      : {best_reward:.2f}")
    print(f"Final-1000 Reward: {final_1000_reward:.2f}")
    print(f"Best Iteration   : {best_iteration}")
    print(f"AUC              : {auc:.2f}")
    print(f"Normalized AUC   : {normalized_auc:.2f}")
    print("-" * 45)

    # --------------------------------------------------------
    # 5. 论文风格设置
    # --------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(
        figsize=(8.5, 5.2),
        dpi=150
    )

    # 给标题和副标题预留空间
    fig.subplots_adjust(
        top=0.84,
        bottom=0.14,
        left=0.12,
        right=0.97
    )

    # --------------------------------------------------------
    # 6. Reward 曲线
    # --------------------------------------------------------
    ax.plot(
        x,
        y,
        marker="o",
        linestyle="-",
        linewidth=1.8,
        markersize=4,
        label="Episode Reward"
    )

    # --------------------------------------------------------
    # 7. 标记 Best Reward
    # --------------------------------------------------------
    ax.scatter(
        best_iteration,
        best_reward,
        marker="*",
        s=85,
        zorder=5,
        label="Best Reward"
    )

    # --------------------------------------------------------
    # 8. 标题
    # --------------------------------------------------------
    fig.suptitle(
        TASK_NAME,
        fontsize=14,
        fontweight="bold",
        y=0.965
    )

    # --------------------------------------------------------
    # 9. 实验、参数、seed
    #    放在标题下面，不遮挡曲线
    # --------------------------------------------------------
    if LATENT_ACTION_DIM is not None:
        experiment_info = (
            f"Algorithm: {ALGORITHM_NAME}"
            f"   |   Latent Action Dim: {LATENT_ACTION_DIM}"
            f"   |   Seed: {SEED}"
        )
    else:
        experiment_info = (
            f"Algorithm: {ALGORITHM_NAME}"
            f"   |   Seed: {SEED}"
        )

    fig.text(
        0.5,
        0.905,
        experiment_info,
        ha="center",
        va="center",
        fontsize=10
    )

    # --------------------------------------------------------
    # 10. 坐标轴
    # --------------------------------------------------------
    ax.set_xlabel(
        "Training Iterations",
        labelpad=8
    )

    ax.set_ylabel(
        "Episode Reward",
        labelpad=8
    )

    # --------------------------------------------------------
    # 11. 网格
    # --------------------------------------------------------
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.45
    )

    ax.set_axisbelow(True)

    # --------------------------------------------------------
    # 12. 统计指标
    #     放在右下角，图例上方
    # --------------------------------------------------------
    # metrics_text = (
    #     f"Average Reward: {mean_reward:.2f}\n"
    #     f"Best Reward: {best_reward:.2f}\n"
    #     f"Final-1000 Reward: {final_1000_reward:.2f}\n"
    #     f"Normalized AUC: {normalized_auc:.2f}"
    # )

    # ax.text(
    #     0.98,
    #     0.155,
    #     metrics_text,
    #     transform=ax.transAxes,
    #     fontsize=9,
    #     verticalalignment="bottom",
    #     horizontalalignment="right",
    #     bbox=dict(
    #         boxstyle="round,pad=0.35",
    #         facecolor="white",
    #         edgecolor="0.55",
    #         linewidth=0.7,
    #         alpha=0.92
    #     )
    # )
    metrics_text = (
        f"Average Reward: {mean_reward:.2f}\n"
        f"Best Reward: {best_reward:.2f}\n"
        f"Final-1000 Reward: {final_1000_reward:.2f}\n"
        f"Normalized AUC: {normalized_auc:.2f}"
    )

    ax.text(
        0.98,
        0.16,
        metrics_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="0.55",
            linewidth=0.7,
            alpha=0.92
        )
    )

    # --------------------------------------------------------
    # 13. 图例
    # --------------------------------------------------------
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        edgecolor="0.75"
    )

    # --------------------------------------------------------
    # 14. 保存
    # --------------------------------------------------------
    plt.savefig(
        SAVE_IMAGE_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\n绘制成功！\n"
        f"图片保存为：{SAVE_IMAGE_PATH}"
    )

if __name__ == "__main__":
    plot_reward()