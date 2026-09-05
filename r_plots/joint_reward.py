import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 只画joint training实验的reward
# ============================================================
# 1. 配置区
# ============================================================

# Joint 实验 CSV 路径
CSV_PATH = (
    "/data3/zhangwanying/PWM/results/"
    "joint_test_6D_seed42/logs/"
    "cheetah-run-backwards_results.csv"
)

# ------------------------------------------------------------
# 如果 CSV 中有这些字段，程序会优先自动读取。
# 如果 CSV 中没有，则使用下面手动设置的值。
# ------------------------------------------------------------

TASK_NAME = "cheetah-run-backwards"

EXPERIMENT_NAME = "JSAE-Joint"

# Joint 实验中使用的 latent_action_dim
LATENT_ACTION_DIM = 6

# seed
SEED = 42

# Joint 实验：允许 world model 一起更新
FINETUNE_WM = True


# ------------------------------------------------------------
# 图片保存目录
# ------------------------------------------------------------

OUTPUT_DIR = (
    "/data3/zhangwanying/PWM/r_plots/"
    "joint_figures"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. 辅助函数：读取 CSV 中的实验信息
# ============================================================
def get_first_valid_value(df, column_name, default=None):
    """
    如果 CSV 中存在该列，则读取第一个非空值；
    否则返回 default。
    """
    if column_name in df.columns:
        values = df[column_name].dropna()

        if len(values) > 0:
            return values.iloc[0]

    return default


# ============================================================
# 3. 绘制 Reward 曲线
# ============================================================

def plot_reward():

    # --------------------------------------------------------
    # 检查文件
    # --------------------------------------------------------

    if not os.path.exists(CSV_PATH):
        print(
            f"❌ 错误：没找到 CSV 文件：\n"
            f"{CSV_PATH}"
        )
        return


    # --------------------------------------------------------
    # 读取 CSV
    # --------------------------------------------------------

    df = pd.read_csv(CSV_PATH)

    print("\nCSV 列名：")
    print(list(df.columns))


    # --------------------------------------------------------
    # 检查必要列
    # --------------------------------------------------------

    required_columns = [
        "iteration",
        "episode_reward"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        print(
            "\n❌ CSV 缺少必要列："
            f"{missing_columns}"
        )
        return


    # ========================================================
    # 4. 自动读取实验信息
    # ========================================================

    task_name = get_first_valid_value(
        df,
        "task",
        TASK_NAME
    )

    seed = get_first_valid_value(
        df,
        "seed",
        SEED
    )

    algorithm = get_first_valid_value(
        df,
        "algorithm",
        EXPERIMENT_NAME
    )

    latent_dim = get_first_valid_value(
        df,
        "latent_action_dim",
        LATENT_ACTION_DIM
    )

    # 转换显示格式
    try:
        seed = int(seed)
    except (ValueError, TypeError):
        pass

    try:
        latent_dim = int(latent_dim)
    except (ValueError, TypeError):
        pass


    # ========================================================
    # 5. 清洗 Reward 数据
    # ========================================================

    reward_data = (
        df[
            [
                "iteration",
                "episode_reward"
            ]
        ]
        .dropna()
        .sort_values("iteration")
    )

    if len(reward_data) == 0:
        print(
            "⚠️ CSV 中没有有效的 "
            "episode_reward 数据。"
        )
        return


    # 转成 numpy
    x = reward_data[
        "iteration"
    ].to_numpy(dtype=float)

    y = reward_data[
        "episode_reward"
    ].to_numpy(dtype=float)


    print(
        f"\n📊 成功提取到 "
        f"{len(y)} 个有效 Reward 点。"
    )


    # ========================================================
    # 6. 计算评价指标
    # ========================================================

    # --------------------------------------------------------
    # 6.1 Average Reward
    # --------------------------------------------------------

    mean_reward = np.mean(y)


    # --------------------------------------------------------
    # 6.2 Best Reward
    # --------------------------------------------------------

    best_reward = np.max(y)

    best_index = np.argmax(y)

    best_iteration = x[best_index]


    # --------------------------------------------------------
    # 6.3 Final-1000 Reward
    #
    # 定义：
    # 最后 1000 个 training iterations 范围内，
    # 所有有效 episode_reward 的平均值
    # --------------------------------------------------------

    max_iteration = np.max(x)

    final_1000_start = max_iteration - 1000

    final_1000_mask = (
        x > final_1000_start
    )

    final_1000_rewards = (
        y[final_1000_mask]
    )

    if len(final_1000_rewards) > 0:

        final_1000_reward = np.mean(
            final_1000_rewards
        )

    else:

        final_1000_reward = np.nan


    # --------------------------------------------------------
    # 6.4 AUC
    #
    # Reward-Iteration 曲线下面积
    # --------------------------------------------------------

    if hasattr(np, "trapezoid"):

        auc = np.trapezoid(
            y,
            x
        )

    else:

        # 兼容旧版本 NumPy
        auc = np.trapz(
            y,
            x
        )


    # --------------------------------------------------------
    # 6.5 Normalized AUC
    #
    # 暂时计算出来，终端输出。
    # 图中默认使用用户要求的原始 AUC。
    # --------------------------------------------------------

    iteration_range = (
        x.max() - x.min()
    )

    if iteration_range > 0:

        normalized_auc = (
            auc / iteration_range
        )

    else:

        normalized_auc = np.nan


    # ========================================================
    # 7. 终端打印实验结果
    # ========================================================

    print("\n" + "=" * 55)

    print(
        f"Experiment          : "
        f"{EXPERIMENT_NAME}"
    )

    print(
        f"Algorithm           : "
        f"{algorithm}"
    )

    print(
        f"Task                : "
        f"{task_name}"
    )

    print(
        f"Latent Action Dim   : "
        f"{latent_dim}"
    )

    print(
        f"Finetune WM         : "
        f"{FINETUNE_WM}"
    )

    print(
        f"Seed                : "
        f"{seed}"
    )

    print("-" * 55)

    print(
        f"Average Reward      : "
        f"{mean_reward:.2f}"
    )

    print(
        f"Best Reward         : "
        f"{best_reward:.2f}"
    )

    print(
        f"Best Iteration      : "
        f"{best_iteration:.0f}"
    )

    print(
        f"Final-1000 Reward   : "
        f"{final_1000_reward:.2f}"
    )

    print(
        f"AUC                 : "
        f"{auc:.2f}"
    )

    print(
        f"Normalized AUC      : "
        f"{normalized_auc:.2f}"
    )

    print("=" * 55)


    # ========================================================
    # 8. 论文绘图风格
    # ========================================================

    plt.rcParams.update({

        "font.family": "serif",

        "font.size": 10,

        "axes.titlesize": 14,

        "axes.labelsize": 11,

        "xtick.labelsize": 9,

        "ytick.labelsize": 9,

        "legend.fontsize": 9,

        "axes.linewidth": 0.8
    })


    # ========================================================
    # 9. 创建画布
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8.5, 5.2),
        dpi=150
    )

    # 给标题和实验参数留出空间
    fig.subplots_adjust(
        top=0.84,
        bottom=0.14,
        left=0.12,
        right=0.97
    )


    # ========================================================
    # 10. 绘制 Reward 曲线
    # ========================================================

    ax.plot(
        x,
        y,
        marker="o",
        linestyle="-",
        linewidth=1.7,
        markersize=3.8,
        label="Episode Reward"
    )


    # ========================================================
    # 11. 标记 Best Reward
    # ========================================================

    ax.scatter(
        best_iteration,
        best_reward,
        marker="*",
        s=90,
        zorder=5,
        label="Best Reward"
    )


    # ========================================================
    # 12. 标题
    # ========================================================

    fig.suptitle(
        task_name,
        fontsize=14,
        fontweight="bold",
        y=0.965
    )


    # ========================================================
    # 13. Joint 实验参数副标题
    # ========================================================

    experiment_info = (
        f"Joint Training"
        f"   |   Latent Action Dim: {latent_dim}"
        f"   |   Finetune WM: True"
        f"   |   Seed: {seed}"
    )

    fig.text(
        0.5,
        0.905,
        experiment_info,
        ha="center",
        va="center",
        fontsize=9.5
    )


    # ========================================================
    # 14. 坐标轴
    # ========================================================

    ax.set_xlabel(
        "Training Iterations",
        labelpad=8
    )

    ax.set_ylabel(
        "Episode Reward",
        labelpad=8
    )


    # ========================================================
    # 15. 网格
    # ========================================================

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.40
    )

    ax.set_axisbelow(True)


    # ========================================================
    # 16. 图中显示统计指标
    # ========================================================

    metrics_text = (
        f"Average Reward: {mean_reward:.2f}\n"
        f"Best Reward: {best_reward:.2f}\n"
        f"Final-1000 Reward: {final_1000_reward:.2f}\n"
        f"Normalized AUC: {normalized_auc:,.2f}"
    )

    # 放右下区域，不挡后期 reward 曲线
    ax.text(
        0.98,
        0.19,
        metrics_text,
        transform=ax.transAxes,
        fontsize=8.8,
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


    # ========================================================
    # 17. 图例
    # ========================================================

    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.90,
        edgecolor="0.75"
    )


    # ========================================================
    # 18. 自动生成图片文件名
    # ========================================================

    safe_task_name = str(
        task_name
    ).replace("/", "_")

    save_image_name = (
        f"{safe_task_name}_"
        f"joint_"
        f"latent{latent_dim}D_"
        f"seed{seed}.png"
    )

    save_image_path = os.path.join(
        OUTPUT_DIR,
        save_image_name
    )


    # ========================================================
    # 19. 保存图片
    # ========================================================

    plt.savefig(
        save_image_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "\n✅ Reward 曲线绘制完成。"
    )

    print(
        f"图片保存位置：\n"
        f"{save_image_path}"
    )


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    plot_reward()