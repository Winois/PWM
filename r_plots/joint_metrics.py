import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# 画joint training实验的reward, reconstruction, world model loss, dynamics loss 
# ============================================================
# 1. 配置区
# ============================================================

CSV_PATH = (
    "/data3/zhangwanying/PWM/results/"
    "joint_frozen_test_6D_seed42_actor_lr1e-4/logs/"
    "cheetah-run-backwards_results.csv"
)

TASK_NAME = "cheetah-run-backwards"
EXPERIMENT_NAME = "JSAE-Joint"
actor_lr = 1e-4

LATENT_ACTION_DIM = 6
SEED = 42
FINETUNE_WM = True


# 图片保存目录
OUTPUT_DIR = (
    "/data3/zhangwanying/PWM/r_plots/"
    "joint_figures"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. 绘图参数
# ============================================================

# 是否画平滑曲线
USE_SMOOTH = True

# rolling window
SMOOTH_WINDOW = 5


# ============================================================
# 3. 论文风格
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
})


# ============================================================
# 4. 工具函数
# ============================================================

def get_first_valid_value(df, column_name, default=None):
    if column_name in df.columns:
        values = df[column_name].dropna()
        if len(values) > 0:
            return values.iloc[0]
    return default

def find_column(df, candidates):
    """
    在多个候选列名中寻找实际存在的列。

    例如：
    dynamics_loss 可能实际保存成 dyn_loss
    """
    for name in candidates:
        if name in df.columns:
            return name
    return None

def to_numeric_series(series):
    """
    将 CSV 中的普通数字、字符串数字以及
    tensor(39.4192, device='cuda:0', grad_fn=...)
    统一转换为 float。

    无法转换的值设为 NaN。
    """
    def convert_value(value):
        # 已经是空值
        if pd.isna(value):
            return np.nan
        # 本身已经是数字
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        # 转字符串
        text = str(value).strip()
        # ----------------------------------------------------
        # 处理 PyTorch tensor 字符串
        #
        # tensor(39.4192, device='cuda:0', ...)
        # tensor(0.0043)
        # tensor(nan, device='cuda:0')
        # ----------------------------------------------------
        if text.startswith("tensor("):
            match = re.search(
                r"tensor\(\s*"
                r"([-+]?(?:\d*\.?\d+(?:[eE][-+]?\d+)?|nan|inf|-inf))",
                text,
                flags=re.IGNORECASE
            )
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return np.nan
        # ----------------------------------------------------
        # 普通数字字符串
        # ----------------------------------------------------
        try:
            return float(text)
        except (ValueError, TypeError):
            return np.nan
    return series.apply(convert_value)

def rolling_smooth(y, window=5):
    numeric_y = to_numeric_series(
        pd.Series(y)
    )
    return (
        numeric_y
        .rolling(
            window=window,
            min_periods=1
        )
        .mean()
        .to_numpy()
    )

def print_metric_statistics(df, column):
    if column is None:
        return

    values = to_numeric_series(
        df[column]
    )

    # 只保留有限值
    valid = values[
        np.isfinite(values)
    ]

    if len(valid) == 0:

        print(
            f"{column:<22}: no valid values"
        )
        return

    nan_count = (
        len(values) - len(valid)
    )

    print(
        f"{column:<22}: "
        f"N={len(valid):4d} | "
        f"mean={valid.mean():10.5f} | "
        f"min={valid.min():10.5f} | "
        f"max={valid.max():10.5f} | "
        f"last={valid.iloc[-1]:10.5f} | "
        f"NaN/Inf={nan_count}"
    )

# ============================================================
# 5. World Model Loss 图
# ============================================================
def plot_world_model_losses(
    df,
    task_name,
    seed,
    latent_dim
):

    # --------------------------------------------------------
    # 自动兼容不同列名
    # --------------------------------------------------------
    wm_col = find_column(
        df,
        [
            "wm_loss",
            "world_model_loss"
        ]
    )

    dynamics_col = find_column(
        df,
        [
            "dynamics_loss",
            "dyn_loss"
        ]
    )

    reward_col = find_column(
        df,
        [
            "reward_loss",
            "wm_reward_loss"
        ]
    )
    available_metrics = []

    if wm_col is not None:
        available_metrics.append(
            (wm_col, "World Model Loss")
        )

    if dynamics_col is not None:
        available_metrics.append(
            (dynamics_col, "Dynamics Loss")
        )

    if reward_col is not None:
        available_metrics.append(
            (reward_col, "Reward Loss")
        )

    if len(available_metrics) == 0:
        print(
            "\n⚠️ 没有找到 World Model loss 相关列，"
            "跳过 Figure 1。"
        )
        return
    # --------------------------------------------------------
    # 创建图
    # --------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(8.5, 5.2),
        dpi=150
    )

    for column, label in available_metrics:
        metric_data = (
            df[
                [
                    "iteration",
                    column
                ]
            ]
            .sort_values("iteration")
        )

        # iteration 转数值
        x = pd.to_numeric(
            metric_data["iteration"],
            errors="coerce"
        ).to_numpy(dtype=float)

        # loss 自动解析普通 float 和 tensor(...) 字符串
        y = to_numeric_series(
            metric_data[column]
        ).to_numpy(dtype=float)

        # 删除 NaN / Inf
        valid_mask = (
            np.isfinite(x)
            & np.isfinite(y)
        )
        x = x[valid_mask]
        y = y[valid_mask]

        if len(y) == 0:
            print(
                f"⚠️ {column} 没有有效数值，跳过。"
            )
            continue
        # 原始曲线 + 平滑曲线
        if USE_SMOOTH:
            ax.plot(
                x,
                y,
                linewidth=0.8,
                alpha=0.25
            )
            y_smooth = rolling_smooth(
                y,
                SMOOTH_WINDOW
            )
            ax.plot(
                x,
                y_smooth,
                linewidth=1.8,
                label=label
            )
        else:
            ax.plot(
                x,
                y,
                linewidth=1.6,
                label=label
            )

    # --------------------------------------------------------
    # 标题
    # --------------------------------------------------------
    fig.suptitle(
        f"{task_name}: World Model Adaptation",
        fontsize=13,
        fontweight="bold",
        y=0.965
    )
    fig.text(
        0.5,
        0.905,
        (
            f"Joint Training"
            f"   |   Latent Action Dim: {latent_dim}"
            f"   |   Seed: {seed}"
        ),
        ha="center",
        fontsize=9.5
    )


    ax.set_xlabel(
        "Training Iterations"
    )
    ax.set_ylabel(
        "Loss"
    )
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.4
    )
    ax.legend(
        loc="best",
        frameon=True
    )

    fig.subplots_adjust(
        top=0.84,
        bottom=0.14,
        left=0.12,
        right=0.97
    )

    save_path = os.path.join(
        OUTPUT_DIR,
        (
            f"{task_name}_joint_"
            f"latent{latent_dim}D_"
            f"seed{seed}_"
            f"world_model_losses.png"
        )
    )


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        f"✅ World Model 图：\n"
        f"   {save_path}"
    )


# ============================================================
# 6. JSAE Reconstruction 图
# ============================================================

def plot_jsae_reconstruction(
    df,
    task_name,
    seed,
    latent_dim
):

    # --------------------------------------------------------
    # 这里优先找真正的 reconstruction MSE
    # 不自动用 jae_loss 替代
    # --------------------------------------------------------

    recon_col = find_column(
        df,
        [
            "jsae_recon_mse",
            "recon_mse",
            "jae_recon_mse"
        ]
    )


    if recon_col is None:

        print(
            "\n⚠️ CSV 中没有找到："
            "jsae_recon_mse / recon_mse / jae_recon_mse"
        )

        print(
            "   因此不会把 jae_loss 强行当作 "
            "reconstruction MSE。"
        )

        print(
            "   JSAE reconstruction 图暂时跳过。"
        )

        return


    recon_data = (
        df[
            [
                "iteration",
                recon_col
            ]
        ]
        .dropna()
        .sort_values("iteration")
    )


    if len(recon_data) == 0:

        print(
            "\n⚠️ jsae_recon_mse 没有有效数据。"
        )

        return


    x = recon_data[
        "iteration"
    ].to_numpy()

    y = recon_data[
        recon_col
    ].to_numpy()


    fig, ax = plt.subplots(
        figsize=(8.5, 5.2),
        dpi=150
    )


    # 原始 reconstruction MSE
    if USE_SMOOTH:

        ax.plot(
            x,
            y,
            linewidth=0.8,
            alpha=0.30,
            label="Raw Reconstruction MSE"
        )


        smooth_y = rolling_smooth(
            y,
            SMOOTH_WINDOW
        )


        ax.plot(
            x,
            smooth_y,
            linewidth=1.8,
            label=(
                f"Smoothed Reconstruction MSE "
                f"(window={SMOOTH_WINDOW})"
            )
        )

    else:

        ax.plot(
            x,
            y,
            linewidth=1.7,
            label="JSAE Reconstruction MSE"
        )


    fig.suptitle(
        f"{task_name}: JSAE Reconstruction",
        fontsize=13,
        fontweight="bold",
        y=0.965
    )


    fig.text(
        0.5,
        0.905,
        (
            f"Joint Training"
            f"   |   Latent Action Dim: {latent_dim}"
            f"   |   Seed: {seed}"
        ),
        ha="center",
        fontsize=9.5
    )


    ax.set_xlabel(
        "Training Iterations"
    )

    ax.set_ylabel(
        "Reconstruction MSE"
    )


    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.4
    )


    ax.legend(
        loc="best"
    )


    # --------------------------------------------------------
    # 显示 summary
    # --------------------------------------------------------

    mean_recon = np.mean(y)
    final_recon = y[-1]
    best_recon = np.min(y)


    metric_text = (
        f"Mean MSE: {mean_recon:.5f}\n"
        f"Final MSE: {final_recon:.5f}\n"
        f"Minimum MSE: {best_recon:.5f}"
    )


    ax.text(
        0.98,
        0.96,
        metric_text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=8.8,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="0.55",
            alpha=0.90
        )
    )


    fig.subplots_adjust(
        top=0.84,
        bottom=0.14,
        left=0.12,
        right=0.97
    )


    save_path = os.path.join(
        OUTPUT_DIR,
        (
            f"{task_name}_joint_"
            f"latent{latent_dim}D_"
            f"seed{seed}_"
            f"jsae_recon_mse.png"
        )
    )


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        f"✅ JSAE Reconstruction 图：\n"
        f"   {save_path}"
    )


# ============================================================
# 7. Control Performance 图
# ============================================================
def plot_control_performance(
    df,
    task_name,
    seed,
    latent_dim
):
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
            "\n⚠️ 没有 episode_reward 数据。"
        )
        return

    x = reward_data[
        "iteration"
    ].to_numpy(dtype=float)

    y = reward_data[
        "episode_reward"
    ].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Average Reward
    # --------------------------------------------------------
    average_reward = np.mean(y)

    # --------------------------------------------------------
    # Best Reward
    # --------------------------------------------------------
    best_reward = np.max(y)
    best_index = np.argmax(y)
    best_iteration = x[
        best_index
    ]

    # --------------------------------------------------------
    # Final-1000 Reward
    # --------------------------------------------------------
    final_start = (
        x.max() - 1000
    )

    final_mask = (
        x > final_start
    )

    final_rewards = y[
        final_mask
    ]

    if len(final_rewards) > 0:
        final_1000_reward = np.mean(
            final_rewards
        )
    else:
        final_1000_reward = np.nan

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------
    if hasattr(np, "trapezoid"):
        auc = np.trapezoid(
            y,
            x
        )
    else:
        auc = np.trapz(
            y,
            x
        )

    # --------------------------------------------------------
    # Normalized AUC
    # --------------------------------------------------------
    iteration_range = (
        x.max() - x.min()
    )

    if iteration_range > 0:
        normalized_auc = (
            auc /
            iteration_range
        )
    else:
        normalized_auc = np.nan

    # ========================================================
    # 绘图
    # ========================================================
    fig, ax = plt.subplots(
        figsize=(8.5, 5.2),
        dpi=150
    )

    ax.plot(
        x,
        y,
        marker="o",
        linestyle="-",
        linewidth=1.7,
        markersize=3.8,
        label="Episode Reward"
    )

    # Best Reward
    ax.scatter(
        best_iteration,
        best_reward,
        marker="*",
        s=90,
        zorder=5,
        label="Best Reward"
    )

    # --------------------------------------------------------
    # Final-1000 区域
    # --------------------------------------------------------
    ax.axvspan(
        final_start,
        x.max(),
        alpha=0.08,
        label="Final 1000 Iterations"
    )

    fig.suptitle(
        f"{task_name}: Control Performance",
        fontsize=13,
        fontweight="bold",
        y=0.965
    )

    fig.text(
        0.5,
        0.905,
        (
            f"Joint Training"
            f"   |   Latent Action Dim: {latent_dim}"
            f"   |   Seed: {seed}"
            f"   |   Actor LR: {actor_lr}"
        ),
        ha="center",
        fontsize=9.5
    )

    ax.set_xlabel(
        "Training Iterations"
    )

    ax.set_ylabel(
        "Episode Reward"
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.4
    )

    # --------------------------------------------------------
    # 指标框
    # --------------------------------------------------------
    metrics_text = (
        f"Average Reward: {average_reward:.2f}\n"
        f"Best Reward: {best_reward:.2f}\n"
        f"Final-1000 Reward: {final_1000_reward:.2f}\n"
        f"Normalized AUC: {normalized_auc:,.2f}"
    )

    ax.text(
        0.98,
        0.20,
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

    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.90
    )

    fig.subplots_adjust(
        top=0.84,
        bottom=0.14,
        left=0.12,
        right=0.97
    )

    save_path = os.path.join(
        OUTPUT_DIR,
        (
            f"{task_name}_joint_"
            f"latent{latent_dim}D_"
            f"seed{seed}_"
            f"actorlr{actor_lr}_"
            f"control_performance.png"
        )
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------------
    # 终端结果
    # --------------------------------------------------------
    print(
        "\nControl Performance:"
    )

    print(
        f"Average Reward      : "
        f"{average_reward:.2f}"
    )

    print(
        f"Best Reward         : "
        f"{best_reward:.2f}"
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


    print(
        f"\n✅ Control Performance 图：\n"
        f"   {save_path}"
    )

# ============================================================
# 8. 主程序
# ============================================================
def main():
    if not os.path.exists(
        CSV_PATH
    ):
        print(
            f"❌ 找不到 CSV：\n"
            f"{CSV_PATH}"
        )
        return

    df = pd.read_csv(
        CSV_PATH
    )

    print(
        "\nCSV columns:"
    )
    for column in df.columns:
        print(
            f"  - {column}"
        )

    # --------------------------------------------------------
    # 自动读取实验参数
    # --------------------------------------------------------
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

    latent_dim = get_first_valid_value(
        df,
        "latent_action_dim",
        LATENT_ACTION_DIM
    )

    try:
        seed = int(seed)
    except (ValueError, TypeError):
        pass

    try:
        latent_dim = int(
            latent_dim
        )
    except (ValueError, TypeError):
        pass

    # ========================================================
    # 输出 loss 统计
    # ========================================================
    print(
        "\n"
        + "=" * 70
    )

    print(
        "Joint Training Diagnostic Summary"
    )

    print(
        "=" * 70
    )

    wm_col = find_column(
        df,
        [
            "wm_loss",
            "world_model_loss"
        ]
    )

    dynamics_col = find_column(
        df,
        [
            "dynamics_loss",
            "dyn_loss"
        ]
    )

    reward_col = find_column(
        df,
        [
            "reward_loss",
            "wm_reward_loss"
        ]
    )

    recon_col = find_column(
        df,
        [
            "jsae_recon_mse",
            "recon_mse",
            "jae_recon_mse"
        ]
    )

    print_metric_statistics(
        df,
        wm_col
    )

    print_metric_statistics(
        df,
        dynamics_col
    )

    print_metric_statistics(
        df,
        reward_col
    )

    print_metric_statistics(
        df,
        recon_col
    )

    print(
        "=" * 70
    )

    # ========================================================
    # 三类指标分别画图
    # ========================================================
    plot_world_model_losses(
        df,
        task_name,
        seed,
        latent_dim
    )

    plot_jsae_reconstruction(
        df,
        task_name,
        seed,
        latent_dim
    )

    plot_control_performance(
        df,
        task_name,
        seed,
        latent_dim
    )

    print(
        "\n🎉 Joint 实验诊断完成。"
    )

    print(
        f"所有图片保存在：\n"
        f"{OUTPUT_DIR}"
    )

# ============================================================
# 9. Run
# ============================================================
if __name__ == "__main__":
    main()