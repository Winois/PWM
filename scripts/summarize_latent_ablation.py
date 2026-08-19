# 计算几个seed的mean和std用的

#命令：
#cd /data3/zhangwanying/PWM/scripts
# python summarize_latent_ablation.py \
#/data3/zhangwanying/PWM/results/latent_ablation_20260817_081317

import os
import re
import glob
import argparse

import numpy as np
import pandas as pd


def normalized_auc(x, y):
    """
    Normalized AUC:
    learning curve 的面积 / iteration 范围
    这样量纲仍然是 reward，便于和 Average Reward 比较。
    """
    if len(x) < 2:
        return np.nan

    order = np.argsort(x)
    x = np.asarray(x)[order]
    y = np.asarray(y)[order]

    x_range = x[-1] - x[0]
    if x_range <= 0:
        return np.nan

    return np.trapezoid(y, x) / x_range


def parse_run_name(run_name):
    """
    支持：
      PWM_seed_42
      JSAE_4D_seed_42
      JSAE_16D_seed_43
      JSAE_64D_seed_44
    """

    pwm_match = re.fullmatch(r"PWM_seed_(\d+)", run_name)
    if pwm_match:
        return {
            "algorithm": "PWM",
            "latent_action_dim": np.nan,
            "seed": int(pwm_match.group(1)),
        }

    jsae_match = re.fullmatch(
        r"JSAE_(\d+)D_seed_(\d+)",
        run_name
    )

    if jsae_match:
        return {
            "algorithm": "JSAE",
            "latent_action_dim": int(jsae_match.group(1)),
            "seed": int(jsae_match.group(2)),
        }

    return None


def summarize_one_run(run_dir):
    run_name = os.path.basename(run_dir)
    info = parse_run_name(run_name)

    if info is None:
        return None

    csv_files = glob.glob(
        os.path.join(run_dir, "logs", "*results.csv")
    )

    if not csv_files:
        print(f"[WARNING] No results CSV: {run_dir}")
        return None

    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)

    if "episode_reward" not in df.columns:
        print(f"[WARNING] episode_reward missing: {csv_path}")
        return None

    # 只使用真正进行了 evaluation 的行
    eval_df = df[
        df["episode_reward"].notna()
    ].copy()

    if len(eval_df) == 0:
        print(f"[WARNING] No valid rewards: {csv_path}")
        return None

    rewards = eval_df["episode_reward"].to_numpy()
    iterations = eval_df["iteration"].to_numpy()

    # ---------------------------------
    # 单个 seed 的 reward 统计
    # ---------------------------------

    avg_reward = np.mean(rewards)
    reward_std = np.std(rewards, ddof=1) if len(rewards) > 1 else np.nan

    best_reward = np.max(rewards)

    auc = normalized_auc(
        iterations,
        rewards,
    )

    # 最后 5 次 evaluation 的平均 reward
    # eval_freq=200 时，大致对应最后 1000 iterations
    last_n = min(5, len(rewards))
    final_rewards = rewards[-last_n:]

    final_reward_mean = np.mean(final_rewards)
    final_reward_std = (
        np.std(final_rewards, ddof=1)
        if len(final_rewards) > 1
        else np.nan
    )

    result = {
        **info,
        "run_name": run_name,
        "csv_path": csv_path,
        "num_eval_points": len(rewards),

        "average_reward": avg_reward,
        "reward_std_within_run": reward_std,

        "best_reward": best_reward,

        "normalized_auc": auc,

        "final5_reward_mean": final_reward_mean,
        "final5_reward_std": final_reward_std,
    }

    # ---------------------------------
    # JSAE reconstruction statistics
    # ---------------------------------

    if "jsae_recon_mse" in df.columns:
        mse = df["jsae_recon_mse"].dropna()

        if len(mse) > 0:
            result["recon_mse_mean"] = mse.mean()
            result["recon_mse_std"] = (
                mse.std(ddof=1)
                if len(mse) > 1
                else np.nan
            )

    if "jsae_recon_mae" in df.columns:
        mae = df["jsae_recon_mae"].dropna()

        if len(mae) > 0:
            result["recon_mae_mean"] = mae.mean()
            result["recon_mae_std"] = (
                mae.std(ddof=1)
                if len(mae) > 1
                else np.nan
            )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "result_root",
        help="latent_ablation_xxx experiment directory"
    )

    args = parser.parse_args()

    result_root = os.path.abspath(args.result_root)

    print("=" * 70)
    print("Reading experiments from:")
    print(result_root)
    print("=" * 70)

    run_dirs = [
        p for p in glob.glob(
            os.path.join(result_root, "*")
        )
        if os.path.isdir(p)
    ]

    results = []

    for run_dir in sorted(run_dirs):
        result = summarize_one_run(run_dir)

        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError(
            f"No valid experiments found in {result_root}"
        )

    runs_df = pd.DataFrame(results)

    # 排序
    runs_df["_latent_sort"] = (
        runs_df["latent_action_dim"]
        .fillna(-1)
    )

    runs_df = runs_df.sort_values(
        ["algorithm", "_latent_sort", "seed"]
    ).drop(columns=["_latent_sort"])

    # =====================================
    # 保存每个 seed 的详细统计
    # =====================================

    run_summary_path = os.path.join(
        result_root,
        "summary_per_seed.csv"
    )

    runs_df.to_csv(
        run_summary_path,
        index=False
    )

    # =====================================
    # 跨 seed 统计 mean ± std
    # =====================================

    runs_df["method"] = runs_df.apply(
        lambda row:
            "PWM"
            if row["algorithm"] == "PWM"
            else f"JSAE-{int(row['latent_action_dim'])}D",
        axis=1,
    )

    metrics_to_aggregate = [
        "average_reward",
        "best_reward",
        "normalized_auc",
        "final5_reward_mean",
    ]

    # reconstruction 指标存在时才统计
    for col in [
        "recon_mse_mean",
        "recon_mae_mean",
    ]:
        if col in runs_df.columns:
            metrics_to_aggregate.append(col)

    summary_rows = []

    for method, group in runs_df.groupby(
        "method",
        sort=False
    ):
        row = {
            "method": method,
            "num_seeds": len(group),
        }

        for metric in metrics_to_aggregate:
            values = group[metric].dropna()

            if len(values) == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                continue

            row[f"{metric}_mean"] = values.mean()

            # 跨 seed sample std
            row[f"{metric}_std"] = (
                values.std(ddof=1)
                if len(values) > 1
                else np.nan
            )

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # 方法排序：PWM, 4D, 6D, 8D, 16D, 64D
    desired_order = {
        "PWM": 0,
        "JSAE-4D": 1,
        "JSAE-6D": 2,
        "JSAE-8D": 3,
        "JSAE-16D": 4,
        "JSAE-64D": 5,
    }

    summary_df["_order"] = summary_df["method"].map(
        desired_order
    ).fillna(999)

    summary_df = (
        summary_df
        .sort_values("_order")
        .drop(columns="_order")
    )

    summary_path = os.path.join(
        result_root,
        "summary_by_method.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # =====================================
    # Terminal 输出
    # =====================================

    print("\n" + "=" * 70)
    print("PER-SEED RESULTS")
    print("=" * 70)

    display_cols = [
        "method",
        "seed",
        "average_reward",
        "best_reward",
        "normalized_auc",
        "final5_reward_mean",
    ]

    print(
        runs_df[display_cols]
        .round(2)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("MEAN ± STD ACROSS SEEDS")
    print("=" * 70)

    for _, row in summary_df.iterrows():
        print(f"\n{row['method']}  (n={int(row['num_seeds'])})")

        print(
            "  Average Reward : "
            f"{row['average_reward_mean']:.2f} ± "
            f"{row['average_reward_std']:.2f}"
        )

        print(
            "  Best Reward    : "
            f"{row['best_reward_mean']:.2f} ± "
            f"{row['best_reward_std']:.2f}"
        )

        print(
            "  Normalized AUC : "
            f"{row['normalized_auc_mean']:.2f} ± "
            f"{row['normalized_auc_std']:.2f}"
        )

        print(
            "  Final-5 Reward : "
            f"{row['final5_reward_mean_mean']:.2f} ± "
            f"{row['final5_reward_mean_std']:.2f}"
        )

    print("\nSaved:")
    print(run_summary_path)
    print(summary_path)


if __name__ == "__main__":
    main()