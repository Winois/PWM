import os
import pandas as pd
import matplotlib.pyplot as plt

# ================= 配置区 =================
# 1. 替换成你当前想要画图的那个真实的 CSV 文件路径
CSV_PATH = "/data3/zhangwanying/PWM/scripts/multirun/2026-06-10/06-48-09/0/logs/cheetah-run-backwards_results.csv"
# 2. 生成的图片保存名字
SAVE_IMAGE_NAME = "cheetah_run_backwards_reward_curve(6.10).png"
TASK_NAME = "cheetah-run-backwards"
# ==========================================

def plot_reward():
    if not os.path.exists(CSV_PATH):
        print(f"❌ 错误：没找到 CSV 文件，请检查路径是否正确：\n{CSV_PATH}")
        return

    # 1. 读取 CSV 数据
    df = pd.read_csv(CSV_PATH)

    # 2. 【核心清洗】：因为 episode_reward 很多行是空的，我们要剔除掉没有评估奖励的空行
    # 同时确保我们只拿需要的两列：iteration 和 episode_reward
    reward_data = df[['iteration', 'episode_reward']].dropna()

    if len(reward_data) == 0:
        print("⚠️ 警告：该 CSV 文件中没有找到任何有效的 episode_reward 数据！请确认任务是否已经运行到产生奖励的阶段。")
        return

    # 3. 提取 X 和 Y
    x = reward_data['iteration']
    y = reward_data['episode_reward']

    print(f"📊 成功提取到 {len(y)} 个有效奖励记录点，正在绘制...")

    # 4. 开始画图
    plt.figure(figsize=(9, 5), dpi=150) # 设置画布大小和清晰度

    # 画折线图，带小圆点标记
    plt.plot(x, y, marker='o', color='#1f77b4', linestyle='-', linewidth=2, markersize=5, label='Episode Reward')

    # 5. 美化图表
    plt.title(TASK_NAME, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Training Iterations ", fontsize=11, labelpad=10) # (横坐标：轮数/步数)
    plt.ylabel("Episode Reward ", fontsize=11, labelpad=10) # (纵坐标：环境真实奖励)
    plt.grid(True, linestyle='--', alpha=0.6) # 加网格线方便看数字
    plt.legend(loc="lower right") # 标签放右下角

    # 调整布局防止文字被切掉
    plt.tight_layout()

    # 6. 保存图片
    plt.savefig(SAVE_IMAGE_NAME)
    plt.close()

    print(f"🎉 绘制成功！图片已保存为：{SAVE_IMAGE_NAME} ，快去查看机器人的得分有没有一路上扬吧！")

if __name__ == "__main__":
    plot_reward()