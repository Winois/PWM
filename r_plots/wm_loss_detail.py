import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 画 world model loss 的细曲线图，包含：
# 1. wm_loss
# 2. dynamics_loss
# 3. reward_loss
# ===============================
# 配置
# ===============================

CSV_PATH = (
    "/data3/zhangwanying/PWM/results/"
    "joint_test_6D_seed42/logs/"
    "cheetah-run-backwards_results.csv"
)


OUTPUT_DIR = (
    "/data3/zhangwanying/PWM/r_plots/"
    "figures/world_model"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


TASK_NAME = "cheetah-run-backwards"


# ===============================
# 平滑
# ===============================

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

def parse_tensor(x):
    """
    Convert:
    tensor(39.4192, device='cuda:0', grad_fn=<...>)
    to:
    39.4192
    """

    if isinstance(x, (int, float)):
        return float(x)

    x = str(x)

    if "tensor(" in x:

        x = (
            x.replace("tensor(", "")
             .split(",")[0]
             .replace(")", "")
        )

    try:
        return float(x)

    except:
        return np.nan

# ===============================
# 读取
# ===============================

df = pd.read_csv(CSV_PATH)


metrics = [
    "wm_loss",
    "dynamics_loss",
    "reward_loss"
]

data = {}

for m in metrics:

    if m not in df.columns:

        print(
            f"Missing {m}"
        )

        continue


    temp = (
        df[
            [
                "iteration",
                m
            ]
        ]
        .dropna()
        .sort_values(
            "iteration"
        )
    )


    x = temp["iteration"].values

    y = temp[m].apply(parse_tensor).values

    data[m] = (x,y)


    print(
        m,
        "mean=",
        y.mean(),
        "min=",
        y.min(),
        "max=",
        y.max()
    )



# ===============================
# Linear scale
# ===============================

plt.figure(
    figsize=(8,4),
    dpi=200
)


for name,(x,y) in data.items():

    plt.plot(
        x,
        smooth(y),
        linewidth=2,
        label=name
    )


plt.xlabel(
    "Training Iterations"
)

plt.ylabel(
    "Loss"
)


plt.title(
    f"{TASK_NAME}: World Model Loss"
)


plt.grid(
    linestyle="--",
    alpha=0.4
)


plt.legend(
    loc="best"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "world_model_loss_linear.png"
    ),
    dpi=300
)


plt.close()



# ===============================
# Log scale
# ===============================

plt.figure(
    figsize=(8,4),
    dpi=200
)


for name,(x,y) in data.items():

    # 防止 log(0)

    y_safe = np.maximum(
        y,
        1e-8
    )


    plt.plot(
        x,
        smooth(y_safe),
        linewidth=2,
        label=name
    )


plt.yscale(
    "log"
)


plt.xlabel(
    "Training Iterations"
)

plt.ylabel(
    "Loss (log scale)"
)


plt.title(
    f"{TASK_NAME}: World Model Loss (Log Scale)"
)


plt.grid(
    linestyle="--",
    alpha=0.4
)


plt.legend(
    loc="best"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "world_model_loss_log.png"
    ),
    dpi=300
)


plt.close()
print("Done.")