# #!/bin/bash

# List of parameters # 所以是对每个任务单独算吗 对的，单独提取策略
params=(
        # 19 original dmcontrol tasks
        # "walker-stand" \
        # "walker-walk" \
        # "walker-run" \
        # "cheetah-run" \
        # "reacher-easy" \
        # "reacher-hard" \
        # "acrobot-swingup" \
        # "pendulum-swingup" \
        # "cartpole-balance" \
        # "cartpole-balance-sparse" \
        # "cartpole-swingup" \
        # "cartpole-swingup-sparse" \
        # "cup-catch" \
        # "finger-spin" \
        # "finger-turn-easy" \
        # "finger-turn-hard" \
        # "fish-swim" \
        # "hopper-stand" \
        # "hopper-hop" \
        # 11 custom dmcontrol tasks
        #"walker-walk-backwards" \
        #"walker-run-backwards" \
        "cheetah-run-backwards" \
        "cheetah-run-front" \
        "cheetah-run-back" \
        "cheetah-jump" \
        "hopper-hop-backwards" \
        "reacher-three-easy" \
        "reacher-three-hard" \
        "cup-spin" \
        "pendulum-spin" \
        )

# Command to run for each parameter
for param in "${params[@]}"
do
    echo "Running command with parameter: $param"
    # Replace 'your_command' with the actual command you want to run
    # your_command "$param"
    #python train_multitask.py --multirun -cn config_mt30 general.run_wandb=True wandb.group=mt30 alg=pwm_48M task="$param"
    # 正确的命令应包含 data_dir
python train_multitask.py --multirun -cn config_mt30 general.run_wandb=False alg=pwm_48M task="$param" general.data_dir=/data3/zhangwanying/PWM/tdmpc2/mt30 general.finetune_wm=True +seed_steps=5000 general.checkpoint=/data3/zhangwanying/PWM/models/mt30_48M.pt #听说不要这行模型将随机初始化所有参数（包括 JAE、Dynamics 和 Reward），并从 /data3/zhangwanying/PWM/tdmpc2/mt30 里的数据重新开始学习。 # 改了之后用checkpoint就报错了因为没有jae的权重 太好了
    sleep 30 # to ensure that the command has finished before running the next one
done


# #!/bin/bash
# # 新的脚本版本，包含了自动捕获最新训练存档并传递给下一个任务的逻辑
# params=(
#         # 19 original dmcontrol tasks
#         #"walker-stand" \
#         # "walker-walk" \
#         # "walker-run" \
#         # "cheetah-run" \
#         # "reacher-easy" \
#         # "reacher-hard" \
#         # "acrobot-swingup" \
#         # "pendulum-swingup" \
#         # "cartpole-balance" \
#         # "cartpole-balance-sparse" \
#         # "cartpole-swingup" \
#         # "cartpole-swingup-sparse" \
#         # "cup-catch" \
#         # "finger-spin" \
#         # "finger-turn-easy" \
#         # "finger-turn-hard" \
#         # "fish-swim" \
#         # "hopper-stand" \
#         # "hopper-hop" \
#         # 11 custom dmcontrol tasks
#         #"walker-walk-backwards" \
#         #"walker-run-backwards" \
#         "cheetah-run-backwards" \
#         "cheetah-run-front" \
#         "cheetah-run-back" \
#         "cheetah-jump" \
#         "hopper-hop-backwards" \
#         "reacher-three-easy" \
#         "reacher-three-hard" \
#         "cup-spin" \
#         "pendulum-spin" \
#         )

# # 1. 定义一个你专属的、用来存放最新传递存档的路径
# SHARED_CHECKPOINT="/data3/zhangwanying/PWM/models/my_latest_joint_wm.pt"

# # 如果你想完全从头一张白纸开始，在跑脚本前先确保删掉上一次的残留存档
# rm -f $SHARED_CHECKPOINT

# for param in "${params[@]}"
# do
#     echo "========================================="
#     echo "Running command with parameter: $param"
#     echo "========================================="

#     # 2. 判断当前是否有训练好的底子。如果共享存档存在，就加载它；不存在（第一轮）就不加参数
#     if [ -f "$SHARED_CHECKPOINT" ]; then
#         checkpoint_arg="general.checkpoint=$SHARED_CHECKPOINT"
#         echo "发现已有世界模型存档，正在加载继承：$SHARED_CHECKPOINT"
#     else
#         checkpoint_arg=""
#         echo "未发现历史存档，当前任务将从纯随机初始化（白纸）开始训练"
#     fi

#     # 3. 运行训练命令（传入动态的 checkpoint_arg 参数）
#     # 注意：我在这里把 model_dir 强行指定到了当前运行的输出里
#     python train_multitask.py --multirun -cn config_mt30 \
#         general.run_wandb=False \
#         alg=pwm_48M \
#         task="$param" \
#         general.data_dir=/data3/zhangwanying/PWM/tdmpc2/mt30 \
#         general.finetune_wm=True \
#         +seed_steps=0 \
#         $checkpoint_arg

#     # 4. 【核心破局点】：训练完成后，去 Hydra 的输出目录里抓取最新生成的那颗热乎的 checkpoint
#     # 把它复制并重命名覆盖到我们的 SHARED_CHECKPOINT 路径，变成下一个任务的“新底子”
#     # 提示：你需要根据你项目实际生成的 checkpoint 名字（比如 checkpoint.pt 或 model.pt）微调下面这行
#     LATEST_TRAINED_MODEL=$(find ./multirun ./outputs -name "*.pt" -type f -printf "%T@ %p\n" | sort -n | tail -1 | awk '{print $2}')

#     if [ ! -z "$LATEST_TRAINED_MODEL" ]; then
#         cp "$LATEST_TRAINED_MODEL" "$SHARED_CHECKPOINT"
#         echo "成功捕获新存档: $LATEST_TRAINED_MODEL ，已提取并准备传给下一个任务！"
#     else
#         echo "警告：在输出目录中未找到新生成的 .pt 存档，下一个任务可能无法继承！"
#     fi

#     sleep 30
# done
