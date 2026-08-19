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
        # "cheetah-run-front" \
        # "cheetah-run-back" \
        # "cheetah-jump" \
        # "hopper-hop-backwards" \
        # "reacher-three-easy" \
        # "reacher-three-hard" \
        # "cup-spin" \
        # "pendulum-spin" \
        )

# Command to run for each parameter
for param in "${params[@]}"
do
    echo "Running command with parameter: $param"
    # Replace 'your_command' with the actual command you want to run
    # your_command "$param"
    #python train_multitask.py --multirun -cn config_mt30 general.run_wandb=True wandb.group=mt30 alg=pwm_48M task="$param"
    # 正确的命令应包含 data_dir
python train_multitask.py --multirun -cn config_mt30 general.run_wandb=False alg=pwm_48M_jsae task="$param" general.data_dir=/data3/zhangwanying/PWM/tdmpc2/mt30 general.finetune_wm=False seed_steps=5000 general.checkpoint=/data3/zhangwanying/PWM/models/mt30_48M.pt 
done
