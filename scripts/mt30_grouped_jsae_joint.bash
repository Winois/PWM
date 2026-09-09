#!/bin/bash

set -e

# Grouped JSAE + Joint Latent World Model smoke experiment.
TASK="cheetah-run-backwards"
LATENT_DIMS=(6) # e.g. (4 6 8 16)
SEEDS=(42)       # e.g. (42 43 44)

DATA_DIR="/data3/zhangwanying/PWM/tdmpc2/mt30"
CHECKPOINT="/data3/zhangwanying/PWM/models/mt30_48M.pt"
RESULT_ROOT="/data3/zhangwanying/PWM/results/grouped_joint_test"

for latent_dim in "${LATENT_DIMS[@]}"
do
    for seed in "${SEEDS[@]}"
    do
        RUN_DIR="${RESULT_ROOT}/${TASK}_${latent_dim}D_seed${seed}"

        echo "=============================================="
        echo "Running Grouped JSAE + Joint Latent WM"
        echo "Task: ${TASK}"
        echo "Latent dim: ${latent_dim}"
        echo "Seed: ${seed}"
        echo "Output: ${RUN_DIR}"
        echo "=============================================="

        python train_multitask.py \
            -cn config_mt30 \
            general.run_wandb=False \
            alg=pwm_48M_grouped_jsae_joint \
            task="${TASK}" \
            general.seed="${seed}" \
            general.data_dir="${DATA_DIR}" \
            general.finetune_wm=True \
            seed_steps=5000 \
            general.checkpoint="${CHECKPOINT}" \
            alg.jsae_config.latent_action_dim="${latent_dim}" \
            hydra.run.dir="${RUN_DIR}"
    done
done
