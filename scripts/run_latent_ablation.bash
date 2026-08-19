#!/bin/bash

set -e

# ============================================================
# PWM / JSAE latent action dimension ablation 
# 看不同 latent action dimension 对性能的影响。baseline是PWM，其他是PWM+JSAE
# ============================================================

TASK="cheetah-run-backwards"

LATENT_DIMS=(4 6 8 16 64)

SEEDS=(42 43 44)

DATA_DIR="/data3/zhangwanying/PWM/tdmpc2/mt30"
CHECKPOINT="/data3/zhangwanying/PWM/models/mt30_48M.pt"

# 每次运行生成新的实验总目录，避免覆盖以前的结果
#RUN_TAG=$(date +"%Y%m%d_%H%M%S")
#RESULT_ROOT="/data3/zhangwanying/PWM/results/latent_ablation_${RUN_TAG}"
RESULT_ROOT="/data3/zhangwanying/PWM/results/latent_ablation_20260817_091657" # 这个训一半断了 接着来

mkdir -p "${RESULT_ROOT}"

echo "============================================================"
echo "Experiment root:"
echo "${RESULT_ROOT}"
echo "============================================================"


# ============================================================
# 1. PWM BASELINE
# ============================================================

for seed in "${SEEDS[@]}"
do

    RUN_NAME="PWM_seed_${seed}"
    RUN_DIR="${RESULT_ROOT}/${RUN_NAME}"
    LOG_DIR="${RUN_DIR}/logs"

    FINAL_MODEL="${LOG_DIR}/${TASK}_PWM_seed_${seed}_model_final.pt"
    RESULT_CSV="${LOG_DIR}/${TASK}_PWM_seed_${seed}_results.csv"

    if [ -f "${FINAL_MODEL}" ] && [ -f "${RESULT_CSV}" ]; then
        echo "============================================================"
        echo "SKIP: PWM seed ${seed} already completed"
        echo "============================================================"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "Algorithm : PWM"
    echo "Task      : ${TASK}"
    echo "Seed      : ${seed}"
    echo "Output    : ${RUN_DIR}"
    echo "============================================================"

    python train_multitask.py \
        -cn config_mt30 \
        general.run_wandb=False \
        alg=pwm_48M \
        task="${TASK}" \
        general.seed="${seed}" \
        general.data_dir="${DATA_DIR}" \
        general.finetune_wm=False \
        seed_steps=5000 \
        general.checkpoint="${CHECKPOINT}" \
        hydra.run.dir="${RUN_DIR}"

    LOG_DIR="${RUN_DIR}/logs"

    # CSV 改成可直接识别的名字
    if [ -f "${LOG_DIR}/${TASK}_results.csv" ]; then
        mv "${LOG_DIR}/${TASK}_results.csv" \
           "${LOG_DIR}/${TASK}_PWM_seed_${seed}_results.csv"
    fi

    # 最终模型改名
    if [ -f "${LOG_DIR}/model_final.pt" ]; then
        mv "${LOG_DIR}/model_final.pt" \
           "${LOG_DIR}/${TASK}_PWM_seed_${seed}_model_final.pt"
    fi

done


# ============================================================
# 2. PWM + JSAE
# ============================================================

for latent_dim in "${LATENT_DIMS[@]}"
do

    for seed in "${SEEDS[@]}"
    do

        RUN_NAME="JSAE_${latent_dim}D_seed_${seed}"
        RUN_DIR="${RESULT_ROOT}/${RUN_NAME}"
        LOG_DIR="${RUN_DIR}/logs"

        FINAL_MODEL="${LOG_DIR}/${TASK}_JSAE_${latent_dim}D_seed_${seed}_model_final.pt"
        RESULT_CSV="${LOG_DIR}/${TASK}_JSAE_${latent_dim}D_seed_${seed}_results.csv"

        if [ -f "${FINAL_MODEL}" ] && [ -f "${RESULT_CSV}" ]; then
            echo "============================================================"
            echo "SKIP: JSAE ${latent_dim}D seed ${seed} already completed"
            echo "============================================================"
            continue
        fi

        echo ""
        echo "============================================================"
        echo "Algorithm  : PWM + JSAE"
        echo "Task       : ${TASK}"
        echo "Latent dim : ${latent_dim}"
        echo "Seed       : ${seed}"
        echo "Output     : ${RUN_DIR}"
        echo "============================================================"

        python train_multitask.py \
            -cn config_mt30 \
            general.run_wandb=False \
            alg=pwm_48M_jsae \
            alg.jsae_config.latent_action_dim="${latent_dim}" \
            task="${TASK}" \
            general.seed="${seed}" \
            general.data_dir="${DATA_DIR}" \
            general.finetune_wm=False \
            seed_steps=5000 \
            general.checkpoint="${CHECKPOINT}" \
            hydra.run.dir="${RUN_DIR}"

        LOG_DIR="${RUN_DIR}/logs"

        # CSV 改名
        if [ -f "${LOG_DIR}/${TASK}_results.csv" ]; then
            mv "${LOG_DIR}/${TASK}_results.csv" \
               "${LOG_DIR}/${TASK}_JSAE_${latent_dim}D_seed_${seed}_results.csv"
        fi

        # 最终模型改名
        if [ -f "${LOG_DIR}/model_final.pt" ]; then
            mv "${LOG_DIR}/model_final.pt" \
               "${LOG_DIR}/${TASK}_JSAE_${latent_dim}D_seed_${seed}_model_final.pt"
        fi

    done
done


echo ""
echo "============================================================"
echo "ALL EXPERIMENTS FINISHED"
echo "Results:"
echo "${RESULT_ROOT}"
echo "============================================================"