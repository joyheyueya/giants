#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

if [[ -n "${CONDA_ENV_NAME:-}" ]] && command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
fi

cd "${REPO_ROOT}"

: "${GPU_IDS:=0,1,2,3}"
IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
: "${N_GPUS:=${#GPU_LIST[@]}}"

: "${BASE_MODEL:=Qwen/Qwen3-4B}"
: "${TRAIN_DATA_DIR:=${REPO_ROOT}/data/insight_anticipation_sft}"
: "${EVAL_DATA_DIR:=${TRAIN_DATA_DIR}}"
: "${PROJECT_NAME:=insight-anticipation-sft}"
: "${EXPERIMENT_NAME:=qwen3-4b-sft}"
: "${TRAINER_DEFAULT_LOCAL_DIR:=${REPO_ROOT}/outputs/sft/${EXPERIMENT_NAME}}"
: "${TRAINER_DEFAULT_HDFS_DIR:=null}"
: "${PROMPT_KEY:=query}"
: "${RESPONSE_KEY:=completion}"
: "${TRAIN_BATCH_SIZE:=64}"
: "${MICRO_BATCH_SIZE:=4}"
: "${MICRO_BATCH_SIZE_PER_GPU:=1}"
: "${MAX_LENGTH:=8192}"
: "${TOTAL_EPOCHS:=10}"
: "${VALIDATION_INTERVAL:=10}"
: "${LEARNING_RATE:=1e-6}"
: "${TRUNCATION:=right}"
: "${APPLY_CHAT_TEMPLATE:=True}"
: "${MASTER_PORT:=29500}"
: "${HF_HOME:=${REPO_ROOT}/.cache/huggingface}"
: "${HF_DATASETS_CACHE:=${HF_HOME}/datasets}"
: "${DRY_RUN:=false}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    : "${TRAINER_LOGGERS:=['console','wandb']}"
else
    : "${TRAINER_LOGGERS:=['console']}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi

if command -v lsof >/dev/null 2>&1; then
    while lsof -Pi :"${MASTER_PORT}" -sTCP:LISTEN -t >/dev/null 2>&1; do
        MASTER_PORT=$((MASTER_PORT + 1))
    done
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export HF_HOME
export HF_DATASETS_CACHE

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRAINER_DEFAULT_LOCAL_DIR}"

for required_file in "${TRAIN_DATA_DIR}/train.parquet" "${EVAL_DATA_DIR}/test.parquet"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing required file: ${required_file}" >&2
        exit 1
    fi
done

cmd=(
    torchrun
    "--nproc_per_node=${N_GPUS}"
    "--master_port=${MASTER_PORT}"
    -m
    verl.trainer.fsdp_sft_trainer
    "data.train_files=${TRAIN_DATA_DIR}/train.parquet"
    "data.val_files=${EVAL_DATA_DIR}/test.parquet"
    "data.prompt_key=${PROMPT_KEY}"
    "data.truncation=${TRUNCATION}"
    "data.apply_chat_template=${APPLY_CHAT_TEMPLATE}"
    "data.response_key=${RESPONSE_KEY}"
    "data.micro_batch_size=${MICRO_BATCH_SIZE}"
    "data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_length=${MAX_LENGTH}"
    "model.partial_pretrain=${BASE_MODEL}"
    "trainer.default_hdfs_dir=${TRAINER_DEFAULT_HDFS_DIR}"
    "trainer.default_local_dir=${TRAINER_DEFAULT_LOCAL_DIR}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.logger=${TRAINER_LOGGERS}"
    "trainer.validation_interval=${VALIDATION_INTERVAL}"
    "optim.lr=${LEARNING_RATE}"
    model.enable_gradient_checkpointing=True
    model.use_liger=True
)

cmd+=("$@")

echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Base model: ${BASE_MODEL}"
echo "Train set: ${TRAIN_DATA_DIR}/train.parquet"
echo "Eval set: ${EVAL_DATA_DIR}/test.parquet"
echo "Run name: ${EXPERIMENT_NAME}"
echo "Output dir: ${TRAINER_DEFAULT_LOCAL_DIR}"
echo "Master port: ${MASTER_PORT}"

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "Dry run enabled. Skipping launch."
    exit 0
fi

"${cmd[@]}"
