#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

if [[ -n "${CONDA_ENV_NAME:-}" ]] && command -v conda >/dev/null 2>&1; then
    # Optional convenience for clusters where the environment is not pre-activated.
    # If CONDA_ENV_NAME is unset, we assume the caller already activated the runtime.
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
fi

cd "${REPO_ROOT}"

: "${GPU_IDS:=0,1,2,3}"
IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
: "${N_GPUS:=${#GPU_LIST[@]}}"

: "${BASE_MODEL:=Qwen/Qwen3-4B}"
: "${TRAIN_DATA_DIR:=${REPO_ROOT}/data/insight_anticipation_grpo}"
: "${EVAL_DATA_DIR:=${TRAIN_DATA_DIR}}"
: "${PROJECT_NAME:=insight-anticipation-grpo}"
: "${EXPERIMENT_NAME:=qwen3-4b-grpo-similarity}"
: "${TRAINER_DEFAULT_LOCAL_DIR:=${REPO_ROOT}/outputs/grpo/${EXPERIMENT_NAME}}"
: "${ROLLOUT_TP_SIZE:=1}"
: "${MAX_MODEL_LEN:=1296}"
: "${MAX_PROMPT_LENGTH:=3000}"
: "${EPOCHS:=16}"
: "${DRY_RUN:=false}"
: "${VLLM_ATTENTION_BACKEND:=XFORMERS}"
: "${HF_HOME:=${REPO_ROOT}/.cache/huggingface}"
: "${HF_DATASETS_CACHE:=${HF_HOME}/datasets}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    : "${TRAINER_LOGGERS:=['console','wandb']}"
else
    : "${TRAINER_LOGGERS:=['console']}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export N_GPUS
export BASE_MODEL
export TRAIN_DATA_DIR
export EVAL_DATA_DIR
export ROLLOUT_TP_SIZE
export EXPERIMENT_NAME
export PROJECT_NAME
export MAX_MODEL_LEN
export MAX_PROMPT_LENGTH
export EPOCHS
export TRAINER_DEFAULT_LOCAL_DIR
export VLLM_ATTENTION_BACKEND
export HF_HOME
export HF_DATASETS_CACHE
export TRAINER_LOGGERS

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRAINER_DEFAULT_LOCAL_DIR}"

for required_file in "${TRAIN_DATA_DIR}/train.parquet" "${EVAL_DATA_DIR}/test.parquet"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing required file: ${required_file}" >&2
        exit 1
    fi
done

echo "Repository root: ${REPO_ROOT}"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Base model: ${BASE_MODEL}"
echo "Train set: ${TRAIN_DATA_DIR}/train.parquet"
echo "Eval set: ${EVAL_DATA_DIR}/test.parquet"
echo "Run name: ${EXPERIMENT_NAME}"
echo "Output dir: ${TRAINER_DEFAULT_LOCAL_DIR}"

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "Dry run enabled. Skipping launch."
    exit 0
fi

bash "${SCRIPT_DIR}/grpo_run.sh" "$@"
