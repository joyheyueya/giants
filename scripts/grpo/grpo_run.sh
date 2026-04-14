#!/usr/bin/env bash

set -euo pipefail
set -x

: "${TRAIN_DATA_DIR:?Set TRAIN_DATA_DIR to a directory containing train.parquet.}"
: "${EVAL_DATA_DIR:?Set EVAL_DATA_DIR to a directory containing test.parquet.}"
: "${BASE_MODEL:?Set BASE_MODEL to a Hugging Face identifier or local checkpoint.}"
: "${PROJECT_NAME:=insight-anticipation-grpo}"
: "${EXPERIMENT_NAME:=qwen3-4b-grpo-similarity}"
: "${TRAINER_DEFAULT_LOCAL_DIR:=outputs/grpo/${EXPERIMENT_NAME}}"
: "${N_GPUS:=4}"
: "${MAX_MODEL_LEN:=1296}"
: "${MAX_PROMPT_LENGTH:=3000}"
: "${EPOCHS:=16}"
: "${ROLLOUT_TP_SIZE:=1}"
: "${TRAINER_LOGGERS:=['console']}"
: "${TRAIN_BATCH_SIZE:=64}"
: "${ACTOR_LR:=1e-6}"
: "${PPO_MINI_BATCH_SIZE:=64}"
: "${PPO_MICRO_BATCH_SIZE:=8}"
: "${ROLLOUT_TEMPERATURE:=0.6}"
: "${ROLLOUT_N:=8}"
: "${ROLLOUT_VAL_N:=${ROLLOUT_N}}"
: "${ROLLOUT_GPU_MEMORY_UTILIZATION:=0.6}"
: "${PPO_MAX_TOKEN_LEN_PER_GPU:=32768}"
: "${ROLLOUT_MAX_BATCHED_TOKENS:=32768}"
: "${KL_LOSS_COEF:=0.001}"
: "${ENTROPY_COEFF:=0.002}"
: "${SAVE_FREQ:=10}"
: "${TEST_FREQ:=10}"
: "${TOTAL_TRAINING_STEPS:=2048}"
: "${MAX_EXTRAPOLATION_LENGTH:=16384}"

cmd=(
    python3 -m verl.trainer.main_ppo
    algorithm.adv_estimator=grpo
    "data.train_files=${TRAIN_DATA_DIR}/train.parquet"
    "data.val_files=${EVAL_DATA_DIR}/test.parquet"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_MODEL_LEN}"
    data.filter_overlong_prompts=True
    "actor_rollout_ref.model.path=${BASE_MODEL}"
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.5
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_micro_batch_size=${PPO_MICRO_BATCH_SIZE}"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.only_train_on_positive=False
    actor_rollout_ref.actor.remove_truncated=False
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS}"
    actor_rollout_ref.actor.use_kl_loss=True
    "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    "actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE}"
    actor_rollout_ref.rollout.name=vllm
    "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${ROLLOUT_TEMPERATURE}"
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.val_kwargs.n=${ROLLOUT_VAL_N}"
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=False
    algorithm.use_kl_in_reward=False
    custom_reward_function.path=verl/utils/reward_score/insight_similarity/compute_score.py
    trainer.critic_warmup=0
    "trainer.logger=${TRAINER_LOGGERS}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    trainer.val_before_train=True
    trainer.default_hdfs_dir=null
    "trainer.n_gpus_per_node=${N_GPUS}"
    trainer.nnodes=1
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
    "trainer.default_local_dir=${TRAINER_DEFAULT_LOCAL_DIR}"
    trainer.extrapolation_val=False
    "data.max_extrapolation_length=${MAX_EXTRAPOLATION_LENGTH}"
    "trainer.total_epochs=${EPOCHS}"
)

cmd+=("$@")
"${cmd[@]}"
