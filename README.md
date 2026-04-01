# Insight Anticipation

<p align="center">
  <img src="assets/fig1.png" alt="Insight Anticipation overview" width="100%">
</p>

<p align="center">
  <strong>Anonymous code release for literature-grounded scientific insight prediction.</strong>
</p>

Insight anticipation asks a model to predict a downstream paper's core contribution from the summaries of its two foundational parent papers. This repository contains the training and reward code used for our reinforcement-learning experiments, packaged as a clean, submission-ready release on top of the `verl` training stack.

## Why this repo exists

Scientific progress is often evolutionary: new papers synthesize prior ideas into focused, mechanistic advances. We operationalize that process as a grounded generation task and train language models to anticipate those advances rather than brainstorm in the abstract.

This release includes:

- A portable supervised fine-tuning entrypoint for bootstrapping the policy.
- A GRPO training pipeline that optimizes an LM-judge similarity reward.
- An anonymized reward implementation for scoring predicted insights against held-out downstream contributions.
- A trimmed codebase with credentials, personal paths, upload utilities, and local-only artifacts removed.

## At a glance

| Component | Purpose | Entry point |
| --- | --- | --- |
| SFT | Bootstrap a policy on parent-summary to insight pairs | `scripts/sft_mult_4gpu_insight_ampere.sh` |
| GRPO | Optimize the policy with similarity-based rewards | `scripts/grpo/train_local_insight_similarity_ampere.sh` |
| Reward judge | Score generated insights against ground truth | `verl/utils/reward_score/insight_similarity/compute_score.py` |
| Core training stack | Distributed training and rollout infrastructure | `verl/` |

## Main result path

If you only need the script used for the primary RL experiments, start with:

```bash
bash scripts/grpo/train_local_insight_similarity_ampere.sh
```

That script is now environment-driven and no longer depends on institution-specific paths, credentials, or cluster setup.

## Installation

We recommend Python 3.10 and a fresh environment.

```bash
conda create -n insight-anticipation python=3.10
conda activate insight-anticipation

pip install -e .
pip install -r requirements.txt
pip install vllm==0.8.4
```

Depending on your CUDA and PyTorch stack, you may also want a local `flash-attn` install that matches your environment.

## Data layout

The release expects dataset directories of the form:

```text
data/
  insight_anticipation_sft/
    train.parquet
    test.parquet
  insight_anticipation_grpo/
    train.parquet
    test.parquet
```

Expected columns:

- SFT data uses `query` and `completion`.
- GRPO data follows the `verl` RL format and should include a prompt field, `data_source`, and `reward_model.ground_truth`.
- Optional `extra_info` metadata can be stored alongside each GRPO example for debugging or evaluation.

See `verl/utils/dataset/README.md` for the base RL dataset contract used by `verl`.

## Running SFT

```bash
BASE_MODEL=Qwen/Qwen3-4B \
TRAIN_DATA_DIR=$PWD/data/insight_anticipation_sft \
EVAL_DATA_DIR=$PWD/data/insight_anticipation_sft \
GPU_IDS=0,1,2,3 \
EXPERIMENT_NAME=qwen3-4b-sft \
bash scripts/sft_mult_4gpu_insight_ampere.sh
```

Important knobs:

- `BASE_MODEL`: Hugging Face model ID or local checkpoint.
- `GPU_IDS`: Comma-separated GPU list.
- `TRAINER_DEFAULT_LOCAL_DIR`: Output directory for checkpoints and logs.
- `TRAINER_LOGGERS`: Hydra list string, for example `['console']` or `['console','wandb']`.

## Running GRPO

```bash
BASE_MODEL=$PWD/outputs/sft/qwen3-4b-sft \
TRAIN_DATA_DIR=$PWD/data/insight_anticipation_grpo \
EVAL_DATA_DIR=$PWD/data/insight_anticipation_grpo \
GPU_IDS=0,1,2,3 \
EXPERIMENT_NAME=qwen3-4b-grpo-similarity \
ROLLOUT_TP_SIZE=1 \
bash scripts/grpo/train_local_insight_similarity_ampere.sh
```

The GRPO launcher exposes the most important hyperparameters as environment variables, including:

- `MAX_PROMPT_LENGTH`
- `MAX_MODEL_LEN`
- `ROLLOUT_N`
- `ACTOR_LR`
- `TRAIN_BATCH_SIZE`
- `TOTAL_TRAINING_STEPS`

## Judge configuration

The insight-similarity reward uses `google-genai` and supports either:

1. Gemini API keys via `GEMINI_API_KEY` or `GOOGLE_API_KEY`
2. Vertex AI via `GOOGLE_CLOUD_PROJECT` plus your usual Google credentials flow

Minimal examples:

```bash
export GEMINI_API_KEY=...
```

or

```bash
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

Optional reward controls:

- `INSIGHT_SIMILARITY_MODEL` defaults to `gemini-2.5-flash`
- `INSIGHT_SIMILARITY_MAX_TOKENS` defaults to `8192`
- `INSIGHT_SIMILARITY_DEBUG_DIR` writes prompt/response traces for debugging

## What changed for release

- Removed hard-coded tokens, usernames, email addresses, and service-account paths.
- Removed upload helpers, notebooks, and cluster-local experiment leftovers that were not needed for the submission artifact.
- Replaced site-specific shell scripts with portable launchers rooted at the current checkout.
- Kept the repo centered on the insight-anticipation training path rather than unrelated side experiments.

## Notes for anonymous review

- External dataset and checkpoint links are intentionally omitted here.
- The code is organized so anonymous artifacts can be mounted locally without modifying source files.
- All experiment launchers are now parameterized through environment variables.

## Acknowledgments

This project builds on the `verl` reinforcement-learning stack. Upstream license and notices are preserved in this repository.
