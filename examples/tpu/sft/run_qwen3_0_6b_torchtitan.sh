#!/usr/bin/env bash
# Copyright (c) 2026 Google LLC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SFT of Qwen3-0.6B on GSM8K with the TorchTitan engine, on one TPU v6e-8 slice
# (2 hosts x 4 chips). Override NNODES_TRAINER / N_CHIPS_TRAINER for other
# topologies, e.g. a single-host v6e-4 slice:
#
#   NNODES_TRAINER=1 N_CHIPS_TRAINER=4 DATA_PARALLEL_SHARD_SIZE=4 \
#     bash examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh
#
# Parallelism: pure FSDP2 (TP=1). Do not raise tensor_parallel_size on TPU
# without re-testing, see examples/tpu/sft/README.md.

set -xeuo pipefail

export VERL_PLATFORM=tpu
export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export PYTHONUNBUFFERED=1
# the TPU runtime holds most of the host RAM, let Ray keep scheduling anyway
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false"

project_name="${PROJECT_NAME:-verl_tpu_sft}"
exp_name="${EXPERIMENT_NAME:-qwen3_0.6b_gsm8k_sft_torchtitan}"

# Data and model locations, typically a bucket mounted into every pod
DATA_HOME="${DATA_HOME:-${HOME}/data}"
MODEL_PATH="${MODEL_PATH:-${DATA_HOME}/assets/hf/Qwen3-0.6B}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_HOME}/gsm8k_sft/train.parquet}"
TEST_FILE="${TEST_FILE:-${DATA_HOME}/gsm8k_sft/test.parquet}"

# TPU slice topology: hosts x chips per host
export NNODES_TRAINER="${NNODES_TRAINER:-2}"
export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-4}"
TOTAL_TRAINER_CHIPS=$((NNODES_TRAINER * N_CHIPS_TRAINER))

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-${TOTAL_TRAINER_CHIPS}}"

SMOKE_TEST="${SMOKE_TEST:-0}"
if [[ "${SMOKE_TEST}" == "1" ]]; then
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-8}"
    TEST_FREQ="${TEST_FREQ:-4}"
else
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"
    TEST_FREQ="${TEST_FREQ:-5}"
fi

python3 -m verl.trainer.sft_trainer_ray \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.val_max_samples=32 \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=2 \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.use_dynamic_bsz=False \
    data.max_length=2048 \
    data.max_token_len_per_gpu=2048 \
    data.ignore_input_ids_mismatch=True \
    model.use_remove_padding=True \
    engine=torchtitan \
    model=hf_model \
    model.path="${MODEL_PATH}" \
    optim=torchtitan \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.2 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.min_lr_factor=0.1 \
    optim.decay_type=cosine \
    engine.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    engine.pipeline_parallel_size=1 \
    engine.context_parallel_size=1 \
    engine.data_parallel_shard_size="${DATA_PARALLEL_SHARD_SIZE}" \
    engine.use_torch_compile=False \
    engine.attn_type=varlen \
    engine.max_seq_len=2048 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq=-1 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=2 \
    trainer.resume_mode=disable \
    trainer.device=tpu \
    trainer.nnodes="${NNODES_TRAINER}" \
    trainer.n_gpus_per_node="${N_CHIPS_TRAINER}" \
    "$@"
