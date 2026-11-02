#!/usr/bin/env bash
# GRPO | Qwen3-30B-A3B (MoE) | GSM8K | TorchTitan Training (SimpleFSDP, torch.compile + splash attention, DEFER_AND_FUSE) & vLLM Rollout | TPU v5p 2x2x4 x2 Slices
# V1 PPOTrainer (Separate Async Overlap).
#
# Qwen3-30B-A3B: 48 layers, hidden 2048, 32 q / 4 KV heads (head_dim 128), 128 experts (8 active per token,
# moe_intermediate_size 768), vocab 151936; 30.5B parameters in total, 3.3B active per token.
#
# Memory budget per trainer chip (95 GiB HBM, FSDP over 16 chips): fp32 params + grads + Adam m/v of 30.5B params
# is 30.5B * 16 B / 16 = ~30.5 GiB.
#
# Rollout: 4 vLLM replicas, one per rollout host (ROLLOUT_TP=4, ROLLOUT_DP=4). With 4 KV heads, TP4 holds one KV head
# per rank (TP16 would replicate each KV head on 4 ranks). Expert parallelism over each replica's TP group (EP4, i.e.
# 32 whole experts per rank) avoids splitting moe_intermediate_size 768 into TP shards; verl turns it into vLLM's
# enable_expert_parallel. On v5p 2x2x4 at 128 prompts x 8 this halved the step time vs one TP16/EP16 replica
# (~110 s -> ~50 s; rollout wait 66-82 s -> 8-12 s; weight sync 22 s -> 27 s). ROLLOUT_TP=16 ROLLOUT_DP=1 restores
# the single-replica layout.
#
# Submit with USE_MOE_SPARSE_CORE=0 (and VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=USE_MOE_SPARSE_CORE) in the Ray job's
# runtime env so the vLLM workers on v5p use the TensorCore MoE path.

set -xeuo pipefail

export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
export VERL_PLATFORM=tpu
# TPU v5p: selects TPU_V5P_TOPOLOGY_MAP in platform_tpu.
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v5p}"
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export VLLM_USE_V1=0
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99

# libtpu flags (forwarded to the Ray workers by verl/trainer/constants_ppo.py):
# - enhanced launch barrier off (distributed hangs, b/477673365);
# - 64 MiB scoped VMEM, as in the v5p runs this recipe was tuned with;
# - async FSDP all-gather offloaded to the SparseCores: update_actor -2.7% on v5p 2x2x4 (9.35 s -> 9.10 s at
#   128 prompts x 8, numerics unchanged). Do not add --xla_tpu_enable_async_collective_fusion* /
#   --xla_tpu_overlap_compute_collective_tc on top: together with these, libtpu fails to initialize.
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_scoped_vmem_limit_kib=65536 --xla_enable_async_all_gather=true --xla_tpu_prefer_async_allgather_to_allreduce=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true}"

# PyTorch Dynamo / Compile Logging and Recompile Limits
export TORCH_LOGS="${TORCH_LOGS:-recompiles}"
export TORCH_DYNAMO_RECOMPILE_LIMIT="${TORCH_DYNAMO_RECOMPILE_LIMIT:-64}"

exp_name="${EXPERIMENT_NAME:-qwen3_30b_a3b_gsm8k_fsdp_compile}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"
# Qwen3 chat template thinking mode (enable_thinking), passed to the dataset and the rollout agent loop via
# data.apply_chat_template_kwargs. Off by default: with thinking on, 26-44% of responses hit MAX_RESPONSE_LEN=1024
# before writing the "#### <answer>" line that the strict GSM8K reward needs, so they score 0.
ENABLE_THINKING="${ENABLE_THINKING:-False}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
# Weight sync backend: raiden (P2P, as in the 4B/8B recipes) or tpu (gather to rank 0 + Ray object store).
CHECKPOINT_BACKEND="${CHECKPOINT_BACKEND:-raiden}"
# Free the bf16 tensors Raiden sends from and the WeightSynchronizer pinning them after each transfer, so they
# do not stay resident through training until the next sync.
RELEASE_SYNC_BUFFERS="${RELEASE_SYNC_BUFFERS:-True}"
# Full AC, as in the 8B/32B recipes and torchtitan's Qwen3-30B-A3B TPU v5p recipe (full_ac).
ACTIVATION_CHECKPOINT="${ACTIVATION_CHECKPOINT:-full}"
# SimpleFSDP: FSDP all-gather/reduce-scatter traced into each compiled TransformerBlock. Set False for FSDP2.
USE_SIMPLE_FSDP="${USE_SIMPLE_FSDP:-True}"
# torch_tpu eager mode for the ops outside the compiled blocks; see run_qwen3_8b_torchtitan.sh.
TPU_EAGER_MODE="${TPU_EAGER_MODE:-DEFER_AND_FUSE}"
# KL-to-reference loss, off by default: the reference model keeps a second fp32 copy of the weights
# (~7.6 GiB per chip at 30B on 16 chips) and doubles the trainer's checkpoint load time.
USE_KL_LOSS="${USE_KL_LOSS:-False}"
# Token budget per micro-batch (use_dynamic_bsz packs sequences up to it, for training and log-prob passes); must be
# >= MAX_MODEL_LEN. Every micro-batch costs ~2.6 s on v5p 2x2x4 regardless of size (FSDP Zero-3 gathers all 30.5B
# params), so size the budget to fit each chip's whole mini-batch share in ONE micro-batch:
#   - 32 prompts x 8 (~5.4k tokens/chip): 8192 (update_actor 7.0 s -> 4.1 s vs 4096)
#   - TRAIN_BATCH_SIZE=128 (~21k tokens/chip): 24576 (update_actor 9.3 s, ~11% MFU)
# Optional: VERL_TPU_STATIC_MICRO_BATCH_LEN=<budget> pads every micro-batch to that length, so only one shape is
# compiled (avoids ~60 s recompiles when the packed length moves to a new bucket).
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-8192}"
# Packed micro-batch lengths are padded to a multiple of this; a coarse bucket keeps the set of compiled shapes small
# (with 256, almost every step hit a new length and recompiled for ~40 s).
export VERL_TPU_SEQ_BUCKET_SIZE="${VERL_TPU_SEQ_BUCKET_SIZE:-1024}"
# Splash attention tile cap: packed micro-batches are long (8-24k tokens) and Splash's grid grows as (len / block)^2;
# 1024 instead of 512 cut update_actor by 17% at 24k tokens (2048 is the same, 4096 exceeds VMEM on v5p).
export VERL_TPU_SPLASH_MAX_BLOCK="${VERL_TPU_SPLASH_MAX_BLOCK:-1024}"
# TorchTitan's paired W1/W3 Pallas expert kernel with the kernel-native expert weight layout (as in its Qwen3-30B-A3B
# TPU recipe): -1.6% update_actor; checkpoint/HF names unchanged (state-dict hooks), Raiden exact parity verified.
export VERL_TPU_MOE_PAIRED_GMM="${VERL_TPU_MOE_PAIRED_GMM:-1}"
export VERL_TPU_MOE_NATIVE_LAYOUT="${VERL_TPU_MOE_NATIVE_LAYOUT:-1}"
# FSDP2 reshard policy (ignored with USE_SIMPLE_FSDP=True).
RESHARD_AFTER_FORWARD="${RESHARD_AFTER_FORWARD:-always}"
TEST_FREQ="${TEST_FREQ:-10}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"

# Project details
project_name='verl_tpu_grpo'

# Paths
RAY_DATA_HOME="${RAY_DATA_HOME:-/data/jialei}"
# Shared GCS Fuse copy of the checkpoint (~61 GB, 16 shards), staged to MODEL_PATH (host /tmp) on every TPU node
# before training. TorchTitan's HF reader and vLLM mmap the shards, and mmap over GCS Fuse runs at ~20 MB/s per
# host, versus ~1 GB/s for the parallel copy. Staging skips hosts that already have a complete copy and copies only
# config/tokenizer files to the CPU head (where the TaskRunner loads them). Set STAGE_MODEL=0 to skip it.
MODEL_SRC_PATH="${MODEL_SRC_PATH:-${RAY_DATA_HOME}/assets/hf/Qwen3-30B-A3B}"
MODEL_PATH="${MODEL_PATH:-/tmp/models/Qwen3-30B-A3B}"
STAGE_MODEL="${STAGE_MODEL:-1}"

TRAIN_FILE="${RAY_DATA_HOME}/data/gsm8k/train.parquet"
TEST_FILE="${RAY_DATA_HOME}/data/gsm8k/test.parquet"

if [[ "${STAGE_MODEL}" == "1" ]]; then
    python3 "$(dirname "${BASH_SOURCE[0]}")/stage_model_local.py" "${MODEL_SRC_PATH}" "${MODEL_PATH}" 8
fi

# 2 x TPU v5p 2x2x4 slices
export NNODES_TRAINER="${NNODES_TRAINER:-4}"     # 4 VM hosts in the training slice
export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-4}"   # 4 v5p chips per training host

export NNODES_ROLLOUT="${NNODES_ROLLOUT:-4}"     # 4 VM hosts in the rollout slice
export N_CHIPS_ROLLOUT="${N_CHIPS_ROLLOUT:-4}"   # 4 v5p chips per rollout host

TOTAL_ROLLOUT_CHIPS=$((NNODES_ROLLOUT * N_CHIPS_ROLLOUT))
TOTAL_TRAINER_CHIPS=$((NNODES_TRAINER * N_CHIPS_TRAINER))
# One TP=ROLLOUT_TP replica per ROLLOUT_TP chips; NNODES_ROLLOUT x N_CHIPS_ROLLOUT must equal ROLLOUT_DP x ROLLOUT_TP.
ROLLOUT_TP="${ROLLOUT_TP:-${N_CHIPS_ROLLOUT}}"
ROLLOUT_DP="${ROLLOUT_DP:-$((TOTAL_ROLLOUT_CHIPS / ROLLOUT_TP))}"
# Must be 1 (off) or ROLLOUT_TP * ROLLOUT_DP (verl/workers/config/rollout.py); each TPU replica then runs expert
# parallelism over its own TP group (tpu_replica_rollout_config).
ROLLOUT_EP="${ROLLOUT_EP:-$((ROLLOUT_TP * ROLLOUT_DP))}"

# Sequence budget
MAX_PROMPT_LEN=512
MAX_MODEL_LEN=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))

# Actor parallelism (default: Full FSDP across all 16 trainer chips: tp=1, dp_shard=16, no expert parallelism,
# like torchtitan's FSDP-only Qwen3-30B-A3B TPU recipe).
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DEFAULT_DP_SHARD=$((TOTAL_TRAINER_CHIPS / TENSOR_PARALLEL_SIZE))
DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-${DEFAULT_DP_SHARD}}"
TRAINER_EP="${TRAINER_EP:-1}"

# Off-policy correction (truncated importance sampling). Required in the decoupled
# separate_async regime; see run_qwen3_0_6b_torchtitan.sh for the full derivation.
ROLLOUT_IS="${ROLLOUT_IS:-token}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"

python3 -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches=1 \
    trainer.v1.separate_async.parameter_sync_step=1 \
    transfer_queue.enable=True \
    model_engine=torchtitan \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is="${ROLLOUT_IS}" \
    algorithm.rollout_correction.rollout_is_threshold="${ROLLOUT_IS_THRESHOLD}" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.val_max_samples="${VAL_MAX_SAMPLES}" \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESPONSE_LEN}" \
    +data.max_length=4096 \
    +data.max_token_len_per_gpu=4096 \
    data.filter_overlong_prompts=True \
    +data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING}" \
    data.truncation='error' \
    +data.pad_mode=no_padding \
    actor_rollout_ref.actor.strategy=torchtitan \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.torchtitan.use_torch_compile=True \
    actor_rollout_ref.actor.torchtitan.use_splash_attention=True \
    actor_rollout_ref.actor.torchtitan.use_simple_fsdp="${USE_SIMPLE_FSDP}" \
    actor_rollout_ref.actor.torchtitan.tpu_eager_mode="${TPU_EAGER_MODE}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${DATA_PARALLEL_SHARD_SIZE}" \
    actor_rollout_ref.actor.torchtitan.expert_parallel_size="${TRAINER_EP}" \
    actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1 \
    actor_rollout_ref.actor.torchtitan.attn_type=varlen \
    actor_rollout_ref.actor.torchtitan.activation_checkpoint="${ACTIVATION_CHECKPOINT}" \
    actor_rollout_ref.actor.torchtitan.reshard_after_forward="${RESHARD_AFTER_FORWARD}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.data_parallel_size="${ROLLOUT_DP}" \
    actor_rollout_ref.rollout.expert_parallel_size="${ROLLOUT_EP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.checkpoint_engine.backend="${CHECKPOINT_BACKEND}" \
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden.release_buffers_after_sync="${RELEASE_SYNC_BUFFERS}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.log_val_generations=4 \
    trainer.rollout_data_dir=/tmp/verl_dump/rollout \
    trainer.validation_data_dir=/tmp/verl_dump/validation \
    trainer.save_freq=-1 \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs=10 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.nnodes="${NNODES_TRAINER}" \
    trainer.n_gpus_per_node="${N_CHIPS_TRAINER}" \
    actor_rollout_ref.rollout.nnodes="${NNODES_ROLLOUT}" \
    actor_rollout_ref.rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" \
    +rollout.nnodes="${NNODES_ROLLOUT}" \
    +rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" "$@"
