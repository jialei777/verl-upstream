#!/usr/bin/env bash
# GRPO | Qwen3-0.6B (Fast Smoke Test) | TorchTitan Training & vLLM Rollout | 2x TPU7x 2x2x1 Slices | V1 PPOTrainer (Separate Async Overlap)
#
# TPU7x (Ironwood) geometry, which differs from every earlier generation:
#
#   1 slice (2x2x1) = 4 chips = 1 host VM (tpu7x-standard-4t)
#   1 chip          = 2 independent TensorCores => 2 addressable devices
#   => 8 devices per host, 8 worker processes per host, TORCH_TPU_TOPOLOGY="2,2,1,2"
#
# Everywhere below, "chips" in a verl/Hydra option name really means *devices*:
# verl runs exactly one worker process per addressable device, so
# `n_gpus_per_node` is 8 here even though the VM only has 4 physical chips.
# The corresponding v6e script uses 4 because a v6e chip is a single device.

set -xeuo pipefail

# export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
# export TPU_VISIBLE_CHIPS="0,1,2,3,4,5,6,7"
export VERL_PLATFORM=tpu
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export VLLM_USE_V1=0
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99

# Pin the TPU generation instead of relying on autodetection. verl can infer it
# from the Ray node labels (`ray.io/accelerator-type: TPU-V7X`), but being
# explicit keeps the run correct even if the driver starts before any TPU node
# has registered with the cluster.
export VERL_TPU_GENERATION=v7x

# XLA flags recommended for Ironwood.
#   dvfs_p_state=7                                  : highest clock P-state.
#   scoped_vmem_limit_kib=65472                     : TPU7x has a larger VMEM.
#   sparse_core_collective_aggregator=true          : offload collectives to the SparseCore.
#   latency_hiding_layer_scheduler=true             : overlap compute with collectives.
#   single_sparse_core_for_reduce_scatter_offload   : disabled, all SparseCores participate.
#   use_enhanced_launch_barrier=false               : required by torch_tpu (it rewrites
#                                                     this to false anyway, with a warning).
export LIBTPU_INIT_ARGS="--xla_tpu_dvfs_p_state=7 --xla_tpu_scoped_vmem_limit_kib=65472 --xla_tpu_enable_sparse_core_collective_aggregator=true --xla_tpu_enable_latency_hiding_layer_scheduler=true --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false --xla_tpu_use_enhanced_launch_barrier=false"
export TORCH_TPU_INTERNAL_XLA_OPTIONS="xla_tpu_scoped_vmem_limit_kib=65472"

# Project and Experiment details
project_name='verl_tpu_grpo'
exp_name="qwen3_0.6b_fast_smoke_test_tpu7x"

# Paths
RAY_DATA_HOME="/data/jialei"
MODEL_PATH="${MODEL_PATH:-${RAY_DATA_HOME}/assets/hf/Qwen3-0.6B}"

TRAIN_FILE="${RAY_DATA_HOME}/data/gsm8k/train.parquet"
TEST_FILE="${RAY_DATA_HOME}/data/gsm8k/test.parquet"

# TPU 2-slice tpu7x 2x2x1 configuration.
# Each slice is a single host holding 4 chips == 8 addressable devices.
export NNODES_TRAINER=1        # 1 physical VM host for the training slice
export N_DEVICES_TRAINER=8     # 8 TPU devices (4 chips x 2 cores) per training host

export NNODES_ROLLOUT=1        # 1 physical VM host for the rollout slice
export N_DEVICES_ROLLOUT=8     # 8 TPU devices (4 chips x 2 cores) per rollout host

TOTAL_ROLLOUT_DEVICES=$((NNODES_ROLLOUT * N_DEVICES_ROLLOUT))

python3 -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches=1 \
    trainer.v1.separate_async.parameter_sync_step=1 \
    transfer_queue.enable=True \
    model_engine=torchtitan \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.val_max_samples=8 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    +data.max_length=4096 \
    +data.max_token_len_per_gpu=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.pad_mode=no_padding \
    actor_rollout_ref.actor.strategy=torchtitan \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.torchtitan.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    +actor_rollout_ref.ref_in_actor=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size=2 \
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size=4 \
    actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1 \
    actor_rollout_ref.actor.torchtitan.attn_type=varlen \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TOTAL_ROLLOUT_DEVICES}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=tpu \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_model_len=512 \
    trainer.val_before_train=False \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.save_freq=-1 \
    trainer.test_freq=2 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=5 \
    trainer.nnodes="${NNODES_TRAINER}" \
    trainer.n_gpus_per_node="${N_DEVICES_TRAINER}" \
    actor_rollout_ref.rollout.nnodes="${NNODES_ROLLOUT}" \
    actor_rollout_ref.rollout.n_gpus_per_node="${N_DEVICES_ROLLOUT}" \
    +rollout.nnodes="${NNODES_ROLLOUT}" \
    +rollout.n_gpus_per_node="${N_DEVICES_ROLLOUT}" "$@"
