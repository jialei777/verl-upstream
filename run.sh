#!/bin/bash
set -e

# Add Ray binary and gke auth plugin path
export PATH="/usr/bin:/usr/local/google/home/wenjung/.local/bin:/mnt/pd/daily/verl/bin:$PATH"

# Specify cluster kubeconfig
export KUBECONFIG="/tmp/alekseyv-kubeconfig"

RAY_PORT=23335

# Reset job log file
> /tmp/job_logs.txt

# Ensure fresh Ray cluster port-forward to svc/ray-tpu-v6e-cluster-head-svc
echo "Establishing fresh port-forward to ray-tpu-v6e-cluster-head-svc on port ${RAY_PORT}..."
pkill -9 -f "port-forward.*${RAY_PORT}" || true
sleep 1
nohup kubectl --kubeconfig="${KUBECONFIG}" port-forward svc/ray-tpu-v6e-cluster-head-svc "${RAY_PORT}:8265" </dev/null >/tmp/pf.log 2>&1 &
PF_PID=$!

for i in {1..30}; do
    if curl -s --max-time 2 "http://127.0.0.1:${RAY_PORT}/api/version" >/dev/null 2>&1; then
        echo "Port-forward established."
        break
    fi
    sleep 1
done

# Set active Ray cluster address
export RAY_ADDRESS="http://127.0.0.1:${RAY_PORT}"

# Submit GRPO RL training job
ray job submit --address "${RAY_ADDRESS}" \
  --working-dir . \
  --runtime-env-json '{
    "excludes": [".git", "logs", "*.log", "*.pt", "*.bin", ".system_generated", ".gemini", "*.zip", "*.parquet", "*.whl", "**/*.whl"],
    "env_vars": {
      "PYTHONPATH": ".:/tmp/tpu-sync",
      "PYTHONUNBUFFERED": "1",
      "VERL_PLATFORM": "tpu",
      "VLLM_USE_V1": "0",
      "RAY_memory_monitor_refresh_ms": "0",
      "RAY_memory_usage_threshold": "0.99",
      "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
      "ALLOW_MULTIPLE_LIBTPU_LOAD": "1",
      "TPU_SKIP_MDS_QUERY": "true",
      "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
      "LIBTPU_INIT_ARGS": "--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_scoped_vmem_limit_kib=65536",
      "XLA_FLAGS": "--xla_disable_hlo_passes=instruction-fusion,fusion-merger,multi-output-fusion,horizontal-fusion",
      "WANDB_API_KEY": "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"
    }
  }' \
  -- bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh 2>&1 | tee /tmp/job_logs.txt
