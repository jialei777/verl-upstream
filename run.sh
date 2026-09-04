#!/bin/bash
set -e

# Add Ray binary path
export PATH="/mnt/pd/daily/verl/bin:$PATH"

# Specify cluster kubeconfig
export KUBECONFIG="/tmp/jialei-kubeconfig"

# Ensure Ray cluster port-forward to svc/ray-tpu-v6e-cluster-head-svc is active
if ! curl -s http://localhost:23333/api/version >/dev/null 2>&1; then
    echo "Establishing port-forward to ray-tpu-v6e-cluster-head-svc on cluster /tmp/jialei-kubeconfig..."
    kubectl --kubeconfig="${KUBECONFIG}" port-forward svc/ray-tpu-v6e-cluster-head-svc 23333:8265 >/dev/null 2>&1 &
    sleep 3
fi

# Set active Ray cluster address
export RAY_ADDRESS="http://localhost:23333"

# Submit GRPO RL training job
ray job submit --address "${RAY_ADDRESS}" \
  --working-dir . \
  --runtime-env-json '{
    "excludes": [".git", "logs", "*.log", "*.pt", "*.bin"],
    "env_vars": {
      "PYTHONPATH": ".",
      "PYTHONUNBUFFERED": "1",
      "VERL_PLATFORM": "tpu",
      "VLLM_USE_V1": "0",
      "RAY_memory_monitor_refresh_ms": "0",
      "RAY_memory_usage_threshold": "0.99",
      "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
      "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
      "WANDB_API_KEY": "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"
    }
  }' \
  -- bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh
