#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="${HOME}/.local/bin:$PATH"
export KUBECONFIG="${KUBECONFIG:-/tmp/alekseyv-kubeconfig}"
export RAY_ADDRESS="${RAY_ADDRESS:-http://localhost:23333}"

# Ensure port-forward is alive
if ! curl -s "${RAY_ADDRESS}/api/version" >/dev/null 2>&1; then
    fuser -k 23333/tcp || true
    kubectl --kubeconfig="${KUBECONFIG}" port-forward svc/ray-tpu-v6e-cluster-head-svc 23333:8265 >/dev/null 2>&1 &
    sleep 3
fi

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"

RUNTIME_ENV_JSON=$(cat <<EOF
{
  "excludes": [
    ".git",
    ".venv",
    "venv",
    "logs",
    "*.log",
    "*.pt",
    "*.bin",
    "*.whl",
    "*.docx",
    "__pycache__"
  ],
  "env_vars": {
    "PYTHONPATH": ".",
    "PYTHONUNBUFFERED": "1",
    "VERL_PLATFORM": "tpu",
    "VLLM_USE_V1": "0",
    "RAY_memory_monitor_refresh_ms": "0",
    "RAY_memory_usage_threshold": "0.99",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
    "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
    "TORCH_LOGS": "recompiles",
    "TORCH_DYNAMO_RECOMPILE_LIMIT": "64",
    "TOTAL_TRAINING_STEPS": "${TOTAL_TRAINING_STEPS}"
  }
}
EOF
)

ray job submit --address "${RAY_ADDRESS}" \
  --working-dir "${SCRIPT_DIR}" \
  --runtime-env-json "${RUNTIME_ENV_JSON}" \
  -- bash examples/tpu/grpo/run_qwen3_4b_torchtitan.sh "$@"
