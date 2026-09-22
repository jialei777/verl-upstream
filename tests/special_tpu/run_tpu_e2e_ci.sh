#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
# End-to-End CI Runner for TPU v6e on GKE KubeRay (SFT & GRPO)
#
# Usage:
#   bash tests/special_tpu/run_tpu_e2e_ci.sh [sft|grpo|all]

set -euo pipefail

MODE="${1:-all}"
export CLUSTER_NAME="${CLUSTER_NAME:-jialeic-ci-v6e8-2s-spot}"
export REGION="${REGION:-us-central2}"
export PROJECT="${PROJECT:-tpu-pytorch}"
export RAY_CLUSTER_NAME="${RAY_CLUSTER_NAME:-ray-tpu-v6e-cluster}"
export RAY_NAMESPACE="${RAY_NAMESPACE:-default}"
export SMOKE_TEST="${SMOKE_TEST:-1}"
export EXPECTED_TPU_CHIPS="${EXPECTED_TPU_CHIPS:-16.0}"
export PORT_FORWARD_PORT="${PORT_FORWARD_PORT:-28265}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PF_PID=""
cleanup() {
    if [[ -n "${PF_PID}" ]] && kill -0 "${PF_PID}" 2>/dev/null; then
        kill "${PF_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

connect_gke_cluster() {
    echo "=================================================================="
    echo "[TPU CI] Connecting to GKE cluster: ${CLUSTER_NAME} (${REGION}, ${PROJECT})"
    echo "=================================================================="
    gcloud container clusters get-credentials "${CLUSTER_NAME}" \
        --region "${REGION}" \
        --project "${PROJECT}" \
        --dns-endpoint

    if ! kubectl get raycluster "${RAY_CLUSTER_NAME}" -n "${RAY_NAMESPACE}" >/dev/null 2>&1; then
        echo "[TPU CI] RayCluster ${RAY_CLUSTER_NAME} not found; applying manifest..."
        kubectl apply -n "${RAY_NAMESPACE}" -f examples/tpu/gke/ray-tpu-v6e8-2slice.yaml
    fi
}

ensure_clean_tpu_cluster() {
    local force_reset="${1:-0}"

    if [[ "${force_reset}" == "1" ]]; then
        echo "[TPU CI] Resetting KubeRay cluster pods to ensure clean TPU state..."
        kubectl delete pod -n "${RAY_NAMESPACE}" -l "ray.io/cluster=${RAY_CLUSTER_NAME}" --wait=true --timeout=180s || true
        sleep 5
    fi

    echo "[TPU CI] Waiting for 1 head pod + 4 TPU worker pods to reach Running/Ready..."
    local deadline=$((SECONDS + 900))
    while (( SECONDS < deadline )); do
        local ready_pods
        ready_pods="$(kubectl get pods -n "${RAY_NAMESPACE}" -l "ray.io/cluster=${RAY_CLUSTER_NAME}" --no-headers 2>/dev/null \
            | awk '$2 == "2/2" && $3 == "Running" {count++} END {print count+0}')"
        if [[ "${ready_pods}" -ge 5 ]]; then
            echo "[TPU CI] All 5 KubeRay pods (1 head + 4 TPU workers) are 2/2 Running."
            break
        fi
        echo "[TPU CI] Ready pods: ${ready_pods}/5. Waiting 10s..."
        sleep 10
    done

    local head_pod
    head_pod="$(kubectl get pods -n "${RAY_NAMESPACE}" -l "ray.io/cluster=${RAY_CLUSTER_NAME},ray.io/node-type=head" \
        -o jsonpath='{.items[0].metadata.name}')"

    echo "[TPU CI] Waiting for Ray cluster (${head_pod}) to register 0.0/${EXPECTED_TPU_CHIPS} TPU..."
    deadline=$((SECONDS + 300))
    while (( SECONDS < deadline )); do
        local tpu_usage
        tpu_usage="$(kubectl exec -n "${RAY_NAMESPACE}" "${head_pod}" -c ray-head -- ray status 2>/dev/null \
            | grep -oE '[0-9.]+/[0-9.]+ TPU$' | head -n 1 || true)"
        echo "[TPU CI] Ray TPU status: ${tpu_usage:-<initializing>}"
        if [[ "${tpu_usage}" == "0.0/${EXPECTED_TPU_CHIPS} TPU" ]]; then
            return 0
        fi
        # If TPUs are held by a leftover process from a previous run, force-reset pods once
        if [[ -n "${tpu_usage}" && "${tpu_usage}" == */"${EXPECTED_TPU_CHIPS} TPU" && "${tpu_usage}" != 0.0/* && "${force_reset}" == "0" ]]; then
            echo "[TPU CI] Detected held TPUs (${tpu_usage}); recycling cluster pods..."
            ensure_clean_tpu_cluster 1
            return 0
        fi
        sleep 10
    done

    echo "[TPU CI] ERROR: Timed out waiting for 0.0/${EXPECTED_TPU_CHIPS} TPU." >&2
    kubectl get pods -n "${RAY_NAMESPACE}" -o wide >&2
    exit 1
}

start_port_forward() {
    cleanup
    # Check if we are running inside the same Kubernetes cluster and can reach the service directly
    if curl -sf "http://${RAY_CLUSTER_NAME}-head-svc.${RAY_NAMESPACE}.svc.cluster.local:8265/api/version" >/dev/null 2>&1; then
        export RAY_ADDRESS="http://${RAY_CLUSTER_NAME}-head-svc.${RAY_NAMESPACE}.svc.cluster.local:8265"
        echo "[TPU CI] Using in-cluster Ray head service: ${RAY_ADDRESS}"
        return 0
    fi

    echo "[TPU CI] Starting port-forward to svc/${RAY_CLUSTER_NAME}-head-svc on localhost:${PORT_FORWARD_PORT}..."
    kubectl port-forward -n "${RAY_NAMESPACE}" "svc/${RAY_CLUSTER_NAME}-head-svc" "${PORT_FORWARD_PORT}:8265" >/dev/null 2>&1 &
    PF_PID=$!

    local deadline=$((SECONDS + 30))
    while (( SECONDS < deadline )); do
        if curl -sf "http://127.0.0.1:${PORT_FORWARD_PORT}/api/version" >/dev/null 2>&1; then
            export RAY_ADDRESS="http://127.0.0.1:${PORT_FORWARD_PORT}"
            echo "[TPU CI] Port-forward established at ${RAY_ADDRESS}"
            return 0
        fi
        sleep 1
    done

    echo "[TPU CI] ERROR: Failed to establish port-forward to Ray head service." >&2
    exit 1
}

submit_and_verify_ray_job() {
    local suite_name="$1"
    local script_path="$2"
    local timeout_mins="$3"

    ensure_clean_tpu_cluster 0
    start_port_forward

    local head_pod
    head_pod="$(kubectl get pods -n "${RAY_NAMESPACE}" -l "ray.io/cluster=${RAY_CLUSTER_NAME},ray.io/node-type=head" \
        -o jsonpath='{.items[0].metadata.name}')"

    local sub_id="ci_${suite_name}_$(date +%Y%m%d_%H%M%S)"
    local log_file="/tmp/${sub_id}.log"

    local runtime_env
    runtime_env="$(python3 -c '
import json, os
print(json.dumps({
    "excludes": [".git", "logs", "*.log", "*.pt", "*.bin", "__pycache__", ".ruff_cache", ".mypy_cache"],
    "env_vars": {
        "PYTHONPATH": ".",
        "PYTHONUNBUFFERED": "1",
        "VERL_PLATFORM": "tpu",
        "VLLM_USE_V1": "0",
        "SMOKE_TEST": os.environ.get("SMOKE_TEST", "1"),
        "RAY_memory_monitor_refresh_ms": "0",
        "RAY_memory_usage_threshold": "0.99",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
        "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
    }
}))
')"

    local extra_args=""
    if [[ "${suite_name}" == "grpo" ]]; then
        # Disable Qwen3 <think> truncation in CI smoke runs so 512-token rollouts emit
        # "#### <answer>", enabling real GSM8K test pass-rate and non-zero gradient checks
        # even in 5-step smoke mode.
        extra_args="+data.apply_chat_template_kwargs.enable_thinking=False trainer.val_before_train=True data.val_max_samples=32 data.train_batch_size=8 actor_rollout_ref.actor.ppo_mini_batch_size=8 actor_rollout_ref.rollout.n=4"
    fi

    echo "=================================================================="
    echo "[TPU CI] Submitting ${suite_name^^} job (${sub_id}): ${script_path}"
    echo "=================================================================="
    ray job submit \
        --address "${RAY_ADDRESS}" \
        --submission-id "${sub_id}" \
        --working-dir "${REPO_ROOT}" \
        --runtime-env-json "${runtime_env}" \
        --no-wait \
        -- bash "${script_path}" ${extra_args}

    # Poll status via kubectl exec so transient port-forward drops never kill the CI run
    local deadline=$((SECONDS + timeout_mins * 60))
    local final_status="TIMEOUT"
    while (( SECONDS < deadline )); do
        local status_out
        status_out="$(kubectl exec -n "${RAY_NAMESPACE}" "${head_pod}" -c ray-head -- \
            ray job status --address http://127.0.0.1:8265 "${sub_id}" 2>/dev/null || true)"
        echo "[$(date +%H:%M:%S)] ${sub_id}: ${status_out}"
        local status_lower="${status_out,,}"
        if [[ "${status_lower}" == *"succeeded"* ]]; then
            final_status="SUCCEEDED"
            break
        elif [[ "${status_lower}" == *"failed"* || "${status_lower}" == *"stopped"* ]]; then
            final_status="FAILED"
            break
        fi
        sleep 15
    done

    echo "[TPU CI] Fetching full job logs for ${sub_id}..."
    kubectl exec -n "${RAY_NAMESPACE}" "${head_pod}" -c ray-head -- \
        ray job logs --address http://127.0.0.1:8265 "${sub_id}" > "${log_file}" 2>&1 || true
    tail -n 120 "${log_file}"

    if [[ "${final_status}" != "SUCCEEDED" ]]; then
        echo "[TPU CI] ERROR: Job ${sub_id} ended with status ${final_status}" >&2
        exit 1
    fi

    # Verify training convergence & test pass rate in log output
    python3 - "${suite_name}" "${log_file}" "${SMOKE_TEST}" <<'PY'
import math
import re
import sys

suite, log_path, smoke_test = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
text = open(log_path, encoding="utf-8", errors="replace").read()

steps = re.findall(r"step[:\s]+([1-9][0-9]*)", text, flags=re.IGNORECASE)
if not steps:
    print(f"[TPU CI] ERROR: No training steps logged in {suite} output.", file=sys.stderr)
    sys.exit(1)

if suite == "sft":
    losses = [float(x) for x in re.findall(r"train/loss[:\s]+([0-9.eE+-]+)", text)]
    val_losses = [float(x) for x in re.findall(r"val/loss[:\s]+([0-9.eE+-]+)", text)]
    grad_norms = [float(x) for x in re.findall(r"train/grad_norm[:\s]+([0-9.eE+-]+)", text)]
    if not losses or not val_losses:
        print("[TPU CI] ERROR: Missing train/loss or val/loss metrics in SFT output.", file=sys.stderr)
        sys.exit(1)
    if any(not math.isfinite(g) for g in grad_norms):
        print("[TPU CI] ERROR: Non-finite train/grad_norm in SFT output.", file=sys.stderr)
        sys.exit(1)
    assert losses[-1] < losses[0] * 0.80, (
        f"[TPU CI] SFT train/loss did not converge sufficiently: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
    assert val_losses[-1] <= val_losses[0] and val_losses[-1] < 0.85, (
        f"[TPU CI] SFT val/loss did not converge sufficiently: {val_losses[0]:.4f} -> {val_losses[-1]:.4f}"
    )
    print(
        f"[TPU CI] SFT convergence verified: train/loss {losses[0]:.4f} -> {losses[-1]:.4f}, "
        f"val/loss {val_losses[0]:.4f} -> {val_losses[-1]:.4f}"
    )
elif suite == "grpo":
    if re.search(r"actor/grad_norm[:\s]+(nan|inf)", text, flags=re.IGNORECASE):
        print("[TPU CI] ERROR: Non-finite actor/grad_norm detected in GRPO output!", file=sys.stderr)
        sys.exit(1)
    grad_norms = [float(x) for x in re.findall(r"actor/grad_norm[:\s]+([0-9.eE+-]+)", text)]
    rewards = [float(x) for x in re.findall(r"critic/rewards/mean[:\s]+([0-9.eE+-]+)", text)]
    corrs = [float(x) for x in re.findall(r"training/rollout_actor_probs_pearson_corr[:\s]+([0-9.eE+-]+)", text)]
    val_accs = [
        float(x)
        for x in re.findall(r"val-core/openai/gsm8k/acc/mean@1['\"]?:\s*(?:np\.float64\()?([0-9.eE+-]+)", text)
    ]
    if not rewards or not grad_norms:
        print("[TPU CI] ERROR: Missing critic/rewards/mean or actor/grad_norm in GRPO output.", file=sys.stderr)
        sys.exit(1)
    assert max(grad_norms) > 0.0, "[TPU CI] GRPO actor/grad_norm was 0.0 on all steps (no gradient flowed)!"
    assert max(rewards) >= 0.15, f"[TPU CI] GRPO best training reward {max(rewards):.4f} < 0.15 target!"
    if corrs:
        assert min(corrs) >= 0.90, (
            f"[TPU CI] Rollout-Actor logprob Pearson correlation dropped below 0.90: min={min(corrs):.4f}"
        )
    if val_accs:
        min_val_acc = 0.15 if smoke_test else 0.25
        assert max(val_accs) >= min_val_acc, (
            f"[TPU CI] GSM8K test pass rate (val-core/openai/gsm8k/acc/mean@1) {max(val_accs):.4f} < {min_val_acc}!"
        )
        if not smoke_test and len(val_accs) >= 2:
            assert val_accs[-1] > val_accs[0], (
                f"[TPU CI] GSM8K test pass rate did not improve over full run: {val_accs[0]:.4f} -> {val_accs[-1]:.4f}"
            )
    print(
        f"[TPU CI] GRPO convergence & quality verified: best_reward={max(rewards):.4f}, "
        f"val_acc={val_accs[-1] if val_accs else 'N/A'}, min_corr={min(corrs) if corrs else 'N/A'}, "
        f"max_grad_norm={max(grad_norms):.4f}"
    )
PY

    echo "[TPU CI] ${suite_name^^} E2E test PASSED!"
}

connect_gke_cluster

case "${MODE}" in
    sft)
        submit_and_verify_ray_job "sft" "examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh" 25
        ;;
    grpo)
        submit_and_verify_ray_job "grpo" "examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh" 35
        ;;
    all)
        submit_and_verify_ray_job "sft" "examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh" 25
        # Recycle pods between SFT and GRPO so all 16 TPU chips start completely clean
        ensure_clean_tpu_cluster 1
        submit_and_verify_ray_job "grpo" "examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh" 35
        ;;
    *)
        echo "Usage: $0 [sft|grpo|all]" >&2
        exit 1
        ;;
esac
