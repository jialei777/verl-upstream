#!/usr/bin/env bash
set -euo pipefail

KUBECONFIG="${KUBECONFIG:-/tmp/alekseyv-kubeconfig}"
HEAD_POD=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

echo "Head Pod: ${HEAD_POD}"
echo "Copying mini_rl_reproduce.py to head pod..."
kubectl --kubeconfig="${KUBECONFIG}" cp mini_rl_reproduce.py "${HEAD_POD}:/tmp/mini_rl_reproduce.py" -c ray-head

echo "Submitting Ray Job via ray job submit..."
JOB_OUTPUT=$(kubectl --kubeconfig="${KUBECONFIG}" exec -i "${HEAD_POD}" -c ray-head -- \
    ray job submit --address="http://127.0.0.1:8265" -- python3 -u /tmp/mini_rl_reproduce.py)

echo "${JOB_OUTPUT}"

# Extract Job ID
JOB_ID=$(echo "${JOB_OUTPUT}" | grep -o "raysubmit_[a-zA-Z0-9_]*" | head -n 1 || true)

if [ -n "${JOB_ID}" ]; then
    echo "=================================================="
    echo "Submitted Ray Job ID: ${JOB_ID}"
    echo "Follow logs with: ray job logs --address http://127.0.0.1:8265 ${JOB_ID} --follow"
    echo "Or query status via: ./get_job_status.sh ${JOB_ID}"
    echo "=================================================="
    echo "Streaming logs for ${JOB_ID}..."
    kubectl --kubeconfig="${KUBECONFIG}" exec -i "${HEAD_POD}" -c ray-head -- \
        ray job logs --address="http://127.0.0.1:8265" "${JOB_ID}" --follow || true
fi
