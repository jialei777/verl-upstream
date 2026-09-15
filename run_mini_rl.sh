#!/usr/bin/env bash
set -euo pipefail

KUBECONFIG="${KUBECONFIG:-/tmp/alekseyv-kubeconfig}"
HEAD_POD=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

echo "Head Pod: ${HEAD_POD}"
echo "Copying mini_rl_reproduce.py to head pod..."
kubectl --kubeconfig="${KUBECONFIG}" cp mini_rl_reproduce.py "${HEAD_POD}:/tmp/mini_rl_reproduce.py" -c ray-head

echo "Submitting Ray Job via ray job submit..."
SUBMIT_LOG=$(mktemp)
kubectl --kubeconfig="${KUBECONFIG}" exec -i "${HEAD_POD}" -c ray-head -- \
    ray job submit --address="http://127.0.0.1:8265" -- python3 -u /tmp/mini_rl_reproduce.py 2>&1 | tee "${SUBMIT_LOG}"

# Extract Job ID
JOB_ID=$(grep -o "raysubmit_[a-zA-Z0-9_]*" "${SUBMIT_LOG}" | head -n 1 || true)
rm -f "${SUBMIT_LOG}"

if [ -n "${JOB_ID}" ]; then
    echo "=================================================="
    echo "Submitted Ray Job ID: ${JOB_ID}"
    echo "Query status via: bash get_job_status.sh ${JOB_ID}"
    echo "=================================================="
fi
