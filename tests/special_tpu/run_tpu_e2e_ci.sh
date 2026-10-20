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
# End-to-end TPU CI runner (Cloud TPU v6e on GKE KubeRay) for the TPU examples in examples/tpu.
#
# Every invocation runs one CI test and owns the full lifecycle of its own RayCluster:
#   1. provision - render tests/special_tpu/gke/raycluster-ci.yaml into a uniquely named, Kueue-queued
#                  RayCluster: single-host v6e-4 (2x2) subslices of the v6e-8 (2x4) node pools, or
#                  whole 1-chip ct6e-standard-1t nodes. Kueue keeps it suspended until the chip-count
#                  queue has TPU quota, so concurrent CI runs queue (FIFO) instead of fighting for hosts.
#   2. run       - submit the test as Ray job(s) the way examples/tpu/grpo/README.md does (this checkout
#                  as the working dir, verl-hardware-plugin as a py_module) and verify the job log with
#                  tests/special_tpu/verify_tpu_e2e_log.py.
#   3. tear down - delete the RayCluster on exit, whether the test passed, failed, timed out or was
#                  cancelled.
#
# Tests (one per workflow, see tests/special_tpu/README.md):
#   tpu_platform  1 x v6e-1  queue 1chip  tpu_unit_tests.yml             tests/special_tpu/tpu_smoke_test.py
#   sft           1 x v6e-4  queue 4chip  e2e_sft_llm_tpu.yml            examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh
#   grpo          2 x v6e-1  queue 2chip  e2e_v1_separate_async_tpu.yml  examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh
#   grpo_4b       2 x v6e-4  queue 8chip  nightly_tpu.yml                examples/tpu/grpo/run_qwen3_4b_torchtitan.sh
#
# Usage:
#   bash tests/special_tpu/run_tpu_e2e_ci.sh <tpu_platform|sft|grpo|grpo_4b>
#   bash tests/special_tpu/run_tpu_e2e_ci.sh teardown [test]   # delete this run's RayClusters

set -euo pipefail

USAGE="usage: $0 <tpu_platform|sft|grpo|grpo_4b|teardown> [test]"
MODE="${1:?${USAGE}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=tests/special_tpu/gke/tpu-ci.env
set -a
source "${REPO_ROOT}/tests/special_tpu/gke/tpu-ci.env"
set +a
export CLUSTER_NAME="${CLUSTER_NAME:-jialeic-ci-v6e8-2s-spot}"
export REGION="${REGION:-us-central2}"
export PROJECT="${PROJECT:-tpu-pytorch}"
export RAY_NAMESPACE="${RAY_NAMESPACE:-default}"
# One Kueue LocalQueue per chip count: ${KUEUE_QUEUE_PREFIX}-{1chip,2chip,4chip,8chip}.
export KUEUE_QUEUE_PREFIX="${KUEUE_QUEUE_PREFIX:-verl-tpu-ci}"
export PORT_FORWARD_PORT="${PORT_FORWARD_PORT:-$((28000 + RANDOM % 1000))}"
# Maximum time to wait in the Kueue queue before giving up.
export QUEUE_TIMEOUT_MINS="${QUEUE_TIMEOUT_MINS:-120}"
# Maximum time from admission until all pods run (TPU node scale-up from zero + image pull, or waiting for
# hosts that something outside Kueue still holds).
export POD_READY_TIMEOUT_MINS="${POD_READY_TIMEOUT_MINS:-25}"
# CI RayClusters older than this are considered leaked (e.g. the runner pod crashed) and reaped.
export STALE_CLUSTER_SECONDS="${STALE_CLUSTER_SECONDS:-10800}"
# The TPU node pools are spot VMs: re-run a test this many times on a new RayCluster when a TPU node it ran
# on was preempted.
export PREEMPTION_RETRIES="${PREEMPTION_RETRIES:-1}"
# Maximum time to build the CI image from examples/tpu/grpo/Dockerfile when the registry does not have it yet.
export IMAGE_BUILD_TIMEOUT_MINS="${IMAGE_BUILD_TIMEOUT_MINS:-90}"

RUN_ID="${GITHUB_RUN_ID:-local$(date +%s)}"
RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
WORK_DIR="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/verl-tpu-ci-${RUN_ID}"

PF_PID=""
LOG_FOLLOW_PID=""
CURRENT_RAY_JOB_ID=""
RAY_CLUSTER_NAME=""
HEAD_POD=""
TPU_NODES=""
KUEUE_QUEUE=""
TIER=""
WORKER_REPLICAS=1
WORKER_TPU_CHIPS=4
PLUGIN_PKG=""
RUNTIME_ENV=""
# Appended to the RayCluster name when a test is re-run after a preemption.
CLUSTER_SUFFIX=""

log() {
    echo "[TPU CI $(date +%H:%M:%S)] $*"
}

connect_gke_cluster() {
    if [[ -n "${KUBERNETES_SERVICE_HOST:-}" ]]; then
        # Self-hosted runner pod in the CI cluster: in-cluster service account (ray-ksa).
        unset KUBECONFIG
        if ! command -v kubectl >/dev/null 2>&1; then
            curl -sLO https://dl.k8s.io/release/v1.31.0/bin/linux/amd64/kubectl
            chmod +x kubectl
            mv kubectl /usr/local/bin/kubectl
        fi
        log "Running inside GKE pod; using in-cluster Kubernetes service account."
    else
        # Use a private kubeconfig so a local run does not switch the user's current kubectl context.
        export KUBECONFIG="${TPU_CI_KUBECONFIG:-${WORK_DIR}/kubeconfig}"
        mkdir -p "$(dirname "${KUBECONFIG}")"
        log "Connecting to GKE cluster ${CLUSTER_NAME} (${REGION}, ${PROJECT})"
        gcloud container clusters get-credentials "${CLUSTER_NAME}" \
            --region "${REGION}" \
            --project "${PROJECT}" \
            --dns-endpoint
    fi
}

delete_ray_clusters() {
    # Delete CI RayClusters matching a label selector without blocking on pod termination.
    local selector="$1"
    kubectl delete raycluster -n "${RAY_NAMESPACE}" -l "${selector}" --wait=false --ignore-not-found || true
}

cleanup() {
    local exit_code=$?
    set +e
    if [[ -n "${CURRENT_RAY_JOB_ID}" && -n "${HEAD_POD}" ]]; then
        log "Stopping active Ray job ${CURRENT_RAY_JOB_ID}..."
        kubectl exec -n "${RAY_NAMESPACE}" "${HEAD_POD}" -c ray-head -- \
            ray job stop --address http://127.0.0.1:8265 "${CURRENT_RAY_JOB_ID}" >/dev/null 2>&1
        CURRENT_RAY_JOB_ID=""
    fi
    local pid
    for pid in ${PF_PID} ${LOG_FOLLOW_PID}; do
        kill "${pid}" 2>/dev/null
    done
    if [[ -n "${RAY_CLUSTER_NAME}" ]]; then
        log "Tearing down RayCluster ${RAY_CLUSTER_NAME} (exit code ${exit_code})..."
        kubectl delete raycluster -n "${RAY_NAMESPACE}" "${RAY_CLUSTER_NAME}" --wait=false --ignore-not-found
        RAY_CLUSTER_NAME=""
    fi
    exit "${exit_code}"
}

ensure_kueue_queue() {
    # Bootstrap only: create the Kueue objects if the test's LocalQueue does not exist yet. The queues
    # and quotas are shared by every branch and PR, so an existing setup is never modified from CI;
    # changes to kueue-tpu-ci.yaml are rolled out by an admin with `kubectl apply -f`.
    local queue="$1"
    if kubectl get localqueue -n "${RAY_NAMESPACE}" "${queue}" >/dev/null 2>&1; then
        log "Kueue LocalQueue ${queue} exists."
    else
        log "Kueue LocalQueue ${queue} not found; applying tests/special_tpu/gke/kueue-tpu-ci.yaml."
        kubectl apply -f tests/special_tpu/gke/kueue-tpu-ci.yaml
    fi
}

reap_stale_clusters() {
    local stale
    stale="$(kubectl get raycluster -n "${RAY_NAMESPACE}" -l verl-ci/managed=true -o json | python3 -c '
import datetime, json, sys
max_age = int(sys.argv[1])
now = datetime.datetime.now(datetime.timezone.utc)
for item in json.load(sys.stdin)["items"]:
    created = datetime.datetime.fromisoformat(item["metadata"]["creationTimestamp"].replace("Z", "+00:00"))
    if (now - created).total_seconds() > max_age:
        print(item["metadata"]["name"])
' "${STALE_CLUSTER_SECONDS}")"
    for name in ${stale}; do
        log "Reaping leaked CI RayCluster ${name} (older than ${STALE_CLUSTER_SECONDS}s)..."
        kubectl delete raycluster -n "${RAY_NAMESPACE}" "${name}" --wait=false --ignore-not-found || true
    done
}

fetch_hardware_plugin() {
    # verl-hardware-plugin provides the TPU platform, engines, weight sync and vLLM rollout. Every job ships it
    # like examples/tpu/grpo/README.md does: VERL_HARDWARE_PLUGIN_REPO at VERL_HARDWARE_PLUGIN_COMMIT (pinned in
    # gke/tpu-ci.env; the commit may come from a fork of the plugin). VERL_HARDWARE_PLUGIN_DIR may point at a
    # local checkout instead (e.g. an uncommitted plugin change when running a test by hand).
    local plugin_dir="${VERL_HARDWARE_PLUGIN_DIR:-}" plugin_source
    if [[ -z "${plugin_dir}" ]]; then
        local repo="${VERL_HARDWARE_PLUGIN_REPO}" commit="${VERL_HARDWARE_PLUGIN_COMMIT}" readme_commit
        plugin_dir="${WORK_DIR}/verl-hardware-plugin"
        plugin_source="${repo}"
        log "Fetching verl-hardware-plugin ${commit} from ${repo}"
        rm -rf "${plugin_dir}"
        git init -q "${plugin_dir}"
        if ! git -C "${plugin_dir}" fetch -q --depth 1 "${repo}" "${commit}"; then
            log "ERROR: cannot fetch ${commit} from ${repo}. Use a full 40-character commit id (or a branch or" \
                "tag) of that repository; for a commit of a fork, set VERL_HARDWARE_PLUGIN_REPO to the fork." >&2
            exit 1
        fi
        git -C "${plugin_dir}" checkout -q FETCH_HEAD
        if [[ "${commit}" =~ ^[0-9a-f]{40}$ && "$(git -C "${plugin_dir}" rev-parse HEAD)" != "${commit}" ]]; then
            log "ERROR: fetched $(git -C "${plugin_dir}" rev-parse HEAD) instead of ${commit}." >&2
            exit 1
        fi
        readme_commit="$(sed -n 's/^git -C verl-hardware-plugin checkout \([0-9a-f]*\).*/\1/p' \
            examples/tpu/grpo/README.md | head -n 1)"
        if [[ -n "${readme_commit}" && "${readme_commit}" != "$(git -C "${plugin_dir}" rev-parse HEAD)" ]]; then
            log "Note: examples/tpu/grpo/README.md checks out verl-hardware-plugin ${readme_commit}; CI tests ${commit}."
        fi
    else
        plugin_source="local checkout ${plugin_dir}"
    fi
    PLUGIN_PKG="$(cd "${plugin_dir}" && pwd)/verl_hardware_plugin"
    if [[ ! -f "${PLUGIN_PKG}/__init__.py" ]]; then
        log "ERROR: ${PLUGIN_PKG} is not the verl_hardware_plugin package." >&2
        exit 1
    fi
    log "verl-hardware-plugin: $(git -C "${plugin_dir}" log -1 --format='%H %s' 2>/dev/null || echo "?") (${plugin_source})"
    # The runtime environment of examples/tpu/grpo/README.md (step 5), with the plugin path of this run.
    RUNTIME_ENV="$(PLUGIN_PKG="${PLUGIN_PKG}" python3 -c '
import json, os
print(json.dumps({
    "py_modules": [os.environ["PLUGIN_PKG"]],
    "excludes": [".git", ".venv"],
    "env_vars": {
        "PYTHONPATH": ".",
        "PYTHONUNBUFFERED": "1",
        "VERL_PLATFORM": "tpu",
        "VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin",
        "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
    },
}))
')"
}

registry_access_token() {
    # Artifact Registry access token: of ray-ksa's Google service account (Workload Identity) in a runner pod,
    # of the gcloud account otherwise.
    if [[ -n "${KUBERNETES_SERVICE_HOST:-}" ]]; then
        curl -sf -H "Metadata-Flavor: Google" \
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" |
            python3 -c 'import json, sys; print(json.load(sys.stdin)["access_token"])'
    else
        gcloud auth print-access-token
    fi
}

image_in_registry() {
    # Whether the image reference $1 (<host>/<path>:<tag>) exists, by asking the registry for its manifest.
    local image="$1" token="$2" host path tag code
    host="${image%%/*}" path="${image#*/}" tag="${path##*:}" path="${path%:*}"
    code="$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer ${token}" \
        -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json" \
        "https://${host}/v2/${path}/manifests/${tag}")"
    [[ "${code}" == "200" ]]
}

ensure_ci_image() {
    # The Ray pods run the image built from examples/tpu/grpo/Dockerfile at this checkout, so every run tests the
    # Dockerfile in the tree. The image is tagged by the Dockerfile's content hash and built (by a BuildKit Job in
    # the CI cluster, gke/image-build-job.yaml) only when that tag is not in TPU_CI_IMAGE_REPO yet; concurrent runs
    # of the same Dockerfile share one build. Set TPU_CI_IMAGE to use a prebuilt image instead.
    if [[ -n "${TPU_CI_IMAGE:-}" ]]; then
        log "Using the prebuilt image TPU_CI_IMAGE=${TPU_CI_IMAGE}."
        return 0
    fi
    local dockerfile="examples/tpu/grpo/Dockerfile" hash image name token deadline succeeded failed
    hash="$(sha256sum "${dockerfile}" | cut -c1-16)"
    image="${TPU_CI_IMAGE_REPO}:df-${hash}"
    name="verl-ci-image-${hash}"
    token="$(registry_access_token)"
    if image_in_registry "${image}" "${token}"; then
        log "Image ${image} (built from ${dockerfile}) is already in the registry."
        export TPU_CI_IMAGE="${image}"
        return 0
    fi

    # A failed build of the same Dockerfile is kept for a day (ttlSecondsAfterFinished): start over.
    if [[ "$(kubectl get job -n "${RAY_NAMESPACE}" "${name}" -o jsonpath='{.status.failed}' 2>/dev/null)" == "1" ]]; then
        log "Deleting the failed image build Job ${name} to retry."
        kubectl delete job,configmap,secret -n "${RAY_NAMESPACE}" "${name}" --ignore-not-found --wait=true
    fi
    if ! kubectl get job -n "${RAY_NAMESPACE}" "${name}" >/dev/null 2>&1; then
        log "Building ${image} from ${dockerfile} (BuildKit Job ${name}, about 25 minutes)."
        python3 - "${token}" "${TPU_CI_IMAGE_REPO%%/*}" > "${WORK_DIR}/docker-config.json" <<'PY'
import base64, json, sys
auth = base64.b64encode(f"oauth2accesstoken:{sys.argv[1]}".encode()).decode()
print(json.dumps({"auths": {sys.argv[2]: {"auth": auth}}}))
PY
        # Another run may create the same objects concurrently: AlreadyExists means its build is the one to wait for.
        kubectl create configmap -n "${RAY_NAMESPACE}" "${name}" --from-file=Dockerfile="${dockerfile}" 2>/dev/null || true
        kubectl create secret generic -n "${RAY_NAMESPACE}" "${name}" \
            --from-file=config.json="${WORK_DIR}/docker-config.json" 2>/dev/null || true
        rm -f "${WORK_DIR}/docker-config.json"
        NAME="${name}" NAMESPACE="${RAY_NAMESPACE}" IMAGE="${image}" BUILDKIT_IMAGE="${TPU_CI_BUILDKIT_IMAGE}" \
            TIMEOUT_SECONDS="$((IMAGE_BUILD_TIMEOUT_MINS * 60))" \
            python3 - tests/special_tpu/gke/image-build-job.yaml <<'PY' | kubectl create -f - 2>/dev/null || true
import os
import string
import sys

keys = ["NAME", "NAMESPACE", "IMAGE", "BUILDKIT_IMAGE", "TIMEOUT_SECONDS"]
print(string.Template(open(sys.argv[1]).read()).substitute({k: os.environ[k] for k in keys}))
PY
    else
        log "Waiting for the image build Job ${name} that another CI run started."
    fi

    deadline=$(( $(date +%s) + IMAGE_BUILD_TIMEOUT_MINS * 60 + 600 ))
    while true; do
        succeeded="$(kubectl get job -n "${RAY_NAMESPACE}" "${name}" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
        failed="$(kubectl get job -n "${RAY_NAMESPACE}" "${name}" -o jsonpath='{.status.failed}' 2>/dev/null || true)"
        if [[ "${succeeded}" == "1" ]]; then
            break
        fi
        if [[ "${failed}" == "1" || "$(date +%s)" -gt "${deadline}" ]]; then
            log "ERROR: building ${image} failed or timed out. Last lines of the build log:" >&2
            kubectl logs -n "${RAY_NAMESPACE}" "job/${name}" --tail=80 >&2 || true
            exit 1
        fi
        log "Image build ${name}: $(kubectl logs -n "${RAY_NAMESPACE}" "job/${name}" --tail=1 2>/dev/null | cut -c1-160 || echo pending)"
        sleep 60
    done
    kubectl delete configmap,secret -n "${RAY_NAMESPACE}" "${name}" --ignore-not-found >/dev/null || true
    if ! image_in_registry "${image}" "$(registry_access_token)"; then
        log "ERROR: the build Job ${name} succeeded but ${image} is not in the registry." >&2
        exit 1
    fi
    log "Built ${image}."
    export TPU_CI_IMAGE="${image}"
}

provision_ray_cluster() {
    # shape: "v6e-4" = 2x2 single-host subslice of a 2x4 node pool (4 chips per worker);
    #        "v6e-1" = whole 1x1 node from the ct6e-standard-1t pool (1 chip per worker; the pool's node
    #                  count caps the concurrent 1-chip and 2-chip tests).
    local test_name="$1" shape="$2" replicas="$3" head_cpu="$4" head_mem="$5" worker_cpu="$6" worker_mem="$7"
    local node_topology torch_topology annotation_key annotation_value slicebuilder
    WORKER_REPLICAS="${replicas}"
    case "${shape}" in
        v6e-4)
            WORKER_TPU_CHIPS=4 node_topology=2x4 torch_topology=2,2,1
            annotation_key=cloud.google.com/gke-tpu-slice-topology annotation_value=2x2
            slicebuilder=localhost:8471,localhost:8472,localhost:8473,localhost:8474
            ;;
        v6e-1)
            # No subslice annotation: the pod owns the whole 1-chip node.
            WORKER_TPU_CHIPS=1 node_topology=1x1 torch_topology=1,1,1
            annotation_key=verl-ci/tpu-slice-topology annotation_value=1x1
            slicebuilder=localhost:8471
            ;;
        *)
            log "ERROR: unknown worker shape ${shape}" >&2
            exit 1
            ;;
    esac
    local chips=$((WORKER_TPU_CHIPS * WORKER_REPLICAS))
    TIER="v6e-${chips}chip"
    KUEUE_QUEUE="${KUEUE_QUEUE_PREFIX}-${chips}chip"
    ensure_kueue_queue "${KUEUE_QUEUE}"
    # Keep the name short: KubeRay derives pod/service names from it (63-char limit).
    RAY_CLUSTER_NAME="$(echo "verl-ci-${test_name}-${RUN_ID}-${RUN_ATTEMPT}" | tr '[:upper:]_' '[:lower:]-' | cut -c1-37 | sed 's/-*$//')${CLUSTER_SUFFIX}"

    log "Provisioning RayCluster ${RAY_CLUSTER_NAME} (${replicas} x ${shape}, queue=${KUEUE_QUEUE}, image=${TPU_CI_IMAGE})"
    RAY_CLUSTER_NAME="${RAY_CLUSTER_NAME}" TIER="${TIER}" TEST_NAME="${test_name//_/-}" \
        HEAD_CPU="${head_cpu}" HEAD_MEMORY="${head_mem}" WORKER_CPU="${worker_cpu}" WORKER_MEMORY="${worker_mem}" \
        WORKER_REPLICAS="${replicas}" CI_RUN_ID="${RUN_ID}" KUEUE_QUEUE="${KUEUE_QUEUE}" \
        WORKER_TPU_CHIPS="${WORKER_TPU_CHIPS}" WORKER_NODE_TOPOLOGY="${node_topology}" \
        SLICE_ANNOTATION_KEY="${annotation_key}" SLICE_ANNOTATION_VALUE="${annotation_value}" \
        TORCH_TPU_TOPOLOGY="${torch_topology}" SLICEBUILDER_ADDRESSES="${slicebuilder}" \
        CI_OWNER="${GITHUB_REPOSITORY:-local}/${GITHUB_REF_NAME:-$(hostname)}#${RUN_ID}" \
        python3 - tests/special_tpu/gke/raycluster-ci.yaml <<'PY' | kubectl apply -f -
import os
import string
import sys

keys = ["RAY_CLUSTER_NAME", "RAY_NAMESPACE", "KUEUE_QUEUE", "TIER", "TEST_NAME", "CI_OWNER", "CI_RUN_ID",
        "HEAD_CPU", "HEAD_MEMORY", "WORKER_CPU", "WORKER_MEMORY", "WORKER_REPLICAS", "WORKER_TPU_CHIPS",
        "WORKER_NODE_TOPOLOGY", "SLICE_ANNOTATION_KEY", "SLICE_ANNOTATION_VALUE", "TORCH_TPU_TOPOLOGY",
        "SLICEBUILDER_ADDRESSES"]
values = {k: os.environ[k] for k in keys}
values["IMAGE"] = os.environ["TPU_CI_IMAGE"]
values["BUCKET"] = os.environ["TPU_CI_BUCKET"]
print(string.Template(open(sys.argv[1]).read()).substitute(values))
PY
}

wait_for_ray_cluster() {
    # 1. Wait for Kueue admission (RayCluster un-suspended).
    local deadline=$((SECONDS + QUEUE_TIMEOUT_MINS * 60))
    local last_report=-1000
    while (( SECONDS < deadline )); do
        local suspended
        suspended="$(kubectl get raycluster -n "${RAY_NAMESPACE}" "${RAY_CLUSTER_NAME}" -o jsonpath='{.spec.suspend}')"
        if [[ "${suspended}" != "true" ]]; then
            log "RayCluster ${RAY_CLUSTER_NAME} admitted by Kueue."
            break
        fi
        if (( SECONDS - last_report >= 60 )); then
            log "Queued in Kueue (${KUEUE_QUEUE}); waiting for TPU quota. Queue state:"
            kubectl get localqueue -n "${RAY_NAMESPACE}" "${KUEUE_QUEUE}" --no-headers 2>/dev/null || true
            kubectl get raycluster -n "${RAY_NAMESPACE}" -l "verl-ci/managed=true,verl-ci/tier=${TIER}" \
                -o custom-columns=NAME:.metadata.name,SUSPENDED:.spec.suspend,STATE:.status.state --no-headers || true
            last_report=${SECONDS}
        fi
        sleep 10
    done
    if [[ "$(kubectl get raycluster -n "${RAY_NAMESPACE}" "${RAY_CLUSTER_NAME}" -o jsonpath='{.spec.suspend}')" == "true" ]]; then
        log "ERROR: RayCluster not admitted within ${QUEUE_TIMEOUT_MINS} minutes." >&2
        exit 1
    fi

    # 2. Wait for head + TPU worker pods to be Running and all containers ready.
    local want_pods=$((1 + WORKER_REPLICAS))
    local want_tpu="0.0/$((WORKER_TPU_CHIPS * WORKER_REPLICAS)).0 TPU"
    local ready_pods=0
    deadline=$((SECONDS + POD_READY_TIMEOUT_MINS * 60))
    while (( SECONDS < deadline )); do
        ready_pods="$(kubectl get pods -n "${RAY_NAMESPACE}" -l "ray.io/cluster=${RAY_CLUSTER_NAME}" --no-headers 2>/dev/null \
            | awk '{split($2, r, "/"); if (r[1] == r[2] && $3 == "Running") c++} END {print c+0}')"
        if [[ "${ready_pods}" -ge "${want_pods}" ]]; then
            break
        fi
        log "Ready pods: ${ready_pods}/${want_pods} (1 head + ${WORKER_REPLICAS} TPU worker(s))."
        sleep 10
    done
    kubectl get pods -n "${RAY_NAMESPACE}" -l "ray.io/cluster=${RAY_CLUSTER_NAME}" -o wide
    if [[ "${ready_pods}" -lt "${want_pods}" ]]; then
        log "ERROR: RayCluster ${RAY_CLUSTER_NAME} pods not ready within ${POD_READY_TIMEOUT_MINS} minutes of admission." >&2
        kubectl get events -n "${RAY_NAMESPACE}" --sort-by=.lastTimestamp 2>/dev/null \
            | awk -v name="${RAY_CLUSTER_NAME}" 'index($0, name)' | tail -n 20 >&2 || true
        exit 1
    fi

    HEAD_POD="$(kubectl get pods -n "${RAY_NAMESPACE}" -l "ray.io/cluster=${RAY_CLUSTER_NAME},ray.io/node-type=head" \
        -o jsonpath='{.items[0].metadata.name}')"
    TPU_NODES="$(kubectl get pods -n "${RAY_NAMESPACE}" -l "ray.io/cluster=${RAY_CLUSTER_NAME},ray.io/node-type=worker" \
        -o jsonpath='{.items[*].spec.nodeName}')"

    # 3. Wait for the TPU workers to register their chips with Ray.
    deadline=$((SECONDS + 300))
    while (( SECONDS < deadline )); do
        local tpu_usage
        tpu_usage="$(kubectl exec -n "${RAY_NAMESPACE}" "${HEAD_POD}" -c ray-head -- ray status 2>/dev/null \
            | awk '/[0-9.]+\/[0-9.]+ TPU([[:space:]]|$)/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0); print $0; exit}' || true)"
        log "Ray TPU status: ${tpu_usage:-<initializing>} (want ${want_tpu})"
        if [[ "${tpu_usage}" == "${want_tpu}" ]]; then
            return 0
        fi
        sleep 10
    done
    log "ERROR: Ray cluster ${RAY_CLUSTER_NAME} did not register $((WORKER_TPU_CHIPS * WORKER_REPLICAS)) TPU chips." >&2
    kubectl get pods -n "${RAY_NAMESPACE}" -l "ray.io/cluster=${RAY_CLUSTER_NAME}" -o wide >&2
    exit 1
}

connect_ray_dashboard() {
    local svc="${RAY_CLUSTER_NAME}-head-svc"
    local deadline=$((SECONDS + 60))
    # Runner pods live in the same GKE cluster: talk to the head service directly.
    while (( SECONDS < deadline )); do
        if curl -sf "http://${svc}.${RAY_NAMESPACE}.svc.cluster.local:8265/api/version" >/dev/null 2>&1; then
            export RAY_ADDRESS="http://${svc}.${RAY_NAMESPACE}.svc.cluster.local:8265"
            log "Using in-cluster Ray head service: ${RAY_ADDRESS}"
            return 0
        fi
        if [[ -z "${KUBERNETES_SERVICE_HOST:-}" ]]; then
            break
        fi
        sleep 5
    done

    log "Starting port-forward to svc/${svc} on localhost:${PORT_FORWARD_PORT}..."
    kubectl port-forward -n "${RAY_NAMESPACE}" "svc/${svc}" "${PORT_FORWARD_PORT}:8265" >/dev/null 2>&1 &
    PF_PID=$!
    deadline=$((SECONDS + 60))
    while (( SECONDS < deadline )); do
        if curl -sf "http://127.0.0.1:${PORT_FORWARD_PORT}/api/version" >/dev/null 2>&1; then
            export RAY_ADDRESS="http://127.0.0.1:${PORT_FORWARD_PORT}"
            log "Port-forward established at ${RAY_ADDRESS}"
            return 0
        fi
        sleep 1
    done
    log "ERROR: Failed to reach the Ray head service." >&2
    exit 1
}

print_environment() {
    # Like the "Check final pip list" / "Check npu and CANN info" steps of the GPU / NPU workflows: record
    # the TPU slices and the image's key packages, so a failure can be attributed to an environment change.
    log "Ray cluster status:"
    kubectl exec -n "${RAY_NAMESPACE}" "${HEAD_POD}" -c ray-head -- ray status 2>/dev/null \
        | sed -n '/^Resources/,/^Demands/p' || true
    local worker
    worker="$(kubectl get pods -n "${RAY_NAMESPACE}" -l "ray.io/cluster=${RAY_CLUSTER_NAME},ray.io/node-type=worker" \
        -o jsonpath='{.items[0].metadata.name}')"
    log "TPU environment of worker ${worker}:"
    kubectl exec -n "${RAY_NAMESPACE}" "${worker}" -c ray-worker -- \
        sh -c 'env | sort | awk -F= "/^(TPU_|TORCH_TPU_)/"' || true
    log "Key packages in ${TPU_CI_IMAGE}:"
    kubectl exec -n "${RAY_NAMESPACE}" "${worker}" -c ray-worker -- python3 -m pip list 2>/dev/null \
        | awk '{name = tolower($1); gsub(/_/, "-", name)}
               name ~ /^(torch|torchvision|torch-tpu|tpu-sync-torch|jax|jaxlib|libtpu|vllm|vllm-torchtpu|transformers|ray|tensordict|transferqueue|spmd-types)$/' || true
    # shellcheck disable=SC2016  # expanded by the shell in the pod
    kubectl exec -n "${RAY_NAMESPACE}" "${worker}" -c ray-worker -- \
        sh -c 'echo "torchtitan: $(git -C /torchtitan rev-parse HEAD 2>/dev/null || echo "(no .git; see TORCHTITAN_COMMIT in examples/tpu/grpo/Dockerfile)")"' || true
}

warm_model_cache() {
    # The workers read the model through gcsfuse. The first read of a cold multi-GB checkpoint by several
    # trainer processes at once (mmap page faults) can take >10 minutes; read it once sequentially first.
    local model_dir="$1"
    local worker pids=()
    for worker in $(kubectl get pods -n "${RAY_NAMESPACE}" -l "ray.io/cluster=${RAY_CLUSTER_NAME},ray.io/node-type=worker" \
        -o jsonpath='{.items[*].metadata.name}'); do
        log "Warming the page cache of ${worker} with ${model_dir}..."
        kubectl exec -n "${RAY_NAMESPACE}" "${worker}" -c ray-worker -- \
            sh -c "cat ${model_dir}/*.safetensors > /dev/null" &
        pids+=($!)
    done
    wait "${pids[@]}"
}

diagnose_cluster() {
    # Explain a failed job: show the cluster's pods and whether the TPU nodes it started on are still there.
    # The TPU node pools are spot VMs, so a preempted node kills its worker pod (and the job) mid-run.
    log "RayCluster ${RAY_CLUSTER_NAME} pods:" >&2
    kubectl get pods -n "${RAY_NAMESPACE}" -l "ray.io/cluster=${RAY_CLUSTER_NAME}" -o wide >&2 || true
    local node status
    for node in ${TPU_NODES}; do
        status="$(kubectl get node "${node}" --no-headers 2>/dev/null | awk '{print $2}')"
        if [[ "${status}" != "Ready" ]]; then
            log "TPU node ${node} is ${status:-gone}: most likely a spot preemption." >&2
            touch "${WORK_DIR}/preempted"
        fi
    done
}

submit_and_verify_ray_job() {
    # Usage: submit_and_verify_ray_job <verifier> <timeout_mins> <verifier args> [ray job submit options] -- <entrypoint...>
    local verifier="$1" timeout_mins="$2" verifier_args="$3"
    shift 3
    local submit_opts=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do
        submit_opts+=("$1")
        shift
    done
    shift
    local entrypoint=("$@")

    local sub_id log_file
    sub_id="ci_${verifier}_$(date +%Y%m%d_%H%M%S)"
    log_file="${WORK_DIR}/${sub_id}.log"

    log "Submitting job ${sub_id}: ${entrypoint[*]}"
    ray job submit \
        --address "${RAY_ADDRESS}" \
        --submission-id "${sub_id}" \
        --working-dir "${REPO_ROOT}" \
        --runtime-env-json "${RUNTIME_ENV}" \
        --no-wait \
        "${submit_opts[@]}" \
        -- "${entrypoint[@]}"
    CURRENT_RAY_JOB_ID="${sub_id}"

    # Stream the log while the job runs: the driver log lives on the node that runs the entrypoint, and
    # it is gone if that node is preempted before the job ends.
    kubectl exec -n "${RAY_NAMESPACE}" "${HEAD_POD}" -c ray-head -- \
        ray job logs --address http://127.0.0.1:8265 --follow "${sub_id}" > "${log_file}.stream" 2>&1 &
    LOG_FOLLOW_PID=$!

    # Poll via kubectl exec so transient dashboard/port-forward drops never kill the CI run.
    local deadline=$((SECONDS + timeout_mins * 60))
    local final_status="TIMEOUT"
    while (( SECONDS < deadline )); do
        local status_out
        status_out="$(kubectl exec -n "${RAY_NAMESPACE}" "${HEAD_POD}" -c ray-head -- \
            ray job status --address http://127.0.0.1:8265 "${sub_id}" 2>/dev/null \
            | awk "/Job '.*' (succeeded|failed|was stopped)|Status message|Status for job/" || true)"
        log "${sub_id}: $(tail -n 1 <<<"${status_out}")"
        if [[ -z "${status_out}" ]] && ! kubectl get pod -n "${RAY_NAMESPACE}" "${HEAD_POD}" >/dev/null 2>&1; then
            log "ERROR: Ray head pod ${HEAD_POD} is gone (evicted or preempted)." >&2
            final_status="HEAD_LOST"
            break
        fi
        local status_lower="${status_out,,}"
        if [[ "${status_lower}" == *"' succeeded"* || "${status_lower}" == *"': succeeded"* ]]; then
            final_status="SUCCEEDED"
            break
        elif [[ "${status_lower}" =~ \':?\ (failed|stopped|was\ stopped) ]]; then
            final_status="FAILED"
            break
        fi
        sleep 20
    done
    if [[ "${final_status}" == "TIMEOUT" ]]; then
        log "Job ${sub_id} exceeded ${timeout_mins} minutes; stopping it."
        kubectl exec -n "${RAY_NAMESPACE}" "${HEAD_POD}" -c ray-head -- \
            ray job stop --address http://127.0.0.1:8265 "${sub_id}" >/dev/null 2>&1 || true
    fi
    CURRENT_RAY_JOB_ID=""

    log "Fetching full job logs for ${sub_id}..."
    if kubectl exec -n "${RAY_NAMESPACE}" "${HEAD_POD}" -c ray-head -- \
        ray job logs --address http://127.0.0.1:8265 "${sub_id}" > "${log_file}" 2>/dev/null; then
        rm -f "${log_file}.stream"
    else
        log "Could not fetch the job logs; using the log streamed during the run."
        mv "${log_file}.stream" "${log_file}"
    fi
    kill "${LOG_FOLLOW_PID}" 2>/dev/null || true
    wait "${LOG_FOLLOW_PID}" 2>/dev/null || true
    LOG_FOLLOW_PID=""
    tail -n 150 "${log_file}"
    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
        echo "log_file=${log_file}" >> "${GITHUB_OUTPUT}"
    fi

    if [[ "${final_status}" != "SUCCEEDED" ]]; then
        log "ERROR: Job ${sub_id} ended with status ${final_status}" >&2
        diagnose_cluster
        exit 1
    fi

    # shellcheck disable=SC2086  # verifier_args is a list of flags
    python3 tests/special_tpu/verify_tpu_e2e_log.py "${verifier}" "${log_file}" ${verifier_args}
    log "${sub_id} PASSED."
}

run_in_tpu_worker() {
    # Usage: run_in_tpu_worker <verifier> <timeout_mins> <verifier args> -- <command...>
    # Run a command directly in the (first) TPU worker container instead of as a Ray job, e.g. torchrun on a
    # single-host slice: copy this checkout and the plugin into the pod, run the command there with the same
    # environment the Ray jobs get, then verify its output.
    local verifier="$1" timeout_mins="$2" verifier_args="$3"
    shift 4
    local worker remote=/tmp/verl-tpu-ci log_file rc=0
    worker="$(kubectl get pods -n "${RAY_NAMESPACE}" -l "ray.io/cluster=${RAY_CLUSTER_NAME},ray.io/node-type=worker" \
        -o jsonpath='{.items[0].metadata.name}')"
    log_file="${WORK_DIR}/${verifier}_in_worker_$(date +%Y%m%d_%H%M%S).log"
    local kexec=(kubectl exec -n "${RAY_NAMESPACE}" "${worker}" -c ray-worker --)

    log "Copying the checkout and verl-hardware-plugin to ${worker}:${remote}"
    "${kexec[@]}" sh -c "rm -rf ${remote} && mkdir -p ${remote}/repo ${remote}/plugin"
    tar -C "${REPO_ROOT}" --exclude=.git --exclude=.venv -cf - . | \
        kubectl exec -i -n "${RAY_NAMESPACE}" "${worker}" -c ray-worker -- tar -xf - -C "${remote}/repo"
    tar -C "$(dirname "${PLUGIN_PKG}")" -cf - "$(basename "${PLUGIN_PKG}")" | \
        kubectl exec -i -n "${RAY_NAMESPACE}" "${worker}" -c ray-worker -- tar -xf - -C "${remote}/plugin"

    log "Running in ${worker}: $*"
    # shellcheck disable=SC2016  # expanded by the shell in the pod
    timeout "$((timeout_mins * 60))" "${kexec[@]}" bash -c \
        'cd "$0/repo" && export PYTHONPATH="$0/plugin:$0/repo:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 VERL_PLATFORM=tpu VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin && exec "$@"' \
        "${remote}" "$@" > "${log_file}" 2>&1 || rc=$?
    tail -n 150 "${log_file}"
    if [[ "${rc}" -ne 0 ]]; then
        log "ERROR: '$*' in ${worker} exited with code ${rc}" >&2
        diagnose_cluster
        exit 1
    fi
    # shellcheck disable=SC2086  # verifier_args is a list of flags
    python3 tests/special_tpu/verify_tpu_e2e_log.py "${verifier}" "${log_file}" ${verifier_args}
    log "${verifier} in ${worker} PASSED."
}

# ---------------------------------------------------------------------------------------------
# Tests. Each runs as Ray job(s) on its own RayCluster. Settings are passed to the example scripts as
# environment variables in the entrypoint, like the examples' READMEs do.
# ---------------------------------------------------------------------------------------------

test_tpu_platform() {
    # torch_tpu basics + TPU-vs-CPU MLP + verl/verl-hardware-plugin TPU platform on 1 chip.
    submit_and_verify_ray_job smoke 15 "" -- python3 tests/special_tpu/tpu_smoke_test.py
}

test_sft() {
    # Qwen3-0.6B GSM8K multi-turn SFT, TorchTitan FSDP2 on one v6e-4 slice, in both launch modes of the script.
    # 1. Ray single controller (verl.trainer.sft_trainer_ray), submitted as a Ray job.
    submit_and_verify_ray_job sft 30 "--steps 8" -- \
        env RAY_DATA_HOME="${TPU_CI_DATA_HOME}" SMOKE_TEST=1 NNODES_TRAINER=1 N_CHIPS_TRAINER=4 SFT_TRAINER_MODE=ray \
        bash examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh
    # 2. torchrun (verl.trainer.sft_trainer, the script's default) inside the TPU container. Not as a Ray job:
    #    a Ray driver that reserves all 4 chips of the host gets the host-default TPU environment from Ray (it
    #    drops TPU_HOST_BOUNDS / TPU_CHIPS_PER_HOST_BOUNDS), which does not describe a 2x2 subslice.
    #    Default MASTER_ADDR / TPU_WORKER_HOSTNAMES to 127.0.0.1 for single-host v6e-4 pods on node pools
    #    where the GKE TPU webhook does not inject TPU_WORKER_HOSTNAMES (run_qwen3_0_6b_torchtitan.sh uses
    #    `${TPU_WORKER_HOSTNAMES%%,*}` under `set -u`).
    run_in_tpu_worker sft 30 "--steps 8" -- \
        env RAY_DATA_HOME="${TPU_CI_DATA_HOME}" SMOKE_TEST=1 NNODES_TRAINER=1 N_CHIPS_TRAINER=4 \
        TPU_WORKER_HOSTNAMES=127.0.0.1 \
        bash examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh
}

test_grpo() {
    # The 5-step Qwen3-0.6B GRPO smoke test of examples/tpu/grpo as is (2 x v6e-1, the script's default
    # N_CHIPS_TRAINER=1 N_CHIPS_ROLLOUT=1): TorchTitan trainer on tpu-group-0, vLLM rollout on tpu-group-1,
    # tpu-sync weight sync, V1 separate-async trainer.
    submit_and_verify_ray_job grpo 45 "--steps 5" -- \
        env RAY_DATA_HOME="${TPU_CI_DATA_HOME}" bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh
}

test_grpo_4b() {
    # Qwen3-4B GRPO of examples/tpu/grpo (realistic config: compile + splash attention + SimpleFSDP), shortened
    # to GRPO_4B_STEPS steps with validation before training and at the end.
    local steps="${GRPO_4B_STEPS:-20}"
    warm_model_cache "${TPU_CI_DATA_HOME}/assets/hf/Qwen3-4B"
    submit_and_verify_ray_job grpo_4b 120 "--steps ${steps}" -- \
        env RAY_DATA_HOME="${TPU_CI_DATA_HOME}" TOTAL_TRAINING_STEPS="${steps}" TEST_FREQ="${steps}" \
        bash examples/tpu/grpo/run_qwen3_4b_torchtitan.sh
}

provision_test_cluster() {
    local test_name="$1"
    case "${test_name}" in
        tpu_platform) provision_ray_cluster "${test_name}" v6e-1 1 2 8Gi 32 128Gi ;;
        sft) provision_ray_cluster "${test_name}" v6e-4 1 4 16Gi 96 400Gi ;;
        grpo) provision_ray_cluster "${test_name}" v6e-1 2 4 16Gi 32 128Gi ;;
        grpo_4b) provision_ray_cluster "${test_name}" v6e-4 2 8 32Gi 96 400Gi ;;
        *)
            echo "${USAGE}" >&2
            exit 1
            ;;
    esac
}

delete_test_cluster() {
    if [[ -n "${PF_PID}" ]]; then
        kill "${PF_PID}" 2>/dev/null || true
        PF_PID=""
    fi
    log "Deleting RayCluster ${RAY_CLUSTER_NAME}..."
    kubectl delete raycluster -n "${RAY_NAMESPACE}" "${RAY_CLUSTER_NAME}" --wait=false --ignore-not-found || true
    RAY_CLUSTER_NAME="" HEAD_POD="" TPU_NODES=""
}

run_test() {
    local test_name="$1" attempt rc
    for (( attempt = 1; ; attempt++ )); do
        rm -f "${WORK_DIR}/preempted"
        provision_test_cluster "${test_name}"
        wait_for_ray_cluster
        connect_ray_dashboard
        print_environment
        # Run the test in a subshell, with errexit, so that its failure (`exit 1`) comes back here.
        set +e
        (
            set -e
            "test_${test_name}"
        )
        rc=$?
        set -e
        if [[ "${rc}" -eq 0 ]]; then
            return 0
        fi
        if [[ ! -f "${WORK_DIR}/preempted" || "${attempt}" -gt "${PREEMPTION_RETRIES}" ]]; then
            exit "${rc}"
        fi
        log "A TPU node of ${RAY_CLUSTER_NAME} was preempted; re-running ${test_name} on a new RayCluster" \
            "(retry ${attempt} of ${PREEMPTION_RETRIES})."
        delete_test_cluster
        CLUSTER_SUFFIX="-r${attempt}"
    done
}

mkdir -p "${WORK_DIR}"
connect_gke_cluster

if [[ "${MODE}" == "teardown" ]]; then
    # Safety net for cancelled workflows: delete every cluster created by this CI run.
    selector="verl-ci/managed=true,verl-ci/run-id=${RUN_ID}"
    if [[ -n "${2:-}" ]]; then
        selector="${selector},verl-ci/test=${2//_/-}"
    fi
    log "Deleting RayClusters matching ${selector}"
    delete_ray_clusters "${selector}"
    exit 0
fi

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

reap_stale_clusters
ensure_ci_image
fetch_hardware_plugin
run_test "${MODE}"
