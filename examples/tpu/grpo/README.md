# GRPO RL Training on Google Cloud TPU (v6e and TPU7x)

This directory contains examples and scripts for running **GRPO (Group Relative Policy Optimization) RL Training** on Google Cloud TPU v6e and TPU7x (Ironwood) instances using `verl`.

The training setup uses:
- **Actor Engine**: TorchTitan (`model_engine=torchtitan`)
- **Rollout Engine**: vLLM (`actor_rollout_ref.rollout.name=vllm`)
- **Placement Strategy**: Non-colocated multi-slice execution (Slice 0 for Trainer/Actor, Slice 1 for Rollout)

---

## Supported hardware

| | TPU v6e | TPU7x (Ironwood) |
| --- | --- | --- |
| Example script | [`run_qwen3_0_6b_torchtitan.sh`](run_qwen3_0_6b_torchtitan.sh) | [`run_qwen3_0_6b_torchtitan_tpu7x.sh`](run_qwen3_0_6b_torchtitan_tpu7x.sh) |
| Cluster manifest | [`ray-tpu-v6e8-2slice.yaml`](../gke/ray-tpu-v6e8-2slice.yaml) | [`ray-tpu7x-2x2x1-2slice.yaml`](../gke/ray-tpu7x-2x2x1-2slice.yaml) |
| Slice in the example | `2x4` (v6e-8) | `2x2x1` |
| Hosts per slice | 2 | 1 |
| Chips per host | 4 | 4 |
| **Devices (TensorCores) per chip** | **1** | **2** |
| **Devices per host** | **4** | **8** |
| HBM per device | 32 GiB | 96 GiB |
| `TORCH_TPU_TOPOLOGY` for the slice | `2,4,1` | `2,2,1,2` |
| GKE accelerator label | `tpu-v6e-slice` | `tpu7x` |

> [!IMPORTANT]
> A TPU7x chip exposes **two** independent TensorCores, and verl runs one worker
> process per *addressable device*. So a single 4-chip TPU7x host is configured
> with `n_gpus_per_node=8`, not 4. This is the only structural difference between
> the two example scripts. Note that `google.com/tpu` in the Kubernetes manifest
> still counts **chips** (`"4"`), while Ray reports `TPU: 8.0` for the same node.

> [!WARNING]
> TPU7x additionally requires `ray >= 2.58` (earlier releases do not know the
> `v7x` pod type and mis-report per-node TPU resources) and
> GKE `>= 1.34.1-gke.1829001`. Ironwood cannot be provisioned through
> `gcloud compute tpus tpu-vm create`; use GKE, Vertex AI, or a GCE MIG.

verl detects the generation automatically from the Ray node labels
(`ray.io/accelerator-type`, `ray.io/tpu-pod-type`) and from `TPU_ACCELERATOR_TYPE`.
Set `VERL_TPU_GENERATION=v7x` (or `v6e`) to bypass detection.

---


## 🚀 Quick Start

### 1. Prerequisites
Ensure you have a running Ray cluster on TPU v6e or TPU7x nodes with `verl` installed across all head and worker nodes. Apply one of the manifests in [`../gke/`](../gke/) to create it.

Environment variables required:
- `MODEL_PATH`: Path to HuggingFace model checkpoint (e.g. `/data/jialei/assets/hf/Qwen3-0.6B`)
- `TRAIN_FILE` & `TEST_FILE`: Parquet dataset files (e.g. GSM8K dataset)
- `WANDB_API_KEY` (Optional): For experiment tracking on Weights & Biases

The rest of this guide is the same for both generations; only `CLUSTER` and the
launch script differ:

```bash
# TPU v6e
export CLUSTER=ray-tpu-v6e-cluster
export RUN_SCRIPT=examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh

# TPU7x (Ironwood)
export CLUSTER=ray-tpu7x-cluster
export RUN_SCRIPT=examples/tpu/grpo/run_qwen3_0_6b_torchtitan_tpu7x.sh
```

---

### 2. Ensure Clean Cluster and Set Up Port Forwarding (Optional)

Before submitting a new job, you can reset TPU cluster state by deleting all cluster pods (or worker pods), which will be automatically recreated by the KubeRay operator:

```bash
# Delete all pods to reset head and worker nodes
kubectl delete pod -l ray.io/cluster="${CLUSTER}"

# Port forward Ray head dashboard service to local port 23333 in the background
kubectl port-forward "svc/${CLUSTER}-head-svc" 23333:8265 > /dev/null 2>&1 &
```

---

### 3. Submit a GRPO Training Job

You can submit the training job to your Ray cluster using the Ray CLI (`ray job submit`):

```bash
# Set active Ray cluster address (e.g. localhost:23333 if port-forwarded)
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
      "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1"
    }
  }' \
  -- bash "${RUN_SCRIPT}"
```

> [!TIP]
> On TPU7x, add `"VERL_TPU_GENERATION": "v7x"` to `env_vars` if the driver may
> start before any TPU node has joined the cluster; otherwise generation
> detection has nothing to read and falls back to `v6e`. The
> [TPU7x manifest](../gke/ray-tpu7x-2x2x1-2slice.yaml) already sets it on the
> worker pods.

---

## 📊 Monitoring Progress

### Check Job Status
```bash
ray job status --address http://localhost:23333 <JOB_ID>
```

### Stream Live Logs
```bash
ray job logs --follow --address http://localhost:23333 <JOB_ID>
```
