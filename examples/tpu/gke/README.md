# GKE KubeRay Cluster & Docker Image for TPU v6e

This directory contains the Docker image definition and KubeRay manifest for running `verl` (SFT and GRPO RL) on Google Kubernetes Engine (GKE) with **Cloud TPU v6e** (`tpu-v6e-slice`) node pools:

- [`Dockerfile.tpu`](Dockerfile.tpu): Builds the unified TPU runtime image containing `torch==2.11.0`, `torch_tpu`, `torchtitan`, `vllm==0.14.0`, `vllm_tpu`, `jax==0.9.0`, `libtpu`, and `verl` (`us-west2-docker.pkg.dev/tpu-pytorch/raycluster/verl-tpu:v20260918-fi0918`).
- [`ray-tpu-v6e8-2slice.yaml`](ray-tpu-v6e8-2slice.yaml): KubeRay `RayCluster` manifest provisioning 1 CPU Ray head pod (`ray-head`) and 2 multi-host TPU v6e-8 slices (`numOfHosts: 2`, `google.com/tpu: 4` per host = 16 TPU v6e chips total) with GCS Fuse mounted at `/data` and `400Gi` host memory per TPU pod.

---

## 🛠️ Deploying the RayCluster on GKE

```bash
# Apply the KubeRay cluster manifest
kubectl apply -f examples/tpu/gke/ray-tpu-v6e8-2slice.yaml

# Wait for 1 head pod + 4 TPU worker pods (2 slices x 2 hosts) to reach 2/2 Running
kubectl get pods -l ray.io/cluster=ray-tpu-v6e-cluster -w

# Port-forward the Ray Dashboard / Job Submission API to localhost:23333
kubectl port-forward svc/ray-tpu-v6e-cluster-head-svc 23333:8265 > /dev/null 2>&1 &
```

Once the cluster is running, see:
- **GRPO RL Training**: [`examples/tpu/grpo/README.md`](../grpo/README.md)
- **SFT Training**: [`examples/tpu/sft/README.md`](../sft/README.md)

---

## 🧪 TPU CI (`.github/workflows/e2e_tpu.yml`)

The CI does **not** use the long-lived cluster above. Each suite provisions its own short-lived RayCluster on the CI GKE cluster (`jialeic-ci-v6e8-2s-spot`) with [`tests/special_tpu/run_tpu_e2e_ci.sh`](../../../tests/special_tpu/run_tpu_e2e_ci.sh):

| Suite   | TPU workers | Chips used | What it runs |
|---------|-------------|-----------:|--------------|
| `smoke` | 1 x v6e-4 subslice | 1 | [`tests/special_tpu/tpu_smoke_test.py`](../../../tests/special_tpu/tpu_smoke_test.py): `import torch_tpu`, `.to("tpu")`, bf16 matmul, autograd |
| `sft`   | 1 x v6e-4 subslice | 4 | Qwen3-0.6B GSM8K SFT (TorchTitan FSDP2) |
| `grpo`  | 2 x v6e-4 subslices | 8 | Qwen3-0.6B GSM8K GRPO: 4-chip trainer on `tpu-group-0` + 4-chip vLLM rollout on `tpu-group-1` |

- **Queueing**: the RayCluster ([`tests/special_tpu/gke/raycluster-ci.yaml`](../../../tests/special_tpu/gke/raycluster-ci.yaml)) is labelled `kueue.x-k8s.io/queue-name: verl-tpu-ci`. Kueue ([`tests/special_tpu/gke/kueue-tpu-ci.yaml`](../../../tests/special_tpu/gke/kueue-tpu-ci.yaml), 16-chip quota) keeps it suspended until enough TPU quota is free, then admits the whole cluster at once, so concurrent CI runs wait in FIFO order instead of competing for hosts.
- **Subslicing**: each TPU worker replica requests one v6e host (`google.com/tpu: 4`) with the `cloud.google.com/gke-tpu-slice-topology: 2x2` annotation, i.e. a single-host KubeRay subslice (v6e-4) of the 2x4 (v6e-8) node pools, and shows up in Ray as its own slice (`tpu-group-<i>`). A whole host is the smallest multi-chip unit: GKE rejects smaller TPU requests on these node pools and libtpu cannot start an ICI session on a 2-chip subset of a host (`START_SESSION failed`). A single chip without ICI works, which the smoke suite uses via `TPU_VISIBLE_CHIPS`.
- **Teardown**: the script deletes its RayCluster on exit (success, failure, timeout or signal), the workflow repeats the deletion in an `if: always()` step, and every run reaps CI RayClusters older than 3 hours.

> [!WARNING]
> Do not apply `ray-tpu-v6e8-2slice.yaml` on the CI cluster: it pins all 16 chips outside of Kueue and CI RayClusters would stay unschedulable.

Run a suite by hand (from a machine with `kubectl` access and the `ray` CLI):

```bash
bash tests/special_tpu/run_tpu_e2e_ci.sh smoke   # or sft / grpo
```
