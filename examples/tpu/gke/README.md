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
