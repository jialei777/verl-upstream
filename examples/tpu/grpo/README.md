# GRPO RL Training on Google Cloud TPU with `verl` + `verl-hardware-plugin`

This directory contains the reference script (`run_qwen3_0_6b_torchtitan.sh`) for running **GRPO (Group Relative Policy Optimization) RL training** on Google Cloud TPU v6e using `verl` core together with [`verl-hardware-plugin`](https://github.com/verl-project/verl-hardware-plugin).

## Architecture Overview

Unlike the monolithic `tpu-main` fork, this branch keeps all TPU-specific runtime implementations out-of-tree in `verl-hardware-plugin` while using `verl` core's `separate_async` V1 trainer:

- **`verl` core (`pr34-grpo-0.6b-core-fixes`)** provides:
  - Generic platform hooks (`current_platform.ray_init_kwargs()`, `current_platform.ray_local_rank_override()`, `current_platform.supports_device_sharing`, `vLLMHttpServer.collective_rpc` return propagation, and per-slice Ray resource pool pinning).
  - V1 disaggregated async PPO/GRPO trainer (`trainer.use_v1=True`, `trainer.v1.trainer_mode=separate_async`).
- **`verl-hardware-plugin`** registers:
  - `PlatformTPU` (`@PlatformRegistry.register("tpu")`)
  - `TorchTitanTPUEngineWithLMHead` (`@EngineRegistry.register(model_type="language_model", backend=["torchtitan"], device=["tpu"])`)
  - `TPUCheckpointEngine` (`@CheckpointEngineRegistry.register("tpu")`) and vLLM TPU rollout patches (`verl_hardware_plugin/rollout/tpu_vllm.py`).

---

## FAQ: Do I Need a New Docker Image? How Do I Install `verl-hardware-plugin`?

**No new Docker image is needed.** You can use the existing TPU RayCluster image (`us-west2-docker.pkg.dev/tpu-pytorch/raycluster/verl-tpu:v20261001-tsync990787257`) directly.

`verl-hardware-plugin` is a pure-Python package (`verl_hardware_plugin/`) whose runtime dependencies (`torch`, `torch_tpu`, `torchtitan`, `vllm`, `vllm_tpu`, `ray`) are already installed in the existing TPU image. You can provide `verl-hardware-plugin` to the cluster in two ways:

1. **Without rebuilding the image (Recommended — via Ray `runtime_env.py_modules`):**
   Clone `verl-hardware-plugin` on the machine where you run `ray job submit`:
   ```bash
   git clone https://github.com/verl-project/verl-hardware-plugin.git
   ```
   Then pass `"py_modules": ["/path/to/verl-hardware-plugin/verl_hardware_plugin"]` and `"VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin"` in `--runtime-env-json` (see **Option A** below). Ray automatically ships the `verl_hardware_plugin` module to the head pod and all TPU worker pods at job startup.
2. **Installing into an image or persistent environment (`pip install`):**
   If you are building a custom Docker image or running inside a shared virtualenv on the nodes, install it with `--no-build-isolation`:
   ```bash
   git clone https://github.com/verl-project/verl-hardware-plugin.git
   pip install --no-build-isolation -e ./verl-hardware-plugin
   ```
   Once installed via `pip`, `verl` automatically discovers `verl_hardware_plugin` through its `[project.entry-points."verl.plugins"]` hook (see **Option B** below).

---

## Prerequisites

1. **Repositories on your submission host**:
   - `verl` checked out at `pr34-grpo-0.6b-core-fixes`
   - `verl-hardware-plugin` checked out at `main` (`https://github.com/verl-project/verl-hardware-plugin.git`)
2. **Ray TPU Cluster**:
   - A KubeRay cluster on TPU v6e (2 slices of `v6e-8`, 4 hosts $\times$ 4 chips = 16 TPU chips total: 1 slice for the 8-chip TorchTitan FSDP2 trainer and 1 slice for the 8-chip TP=8 vLLM rollout server), running image `us-west2-docker.pkg.dev/tpu-pytorch/raycluster/verl-tpu:v20261001-tsync990787257`.
   - Port-forward the Ray head dashboard to `localhost:23333`:
     ```bash
     kubectl port-forward svc/ray-tpu-v6e-cluster-head-svc 23333:8265 > /dev/null 2>&1 &
     export RAY_ADDRESS="http://localhost:23333"
     ```
3. **Model & Dataset Paths** (visible inside the Ray pods, e.g. under `/data/jialei`):
   - `MODEL_PATH`: `/data/jialei/assets/hf/Qwen3-0.6B`
   - `TRAIN_FILE`: `/data/jialei/data/gsm8k/train.parquet`
   - `TEST_FILE`: `/data/jialei/data/gsm8k/test.parquet`

---

## How to Run (`verl` Core + `verl-hardware-plugin`)

### Option A (Recommended): Ship `verl_hardware_plugin` via Ray `runtime_env.py_modules` (No Image Rebuild)

This method requires **no changes to the container image** on the cluster. Ray packages your local `verl` working directory and `verl_hardware_plugin` package directory and distributes both to the head and worker pods. Setting `"VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin"` tells `verl/__init__.py` to import `verl_hardware_plugin` at startup on every process (driver, `TaskRunnerV1`, `WorkerDict`, and `vLLMHttpServer`), registering `PlatformTPU`, `TorchTitanTPUEngineWithLMHead`, and `TPUCheckpointEngine`.

Run from the root of the `verl` repository:

```bash
export RAY_ADDRESS="http://localhost:23333"
export PLUGIN_REPO="/path/to/verl-hardware-plugin"

ray job submit --address "${RAY_ADDRESS}" \
  --working-dir . \
  --runtime-env-json "{
    \"py_modules\": [\"${PLUGIN_REPO}/verl_hardware_plugin\"],
    \"excludes\": [\".git\", \"logs\", \"*.log\", \"*.pt\", \"*.bin\", \".venv\", \"__pycache__\", \".ruff_cache\", \".mypy_cache\"],
    \"env_vars\": {
      \"PYTHONPATH\": \".\",
      \"PYTHONUNBUFFERED\": \"1\",
      \"VERL_PLATFORM\": \"tpu\",
      \"VERL_USE_EXTERNAL_MODULES\": \"verl_hardware_plugin\",
      \"VERL_LOGGING_LEVEL\": \"INFO\",
      \"RAY_memory_monitor_refresh_ms\": \"0\",
      \"RAY_memory_usage_threshold\": \"0.99\",
      \"RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS\": \"1\",
      \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\"
    }
  }" \
  -- bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh
```

#### Quick 5-step Smoke Test

To verify that the cluster, TorchTitan trainer, vLLM rollout server, and weight synchronization come up cleanly (5 steps, `train_batch_size=4`, `max_response_length=512`), pass `SMOKE_TEST=1`:

```bash
ray job submit --address "${RAY_ADDRESS}" \
  --working-dir . \
  --runtime-env-json "{
    \"py_modules\": [\"${PLUGIN_REPO}/verl_hardware_plugin\"],
    \"excludes\": [\".git\", \"logs\", \"*.log\", \"*.pt\", \"*.bin\", \".venv\", \"__pycache__\", \".ruff_cache\", \".mypy_cache\"],
    \"env_vars\": {
      \"PYTHONPATH\": \".\",
      \"PYTHONUNBUFFERED\": \"1\",
      \"VERL_PLATFORM\": \"tpu\",
      \"VERL_USE_EXTERNAL_MODULES\": \"verl_hardware_plugin\",
      \"SMOKE_TEST\": \"1\",
      \"RAY_memory_monitor_refresh_ms\": \"0\",
      \"RAY_memory_usage_threshold\": \"0.99\",
      \"RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS\": \"1\",
      \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\"
    }
  }" \
  -- bash -c "SMOKE_TEST=1 bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh"
```

---

### Option B: Pre-install `verl-hardware-plugin` with `pip`

If you pre-install `verl-hardware-plugin` on all Ray nodes (or in your Docker image):

```bash
pip install --no-build-isolation -e /path/to/verl-hardware-plugin
```

`verl` automatically discovers and loads it via the `verl.plugins` entry point (`VERL_USE_EXTERNAL_PLUGINS=auto` by default), so `py_modules` is not required:

```bash
ray job submit --address "${RAY_ADDRESS}" \
  --working-dir . \
  --runtime-env-json '{
    "excludes": [".git", "logs", "*.log", "*.pt", "*.bin", ".venv", "__pycache__"],
    "env_vars": {
      "PYTHONPATH": ".",
      "PYTHONUNBUFFERED": "1",
      "VERL_PLATFORM": "tpu",
      "VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin",
      "RAY_memory_monitor_refresh_ms": "0",
      "RAY_memory_usage_threshold": "0.99",
      "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
      "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1"
    }
  }' \
  -- bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh
```

---

## Monitoring & Verification

```bash
# Check job status
ray job status --address "${RAY_ADDRESS}" <JOB_ID>

# Stream live logs
ray job logs --follow --address "${RAY_ADDRESS}" <JOB_ID>
```

Expected signals in the logs:
- **Plugin registration**: `Patched TPUWorker class with dummy_reset_encoder_cache and load_weights_from_ray_registry.`
- **Initial validation (`step:0`)**: `val-core/openai/gsm8k/acc/mean@1` around `0.42–0.51` for `Qwen3-0.6B` before training.
- **Rollout–Actor probability correlation**: `training/rollout_actor_probs_pearson_corr` $\ge 0.996$ from `step:1` onward.
- **Reward progression**: `critic/rewards/mean` rising from `~0.40` (step 1) toward `~0.70–0.80` over 100 steps.
