# GRPO RL Training on Google Cloud TPU (v6e)

GRPO (Group Relative Policy Optimization) RL training on Google Cloud TPU v6e, using:

- **Actor / trainer engine**: TorchTitan (`model_engine=torchtitan`)
- **Rollout engine**: vLLM (`actor_rollout_ref.rollout.name=vllm`)
- **Placement**: non-colocated multi-slice — slice 0 trains, slice 1 generates

There are two ways to get the TPU TorchTitan trainer:

| Mode | Where the TPU engine lives | When to use |
| --- | --- | --- |
| **In-tree** (default) | `verl/workers/engine/torchtitan/transformer_impl.py`, behind `device_name == "tpu"` branches | Baseline / regression reference |
| **Plugin override** | [`verl-hardware-plugin`](https://github.com/jialei777/verl-hardware-plugin), class `TorchTitanTPUEngineWithLMHead` | Validating the out-of-tree engine before the in-tree TPU branches are removed |

Both are covered below. The plugin mode needs **no image rebuild** — both repos are
uploaded to the cluster at job-submission time.

---

## 1. Connect to the cluster

```bash
export CLUSTER_NAME="jialeic-tpu-v6e8-2s-spot"
export REGION="us-central2"
export PROJECT="tpu-pytorch"

gcloud container clusters get-credentials "$CLUSTER_NAME" \
  --region "$REGION" --project "$PROJECT" --dns-endpoint
```

Sanity check — you should see one head pod and four TPU worker pods, all `2/2 Running`:

```bash
kubectl get pods -l ray.io/cluster=ray-tpu-v6e-cluster
```

---

## 2. Reset cluster state and port-forward

Ray keeps TPU chips claimed by a previous job until its actors die. If a prior run
crashed, reset the pods (KubeRay recreates them automatically):

```bash
kubectl delete pod -l ray.io/cluster=ray-tpu-v6e-cluster

# Wait for all pods to return to 2/2 Running before submitting
kubectl get pods -l ray.io/cluster=ray-tpu-v6e-cluster -w
```

Then forward the Ray dashboard:

```bash
kubectl port-forward svc/ray-tpu-v6e-cluster-head-svc 23333:8265 > /dev/null 2>&1 &
export RAY_ADDRESS="http://localhost:23333"
```

---

## 3. Submit the job

Run all submit commands from the **root of this repo** (`verl-upstream`), because
`--working-dir .` uploads the current directory and `PYTHONPATH=.` makes the uploaded
copy shadow the `verl` installed in the image. That is what lets you test local edits
without rebuilding the container.

### 3a. Baseline — in-tree TPU engine

```bash
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
  -- bash -c 'SMOKE_TEST=1 bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh'
```

### 3b. Plugin override — `verl-hardware-plugin`, no image rebuild

Two extra things compared to 3a:

1. `"py_modules": ["<plugin repo>/verl_hardware_plugin"]` — Ray zips the package,
   uploads it with the job, and prepends it to `sys.path` on the driver **and** every
   worker. Nothing is baked into the image.
2. `"VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin"` — verl's normal plugin
   discovery goes through `importlib.metadata.entry_points(group="verl.plugins")`, which
   only finds *pip-installed* packages. A `py_modules` upload is not pip-installed, so
   entry-point discovery will not see it. `VERL_USE_EXTERNAL_MODULES` is the escape
   hatch: `verl/__init__.py` calls `import_external_libs()` on it, before entry-point
   discovery, which imports the package and fires its registration decorators.

```bash
export PLUGIN_REPO="${HOME}/Work/rl/verl-hardware-plugin"

ray job submit --address "${RAY_ADDRESS}" \
  --working-dir . \
  --runtime-env-json "{
    \"py_modules\": [\"${PLUGIN_REPO}/verl_hardware_plugin\"],
    \"excludes\": [\".git\", \"logs\", \"*.log\", \"*.pt\", \"*.bin\"],
    \"env_vars\": {
      \"PYTHONPATH\": \".\",
      \"PYTHONUNBUFFERED\": \"1\",
      \"VERL_PLATFORM\": \"tpu\",
      \"VERL_USE_EXTERNAL_MODULES\": \"verl_hardware_plugin\",
      \"VERL_USE_EXTERNAL_PLUGINS\": \"none\",
      \"VERL_LOGGING_LEVEL\": \"INFO\",
      \"VLLM_USE_V1\": \"0\",
      \"RAY_memory_monitor_refresh_ms\": \"0\",
      \"RAY_memory_usage_threshold\": \"0.99\",
      \"RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS\": \"1\",
      \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\"
    }
  }" \
  -- bash -c 'SMOKE_TEST=1 bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh'
```

Note the quoting flips from `'...'` to `"..."` so `${PLUGIN_REPO}` expands; the inner
JSON quotes are therefore escaped.

`VERL_USE_EXTERNAL_PLUGINS=none` disables entry-point discovery. It is not strictly
required, but if a copy of the plugin is *also* pip-installed in the image you would
otherwise load it twice from two different paths; turning discovery off makes it
unambiguous which copy is in use.

#### Why this overrides the in-tree engine

`EngineRegistry` keys engines on `(model_type, backend, device[, vendor])`:

- in-tree registers `device=["cuda", "npu", "tpu"]` with **no vendor** → key `"tpu"`
- the plugin registers `device="tpu", vendor="google"` → key `("tpu", "google")`

`get_engine_cls()` tries the `(device, vendor)` key **first** and only falls back to the
device-only key. `PlatformTPU.vendor_name` is `"google"`, so on TPU the plugin's engine
always wins. The two keys are distinct, so there is no registration collision and the
in-tree code does not have to be deleted for this test.

The platform itself is a plain override: `PlatformRegistry.register()` replaces an
existing name and logs `PlatformRegistry: overriding tpu (PlatformTPU -> PlatformTPU)`.

---

## 4. Confirm the plugin is actually in use

Do this before trusting a green run — a misconfigured `py_modules` path fails *silently*
and you would just be re-testing the in-tree engine.

```bash
ray job logs --address "${RAY_ADDRESS}" <JOB_ID> | grep -E \
  "verl-hardware-plugin loaded|PlatformRegistry: overriding tpu|TorchTitanTPUEngineWithLMHead initialized"
```

All three lines should appear (they are `INFO`, hence `VERL_LOGGING_LEVEL=INFO` above):

```
verl-hardware-plugin loaded successfully
PlatformRegistry: overriding tpu (PlatformTPU -> PlatformTPU)
TorchTitanTPUEngineWithLMHead initialized
```

If `TorchTitanTPUEngineWithLMHead initialized` is missing, the plugin was imported but
its engine was not selected — check that `VERL_PLATFORM=tpu` reached the actor and that
`get_vendor()` returns `google`.

---

## 5. What a passing smoke test looks like

`SMOKE_TEST=1` runs 5 steps with batch 4 / `rollout.n=2` / 512-token responses. It trains
nothing useful; it only proves the stack comes up end to end. Check all of:

| Check | Expected |
| --- | --- |
| `ray job status <JOB_ID>` | `SUCCEEDED` |
| Step count in logs | `step:5` reached |
| `actor/grad_norm` | finite, and **non-zero on at least one step** |
| `actor/pg_loss` | present and finite |
| Python tracebacks | none |

> [!WARNING]
> `SUCCEEDED` alone is not sufficient. `optimizer_step()` skips the update whenever
> `grad_norm` is not finite, so a job in which the policy never trains still exits 0.
> Always eyeball `grad_norm`.

```bash
ray job logs --address "${RAY_ADDRESS}" <JOB_ID> | grep -E "actor/grad_norm|actor/pg_loss|step:"
```

A healthy tp=1 smoke run looks like `grad_norm` ≈ `1.47 / 0.0 / 1.65 / 0.0 / 0.0` — the
zeros are steps where every sample in a group got the same reward, so GRPO's advantage
collapses to zero. That is expected at `rollout.n=2`.

Once the smoke test passes, run the real thing by dropping `SMOKE_TEST=1` (100 steps on
GSM8K, batch 32, `rollout.n=8`, 1024-token responses) and watch `critic/rewards/mean`
trend upward.

---

## 6. Monitoring

```bash
ray job status --address "${RAY_ADDRESS}" <JOB_ID>
ray job logs --follow --address "${RAY_ADDRESS}" <JOB_ID>
```

TensorBoard output lands under `tensorboard/` in the job's working directory; rollout and
validation dumps go to `/tmp/verl_dump/` inside the pods.

---

## 7. Configuration notes

- `MODEL_PATH` defaults to `/data/jialei/assets/hf/Qwen3-0.6B`; the GSM8K parquet files
  default to `/data/jialei/data/gsm8k/{train,test}.parquet`. Both are pod-local paths.
- `WANDB_API_KEY` is optional; the script logs to console + TensorBoard by default.
- **Do not enable tensor parallelism.** The script pins `TENSOR_PARALLEL_SIZE=1` /
  `DATA_PARALLEL_SHARD_SIZE=8`. At `tp=2` the actor produces non-finite gradients
  (`inf / 8.3e37 / 3.8e24`) which are silently skipped. See the comment block at the top
  of `run_qwen3_0_6b_torchtitan.sh`.
- The plugin engine supports both packed (`use_remove_padding=True`) and padded (`use_remove_padding=False`) paths with `+data.pad_mode=no_padding`.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `No engine registered for device='tpu'` | `VERL_PLATFORM=tpu` did not reach the worker, or the plugin was not imported. Check §4. |
| Plugin log lines missing | `py_modules` path wrong, or it points at the repo root instead of the inner `verl_hardware_plugin/` package directory. |
| `TypeError: ... unexpected keyword argument 'images_seqlens'` | A stale plugin copy is on `sys.path` ahead of the uploaded one. Set `VERL_USE_EXTERNAL_PLUGINS=none`. |
| Job hangs before step 1 | TPU chips still held by a dead job — delete the pods (§2). |
| `path_or_uri must be a string, got NoneType` | Ray's `uv` runtime-env hook. `verl/__init__.py` sets `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`; make sure nothing re-enables it. |
| `ray job submit` prints a blank line then a bare `Aborted!` | **Not a Ray bug.** Something sent the CLI a `SIGINT`; click turns `KeyboardInterrupt` into `click.Abort`, whose only output is `Aborted!`. It shows up when the submit is run alongside another command in the same shell/process group and the upload is slow enough (~30 s for a fresh 68 MB working dir) to still be in flight. Run the submit on its own, or use the Python SDK, which raises real tracebacks. |
| `TypeError: Trainer.forward_backward_step() got an unexpected keyword argument 'input_ids'` | Plugin older than `5e34766`. The forward must go through verl's `self.model_forward_step(inputs=..., extra_inputs=..., extra_kwargs=...)`. |
| Rewards flat at 0 for the whole run | Responses truncated before the `#### <answer>` line. Raise `MAX_RESPONSE_LEN` (this is why the full run uses 1024, not 512). |
