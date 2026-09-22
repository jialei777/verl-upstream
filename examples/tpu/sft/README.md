# SFT on Google Cloud TPU (v6e)

Supervised fine-tuning on Cloud TPU v6e with the **TorchTitan** engine (`engine=torchtitan`,
PyTorch FSDP2 on `torch_tpu`).

## Prerequisites

1. A TPU v6e Ray cluster (Cloud TPU VMs or GKE + KubeRay) whose image provides `torch`,
   `torch_tpu` and `libtpu`.
2. The TPU platform plugin, which registers the `tpu` platform with verl:

   ```bash
   pip install git+https://github.com/verl-project/verl-hardware-plugin.git
   ```

3. `VERL_PLATFORM=tpu` exported in the driver and in the Ray runtime env. Auto-detection also
   works on a TPU host, but setting it explicitly is the supported way to select the platform.

Check that the platform resolves before launching anything:

```bash
VERL_PLATFORM=tpu python3 -c '
from verl.plugin.platform import get_platform
p = get_platform()
print(p.device_name, p.vendor_name, p.communication_backend_name())'
# tpu google tpu_dist
```

## Run

```bash
bash examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh
```

Point `MODEL_PATH`, `TRAIN_FILE` and `TEST_FILE` at your assets (defaults assume a bucket mounted
under `$HOME/data`). `SMOKE_TEST=1` shortens the run to eight steps with a validation pass at step
four, which is enough to tell whether the loss is moving.

On a Ray cluster, submit the same script as a job:

```bash
ray job submit --address "${RAY_ADDRESS}" --working-dir . \
  --runtime-env-json '{
    "excludes": [".git", "*.log", "*.pt", "*.bin"],
    "env_vars": {
      "PYTHONPATH": ".",
      "VERL_PLATFORM": "tpu",
      "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
      "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1"
    }
  }' \
  -- bash examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh
```

## Configuration notes

**Slice topology.** `NNODES_TRAINER x N_CHIPS_TRAINER` must cover a whole slice: `libtpu`'s
slice builder initializes every chip in the slice together. A v6e-8 slice is two hosts of four
chips (`NNODES_TRAINER=2 N_CHIPS_TRAINER=4`), a v6e-4 slice is one host of four
(`NNODES_TRAINER=1 N_CHIPS_TRAINER=4 DATA_PARALLEL_SHARD_SIZE=4`).

**Keep `tensor_parallel_size=1`.** Shard over `data_parallel_shard_size` instead. With
`tensor_parallel_size > 1` TorchTitan routes the logits through a DTensor that has to be
reassembled before the loss, and that backward path intermittently produces non-finite gradients
on TPU. `optimizer_step()` skips non-finite updates, so such a run reports success while the
policy never trains. The engine logs a warning if you set it anyway.

**Sequence bucketing.** Packed sequences are padded on the host to multiples of
`VERL_TPU_SEQ_BUCKET_SIZE` (default `256`) before they are copied to the device, so XLA compiles a
bounded set of shapes (`256, 512, ... , max_seq_len`) and reuses them for the rest of the run.
Raising the bucket size compiles fewer graphs but wastes more compute per step; lowering it does
the opposite. This needs `model.use_remove_padding=True` with `data.pad_mode=no_padding`.

**`engine.attn_type=varlen`.** The packed path uses `VarlenAttention`, which verl re-routes
through `scaled_dot_product_attention` on TPU because there is no flash-attention kernel there.

## Monitoring

```bash
ray job status --address "${RAY_ADDRESS}" <JOB_ID>
ray job logs --follow --address "${RAY_ADDRESS}" <JOB_ID>
```

A healthy run shows `train/loss` decreasing and a non-zero `train/grad_norm`. A `grad_norm` that
is `inf`/`nan` on every step means every update is being skipped - check the notes above.
