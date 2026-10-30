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
"""1-chip TPU test: tpu-sync raw transfer accepts the tensors Raiden binds for MoE models.

Raiden sends each MoE layer's expert stack folded to 2-D (``[E_local * F, D]``, see ``expert_stack_wire`` /
``export_local_shards``). tpu-sync's raw transfer rejects a tensor whose leading dim differs from its base storage
buffer's, and since torch_tpu 449c975e1 a ``clone()`` of the folded view shares the ``[E_local, F, D]`` base, so the
folded stack is rejected unless ``adapt_for_raw_transfer`` rebinds it. The dense-model RL CI test never builds such a
tensor; this test does, in seconds, so a CI image bump that changes torch_tpu / tpu-sync behaviour is caught here.

Checks, on one TPU chip:
  1. A dense 2-D weight is accepted as-is (control).
  2. The folded expert stack, built exactly as ``export_local_shards`` builds it, ends up accepted after
     ``adapt_for_raw_transfer`` (whether or not tpu-sync rejected it first; both are reported).
  3. Rebinding preserves the values and the logical shape.

Usage (inside the Ray cluster):
    python3 tests/special_tpu/raiden_raw_transfer_test.py
"""

import json
import os
import sys

import ray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tpu_smoke_test import _pin_single_chip_env  # noqa: E402


@ray.remote(num_cpus=1, resources={"TPU": 1})
def raiden_raw_transfer_task() -> dict:
    env = _pin_single_chip_env()

    import torch
    import torch_tpu  # noqa: F401  (registers the "tpu" device)

    from verl.checkpoint_engine.raiden_checkpoint_engine import (
        _raw_transfer_error,
        adapt_for_raw_transfer,
        validate_and_sanitize_tensors,
    )

    device = torch.device("tpu")
    bind_ip = "127.0.0.1"
    torch.manual_seed(0)

    # Shapes are tile multiples like the real ones (Qwen3-30B-A3B: [8, 768, 2048] per rank at FSDP16).
    experts, rows, cols = 4, 64, 256
    dense_cpu = torch.randn(256, 512).to(torch.bfloat16)
    stack_cpu = torch.randn(experts, rows, cols).to(torch.bfloat16)

    dense = dense_cpu.to(device)
    local = stack_cpu.to(device)  # this rank's FSDP Shard(0) of an expert stack: whole experts
    folded = local.reshape(experts * rows, cols).clone()  # as export_local_shards builds the wire tensor

    named = [
        ("model.layers.0.mlp.experts.gate_proj.weight", folded),
        ("model.layers.0.self_attn.q_proj.weight", dense),
    ]
    named = validate_and_sanitize_tensors(named, device=device)
    initial = {n: _raw_transfer_error(t, bind_ip) for n, t in named}
    print(f"[raiden raw-transfer] tpu-sync before rebinding: {initial}")
    assert initial["model.layers.0.self_attn.q_proj.weight"] is None, "a plain dense 2-D weight was rejected"

    adapted, methods, report = adapt_for_raw_transfer(named, bind_ip, "raw-transfer test")
    print(f"[raiden raw-transfer] rebind methods: {methods}; report: {report}")

    for n, t in adapted:
        err = _raw_transfer_error(t, bind_ip)
        assert err is None, f"{n} {tuple(t.shape)} still rejected after adapt_for_raw_transfer: {err}"
    adapted = dict(adapted)
    got_folded = adapted["model.layers.0.mlp.experts.gate_proj.weight"]
    assert tuple(got_folded.shape) == (experts * rows, cols), f"logical shape changed: {tuple(got_folded.shape)}"
    torch.testing.assert_close(got_folded.cpu(), stack_cpu.reshape(experts * rows, cols), rtol=0, atol=0)
    torch.testing.assert_close(adapted["model.layers.0.self_attn.q_proj.weight"].cpu(), dense_cpu, rtol=0, atol=0)

    return {
        "torch_tpu_version": getattr(torch_tpu, "__version__", "unknown"),
        "pinned_env": env,
        "folded_rejected_before_rebind": initial["model.layers.0.mlp.experts.gate_proj.weight"],
        "rebind_methods": methods,
    }


def main() -> int:
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
    assert ray.cluster_resources().get("TPU", 0) >= 1, "no TPU resources registered in the Ray cluster"
    result = ray.get(raiden_raw_transfer_task.remote(), timeout=900)
    print("[raiden raw-transfer] result:\n" + json.dumps(result, indent=2, default=str))
    print(
        "[raiden raw-transfer] PASSED: dense weight and folded MoE expert stack are accepted by tpu-sync raw "
        "transfer (after adapt_for_raw_transfer), values and shapes preserved."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
