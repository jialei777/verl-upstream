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
"""1-chip TPU smoke test for the verl TPU CI.

Submitted as a Ray job to the per-run CI RayCluster. Schedules a single Ray task that
reserves exactly one TPU chip, pins the process to it, imports ``torch_tpu``, moves
tensors to the ``tpu`` device and checks a few basic ops against CPU results. It is meant
to fail fast (minutes) on broken images, driver/libtpu issues or device plugin problems
before the heavier SFT / GRPO suites spend time on them.

Usage (inside the Ray cluster):
    python3 tests/special_tpu/tpu_smoke_test.py
"""

import json
import os
import socket
import sys

import ray


def _pin_single_chip_env() -> dict:
    """Restrict this process to the single TPU chip Ray assigned to the task.

    Ray sets TPU_VISIBLE_CHIPS / TPU_CHIPS_PER_HOST_BOUNDS / TPU_HOST_BOUNDS for tasks that
    reserve a subset of a host's chips; the torch_tpu slice-builder variables injected by the
    KubeRay webhook describe the whole host, so override them for a 1x1x1 slice.
    """
    chip_ids = ray.get_runtime_context().get_accelerator_ids().get("TPU", [])
    assert len(chip_ids) == 1, f"expected exactly one assigned TPU chip, got {chip_ids}"
    port = 8471 + int(chip_ids[0])
    env = {
        "TPU_VISIBLE_CHIPS": str(chip_ids[0]),
        "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
        "TPU_HOST_BOUNDS": "1,1,1",
        "TPU_WORKER_ID": "0",
        "TPU_WORKER_HOSTNAMES": "localhost",
        "CLOUD_TPU_TASK_ID": "0",
        "TORCH_TPU_TOPOLOGY": "1,1,1",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": f"localhost:{port}",
        "TPU_PROCESS_ADDRESSES": f"localhost:{port}",
        "TPU_PROCESS_PORT": str(port),
    }
    os.environ.update(env)
    return env


@ray.remote(num_cpus=1, resources={"TPU": 1})
def tpu_smoke_task() -> dict:
    env = _pin_single_chip_env()

    import torch
    import torch_tpu  # noqa: F401  (registers the "tpu" device)

    device = torch.device("tpu")
    result = {
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "torch_tpu_version": getattr(torch_tpu, "__version__", "unknown"),
        "pinned_env": env,
    }

    # 1. Host -> device -> host round trip.
    cpu_tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    tpu_tensor = cpu_tensor.to(device)
    assert tpu_tensor.device.type == "tpu", f"tensor landed on {tpu_tensor.device}"
    torch.testing.assert_close(tpu_tensor.cpu(), cpu_tensor)

    # 2. Elementwise + reduction.
    torch.testing.assert_close((tpu_tensor * 2 + 1).sum().cpu(), (cpu_tensor * 2 + 1).sum())

    # 3. bf16 matmul (MXU path) against a CPU fp32 reference.
    torch.manual_seed(0)
    a = torch.randn(128, 256)
    b = torch.randn(256, 64)
    out = (a.to(device, torch.bfloat16) @ b.to(device, torch.bfloat16)).float().cpu()
    torch.testing.assert_close(out, a @ b, atol=0.5, rtol=5e-2)

    # 4. Autograd on device.
    w = torch.randn(64, 32, device=device, requires_grad=True)
    x = torch.randn(8, 64, device=device)
    loss = torch.nn.functional.gelu(x @ w).pow(2).mean()
    loss.backward()
    assert w.grad is not None and w.grad.device.type == "tpu"
    grad_norm = w.grad.float().norm().cpu().item()
    assert grad_norm > 0 and grad_norm == grad_norm, f"bad grad norm {grad_norm}"

    result.update({"device": str(tpu_tensor.device), "matmul_max_abs_err": (out - a @ b).abs().max().item()})
    result["grad_norm"] = grad_norm
    return result


def main() -> int:
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
    total_tpus = ray.cluster_resources().get("TPU", 0)
    print(f"[TPU smoke] cluster TPU resources: {total_tpus}")
    assert total_tpus >= 1, "no TPU resources registered in the Ray cluster"

    result = ray.get(tpu_smoke_task.remote(), timeout=900)
    print("[TPU smoke] result:\n" + json.dumps(result, indent=2, default=str))
    print("[TPU smoke] PASSED: torch_tpu import, host<->device copy, bf16 matmul and autograd on 1 TPU chip.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
