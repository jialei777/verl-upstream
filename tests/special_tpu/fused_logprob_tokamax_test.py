# Copyright 2026 Google LLC
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
"""1-chip TPU test: Tokamax fused LM-head log-probs (``VERL_TPU_FUSED_LOGPROB=1``) match the unfused math.

``verl/workers/engine/torchtitan/tokamax_logprob.py`` computes per-token ``log_softmax(x @ W.T / T)[label]`` with
Tokamax's fused linear + softmax + cross-entropy kernels (Pallas forward, chunked-XLA backward by default) so the
``[T, vocab]`` logits never reach HBM. This test compares its log-probs and its gradients (dX, dW, with a random
per-token cotangent) against the plain PyTorch math on the same chip:

  * reference "fp32": FP32-accumulated logits of the BF16 operands, FP32 log-softmax (the kernel's math);
  * reference "bf16": BF16 logits (the unfused head under autocast), FP32 log-softmax (reported only).

Shapes include the Qwen3 vocabulary (151,936, not a multiple of the 2048 vocab block) and a ragged token count.

Usage (inside the Ray cluster):
    python3 tests/special_tpu/fused_logprob_tokamax_test.py
"""

import json
import os
import sys

import ray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tpu_smoke_test import _pin_single_chip_env  # noqa: E402

# (tokens, hidden, vocab, temperature)
CASES = [
    (2048, 2048, 151936, 1.0),
    (3000, 2048, 151936, 1.0),
    (2048, 512, 4992, 0.7),  # 4992 = 2 x 2048 + 896: ragged vocab tail (a multiple of 128, see require_tokamax)
]
# Tolerances vs the FP32-accumulated reference. dX / dW leave the kernel in BF16 (relative rounding ~2e-3); with
# T != 1 the temperature is applied by scaling x in BF16, which adds one BF16 rounding of the logits' input.
TOL = {
    1.0: {"logp_maxabs": 1e-4, "grad_rel": 5e-3},
    0.7: {"logp_maxabs": 2e-2, "grad_rel": 1e-2},
}


def _reference(x, w, labels, g, temperature: float, *, fp32: bool):
    """Unfused log-probs and their gradients: FP32-accumulated (``fp32``) or BF16 logits, FP32 log-softmax."""
    import torch

    xr = (x.float() if fp32 else x.clone()).requires_grad_()
    wr = (w.float() if fp32 else w.clone()).requires_grad_()
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        z = (xr @ wr.t()) / temperature
    finally:
        torch.set_float32_matmul_precision(previous)
    logp = torch.log_softmax(z, -1, dtype=torch.float32).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    (logp * g).sum().backward()
    return logp.detach(), xr.grad, wr.grad


def _maxabs(a, b) -> float:
    return (a.float() - b.float()).abs().max().item()


def _rel(a, b) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


@ray.remote(num_cpus=1, resources={"TPU": 1})
def fused_logprob_task() -> dict:
    env = _pin_single_chip_env()

    import torch
    import torch_tpu  # noqa: F401  (registers the "tpu" device)

    from verl.workers.engine.torchtitan.tpu_utils import fused_linear_logprob

    device = torch.device("tpu")
    results = []
    for tokens, hidden, vocab, temperature in CASES:
        torch.manual_seed(0)
        x = torch.randn(1, tokens, hidden).to(torch.bfloat16).to(device)
        w = (torch.randn(vocab, hidden) * 0.02).to(torch.bfloat16).to(device)
        labels = torch.randint(0, vocab, (1, tokens)).to(device)
        g = torch.randn(1, tokens).to(device)

        ref32 = _reference(x, w, labels, g, temperature, fp32=True)
        ref16 = _reference(x, w, labels, g, temperature, fp32=False)

        xk = x.clone().requires_grad_()
        wk = w.clone().requires_grad_()
        logp, entropy = fused_linear_logprob(xk, wk, labels, temperature)
        assert entropy is None
        (logp * g).sum().backward()
        got = (logp.detach(), xk.grad, wk.grad)

        row = {
            "case": [tokens, hidden, vocab, temperature],
            "fp32_ref": {
                "logp_maxabs": _maxabs(got[0], ref32[0]),
                "dx_rel": _rel(got[1], ref32[1]),
                "dw_rel": _rel(got[2], ref32[2]),
            },
            "bf16_ref": {
                "logp_maxabs": _maxabs(got[0], ref16[0]),
                "dx_rel": _rel(got[1], ref16[1]),
                "dw_rel": _rel(got[2], ref16[2]),
            },
            "bf16_vs_fp32_ref_logp_maxabs": _maxabs(ref16[0], ref32[0]),
        }
        print(f"[fused logprob] {json.dumps(row)}", flush=True)
        tol = TOL[temperature]
        r = row["fp32_ref"]
        assert all(torch.isfinite(t).all().item() for t in got), f"non-finite output for {row['case']}"
        assert r["logp_maxabs"] <= tol["logp_maxabs"], f"log-prob mismatch {row}"
        assert r["dx_rel"] <= tol["grad_rel"] and r["dw_rel"] <= tol["grad_rel"], f"gradient mismatch {row}"
        results.append(row)
    return {"torch_tpu_version": getattr(torch_tpu, "__version__", "unknown"), "pinned_env": env, "cases": results}


def main() -> int:
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
    assert ray.cluster_resources().get("TPU", 0) >= 1, "no TPU resources registered in the Ray cluster"
    result = ray.get(fused_logprob_task.remote(), timeout=900)
    print("[fused logprob] result:\n" + json.dumps(result, indent=2, default=str))
    print("[fused logprob] PASSED: Tokamax fused log-probs and gradients match the FP32-accumulated reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
