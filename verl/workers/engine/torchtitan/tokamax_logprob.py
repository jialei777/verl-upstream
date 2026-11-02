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
"""Per-token LM-head log-probs on TPU with Tokamax's fused linear + softmax + cross-entropy Pallas kernel.

``tokamax_linear_logprob(x, labels, weight)`` returns ``log_softmax(x @ weight.T)[label]`` per token without writing
the ``[T, vocab]`` logits to HBM, in forward or backward. Both directions use Tokamax with ``reduction="none"``
(per-token loss = ``-log p(label)``, per-token cotangent in the backward):

- forward: the Mosaic-TPU Pallas kernel;
- backward (``VERL_TPU_TOKAMAX_BWD``): ``xla`` (default), Tokamax's chunked-XLA backward, which walks
  ``(b_block, v_block)`` tiles and accumulates dW in FP32; or ``pallas``, the Mosaic-TPU backward kernel. With
  tokamax 0.0.14 on jax 0.11.1 + libtpu 0.0.47 the Pallas backward returned wrong dW (errors in the first hidden
  block at the label columns, NaNs with a single hidden block), so it is opt-in until validated on a newer stack.

TorchTitan's ``TorchLinearLoss`` wraps the same kernel for SFT, but only for a reduced (scalar) loss: its fake
output has one element and its backward reshapes ``dout`` to a scalar. This module registers its own torch ops with
``(B,)`` shapes; the JAX side follows TorchTitan's wrapper (``torch_tpu._internal.pallas.pallas.custom_jax_kernel``).

Numerics: BF16 operands, FP32 accumulation and FP32 log-sum-exp in the kernel; gradients are returned in the
operands' dtypes.
"""

import importlib
import os
from functools import cache
from typing import Any

import torch

# Tokamax's default block sizes (TorchTitan uses the same in its Qwen3-30B-A3B TPU recipe).
B_BLOCK_SIZE = 1024
H_BLOCK_SIZE = 512
V_BLOCK_SIZE = 2048

_REQUIREMENT = (
    "VERL_TPU_FUSED_LOGPROB=1 requires tokamax and torch_tpu's Pallas bridge (torch_tpu._internal.pallas); "
    "unset VERL_TPU_FUSED_LOGPROB to use the unfused LM head"
)


def _import_tokamax_kernel() -> Any:
    """Import Tokamax's Mosaic-TPU linear-softmax-CE kernel module.

    Works around https://github.com/openxla/tokamax/issues/240 as TorchTitan does: importing tokamax with
    ``jaxtyping.jaxtyped`` active can fail, so a pass-through decorator is swapped in for the import.
    """
    import jaxtyping

    real_jaxtyped = jaxtyping.jaxtyped

    def _passthrough_jaxtyped(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f

    jaxtyping.jaxtyped = _passthrough_jaxtyped
    try:
        importlib.import_module("tokamax")
    finally:
        jaxtyping.jaxtyped = real_jaxtyped
    _patch_pallas_tpu_for_old_jax()
    return importlib.import_module("tokamax._src.ops.linear_softmax_cross_entropy_loss.pallas_mosaic_tpu_kernel")


def _patch_pallas_tpu_for_old_jax() -> None:
    """jax < 0.11 compatibility for the Tokamax kernel (the verl-tpu:v20261006 image ships jax 0.10.2).

    Tokamax 0.0.14 builds its core mesh with ``pltpu.TensorCoreMesh(axis_name=...)``; jax 0.10.2 has the same mesh
    object behind the factory ``pltpu.create_tensorcore_mesh(axis_name)``. No-op on jax versions that have it.
    """
    from jax.experimental.pallas import tpu as pltpu

    if not hasattr(pltpu, "TensorCoreMesh") and hasattr(pltpu, "create_tensorcore_mesh"):
        pltpu.TensorCoreMesh = lambda axis_name, **kwargs: pltpu.create_tensorcore_mesh(axis_name, **kwargs)


def _jax_older_than_0_11() -> bool:
    import jax

    major, minor = (int(p) for p in jax.__version__.split(".")[:2])
    return (major, minor) < (0, 11)


def require_tokamax(vocab_size: int | None = None) -> None:
    """Raise a clear error if the kernel, torch_tpu's Pallas bridge, or this vocabulary size is unsupported.

    On jax < 0.11 (verl-tpu:v20261006 has 0.10.2) the forward kernel returned wrong log-probs for a vocabulary whose
    size is not a multiple of 128 (e.g. 5000); multiples of 128 (Qwen3: 151,936) match the reference.
    """
    try:
        _import_tokamax_kernel()
        importlib.import_module("torch_tpu._internal.pallas.pallas")
    except Exception as e:  # ImportError, or a jax / tokamax version mismatch at import time
        raise RuntimeError(f"{_REQUIREMENT}: {e!r}") from e
    if vocab_size is not None and vocab_size % 128 != 0 and _jax_older_than_0_11():
        raise RuntimeError(
            f"{_REQUIREMENT}: on jax < 0.11 the Tokamax kernel needs a vocabulary size that is a multiple of 128, "
            f"got {vocab_size}"
        )


@cache
def _ops(b_block: int, h_block: int, v_block: int, bwd_impl: str) -> tuple[Any, Any]:
    """Register (once per configuration) the forward and backward torch ops around the Tokamax kernels."""
    import jax.numpy as jnp
    from torch_tpu._internal.pallas import pallas

    kernel = _import_tokamax_kernel()
    chunked_xla = importlib.import_module("tokamax._src.ops.linear_softmax_cross_entropy_loss.chunked_xla")

    def _fwd(x, labels, weight, b, h, v):
        # weight is the LM head's [V, H]; the kernel wants [H, V]. The transpose happens inside the executable.
        loss, lse = kernel.linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu(
            x, labels, weight.T, b_block_size=b, h_block_size=h, v_block_size=v, reduction="none"
        )
        return loss, lse

    def _bwd(dout, lse, x, labels, weight, b, h, v):
        if bwd_impl == "pallas":
            grad_x, grad_w = kernel.linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(
                dout, lse, x, labels, weight.T, b_block_size=b, h_block_size=h, v_block_size=v, reduction="none"
            )
        else:
            grad_x, grad_w = chunked_xla.linear_softmax_cross_entropy_loss_bwd_chunked_xla(
                dout,
                lse,
                x,
                labels,
                weight.T,
                b_block_sz=b,
                v_block_sz=v,
                reduction="none",
                preferred_element_type=jnp.float32,
            )
        return grad_x.astype(x.dtype), grad_w.T.astype(weight.dtype)

    compiled_fwd = pallas.custom_jax_kernel(_fwd, static_argnums=(3, 4, 5))
    compiled_bwd = pallas.custom_jax_kernel(_bwd, static_argnums=(5, 6, 7))
    suffix = f"{b_block}_{h_block}_{v_block}_{bwd_impl}"

    fwd_op = torch.library.custom_op(
        f"verl_tpu::tokamax_logprob_fwd_{suffix}",
        lambda x, labels, weight: compiled_fwd(x, labels, weight, b_block, h_block, v_block),
        mutates_args=(),
        schema="(Tensor x, Tensor labels, Tensor weight) -> (Tensor, Tensor)",
    )
    fwd_op.register_fake(
        lambda x, labels, weight: (
            torch.empty(x.shape[0], dtype=torch.float32, device=x.device),
            torch.empty(x.shape[0], dtype=torch.float32, device=x.device),
        )
    )
    bwd_op = torch.library.custom_op(
        f"verl_tpu::tokamax_logprob_bwd_{suffix}",
        lambda dout, lse, x, labels, weight: compiled_bwd(dout, lse, x, labels, weight, b_block, h_block, v_block),
        mutates_args=(),
        schema="(Tensor dout, Tensor lse, Tensor x, Tensor labels, Tensor weight) -> (Tensor, Tensor)",
    )
    bwd_op.register_fake(lambda dout, lse, x, labels, weight: (torch.empty_like(x), torch.empty_like(weight)))
    return fwd_op, bwd_op


class _TokamaxLinearLogprob(torch.autograd.Function):
    """``log p(label)`` per token = ``-loss`` of the kernel with ``reduction="none"``."""

    @staticmethod
    def forward(ctx, x, labels, weight, b_block, h_block, v_block, bwd_impl):
        fwd_op, _ = _ops(b_block, h_block, v_block, bwd_impl)
        loss, lse = fwd_op(x, labels, weight)
        ctx.save_for_backward(x, labels, weight, lse)
        ctx.config = (b_block, h_block, v_block, bwd_impl)
        return -loss

    @staticmethod
    def backward(ctx, grad_logp):
        x, labels, weight, lse = ctx.saved_tensors
        _, bwd_op = _ops(*ctx.config)
        # d(logp)/d(.) = -d(loss)/d(.): pass the negated cotangent as the kernel's per-token dout.
        dout = (-grad_logp).to(torch.float32).contiguous()
        grad_x, grad_w = bwd_op(dout, lse, x, labels, weight)
        return grad_x, None, grad_w, None, None, None, None


def tokamax_linear_logprob(
    x: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    *,
    b_block: int = B_BLOCK_SIZE,
    h_block: int = H_BLOCK_SIZE,
    v_block: int = V_BLOCK_SIZE,
    bwd_impl: str | None = None,
) -> torch.Tensor:
    """Per-token ``log_softmax(x @ weight.T)[labels]``, FP32, shape ``(T,)``.

    Args:
        x: ``(T, H)`` hidden states (cast to BF16).
        labels: ``(T,)`` token ids in ``[0, V)`` (cast to int32).
        weight: ``(V, H)`` LM-head weight (cast to BF16).
        bwd_impl: ``"xla"`` or ``"pallas"``; defaults to ``VERL_TPU_TOKAMAX_BWD`` (``"xla"``).
    """
    bwd_impl = bwd_impl or os.getenv("VERL_TPU_TOKAMAX_BWD", "xla").strip().lower() or "xla"
    if bwd_impl not in ("xla", "pallas"):
        raise ValueError(f"VERL_TPU_TOKAMAX_BWD must be 'xla' or 'pallas', got {bwd_impl!r}")
    x = x.to(torch.bfloat16)
    weight = weight.to(torch.bfloat16).contiguous()
    labels = labels.to(torch.int32)
    # The kernel returned inf for a token count that is not a multiple of b_block (e.g. 3000 with 1024) on
    # verl-tpu:v20261006; pad with zero rows (label 0) and drop them. Padded rows get a zero cotangent.
    num_tokens = x.shape[0]
    pad = -num_tokens % b_block
    if pad:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        labels = torch.nn.functional.pad(labels, (0, pad))
    logp = _TokamaxLinearLogprob.apply(x.contiguous(), labels.contiguous(), weight, b_block, h_block, v_block, bwd_impl)
    return logp[:num_tokens] if pad else logp
