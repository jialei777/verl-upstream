# Copyright 2024-2025 BAAI and Google LLC
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
"""TPU-specific vLLM patches for rotary positional embeddings."""

import logging
import os
import torch

logger = logging.getLogger(__name__)

_TPU_SIGN_CACHE: dict = {}


def _tpu_sign(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Cached [[-1], [1]] used to negate the first half of a split tensor."""
    key = (dtype, device)
    sign = _TPU_SIGN_CACHE.get(key)
    if sign is None:
        sign = torch.tensor([[-1.0], [1.0]], dtype=dtype, device=device)
        _TPU_SIGN_CACHE[key] = sign
    return sign


def _tpu_rotate_neox(x: torch.Tensor) -> torch.Tensor:
    """cat((-x2, x1), -1) without a concat. Bitwise identical under IEEE754."""
    half = x.shape[-1] // 2
    swapped = x.unflatten(-1, (2, half)).flip(-2)
    return (swapped * _tpu_sign(x.dtype, x.device)).flatten(-2)


def _tpu_widen(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """[..., half] -> [..., 1, 2 * half], i.e. cat((t, t), -1).unsqueeze(-2)."""
    lead = t.shape[:-1]
    half = t.shape[-1]
    return (
        t.unsqueeze(-2)
        .expand(*lead, 2, half)
        .reshape(*lead, 2 * half)
        .unsqueeze(-2)
        .to(dtype)
    )


# TODO: Remove this workaround once upstream vLLM PR #56879 is merged and released.
# https://github.com/vllm-project/vllm/pull/56879
def patch_tpu_rotary_emb():
    """Patches vLLM ApplyRotaryEmb and rotate_neox with concat-free implementations.

    On Google TPU, torch.cat along the last dimension lowers to a strided DynamicUpdateSlice (DUS).
    When fused with elementwise ops, it fails the unaligned DUS predicate inside the XLA:TPU
    fusion emitter (b/501165531) and causes a fatal crash:
        Check failed: fusion_util::IsFusibleUnalignedDUS(...)

    This patch replaces the concatenation with an unflatten + flip formulation that is
    bitwise identical and emits no DUS instructions.
    """
    try:
        import vllm.model_executor.layers.rotary_embedding.common as rotary_common
        from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

        if getattr(ApplyRotaryEmb, "_verl_tpu_rotary_patched", False):
            return

        def patched_forward_static(
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            is_neox_style: bool = True,
            enable_fp32_compute: bool = False,
        ) -> torch.Tensor:
            origin_dtype = x.dtype
            if enable_fp32_compute:
                x = x.float()

            if is_neox_style:
                cos_f = _tpu_widen(cos, x.dtype)
                sin_f = _tpu_widen(sin, x.dtype)
                output = x * cos_f + _tpu_rotate_neox(x) * sin_f
            else:
                cos = cos.unsqueeze(-2).to(x.dtype)
                sin = sin.unsqueeze(-2).to(x.dtype)
                x1 = x[..., ::2]
                x2 = x[..., 1::2]
                o1 = x1 * cos - x2 * sin
                o2 = x2 * cos + x1 * sin
                output = torch.stack((o1, o2), dim=-1).flatten(-2)

            if enable_fp32_compute:
                output = output.to(origin_dtype)
            return output

        ApplyRotaryEmb.forward_static = staticmethod(patched_forward_static)
        rotary_common.rotate_neox = _tpu_rotate_neox
        ApplyRotaryEmb._verl_tpu_rotary_patched = True
        logger.info("Successfully applied TPU concat-free RoPE patch to vLLM.")
    except Exception as e:
        logger.warning(f"Failed to apply TPU rotary embedding patch to vLLM: {e}")


def apply_tpu_vllm_patches() -> None:
    """Apply TPU-specific vLLM patches."""
    if os.environ.get("VERL_PLATFORM") == "tpu":
        patch_tpu_rotary_emb()
