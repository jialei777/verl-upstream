# Copyright (c) 2026 Google LLC. All rights reserved.
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
"""TorchTitan model-input helpers for Google TPU (``torch_tpu``).

Two things the TorchTitan engine does are not usable on the TPU backend:

* the packed (``use_remove_padding``) input path hands the model a tensor whose length is the
  number of tokens in the micro batch, so XLA compiles a fresh executable per micro batch;
* ``VarlenAttention`` dispatches to a flash-attention kernel that ``torch_tpu`` does not provide.

:func:`pad_packed_inputs_for_tpu` fixes the first by building bucketed inputs (and an explicit
document mask) on the host, :func:`monkey_patch_varlen_attention_tpu` fixes the second by routing
the varlen attention through ``scaled_dot_product_attention`` with that mask.
"""

import logging
import os
from typing import Any

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.utils.tpu_utils import bucket_length, get_tpu_seq_bucket_size

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def monkey_patch_varlen_attention_tpu() -> None:
    """Route TorchTitan's ``VarlenAttention`` through ``scaled_dot_product_attention``.

    The replacement rebuilds the block-diagonal causal mask from whatever the caller passes
    (cumulative sequence lengths, a 4D mask, or a 2D padding mask) and lets the TPU backend lower
    the attention itself. Repeating k/v covers grouped-query attention, which the flash kernel
    would otherwise handle.
    """
    try:
        from torchtitan.models.common.attention import VarlenAttention
    except Exception as e:  # pragma: no cover - depends on the installed torchtitan version
        logger.warning(f"Failed to patch VarlenAttention for TPU: {e}")
        return

    def tpu_varlen_forward(self, q_BLNH, k_BLNH, v_BLNH, *, attention_masks, scale=None, out_transform=None, **kwargs):
        xq, xk, xv = q_BLNH, k_BLNH, v_BLNH
        cu_seqs = None
        for attr in ("cu_seq_q", "cu_seqlens_q", "cu_seqlens"):
            if hasattr(attention_masks, attr):
                cu_seqs = getattr(attention_masks, attr)
                break

        if cu_seqs is not None:
            positions = torch.arange(xq.shape[1], device=xq.device)
            seq_indices = (positions.unsqueeze(1) >= cu_seqs.unsqueeze(0)).sum(dim=1) - 1
            same_seq_mask = seq_indices.unsqueeze(1) == seq_indices.unsqueeze(0)
            causal_mask = positions.unsqueeze(1) >= positions.unsqueeze(0)
            mask = (same_seq_mask & causal_mask).unsqueeze(0).unsqueeze(0)
        elif isinstance(attention_masks, torch.Tensor) and attention_masks.dim() == 4:
            mask = attention_masks.to(torch.bool)
        else:
            positions = torch.arange(xq.shape[1], device=xq.device)
            causal_mask = (positions.unsqueeze(1) >= positions.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
            if isinstance(attention_masks, torch.Tensor):
                mask = causal_mask & attention_masks.unsqueeze(1).unsqueeze(2).to(torch.bool)
            else:
                mask = causal_mask

        q, k, v = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        if q.shape[1] != k.shape[1]:
            # grouped-query attention: expand the k/v heads to match the query heads
            num_repeat = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(num_repeat, dim=1)
            v = v.repeat_interleave(num_repeat, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        return attn_out.transpose(1, 2)

    VarlenAttention.forward = tpu_varlen_forward
    logger.info("Patched VarlenAttention.forward for TPU execution.")


def pad_packed_inputs_for_tpu(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    micro_batch: TensorDict,
    device: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build bucket-padded packed model inputs on the host and move them to the device once.

    Returns ``(input_ids, position_ids, labels, attention_mask, orig_seq_len)`` where the first four
    are padded to a multiple of the sequence bucket size and ``orig_seq_len`` is the unpadded token
    count, which the caller needs to slice the padding back off the model output.

    The 4D attention mask is materialized here rather than through TorchTitan's mask builder
    because the padding tokens must not attend to (or be attended to by) any real document: each
    padded position is given its own negative document id.
    """
    from verl.utils import tensordict_utils as tu

    bucket_size = get_tpu_seq_bucket_size()
    input_ids_cpu = input_ids.values().detach().cpu().unsqueeze(0)
    if position_ids.dim() == 3:
        position_ids_cpu = position_ids.values().detach().cpu().unsqueeze(1)
    else:
        position_ids_cpu = position_ids.values().detach().cpu().unsqueeze(0)
    labels_cpu = torch.roll(input_ids_cpu, shifts=-1, dims=1)

    orig_seq_len = int(input_ids_cpu.shape[1])
    pad_len = bucket_length(orig_seq_len, bucket_size) - orig_seq_len
    padded_seq_len = orig_seq_len + pad_len

    pos_2d_cpu = position_ids_cpu[0] if position_ids_cpu.dim() == 3 else position_ids_cpu
    if pad_len > 0:
        input_ids_cpu = F.pad(input_ids_cpu, (0, pad_len), value=0)
        labels_cpu = F.pad(labels_cpu, (0, pad_len), value=0)
        position_ids_cpu = F.pad(position_ids_cpu, (0, pad_len), value=0)
        pos_2d_cpu = F.pad(pos_2d_cpu, (0, pad_len), value=0)

    # document id per token: from the nested offsets when available, otherwise from the points
    # where the position ids restart
    if getattr(input_ids, "is_nested", False):
        seq_lens = input_ids.offsets().diff().detach().cpu()
        seq_ids_1d = torch.repeat_interleave(torch.arange(1, len(seq_lens) + 1, dtype=torch.int64), seq_lens)
        if pad_len > 0:
            seq_ids_1d = F.pad(seq_ids_1d, (0, pad_len), value=0)
        seq_ids = seq_ids_1d.unsqueeze(0)
    else:
        first_dummy = pos_2d_cpu[:, :1] - 1
        boundary = torch.diff(pos_2d_cpu, prepend=first_dummy, dim=-1) != 1
        boundary[:, 0] = True
        seq_ids = boundary.cumsum(dim=-1)

    idx = torch.arange(padded_seq_len, dtype=seq_ids.dtype).unsqueeze(0)
    # give every padded position a unique (negative) document id so it matches nothing but itself
    seq_ids = torch.where(idx < orig_seq_len, seq_ids, -idx - 1)
    same_seq_mask = seq_ids.unsqueeze(2) == seq_ids.unsqueeze(1)
    causal_mask = idx.unsqueeze(2) >= idx.unsqueeze(1)
    attention_mask_cpu = (same_seq_mask & causal_mask).unsqueeze(1)

    # bucket the response length too, so the loss downstream also sees static shapes
    if "responses" in micro_batch.keys() and getattr(micro_batch["responses"], "is_nested", False):
        raw_max_resp = int(micro_batch["responses"].offsets().diff().cpu().max().item())
        tu.assign_non_tensor_data(micro_batch, "max_response_len", bucket_length(raw_max_resp, bucket_size))

    return (
        input_ids_cpu.to(device=device).contiguous(),
        position_ids_cpu.to(device=device).contiguous(),
        labels_cpu.to(device=device).contiguous(),
        attention_mask_cpu.to(device=device).contiguous(),
        orig_seq_len,
    )
