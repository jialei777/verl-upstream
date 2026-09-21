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
"""TPU sequence packing utilities, attention monkey patches, and memory layout helpers."""

import logging
import os
from typing import Any

import torch
import torch.distributed
import torch.nn.functional as F
from tensordict import TensorDict

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def unwrap_metadata(val):
    """Recursively unwraps metadata values (lists, single-element tensors) to standard Python types."""
    if isinstance(val, list):
        if len(val) > 0:
            return unwrap_metadata(val[0])
        return None
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            return val.item()
        elif val.numel() > 1:
            return val.flatten()[0].item()
    return val


def monkey_patch_varlen_attention_tpu():
    """Patches TorchTitan's VarlenAttention forward method to use native scaled_dot_product_attention on TPU."""
    try:
        from torchtitan.models.common.attention import VarlenAttention

        def tpu_varlen_forward(
            self,
            q_BLNH,
            k_BLNH,
            v_BLNH,
            *,
            attention_masks,
            scale=None,
            out_transform=None,
            **kwargs,
        ):
            xq, xk, xv = q_BLNH, k_BLNH, v_BLNH
            if (
                hasattr(attention_masks, "cu_seq_q")
                or hasattr(attention_masks, "cu_seqlens_q")
                or hasattr(attention_masks, "cu_seqlens")
            ):
                cu_seqs = getattr(
                    attention_masks,
                    "cu_seq_q",
                    getattr(attention_masks, "cu_seqlens_q", getattr(attention_masks, "cu_seqlens", None)),
                )
                total_tokens = xq.shape[1]
                positions = torch.arange(total_tokens, device=xq.device)
                seq_indices = (positions.unsqueeze(1) >= cu_seqs.unsqueeze(0)).sum(dim=1) - 1
                same_seq_mask = seq_indices.unsqueeze(1) == seq_indices.unsqueeze(0)
                causal_mask = positions.unsqueeze(1) >= positions.unsqueeze(0)

                mask = same_seq_mask & causal_mask
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif isinstance(attention_masks, torch.Tensor):
                if attention_masks.dim() == 4:
                    mask = attention_masks.to(torch.bool)
                else:
                    seq_len = xq.shape[1]
                    positions = torch.arange(seq_len, device=xq.device)
                    causal_mask = (positions.unsqueeze(1) >= positions.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

                    padding_mask = attention_masks.unsqueeze(1).unsqueeze(2).to(torch.bool)
                    mask = causal_mask & padding_mask
            else:
                seq_len = xq.shape[1]
                positions = torch.arange(seq_len, device=xq.device)
                mask = (positions.unsqueeze(1) >= positions.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

            q = xq.transpose(1, 2)
            k = xk.transpose(1, 2)
            v = xv.transpose(1, 2)

            if q.shape[1] != k.shape[1]:
                num_repeat = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(num_repeat, dim=1)
                v = v.repeat_interleave(num_repeat, dim=1)

            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
            return attn_out.transpose(1, 2)

        VarlenAttention.forward = tpu_varlen_forward
        logger.info("Successfully patched VarlenAttention.forward for TPU execution.")
    except Exception as e:
        logger.warning(f"Failed to patch VarlenAttention: {e}")


def compute_global_batch_num_tokens(data: TensorDict, dp_group, tp_size: int) -> Any:
    """Computes global batch token count for loss normalization on TPU.

    On TPU, performing CPU-side all-reduce for loss normalization avoids graph desynchronization
    and JIT compilation lockups across ranks.
    """
    batch_num_tokens = data["loss_mask"].sum().cpu()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(batch_num_tokens, op=torch.distributed.ReduceOp.SUM)
        batch_num_tokens = batch_num_tokens / tp_size
    return batch_num_tokens.item()


def synchronize_tpu_loss(loss: torch.Tensor):
    """Materializes forward graph loss without blocking to split XLA forward and backward compilation passes."""
    try:
        from torch_tpu._internal.sync import synchronize

        synchronize(loss, wait=False)
    except ImportError:
        pass


def get_tpu_seq_bucket_size() -> int:
    """Returns the token bucketing multiple for TPU sequence packing (default 256)."""
    return int(os.getenv("VERL_TPU_SEQ_BUCKET_SIZE", "256"))


def bucket_length(length: int, bucket_size: int | None = None) -> int:
    """Rounds a positive sequence length up to the nearest multiple of `bucket_size`."""
    if bucket_size is None:
        bucket_size = get_tpu_seq_bucket_size()
    if bucket_size <= 1:
        return max(1, int(length))
    return max(bucket_size, ((int(length) + bucket_size - 1) // bucket_size) * bucket_size)


def pad_packed_inputs_for_tpu(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    micro_batch: TensorDict,
    device: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pads 1D packed sequence inputs on CPU to a fixed bucket multiple and transfers to TPU.

    On `torch_tpu` (XLA), passing or padding unbucketed packed token counts `orig_seq_len` on the
    TPU device causes XLA to compile and cache a new HLO executable in host CPU RAM for every
    distinct sequence length. Padding `input_ids`, `position_ids`, `labels`, and the 4D causal
    document mask on CPU to multiples of `bucket_size` (default 256) before H2D transfer ensures
    the TPU only ever sees static bucket shapes across the entire training run.
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
    padded_seq_len = bucket_length(orig_seq_len, bucket_size)
    pad_len = padded_seq_len - orig_seq_len

    pos_2d_cpu = position_ids_cpu[0] if position_ids_cpu.dim() == 3 else position_ids_cpu
    if pad_len > 0:
        input_ids_cpu = F.pad(input_ids_cpu, (0, pad_len), value=0)
        labels_cpu = F.pad(labels_cpu, (0, pad_len), value=0)
        position_ids_cpu = F.pad(position_ids_cpu, (0, pad_len), value=0)
        pos_2d_cpu = F.pad(pos_2d_cpu, (0, pad_len), value=0)

    # Build static 4D causal + document-boundary mask [1, 1, padded_seq_len, padded_seq_len] on CPU
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
    valid = idx < orig_seq_len
    seq_ids = torch.where(valid, seq_ids, -idx - 1)
    same_seq_mask = seq_ids.unsqueeze(2) == seq_ids.unsqueeze(1)
    causal_mask = idx.unsqueeze(2) >= idx.unsqueeze(1)
    attention_mask_cpu = (same_seq_mask & causal_mask).unsqueeze(1)

    # `labels_cpu` was built with a plain roll over the whole packed buffer, so the label at the
    # last position of every document is the *first token of the next document* (and the label at
    # the very last position wraps around to the first token of the buffer). Downstream extraction
    # (`response_from_nested` / `tpu_no_padding_2_padding`) gathers only
    # [seq_end - response_len - 1, seq_end - 1), so those positions are already excluded from both
    # `old_log_probs` and the PPO loss. Zero them anyway so the cross-document leak cannot reach a
    # future consumer that iterates the full packed length. This is a fixed-shape elementwise op,
    # so it does not introduce a new XLA shape.
    same_doc_as_next = seq_ids == torch.roll(seq_ids, shifts=-1, dims=1)
    labels_cpu = torch.where(same_doc_as_next, labels_cpu, torch.zeros_like(labels_cpu))

    # Bucket max_response_len on micro_batch so ppo_loss operates on static bucketed shapes.
    if "responses" in micro_batch.keys() and getattr(micro_batch["responses"], "is_nested", False):
        resp_lens = micro_batch["responses"].offsets().diff().cpu()
        raw_max_resp = int(resp_lens.max().item())
        tu.assign_non_tensor_data(micro_batch, "max_response_len", bucket_length(raw_max_resp, bucket_size))

    return (
        input_ids_cpu.to(device=device).contiguous(),
        position_ids_cpu.to(device=device).contiguous(),
        labels_cpu.to(device=device).contiguous(),
        attention_mask_cpu.to(device=device).contiguous(),
        orig_seq_len,
    )


def tpu_no_padding_2_padding(tensor: torch.Tensor, data: TensorDict) -> torch.Tensor:
    """Extracts and pads response tokens from a bucketed 1D tensor using static-shape index gather on TPU."""
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import no_padding_2_padding

    padded_values = getattr(tensor, "_tpu_padded_values", None)
    if padded_values is None:
        return no_padding_2_padding(tensor, data)

    prompt_ids = data["prompts"]
    response_ids = data["responses"]
    if not (getattr(prompt_ids, "is_nested", False) and getattr(response_ids, "is_nested", False)):
        return no_padding_2_padding(tensor, data)

    prompt_lens = prompt_ids.offsets().diff().cpu()
    response_lens = response_ids.offsets().diff().cpu()
    seq_offsets = (prompt_lens + response_lens).cumsum(dim=0)

    max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)
    if max_response_len < 0:
        max_response_len = bucket_length(int(response_lens.max().item()))
    else:
        max_response_len = bucket_length(int(max_response_len))
        tu.assign_non_tensor_data(data, "max_response_len", max_response_len)

    bsz = int(response_lens.shape[0])
    col_idx = torch.arange(max_response_len, dtype=torch.int64).unsqueeze(0)  # [1, max_response_len]
    starts = (seq_offsets - response_lens - 1).to(torch.int64).unsqueeze(1)  # [bsz, 1]
    valid_mask_cpu = col_idx < response_lens.unsqueeze(1)  # [bsz, max_response_len]
    gather_idx_cpu = torch.where(valid_mask_cpu, starts + col_idx, torch.zeros_like(col_idx)).clamp(
        min=0, max=max(0, int(padded_values.shape[0]) - 1)
    )

    device = padded_values.device
    gather_idx = gather_idx_cpu.to(device=device)
    valid_mask = valid_mask_cpu.to(device=device, dtype=padded_values.dtype)
    values_2d = padded_values.unsqueeze(0).expand(bsz, -1)
    return torch.gather(values_2d, 1, gather_idx) * valid_mask


def safe_to_padded_tensor(nt: Any, padding: Any = 0, output_size: Any = None) -> torch.Tensor:
    """Safely converts a NestedTensor to a padded dense tensor on TPU using CPU-side assembly.

    Assembling the small padded response tensors on CPU avoids compiling a new
    `tt_jit_jagged_to_padded_dense_forward` HLO executable for every unbucketed jagged length.
    """
    from verl.utils.device import get_device_id

    if not getattr(nt, "is_nested", False):
        if isinstance(nt, torch.Tensor) and nt.device.type == "cpu":
            return nt.to(device=get_device_id())
        return nt
    target_device = get_device_id() if nt.device.type == "cpu" else nt.device
    values_cpu = nt.values().detach().cpu()
    offsets_cpu = nt.offsets().detach().cpu()
    batch_size = int(offsets_cpu.shape[0]) - 1
    if batch_size <= 0:
        return torch.empty(output_size if output_size is not None else (0,), device=target_device, dtype=nt.dtype)
    lengths = offsets_cpu.diff().tolist()
    trailing_dims = tuple(values_cpu.shape[1:])
    if output_size is None:
        max_len = bucket_length(max(lengths))
        output_size = (batch_size, max_len, *trailing_dims)
    out_cpu = torch.full(output_size, padding, dtype=values_cpu.dtype)
    for i in range(batch_size):
        start = int(offsets_cpu[i].item())
        length = int(lengths[i])
        if length > 0:
            out_cpu[i, :length] = values_cpu[start : start + length]
    return out_cpu.to(device=target_device)


def select_and_to_padded_tensor(data: TensorDict, *fields: str) -> TensorDict:
    """Selects fields from a TensorDict and converts NestedTensors to bucket-padded dense tensors on TPU."""
    from verl.utils import tensordict_utils as tu
    from verl.utils.device import get_device_id

    max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)
    if max_response_len is not None and int(max_response_len) > 0:
        max_response_len = bucket_length(int(max_response_len))
    else:
        max_response_len = None

    target_device = get_device_id()
    padded_dict = {}
    for k in fields:
        if k in data.keys():
            val = data[k]
            if getattr(val, "is_nested", False):
                output_size = None
                if max_response_len is not None:
                    trailing_dims = tuple(val.values().shape[1:])
                    output_size = (int(data.batch_size[0]), max_response_len, *trailing_dims)
                padded_dict[k] = safe_to_padded_tensor(val, output_size=output_size)
            elif isinstance(val, torch.Tensor) and val.device.type == "cpu":
                padded_dict[k] = val.to(device=target_device)
            else:
                padded_dict[k] = val
    return TensorDict(padded_dict, batch_size=data.batch_size)


class _VocabParallelLogprobsFn(torch.autograd.Function):
    """Computes log-probabilities from Shard(-1) vocab-parallel logits without full-vocab backward all_gather."""

    @staticmethod
    def forward(
        ctx,
        local_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        tp_group: torch.distributed.ProcessGroup,
        tp_rank: int,
    ) -> torch.Tensor:
        orig_dtype = local_logits.dtype
        logits_fp32 = local_logits.float()
        if temperature != 1.0:
            logits_fp32 = logits_fp32 / float(temperature)

        vocab_per_rank = int(logits_fp32.size(-1))
        vocab_start = tp_rank * vocab_per_rank
        vocab_end = vocab_start + vocab_per_rank

        # 1. Global max logit across TP group
        max_logits = torch.max(logits_fp32, dim=-1, keepdim=True).values
        torch.distributed.all_reduce(max_logits, op=torch.distributed.ReduceOp.MAX, group=tp_group)

        # 2. Shifted exponentials and global partition function Z
        shifted_logits = logits_fp32 - max_logits
        exp_logits = torch.exp(shifted_logits)
        sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True)
        torch.distributed.all_reduce(sum_exp, op=torch.distributed.ReduceOp.SUM, group=tp_group)

        # 3. Target logit on the owning TP shard
        target_mask = (labels >= vocab_start) & (labels < vocab_end)
        local_labels = torch.where(target_mask, labels - vocab_start, torch.zeros_like(labels)).clamp(
            0, vocab_per_rank - 1
        )
        gathered_shifted = torch.gather(shifted_logits, dim=-1, index=local_labels.unsqueeze(-1)).squeeze(-1)
        target_shifted = torch.where(target_mask, gathered_shifted, torch.zeros_like(gathered_shifted))
        torch.distributed.all_reduce(target_shifted, op=torch.distributed.ReduceOp.SUM, group=tp_group)

        log_probs = target_shifted - torch.log(sum_exp.squeeze(-1))
        # Keep the saved softmax in fp32. The gradient w.r.t. the *target* logit is (1 - p_t),
        # and bf16 spacing just below 1.0 is 2^-8 = 0.0039. Downcasting here would mean:
        #   p_t = 0.99  -> true grad 0.01, absolute error up to 0.002  => ~20% error
        #   p_t = 0.999 -> bf16(0.999) rounds to exactly 1.0           => gradient exactly 0
        # An RL-finetuned policy is confident on most tokens, so this silently destroys the
        # majority of the gradient signal while low-confidence tokens stay accurate -- a bias,
        # not just noise. This is the most likely cause of the non-finite grad_norm previously
        # observed under tensor_parallel_size > 1.
        softmax_local = exp_logits / sum_exp
        ctx.save_for_backward(softmax_local, local_labels, target_mask)
        ctx.temperature = float(temperature)
        ctx.orig_dtype = orig_dtype
        return log_probs.to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        softmax_local, local_labels, target_mask = ctx.saved_tensors
        # softmax_local is fp32 (see forward); do the whole (1 - p_t) construction in fp32 and
        # only cast back to the parameter dtype at the very end.
        grad_logits = -softmax_local.clone()
        one_hot_update = target_mask.to(grad_logits.dtype).unsqueeze(-1)
        grad_logits.scatter_add_(-1, local_labels.unsqueeze(-1), one_hot_update)
        scale = grad_output.unsqueeze(-1).to(grad_logits.dtype)
        if ctx.temperature != 1.0:
            scale = scale / ctx.temperature
        grad_logits.mul_(scale)
        return grad_logits.to(ctx.orig_dtype), None, None, None, None


def vocab_parallel_logprobs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    tp_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """Computes token logprobs directly from a Shard(-1) DTensor on TPU without calling full_tensor()."""
    from torch.distributed.tensor import DTensor

    if isinstance(logits, DTensor):
        if tp_group is None:
            tp_mesh = logits.device_mesh
            tp_group = tp_mesh.get_group()
        tp_rank = torch.distributed.get_rank(tp_group)
        local_logits = logits.to_local()
    else:
        assert tp_group is not None
        tp_rank = torch.distributed.get_rank(tp_group)
        local_logits = logits
    return _VocabParallelLogprobsFn.apply(local_logits, labels, temperature, tp_group, tp_rank)
