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
"""Helpers for running the training path on Google TPU (``torch_tpu``).

TPU execution goes through PJRT/XLA: every distinct tensor shape a step sees makes XLA trace and
compile a new executable, and the compiled binaries are cached in host RAM for the whole run. A
variable-length RL workload therefore has to be *bucketed* - padded to a small set of static shapes
- or the job spends its time compiling and eventually exhausts host memory.

The helpers here implement that bucketing plus a few host-side substitutes for operations that are
either unimplemented or pathologically slow on the TPU backend. They are plain PyTorch and import
nothing from ``torch_tpu``, so this module is importable (and unit-testable) on any platform.
"""

import logging
import os
from typing import Any, Optional

import torch
import torch.distributed
from tensordict import TensorDict

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Attribute used to carry the bucket-padded, still-on-device copy of a value alongside the
# unpadded nested tensor that the rest of verl consumes. See ``TorchTitanEngineWithLMHead``.
TPU_PADDED_VALUES_ATTR = "_tpu_padded_values"

DEFAULT_SEQ_BUCKET_SIZE = 256


def get_tpu_seq_bucket_size() -> int:
    """Return the token bucket multiple used when padding sequences for TPU.

    Override with ``VERL_TPU_SEQ_BUCKET_SIZE``. A larger bucket wastes more compute per step but
    compiles fewer graphs; a smaller one does the opposite.
    """
    return int(os.getenv("VERL_TPU_SEQ_BUCKET_SIZE", str(DEFAULT_SEQ_BUCKET_SIZE)))


def bucket_length(length: int, bucket_size: Optional[int] = None) -> int:
    """Round ``length`` up to the next multiple of ``bucket_size`` (at least one bucket)."""
    if bucket_size is None:
        bucket_size = get_tpu_seq_bucket_size()
    if bucket_size <= 1:
        return max(1, int(length))
    return max(bucket_size, ((int(length) + bucket_size - 1) // bucket_size) * bucket_size)


def unwrap_metadata(value: Any) -> Any:
    """Normalize metadata read back from a TensorDict to a plain python value.

    Non-tensor metadata that travelled through Ray and ``TensorDict`` can come back wrapped in a
    per-rank list or as a singleton tensor; the engine only ever wants the scalar behind it.
    """
    if isinstance(value, list):
        return unwrap_metadata(value[0]) if value else None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        if value.numel() > 1:
            return value.flatten()[0].item()
    return value


def synchronize_tpu_loss(loss: torch.Tensor) -> None:
    """Materialize the forward graph before the backward pass is traced.

    Without this the forward and backward passes are traced into a single huge XLA graph, which
    compiles slowly and peaks host memory during compilation. ``wait=False`` only cuts the graph,
    it does not block the host on the result.
    """
    try:
        from torch_tpu._internal.sync import synchronize
    except ImportError:
        return
    synchronize(loss, wait=False)


def compute_global_batch_num_tokens(data: TensorDict, dp_group, tp_size: int) -> float:
    """Return the number of loss tokens in the global batch, reduced on the host.

    The device-side equivalent adds a collective to the traced graph on every step, which has to
    stay in lockstep with every other rank's graph. Reducing the (single scalar) count on the host
    keeps the traced graph free of data-dependent collectives.
    """
    batch_num_tokens = data["loss_mask"].sum().cpu()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(batch_num_tokens, op=torch.distributed.ReduceOp.SUM)
        # the reduction above runs over the whole world, undo the tensor-parallel duplication
        batch_num_tokens = batch_num_tokens / tp_size
    return batch_num_tokens.item()


def safe_to_padded_tensor(nested: Any, padding: Any = 0, output_size: Optional[tuple] = None) -> torch.Tensor:
    """Bucket-pad a nested tensor into a dense tensor, assembling the result on the host.

    ``torch.nested.to_padded_tensor`` compiles a new jagged-to-dense kernel for every distinct
    jagged length. The tensors this is used for (per-response log probs, masks, ...) are small, so
    building them on the host and transferring once is both cheaper and shape-stable.
    """
    from verl.utils.device import get_device_id

    if not getattr(nested, "is_nested", False):
        if isinstance(nested, torch.Tensor) and nested.device.type == "cpu":
            return nested.to(device=get_device_id())
        return nested

    target_device = get_device_id() if nested.device.type == "cpu" else nested.device
    values_cpu = nested.values().detach().cpu()
    offsets_cpu = nested.offsets().detach().cpu()
    batch_size = int(offsets_cpu.shape[0]) - 1
    if batch_size <= 0:
        return torch.empty(output_size if output_size is not None else (0,), device=target_device, dtype=nested.dtype)

    lengths = offsets_cpu.diff().tolist()
    if output_size is None:
        output_size = (batch_size, bucket_length(max(lengths)), *tuple(values_cpu.shape[1:]))
    out_cpu = torch.full(output_size, padding, dtype=values_cpu.dtype)
    for i in range(batch_size):
        start = int(offsets_cpu[i].item())
        length = int(lengths[i])
        if length > 0:
            out_cpu[i, :length] = values_cpu[start : start + length]
    return out_cpu.to(device=target_device)


def select_and_to_padded_tensor(data: TensorDict, *fields: str) -> TensorDict:
    """TPU replacement for ``data.select(*fields).to_padded_tensor()``.

    Nested fields are padded to the batch's bucketed response length so the loss always sees the
    same shapes; dense fields that are still on the host are moved to the device.
    """
    from verl.utils import tensordict_utils as tu
    from verl.utils.device import get_device_id

    max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)
    max_response_len = bucket_length(int(max_response_len)) if int(max_response_len or -1) > 0 else None

    target_device = get_device_id()
    padded_dict = {}
    for key in fields:
        if key not in data.keys():
            continue
        value = data[key]
        if getattr(value, "is_nested", False):
            output_size = None
            if max_response_len is not None:
                trailing_dims = tuple(value.values().shape[1:])
                output_size = (int(data.batch_size[0]), max_response_len, *trailing_dims)
            padded_dict[key] = safe_to_padded_tensor(value, output_size=output_size)
        elif isinstance(value, torch.Tensor) and value.device.type == "cpu":
            padded_dict[key] = value.to(device=target_device)
        else:
            padded_dict[key] = value
    return TensorDict(padded_dict, batch_size=data.batch_size)


def tpu_no_padding_2_padding(tensor: torch.Tensor, data: TensorDict) -> torch.Tensor:
    """TPU replacement for :func:`verl.workers.utils.padding.no_padding_2_padding`.

    The engine keeps the bucket-padded model output next to the unpadded nested tensor (see
    ``TPU_PADDED_VALUES_ATTR``). Slicing the response part out of it with a static-shape
    ``gather`` keeps the graph shape-stable, where indexing per sequence would not.
    """
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import no_padding_2_padding

    padded_values = getattr(tensor, TPU_PADDED_VALUES_ATTR, None)
    if padded_values is None:
        return no_padding_2_padding(tensor, data)

    prompt_ids, response_ids = data["prompts"], data["responses"]
    if not (getattr(prompt_ids, "is_nested", False) and getattr(response_ids, "is_nested", False)):
        return no_padding_2_padding(tensor, data)

    prompt_lens = prompt_ids.offsets().diff().cpu()
    response_lens = response_ids.offsets().diff().cpu()
    seq_offsets = (prompt_lens + response_lens).cumsum(dim=0)

    max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)
    if int(max_response_len or -1) < 0:
        max_response_len = bucket_length(int(response_lens.max().item()))
    else:
        max_response_len = bucket_length(int(max_response_len))
        tu.assign_non_tensor_data(data, "max_response_len", max_response_len)

    bsz = int(response_lens.shape[0])
    col_idx = torch.arange(max_response_len, dtype=torch.int64).unsqueeze(0)  # [1, max_response_len]
    # log prob i is the probability of token i + 1, so the response starts one token early
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
