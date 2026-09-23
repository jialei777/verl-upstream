# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.device import get_device_name
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
import os
from typing import Any
from verl.utils.device import get_device_id
_DEFAULT_TPU_SEQ_BUCKET_SIZE = 256
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

from verl.workers.utils.padding import no_padding_2_padding


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = tu.get_non_tensor_data(data=data, key="dp_size", default=1)
    batch_num_tokens = tu.get_non_tensor_data(data=data, key="batch_num_tokens", default=None)

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = getattr(log_prob, "_tpu_padded_values", None)
        if log_prob_flatten is not None:
            loss_mask_flatten = torch.roll(loss_mask.values().detach().cpu(), shifts=-1, dims=0)
            pad_len = int(log_prob_flatten.shape[0]) - int(loss_mask_flatten.shape[0])
            if pad_len > 0:
                loss_mask_flatten = torch.nn.functional.pad(loss_mask_flatten, (0, pad_len), value=0)
            loss_mask_flatten = loss_mask_flatten.to(device=log_prob_flatten.device)
        else:
            log_prob_flatten = log_prob.values()
            loss_mask_flatten = loss_mask.values()
            # left-shift the loss mask by one token to align with log_prob
            loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    pad_fn = tpu_no_padding_2_padding if get_device_name() == "tpu" else no_padding_2_padding
    log_prob = pad_fn(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = pad_fn(entropy, data)

    # global batch info for loss aggregation
    dp_size = tu.get_non_tensor_data(data=data, key="dp_size", default=1)
    batch_num_tokens = tu.get_non_tensor_data(data=data, key="batch_num_tokens", default=None)
    config.global_batch_info["dp_size"] = dp_size
    config.global_batch_info["batch_num_tokens"] = batch_num_tokens
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        dp_size > 1
        or batch_num_tokens is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    data = (
        select_and_to_padded_tensor(data, *fields)
        if get_device_name() == "tpu"
        else data.select(*fields).to_padded_tensor()
    )

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    pad_fn = tpu_no_padding_2_padding if get_device_name() == "tpu" else no_padding_2_padding
    vpreds = pad_fn(model_output["values"], data)  # (bsz, response_length)

    # Normalize the value loss over the global mini-batch (dp_size / batch_num_tokens /
    # global_batch_size) instead of the local micro-batch, so the accumulated critic gradient is
    # invariant to how the mini-batch is split into micro-batches (as the actor's ppo_loss does).
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]
    global_batch_size = data["global_batch_size"]

    # When the loss is normalized over the global batch, each micro-batch contributes a partial sum,
    # so the loss metric must be aggregated with SUM to reflect the global-batch mean.
    if (
        dp_size > 1
        or batch_num_tokens is not None
        or global_batch_size is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    # select fields and convert to padded tensor
    fields = ("values", "returns", "response_mask")
    data = (
        select_and_to_padded_tensor(data, *fields)
        if get_device_name() == "tpu"
        else data.select(*fields).to_padded_tensor()
    )

    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=global_batch_size,
        loss_scale_factor=config.loss_scale_factor,
    )

    metrics = {
        "critic/vf_loss": Metric(value=vf_loss, aggregation=metric_aggregation),
        "critic/vf_clipfrac": vf_clipfrac.detach().item(),
        "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
    }

    return vf_loss, metrics
