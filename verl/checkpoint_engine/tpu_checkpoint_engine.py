# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import asyncio
import logging
import re
import time
from typing import Any, Generator

import ray
import torch
from torch.distributed.tensor import DTensor

from verl.checkpoint_engine.base import (
    CheckpointEngine,
    CheckpointEngineRegistry,
)

from .tpu_weight_registry import TPUWeightRegistry

logger = logging.getLogger(__name__)

# --- GLOBAL CONFIGURATION / CONSTANTS FOR WEIGHT TRANSFER ---
SYNC_LAYER_BY_LAYER = False  # Qwen3-0.6B is small, whole-model sync is fast and safe
TPU_COPY_CHUNK_SIZE_PARAMETERS = 30

# Must stay in sync with verl/workers/rollout/vllm_rollout/tpu_utils.py, which
# looks the same detached actor up from the rollout side.
TPU_WEIGHT_REGISTRY_ACTOR_NAME = "TPUWeightRegistry_v3"
TPU_WEIGHT_REGISTRY_NAMESPACE = "verl"

# vLLM stores q/k/v and gate/up as single fused parameters, while TorchTitan
# exports them under their original HuggingFace names. Maps the fused parameter
# to its source projections, in the order vLLM concatenates them.
_FUSED_PROJECTIONS = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}

# =====================================================================
# Namespace & Formatting Utilities
# =====================================================================


def get_clean_name(name: str) -> str:
    """Strip FSDP/DCP wrapper prefixes from state dict keys to match standard model namespaces."""
    return name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "").replace("module.", "")


def get_layer_group(key: str) -> str:
    """Given a state dict key, returns its group name (e.g. 'embeddings', 'layers.0', 'output')."""
    clean_k = get_clean_name(key)
    match = re.search(r"layers\.(\d+)\.", clean_k)
    if match:
        return f"layers.{match.group(1)}"
    elif "tok_embeddings" in clean_k:
        return "embeddings"
    else:
        return "output"


# =====================================================================
# TPU Worker Weight Injection & Slicing
# =====================================================================


def load_weights_on_worker(vllm_model, state_dict: dict, rank: int) -> int:
    """Worker-side weight loader. Performs host-side CPU sharding (slicing)
    and chunked, memory-safe, JIT-partitioned PCIe copying to TPU.
    """
    if state_dict is None:
        return 0

    t_start = time.perf_counter()

    if isinstance(state_dict, dict) and "grouped" in state_dict:
        grouped_dict = state_dict["grouped"]
    else:
        grouped_dict = {"all": state_dict}

    total_keys = 0
    from concurrent.futures import ThreadPoolExecutor

    temp_tpu_tensors = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for group_name, group_sd in grouped_dict.items():
            keys_loaded = _load_single_group_on_worker(
                vllm_model, group_sd, rank, executor=executor, temp_tpu_tensors=temp_tpu_tensors
            )
            total_keys += keys_loaded

    import torch_tpu

    torch_tpu._internal.sync.synchronize(wait=True)
    del temp_tpu_tensors
    import gc

    gc.collect()

    t_total = time.perf_counter() - t_start
    if rank == 0:
        logger.info(f"Worker 0: Loaded {total_keys} keys in {t_total:.3f}s")
    return total_keys


def _load_single_group_on_worker(
    vllm_model,
    group_sd: dict,
    rank: int,
    executor=None,
    temp_tpu_tensors=None,
    skipped_keys=None,
    written_keys=None,
) -> int:
    if skipped_keys is None:
        skipped_keys = []
    if written_keys is None:
        written_keys = set()

    flat_tensors = group_sd["flat_tensors"]
    metadata = group_sd["metadata"]

    clean_metadata = {}
    num_keys = 0
    for dtype, items in metadata.items():
        clean_items = []
        offset = 0
        for k, shape, numel in items:
            clean_k = get_clean_name(k)
            clean_items.append((clean_k, shape, numel, offset))
            num_keys += 1

            if "tok_embeddings.weight" in clean_k:
                lm_k = clean_k.replace("tok_embeddings", "lm_head")
                clean_items.append((lm_k, shape, numel, offset))
                num_keys += 1

            offset += numel
        clean_metadata[dtype] = clean_items

    model_sd = vllm_model.state_dict() if hasattr(vllm_model, "state_dict") else vllm_model.model.state_dict()
    module_dict = dict(vllm_model.named_modules()) if hasattr(vllm_model, "named_modules") else {}

    def resolve_key(k):
        if k in model_sd:
            return k
        if k.startswith("model.") and k[6:] in model_sd:
            return k[6:]
        if f"model.{k}" in model_sd:
            return f"model.{k}"
        return k

    def get_parent_module(target_key):
        parent_name = target_key.rsplit(".", 1)[0] if "." in target_key else ""
        if parent_name in module_dict:
            return module_dict[parent_name]
        if parent_name.startswith("model.") and parent_name[6:] in module_dict:
            return module_dict[parent_name[6:]]
        if f"model.{parent_name}" in module_dict:
            return module_dict[f"model.{parent_name}"]
        return None

    for dtype, flat_data in flat_tensors.items():
        items = clean_metadata.get(dtype, [])
        if not items:
            continue

        flat_cpu = torch.from_numpy(flat_data) if not isinstance(flat_data, torch.Tensor) else flat_data
        if dtype == torch.bfloat16 and flat_cpu.dtype == torch.int16:
            # bfloat16 is serialized via int16 numpy buffers (identical 16-bit representation)
            # and viewed back as bfloat16 here.
            flat_cpu = flat_cpu.view(torch.bfloat16)

        raw_tensors = {}
        dedup_items = []
        seen_clean_keys = set()
        for item in items:
            k, shape, numel, offset = item
            raw_tensors[k] = flat_cpu[offset : offset + numel].view(shape)
            if k not in seen_clean_keys:
                seen_clean_keys.add(k)
                dedup_items.append(item)

        local_items = []
        local_tensors_to_cat = []
        local_offset = 0

        def to_target_layout(tensor, target_local, flipped):
            """Match vllm-torchtpu's (n_in, n_out) weight layout when required."""
            if tensor.ndim == 2 and (
                flipped or (tensor.shape != target_local.shape and tensor.T.shape == target_local.shape)
            ):
                return tensor.transpose(0, 1).contiguous()
            return tensor.contiguous()

        def build_fused(fused_key, parts):
            """Shard each source projection for this rank, then concatenate."""
            target_v = model_sd[fused_key]
            target_local = target_v.to_local() if isinstance(target_v, DTensor) else target_v
            module = get_parent_module(fused_key)
            flipped = bool(getattr(module, "_tpu_weight_flipped", False))

            out_dim = target_local.shape[1] if (flipped and target_local.ndim == 2) else target_local.shape[0]
            tp_size = getattr(module, "tp_size", max(1, sum(p.shape[0] for p in parts) // out_dim))
            # Grouped-query attention replicates KV heads when tp_size exceeds the
            # KV head count, so k/v span fewer shards than q. For gate_up_proj
            # num_kv_head_replicas is absent and this collapses to plain sharding.
            kv_replicas = getattr(module, "num_kv_head_replicas", 1)
            kv_tp, kv_rank = max(1, tp_size // kv_replicas), rank // kv_replicas

            shards = []
            for i, part in enumerate(parts):
                n_shards, shard_rank = (tp_size, rank) if i == 0 else (kv_tp, kv_rank)
                size = part.shape[0] // n_shards
                shards.append(part[shard_rank * size : (shard_rank + 1) * size])

            fused = to_target_layout(torch.cat(shards, dim=0), target_local, flipped)
            return (fused_key, target_local.shape, target_local.numel(), fused.reshape(-1))

        def process_item_parallel(item, raw_tensors=raw_tensors):
            k, shape, numel, offset = item
            target_key = resolve_key(k)

            if target_key not in model_sd:
                # vLLM fuses q/k/v and gate/up into single parameters while
                # TorchTitan exports them separately. The group is emitted once,
                # by its first source; the rest resolve to nothing.
                for suffix in (".weight", ".bias"):
                    if not k.endswith(suffix):
                        continue
                    base = k[: -len(suffix)]
                    for fused_name, sources in _FUSED_PROJECTIONS.items():
                        if not base.endswith(sources[0]):
                            continue
                        prefix = base[: -len(sources[0])]
                        fused_key = resolve_key(f"{prefix}{fused_name}{suffix}")
                        parts = [raw_tensors.get(f"{prefix}{s}{suffix}") for s in sources]
                        if fused_key not in model_sd or any(p is None for p in parts):
                            return None
                        return build_fused(fused_key, parts)
                return None

            target_v = model_sd[target_key]
            target_local = target_v.to_local() if isinstance(target_v, DTensor) else target_v
            parent_mod = get_parent_module(target_key)
            is_flipped = bool(getattr(parent_mod, "_tpu_weight_flipped", False))

            param_cpu_global = raw_tensors[k]
            if param_cpu_global.ndim == 2 and is_flipped:
                param_cpu_global = param_cpu_global.transpose(0, 1)

            eff_shape = param_cpu_global.shape
            if target_local.shape == eff_shape:
                param_cpu_local = param_cpu_global.contiguous()
            else:
                sharded = False
                for dim in range(len(eff_shape)):
                    if eff_shape[dim] != target_local.shape[dim]:
                        shard_size = target_local.shape[dim]
                        rank_offset = shard_size * rank
                        indices = [slice(None)] * len(eff_shape)
                        indices[dim] = slice(rank_offset, rank_offset + shard_size)
                        param_cpu_local = param_cpu_global[tuple(indices)].contiguous()
                        sharded = True
                        break
                if not sharded:
                    param_cpu_local = param_cpu_global.contiguous()

            return (target_key, target_local.shape, target_local.numel(), param_cpu_local.reshape(-1))

        if executor is None:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=8) as local_exec:
                sliced_results = list(local_exec.map(process_item_parallel, dedup_items))
        else:
            sliced_results = list(executor.map(process_item_parallel, dedup_items))

        for item, res in zip(dedup_items, sliced_results, strict=False):
            if res is None:
                skipped_keys.append(item[0])
                continue
            target_key, target_shape, target_numel, param_cpu_local_flat = res
            local_tensors_to_cat.append(param_cpu_local_flat)
            local_items.append((target_key, target_shape, target_numel, local_offset))
            local_offset += target_numel

        if not local_tensors_to_cat:
            continue

        flat_local_cpu = torch.cat(local_tensors_to_cat)

        chunks = [
            local_items[i : i + TPU_COPY_CHUNK_SIZE_PARAMETERS]
            for i in range(0, len(local_items), TPU_COPY_CHUNK_SIZE_PARAMETERS)
        ]

        for chunk in chunks:
            chunk_start_offset = chunk[0][3]
            chunk_end_offset = chunk[-1][3] + chunk[-1][2]
            flat_chunk_cpu = flat_local_cpu[chunk_start_offset:chunk_end_offset]

            flat_chunk_tpu = flat_chunk_cpu.to("tpu")
            if temp_tpu_tensors is not None:
                temp_tpu_tensors.append(flat_chunk_tpu)

            for target_key, local_shape, local_numel, offset in chunk:
                local_offset = offset - chunk_start_offset
                slice_tpu = flat_chunk_tpu[local_offset : local_offset + local_numel].view(local_shape)

                target_v = model_sd[target_key]
                target_local = target_v.to_local() if isinstance(target_v, DTensor) else target_v
                target_local.copy_(slice_tpu)
                written_keys.add(target_key)

            if temp_tpu_tensors is None:
                del flat_chunk_tpu
                import torch_tpu

                torch_tpu._internal.sync.synchronize(wait=True)

    return num_keys


class TPUStreamingWeightLoader:
    """Streaming worker-side weight loader for vLLM TPU workers.

    Accepts `(name, cpu_tensor)` batches from `BucketedWeightReceiver(use_shm=True)`,
    performs parallel CPU TP-sharding, fuses `(q_proj, k_proj, v_proj) -> qkv_proj`
    and `(gate_proj, up_proj) -> gate_up_proj` (buffering any partial fused group
    across bucket boundaries), transposes 2D weights when `_tpu_weight_flipped=True`,
    and copies to TPU HBM in 30-parameter chunks.
    """

    def __init__(self, vllm_model, rank: int) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self.vllm_model = vllm_model
        self.rank = rank
        self.model_sd = vllm_model.state_dict() if hasattr(vllm_model, "state_dict") else vllm_model.model.state_dict()
        self.module_dict = dict(vllm_model.named_modules()) if hasattr(vllm_model, "named_modules") else {}
        self.pending_raw_tensors: dict[str, torch.Tensor] = {}
        self.temp_tpu_tensors: list[torch.Tensor] = []
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.total_keys = 0
        self.t_start = time.perf_counter()

    def _resolve_key(self, k: str) -> str:
        if k in self.model_sd:
            return k
        if k.startswith("model.") and k[6:] in self.model_sd:
            return k[6:]
        if f"model.{k}" in self.model_sd:
            return f"model.{k}"
        return k

    def _get_parent_module(self, target_key: str):
        parent_name = target_key.rsplit(".", 1)[0] if "." in target_key else ""
        if parent_name in self.module_dict:
            return self.module_dict[parent_name]
        if parent_name.startswith("model.") and parent_name[6:] in self.module_dict:
            return self.module_dict[parent_name[6:]]
        if f"model.{parent_name}" in self.module_dict:
            return self.module_dict[f"model.{parent_name}"]
        return None

    def load_bucket(self, weights: list[tuple[str, torch.Tensor]]) -> int:
        if not weights and not self.pending_raw_tensors:
            return 0

        raw_tensors: dict[str, torch.Tensor] = dict(self.pending_raw_tensors)
        dedup_keys: list[str] = list(self.pending_raw_tensors.keys())
        seen_keys: set[str] = set(dedup_keys)

        for k, tensor in weights:
            clean_k = get_clean_name(k)
            raw_tensors[clean_k] = tensor
            if clean_k not in seen_keys:
                seen_keys.add(clean_k)
                dedup_keys.append(clean_k)

            if "tok_embeddings.weight" in clean_k:
                lm_k = clean_k.replace("tok_embeddings", "lm_head")
                raw_tensors[lm_k] = tensor
                if lm_k not in seen_keys:
                    seen_keys.add(lm_k)
                    dedup_keys.append(lm_k)

        new_pending: dict[str, torch.Tensor] = {}

        def to_target_layout(tensor, target_local, flipped):
            if tensor.ndim == 2 and (
                flipped or (tensor.shape != target_local.shape and tensor.T.shape == target_local.shape)
            ):
                return tensor.transpose(0, 1).contiguous()
            return tensor.contiguous()

        def build_fused(fused_key, parts):
            target_v = self.model_sd[fused_key]
            target_local = target_v.to_local() if isinstance(target_v, DTensor) else target_v
            module = self._get_parent_module(fused_key)
            flipped = bool(getattr(module, "_tpu_weight_flipped", False))

            out_dim = target_local.shape[1] if (flipped and target_local.ndim == 2) else target_local.shape[0]
            tp_size = getattr(module, "tp_size", max(1, sum(p.shape[0] for p in parts) // out_dim))
            kv_replicas = getattr(module, "num_kv_head_replicas", 1)
            kv_tp, kv_rank = max(1, tp_size // kv_replicas), self.rank // kv_replicas

            shards = []
            for i, part in enumerate(parts):
                n_shards, shard_rank = (tp_size, self.rank) if i == 0 else (kv_tp, kv_rank)
                size = part.shape[0] // n_shards
                shards.append(part[shard_rank * size : (shard_rank + 1) * size])

            fused = to_target_layout(torch.cat(shards, dim=0), target_local, flipped)
            return (fused_key, target_local.shape, target_local.numel(), fused.reshape(-1), fused.dtype)

        # Identify any fused group whose source tensors are only partially present in this bucket
        ready_keys: list[str] = []
        for k in dedup_keys:
            target_key = self._resolve_key(k)
            if target_key in self.model_sd:
                ready_keys.append(k)
                continue

            matched_fused = False
            for suffix in (".weight", ".bias"):
                if not k.endswith(suffix):
                    continue
                base = k[: -len(suffix)]
                for fused_name, sources in _FUSED_PROJECTIONS.items():
                    for src_name in sources:
                        if base.endswith(src_name):
                            prefix = base[: -len(src_name)]
                            fused_key = self._resolve_key(f"{prefix}{fused_name}{suffix}")
                            if fused_key in self.model_sd:
                                matched_fused = True
                                parts = [raw_tensors.get(f"{prefix}{s}{suffix}") for s in sources]
                                if any(p is None for p in parts):
                                    # Keep a cloned copy on CPU for the next bucket
                                    if k not in new_pending:
                                        new_pending[k] = (
                                            raw_tensors[k] if k in self.pending_raw_tensors else raw_tensors[k].clone()
                                        )
                                elif src_name == sources[0]:
                                    ready_keys.append(k)
                            break
                    if matched_fused:
                        break

        self.pending_raw_tensors = new_pending

        def process_key(k: str):
            target_key = self._resolve_key(k)
            if target_key not in self.model_sd:
                for suffix in (".weight", ".bias"):
                    if not k.endswith(suffix):
                        continue
                    base = k[: -len(suffix)]
                    for fused_name, sources in _FUSED_PROJECTIONS.items():
                        if not base.endswith(sources[0]):
                            continue
                        prefix = base[: -len(sources[0])]
                        fused_key = self._resolve_key(f"{prefix}{fused_name}{suffix}")
                        parts = [raw_tensors.get(f"{prefix}{s}{suffix}") for s in sources]
                        if fused_key not in self.model_sd or any(p is None for p in parts):
                            return None
                        return build_fused(fused_key, parts)
                return None

            target_v = self.model_sd[target_key]
            target_local = target_v.to_local() if isinstance(target_v, DTensor) else target_v
            parent_mod = self._get_parent_module(target_key)
            is_flipped = bool(getattr(parent_mod, "_tpu_weight_flipped", False))

            param_cpu_global = raw_tensors[k]
            if param_cpu_global.ndim == 2 and is_flipped:
                param_cpu_global = param_cpu_global.transpose(0, 1)

            eff_shape = param_cpu_global.shape
            if target_local.shape == eff_shape:
                param_cpu_local = param_cpu_global.contiguous()
            else:
                sharded = False
                for dim in range(len(eff_shape)):
                    if eff_shape[dim] != target_local.shape[dim]:
                        shard_size = target_local.shape[dim]
                        rank_offset = shard_size * self.rank
                        indices = [slice(None)] * len(eff_shape)
                        indices[dim] = slice(rank_offset, rank_offset + shard_size)
                        param_cpu_local = param_cpu_global[tuple(indices)].contiguous()
                        sharded = True
                        break
                if not sharded:
                    param_cpu_local = param_cpu_global.contiguous()

            return (
                target_key,
                target_local.shape,
                target_local.numel(),
                param_cpu_local.reshape(-1),
                param_cpu_local.dtype,
            )

        if not ready_keys:
            raw_tensors.clear()
            return 0

        sliced_results = list(self.executor.map(process_key, ready_keys))
        raw_tensors.clear()

        by_dtype: dict[torch.dtype, list[tuple[str, torch.Size, int, torch.Tensor]]] = {}
        for res in sliced_results:
            if res is None:
                continue
            target_key, target_shape, target_numel, param_cpu_local_flat, dtype = res
            by_dtype.setdefault(dtype, []).append((target_key, target_shape, target_numel, param_cpu_local_flat))

        loaded_in_bucket = 0
        for _dtype, items in by_dtype.items():
            local_items = []
            local_tensors_to_cat = []
            local_offset = 0
            for target_key, target_shape, target_numel, param_cpu_local_flat in items:
                local_tensors_to_cat.append(param_cpu_local_flat)
                local_items.append((target_key, target_shape, target_numel, local_offset))
                local_offset += target_numel

            if not local_tensors_to_cat:
                continue

            flat_local_cpu = torch.cat(local_tensors_to_cat)
            chunks = [
                local_items[i : i + TPU_COPY_CHUNK_SIZE_PARAMETERS]
                for i in range(0, len(local_items), TPU_COPY_CHUNK_SIZE_PARAMETERS)
            ]

            for chunk in chunks:
                chunk_start_offset = chunk[0][3]
                chunk_end_offset = chunk[-1][3] + chunk[-1][2]
                flat_chunk_cpu = flat_local_cpu[chunk_start_offset:chunk_end_offset]

                flat_chunk_tpu = flat_chunk_cpu.to("tpu")
                self.temp_tpu_tensors.append(flat_chunk_tpu)

                for target_key, local_shape, local_numel, offset in chunk:
                    rel_offset = offset - chunk_start_offset
                    slice_tpu = flat_chunk_tpu[rel_offset : rel_offset + local_numel].view(local_shape)

                    target_v = self.model_sd[target_key]
                    target_local = target_v.to_local() if isinstance(target_v, DTensor) else target_v
                    target_local.copy_(slice_tpu)
                    loaded_in_bucket += 1

        self.total_keys += loaded_in_bucket
        return loaded_in_bucket

    def finalize(self) -> None:
        import gc

        try:
            import torch_tpu

            torch_tpu._internal.sync.synchronize(wait=True)
        except Exception:
            pass
        self.temp_tpu_tensors.clear()
        self.pending_raw_tensors.clear()
        self.executor.shutdown(wait=False)
        gc.collect()
        t_total = time.perf_counter() - self.t_start
        if self.rank == 0:
            logger.info(f"TPUStreamingWeightLoader (rank 0): Loaded {self.total_keys} target params in {t_total:.3f}s")


# =====================================================================
# TPUCheckpointEngine Registration
# =====================================================================


@CheckpointEngineRegistry.register("tpu")
class TPUCheckpointEngine(CheckpointEngine):
    """Unified bucket-streaming CheckpointEngine for Cloud TPU.

    Aligns TPU weight synchronization with `NCCLCheckpointEngine`:
    - `CheckpointEngineManager.update_weights()` builds the topology between
      Trainer Rank 0 (`rank == 0`) and rollout `CheckpointEngineWorker` sidecars
      (`rank == 1..M`, running on CPU with `use_gpu=False`).
    - Trainer Rank 0 packs weights into double-buffered `bucket_size` (512 MiB)
      `uint8` buckets using `split_weight_chunks`, places each bucket into Ray Plasma
      (so Ray's C++ Object Manager transfers it once per rollout VM into `/dev/shm`),
      and coordinates streaming + flow control over ZeroMQ `PUB/SUB` and `PUSH/PULL`.
    - Each rollout `CheckpointEngineWorker` maps the bucket zero-copy from local
      `/dev/shm` Plasma, reconstructs tensors via `merge_weight_chunks`, and feeds
      `vLLMServerAdapter` -> `BucketedWeightSender(use_shm=True)` -> `TPUWorker`
      (`TPUStreamingWeightLoader`).
    """

    def __init__(self, bucket_size: int = 512 << 20, is_master: bool = False, **kwargs) -> None:
        self.is_master = is_master
        self.bucket_size = min(bucket_size, 512 << 20) if bucket_size > 0 else (512 << 20)
        self.rank: int | None = None
        self.world_size: int | None = None
        self._sync_round: int = 0

        # Connect to or create named Ray TPUWeightRegistry actor
        try:
            self.registry = ray.get_actor(TPU_WEIGHT_REGISTRY_ACTOR_NAME, namespace=TPU_WEIGHT_REGISTRY_NAMESPACE)
            if not hasattr(self.registry, "wait_bucket_acks"):
                ray.kill(self.registry)
                raise ValueError("Stale TPUWeightRegistry without wait_bucket_acks")
        except ValueError:
            try:
                self.registry = TPUWeightRegistry.options(
                    name=TPU_WEIGHT_REGISTRY_ACTOR_NAME,
                    namespace=TPU_WEIGHT_REGISTRY_NAMESPACE,
                    lifetime="detached",
                ).remote()
            except Exception:
                self.registry = ray.get_actor(TPU_WEIGHT_REGISTRY_ACTOR_NAME, namespace=TPU_WEIGHT_REGISTRY_NAMESPACE)

        if self.is_master:
            try:
                ray.get(self.registry.clear.remote())
            except Exception as e:
                logger.warning(f"Could not reset TPUWeightRegistry left over from a previous job: {e}")

    def prepare(self) -> dict[str, Any]:
        if not self.is_master:
            return {"is_master": False}
        self._sync_round += 1
        return {"is_master": True, "sync_round": self._sync_round}

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]):
        master_meta = metadata[0]
        for m in metadata[:actor_wg_world_size]:
            if isinstance(m, dict) and m.get("is_master"):
                master_meta = m
                break

        world_size = rollout_world_size + 1
        actor_wg_kwargs = {
            "rank": [0] + [None] * (actor_wg_world_size - 1),
            "world_size": [world_size] + [None] * (actor_wg_world_size - 1),
            "master_metadata": [master_meta] * actor_wg_world_size,
        }
        rollout_kwargs = {
            "rank": list(range(1, world_size)),
            "world_size": [world_size] * rollout_world_size,
            "master_metadata": [master_meta] * rollout_world_size,
        }
        return actor_wg_kwargs, rollout_kwargs

    def init_process_group(
        self,
        rank: int | None = None,
        world_size: int | None = None,
        master_metadata: dict | None = None,
        **kwargs,
    ):
        self.rank = rank
        self.world_size = world_size
        self.master_metadata = master_metadata
        if isinstance(master_metadata, dict) and "sync_round" in master_metadata:
            self._sync_round = int(master_metadata["sync_round"])

    def finalize(self):
        self.rank = None
        self.world_size = None

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        import gc

        from verl.checkpoint_engine.base import TensorMeta, split_weight_chunks

        t_start = time.perf_counter()

        try:
            import torch_tpu

            torch_tpu._internal.sync.synchronize(wait=True)
        except Exception:
            pass

        if not self.is_master:
            for i, (_k, v) in enumerate(weights):
                del v
                if (i + 1) % 16 == 0:
                    try:
                        import torch_tpu

                        torch_tpu._internal.sync.synchronize(wait=True)
                    except Exception:
                        pass
            try:
                import torch_tpu

                torch_tpu._internal.sync.synchronize(wait=True)
            except Exception:
                pass
            gc.collect()
            return

        # Fallback to legacy registry upload if build_process_group was not called
        if not self.world_size or self.world_size <= 1:
            await self._send_weights_legacy(weights, global_steps=global_steps)
            return

        # Extract all FSDP weights in lockstep with trainer ranks 1..7 so TPU
        # collective all_gather_into_tensor completes without pausing mid-model.
        cpu_weights: list[tuple[str, torch.Tensor]] = []
        for i, (k, v) in enumerate(weights):
            cpu_v = v.detach().cpu().contiguous()
            del v
            cpu_weights.append((k, cpu_v))
            if (i + 1) % 16 == 0:
                try:
                    import torch_tpu

                    torch_tpu._internal.sync.synchronize(wait=True)
                except Exception:
                    pass
        try:
            import torch_tpu

            torch_tpu._internal.sync.synchronize(wait=True)
        except Exception:
            pass

        num_receivers = self.world_size - 1
        step_key = self._sync_round if self._sync_round > 0 else (global_steps if global_steps is not None else 0)
        logger.info(
            f"TPUCheckpointEngine: [Step {step_key}] Starting Ray Plasma bucketed stream "
            f"(bucket_size={self.bucket_size >> 20}MB, receivers={num_receivers}, tensors={len(cpu_weights)})..."
        )

        send_buf = torch.empty(self.bucket_size, dtype=torch.uint8, device="cpu")
        recv_buf = torch.empty(self.bucket_size, dtype=torch.uint8, device="cpu")

        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0
        bucket_idx = 0
        prev_bucket_idx: int | None = None

        async for tensor_meta, chunk in split_weight_chunks(cpu_weights, self.bucket_size):
            if offset + tensor_meta.chunk_size > self.bucket_size:
                if prev_bucket_idx is not None:
                    await self.registry.wait_bucket_acks.remote(step_key, prev_bucket_idx)

                ref = ray.put(send_buf[:offset].numpy())
                await self.registry.set_bucket.remote(step_key, bucket_idx, [ref], bucket_meta, False, num_receivers)
                del ref

                prev_bucket_idx = bucket_idx
                bucket_idx += 1
                send_buf, recv_buf = recv_buf, send_buf
                bucket_meta = {}
                offset = 0

            tensor_meta.offset = offset
            bucket_meta[tensor_meta.name] = tensor_meta
            send_buf[offset : offset + tensor_meta.chunk_size].copy_(chunk)
            offset += tensor_meta.chunk_size

        # Send final bucket
        if prev_bucket_idx is not None:
            await self.registry.wait_bucket_acks.remote(step_key, prev_bucket_idx)

        ref = ray.put(send_buf[:offset].numpy())
        await self.registry.set_bucket.remote(step_key, bucket_idx, [ref], bucket_meta, True, num_receivers)
        del ref

        await self.registry.wait_bucket_acks.remote(step_key, bucket_idx)

        del cpu_weights, send_buf, recv_buf
        gc.collect()

        t_total = time.perf_counter() - t_start
        logger.info(f"TPUCheckpointEngine: [Step {step_key}] Streamed {bucket_idx + 1} buckets in {t_total:.3f}s")

    async def _send_weights_legacy(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        import gc

        step_key = global_steps if global_steps is not None else 0
        grouped_weights = {}
        for k, v in weights:
            cpu_v = v.detach().cpu()
            del v
            if "layers." in k:
                parts = k.split(".")
                idx = parts.index("layers")
                group_name = ".".join(parts[: idx + 2])
            else:
                group_name = "other"
            grouped_weights.setdefault(group_name, []).append((k, cpu_v))
        try:
            import torch_tpu

            torch_tpu._internal.sync.synchronize(wait=True)
        except Exception:
            pass

        grouped_dict = {}
        for group_name in list(grouped_weights.keys()):
            group_items = grouped_weights.pop(group_name)
            by_dtype = {}
            for k, cpu_v in group_items:
                by_dtype.setdefault(cpu_v.dtype, []).append((k, cpu_v))
            del group_items

            flat_tensors = {}
            metadata = {}
            for dtype, items in by_dtype.items():
                flat_cpu = torch.cat([v.view(-1) for _, v in items])
                if dtype == torch.bfloat16:
                    flat_tensors[dtype] = flat_cpu.view(torch.int16).numpy()
                else:
                    flat_tensors[dtype] = flat_cpu.numpy()
                metadata[dtype] = [(k, v.shape, v.numel()) for k, v in items]
                del items
            del by_dtype

            grouped_dict[group_name] = {"flat_tensors": flat_tensors, "metadata": metadata}

        state_dict = {"grouped": grouped_dict}
        ref = ray.put(state_dict)
        del grouped_dict, state_dict
        await self.registry.set_weights.remote(step_key, [ref])
        del ref
        gc.collect()

    @torch.no_grad()
    async def receive_weights(
        self,
        global_steps: int | None = None,
    ):
        from verl.checkpoint_engine.base import merge_weight_chunks

        async for name, weight in merge_weight_chunks(
            self._receive_weight_chunks(global_steps=global_steps), self.bucket_size
        ):
            yield name, weight

    async def _receive_weight_chunks(self, global_steps: int | None = None):
        assert self.rank is not None and self.rank > 0, f"Only rollout ranks > 0 receive weights, got {self.rank}"
        step_key = self._sync_round if self._sync_round > 0 else (global_steps if global_steps is not None else 0)

        bucket_idx = 0
        while True:
            bucket_entry = await self.registry.get_bucket.remote(step_key, bucket_idx)
            bucket_ref_list, bucket_meta, is_last = bucket_entry
            bucket_np = ray.get(bucket_ref_list[0])
            bucket_tensor = torch.from_numpy(bucket_np)

            for tensor_meta in bucket_meta.values():
                yield tensor_meta, bucket_tensor[tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size]

            del bucket_tensor, bucket_np, bucket_ref_list, bucket_entry
            await self.registry.ack_bucket.remote(step_key, bucket_idx)

            if is_last:
                break
            bucket_idx += 1


async def update_tpu_weights(manager, global_steps: int | None = None) -> dict:
    """Synchronize weights from actor worker group to rollout replicas on TPU."""
    import gc

    t_abort_start = time.perf_counter()
    if global_steps and global_steps > 0:
        try:
            await manager.abort_replicas()
        except Exception as e:
            logger.warning(f"Failed to abort replicas at step {global_steps}: {e}")
    t_abort = time.perf_counter() - t_abort_start

    t_total_start = time.perf_counter()
    # 1. Extract and upload weights on trainer side (Rank 0 sends to Ray Plasma)
    actor_refs = manager.actor_wg.update_weights(global_steps=global_steps, mode=manager.backend)
    if isinstance(actor_refs, list):
        ray.get(actor_refs)
    elif actor_refs is not None:
        ray.get(actor_refs)

    # 2. Call collective_rpc on all rollout replicas to load weights from the registry
    step_key = global_steps if global_steps is not None else 0

    # Fail loudly if the trainer side did not actually publish anything. Without
    # this check an empty registry degrades into the rollout silently reusing its
    # initial weights, so training "succeeds" while nothing is ever learned.
    registry = ray.get_actor(TPU_WEIGHT_REGISTRY_ACTOR_NAME, namespace=TPU_WEIGHT_REGISTRY_NAMESPACE)
    published = ray.get(registry.get_weights.remote(step_key))
    if published is None:
        raise RuntimeError(
            f"TPU weight sync failed: no weights published under step_key={step_key}. "
            "The trainer-side send_weights never reached the registry. Re-run with "
            "debug logging enabled on verl.checkpoint_engine to confirm which rank "
            "was elected is_master."
        )
    del published

    futures = [
        replica.server_handle.collective_rpc.remote(method="load_weights_from_ray_registry", args=(step_key,))
        for replica in manager.replicas
    ]
    results = await asyncio.gather(*futures)

    # Release the state_dict ObjectRef from TPUWeightRegistry immediately after all
    # rollout replicas have loaded the weights so it does not stay pinned in the
    # actor node's Ray Object Store during the subsequent training step.
    try:
        await registry.clear.remote()
    except Exception:
        pass
    gc.collect()

    t_total = time.perf_counter() - t_total_start

    flat_counts: list = []
    for replica_result in results:
        if isinstance(replica_result, list | tuple):
            flat_counts.extend(replica_result)
        elif replica_result is not None:
            flat_counts.append(replica_result)
    if flat_counts and not any(isinstance(n, int) and n > 0 for n in flat_counts):
        raise RuntimeError(
            f"TPU weight sync failed: no rollout worker loaded any tensor for "
            f"step_key={step_key} (per-worker key counts: {results})."
        )

    logger.info(f"TPU weight sync for step {global_steps} completed in {t_total + t_abort:.3f}s")

    await manager.resume_generation_replicas()
    return {}
