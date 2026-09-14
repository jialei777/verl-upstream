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

# Set by vllm_torchtpu's linear methods once they have transposed a dense
# weight out of vLLM's [n_out, n_in] layout into the (k, n) layout its matmuls
# want. The flip is not recoverable from the shape alone -- most projections
# are square -- so this attribute is the only reliable signal.
TPU_WEIGHT_FLIPPED_ATTR = "_tpu_weight_flipped"

# Keys the rollout model has no parameter for are skipped. That is legitimate
# for aliases we speculatively emit (``lm_head`` on a tied-embedding model),
# and a silent correctness bug for anything else, so each distinct shape of
# name is reported once.
_UNRESOLVED_LOGGED: set[str] = set()


def _resolve_in_state_dict(key: str, model_sd: dict) -> str | None:
    """Match a trainer key against the rollout state dict, modulo a 'model.' prefix."""
    if key in model_sd:
        return key
    if key.startswith("model.") and key[6:] in model_sd:
        return key[6:]
    if f"model.{key}" in model_sd:
        return f"model.{key}"
    return None


class VllmParameterLayout:
    """Describes how the live vLLM model stores parameters, versus HF.

    The trainer publishes tensors in HF's namespace and HF's layout. Neither
    survives into the rollout model unchanged, and both differences are
    invisible in the key alone:

    * **Fusion.** vLLM packs ``q/k/v_proj`` into a single ``qkv_proj`` and
      ``gate/up_proj`` into ``gate_up_proj``, concatenated along the output
      dimension *after* each part has been sharded across TP ranks. The HF
      names do not exist in the model at all, so an unaware loader drops those
      weights on the floor and the rollout policy silently stops tracking the
      trainer.
    * **Transposition.** ``vllm_torchtpu`` transposes every dense linear weight
      to ``[n_in, n_out]`` in ``process_weights_after_loading``, because
      ``(m, k) @ (k, n)`` is much cheaper on TPU than the ``(m, k) @ (n, k).T``
      that vLLM's stock layout forces. An unaware loader compares dimensions
      positionally, picks the wrong axis to shard, and produces a tensor of the
      wrong size.
    """

    def __init__(self, vllm_model, root: torch.nn.Module) -> None:
        # Names are collected against both roots because the state dict this
        # is queried with mixes the two: most parameters come from the inner
        # model, but anything hanging off the wrapper (``lm_head``) keeps its
        # unprefixed top-level name.
        self.flipped: set[str] = set()
        for scope, strip in ((root, False), (vllm_model, True)):
            named_modules = getattr(scope, "named_modules", None)
            if named_modules is None:
                continue
            for mod_name, mod in named_modules():
                if not getattr(mod, TPU_WEIGHT_FLIPPED_ATTR, False):
                    continue
                if strip and mod_name.startswith("model."):
                    mod_name = mod_name[6:]
                self.flipped.add(f"{mod_name}.weight" if mod_name else "weight")

        # {"q_proj": ("qkv_proj", 0, 3), "k_proj": ("qkv_proj", 1, 3), ...}
        self.packed: dict[str, tuple[str, int, int]] = {}
        mapping = getattr(vllm_model, "packed_modules_mapping", None)
        if not mapping:
            mapping = getattr(root, "packed_modules_mapping", None)
        for fused, parts in (mapping or {}).items():
            if not isinstance(parts, (list, tuple)):
                continue
            for index, part in enumerate(parts):
                self.packed[part] = (fused, index, len(parts))

    def is_flipped(self, target_key: str) -> bool:
        return target_key in self.flipped

    def resolve(self, key: str, model_sd: dict) -> tuple[str | None, int, int]:
        """Map a trainer key onto ``(target_key, part_index, num_parts)``.

        ``num_parts`` is 1 for an ordinary parameter and >1 when ``key`` is one
        constituent of a fused vLLM parameter.
        """
        direct = _resolve_in_state_dict(key, model_sd)
        if direct is not None:
            return direct, 0, 1

        parts = key.split(".")
        if len(parts) >= 2 and parts[-2] in self.packed:
            fused, index, num_parts = self.packed[parts[-2]]
            candidate = ".".join(parts[:-2] + [fused, parts[-1]])
            resolved = _resolve_in_state_dict(candidate, model_sd)
            if resolved is not None:
                return resolved, index, num_parts

        return None, 0, 1


def _log_unresolved(keys: list[str], rank: int) -> None:
    if rank != 0:
        return
    for key in keys:
        pattern = re.sub(r"\.\d+\.", ".*.", key)
        if pattern in _UNRESOLVED_LOGGED:
            continue
        _UNRESOLVED_LOGGED.add(pattern)
        logger.warning(
            "Weight sync: no rollout parameter matches '%s' (pattern '%s'), so it will not be "
            "updated. Unless this is a known alias, the rollout policy is now drifting from the "
            "trainer.",
            key,
            pattern,
        )


def load_weights_on_worker(vllm_model, state_dict: dict, rank: int, world_size: int | None = None) -> int:
    """Worker-side weight loader. Performs host-side CPU sharding (slicing)
    and chunked, memory-safe, JIT-partitioned PCIe copying to TPU.

    ``world_size`` is the rollout tensor-parallel degree. When supplied, the
    loader can verify that each incoming tensor really is the *global* weight
    rather than a shard the trainer forgot to gather; see
    ``_validate_incoming_shape``.
    """
    # HACK: Guard against state_dict=None when shared memory weight cache is skipped on multi-host vLLM workers.
    # TODO: remove HACK once shared memory state dict caching is synchronized across all secondary TPU worker nodes.
    if state_dict is None:
        return 0

    t_start = time.perf_counter()

    if isinstance(state_dict, dict) and "grouped" in state_dict:
        grouped_dict = state_dict["grouped"]
    else:
        grouped_dict = {"all": state_dict}

    total_keys = 0
    from concurrent.futures import ThreadPoolExecutor

    layout = VllmParameterLayout(vllm_model, vllm_model.model)

    temp_tpu_tensors = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for group_name, group_sd in grouped_dict.items():
            keys_loaded = _load_single_group_on_worker(
                vllm_model,
                group_sd,
                rank,
                executor=executor,
                temp_tpu_tensors=temp_tpu_tensors,
                world_size=world_size,
                layout=layout,
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


def _validate_incoming_shape(key, incoming_shape, logical_shape, dim, rank, world_size, flipped) -> None:
    """Checks that an incoming tensor is the global weight, not a trainer shard.

    The loader reshards by cutting ``logical_shape[dim]`` entries out of the
    incoming tensor at ``logical_shape[dim] * rank``. That is only valid if the
    incoming tensor spans all ``world_size`` rollout shards along ``dim``.

    If it does not, the arithmetic degrades in two ways, neither of which
    reports itself honestly:

    * High ranks index past the end. Python clamps the slice instead of
      raising, the shortfall desynchronises every subsequent offset in the flat
      buffer, and the first visible symptom is a ``view()`` size error against
      some unrelated parameter much later in the group.
    * Low ranks stay in bounds and load *silently wrong* weights, because the
      entries they read are no longer the ones they own globally.

    ``logical_shape`` is the local parameter's shape in HF orientation, so the
    message stays readable even for the weights ``vllm_torchtpu`` stores
    transposed.
    """
    if world_size in (None, 0):
        return
    expected = logical_shape[dim] * world_size
    if incoming_shape[dim] == expected:
        return
    orientation = " (stored transposed by vllm_torchtpu)" if flipped else ""
    raise ValueError(
        f"Weight sync reshard mismatch for '{key}': received shape {tuple(incoming_shape)} but rollout "
        f"tensor parallelism of {world_size} over dim {dim} requires a global size of {expected} "
        f"(local shard {tuple(logical_shape)} in HF orientation{orientation}). "
        f"Rank {rank} would have read out-of-range or incorrect entries."
    )


def _shard_along(tensor: torch.Tensor, dim: int, shard_size: int, rank: int) -> torch.Tensor:
    indices = [slice(None)] * tensor.dim()
    indices[dim] = slice(shard_size * rank, shard_size * (rank + 1))
    # COMMENT: no .clone() -- the slice stays a zero-copy view until reshape(-1).
    return tensor[tuple(indices)]


def _build_local_tensor(target_key, target_local, entries, flat_cpu, rank, world_size, flipped) -> torch.Tensor:
    """Reshard one target parameter and return it flattened in storage order.

    ``entries`` are ``(part_index, num_parts, shape, numel, offset)`` tuples,
    sorted by ``part_index``: one entry for an ordinary parameter, several when
    the rollout model fuses them.
    """
    local_shape = tuple(target_local.shape)
    # The shape the parameter *would* have in HF orientation. Everything below
    # reasons in this space and transposes back once, at the end.
    logical_shape = local_shape[::-1] if flipped else local_shape

    if len(entries) == 1 and entries[0][1] == 1:
        _, _, shape, numel, offset = entries[0]
        shape = tuple(shape)
        global_tensor = flat_cpu[offset : offset + numel].view(shape)

        if shape == logical_shape:
            local = global_tensor
        else:
            mismatched = [d for d in range(len(shape)) if shape[d] != logical_shape[d]]
            if len(mismatched) != 1:
                raise ValueError(
                    f"Weight sync cannot reshard '{target_key}': incoming shape {shape} and local "
                    f"shape {logical_shape} (HF orientation) differ in {len(mismatched)} dimensions, "
                    "so there is no single axis to shard along."
                )
            dim = mismatched[0]
            _validate_incoming_shape(target_key, shape, logical_shape, dim, rank, world_size, flipped)
            local = _shard_along(global_tensor, dim, logical_shape[dim], rank)
    else:
        # Fused parameter. vLLM concatenates the parts along the output
        # dimension *after* sharding each one, so rank r owns
        # concat(part[0][r], part[1][r], ...) -- not a contiguous slice of the
        # concatenation. Rebuild it in that order.
        num_parts = entries[0][1]
        if len(entries) != num_parts:
            present = [int(e[0]) for e in entries]
            raise ValueError(
                f"Weight sync received {len(entries)} of {num_parts} parts for fused parameter "
                f"'{target_key}' (part indices {present}). A fused parameter can only be rebuilt "
                "from all of its parts."
            )

        total_out = sum(int(e[2][0]) for e in entries)
        local_out = logical_shape[0]
        if local_out <= 0 or total_out % local_out != 0:
            raise ValueError(
                f"Weight sync cannot reshard fused parameter '{target_key}': its parts total "
                f"{total_out} output rows, which is not a multiple of the {local_out} rows this "
                f"rank holds. (Uneven KV-head replication across TP ranks is not supported.)"
            )
        shards = total_out // local_out
        if world_size and shards != world_size:
            raise ValueError(
                f"Weight sync cannot reshard fused parameter '{target_key}': its parts imply "
                f"{shards} shards but rollout tensor parallelism is {world_size}."
            )

        pieces = []
        for _, _, shape, numel, offset in entries:
            shape = tuple(shape)
            if shape[1:] != logical_shape[1:]:
                raise ValueError(
                    f"Weight sync cannot reshard fused parameter '{target_key}': a part of shape "
                    f"{shape} does not agree with the local shape {logical_shape} (HF orientation) "
                    "outside the fused dimension."
                )
            if shape[0] % shards != 0:
                raise ValueError(
                    f"Weight sync cannot reshard fused parameter '{target_key}': a part of shape "
                    f"{shape} does not split evenly across {shards} shards."
                )
            global_tensor = flat_cpu[offset : offset + numel].view(shape)
            pieces.append(_shard_along(global_tensor, 0, shape[0] // shards, rank))
        local = torch.cat(pieces, dim=0)

    if tuple(local.shape) != logical_shape:
        raise ValueError(
            f"Weight sync produced shape {tuple(local.shape)} for '{target_key}' but the local "
            f"parameter needs {logical_shape} in HF orientation (stored as {local_shape}, rank {rank})."
        )

    if flipped:
        local = local.transpose(0, 1)
    # reshape() materialises the transposed/sliced view in the parameter's own
    # storage order, which is what the flat copy buffer downstream assumes.
    return local.reshape(-1)


def _load_single_group_on_worker(
    vllm_model,
    group_sd: dict,
    rank: int,
    executor=None,
    temp_tpu_tensors=None,
    world_size: int | None = None,
    layout: "VllmParameterLayout | None" = None,
) -> int:
    flat_tensors = group_sd["flat_tensors"]
    metadata = group_sd["metadata"]

    if layout is None:
        layout = VllmParameterLayout(vllm_model, vllm_model.model)

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

    model_sd = dict(vllm_model.model.state_dict())
    # `lm_head` hangs off the wrapper, not off `model`, so it is absent from the
    # dict above. On a tied-embedding model that is harmless (it aliases
    # `embed_tokens`), but on an untied one the output head would never be
    # updated. Add the wrapper's own parameters, skipping the `model.`-prefixed
    # duplicates that would otherwise be written twice under two names.
    try:
        for wrapper_key, wrapper_value in vllm_model.state_dict().items():
            if not wrapper_key.startswith("model."):
                model_sd.setdefault(wrapper_key, wrapper_value)
    except Exception as e:  # pragma: no cover - defensive, model_sd is still usable
        logger.debug(f"Could not enumerate wrapper-level parameters for weight sync: {e}")

    for dtype, flat_data in flat_tensors.items():
        items = clean_metadata.get(dtype, [])
        if not items:
            continue

        flat_cpu = torch.from_numpy(flat_data) if not isinstance(flat_data, torch.Tensor) else flat_data
        if dtype == torch.bfloat16 and flat_cpu.dtype == torch.int16:
            # COMMENT: On TPU CPU builds (e.g. torch_tpu), converting torch.bfloat16 to numpy raises a TypeError.
            # Thus, we serialize it as int16 (same bit representation) and view it back to bfloat16 here.
            # TODO: remove HACK once PyTorch CPU native bfloat16 to numpy conversion is universally stable.
            flat_cpu = flat_cpu.view(torch.bfloat16)

        # Group incoming keys by the rollout parameter they land in. Several
        # keys share a target whenever vLLM fuses projections, so resolution
        # has to happen before any slicing.
        targets: dict[str, list] = {}
        unresolved: list[str] = []
        for k, shape, numel, offset in items:
            target_key, part_index, num_parts = layout.resolve(k, model_sd)
            if target_key is None:
                unresolved.append(k)
                continue
            targets.setdefault(target_key, []).append((part_index, num_parts, shape, numel, offset))
        if unresolved:
            _log_unresolved(unresolved, rank)

        local_items = []
        local_tensors_to_cat = []
        local_offset = 0

        def process_target(entry, flat_cpu=flat_cpu):
            target_key, entries = entry
            entries = sorted(entries, key=lambda e: e[0])

            target_v = model_sd[target_key]
            target_local = target_v.to_local() if isinstance(target_v, DTensor) else target_v
            flipped = target_local.dim() == 2 and layout.is_flipped(target_key)

            param_cpu_local_flat = _build_local_tensor(
                target_key, target_local, entries, flat_cpu, rank, world_size, flipped
            )

            # The flat buffer below is indexed using target_local.numel(). A slice
            # that came out a different size would silently shift every later
            # parameter's offset, so refuse to enter it into the buffer at all.
            if param_cpu_local_flat.numel() != target_local.numel():
                raise ValueError(
                    f"Weight sync produced a {param_cpu_local_flat.numel()}-element shard for "
                    f"'{target_key}' but the local parameter needs {target_local.numel()} "
                    f"(local {tuple(target_local.shape)}, rank {rank}). Refusing to continue: the "
                    "flat copy buffer would desynchronise and fail later against an unrelated parameter."
                )

            return (target_key, target_local.shape, target_local.numel(), param_cpu_local_flat)

        target_entries = list(targets.items())
        if executor is None:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=8) as local_exec:
                sliced_results = list(local_exec.map(process_target, target_entries))
        else:
            sliced_results = list(executor.map(process_target, target_entries))

        for res in sliced_results:
            if res is None:
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

            if temp_tpu_tensors is None:
                del flat_chunk_tpu
                import torch_tpu

                torch_tpu._internal.sync.synchronize(wait=True)

    return num_keys


# =====================================================================
# TPUCheckpointEngine Registration
# =====================================================================


@CheckpointEngineRegistry.register("tpu")
class TPUCheckpointEngine(CheckpointEngine):
    def __init__(self, bucket_size: int = 0, is_master: bool = False, **kwargs) -> None:
        self.is_master = is_master
        self.bucket_size = bucket_size

        # Connect to or create named Ray TPUWeightRegistry actor
        try:
            self.registry = ray.get_actor("TPUWeightRegistry", namespace="verl")
        except ValueError:
            try:
                # COMMENT: Since TPUWeightRegistry is already a remote class decorated with @ray.remote,
                # wrapping it with ray.remote(TPUWeightRegistry) throws a TypeError.
                # Calling TPUWeightRegistry.options directly is the correct way to specify options.
                # TODO: remove HACK once a unified and clean TPU checkpoint/weight registry engine is standard.
                self.registry = TPUWeightRegistry.options(
                    name="TPUWeightRegistry", namespace="verl", lifetime="detached"
                ).remote()
            except Exception:
                self.registry = ray.get_actor("TPUWeightRegistry", namespace="verl")

    def prepare(self) -> dict[str, Any]:
        return {}

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]):
        return {}, {}

    def init_process_group(self, **kwargs):
        pass

    def finalize(self):
        pass

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        t_start = time.perf_counter()

        try:
            import torch_tpu

            torch_tpu._internal.sync.synchronize(wait=True)
        except Exception:
            pass

        if not self.is_master:
            # Non-master ranks must consume the generator to prevent hangs
            for _ in weights:
                pass
            return

        step_key = global_steps if global_steps is not None else 0
        logger.info(f"@@@ TPUCheckpointEngine: [Step {step_key}] Start send_weights...")

        # Time generator consumption and CPU offloading
        t_offload_start = time.perf_counter()
        grouped_weights = {}
        for k, v in weights:
            cpu_v = v.detach().cpu()
            if "layers." in k:
                # Extract layer part: model.layers.12.self_attn... -> model.layers.12
                parts = k.split(".")
                idx = parts.index("layers")
                group_name = ".".join(parts[: idx + 2])
            else:
                group_name = "other"
            grouped_weights.setdefault(group_name, []).append((k, cpu_v))
        t_offload = time.perf_counter() - t_offload_start

        # Time grouping and flattening
        t_group_start = time.perf_counter()
        grouped_dict = {}
        for group_name, group_items in grouped_weights.items():
            by_dtype = {}
            for k, cpu_v in group_items:
                by_dtype.setdefault(cpu_v.dtype, []).append((k, cpu_v))

            flat_tensors = {}
            metadata = {}
            for dtype, items in by_dtype.items():
                flat_cpu = torch.cat([v.view(-1) for _, v in items])
                if dtype == torch.bfloat16:
                    flat_tensors[dtype] = flat_cpu.view(torch.int16).numpy()
                else:
                    flat_tensors[dtype] = flat_cpu.numpy()
                metadata[dtype] = [(k, v.shape, v.numel()) for k, v in items]

            grouped_dict[group_name] = {"flat_tensors": flat_tensors, "metadata": metadata}

        state_dict = {"grouped": grouped_dict}
        t_group = time.perf_counter() - t_group_start

        # Time Ray Put upload
        t_put_start = time.perf_counter()
        ref = ray.put(state_dict)
        t_put = time.perf_counter() - t_put_start

        # Time Registry update
        t_reg_start = time.perf_counter()
        await self.registry.set_weights.remote(step_key, ref)
        t_reg = time.perf_counter() - t_reg_start

        t_total = time.perf_counter() - t_start
        logger.debug(
            f"TPUCheckpointEngine Phase A [Step {step_key}]: Total={t_total:.3f}s, "
            f"Offload={t_offload:.3f}s, GroupFlatten={t_group:.3f}s, RayPut={t_put:.3f}s, Registry={t_reg:.3f}s"
        )

    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        # Rollout uses load_weights_from_ray_registry directly, receive_weights is unused
        raise NotImplementedError("Rollout on TPU uses direct load_weights_from_ray_registry via collective_rpc.")


async def update_tpu_weights(manager, global_steps: int | None = None) -> dict:
    """Synchronize weights from actor worker group to rollout replicas on TPU."""
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
        await asyncio.gather(*actor_refs)
    elif actor_refs is not None:
        await actor_refs

    # 2. Call collective_rpc on all rollout replicas to load weights from the registry
    step_key = global_steps if global_steps is not None else 0
    futures = [
        replica.server_handle.collective_rpc.remote(method="load_weights_from_ray_registry", args=(step_key,))
        for replica in manager.replicas
    ]
    await asyncio.gather(*futures)
    t_total = time.perf_counter() - t_total_start

    logger.info(f"TPU weight sync for step {global_steps} completed in {t_total + t_abort:.3f}s")

    await manager.resume_generation_replicas()
    return {}
