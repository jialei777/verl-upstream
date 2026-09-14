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
import copy
import gc
import importlib
import json
import logging
import multiprocessing
import multiprocessing.process
import os
import sys
import time
import types
from collections import defaultdict
from typing import Any

import numpy as np
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from verl.plugin.platform.tpu_topology import (
    detect_slice_chip_mesh,
    detect_tpu_generation,
    get_chips_per_host_bounds,
    get_devices_per_chip,
    get_devices_per_host,
    get_known_device_counts,
    get_torch_tpu_topology,
)
from verl.utils.device import get_resource_name

# --- Google TPU specific global constants ---
TPU_WEIGHT_REGISTRY_ACTOR_NAME = "TPUWeightRegistry"
TPU_WEIGHT_REGISTRY_NAMESPACE = "verl"
TPU_ROLLOUT_BASE_PORT = 8070

# Environment variables that name the *one* device a single process owns.
# They are meaningful only inside the process they were computed for, so they
# must never be copied from one process's environment into another's.
TPU_PER_PROCESS_DEVICE_ENV_VARS = frozenset(
    {
        "TPU_VISIBLE_DEVICES",
        "TPU_VISIBLE_CHIPS",
    }
)
# -------------------------------------------

try:
    import torch_tpu
except ImportError:
    torch_tpu = None

try:
    from verl.checkpoint_engine.tpu_checkpoint_engine import load_weights_on_worker
except ImportError:
    load_weights_on_worker = None

# NOTE: deliberately no ``setLevel`` here. The TPU patches emit their resolved
# topology and per-worker environment at INFO, and those lines are the only
# record of what the rollout was actually configured with. Pinning the level
# to WARN hides them; let the process-wide logging configuration decide.
logger = logging.getLogger(__name__)

# --- Optional TPU vLLM backend resolution ------------------------------------
#
# Two mutually exclusive backends can supply the symbols below:
#   * ``vllm_torchtpu`` -- the torch_tpu / PyTorch backend, which is what verl
#     drives on TPU.
#   * ``tpu_inference`` -- the JAX backend.
#
# Both are optional, and crucially either can be *installed but broken*.
# Probing must therefore catch ``Exception``, not just ``ImportError``: in the
# current TPU image, importing ``tpu_inference`` pulls in ``torchax``, which
# raises ``AttributeError: The underlying op of 'aten.prod' has no overload
# name 'dim_Dimname'`` against torch 2.12. An ``ImportError``-only guard lets
# that escape and makes this whole module unimportable -- even though the
# backend actually in use (``vllm_torchtpu``) is perfectly healthy.


def _optional_import(candidates: tuple[tuple[str, str | None], ...], what: str) -> Any:
    """Return the first resolvable candidate, or ``None`` if none resolve.

    Args:
        candidates: ``(module_path, attribute)`` pairs, tried in order. An
            attribute of ``None`` means "import the module itself".
        what: Human-readable name of the symbol, used in the debug logs.

    Returns:
        The first successfully resolved module or attribute, else ``None``.
    """
    for module_path, attribute in candidates:
        target = f"{module_path}.{attribute}" if attribute else module_path
        try:
            module = importlib.import_module(module_path)
            return getattr(module, attribute) if attribute else module
        except Exception as e:
            logger.debug(f"[TPU] {what}: {target} unavailable ({type(e).__name__}: {e})")
    return None


ray_distributed_executor = _optional_import(
    (
        ("vllm_torchtpu.executors.ray_distributed_executor", None),
        ("tpu_inference.executors.ray_distributed_executor", None),
    ),
    "ray_distributed_executor",
)

tpu_platform = _optional_import(
    (
        ("vllm_torchtpu.platforms.tpu_platform", None),
        ("tpu_inference.platforms.tpu_platform", None),
    ),
    "tpu_platform",
)

tpu_envs = _optional_import(
    (("vllm_torchtpu.envs", None), ("tpu_inference.envs", None)),
    "tpu_envs",
)

TPUWorker = _optional_import(
    (
        ("vllm_torchtpu.worker.tpu_worker", "TPUWorker"),
        ("tpu_inference.worker.tpu_worker", "TPUWorker"),
    ),
    "TPUWorker",
)

if tpu_platform is None and ray_distributed_executor is None:
    logger.warning(
        "[TPU] Neither 'vllm_torchtpu' nor 'tpu_inference' could be imported. "
        "TPU rollout patches will be skipped; run with VERL_LOGGING_LEVEL=DEBUG "
        "to see why each candidate was rejected."
    )

AttentionConfig = _optional_import((("vllm.config", "AttentionConfig"),), "AttentionConfig")

AttentionBackendEnum = _optional_import(
    (("vllm.v1.attention.backends.registry", "AttentionBackendEnum"),),
    "AttentionBackendEnum",
)

EngineArgs = _optional_import((("vllm.engine.arg_utils", "EngineArgs"),), "EngineArgs")
AsyncEngineArgs = _optional_import((("vllm.engine.arg_utils", "AsyncEngineArgs"),), "AsyncEngineArgs")

vllm_envs = _optional_import((("vllm.envs", None),), "vllm_envs")

get_ip = _optional_import(
    (("vllm.utils", "get_ip"), ("vllm.utils.network_utils", "get_ip")),
    "get_ip",
)

get_open_port = _optional_import(
    (
        ("vllm.utils", "get_open_port"),
        ("vllm_torchtpu.utils", "get_open_port"),
        ("vllm.utils.network_utils", "get_open_port"),
    ),
    "get_open_port",
)

# NOTE: vLLM's own generic helper is preferred over the JAX backend here. This
# is only a "build a tcp://host:port string" utility, so when the active
# backend simply stopped exporting it, reaching for a whole different platform
# module is both unnecessary and (see above) actively dangerous.
get_distributed_init_method = _optional_import(
    (
        ("vllm_torchtpu.platforms.tpu_platform", "get_distributed_init_method"),
        ("vllm.utils.network_utils", "get_distributed_init_method"),
        ("vllm.utils", "get_distributed_init_method"),
        ("tpu_inference.platforms.tpu_platform", "get_distributed_init_method"),
    ),
    "get_distributed_init_method",
)

current_platform = _optional_import((("vllm.platforms", "current_platform"),), "current_platform")

RayWorkerMetaData = _optional_import(
    (
        ("vllm.v1.executor.ray_executor", "RayWorkerMetaData"),
        ("vllm.v1.executor.ray_utils", "RayWorkerMetaData"),
    ),
    "RayWorkerMetaData",
)

if RayWorkerMetaData is None:

    class RayWorkerMetaData:  # type: ignore[no-redef]
        def __init__(self, worker, created_rank):
            self.worker = worker
            self.created_rank = created_rank
            self.adjusted_rank = None
            self.ip = None


class PickleableProcessWrapper:
    """
    A pickle-compatible wrapper for multiprocessing target functions to ensure
    vLLM-on-TPU worker patches are automatically applied in the spawned processes.
    """

    def __init__(self, target):
        self.target = target

    def __call__(self, *args, **kwargs):
        patch_vllm_for_tpu()
        if self.target is not None:
            return self.target(*args, **kwargs)


def patch_multiprocessing_for_tpu() -> None:
    """
    Monkey-patch multiprocessing.process.BaseProcess to wrap target entrypoints
    with PickleableProcessWrapper, propagating vLLM-on-TPU patches to child processes.
    """
    if getattr(multiprocessing.process.BaseProcess, "_tpu_patched", False):
        return

    original_init = multiprocessing.process.BaseProcess.__init__

    def patched_init(self, *args, **kwargs):
        target = kwargs.get("target", None)
        if target is None and len(args) > 1:
            target = args[1]

        if target is not None:
            wrapped_target = PickleableProcessWrapper(target)
            if "target" in kwargs:
                kwargs["target"] = wrapped_target
            elif len(args) > 1:
                args = list(args)
                args[1] = wrapped_target
                args = tuple(args)

        original_init(self, *args, **kwargs)

    multiprocessing.process.BaseProcess.__init__ = patched_init
    multiprocessing.process.BaseProcess._tpu_patched = True


_orig_run_engine_core = None


def _patched_run_engine_core(*args, **kwargs):
    try:
        patch_vllm_for_tpu()
    except Exception as e:
        print(f"[TPU DEBUG] Error re-applying patch in EngineCoreProc: {e}", flush=True)

    global _orig_run_engine_core
    if _orig_run_engine_core is None:
        import vllm.v1.engine.core as v1_core

        _orig_run_engine_core = getattr(v1_core.EngineCoreProc, "_unpatched_run_engine_core", None)

    if hasattr(_orig_run_engine_core, "__func__"):
        _orig_run_engine_core = _orig_run_engine_core.__func__

    if _orig_run_engine_core is not None:
        return _orig_run_engine_core(*args, **kwargs)
    raise RuntimeError("[TPU ERROR] _orig_run_engine_core could not be resolved in _patched_run_engine_core")


def _seed_vllm_tpu_topology_map() -> None:
    """Seed vLLM's TPU topology lookup table for the TPU generation in use.

    vLLM's TPU backend keeps a ``{num_devices: topology}`` dict that it consults
    when it has to synthesize ``TORCH_TPU_TOPOLOGY`` on its own. Upstream only
    ships the entries it needs for its own launch paths, and the module owning
    the dict has moved between releases (``TPU_TOPOLOGY_MAP`` in
    ``executors.ray_distributed_executor`` on older torchtpu-vLLM builds,
    ``TPU_MULTIHOST_TOPOLOGY_MAP`` in ``platforms.tpu_platform`` on newer ones),
    so probe for every known location rather than assuming one.

    The values are generation-aware, which matters because the same device count
    means different things per generation: 8 devices is *two* v6e hosts
    (``"2,4,1"``) but a *single* tpu7x host (``"2,2,1,2"``, 4 dual-core chips).
    """
    logger = logging.getLogger(__name__)

    generation = detect_tpu_generation()
    chip_mesh = detect_slice_chip_mesh()

    tables = []
    for module, attr in (
        (ray_distributed_executor, "TPU_TOPOLOGY_MAP"),
        (ray_distributed_executor, "TPU_MULTIHOST_TOPOLOGY_MAP"),
        (tpu_platform, "TPU_TOPOLOGY_MAP"),
        (tpu_platform, "TPU_MULTIHOST_TOPOLOGY_MAP"),
    ):
        if module is None:
            continue
        table = getattr(module, attr, None)
        if isinstance(table, dict):
            tables.append((f"{getattr(module, '__name__', module)}.{attr}", table))

    if not tables:
        # Not fatal: verl computes the topology itself in patched_init_workers_ray
        # and exports it before the workers import vLLM, so the map is only a
        # backstop for code paths verl does not drive.
        logger.warning("[TPU] No vLLM TPU topology map found to seed; relying on verl-computed topology.")
        return

    # One entry per plausible rollout size, from a single chip up to a few hosts.
    device_counts = get_known_device_counts(generation)
    overrides = {
        num_devices: get_torch_tpu_topology(num_devices, generation, chip_mesh) for num_devices in device_counts
    }
    for name, table in tables:
        table.update(overrides)
        logger.info(f"[TPU HACK 21] Seeded {name} for TPU '{generation}': {overrides}")


def patch_vllm_for_tpu() -> None:
    """
    Apply TPU-specific patches and workarounds to vLLM and torchtpu-vllm workers.
    Ensures correct topology routing, un-clashed TCP ports, custom weight registry loading,
    and driver-worker environment synchronization on GKE TPU v6e instances.
    """
    logger = logging.getLogger(__name__)

    try:
        import torch
        import torch._dynamo
        import torch._ops
        import torch.compiler

        def _allow(op):
            if op is not None:
                for fn in (
                    getattr(torch.compiler, "allow_in_graph", None),
                    getattr(torch._dynamo, "allow_in_graph", None),
                ):
                    if fn:
                        try:
                            fn(op)
                        except Exception:
                            pass

        for ns_name in ["_c10d_functional", "c10d_functional"]:
            if hasattr(torch.ops, ns_name):
                ns = getattr(torch.ops, ns_name)
                _allow(ns)
                for name in dir(ns):
                    try:
                        attr = getattr(ns, name)
                        _allow(attr)
                        if hasattr(attr, "default"):
                            _allow(attr.default)
                    except Exception:
                        pass
    except Exception:
        pass

    try:
        import tpu_inference.worker.tpu_worker as tw

        if hasattr(tw, "TPUWorker") and not getattr(tw.TPUWorker, "_patched_dynamo", False):
            orig_determine = tw.TPUWorker.determine_available_memory

            def patched_determine(self, *args, **kwargs):
                patch_vllm_for_tpu()
                return orig_determine(self, *args, **kwargs)

            tw.TPUWorker.determine_available_memory = patched_determine
            tw.TPUWorker._patched_dynamo = True
    except Exception:
        pass

    try:
        import tpu_inference.runner.tpu_runner as tr

        if hasattr(tr, "TPUModelRunner") and not getattr(tr.TPUModelRunner, "_patched_dynamo", False):
            orig_profile = tr.TPUModelRunner.profile_run

            def patched_profile(self, *args, **kwargs):
                patch_vllm_for_tpu()
                return orig_profile(self, *args, **kwargs)

            tr.TPUModelRunner.profile_run = patched_profile
            tr.TPUModelRunner._patched_dynamo = True
    except Exception:
        pass

    def dummy_reset_encoder_cache(*args, **kwargs):
        pass

    def load_weights_from_ray_registry(self, step_key: int):
        rank_val = getattr(
            self, "rank", getattr(self, "adjusted_rank", getattr(self, "rpc_rank", int(os.environ.get("RANK", "0"))))
        )

        shm_dir = "/tmp/verl_weight_cache/shared"
        os.makedirs(shm_dir, exist_ok=True)
        shm_file_path = f"{shm_dir}/state_dict_{step_key}.pt"
        shm_tmp_path = f"{shm_dir}/state_dict_{step_key}.tmp"
        shm_ready_path = f"{shm_dir}/state_dict_{step_key}.ready"
        lock_path = f"{shm_dir}/state_dict_{step_key}.lock"
        state_dict_data = None

        # Atomic lock to decide the designated master for this step on this physical VM node
        is_master = False
        lock_fd = None
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            is_master = True
        except FileExistsError:
            is_master = False

        if is_master:
            # Clean up older steps' cache files from /tmp to ensure NVMe disk space never exhausts
            try:
                for file_name in os.listdir(shm_dir):
                    if file_name.startswith("state_dict_"):
                        parts = file_name.split("_")
                        if len(parts) >= 3:
                            try:
                                old_step_str = parts[2].split(".")[0]
                                old_step = int(old_step_str)
                                if old_step < step_key:
                                    old_path = os.path.join(shm_dir, file_name)
                                    os.remove(old_path)
                            except ValueError:
                                pass
            except Exception as e:
                logger.warning(f"Error during cache cleanup: {e}")

            # Node master block: fetch weights once, convert, and cache to /tmp
            try:
                registry = ray.get_actor(TPU_WEIGHT_REGISTRY_ACTOR_NAME, namespace=TPU_WEIGHT_REGISTRY_NAMESPACE)
                state_dict_ref = ray.get(registry.get_weights.remote(step_key))
            except Exception as e:
                logger.warning(f"Failed to get weights from TPUWeightRegistry: {e}")
                if lock_fd is not None:
                    os.close(lock_fd)
                    try:
                        os.remove(lock_path)
                    except Exception:
                        pass
                return 0

            if state_dict_ref is None:
                logger.warning(f"No weights registered under step_key={step_key}")
                if lock_fd is not None:
                    os.close(lock_fd)
                    try:
                        os.remove(lock_path)
                    except Exception:
                        pass
                return 0

            if isinstance(state_dict_ref, str):
                state_dict_data = torch.load(state_dict_ref, map_location="cpu", weights_only=False)
            elif isinstance(state_dict_ref, ray.ObjectRef):
                state_dict_data = ray.get(state_dict_ref)
            else:
                state_dict_data = state_dict_ref

            # Convert NumPy arrays back to native PyTorch CPU tensors
            if isinstance(state_dict_data, dict) and "grouped" in state_dict_data:
                for group_name, group_sd in state_dict_data["grouped"].items():
                    flat_tensors = group_sd["flat_tensors"]
                    for dtype, arr in list(flat_tensors.items()):
                        if isinstance(arr, np.ndarray):
                            flat_tensors[dtype] = torch.from_numpy(arr)

            # Save atomically using temp file and rename
            torch.save(state_dict_data, shm_tmp_path)
            os.replace(shm_tmp_path, shm_file_path)

            # Release memory before mapping
            del state_dict_data
            gc.collect()

            # Signal ready to all other processes on this node
            with open(shm_ready_path, "w") as f:
                f.write("ready")

            if lock_fd is not None:
                os.close(lock_fd)

            # Stagger ranks before loading to serialize memory traffic
            time.sleep((rank_val % 4) * 0.4)

            # Now load memory-mapped version to share pages
            gc.collect()
            # HACK: Catch FileNotFoundError/Timeout when state_dict_0.pt is absent on secondary TPU nodes, falling back to direct HF loading.
            # TODO: remove HACK once shared memory weight cache propagation across Ray multi-node TPU VM topology is fully guaranteed.
            try:
                state_dict_data = torch.load(shm_file_path, map_location="cpu", weights_only=False, mmap=True)
            except Exception as e:
                logger.warning(f"Shared memory state dict {shm_file_path} not available: {e}")
                state_dict_data = None
        else:
            # Node worker processes: wait for ready marker
            t_wait_start = time.time()
            while not os.path.exists(shm_ready_path):
                time.sleep(0.05)
                if time.time() - t_wait_start > 30:
                    break

            # Stagger ranks before loading to serialize memory traffic
            time.sleep((rank_val % 4) * 0.4)

            gc.collect()
            # HACK: Catch FileNotFoundError/Timeout when state_dict_0.pt is absent on secondary TPU nodes, falling back to direct HF loading.
            # TODO: remove HACK once shared memory weight cache propagation across Ray multi-node TPU VM topology is fully guaranteed.
            try:
                state_dict_data = torch.load(shm_file_path, map_location="cpu", weights_only=False, mmap=True)
            except Exception as e:
                logger.warning(f"Shared memory state dict {shm_file_path} not available: {e}")
                state_dict_data = None

        if hasattr(self, "worker") and self.worker is not None:
            worker_inst = self.worker
        else:
            worker_inst = self

        if hasattr(worker_inst, "get_model"):
            vllm_model = worker_inst.get_model()
        else:
            vllm_model = worker_inst.model_runner.model

        num_keys = 0
        if load_weights_on_worker is not None:
            # The loader reshards by rank, so it needs to know how many shards
            # the incoming global tensors are expected to split into. Without
            # this it cannot tell a correctly-gathered weight from a trainer
            # shard, and a mismatch only surfaces much later as a confusing
            # view() error (or, on low ranks, as silently wrong weights).
            world_size_val = getattr(self, "world_size", None)
            if not world_size_val:
                try:
                    world_size_val = int(os.environ.get("WORLD_SIZE", "0")) or None
                except ValueError:
                    world_size_val = None

            res_loader = load_weights_on_worker(vllm_model, state_dict_data, rank_val, world_size_val)
            if isinstance(res_loader, tuple):
                num_keys = res_loader[0]
            else:
                num_keys = res_loader

        del state_dict_data
        gc.collect()

        return num_keys

    patch_multiprocessing_for_tpu()

    if "RAY_RUNTIME_ENV_WORKER_PROCESS_SETUP_HOOK" in os.environ:
        del os.environ["RAY_RUNTIME_ENV_WORKER_PROCESS_SETUP_HOOK"]

    try:
        orig_ray_init = ray.init

        def patched_ray_init(*args, **kwargs):
            if "runtime_env" in kwargs and kwargs["runtime_env"]:
                rt_env = kwargs["runtime_env"]
                if isinstance(rt_env, dict) and "worker_process_setup_hook" in rt_env:
                    rt_env = dict(rt_env)
                    del rt_env["worker_process_setup_hook"]
                    kwargs["runtime_env"] = rt_env
            if "RAY_RUNTIME_ENV_WORKER_PROCESS_SETUP_HOOK" in os.environ:
                del os.environ["RAY_RUNTIME_ENV_WORKER_PROCESS_SETUP_HOOK"]
            return orig_ray_init(*args, **kwargs)

        ray.init = patched_ray_init
    except Exception as e:
        logger.warning(f"Failed to patch ray.init to clear worker_process_setup_hook: {e}")

    try:
        if "vllm.utils.import_utils" not in sys.modules:
            sys.modules["vllm.utils.import_utils"] = types.ModuleType("import_utils")
        sys.modules["vllm.utils.import_utils"].init_cached_hf_modules = lambda: None

        os.environ["VLLM_USE_V1"] = "0"

        if ray_distributed_executor is None:
            return

        try:
            if tpu_platform is not None and AttentionBackendEnum is not None:
                orig_wrap = tpu_platform.TpuPlatform.wrap_engine_kwargs

                def patched_wrap(self, engine_kwargs):
                    orig_wrap(self, engine_kwargs)
                    if "attention_config" in engine_kwargs:
                        engine_kwargs["attention_config"].backend = AttentionBackendEnum.MATH

                tpu_platform.TpuPlatform.wrap_engine_kwargs = patched_wrap
        except Exception as e:
            logger.warning(f"Failed to patch TPUPlatform.wrap_engine_kwargs: {e}")

        try:
            if EngineArgs is not None:
                orig_create_engine_config = EngineArgs.create_engine_config

                def patched_create_engine_config(self, *args, **kwargs):
                    is_tpu = get_resource_name() == "TPU" or os.environ.get("VLLM_USE_V1") == "0"
                    # A rollout spans multiple hosts only once it needs more
                    # devices than one VM exposes. That is 4 on v6e but 8 on
                    # TPU7x, whose 4 chips each present two chiplets.
                    is_multi_host = (
                        self.tensor_parallel_size > get_devices_per_host()
                        or int(os.environ.get("NNODES_ROLLOUT", "1")) > 1
                        or os.environ.get("TPU_MULTIHOST_BACKEND") == "ray"
                    )
                    if is_tpu:
                        os.environ["VLLM_USE_V1"] = "0"
                        if hasattr(self, "use_v1"):
                            self.use_v1 = False

                    if getattr(self, "data_parallel_size", 1) <= 1:
                        if hasattr(self, "data_parallel_external_lb"):
                            self.data_parallel_external_lb = False
                        if hasattr(self, "data_parallel_rank"):
                            self.data_parallel_rank = None
                        if hasattr(self, "data_parallel_size_local"):
                            self.data_parallel_size_local = None
                        if hasattr(self, "data_parallel_start_rank"):
                            self.data_parallel_start_rank = None
                        if hasattr(self, "data_parallel_hybrid_lb"):
                            self.data_parallel_hybrid_lb = False

                    if not is_multi_host:
                        if "TPU_MULTIHOST_BACKEND" in os.environ:
                            del os.environ["TPU_MULTIHOST_BACKEND"]
                        if vllm_envs is not None and hasattr(vllm_envs, "TPU_MULTIHOST_BACKEND"):
                            vllm_envs.TPU_MULTIHOST_BACKEND = None
                        if tpu_envs is not None and hasattr(tpu_envs, "TPU_MULTIHOST_BACKEND"):
                            tpu_envs.TPU_MULTIHOST_BACKEND = None

                        vllm_config = orig_create_engine_config(self, *args, **kwargs)
                        if is_tpu and hasattr(vllm_config, "use_v1"):
                            vllm_config.use_v1 = False

                        logger.info(
                            "[TPU HACK 16] Single-host rollout detected. "
                            "Bypassed forcing Ray distributed executor backend."
                        )
                        if hasattr(vllm_config, "scheduler_config") and hasattr(
                            vllm_config.scheduler_config, "async_scheduling"
                        ):
                            vllm_config.scheduler_config.async_scheduling = True
                            logger.info("[TPU HACK 20] Enabled async_scheduling on TPU for single-host.")
                    else:
                        os.environ["TPU_MULTIHOST_BACKEND"] = "ray"
                        if vllm_envs is not None and hasattr(vllm_envs, "TPU_MULTIHOST_BACKEND"):
                            vllm_envs.TPU_MULTIHOST_BACKEND = "ray"
                        if tpu_envs is not None and hasattr(tpu_envs, "TPU_MULTIHOST_BACKEND"):
                            tpu_envs.TPU_MULTIHOST_BACKEND = "ray"

                        vllm_config = orig_create_engine_config(self, *args, **kwargs)
                        if is_tpu and hasattr(vllm_config, "use_v1"):
                            vllm_config.use_v1 = False
                        vllm_config.parallel_config.distributed_executor_backend = "ray"
                        logger.info(
                            "[TPU HACK 16] Directly forced 'ray' distributed executor backend on TPU for multi-host."
                        )
                        if hasattr(vllm_config, "scheduler_config") and hasattr(
                            vllm_config.scheduler_config, "async_scheduling"
                        ):
                            vllm_config.scheduler_config.async_scheduling = False
                            logger.info(
                                "[TPU HACK 20] Disabled async_scheduling on TPU "
                                "inside patched_create_engine_config because Ray does not support it."
                            )
                    return vllm_config

                EngineArgs.create_engine_config = patched_create_engine_config
                if AsyncEngineArgs is not None:
                    AsyncEngineArgs.create_engine_config = patched_create_engine_config
        except Exception as e:
            logger.warning(f"Failed to patch EngineArgs.create_engine_config: {e}")

        try:
            if TPUWorker is not None:
                TPUWorker.reset_encoder_cache = dummy_reset_encoder_cache
                TPUWorker.load_weights_from_ray_registry = load_weights_from_ray_registry
                logger.info(
                    "[TPU HACK 13] Successfully patched TPUWorker class with "
                    "dummy_reset_encoder_cache and load_weights_from_ray_registry."
                )
        except Exception as e:
            logger.warning(f"Failed to patch TPUWorker class directly: {e}")

        _seed_vllm_tpu_topology_map()

        original_driver_environ_setitem = os.environ.__class__.__setitem__

        def patched_driver_environ_setitem(self, key, value):
            if key in (
                "TORCH_TPU_SLICEBUILDER_ADDRESSES",
                "TPU_PROCESS_ADDRESSES",
                "TPU_CHIPS_PER_HOST_BOUNDS",
                "TPU_HOST_BOUNDS",
                "TORCH_TPU_TOPOLOGY",
                "TPU_WORKER_HOSTNAMES",
            ):
                value = os.environ.get(key, value)
            elif key == "LIBTPU_INIT_ARGS":
                value = value.replace("--deepsea_chip_config_name=megachip_tccontrol", "")
            original_driver_environ_setitem(self, key, value)

        os.environ.__class__.__setitem__ = patched_driver_environ_setitem

        try:
            import vllm.v1.executor.ray_utils as v1_ray_utils

            orig_avail_res = v1_ray_utils.available_resources_per_node

            # Ray can report 0 free TPUs for a node whose devices verl has already
            # reserved through its own placement group, which makes vLLM refuse to
            # schedule. Floor the count at one full host: 4 devices on v6e, 8 on
            # tpu7x (4 chips x 2 addressable cores).
            devices_per_host = float(get_devices_per_host())

            def patched_avail_res(*args, **kwargs):
                res_map = orig_avail_res(*args, **kwargs)
                for node_id, res in res_map.items():
                    res["TPU"] = max(res.get("TPU", 0.0), devices_per_host)
                return res_map

            v1_ray_utils.available_resources_per_node = patched_avail_res

            orig_init_ray_cluster = v1_ray_utils.initialize_ray_cluster

            def patched_initialize_ray_cluster(parallel_config):
                if parallel_config.placement_group is None:
                    curr_pg = ray.util.get_current_placement_group()
                    if curr_pg is None:
                        pg_name = os.environ.get("VERL_ROLLOUT_PG_NAME")
                        if pg_name:
                            try:
                                curr_pg = ray.util.get_placement_group(pg_name)
                            except Exception:
                                pass
                        if curr_pg is None:
                            try:
                                pgs = ray.util.placement_group_table()
                                for pg_id, pg_info in pgs.items():
                                    state = (
                                        pg_info.get("state")
                                        if isinstance(pg_info, dict)
                                        else getattr(pg_info, "state", None)
                                    )
                                    if hasattr(state, "name"):
                                        state = state.name
                                    name = (
                                        pg_info.get("name")
                                        if isinstance(pg_info, dict)
                                        else getattr(pg_info, "name", None)
                                    )
                                    if (
                                        str(state) in ("CREATED", "1")
                                        and name
                                        and ("rollout" in str(name) or "global" in str(name))
                                    ):
                                        try:
                                            candidate_pg = ray.util.get_placement_group(name)
                                            if candidate_pg is not None:
                                                num_bundles = len(getattr(candidate_pg, "bundle_specs", []))
                                                if num_bundles >= parallel_config.world_size:
                                                    curr_pg = candidate_pg
                                                    break
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                    parallel_config.placement_group = curr_pg
                try:
                    return orig_init_ray_cluster(parallel_config)
                except ValueError as e:
                    if "exceeds the total number of available" in str(e) or "placement group" in str(e):
                        logger.warning(
                            f"[TPU HACK] Bypassed vLLM placement group size validation on multi-node TPU: {e}"
                        )
                        return
                    raise

            try:
                import vllm.v1.engine.core as v1_core

                if not getattr(v1_core, "_verl_tpu_patched", False):
                    v1_core._verl_tpu_patched = True
                    global _orig_run_engine_core
                    raw_fn = v1_core.EngineCoreProc.run_engine_core
                    if hasattr(raw_fn, "__func__"):
                        raw_fn = raw_fn.__func__
                    _orig_run_engine_core = raw_fn
                    v1_core.EngineCoreProc._unpatched_run_engine_core = raw_fn
                    v1_core.EngineCoreProc.run_engine_core = staticmethod(_patched_run_engine_core)
                    v1_core.run_engine_core = _patched_run_engine_core
            except Exception as e3:
                logger.warning(f"Failed to patch v1_core.run_engine_core: {e3}")

            try:
                import vllm.v1.executor.ray_executor as v1_ray_executor

                v1_ray_executor.initialize_ray_cluster = patched_initialize_ray_cluster

                executor_cls = getattr(
                    v1_ray_executor, "RayDistributedExecutor", getattr(v1_ray_executor, "RayExecutor", None)
                )
                if executor_cls is not None and hasattr(executor_cls, "_init_workers_ray"):
                    orig_init_workers_ray = executor_cls._init_workers_ray

                    def patched_init_workers_ray(self, placement_group, **ray_remote_kwargs):
                        if not os.environ.get("VLLM_RAY_BUNDLE_INDICES"):
                            indices = [str(i) for i in range(len(placement_group.bundle_specs))]
                            if indices and len(indices) >= self.parallel_config.world_size:
                                os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(
                                    indices[: self.parallel_config.world_size]
                                )
                        return orig_init_workers_ray(self, placement_group, **ray_remote_kwargs)

                    executor_cls._init_workers_ray = patched_init_workers_ray
            except Exception as e2:
                logger.warning(f"Failed to patch v1_ray_executor: {e2}")
        except Exception as e:
            logger.warning(f"Failed to patch v1_ray_utils: {e}")

        OriginalRayWorkerWrapper = ray_distributed_executor.RayWorkerWrapper
        original_init_worker = OriginalRayWorkerWrapper.init_worker
        original_wrapper_init = OriginalRayWorkerWrapper.__init__

        def patched_wrapper_init(self, *args, **kwargs):
            patch_vllm_for_tpu()
            try:
                local_ip = ray.util.get_node_ip_address()
                pod_map_str = os.environ.get("TPU_POD_IP_TO_SLICE")
                if pod_map_str:
                    pod_map = json.loads(pod_map_str)
                    current_slice = pod_map.get(local_ip)
                    if current_slice:
                        slice_ips = sorted([ip for ip, sl in pod_map.items() if sl == current_slice])
                        if local_ip in slice_ips:
                            worker_id = slice_ips.index(local_ip)
                            os.environ["TPU_WORKER_ID"] = str(worker_id)
                            os.environ["TPU_WORKER_HOSTNAMES"] = ",".join(slice_ips)
                            logger.info(
                                f"[TPU WORKER FIX] Node {local_ip} (slice {current_slice}) "
                                f"configured TPU_WORKER_ID={worker_id}, hostnames={slice_ips}"
                            )
            except Exception as e:
                logger.warning(f"Failed setting local TPU_WORKER_ID: {e}")
            return original_wrapper_init(self, *args, **kwargs)

        OriginalRayWorkerWrapper.__init__ = patched_wrapper_init

        def patched_setup_device_if_necessary(self):
            patch_vllm_for_tpu()
            if not getattr(self, "compiled_dag_cuda_device_set", False):
                try:
                    from vllm.platforms import current_platform

                    if current_platform.is_cuda() and hasattr(self.worker, "device") and self.worker.device is not None:
                        current_platform.set_device(self.worker.device)
                except Exception:
                    pass
                self.compiled_dag_cuda_device_set = True

        OriginalRayWorkerWrapper.setup_device_if_necessary = patched_setup_device_if_necessary

        def patched_init_worker(self, *args, **kwargs):
            patch_vllm_for_tpu()
            if "vllm.utils.import_utils" not in sys.modules:
                sys.modules["vllm.utils.import_utils"] = types.ModuleType("import_utils")
            sys.modules["vllm.utils.import_utils"].init_cached_hf_modules = lambda: None

            try:
                import vllm.model_executor.model_loader.weight_utils as vllm_weight_utils

                vllm_weight_utils.initialize_dummy_weights = lambda *args, **kwargs: None
            except Exception:
                pass
            try:
                import vllm.model_executor.model_loader.dummy_loader as vllm_dummy_loader

                vllm_dummy_loader.initialize_dummy_weights = lambda *args, **kwargs: None
            except Exception:
                pass

            self.__class__.load_weights_from_ray_registry = load_weights_from_ray_registry
            self.__class__.load_weights_from_state_dict_on_worker = lambda self, sd: load_weights_from_ray_registry(
                self, 0
            )
            self.__class__.reset_encoder_cache = dummy_reset_encoder_cache

            torch.set_grad_enabled(False)

            res = original_init_worker(self, *args, **kwargs)
            if hasattr(self, "worker") and self.worker is not None:
                self.worker.reset_encoder_cache = dummy_reset_encoder_cache
                self.worker.__class__.reset_encoder_cache = dummy_reset_encoder_cache
            return res

        OriginalRayWorkerWrapper.init_worker = patched_init_worker

        def patched_init_workers_ray(self, placement_group, **ray_remote_kwargs):
            RayWorkerWrapper_local = ray_distributed_executor.RayWorkerWrapper

            self.workers = []
            self.pp_tp_workers = []

            if self.parallel_config.ray_workers_use_nsight:
                ray_remote_kwargs = self._configure_ray_workers_use_nsight(ray_remote_kwargs)

            bundle_indices = []
            if vllm_envs is not None and vllm_envs.VLLM_RAY_BUNDLE_INDICES:
                bundle_indices = list(map(int, vllm_envs.VLLM_RAY_BUNDLE_INDICES.split(",")))
                assert len(bundle_indices) == self.parallel_config.world_size, (
                    "VLLM_RAY_BUNDLE_INDICES must have the same size"
                    f" as the world size, but got {bundle_indices=} "
                    f"and {self.parallel_config.world_size=}"
                )
                assert len(set(bundle_indices)) == len(bundle_indices), (
                    f"VLLM_RAY_BUNDLE_INDICES cannot have duplicate values, but got {bundle_indices=}"
                )
            else:
                for bundle_id, bundle in enumerate(placement_group.bundle_specs):
                    if current_platform is not None and bundle.get(current_platform.ray_device_key, 0):
                        bundle_indices.append(bundle_id)

            worker_metadata = []
            driver_ip = get_ip() if get_ip is not None else ""
            num_tpu_per_worker = 1.0
            for rank, bundle_id in enumerate(bundle_indices):
                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=bundle_id,
                )
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=0,
                    resources={current_platform.ray_device_key: num_tpu_per_worker}
                    if current_platform is not None
                    else {},
                    scheduling_strategy=scheduling_strategy,
                    **ray_remote_kwargs,
                )(RayWorkerWrapper_local).remote(rpc_rank=rank)
                worker_metadata.append(
                    RayWorkerMetaData(worker=worker, created_rank=rank) if RayWorkerMetaData is not None else None
                )

            worker_ips = ray.get([each.worker.get_node_ip.remote() for each in worker_metadata])

            for each, ip in zip(worker_metadata, worker_ips, strict=False):
                each.ip = ip

            logger.info(f"Initialized worker_metadata: {worker_metadata}")

            ip_counts = {}
            for ip in worker_ips:
                ip_counts[ip] = ip_counts.get(ip, 0) + 1

            def sort_by_driver_then_worker_ip(item):
                ip = item.ip
                return (0 if ip == driver_ip else 1, ip_counts[ip], ip)

            sorted_worker_metadata = sorted(worker_metadata, key=sort_by_driver_then_worker_ip)
            start_rank = 0
            for i, item in enumerate(sorted_worker_metadata):
                item.adjusted_rank = i + start_rank
            logger.info(f"Initialized sorted worker_metadata: {sorted_worker_metadata}")

            self.workers = [item.worker for item in sorted_worker_metadata]
            rerank_mapping = {item.created_rank: item.adjusted_rank for item in sorted_worker_metadata}
            self.collective_rpc("adjust_rank", args=(rerank_mapping,))

            worker_node_and_tpu_ids = []
            for worker in self.workers:
                worker_node_and_tpu_ids.append(ray.get(worker.get_node_and_gpu_ids.remote()))

            node_workers = defaultdict(list)
            node_tpus = defaultdict(list)

            for i, (node_id, tpu_ids) in enumerate(worker_node_and_tpu_ids):
                node_workers[node_id].append(i)
                tpu_ids = [int(x) for x in tpu_ids]
                node_tpus[node_id].extend(tpu_ids)
            for node_id, tpu_ids in node_tpus.items():
                node_tpus[node_id] = sorted(tpu_ids)
            logger.info(f"RayDistributedExecutor | node_workers={node_workers} | node_tpus={node_tpus}")

            all_ips = set(worker_ips + [driver_ip])
            n_ips = len(all_ips)
            n_nodes = len(node_workers)

            if n_nodes != n_ips:
                logger.warning(
                    f"Got {n_nodes} nodes but with {n_ips} IP addresses. "
                    "This is not a typical production setup whose "
                    "number of nodes and IPs is equal. This setup may "
                    "lead to unexpected behaviors."
                )

            unique_node_ids = list(node_workers.keys())
            num_nodes = len(unique_node_ids)

            sb_addresses = []
            base_port = int(os.environ.get("TORCH_TPU_BASE_PORT", TPU_ROLLOUT_BASE_PORT))
            for node_id in unique_node_ids:
                w_idx = node_workers[node_id][0]
                host_ip = sorted_worker_metadata[w_idx].ip
                chips_on_node = len(node_workers[node_id])
                for lr in range(chips_on_node):
                    sb_addresses.append(f"{host_ip}:{base_port + lr}")

            sb_addresses_str = ",".join(sb_addresses)
            os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = sb_addresses_str
            logger.info(f"Constructed TORCH_TPU_SLICEBUILDER_ADDRESSES: {sb_addresses_str}")

            # NOTE: there is one worker per *addressable device*, not per chip.
            # The distinction only bites on TPU7x, where every chip exposes two
            # independent cores: a single 4-chip tpu7x host produces 8 workers,
            # the same count as a *two-host* v6e slice. Deriving the geometry
            # from the detected generation keeps both readings correct.
            total_devices = len(self.workers)
            generation = detect_tpu_generation()
            devices_per_chip = get_devices_per_chip(generation)
            devices_per_host = get_devices_per_host(generation)
            topology = get_torch_tpu_topology(total_devices, generation, detect_slice_chip_mesh())
            chips_per_host = str(devices_per_host)

            if devices_per_chip > 1:
                # TPU7x. torch_tpu itself overwrites TPU_HOST_BOUNDS with the
                # topology verbatim and TPU_CHIPS_PER_HOST_BOUNDS with an
                # all-ones string of matching arity ("1,1,1,1" for a 4D
                # topology), so mirror that for the executor paths that read
                # these variables before torch_tpu initializes.
                host_bounds = topology
                chips_per_host_bounds = get_chips_per_host_bounds(topology)
            elif total_devices == 4:
                # One device per chip: a 2x2 slice fits in a single host, so the
                # whole mesh is described by the per-host bounds instead.
                host_bounds = "1,1,1"
                chips_per_host_bounds = "2,2,1" if num_nodes == 1 else "1,1,1"
            elif total_devices in (8, 32):
                host_bounds = topology
                chips_per_host_bounds = "1,1,1"
            else:
                host_bounds = "1,1,1"
                chips_per_host_bounds = "1,1,1"

            logger.info(
                f"RayDistributedExecutor | TPU '{generation}': {total_devices} devices across {num_nodes} node(s) -> "
                f"TORCH_TPU_TOPOLOGY={topology}, TPU_HOST_BOUNDS={host_bounds}, "
                f"TPU_CHIPS_PER_HOST_BOUNDS={chips_per_host_bounds}, CHIPS_PER_HOST={chips_per_host}"
            )

            rank_0_node_id = unique_node_ids[0]
            rank_0_worker_index = node_workers[rank_0_node_id][0]
            master_addr = sorted_worker_metadata[rank_0_worker_index].ip
            master_port = str(get_open_port()) if get_open_port is not None else ""

            all_args_to_update_environment_variables = []
            for i in range(total_devices):
                node_id = worker_node_and_tpu_ids[i][0]
                node_rank = unique_node_ids.index(node_id)
                args = {
                    "NNODES": str(num_nodes),
                    "NODE_RANK": str(node_rank),
                    "MASTER_ADDR": master_addr,
                    "MASTER_PORT": master_port,
                    "TORCH_TPU_TOPOLOGY": topology,
                    "LOCAL_WORLD_SIZE": str(len(node_tpus[node_id])),
                }
                if "TORCH_TPU_XPROF_SESSION_ID" not in os.environ:
                    os.environ["TORCH_TPU_XPROF_SESSION_ID"] = str(time.time_ns())

                args["TORCH_TPU_XPROF_SESSION_ID"] = os.environ["TORCH_TPU_XPROF_SESSION_ID"]
                all_args_to_update_environment_variables.append(args)

            env_vars_to_copy_list = []
            if get_env_vars_to_copy is not None:
                env_vars_to_copy_list = get_env_vars_to_copy(
                    exclude_vars=self.WORKER_SPECIFIC_ENV_VARS,
                    additional_vars=set(current_platform.additional_env_vars)
                    if current_platform is not None
                    else set(),
                    destination="workers",
                )

            for i, args in enumerate(all_args_to_update_environment_variables):
                for name in env_vars_to_copy_list:
                    if name in os.environ:
                        args[name] = os.environ[name]
                logger.debug(f"RayDistributedExecutor | Worker {i} environment variables before patch: {args}")

            self._env_vars_for_all_workers = all_args_to_update_environment_variables

            try:
                unique_host_ips = [sorted_worker_metadata[node_workers[nid][0]].ip for nid in unique_node_ids]
                host_names_str = ",".join(unique_host_ips)
                for i, worker in enumerate(self.workers):
                    node_id = worker_node_and_tpu_ids[i][0]
                    host_idx = unique_node_ids.index(node_id)
                    local_chip_id = node_workers[node_id].index(i)
                    args = self._env_vars_for_all_workers[i]
                    args["RANK"] = str(i)
                    args["LOCAL_RANK"] = str(local_chip_id)
                    args["TPU_VISIBLE_CHIPS"] = str(local_chip_id)
                    args["TPU_VISIBLE_DEVICES"] = str(local_chip_id)
                    args["TPU_PROCESS_PORT"] = str(base_port + local_chip_id)
                    args["CLOUD_TPU_TASK_ID"] = str(host_idx)
                    args["TPU_WORKER_HOSTNAMES"] = host_names_str
                    args["TPU_HOST_BOUNDS"] = host_bounds
                    args["TPU_CHIPS_PER_HOST_BOUNDS"] = chips_per_host_bounds
                    args["CHIPS_PER_HOST"] = chips_per_host
                    args["TORCH_TPU_TOPOLOGY"] = topology
                    args["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = sb_addresses_str
                    args["TPU_PROCESS_ADDRESSES"] = sb_addresses_str
                    if total_devices > devices_per_host or num_nodes > 1:
                        args["TPU_MULTIHOST_BACKEND"] = "ray"

                    logger.info(
                        f"[TPU HACK 11] Patched worker {i} (host {host_idx}, chip {local_chip_id}) env vars: "
                        f"TPU_VISIBLE_CHIPS={local_chip_id}, TPU_PROCESS_PORT={base_port + local_chip_id}, "
                        f"CLOUD_TPU_TASK_ID={host_idx}"
                    )
            except Exception as patch_err:
                logger.warning(f"Failed to inject TPU HACK 11 env vars: {patch_err}")

            self.collective_rpc("update_environment_variables", args=(self._get_env_vars_to_be_updated(),))

            distributed_init_method = (
                get_distributed_init_method(driver_ip, get_open_port())
                if get_distributed_init_method is not None
                else ""
            )

            driver_node_id = ray.get_runtime_context().get_node_id()

            all_kwargs = []
            for rank, (node_id, _) in enumerate(worker_node_and_tpu_ids):
                local_rank = node_workers[node_id].index(rank)
                ip = sorted_worker_metadata[rank].ip
                prev_ip = sorted_worker_metadata[rank - 1].ip if rank > 0 else ""

                worker_vllm_config = self.vllm_config

                if (
                    node_id != driver_node_id
                    and getattr(self.vllm_config, "model_config", None)
                    and getattr(self.vllm_config.model_config, "model_weights", None)
                ):
                    worker_vllm_config = copy.deepcopy(self.vllm_config)
                    worker_vllm_config.model_config.model = worker_vllm_config.model_config.model_weights
                    worker_vllm_config.model_config.model_weights = None

                kwargs = dict(
                    vllm_config=worker_vllm_config,
                    local_rank=local_rank,
                    rank=rank,
                    distributed_init_method=distributed_init_method,
                    is_driver_worker=(not self.parallel_config)
                    or (rank % self.parallel_config.tensor_parallel_size == 0),
                    ip=ip,
                    prev_worker_ip=prev_ip,
                )
                all_kwargs.append(kwargs)
            self.collective_rpc("init_worker", args=(all_kwargs,))
            self.collective_rpc("init_device")
            if self.parallel_config.pipeline_parallel_size > 1:
                self.collective_rpc("initialize_pp_transfer_connect")
            self.collective_rpc("load_model")
            if hasattr(self, "pp_tp_workers"):
                self.pp_tp_workers = []
                pp_size = self.parallel_config.pipeline_parallel_size if self.parallel_config else 1
                tp_size = self.parallel_config.tensor_parallel_size if self.parallel_config else len(self.workers)
                for pp_rank in range(pp_size):
                    self.pp_tp_workers.append([])
                    for tp_rank in range(tp_size):
                        rank = (pp_rank * tp_size) + tp_rank
                        if rank < len(self.workers):
                            self.pp_tp_workers[pp_rank].append(self.workers[rank])

        def patched_execute_dag(
            self,
            scheduler_output,
            grammar_output,
            non_block: bool = False,
        ):
            refs = [worker.execute_model_ray.remote((scheduler_output, grammar_output)) for worker in self.workers]
            if not self.has_connector:
                if not non_block:
                    all_results = ray.get(refs)
                    return all_results[0]
                from vllm.v1.executor.ray_utils import FutureWrapper

                return FutureWrapper(refs[0])

            assert self.kv_output_aggregator is not None
            if not non_block:
                return self.kv_output_aggregator.aggregate(ray.get(refs))
            from vllm.v1.executor.ray_utils import FutureWrapper

            return FutureWrapper(refs, self.kv_output_aggregator)

        if ray_distributed_executor is not None:
            ray_distributed_executor.RayDistributedExecutor._init_workers_ray = patched_init_workers_ray
        try:
            import vllm.executor.ray_distributed_executor as vllm_ray_dist_exec

            vllm_ray_dist_exec.RayDistributedExecutor._init_workers_ray = patched_init_workers_ray
        except Exception:
            pass
        try:
            import vllm.v1.executor.ray_executor as vllm_v1_ray_exec

            vllm_v1_ray_exec.RayDistributedExecutor._init_workers_ray = patched_init_workers_ray
            vllm_v1_ray_exec.RayDistributedExecutor._execute_dag = patched_execute_dag
        except Exception:
            pass

        logger.info("Successfully applied all TPU patches and hacks to vLLM & torchtpu-vllm")
    except Exception as e:
        logger.warning(f"Failed to apply TPU patches: {e}")


# Helper imports from torchtpu-vllm and other places used inside vllm_async_server.
# Same probing rules as the block at the top of this module: catch Exception, and
# never let an unhealthy optional backend break the healthy one.
#
# ``vllm.ray.ray_env`` is upstream vLLM's canonical home for this helper (same
# signature); the TPU forks used to re-export it from their platform module and
# newer builds no longer do. Resolving it matters beyond avoiding a crash: this
# is what copies driver env vars (LIBTPU_INIT_ARGS, HF_TOKEN, VLLM_*) onto the
# rollout workers, so silently losing it would surface much later as a
# misconfigured worker rather than an import error.
get_env_vars_to_copy = _optional_import(
    (
        ("vllm_torchtpu.platforms.tpu_platform", "get_env_vars_to_copy"),
        ("vllm.ray.ray_env", "get_env_vars_to_copy"),
        ("tpu_inference.platforms.tpu_platform", "get_env_vars_to_copy"),
    ),
    "get_env_vars_to_copy",
)

if get_env_vars_to_copy is None:
    logger.warning(
        "[TPU] Could not resolve get_env_vars_to_copy; driver environment variables "
        "will NOT be propagated to rollout workers. Anything the rollout needs "
        "(e.g. LIBTPU_INIT_ARGS) must be set on the workers by other means."
    )


def is_tpu_vllm_run() -> bool:
    """Returns True if executing on a Google TPU resource or with V1 explicitly disabled."""
    return get_resource_name() == "TPU" or os.environ.get("VLLM_USE_V1") == "0"


def override_vllm_configs_for_tpu(args_or_config: Any):
    """Enforces VLLM_USE_V1=0 and use_v1=False across dictionary and namespace objects on TPU."""
    os.environ["VLLM_USE_V1"] = "0"
    try:
        import vllm.envs as vllm_envs

        vllm_envs.VLLM_USE_V1 = False
    except Exception:
        pass

    if isinstance(args_or_config, dict):
        args_or_config["use_v1"] = False
        return

    for obj in [args_or_config, getattr(args_or_config, "model_config", None)]:
        if obj is None:
            continue
        for attr in ["use_v1", "_use_v1"]:
            if hasattr(obj, attr):
                try:
                    setattr(obj, attr, False)
                except Exception:
                    pass
            if hasattr(obj, "__dict__") and attr in obj.__dict__:
                try:
                    obj.__dict__[attr] = False
                except Exception:
                    pass


async def get_tpu_server_launch_config(workers):
    """
    Asynchronously queries node ID, visible chips, and TPU specific environment
    variables from all TPU workers for launching the server actor on TPU.
    """
    worker_infos = await asyncio.gather(
        *[
            worker.__ray_call__.remote(
                lambda self: (
                    ray.get_runtime_context().get_node_id(),
                    os.environ.get("TPU_VISIBLE_CHIPS", "0"),
                )
            )
            for worker in workers
        ]
    )

    worker_tpu_envs = await asyncio.gather(
        *[
            worker.__ray_call__.remote(
                lambda self: {
                    k: v
                    for k, v in os.environ.items()
                    if k.startswith("TPU_")
                    or k.startswith("TORCH_TPU_")
                    or k
                    in (
                        "CLOUD_TPU_TASK_ID",
                        "CHIPS_PER_HOST",
                        "JAX_MEM_FRACTION",
                        "JAX_THREE_G_MEM_ALLOC_ON_FREE",
                        "XLA_PYTHON_CLIENT_PREALLOCATE",
                        "XLA_PYTHON_CLIENT_MEM_FRACTION",
                        "LIBTPU_INIT_ARGS",
                        "TORCH_DYNAMO_RECOMPILE_LIMIT",
                        "SKIP_JAX_PRECOMPILE",
                        "VLLM_ENABLE_V1_MULTIPROCESSING",
                        "XLA_FLAGS",
                    )
                }
            )
            for worker in workers
        ]
    )

    node_id = worker_infos[0][0]
    visible_chips = ",".join([info[1] for info in worker_infos])
    sampled_env = worker_tpu_envs[0] if worker_tpu_envs else {}

    # This snapshot is taken from worker 0 but applied to the vLLM server
    # actor's runtime_env, which Ray then hands down to every engine worker on
    # the host. Host-level settings (topology, slicebuilder addresses, XLA
    # flags) are exactly what we want to share; a device pin is not.
    #
    # torch_tpu treats a single-valued TPU_VISIBLE_DEVICES as authoritative and
    # only falls back to the process's LOCAL_RANK when it is absent or names
    # more than one device (see IsSingleDeviceSpecified in
    # torch_tpu/csrc/common/device_utils.h). Leaking worker 0's "0" therefore
    # makes every engine worker open the same /dev/vfio node and the second one
    # to get there dies with FAILED_PRECONDITION.
    tpu_env_vars = {k: v for k, v in sampled_env.items() if k not in TPU_PER_PROCESS_DEVICE_ENV_VARS}

    dropped = sorted(k for k in sampled_env if k in TPU_PER_PROCESS_DEVICE_ENV_VARS)
    if dropped:
        logger.info(
            f"[TPU] Withheld per-process device variables {dropped} from the vLLM server environment; "
            "each engine worker selects its device from LOCAL_RANK."
        )

    return node_id, visible_chips, tpu_env_vars


def prepare_tpu_server_args(args: dict):
    """Configures TPU-specific CLI/server arguments and environment variables."""
    if not is_tpu_vllm_run():
        return

    args["enable_sleep_mode"] = False
    args["distributed_executor_backend"] = "external_launcher"
    os.environ["TPU_MULTIHOST_BACKEND"] = "ray"

    # ``tpu_envs`` was already resolved against both backends at import time; do
    # not retry the import here, or a broken optional backend gets re-imported
    # (slowly, and noisily) on every call.
    if tpu_envs is None:
        logger.warning("Cannot force TPU_MULTIHOST_BACKEND to ray: no TPU vLLM envs module was resolved.")
        return

    try:
        tpu_envs.TPU_MULTIHOST_BACKEND = "ray"
        if hasattr(tpu_envs, "__getattr__") and hasattr(tpu_envs.__getattr__, "cache_clear"):
            tpu_envs.__getattr__.cache_clear()
    except Exception as env_err:
        logger.warning(f"Failed to force TPU_MULTIHOST_BACKEND to ray: {env_err}")
