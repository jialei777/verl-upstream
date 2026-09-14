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
"""Google TPU platform implementation.

TPU with PyTorch/XLA (torch_tpu) reuses the ``torch.cuda.*`` API surface, so most of
``PlatformCUDA`` works unchanged. This class subclasses ``PlatformCUDA`` and overrides
device-specific environment configuration, resource options, and memory management proxies.
"""

import logging
import os
import sys
from typing import Any, Optional

import ray
import torch

from .platform_cuda import PlatformCUDA
from .platform_manager import PlatformRegistry, get_platform
from .platform_tpu_workarounds import convert_tensors_to_scalars, ensure_absl_flags_parsed, patch_ray_worker
from .tpu_topology import (
    DEFAULT_HBM_BYTES_PER_DEVICE,
    detect_slice_chip_mesh,
    detect_tpu_generation,
    get_chips_per_host_bounds,
    get_devices_per_host,
    get_hbm_bytes_per_device,
    get_torch_tpu_topology,
    normalize_tpu_generation,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Communication port defaults for TPU distributed slice builder meshes
ROLLOUT_BASE_PORT = 8070
TRAINER_BASE_PORT = 8471

# Fallback HBM capacity used before the TPU generation can be resolved.
DEFAULT_HBM_BYTES = DEFAULT_HBM_BYTES_PER_DEVICE


def get_tpu_accelerator_type() -> str:
    """Return the raw accelerator string advertised by Ray or the Cloud TPU environment.

    Examples: ``"TPU-V7X"`` / ``"v7x-8"`` (Ray labels), ``"tpu7x-8"`` or
    ``"v6e-8"`` (``TPU_ACCELERATOR_TYPE``). Returns ``""`` when unavailable.
    """
    tpu_type = ""

    # Query Ray cluster node labels for TPU resource type
    try:
        if ray.is_initialized():
            tpu_nodes = [node for node in ray.nodes() if "TPU" in node.get("Resources", {}) and node.get("Alive")]
            if tpu_nodes:
                labels = tpu_nodes[0].get("Labels", {})
                tpu_type = (labels.get("ray.io/accelerator-type") or labels.get("ray.io/tpu-pod-type") or "").lower()
    except Exception as e:
        logger.warning(f"Unable to query Ray node labels for TPU chip type: {e}")

    # Fallback to environment variables
    if not tpu_type:
        tpu_type = (
            os.environ.get("TPU_ACCELERATOR_TYPE")
            or os.environ.get("ACCELERATOR_TYPE")
            or os.environ.get("TPU_TYPE")
            or ""
        ).lower()

    return tpu_type


def get_tpu_chip_hbm_bytes() -> int:
    """Returns the HBM capacity addressable by a single TPU device, in bytes.

    NOTE: this is deliberately *per device*, not per physical chip. Every verl
    worker drives exactly one addressable device, and on TPU7x a chip's 192 GiB
    is split into two independent 96 GiB chiplet memory spaces. Reporting the
    per-chip figure here would make ``torch.tpu.get_device_properties()`` and
    ``mem_get_info()`` overstate available memory by 2x and cause vLLM to
    over-allocate its KV cache.

    Returns:
        HBM bytes per device, or -1 if the TPU generation could not be resolved.
    """
    tpu_type = get_tpu_accelerator_type()

    generation = normalize_tpu_generation(tpu_type)
    if generation is not None:
        return get_hbm_bytes_per_device(generation)

    logger.warning(f"Unable to determine TPU device HBM bytes for tpu_type='{tpu_type}'. Returning -1.")
    return -1


# Enforce static compilation graph for torch.compile on TPU
try:
    _orig_compile = torch.compile

    def patched_compile(*args, **kwargs):
        if get_platform().device_name == "tpu":
            kwargs["dynamic"] = False
        return _orig_compile(*args, **kwargs)

    torch.compile = patched_compile
except Exception as e:
    logger.warning(f"Failed to patch torch.compile for TPU: {e}")


class DummyTpuDeviceModule:
    """Fallback device module for CPU-only nodes and driver processes.

    Provides no-op implementations for torch.tpu APIs on processes where torch_tpu is not imported
    or no TPU devices are attached.
    """

    def is_available(self) -> bool:
        return False

    def set_device(self, device_index: Any) -> None:
        pass

    def current_device(self) -> int:
        return 0

    def device_count(self) -> int:
        return 0

    def synchronize(self) -> None:
        pass

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)


class TPUDeviceModuleProxy:
    """Proxy wrapper for torch.tpu to emulate PyTorch CUDA memory management APIs.

    Provides default fallback implementations for CUDA memory tracking methods
    (e.g., memory_reserved, memory_allocated, get_device_properties) that are called throughout verl's codebase
    but not natively provided by torch_tpu.
    """

    def __init__(self, original_module):
        self.__dict__["_original_module"] = original_module

    def __getattr__(self, name):
        if name == "set_device":
            return self.set_device

        if hasattr(self._original_module, name):
            return getattr(self._original_module, name)

        if name == "memory_reserved":
            return lambda *args, **kwargs: 0
        elif name == "memory_allocated":
            return lambda *args, **kwargs: 0
        elif name == "max_memory_reserved":
            return lambda *args, **kwargs: 0
        elif name == "max_memory_allocated":
            return lambda *args, **kwargs: 0
        elif name == "reset_peak_memory_stats":
            return lambda *args, **kwargs: None
        elif name == "get_device_properties":

            class DummyDeviceProperties:
                def __init__(self, total_memory=DEFAULT_HBM_BYTES):
                    self.total_memory = total_memory
                    self.name = "Google TPU"
                    self.major = 1
                    self.minor = 0

            hbm_bytes = get_tpu_chip_hbm_bytes()
            total_mem = hbm_bytes if hbm_bytes > 0 else DEFAULT_HBM_BYTES
            return lambda *args, **kwargs: DummyDeviceProperties(total_memory=total_mem)
        elif name == "mem_get_info":
            hbm_bytes = get_tpu_chip_hbm_bytes()
            total_mem = hbm_bytes if hbm_bytes > 0 else DEFAULT_HBM_BYTES
            return lambda *args, **kwargs: (total_mem, total_mem)

        raise AttributeError(f"'TPUDeviceModuleProxy' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(self._original_module, name, value)

    def is_available(self) -> bool:
        if hasattr(self._original_module, "is_available"):
            try:
                return self._original_module.is_available()
            except Exception as e:
                logger.warning(f"torch.tpu.is_available() check failed: {e}")
                return False
        return False

    def set_device(self, device_index: Any) -> None:
        pass

    def current_device(self) -> int:
        if hasattr(self._original_module, "current_device"):
            try:
                return self._original_module.current_device()
            except Exception as e:
                logger.warning(f"torch.tpu.current_device() failed: {e}")
                return 0
        return 0

    def device_count(self) -> int:
        if hasattr(self._original_module, "device_count"):
            try:
                return self._original_module.device_count()
            except Exception as e:
                logger.warning(f"torch.tpu.device_count() failed: {e}")
                return 0
        return 0

    def synchronize(self, device_index: int | None = None) -> None:
        if hasattr(self._original_module, "synchronize"):
            try:
                self._original_module.synchronize()
            except Exception as e:
                logger.warning(f"torch.tpu.synchronize() failed: {e}")

    def empty_cache(self) -> None:
        if hasattr(self._original_module, "_clear_cache"):
            try:
                self._original_module._clear_cache()
            except Exception as e:
                logger.warning(f"Failed to clear TPU cache: {e}")


@PlatformRegistry.register(platform="tpu")
class PlatformTPU(PlatformCUDA):
    """Platform backend for Google TPUs (subclasses PlatformCUDA for API compatibility)."""

    def __init__(self):
        super().__init__()
        original_tpu = getattr(torch, "tpu", DummyTpuDeviceModule())
        self._device_module = TPUDeviceModuleProxy(original_tpu)

    @property
    def vendor_name(self) -> str:
        return "google"

    @property
    def device_name(self) -> str:
        return "tpu"

    @property
    def device_module(self):
        return self._device_module

    def current_device(self) -> int:
        return self.device_module.current_device()

    def device_count(self) -> int:
        return self.device_module.device_count()

    def set_device(self, device_index: int) -> None:
        self.device_module.set_device(device_index)

    def synchronize(self, device_index: int | None = None) -> None:
        self.device_module.synchronize()

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        self.device_module.manual_seed_all(seed)

    def is_available(self) -> bool:
        if hasattr(torch, "tpu"):
            try:
                return torch.tpu.is_available()
            except Exception as e:
                logger.warning(f"torch.tpu.is_available() check failed: {e}")
        return False

    def is_platform_available(self, use_smi_check=False) -> bool:
        if os.environ.get("VERL_PLATFORM") == "tpu":
            return True
        if "TPU_NAME" in os.environ or "TPU_VISIBLE_DEVICES" in os.environ:
            return True
        return False

    def ray_resource_name(self) -> str:
        return "TPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        tpu_chips = int(num_gpus)
        return {"resources": {"TPU": tpu_chips}} if tpu_chips >= 1 else {}

    def communication_backend_name(self) -> str:
        return "tpu_dist"

    def ray_noset_envvars(self) -> list[str]:
        return super().ray_noset_envvars() + [
            "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        ]

    def get_tpu_env_vars(
        self,
        rank: int,
        world_size: int,
        local_rank: int,
        local_world_size: int,
        name_prefix: str,
        pgs: list,
    ) -> dict[str, str]:
        """Generates TPU-specific distributed environment variables for PJRT mesh initialization."""
        node_ip_map = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes() if node.get("Alive", False)}
        bundle_ips = []
        local_ip = ray.util.get_node_ip_address()
        clean_prefix = name_prefix.lower().split("_")[0] if name_prefix else ""
        matching_pgs = []

        # 1. Primary filter: Select placement group containing current worker's node IP
        for p in pgs:
            specs = ray._private.state.state.placement_group_table(p.id)
            if specs.get("state") != "CREATED":
                continue
            bundles_map = specs.get("bundles_to_node_id", {})
            pg_ips = [node_ip_map[node_id] for b_idx, node_id in sorted(bundles_map.items()) if node_id in node_ip_map]
            if local_ip in pg_ips:
                matching_pgs.append(p)

        # 2. Secondary fallback: Filter by clean_prefix if placement group names are explicitly set
        if not matching_pgs and clean_prefix:
            for p in pgs:
                p_name = ray._private.state.state.placement_group_table(p.id).get("name", "").lower()
                if clean_prefix in p_name:
                    matching_pgs.append(p)

        target_pgs = matching_pgs if matching_pgs else pgs

        for pg in target_pgs:
            specs = ray._private.state.state.placement_group_table(pg.id)
            if specs.get("state") != "CREATED":
                continue
            bundles_map = specs.get("bundles_to_node_id", {})
            for b_idx in sorted(bundles_map.keys()):
                node_id = bundles_map[b_idx]
                if node_id in node_ip_map:
                    bundle_ips.append(node_ip_map[node_id])

        is_rollout = "rollout" in name_prefix.lower()
        base_port = ROLLOUT_BASE_PORT if is_rollout else TRAINER_BASE_PORT

        sb_addresses = [f"{ip}:{base_port + (b_idx % local_world_size)}" for b_idx, ip in enumerate(bundle_ips)]

        # Extract unique worker hostnames preserving rank order
        unique_hostnames = list(dict.fromkeys(bundle_ips))

        env_vars = {
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(sb_addresses),
            "TPU_PROCESS_ADDRESSES": ",".join(sb_addresses),
            "TPU_PROCESS_PORT": str(base_port + local_rank),
            "CLOUD_TPU_TASK_ID": str(rank // local_world_size),
            "TPU_WORKER_HOSTNAMES": ",".join(unique_hostnames),
            # One worker process drives exactly one addressable device. On TPU7x
            # that is a single chiplet, so these index devices (0..7 on a 4-chip
            # host), not chips. TPU_VISIBLE_DEVICES is torch_tpu's source of
            # truth; TPU_VISIBLE_CHIPS is kept in sync because libtpu gives it
            # priority when non-empty.
            "TPU_VISIBLE_CHIPS": str(local_rank),
            "TPU_VISIBLE_DEVICES": str(local_rank),
        }

        # Apply TPU topology and host bounds based on the detected TPU generation.
        #
        # ``world_size`` counts worker processes, i.e. addressable devices. The
        # generation decides how those map onto physical chips: a v6e slice with
        # 8 devices is 8 chips across 2 hosts ("2,4,1"), whereas a TPU7x slice
        # with 8 devices is 4 chips on a single host ("2,2,1,2").
        tpu_nodes = [node for node in ray.nodes() if "TPU" in node.get("Resources", {}) and node.get("Alive")]
        tpu_type = tpu_nodes[0].get("Labels", {}).get("ray.io/tpu-pod-type", "") if tpu_nodes else ""

        generation = detect_tpu_generation(tpu_type)
        topo = get_torch_tpu_topology(
            num_devices=world_size,
            generation=generation,
            chip_mesh=detect_slice_chip_mesh(),
        )

        env_vars.update(
            {
                "TORCH_TPU_TOPOLOGY": topo,
                "TPU_HOST_BOUNDS": topo,
                # Arity must match the topology string: "1,1,1,1" for the 4D
                # TPU7x mesh, "1,1,1" otherwise. Every dimension is 1 because
                # each process owns a single device.
                "TPU_CHIPS_PER_HOST_BOUNDS": get_chips_per_host_bounds(topo),
                "CHIPS_PER_HOST": str(get_devices_per_host(generation)),
            }
        )

        if is_rollout:
            env_vars.update(
                {
                    "SKIP_JAX_PRECOMPILE": "1",
                    "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
                }
            )
            if world_size > 1:
                env_vars["TPU_MULTIHOST_BACKEND"] = "ray"

        return env_vars

    def auto_assign_accelerator_type(self, name_prefix: str, accelerator_type: Optional[str]) -> Optional[str]:
        """Dynamically assign a TPU slice/group affinity to a resource pool on multi-slice clusters."""
        if accelerator_type is not None:
            return accelerator_type

        try:
            if ray.is_initialized():
                tpu_slices = set()
                for node in ray.nodes():
                    if node.get("Alive"):
                        for res in node.get("Resources", {}).keys():
                            if res.startswith("tpu-group-"):
                                tpu_slices.add(res)
                tpu_slices = sorted(list(tpu_slices))
                if len(tpu_slices) >= 2:
                    if any(k in name_prefix.lower() for k in ["rollout", "reward", "teacher"]):
                        return tpu_slices[1]
                    elif any(k in name_prefix.lower() for k in ["trainer", "actor", "global"]):
                        return tpu_slices[0]
        except Exception:
            pass

        return accelerator_type

    def configure_placement_group_bundle(
        self, bundle: dict, use_gpu: bool, device_name: str, name_prefix: str, accelerator_type: Optional[str] = None
    ) -> None:
        """Configure placement group bundle resources to prevent vLLM resource lockups on GKE TPU."""
        is_rollout_pool = any(k in name_prefix.lower() for k in ["rollout", "reward", "teacher"])
        if use_gpu and not is_rollout_pool:
            bundle[device_name] = 1
        if accelerator_type is not None:
            bundle[accelerator_type] = 1e-4

    def get_worker_env_vars(
        self,
        resource_pool,
        rank: int,
        world_size: int,
        local_rank: int,
        local_world_size: int,
        name_prefix: str,
        device_name: str,
    ) -> dict[str, str]:
        """Return platform-specific TPU environment variables for worker nodes."""
        env_vars = {}
        # Driver-side settings that must reach the worker actors. A launch
        # script's `export` happens in the driver process, i.e. after the Ray
        # job's runtime_env is already fixed, so without copying them here the
        # workers silently run with none of the XLA tuning flags the script set.
        for var in (
            "VERL_PLATFORM",
            "VERL_TPU_GENERATION",
            "LIBTPU_INIT_ARGS",
            "TORCH_TPU_INTERNAL_XLA_OPTIONS",
        ):
            if var in os.environ:
                env_vars[var] = os.environ[var]
        for var in self.ray_noset_envvars():
            env_vars[var] = "1"
        pgs = resource_pool.get_placement_groups(device_name=device_name)
        tpu_env = self.get_tpu_env_vars(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            local_world_size=local_world_size,
            name_prefix=name_prefix,
            pgs=pgs,
        )
        env_vars.update(tpu_env)
        return env_vars

    def sanitize_metrics(self, metrics: Any) -> Any:
        """Convert any TPU tensor in metrics to a standard Python scalar before Ray RPC transfer."""
        return convert_tensors_to_scalars(metrics)

    def get_ray_init_kwargs(self) -> dict[str, Any]:
        """Return Ray initialization arguments with runtime_env configured for GKE TPU workers."""
        return {
            "runtime_env": {
                "worker_process_setup_hook": patch_ray_worker,
            }
        }


# ``platform_manager`` imports this module unconditionally, and every process
# that touches ``get_platform()`` -- including the vLLM engine workers Ray
# starts for the rollout -- executes it. That makes this the one place we know
# runs inside those workers, so it is where we guarantee torch_tpu's absl flags
# are readable before anything reaches torch.compile. The guard keeps the call
# off non-TPU deployments, which import this module too.
if os.environ.get("VERL_PLATFORM", "").strip().lower() == "tpu" or "torch_tpu" in sys.modules:
    ensure_absl_flags_parsed()
