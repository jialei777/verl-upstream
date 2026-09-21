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
from typing import Any, Optional

import ray
import torch

from .platform_cuda import PlatformCUDA
from .platform_manager import PlatformRegistry, get_platform
from .platform_tpu_workarounds import convert_tensors_to_scalars, patch_ray_worker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Communication port defaults for TPU distributed slice builder meshes
ROLLOUT_BASE_PORT = 8070
TRAINER_BASE_PORT = 8471

# TPU Chip HBM capacities in bytes
HBM_BYTES_TPU_V5P = 95 * 1024 * 1024 * 1024  # 95 GB
HBM_BYTES_TPU_V6E = 32 * 1024 * 1024 * 1024  # 32 GB
HBM_BYTES_TPU_V7X = 192 * 1024 * 1024 * 1024  # 192 GB per dual-core chip (96 GB per TensorCore device)
HBM_BYTES_TPU_V7X_CORE = 96 * 1024 * 1024 * 1024  # 96 GB per logical TensorCore device

TPU_HBM_BYTES_MAP = {
    "v5p": HBM_BYTES_TPU_V5P,
    "v6e": HBM_BYTES_TPU_V6E,
    "tpu7x": HBM_BYTES_TPU_V7X_CORE,
    "v7x": HBM_BYTES_TPU_V7X_CORE,
    "7x": HBM_BYTES_TPU_V7X_CORE,
    "v7": HBM_BYTES_TPU_V7X_CORE,
}

# TPU default 3D mesh topology mappings (v6e / single-core TPUs) by pod type or total chips
TPU_TOPOLOGY_MAP = {
    "v6e-32": "4,8,1",
    "v6e-8": "2,4,1",
    "v6e-4": "2,2,1",
    32: "4,8,1",
    8: "2,4,1",
    4: "2,2,1",
    2: "1,2,1",
    1: "1,1,1",
}

# TPU 7x (Ironwood) 4D mesh topology mappings (X, Y, Z, CoresPerChip=2)
# A single-host 2x2x1 TPU 7x node has 4 physical chips x 2 TensorCores = 8 logical devices ("2,2,1,2").
TPU_V7X_TOPOLOGY_MAP = {
    "tpu7x-2x2x1": "2,2,1,2",
    "tpu7x-8": "2,2,1,2",
    "tpu7x-4": "2,2,1,2",
    "v7x-8": "2,2,1,2",
    "v7x-4": "2,2,1,2",
    "2x2x1": "2,2,1,2",
    16: "2,2,2,2",
    8: "2,2,1,2",
    4: "1,2,1,2",
    2: "1,1,1,2",
    1: "1,1,1,1",
}


def _is_node_tpu_v7x(node: dict) -> bool:
    """Checks whether a specific Ray node dictionary corresponds to a TPU 7x (Ironwood) host."""
    labels = node.get("Labels", {}) or {}
    for label_key in (
        "ray.io/accelerator-type",
        "ray.io/tpu-pod-type",
        "cloud.google.com/gke-tpu-accelerator",
        "cloud.google.com/gke-nodepool",
    ):
        val = str(labels.get(label_key, "")).lower()
        if any(k in val for k in ("tpu7x", "v7x", "7x", "v7", "ironwood")):
            return True

    for res_key in (node.get("Resources", {}) or {}).keys():
        res_lower = res_key.lower()
        if any(k in res_lower for k in ("tpu7x", "v7x", "tpu-v7", "ironwood")):
            return True

    return False


def is_tpu_v7x() -> bool:
    """Detects whether the current environment or Ray cluster is running on TPU 7x (Ironwood)."""
    # 1. Check explicit environment variables
    for env_key in ("TPU_ACCELERATOR_TYPE", "ACCELERATOR_TYPE", "TPU_TYPE", "GKE_TPU_ACCELERATOR"):
        val = os.environ.get(env_key, "").lower()
        if any(k in val for k in ("tpu7x", "v7x", "7x", "v7", "ironwood")):
            return True
        if any(k in val for k in ("v6e", "v5p", "v5e", "v4")):
            return False

    # 2. Check if a 4D topology is already configured in the environment
    for topo_key in ("TORCH_TPU_TOPOLOGY", "TPU_HOST_BOUNDS"):
        topo_val = os.environ.get(topo_key, "").strip()
        if topo_val and len(topo_val.split(",")) == 4:
            return True

    # 3. Check local PCI devices for TPU v7x (Google vendor 0x1ae0, device 0x0076)
    try:
        pci_root = "/sys/bus/pci/devices"
        if os.path.isdir(pci_root):
            for dev_dir in os.listdir(pci_root):
                vendor_path = os.path.join(pci_root, dev_dir, "vendor")
                device_path = os.path.join(pci_root, dev_dir, "device")
                if os.path.exists(vendor_path) and os.path.exists(device_path):
                    with open(vendor_path) as vf, open(device_path) as df:
                        vendor = vf.read().strip().lower()
                        device = df.read().strip().lower()
                    if vendor == "0x1ae0" and device == "0x0076":
                        return True
                    if vendor == "0x1ae0" and device in ("0x006f", "0x0062", "0x0063"):
                        return False
    except Exception:
        pass

    # 4. Check torch_tpu hardware detection if available
    try:
        from torch_tpu._internal.utils.hardware import detect_tpu_devices

        info = detect_tpu_devices()
        if info is not None and "v7" in str(getattr(info, "version", "")).lower():
            return True
    except Exception:
        pass

    # 5. Query Ray cluster node labels and resources
    try:
        if ray.is_initialized():
            tpu_nodes = [node for node in ray.nodes() if "TPU" in node.get("Resources", {}) and node.get("Alive")]
            if any(_is_node_tpu_v7x(node) for node in tpu_nodes):
                return True
    except Exception:
        pass

    return False


def get_tpu_chip_hbm_bytes() -> int:
    """Detects the TPU chip generation from Ray node labels, PCI IDs, or environment variables and returns HBM bytes."""
    tpu_type = ""

    # Query Ray cluster node labels for TPU resource type
    try:
        if ray.is_initialized():
            tpu_nodes = [node for node in ray.nodes() if "TPU" in node.get("Resources", {}) and node.get("Alive")]
            if tpu_nodes:
                labels = tpu_nodes[0].get("Labels", {})
                tpu_type = (
                    labels.get("ray.io/accelerator-type")
                    or labels.get("ray.io/tpu-pod-type")
                    or labels.get("cloud.google.com/gke-tpu-accelerator")
                    or ""
                ).lower()
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

    for chip_gen, hbm_bytes in TPU_HBM_BYTES_MAP.items():
        if chip_gen in tpu_type:
            return hbm_bytes

    if is_tpu_v7x():
        return HBM_BYTES_TPU_V7X_CORE

    logger.warning(f"Unable to determine TPU chip HBM bytes for tpu_type='{tpu_type}'. Returning -1.")
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

    def get_device_name(self, device: Any = None) -> str:
        return "TPU v7x" if is_tpu_v7x() else "TPU v6e"


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
        elif name == "get_device_name":
            return lambda *args, **kwargs: ("TPU v7x" if is_tpu_v7x() else "TPU v6e")
        elif name == "get_device_properties":

            class DummyDeviceProperties:
                def __init__(self, total_memory=32 * 1024 * 1024 * 1024):
                    self.total_memory = total_memory
                    self.name = "TPU v7x" if is_tpu_v7x() else "Google TPU"
                    self.major = 1
                    self.minor = 0

            hbm_bytes = get_tpu_chip_hbm_bytes()
            default_mem = HBM_BYTES_TPU_V7X_CORE if is_tpu_v7x() else 32 * 1024 * 1024 * 1024
            total_mem = hbm_bytes if hbm_bytes > 0 else default_mem
            return lambda *args, **kwargs: DummyDeviceProperties(total_memory=total_mem)
        elif name == "mem_get_info":
            hbm_bytes = get_tpu_chip_hbm_bytes()
            default_mem = HBM_BYTES_TPU_V7X_CORE if is_tpu_v7x() else 32 * 1024 * 1024 * 1024
            total_mem = hbm_bytes if hbm_bytes > 0 else default_mem
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

    def get_logical_device_count_on_node(self, node: dict) -> float:
        """Returns the number of addressable TPU devices (TensorCores) on a Ray node.

        On TPU 7x (Ironwood) 2x2x1 single-host nodes, GKE/KubeRay may register ``google.com/tpu: 4``
        (4 physical chips -> ``node['Resources']['TPU'] == 4.0``), while each chip exposes 2 separately
        addressable TensorCore devices (8 logical devices total per host).
        """
        tpu_res = float(node.get("Resources", {}).get("TPU", 0.0))
        if tpu_res <= 0:
            return 0.0
        if (_is_node_tpu_v7x(node) or is_tpu_v7x()) and tpu_res <= 4.0:
            return tpu_res * 2.0
        return tpu_res

    def _get_per_worker_tpu_resource(self, process_count: Optional[int] = None, default: float = 1.0) -> float:
        """Determines the Ray ``TPU`` resource quantity per worker to match node capacity.

        When a TPU 7x 2x2x1 node reports ``TPU: 4.0`` in Ray (4 physical chips) for 8 TensorCore workers,
        each worker bundle/actor requests ``0.5`` ``TPU`` so all 8 workers fit on the single host.
        """
        try:
            if ray.is_initialized():
                tpu_nodes = [node for node in ray.nodes() if "TPU" in node.get("Resources", {}) and node.get("Alive")]
                if tpu_nodes:
                    min_node_tpu = min(float(n["Resources"]["TPU"]) for n in tpu_nodes)
                    if process_count is not None and process_count > 0 and min_node_tpu < process_count:
                        return min_node_tpu / float(process_count)
                    if (any(_is_node_tpu_v7x(n) for n in tpu_nodes) or is_tpu_v7x()) and min_node_tpu <= 4.0:
                        return 0.5
        except Exception:
            pass
        return default

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        if num_gpus <= 0:
            return {}
        tpu_val = min(float(num_gpus), self._get_per_worker_tpu_resource(default=float(num_gpus)))
        if tpu_val == int(tpu_val):
            tpu_val = int(tpu_val)
        return {"resources": {"TPU": tpu_val}} if tpu_val > 0 else {}

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

        tpu_nodes = [node for node in ray.nodes() if "TPU" in node.get("Resources", {}) and node.get("Alive")]
        tpu_type = tpu_nodes[0].get("Labels", {}).get("ray.io/tpu-pod-type", "") if tpu_nodes else ""
        is_v7x = is_tpu_v7x() or any(_is_node_tpu_v7x(n) for n in tpu_nodes)

        # Support single-host split sharing on TPU 7x when only 1 TPU node exists and each role uses <= 4 devices
        device_offset = 0
        if is_v7x and is_rollout and len(tpu_nodes) == 1 and local_world_size <= 4:
            device_offset = int(os.environ.get("DEBUG_TPU_LOCAL_RANK_OFFSET", "4"))

        effective_local_rank = local_rank + device_offset

        env_vars = {
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(sb_addresses),
            "TPU_PROCESS_ADDRESSES": ",".join(sb_addresses),
            "TPU_PROCESS_PORT": str(base_port + local_rank),
            "CLOUD_TPU_TASK_ID": str(rank),
            "TPU_WORKER_HOSTNAMES": ",".join(unique_hostnames),
            "TPU_VISIBLE_CHIPS": str(effective_local_rank),
            "TPU_VISIBLE_DEVICES": str(effective_local_rank),
            "ALLOW_MULTIPLE_LIBTPU_LOAD": "1",
        }
        if device_offset > 0:
            env_vars["DEBUG_TPU_LOCAL_RANK_OFFSET"] = str(device_offset)

        # Apply TPU topology and host bounds based on TPU generation (4D for TPU 7x, 3D for v6e)
        if is_v7x:
            topo = TPU_V7X_TOPOLOGY_MAP.get(
                tpu_type, TPU_V7X_TOPOLOGY_MAP.get(world_size, "2,2,1,2" if world_size >= 8 else "1,2,1,2")
            )
            chips_bounds = "1,1,1,1"
            chips_per_host = str(max(local_world_size, 8 if world_size >= 8 else local_world_size))
        else:
            topo = TPU_TOPOLOGY_MAP.get(tpu_type, TPU_TOPOLOGY_MAP.get(world_size, "1,1,1"))
            chips_bounds = "1,1,1"
            chips_per_host = str(max(local_world_size, 4))

        env_vars.update(
            {
                "TORCH_TPU_TOPOLOGY": topo,
                "TPU_HOST_BOUNDS": topo,
                "TPU_PROCESS_BOUNDS": topo,
                "TPU_CHIPS_PER_HOST_BOUNDS": chips_bounds,
                "TPU_CHIPS_PER_PROCESS_BOUNDS": chips_bounds,
                "CHIPS_PER_HOST": chips_per_host,
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
        """Dynamically assign a TPU slice/group or node affinity to a resource pool on single/multi-slice clusters."""
        if accelerator_type is not None:
            return accelerator_type

        is_rollout_pool = any(k in name_prefix.lower() for k in ["rollout", "reward", "teacher"])

        try:
            if ray.is_initialized():
                tpu_slices = set()
                alive_tpu_nodes = []
                for node in ray.nodes():
                    if node.get("Alive") and "TPU" in node.get("Resources", {}):
                        alive_tpu_nodes.append(node)
                        for res in node.get("Resources", {}).keys():
                            if res.startswith("tpu-group-"):
                                tpu_slices.add(res)
                tpu_slices = sorted(list(tpu_slices))
                if len(tpu_slices) >= 1:
                    if len(tpu_slices) >= 2 and is_rollout_pool:
                        return tpu_slices[1]
                    return tpu_slices[0]

                # Fallback for single-host TPU slices (e.g., 2x2x1 TPU 7x) where KubeRay does not inject tpu-group-*:
                # Pin Trainer and Rollout pools to distinct TPU nodes via node:<ip> resources, avoiding ray-head.
                if alive_tpu_nodes:
                    alive_tpu_nodes.sort(
                        key=lambda n: (
                            n.get("Labels", {}).get("ray.io/tpu-slice-name", ""),
                            n.get("NodeManagerAddress", ""),
                            n.get("NodeID", ""),
                        )
                    )
                    target_node = (
                        alive_tpu_nodes[1] if (len(alive_tpu_nodes) >= 2 and is_rollout_pool) else alive_tpu_nodes[0]
                    )
                    node_ip = target_node.get("NodeManagerAddress")
                    node_res_key = f"node:{node_ip}"
                    if node_ip and node_res_key in target_node.get("Resources", {}):
                        return node_res_key
        except Exception:
            pass

        return accelerator_type

    def configure_placement_group_bundle(
        self,
        bundle: dict,
        use_gpu: bool,
        device_name: str,
        name_prefix: str,
        accelerator_type: Optional[str] = None,
        process_count: Optional[int] = None,
    ) -> None:
        """Configure placement group bundle resources to prevent vLLM resource lockups on GKE TPU."""
        is_rollout_pool = any(k in name_prefix.lower() for k in ["rollout", "reward", "teacher"])
        if use_gpu and not is_rollout_pool:
            tpu_val = self._get_per_worker_tpu_resource(process_count=process_count, default=1.0)
            bundle[device_name] = int(tpu_val) if tpu_val == int(tpu_val) else tpu_val
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
        if "VERL_PLATFORM" in os.environ:
            env_vars["VERL_PLATFORM"] = os.environ["VERL_PLATFORM"]
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

    def rollout_env_vars(self) -> dict[str, str]:
        """Return env vars to inject into the rollout engine.

        Overrides PlatformCUDA, whose NCCL_CUMEM_ENABLE is meaningless here.

        The rollout server actor is created with an explicit runtime_env, which
        replaces inheritance from the job, so compiler flags set on the driver
        have to be named to survive. The caller appends VERL_TPU_EXTRA_<VAR> and
        falls back to the value forwarded off the worker.
        """
        return {var: os.environ[var] for var in ("XLA_FLAGS", "LIBTPU_INIT_ARGS") if os.environ.get(var)}

    def get_ray_init_kwargs(self) -> dict[str, Any]:
        """Return Ray initialization arguments with runtime_env configured for GKE TPU workers."""
        return {
            "runtime_env": {
                "worker_process_setup_hook": patch_ray_worker,
                "env_vars": {"VERL_PLATFORM": "tpu"},
            }
        }
