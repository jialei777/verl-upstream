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
HBM_BYTES_TPU_V7X = 192 * 1024 * 1024 * 1024  # 192 GB

TPU_HBM_BYTES_MAP = {
    "v5p": HBM_BYTES_TPU_V5P,
    "v6e": HBM_BYTES_TPU_V6E,
    "v7x": HBM_BYTES_TPU_V7X,
}

# TPU default 3D mesh topology mappings by pod type or total chips
TPU_TOPOLOGY_MAP = {
    "v6e-32": "4,8,1",
    "v6e-8": "2,4,1",
    "v6e-4": "2,2,1",
    32: "4,8,1",
    8: "2,4,1",
    4: "2,2,1",
}


def get_tpu_chip_hbm_bytes() -> int:
    """Detects the TPU chip generation from Ray node labels or environment variables and returns its HBM capacity."""
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

    for chip_gen, hbm_bytes in TPU_HBM_BYTES_MAP.items():
        if chip_gen in tpu_type:
            return hbm_bytes

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
                def __init__(self, total_memory=32 * 1024 * 1024 * 1024):
                    self.total_memory = total_memory
                    self.name = "Google TPU"
                    self.major = 1
                    self.minor = 0

            hbm_bytes = get_tpu_chip_hbm_bytes()
            total_mem = hbm_bytes if hbm_bytes > 0 else 32 * 1024 * 1024 * 1024
            return lambda *args, **kwargs: DummyDeviceProperties(total_memory=total_mem)
        elif name == "mem_get_info":
            hbm_bytes = get_tpu_chip_hbm_bytes()
            total_mem = hbm_bytes if hbm_bytes > 0 else 32 * 1024 * 1024 * 1024
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
    """In-tree PlatformTPU gutted on jialei/tpu-plugin-gut-test to verify verl-hardware-plugin override."""
    def __init__(self, *args, **kwargs):
        raise RuntimeError("In-tree PlatformTPU was instantiated! Expected verl_hardware_plugin.platforms.platform_tpu.PlatformTPU.")
