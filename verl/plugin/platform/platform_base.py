# Copyright (c) 2026 BAAI. All rights reserved.
# Adopted from https://github.com/microsoft/DeepSpeed/blob/master/accelerator/abstract_accelerator.py
"""Abstract base class defining the platform interface for device backends.

To add support for a new chip/accelerator, subclass ``PlatformBase`` and
implement all abstract methods.  Then register the platform name in
``platform_manager.py`` so that auto-detection or ``VERL_PLATFORM`` can
pick it up.
"""

import abc
import os
import shutil
import subprocess
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional


class PlatformBase(abc.ABC):
    """Hardware-agnostic interface for accelerator backends.

    Every concrete platform (CUDA, NPU, CPU, XPU, …) must implement the
    methods below so that the rest of the verl codebase can remain
    device-agnostic.

    For profiling methods (``nvtx_range``, ``profiler_start``, ``profiler_stop``),
    platforms that do not support profiling should implement them as no-ops.
    """

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @staticmethod
    def check_smi_command(cmd: str) -> bool:
        """Run an SMI command (e.g. nvidia-smi, mx-smi) and return True if it exits successfully.

        Useful for CUDA-compatible hardware that needs to be distinguished
        from NVIDIA during auto-detection.
        """
        cmd_path = shutil.which(cmd)
        if cmd_path is None:
            # Fallback to common absolute paths if not found in PATH
            common_paths = [
                f"/usr/bin/{cmd}",
                f"/usr/local/bin/{cmd}",
                f"/usr/local/cuda/bin/{cmd}",
            ]
            for path in common_paths:
                if os.path.isfile(path) and os.access(path, os.X_OK):
                    cmd_path = path
                    break
            if cmd_path is None:
                return False
        try:
            result = subprocess.run([cmd_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    @property
    @abc.abstractmethod
    def device_name(self) -> str:
        """Return the device type string (e.g. ``'cuda'``, ``'npu'``, ``'cpu'``)."""
        ...

    @property
    @abc.abstractmethod
    def vendor_name(self) -> str:
        """Return the hardware vendor name (e.g. ``'nvidia'``, ``'metax'``, ``'huawei'``, ``'intel'``)."""
        ...

    @property
    @abc.abstractmethod
    def device_module(self) -> ModuleType:
        """Return the ``torch.<device>`` namespace module (e.g. ``torch.cuda``)."""
        ...

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if the accelerator device is visible and usable in this process."""
        ...

    def is_platform_available(self, use_smi_check=False) -> bool:
        """Return ``True`` if this platform is available on this host.

        Used during auto-detection to determine if the environment targets
        this platform.  When ``use_smi_check=True``, may use relaxed checks
        (e.g. package importability, system commands) that work even in
        processes without device visibility (CPU-only Ray actors).

        Default implementation delegates to ``is_available()``.  Subclasses
        can override to provide more sophisticated detection logic.
        """
        return self.is_available()

    @abc.abstractmethod
    def current_device(self) -> int:
        """Return the index of the currently selected device."""
        ...

    @abc.abstractmethod
    def device_count(self) -> int:
        """Return the number of available devices of this type."""
        ...

    @abc.abstractmethod
    def set_device(self, device_index: int) -> None:
        """Select the device at *device_index*."""
        ...

    @abc.abstractmethod
    def synchronize(self, device_index: Optional[int] = None) -> None:
        """Block until all pending work on the device completes."""
        ...

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def manual_seed(self, seed: int) -> None:
        """Seed the current device's RNG."""
        ...

    @abc.abstractmethod
    def manual_seed_all(self, seed: int) -> None:
        """Seed **all** devices' RNG."""
        ...

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def set_allocator_settings(self, settings: str) -> None:
        """Configure the memory allocator (e.g. expandable segments)."""
        ...

    @abc.abstractmethod
    def empty_cache(self) -> None:
        """Release all unused cached memory."""
        ...

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        """Return ``(major, minor)`` compute capability, or ``(None, None)``."""
        ...

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def communication_backend_name(self) -> str:
        """Return the default collective-communication backend name (e.g. ``'nccl'``)."""
        ...

    @abc.abstractmethod
    def visible_devices_envvar(self) -> str:
        """Return the environment-variable name that controls visible devices."""
        ...

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    @abc.abstractmethod
    @contextmanager
    def nvtx_range(self, msg: str):
        """Context manager that wraps a block with an NVTX / profiler range.

        Platforms without profiling support should yield immediately (no-op).
        """
        ...

    @abc.abstractmethod
    def profiler_start(self) -> None:
        """Start the device profiler (no-op on unsupported platforms)."""
        ...

    @abc.abstractmethod
    def profiler_stop(self) -> None:
        """Stop the device profiler (no-op on unsupported platforms)."""
        ...

    # ------------------------------------------------------------------
    # vllm integration
    # ------------------------------------------------------------------
    def get_device_uuid(self, device_id):
        if os.getenv(self.visible_devices_envvar()) is not None:
            visible_devices = os.environ[self.visible_devices_envvar()].split(",")
            assert device_id < len(visible_devices), f"device_id {device_id} must less than {visible_devices}"
            return self.ray_resource_name() + visible_devices[device_id]
        else:
            return f"{self.ray_resource_name()}-{device_id}"

    def apply_model_patches(self, model_type: str) -> None:
        """Apply platform-specific model patches (e.g. replace ops unsupported on this device)."""
        return  # default no-op

    # ------------------------------------------------------------------
    # Ray integration
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def ray_resource_name(self) -> str:
        """Return the Ray accelerator resource name (e.g. ``'GPU'``, ``'NPU'``)."""
        ...

    @abc.abstractmethod
    def ray_noset_envvars(self) -> list[str]:
        """Return ``RAY_EXPERIMENTAL_NOSET_*`` env var names for this platform."""
        ...

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        """Return Ray actor resource options for allocating accelerators.

        CUDA uses ``{"num_gpus": N}`` while custom resources like NPU use
        ``{"resources": {"NPU": N}}``.  Subclasses may override for
        platform-specific behavior.
        """
        resource_name = self.ray_resource_name()
        if resource_name == "GPU":
            return {"num_gpus": num_gpus}
        return {"resources": {resource_name: num_gpus}}

    def ray_local_rank_override(self) -> Optional[str]:
        """Return the local rank of the current worker, or ``None`` to let Ray decide.

        verl normally reads the local rank from ``ray.get_runtime_context().get_accelerator_ids()``.
        That only works for accelerators Ray enumerates (``GPU``, ``NPU``, ...). A platform whose
        devices Ray does not enumerate individually can report the local rank here instead.
        """
        return None

    def supports_colocated_worker_groups(self) -> bool:
        """Return ``True`` if several WorkerGroups may be colocated on the same device.

        Platforms where one device is owned by exactly one process (so two WorkerGroups cannot
        share it) return ``False``; verl then caps ``max_colocate_count`` at 1.
        """
        return True

    def get_worker_env_vars(
        self,
        resource_pool: Any,
        rank: int,
        world_size: int,
        local_rank: int,
        local_world_size: int,
        name_prefix: str,
        device_name: str,
    ) -> dict[str, str]:
        """Return extra env vars to inject into a Ray worker's ``runtime_env``.

        Called by :class:`~verl.single_controller.ray.base.RayWorkerGroup` once per worker, after
        the placement groups exist and before the actor is created. Platforms that need the
        topology of the whole placement group to configure their runtime (mesh addresses, device
        index, ...) build those variables here.

        Args:
            resource_pool: The ``RayResourcePool`` the worker is created from.
            rank: Global rank of the worker.
            world_size: Total number of workers in the group.
            local_rank: Rank of the worker within its node.
            local_world_size: Number of workers on the worker's node.
            name_prefix: Name prefix of the WorkerGroup.
            device_name: Device name the WorkerGroup was created with.
        """
        return {}

    def get_ray_init_kwargs(self) -> dict[str, Any]:
        """Return extra kwargs to merge into ``ray.init()``.

        Platforms that need a ``runtime_env`` entry for every Ray worker in the job (for example a
        ``worker_process_setup_hook``) provide it here. The keys are merged into the kwargs verl
        builds from the config; ``runtime_env.env_vars`` is merged key by key.
        """
        return {}

    def requires_remote_driver(self) -> bool:
        """Return ``True`` if the training driver has to run inside a Ray actor on a worker node.

        Some clusters run the Ray driver on a head node that has no accelerator attached, while the
        device runtime has to be initialized from a node that owns devices. Those platforms return
        ``True`` and verl wraps the trainer in a Ray actor.
        """
        return False

    # ------------------------------------------------------------------
    # Collective communication capabilities
    # ------------------------------------------------------------------

    def supports_eager_collectives(self) -> bool:
        """Return ``True`` if ``torch.distributed`` collectives can be issued eagerly.

        Backends that trace and compile the execution graph (XLA/PJRT style) deadlock when a
        collective is issued outside the traced graph, which is what verl does to reduce metrics
        after a training step. Those platforms return ``False``, and verl aggregates the per-rank
        metrics on the driver instead.
        """
        return True

    # ------------------------------------------------------------------
    # IPC support
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def is_ipc_supported(self) -> bool:
        """Return ``True`` if the platform supports IPC for tensor sharing."""
        ...

    # ------------------------------------------------------------------
    # Rollout engine integration
    # ------------------------------------------------------------------

    def rollout_env_vars(self) -> dict[str, str]:
        """Return platform-specific env vars to inject when launching rollout engines."""
        return {}

    # ------------------------------------------------------------------
    # Collective communication
    # ------------------------------------------------------------------

    def get_collective_module(self) -> Any:
        """Return the collective communication module (e.g. ``cupy.cuda.nccl``).

        Returns ``None`` if not available. Subclasses should override.
        """
        return None

    # ------------------------------------------------------------------
    # Low-level runtime API
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def cudart(self) -> Any:
        """Return the CUDA runtime API object, or ``None`` if not applicable."""
        ...
