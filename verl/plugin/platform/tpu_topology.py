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
"""TPU generation detection and device-geometry helpers.

Cloud TPU generations do not share a single device geometry, and the distributed
bootstrap that ``torch_tpu`` (and therefore TorchTitan training and the vLLM TPU
rollout) expects is derived directly from it:

* Up to and including TPU v6e, a chip exposes exactly **one** addressable device.
  ``world_size == number of chips`` and the mesh string is 3D (``"x,y,z"``).
* TPU7x (Ironwood, a.k.a. ``v7x``) is a dual-chiplet part with MegaCore disabled.
  Each chip exposes **two** separately addressable devices, each owning one
  TensorCore and half of the chip's 192 GiB of HBM. ``world_size == 2 * chips``
  and the mesh string gains a fourth "chiplet" dimension (``"x,y,z,2"``).

verl runs exactly one worker process per addressable device, so every quantity
in this module is expressed in **devices**. "Chips" only appear where a
Cloud/GKE-facing value is involved (e.g. the ``2x2x1`` value of the
``cloud.google.com/gke-tpu-topology`` node label, which always counts chips).

Concretely, a single-host ``2x2x1`` TPU7x slice is 4 chips but **8 devices**,
so it needs ``TORCH_TPU_TOPOLOGY="2,2,1,2"`` and 8 worker processes, whereas a
single-host ``2x2`` v6e slice is 4 chips, 4 devices, ``TORCH_TPU_TOPOLOGY="2,2,1"``
and 4 worker processes.

References:
  * https://cloud.google.com/tpu/docs/tpu7x — "Dual-chiplet architecture"
  * ``torch_tpu._internal.utils.hardware`` — ``_V7_TOPOLOGY``, ``_DEVICES_PER_CHIP``
  * ``torch_tpu`` ``csrc/common/environment.cc`` — 3D vs 4D topology handling
"""

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_GIB = 1024 * 1024 * 1024

# Canonical (normalized) TPU generation identifiers used as keys throughout verl.
TPU_V4 = "v4"
TPU_V5E = "v5e"
TPU_V5P = "v5p"
TPU_V6E = "v6e"
TPU_V7X = "v7x"

# Generation assumed when auto-detection fails. v6e keeps the historical verl
# behaviour for clusters that do not surface any accelerator metadata.
DEFAULT_TPU_GENERATION = TPU_V6E

# Substring patterns matched (in order) against a lower-cased accelerator string.
# Ordering matters: the most specific spelling of a generation must come first so
# that e.g. "tpu-v7x" is not shortened to "v7" and "v6e" is not matched by "v6".
#
# Accepted spellings seen in the wild:
#   "tpu7x", "tpu7x-8", "tpu7x-2x2x1"  GKE label / TPU_ACCELERATOR_TYPE / xpk
#   "v7x-8", "TPU-V7X"                 ray.io/tpu-pod-type, ray.io/accelerator-type
#   "TPU v7"                           torch_tpu.get_tpu_device_name()
#   "tpu-v6e-slice", "v6e-8", "TPU-V6E"
#   "tpu-v5-lite-podslice", "v5litepod-8"  GKE label / gcloud accelerator type
#   "tpu7x-standard-4t", "ct6e-standard-4t", "ct5lp-hightpu-4t"  GCE machine types
_GENERATION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("tpu7x", TPU_V7X),
    ("v7x", TPU_V7X),
    ("tpu v7", TPU_V7X),
    ("v7", TPU_V7X),
    ("ct6e", TPU_V6E),
    ("v6e", TPU_V6E),
    ("v6", TPU_V6E),
    ("ct5lp", TPU_V5E),
    ("v5litepod", TPU_V5E),
    ("v5-lite", TPU_V5E),
    ("v5e", TPU_V5E),
    ("ct5p", TPU_V5P),
    ("v5p", TPU_V5P),
    ("ct4p", TPU_V4),
    ("v4", TPU_V4),
)

# Separately addressable devices exposed by one physical chip. Only generations
# that expose more than one are listed; everything else (including the MegaCore
# generations v4/v5p, which fuse their two cores into a single device) is 1.
_DEVICES_PER_CHIP: dict[str, int] = {
    TPU_V7X: 2,
}
DEFAULT_DEVICES_PER_CHIP = 1

# Addressable devices attached to a single host VM.
#   v6e   -> ct6e-standard-4t, 4 chips x 1 device
#   TPU7x -> tpu7x-standard-4t, 4 chips x 2 chiplets = 8 devices
_DEVICES_PER_HOST: dict[str, int] = {
    TPU_V7X: 8,
}
DEFAULT_DEVICES_PER_HOST = 4

# HBM addressable by a single device (NOT per chip). On TPU7x the chip's 192 GiB
# is split into two independent 96 GiB memory spaces, one per chiplet, so the
# per-device figure is the one that matters for memory accounting and for the
# vLLM KV-cache/`gpu_memory_utilization` math.
_HBM_BYTES_PER_DEVICE: dict[str, int] = {
    TPU_V4: 32 * _GIB,
    TPU_V5E: 16 * _GIB,
    TPU_V5P: 95 * _GIB,
    TPU_V6E: 32 * _GIB,
    TPU_V7X: 96 * _GIB,
}
DEFAULT_HBM_BYTES_PER_DEVICE = 32 * _GIB

# ``TORCH_TPU_TOPOLOGY`` strings keyed by **device count**. The product of the
# dimensions always equals the device count (and therefore ``WORLD_SIZE``).
#
# The v7x table extends ``torch_tpu``'s single-host ``_V7_TOPOLOGY`` with the
# multi-host slices published for Ironwood; the fourth element is always 2.
_TOPOLOGY_BY_DEVICE_COUNT: dict[str, dict[int, str]] = {
    # 3D torus generations, one device per chip.
    TPU_V4: {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,2,2", 16: "2,2,4", 32: "2,4,4", 64: "4,4,4"},
    TPU_V5P: {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,2,2", 16: "2,2,4", 32: "2,4,4", 64: "4,4,4"},
    # 2D mesh generations, one device per chip (padded to the 3D form libtpu wants).
    TPU_V5E: {1: "1,1,1", 4: "2,2,1", 8: "2,4,1", 16: "4,4,1", 32: "4,8,1", 64: "8,8,1", 128: "8,16,1"},
    TPU_V6E: {1: "1,1,1", 4: "2,2,1", 8: "2,4,1", 16: "4,4,1", 32: "4,8,1", 64: "8,8,1", 128: "8,16,1"},
    # 3D torus, two devices per chip. Keys are devices, i.e. 2 x chips.
    TPU_V7X: {
        2: "1,1,1,2",  # 1 chip
        4: "1,2,1,2",  # 2 chips
        8: "2,2,1,2",  # 4 chips, one full tpu7x-standard-4t host
        16: "2,2,2,2",  # 8 chips, 2 hosts
        32: "2,2,4,2",  # 16 chips, 4 hosts
        64: "2,4,4,2",  # 32 chips, 8 hosts
        128: "4,4,4,2",  # 64 chips, 16 hosts (one cube)
        256: "4,4,8,2",
        512: "4,8,8,2",
        1024: "8,8,8,2",
    },
}


def normalize_tpu_generation(raw: Optional[str]) -> Optional[str]:
    """Normalize any Cloud/GKE/Ray/torch_tpu accelerator string to a verl generation key.

    Args:
        raw: An accelerator description such as ``"tpu7x-8"``, ``"TPU-V7X"``,
            ``"tpu-v6e-slice"``, ``"v6e-8"`` or ``"TPU v7"``.

    Returns:
        One of ``"v4" | "v5e" | "v5p" | "v6e" | "v7x"``, or ``None`` if ``raw``
        does not name a recognized TPU generation.
    """
    if not raw:
        return None
    lowered = str(raw).strip().lower()
    for pattern, generation in _GENERATION_PATTERNS:
        if pattern in lowered:
            return generation
    return None


def _ray_tpu_node_labels() -> dict[str, str]:
    """Return the accelerator labels of an arbitrary alive TPU node, or ``{}``."""
    try:
        import ray

        if not ray.is_initialized():
            return {}
        for node in ray.nodes():
            if node.get("Alive") and "TPU" in node.get("Resources", {}):
                return node.get("Labels", {}) or {}
    except Exception as e:  # pragma: no cover - depends on a live Ray cluster
        logger.warning(f"Unable to query Ray node labels for TPU metadata: {e}")
    return {}


def detect_tpu_generation(hint: Optional[str] = None) -> str:
    """Detect the TPU generation of the current cluster.

    Resolution order (first match wins):
      1. An explicit ``hint`` (e.g. a Ray pod-type label the caller already read).
      2. The ``VERL_TPU_GENERATION`` escape hatch.
      3. Ray node labels (``ray.io/accelerator-type``, ``ray.io/tpu-pod-type``).
      4. Cloud TPU environment variables (``TPU_ACCELERATOR_TYPE`` and friends).
      5. ``torch_tpu``'s PCI-based probe, which only works on a machine that
         actually has chips attached.

    Returns:
        A normalized generation key, falling back to :data:`DEFAULT_TPU_GENERATION`.
    """
    generation = normalize_tpu_generation(hint)
    if generation:
        return generation

    generation = normalize_tpu_generation(os.environ.get("VERL_TPU_GENERATION"))
    if generation:
        return generation

    labels = _ray_tpu_node_labels()
    for key in ("ray.io/accelerator-type", "ray.io/tpu-pod-type"):
        generation = normalize_tpu_generation(labels.get(key))
        if generation:
            return generation

    for env_var in ("TPU_ACCELERATOR_TYPE", "ACCELERATOR_TYPE", "TPU_TYPE"):
        generation = normalize_tpu_generation(os.environ.get(env_var))
        if generation:
            return generation

    try:
        from torch_tpu._internal.utils import hardware

        generation = normalize_tpu_generation(hardware.get_tpu_device_name())
        if generation:
            return generation
    except Exception:
        pass

    logger.warning(
        f"Unable to detect the TPU generation; assuming '{DEFAULT_TPU_GENERATION}'. "
        "Set VERL_TPU_GENERATION (e.g. 'v7x') to override."
    )
    return DEFAULT_TPU_GENERATION


def detect_slice_chip_mesh() -> Optional[str]:
    """Return the chip mesh of the current slice (e.g. ``"2x2x1"``), if advertised.

    This is the ``cloud.google.com/gke-tpu-topology`` value as surfaced by the
    ``ray.io/tpu-topology`` node label or the ``TPU_TOPOLOGY`` environment
    variable. It always counts **chips**, never devices.
    """
    mesh = _ray_tpu_node_labels().get("ray.io/tpu-topology")
    if not mesh:
        mesh = os.environ.get("TPU_TOPOLOGY")
    return mesh.strip().lower() if mesh else None


def get_devices_per_chip(generation: Optional[str] = None) -> int:
    """Number of separately addressable devices exposed by one physical chip."""
    return _DEVICES_PER_CHIP.get(detect_tpu_generation(generation), DEFAULT_DEVICES_PER_CHIP)


def get_devices_per_host(generation: Optional[str] = None) -> int:
    """Number of addressable devices attached to a single host VM."""
    return _DEVICES_PER_HOST.get(detect_tpu_generation(generation), DEFAULT_DEVICES_PER_HOST)


def get_hbm_bytes_per_device(generation: Optional[str] = None) -> int:
    """HBM in bytes addressable by a single device (half a chip on TPU7x)."""
    return _HBM_BYTES_PER_DEVICE.get(detect_tpu_generation(generation), DEFAULT_HBM_BYTES_PER_DEVICE)


def parse_topology_dims(topology: str) -> list[int]:
    """Parse ``"2,2,1,2"`` / ``"2x2x1"`` into ``[2, 2, 1, 2]`` / ``[2, 2, 1]``."""
    return [int(dim) for dim in re.split(r"[x,]", str(topology).strip().lower()) if dim]


def topology_num_devices(topology: str) -> int:
    """Total device count described by a ``TORCH_TPU_TOPOLOGY`` string."""
    total = 1
    for dim in parse_topology_dims(topology):
        total *= dim
    return total


def topology_from_chip_mesh(chip_mesh: str, generation: Optional[str] = None) -> str:
    """Convert a Cloud/GKE chip mesh into a ``TORCH_TPU_TOPOLOGY`` string.

    Args:
        chip_mesh: The chip mesh advertised by GKE/Ray, e.g. ``"2x2x1"`` (TPU7x,
            3D) or ``"2x4"`` (v6e, 2D). Both ``x`` and ``,`` separators work.
        generation: Optional generation hint; auto-detected when omitted.

    Returns:
        A topology string whose dimension product equals the device count:
        ``"2x2x1"`` on TPU7x becomes ``"2,2,1,2"`` (4 chips, 8 devices) while
        ``"2x4"`` on v6e becomes ``"2,4,1"`` (8 chips, 8 devices).
    """
    dims = parse_topology_dims(chip_mesh)
    if not dims:
        raise ValueError(f"Unable to parse TPU chip mesh {chip_mesh!r}")

    # libtpu always wants a 3D chip mesh; 2D generations (v5e/v6e) are padded.
    dims = (dims + [1, 1, 1])[:3]

    devices_per_chip = get_devices_per_chip(generation)
    if devices_per_chip > 1:
        # Multi-chiplet chips (TPU7x) carry the chiplet count as a 4th dimension.
        dims.append(devices_per_chip)
    return ",".join(str(dim) for dim in dims)


def get_torch_tpu_topology(
    num_devices: int,
    generation: Optional[str] = None,
    chip_mesh: Optional[str] = None,
) -> str:
    """Compute the ``TORCH_TPU_TOPOLOGY`` string for ``num_devices`` worker processes.

    ``torch_tpu`` requires the product of the topology dimensions to equal
    ``WORLD_SIZE``, i.e. one rank per addressable device.

    Args:
        num_devices: Number of worker processes / devices in the slice.
        generation: Optional generation hint; auto-detected when omitted.
        chip_mesh: Optional chip mesh from GKE/Ray (e.g. ``"2x2x1"``). Preferred
            when it is consistent with ``num_devices`` because it describes the
            real physical wiring rather than a canonical guess.

    Returns:
        A topology string such as ``"2,2,1,2"`` (TPU7x, 8 devices) or ``"2,4,1"``
        (v6e, 8 devices).
    """
    generation = detect_tpu_generation(generation)

    if chip_mesh:
        try:
            candidate = topology_from_chip_mesh(chip_mesh, generation)
        except ValueError:
            candidate = None
        # Only trust the node label when the worker group spans the whole slice;
        # otherwise verl is running on a sub-slice and we must size the mesh from
        # the actual device count instead.
        if candidate and topology_num_devices(candidate) == num_devices:
            return candidate

    table = _TOPOLOGY_BY_DEVICE_COUNT.get(generation, {})
    if num_devices in table:
        return table[num_devices]

    # Unlisted size: fall back to a 1D mesh, which is still a valid slice
    # description as long as the product matches WORLD_SIZE.
    devices_per_chip = _DEVICES_PER_CHIP.get(generation, DEFAULT_DEVICES_PER_CHIP)
    if devices_per_chip > 1 and num_devices % devices_per_chip == 0:
        fallback = f"{num_devices // devices_per_chip},1,1,{devices_per_chip}"
    else:
        fallback = f"{num_devices},1,1"
    logger.warning(
        f"No canonical TPU topology for {num_devices} devices on '{generation}'; "
        f"falling back to '{fallback}'. Set TORCH_TPU_TOPOLOGY explicitly to override."
    )
    return fallback


def get_known_device_counts(generation: Optional[str] = None) -> list[int]:
    """Return the device counts that have a canonical topology for ``generation``.

    Useful for pre-populating third-party ``{num_devices: topology}`` lookup
    tables (e.g. vLLM's) with generation-correct values.
    """
    generation = detect_tpu_generation(generation)
    return sorted(_TOPOLOGY_BY_DEVICE_COUNT.get(generation, {}))


def get_chips_per_host_bounds(topology: str) -> str:
    """Return the ``TPU_CHIPS_PER_HOST_BOUNDS`` value matching ``topology``.

    verl always drives exactly one device per process, so every dimension is 1;
    only the arity has to match the topology string, which is what ``torch_tpu``
    does in ``InitializeDistributedEnvironment``: ``"1,1,1,1"`` for a 4D (TPU7x)
    topology and ``"1,1,1"`` otherwise.
    """
    return "1,1,1,1" if len(parse_topology_dims(topology)) == 4 else "1,1,1"
