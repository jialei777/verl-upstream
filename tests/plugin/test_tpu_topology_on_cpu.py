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

"""CPU-only tests for the generation-aware TPU geometry helpers.

These cover the TPU7x (Ironwood) chip-vs-device distinction, which is the main
source of silent misconfiguration: a TPU7x chip exposes two addressable devices,
so device counts do not mean the same thing across generations.
"""

import importlib.util
import os
import pathlib

import pytest

_MODULE_PATH = pathlib.Path(__file__).resolve().parents[2] / "verl" / "plugin" / "platform" / "tpu_topology.py"


def _load_module():
    """Import tpu_topology by path so the test does not drag in torch/ray."""
    spec = importlib.util.spec_from_file_location("verl_tpu_topology_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tpu_topology = _load_module()


@pytest.mark.parametrize(
    "raw,expected",
    [
        # TPU7x spellings: GKE label, TPU_ACCELERATOR_TYPE, Ray labels, machine type.
        ("tpu7x", "v7x"),
        ("tpu7x-8", "v7x"),
        ("tpu7x-2x2x1", "v7x"),
        ("tpu7x-standard-4t", "v7x"),
        ("v7x-8", "v7x"),
        ("TPU-V7X", "v7x"),
        ("TPU v7", "v7x"),
        # Older generations must keep resolving exactly as before.
        ("tpu-v6e-slice", "v6e"),
        ("v6e-8", "v6e"),
        ("ct6e-standard-4t", "v6e"),
        ("tpu-v5-lite-podslice", "v5e"),
        ("v5litepod-8", "v5e"),
        ("tpu-v5p-slice", "v5p"),
        ("tpu-v4-podslice", "v4"),
        # Non-TPU / unknown strings must not be guessed.
        ("", None),
        (None, None),
        ("nvidia-h100", None),
    ],
)
def test_normalize_tpu_generation(raw, expected):
    assert tpu_topology.normalize_tpu_generation(raw) == expected


def test_v7x_geometry_differs_from_v6e():
    assert tpu_topology.get_devices_per_chip("v7x") == 2
    assert tpu_topology.get_devices_per_chip("v6e") == 1
    assert tpu_topology.get_devices_per_host("v7x") == 8
    assert tpu_topology.get_devices_per_host("v6e") == 4
    # Per *device*, not per chip: a TPU7x chip has 192 GiB split across two chiplets.
    assert tpu_topology.get_hbm_bytes_per_device("v7x") == 96 * 1024**3
    assert tpu_topology.get_hbm_bytes_per_device("v6e") == 32 * 1024**3


@pytest.mark.parametrize(
    "generation,num_devices,expected",
    [
        # The trap: 8 devices is two v6e hosts but a single tpu7x host.
        ("v6e", 8, "2,4,1"),
        ("v7x", 8, "2,2,1,2"),
        ("v6e", 4, "2,2,1"),
        ("v7x", 4, "1,2,1,2"),
        ("v6e", 32, "4,8,1"),
        ("v7x", 16, "2,2,2,2"),
        ("v7x", 2, "1,1,1,2"),
    ],
)
def test_get_torch_tpu_topology(generation, num_devices, expected):
    assert tpu_topology.get_torch_tpu_topology(num_devices, generation) == expected


@pytest.mark.parametrize("generation", ["v4", "v5e", "v5p", "v6e", "v7x"])
def test_topology_product_always_equals_device_count(generation):
    """torch_tpu hard-requires prod(topology) == WORLD_SIZE, including the 4th dim."""
    for num_devices in tpu_topology.get_known_device_counts(generation):
        topology = tpu_topology.get_torch_tpu_topology(num_devices, generation)
        assert tpu_topology.topology_num_devices(topology) == num_devices, topology


@pytest.mark.parametrize("num_devices", [3, 6, 12, 100])
def test_unlisted_device_counts_still_produce_a_valid_mesh(num_devices):
    for generation in ("v6e", "v7x"):
        topology = tpu_topology.get_torch_tpu_topology(num_devices, generation)
        assert tpu_topology.topology_num_devices(topology) == num_devices, topology


def test_chips_per_host_bounds_arity_matches_topology():
    # torch_tpu derives "1,1,1,1" for 4D topologies and "1,1,1" otherwise.
    assert tpu_topology.get_chips_per_host_bounds("2,2,1,2") == "1,1,1,1"
    assert tpu_topology.get_chips_per_host_bounds("2,4,1") == "1,1,1"


def test_topology_from_chip_mesh():
    # GKE advertises chips; TPU7x needs the chiplet count appended.
    assert tpu_topology.topology_from_chip_mesh("2x2x1", "v7x") == "2,2,1,2"
    assert tpu_topology.topology_from_chip_mesh("2x2x2", "v7x") == "2,2,2,2"
    # v6e meshes are 2D and get padded to the 3D form libtpu expects.
    assert tpu_topology.topology_from_chip_mesh("2x4", "v6e") == "2,4,1"


def test_chip_mesh_is_ignored_when_it_disagrees_with_the_device_count():
    """A sub-slice worker group must be sized from its own device count."""
    # The node advertises a whole 2x2x1 slice (8 devices) but only 4 are in use.
    assert tpu_topology.get_torch_tpu_topology(4, "v7x", chip_mesh="2x2x1") == "1,2,1,2"
    # When they agree, the physical wiring wins.
    assert tpu_topology.get_torch_tpu_topology(8, "v7x", chip_mesh="2x2x1") == "2,2,1,2"


def test_detect_generation_env_override(monkeypatch):
    monkeypatch.setenv("VERL_TPU_GENERATION", "v7x")
    assert tpu_topology.detect_tpu_generation() == "v7x"
    # An explicit hint still takes precedence over the escape hatch.
    assert tpu_topology.detect_tpu_generation("tpu-v6e-slice") == "v6e"


def test_detect_generation_falls_back_to_v6e(monkeypatch):
    """Unchanged default keeps existing v6e clusters working."""
    for var in ("VERL_TPU_GENERATION", "TPU_ACCELERATOR_TYPE", "ACCELERATOR_TYPE", "TPU_TYPE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(tpu_topology, "_ray_tpu_node_labels", dict)
    assert tpu_topology.detect_tpu_generation() == tpu_topology.DEFAULT_TPU_GENERATION == "v6e"


def test_detect_slice_chip_mesh_from_env(monkeypatch):
    monkeypatch.setattr(tpu_topology, "_ray_tpu_node_labels", dict)
    monkeypatch.setenv("TPU_TOPOLOGY", "2X2X1")
    assert tpu_topology.detect_slice_chip_mesh() == "2x2x1"
    monkeypatch.delenv("TPU_TOPOLOGY")
    assert tpu_topology.detect_slice_chip_mesh() is None


def test_detect_generation_reads_ray_labels(monkeypatch):
    for var in ("VERL_TPU_GENERATION", "TPU_ACCELERATOR_TYPE", "ACCELERATOR_TYPE", "TPU_TYPE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        tpu_topology,
        "_ray_tpu_node_labels",
        lambda: {"ray.io/accelerator-type": "TPU-V7X", "ray.io/tpu-pod-type": "v7x-8"},
    )
    assert tpu_topology.detect_tpu_generation() == "v7x"


def test_detect_generation_reads_cloud_env(monkeypatch):
    monkeypatch.delenv("VERL_TPU_GENERATION", raising=False)
    monkeypatch.setattr(tpu_topology, "_ray_tpu_node_labels", dict)
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "tpu7x-8")
    assert tpu_topology.detect_tpu_generation() == "v7x"


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-v"]))
