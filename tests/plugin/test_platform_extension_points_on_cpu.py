# Copyright (c) 2026 Google LLC. All rights reserved.
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
"""CPU tests for the platform extension points a hardware plugin can implement."""

import torch

import verl.plugin.platform as platform_pkg
from verl.plugin.platform import get_platform
from verl.utils.metric import convert_tensors_to_scalars
from verl.utils.ray_utils import merge_platform_ray_init_kwargs
from verl.utils.tensordict_utils import concat_tensordict_with_none_bsz, get, get_tensordict


class _StubPlatform:
    """Minimal stand-in for a platform that cannot run collectives eagerly."""

    def __init__(self, ray_init_kwargs=None):
        self._ray_init_kwargs = ray_init_kwargs or {}

    def supports_eager_collectives(self) -> bool:
        return False

    def get_ray_init_kwargs(self) -> dict:
        return self._ray_init_kwargs


def test_platform_extension_points_default_to_no_op():
    platform = get_platform()
    assert platform.ray_local_rank_override() is None
    assert platform.supports_colocated_worker_groups() is True
    assert platform.supports_eager_collectives() is True
    assert platform.requires_remote_driver() is False
    assert platform.get_ray_init_kwargs() == {}
    assert (
        platform.get_worker_env_vars(
            resource_pool=None,
            rank=0,
            world_size=1,
            local_rank=0,
            local_world_size=1,
            name_prefix="wg",
            device_name=platform.device_name,
        )
        == {}
    )


def test_merge_platform_ray_init_kwargs_is_a_no_op_without_platform_kwargs():
    config_kwargs = {"runtime_env": {"env_vars": {"A": "1"}}, "num_cpus": 4}
    assert merge_platform_ray_init_kwargs(config_kwargs) == config_kwargs


def test_merge_platform_ray_init_kwargs_merges_without_mutating_the_input(monkeypatch):
    def setup_hook():
        return None

    stub = _StubPlatform(
        ray_init_kwargs={"runtime_env": {"worker_process_setup_hook": setup_hook, "env_vars": {"A": "0", "B": "2"}}}
    )
    monkeypatch.setattr(platform_pkg, "get_platform", lambda: stub)

    config_kwargs = {"runtime_env": {"env_vars": {"A": "1"}}}
    merged = merge_platform_ray_init_kwargs(config_kwargs)

    # config values win, platform values fill in the rest
    assert merged["runtime_env"]["env_vars"] == {"A": "1", "B": "2"}
    assert merged["runtime_env"]["worker_process_setup_hook"] is setup_hook
    assert config_kwargs == {"runtime_env": {"env_vars": {"A": "1"}}}


def test_convert_tensors_to_scalars_leaves_the_container_shape_alone():
    converted = convert_tensors_to_scalars(
        {"loss": torch.tensor(1.5), "per_micro_batch": [torch.tensor(1.0), 2.0], "name": "actor"}
    )
    assert converted["loss"] == 1.5
    assert converted["per_micro_batch"] == [1.0, 2.0]
    assert converted["name"] == "actor"


def test_metrics_are_merged_on_the_driver_without_eager_collectives(monkeypatch):
    per_rank = [
        get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": {"loss": 1.0, "pg": [1, 2]}}),
        get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": {"loss": 3.0, "pg": [3, 4]}}),
    ]

    # by default the workers all-gathered the metrics themselves, only the first dict is kept
    assert get(concat_tensordict_with_none_bsz(per_rank), "metrics") == {"loss": 1.0, "pg": [1, 2]}

    monkeypatch.setattr(platform_pkg, "get_platform", lambda: _StubPlatform())
    merged = get(concat_tensordict_with_none_bsz(per_rank), "metrics")
    assert merged == {"loss": [1.0, 3.0], "pg": [[1, 2], [3, 4]]}
