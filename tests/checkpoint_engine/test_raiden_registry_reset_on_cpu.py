# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Google LLC
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
"""setup_raiden_controller() must reset the detached TPUWeightRegistry before this job publishes anything."""

import sys
import types

import ray

from verl.checkpoint_engine import raiden_checkpoint_engine, tpu_weight_registry


class _RemoteMethod:
    def __init__(self, log: list, name: str):
        self.log, self.name = log, name

    def remote(self, *args, **kwargs):
        self.log.append((self.name, args))
        return None


def test_setup_raiden_controller_clears_stale_registry_first(monkeypatch):
    log = []
    registry = types.SimpleNamespace(
        clear=_RemoteMethod(log, "clear"),
        set_controller_address=_RemoteMethod(log, "set_controller_address"),
    )
    fake_controller_module = types.SimpleNamespace(
        RaidenController=lambda port: object(),
        RaidenControllerServer=lambda controller: types.SimpleNamespace(start=lambda: 4321),
    )
    tpu_sync_rpc = types.ModuleType("tpu_sync.rpc")
    tpu_sync_rpc.raiden_controller = fake_controller_module
    monkeypatch.setitem(sys.modules, "tpu_sync", types.ModuleType("tpu_sync"))
    monkeypatch.setitem(sys.modules, "tpu_sync.rpc", tpu_sync_rpc)
    monkeypatch.setattr(tpu_weight_registry, "get_tpu_weight_registry", lambda: registry)
    monkeypatch.setattr(ray, "get", lambda ref: ref)
    monkeypatch.setattr(ray.util, "get_node_ip_address", lambda: "10.0.0.1")

    _, _, address = raiden_checkpoint_engine.setup_raiden_controller()

    assert address == "10.0.0.1:4321"
    # A previous job's steps 6..10 left in the registry would otherwise evict this job's steps 1..5 parity stats.
    assert log == [("clear", ()), ("set_controller_address", ("10.0.0.1:4321",))]
