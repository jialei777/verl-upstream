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
"""CPU unit tests for local fused-QKV shard splitting during parameter export."""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard


def _load_torchtitan_engine_cls():
    """Import TorchTitanEngine without requiring the optional torchtitan package on CPU CI."""
    if "torchtitan" not in sys.modules:
        stubs = [
            "torchtitan",
            "torchtitan.components",
            "torchtitan.components.checkpoint",
            "torchtitan.components.loss",
            "torchtitan.components.lr_scheduler",
            "torchtitan.components.optimizer",
            "torchtitan.components.dataloader",
            "torchtitan.config",
            "torchtitan.distributed",
            "torchtitan.distributed.activation_checkpoint",
            "torchtitan.distributed.context_parallel",
            "torchtitan.distributed.parallel_dims",
            "torchtitan.models",
            "torchtitan.models.common",
            "torchtitan.models.common.attention",
            "torchtitan.train",
        ]
        for mod_name in stubs:
            sys.modules[mod_name] = types.ModuleType(mod_name)
        sys.modules["torchtitan.components.checkpoint"].CheckpointManager = MagicMock()
        sys.modules["torchtitan.components.loss"].CrossEntropyLoss = MagicMock()
        sys.modules["torchtitan.components.lr_scheduler"].LRSchedulersContainer = MagicMock()
        sys.modules["torchtitan.components.optimizer"].OptimizersContainer = MagicMock()
        sys.modules["torchtitan.components.optimizer"].ParamGroupConfig = MagicMock()

        class _BaseDataLoader:
            class Config:
                pass

        sys.modules["torchtitan.components.dataloader"].BaseDataLoader = _BaseDataLoader
        sys.modules["torchtitan.config"].CompileConfig = MagicMock()
        sys.modules["torchtitan.config"].DebugConfig = MagicMock()
        sys.modules["torchtitan.config"].ParallelismConfig = MagicMock()
        sys.modules["torchtitan.config"].TrainingConfig = MagicMock()
        sys.modules["torchtitan.distributed"].utils = MagicMock()
        sys.modules["torchtitan.distributed.activation_checkpoint"].FullAC = MagicMock()
        sys.modules["torchtitan.distributed.activation_checkpoint"].SelectiveAC = MagicMock()
        sys.modules["torchtitan.distributed.context_parallel"].prepare_context_parallel_input = MagicMock()
        sys.modules["torchtitan.distributed.parallel_dims"].ParallelDims = MagicMock()
        attn = sys.modules["torchtitan.models.common.attention"]
        attn.AttentionMasksType = object
        attn.VarlenMetadata = object
        attn.create_attention_mask = MagicMock()
        attn.get_causal_mask_mod = MagicMock()

        class _FakeFusedQKVLinear(nn.Module):
            @staticmethod
            def _split_qkv_on_save(module, state_dict, prefix, local_metadata):
                raise AssertionError("original _split_qkv_on_save should have been swapped out")

        attn.FusedQKVLinear = _FakeFusedQKVLinear
        sys.modules["torchtitan.train"].Trainer = MagicMock()

    from verl.workers.engine.torchtitan.transformer_impl import TorchTitanEngine

    return TorchTitanEngine


def _fake_dtensor(mesh_sizes: tuple[int, ...], placements: tuple):
    from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

    mesh = SimpleNamespace(ndim=len(mesh_sizes), size=lambda d: mesh_sizes[d])
    meta = TensorMeta(torch.Size([64, 4]), (4, 1), torch.float32)
    spec = DTensorSpec(mesh=mesh, placements=placements, tensor_meta=meta)
    return DTensor.__new__(DTensor, torch.zeros(1), spec, False)


def test_can_split_qkv_locally_checks_divisibility_and_placement():
    TorchTitanEngine = _load_torchtitan_engine_cls()

    # Plain Tensor: always local
    assert TorchTitanEngine._can_split_qkv_locally(torch.zeros(16, 4), n_kv=8)

    # Qwen3-8B (n_kv=8) on dp_shard=8 -> 8 % 8 == 0 -> local split
    dt_8 = _fake_dtensor((8,), (Shard(0),))
    assert TorchTitanEngine._can_split_qkv_locally(dt_8, n_kv=8)

    # HSDP (Replicate, Shard(0)) with shard degree 4 -> 8 % 4 == 0 -> local split
    dt_hsdp = _fake_dtensor((2, 4), (Replicate(), Shard(0)))
    assert TorchTitanEngine._can_split_qkv_locally(dt_hsdp, n_kv=8)

    # Qwen3-32B (n_kv=8) on dp_shard=32 -> 8 % 32 != 0 -> must all-gather
    dt_32 = _fake_dtensor((32,), (Shard(0),))
    assert not TorchTitanEngine._can_split_qkv_locally(dt_32, n_kv=8)

    # _StridedShard or Shard(1) -> must all-gather
    dt_strided = _fake_dtensor((4, 2), (Shard(0), _StridedShard(0, split_factor=2)))
    assert not TorchTitanEngine._can_split_qkv_locally(dt_strided, n_kv=8)
    dt_dim1 = _fake_dtensor((4,), (Shard(1),))
    assert not TorchTitanEngine._can_split_qkv_locally(dt_dim1, n_kv=8)


def test_split_qkv_on_save_local_matches_stock_split_and_avoids_redistribute():
    TorchTitanEngine = _load_torchtitan_engine_cls()

    # Qwen3-8B geometry scaled down in dim: n_kv=8, hpk=4, r=6, hd=128, dim=64
    n_kv, hpk, hd, dim = 8, 4, 128, 64
    r = hpk + 2
    module = SimpleNamespace(head_dim=hd, heads_per_kv=hpk, r_dim=r)
    full_wqkv = torch.arange(n_kv * r * hd * dim, dtype=torch.float32).reshape(n_kv * r * hd, dim)

    # Stock reference on the full tensor
    ref_sd = {"layers.0.wqkv.weight": full_wqkv.clone()}
    TorchTitanEngine._split_qkv_on_save_local(module, ref_sd, "layers.0.", {})
    assert ref_sd["layers.0.wq.weight"].shape == (n_kv * hpk * hd, dim)
    assert ref_sd["layers.0.wk.weight"].shape == (n_kv * hd, dim)
    assert ref_sd["layers.0.wv.weight"].shape == (n_kv * hd, dim)

    # Simulate each rank of 8: holds 1 KV group (1/8 of dim 0). Each rank's local slice of wqkv, split
    # locally, must equal that rank's Shard(0) chunk of the full wq/wk/wv (512 / 128 / 128 rows).
    for rank in range(8):
        local_wqkv = full_wqkv.chunk(8, dim=0)[rank].contiguous()
        sd = {"layers.0.wqkv.weight": local_wqkv}
        TorchTitanEngine._split_qkv_on_save_local(module, sd, "layers.0.", {})
        assert sd["layers.0.wq.weight"].shape == (512, dim)
        assert sd["layers.0.wk.weight"].shape == (128, dim)
        assert sd["layers.0.wv.weight"].shape == (128, dim)
        assert torch.equal(sd["layers.0.wq.weight"], ref_sd["layers.0.wq.weight"].chunk(8, dim=0)[rank])
        assert torch.equal(sd["layers.0.wk.weight"], ref_sd["layers.0.wk.weight"].chunk(8, dim=0)[rank])
        assert torch.equal(sd["layers.0.wv.weight"], ref_sd["layers.0.wv.weight"].chunk(8, dim=0)[rank])


def test_get_per_tensor_param_shard_swaps_and_restores_instance_hook(monkeypatch):
    TorchTitanEngine = _load_torchtitan_engine_cls()
    from torchtitan.models.common.attention import FusedQKVLinear

    import verl.workers.engine.torchtitan.transformer_impl as ti

    monkeypatch.setattr(ti, "get_device_id", lambda: "cpu")

    class _DummyQKV(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = 4
            self.heads_per_kv = 2
            self.r_dim = 4
            self.wqkv = nn.Linear(8, 2 * 4 * 4, bias=False)
            self.register_state_dict_post_hook(FusedQKVLinear._split_qkv_on_save)

    qkv = _DummyQKV()
    engine = TorchTitanEngine.__new__(TorchTitanEngine)
    engine.parallel_dims = SimpleNamespace(pp_enabled=False)
    engine.module = [qkv]
    engine._expert_stack_slots = lambda name, param: None
    engine._to_hf_named_params = lambda d: d

    gen, _ = engine.get_per_tensor_param_shard()
    exported = {name: (flat, spec) for name, flat, spec in gen}

    assert set(exported.keys()) == {"wq.weight", "wk.weight", "wv.weight"}
    assert exported["wq.weight"][1].full_shape == (16, 8)
    assert exported["wk.weight"][1].full_shape == (8, 8)
    assert exported["wv.weight"][1].full_shape == (8, 8)
    # Instance hook restored after export
    assert list(qkv._state_dict_hooks.values()) == [FusedQKVLinear._split_qkv_on_save]
