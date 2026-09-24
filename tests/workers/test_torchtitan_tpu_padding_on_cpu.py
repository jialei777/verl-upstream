# Copyright 2025 Google LLC
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
"""CPU tests for the TPU padding helpers used by the TorchTitan ``use_remove_padding=False`` path."""

import importlib.util
from pathlib import Path

import pytest
import torch

# Load tpu_utils by path: importing the `verl.workers.engine.torchtitan` package pulls in
# transformer_impl and therefore torchtitan, which the plain CPU unit-test image does not have.
_TPU_UTILS_PATH = Path(__file__).resolve().parents[2] / "verl/workers/engine/torchtitan/tpu_utils.py"
_spec = importlib.util.spec_from_file_location("_torchtitan_tpu_utils_under_test", _TPU_UTILS_PATH)
tpu_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tpu_utils)
build_packed_gather_index = tpu_utils.build_packed_gather_index
safe_to_padded_tensor = tpu_utils.safe_to_padded_tensor


@pytest.fixture(autouse=True)
def _cpu_device(monkeypatch):
    monkeypatch.setattr("verl.utils.device.get_device_id", lambda: "cpu")


def _njt(tensors):
    return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)


def test_safe_to_padded_tensor_matches_torch_2d():
    nt = _njt([torch.arange(3), torch.arange(5) + 10, torch.arange(1) + 20])
    expected = torch.nested.to_padded_tensor(nt, padding=-1, output_size=(3, 8))
    actual = safe_to_padded_tensor(nt, padding=-1, output_size=(3, 8))
    torch.testing.assert_close(actual, expected)


def test_safe_to_padded_tensor_matches_torch_3d_ragged_last_dim():
    # 3D mrope position_ids: logical [bsz, 4, j] with the ragged dim at index 2.
    per_sample = [torch.arange(4 * n).reshape(4, n) for n in (3, 6, 2)]
    nt = _njt([t.transpose(0, 1) for t in per_sample]).transpose(1, 2)
    assert nt._ragged_idx == 2
    expected = torch.nested.to_padded_tensor(nt, padding=0, output_size=(3, 4, 8))
    actual = safe_to_padded_tensor(nt, padding=0, output_size=(3, 4, 8))
    torch.testing.assert_close(actual, expected)


def test_build_packed_gather_index_repacks_dense_layout(monkeypatch):
    monkeypatch.setattr(tpu_utils, "get_tpu_seq_bucket_size", lambda: 8)
    seq_lens = torch.tensor([3, 5, 1])
    padded_seq_len, vocab = 8, 11
    lengths = seq_lens.tolist()
    dense = torch.randn(len(lengths), padded_seq_len, vocab)

    gather_idx, total = build_packed_gather_index(seq_lens, padded_seq_len=padded_seq_len)

    assert total == sum(lengths)
    assert gather_idx.shape[0] == tpu_utils.bucket_length(total)
    packed = dense.reshape(-1, vocab).index_select(0, gather_idx)
    expected = torch.cat([dense[i, :n] for i, n in enumerate(lengths)])
    torch.testing.assert_close(packed[:total], expected)
