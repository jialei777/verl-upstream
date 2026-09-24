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
"""pad_packed_inputs_for_tpu must not let a document's last label point into the next document."""

import importlib.util
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

# Load by path: the `verl.workers.engine.torchtitan` package imports torchtitan.
_TPU_UTILS_PATH = Path(__file__).resolve().parents[2] / "verl/workers/engine/torchtitan/tpu_utils.py"
_spec = importlib.util.spec_from_file_location("_torchtitan_tpu_utils_under_test", _TPU_UTILS_PATH)
tpu_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tpu_utils)


def _pack(docs):
    input_ids = torch.nested.as_nested_tensor([torch.tensor(d) for d in docs], layout=torch.jagged)
    position_ids = torch.nested.as_nested_tensor([torch.arange(len(d)) for d in docs], layout=torch.jagged)
    return input_ids, position_ids


@pytest.mark.parametrize(
    ("docs", "bucket"),
    [
        ([[11, 12, 13], [21, 22], [31, 32, 33, 34]], 16),  # bucket padding after the last document
        ([[11, 12, 13], [21, 22, 23, 24, 25]], 8),  # exactly one bucket: roll wraps to position 0
        ([[11, 12, 13, 14, 15, 16, 17, 18]], 8),  # single document, no padding
    ],
)
def test_last_label_of_each_document_is_zeroed(monkeypatch, docs, bucket):
    monkeypatch.setattr(tpu_utils, "get_tpu_seq_bucket_size", lambda: bucket)
    input_ids, position_ids = _pack(docs)

    _, _, labels, _, orig_seq_len = tpu_utils.pad_packed_inputs_for_tpu(
        input_ids=input_ids, position_ids=position_ids, micro_batch=TensorDict({}, batch_size=[]), device="cpu"
    )

    expected = []
    for doc in docs:
        expected += doc[1:] + [0]  # in-document next-token labels, 0 at the document end
    expected += [0] * (labels.shape[1] - orig_seq_len)
    assert labels.squeeze(0).tolist() == expected
