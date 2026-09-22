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
"""CPU tests for the shape-stable helpers used by the TPU training path."""

import torch

from verl.utils.tensordict_utils import get_tensordict
from verl.utils.tpu_utils import (
    TPU_PADDED_VALUES_ATTR,
    bucket_length,
    get_tpu_seq_bucket_size,
    tpu_no_padding_2_padding,
    unwrap_metadata,
)


def test_bucket_length_rounds_up_to_bucket_multiples():
    assert bucket_length(1, bucket_size=256) == 256
    assert bucket_length(256, bucket_size=256) == 256
    assert bucket_length(257, bucket_size=256) == 512
    # a bucket size of one (or less) disables bucketing but never returns zero
    assert bucket_length(13, bucket_size=1) == 13
    assert bucket_length(0, bucket_size=1) == 1


def test_bucket_length_uses_the_configured_bucket_size(monkeypatch):
    monkeypatch.setenv("VERL_TPU_SEQ_BUCKET_SIZE", "128")
    assert get_tpu_seq_bucket_size() == 128
    assert bucket_length(129) == 256


def test_unwrap_metadata_returns_plain_python_values():
    assert unwrap_metadata(True) is True
    assert unwrap_metadata([1.5, 2.5]) == 1.5
    assert unwrap_metadata([]) is None
    assert unwrap_metadata(torch.tensor([2.0])) == 2.0
    assert unwrap_metadata(torch.tensor([[3.0, 4.0]])) == 3.0


def test_tpu_no_padding_2_padding_slices_responses_out_of_the_padded_values():
    # two sequences: prompt/response lengths (2, 3) and (1, 2), packed into one flat tensor
    prompts = torch.nested.nested_tensor_from_jagged(torch.zeros(3, dtype=torch.int64), offsets=torch.tensor([0, 2, 3]))
    responses = torch.nested.nested_tensor_from_jagged(
        torch.zeros(5, dtype=torch.int64), offsets=torch.tensor([0, 3, 5])
    )
    data = get_tensordict(
        tensor_dict={"prompts": prompts, "responses": responses},
        non_tensor_dict={"max_response_len": 4},
    )

    # log prob i predicts token i + 1, so the responses start one position early
    padded_values = torch.arange(16, dtype=torch.float32)
    log_probs = torch.nested.nested_tensor_from_jagged(torch.zeros(8), offsets=torch.tensor([0, 5, 8]))
    setattr(log_probs, TPU_PADDED_VALUES_ATTR, padded_values)

    out = tpu_no_padding_2_padding(log_probs, data)

    # shape is bucketed, not the raw max response length
    assert out.shape == (2, bucket_length(4))
    torch.testing.assert_close(out[0, :3], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(out[1, :2], torch.tensor([5.0, 6.0]))
    # everything past a sequence's response length is masked out
    assert out[0, 3:].abs().sum() == 0
    assert out[1, 2:].abs().sum() == 0


def test_tpu_no_padding_2_padding_falls_back_without_padded_values():
    from verl.workers.utils.padding import no_padding_2_padding

    prompts = torch.nested.nested_tensor_from_jagged(torch.zeros(3, dtype=torch.int64), offsets=torch.tensor([0, 2, 3]))
    responses = torch.nested.nested_tensor_from_jagged(
        torch.zeros(5, dtype=torch.int64), offsets=torch.tensor([0, 3, 5])
    )
    data = get_tensordict(tensor_dict={"prompts": prompts, "responses": responses}, non_tensor_dict={})
    log_probs = torch.nested.nested_tensor_from_jagged(torch.arange(8, dtype=torch.float32), torch.tensor([0, 5, 8]))

    torch.testing.assert_close(tpu_no_padding_2_padding(log_probs, data), no_padding_2_padding(log_probs, data))
