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
"""Tests for the TPU rollout log-prob repair applied before rollout importance sampling."""

import pytest
import torch

from verl.trainer.ppo.rollout_corr_helper import (
    compute_rollout_correction_and_rejection_mask,
    sanitize_tpu_rollout_log_prob,
)


def _inputs():
    old_log_prob = torch.tensor(
        [
            [-20.0, 0.0, 0.0],  # single-token response (immediate EOS)
            [-1.0, -2.0, -3.0],  # second token missing (reported as 0.0)
            [-0.5, -0.7, -0.9],  # healthy
        ]
    )
    rollout_log_prob = torch.tensor(
        [
            [-3.0, 0.0, 0.0],
            [-1.1, 0.0, -2.9],
            [-0.4, -0.8, 0.0],  # last position is padding, 0.0 there must stay untouched
        ]
    )
    response_mask = torch.tensor([[1, 0, 0], [1, 1, 1], [1, 1, 0]])
    return old_log_prob, rollout_log_prob, response_mask


def test_sanitize_repairs_only_unreported_in_mask_entries():
    old_log_prob, rollout_log_prob, response_mask = _inputs()

    repaired_lp, repaired = sanitize_tpu_rollout_log_prob(old_log_prob, rollout_log_prob, response_mask)

    expected_mask = torch.tensor([[True, False, False], [False, True, False], [False, False, False]])
    assert torch.equal(repaired, expected_mask)
    expected = torch.where(expected_mask, old_log_prob, rollout_log_prob)
    torch.testing.assert_close(repaired_lp, expected)

    # Idempotent, so compute_offpolicy_metrics may re-apply it safely.
    again, _ = sanitize_tpu_rollout_log_prob(old_log_prob, repaired_lp, response_mask)
    torch.testing.assert_close(again, repaired_lp)


def test_rollout_correction_on_tpu_keeps_repaired_tokens_at_unit_weight(monkeypatch):
    old_log_prob, rollout_log_prob, response_mask = _inputs()
    monkeypatch.setattr("verl.utils.device.get_device_name", lambda: "tpu")

    weights_proto, _, metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
        rollout_is="token",
        rollout_is_threshold=2.0,
    )

    weights = weights_proto.batch["rollout_is_weights"]
    torch.testing.assert_close(weights[0, 0], torch.tensor(1.0))
    torch.testing.assert_close(weights[1, 1], torch.tensor(1.0))
    assert metrics["rollout_corr/tpu_repaired_rollout_log_prob_frac"] == pytest.approx(2 / int(response_mask.sum()))
    # One of the two repaired tokens sits at response position 0.
    assert metrics["rollout_corr/tpu_repaired_first_token_frac"] == pytest.approx(0.5)
    # Raw (pre-repair) KL keeps the generator's unrepaired signal; post-repair KL does not.
    expected_raw_kl = ((rollout_log_prob - old_log_prob) * response_mask).sum() / response_mask.sum()
    assert metrics["rollout_corr/tpu_raw_kl"] == pytest.approx(expected_raw_kl.item())
    assert metrics["rollout_corr/tpu_raw_kl"] != pytest.approx(metrics["rollout_corr/kl"])


def test_rollout_correction_off_tpu_is_unchanged(monkeypatch):
    old_log_prob, rollout_log_prob, response_mask = _inputs()
    monkeypatch.setattr("verl.utils.device.get_device_name", lambda: "cuda")

    weights_proto, _, metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
        rollout_is="token",
        rollout_is_threshold=2.0,
    )

    assert not any(k.startswith("rollout_corr/tpu_") for k in metrics)
    expected_w00 = torch.exp(old_log_prob[0, 0] - rollout_log_prob[0, 0]).clamp(max=2.0)
    torch.testing.assert_close(weights_proto.batch["rollout_is_weights"][0, 0], expected_w00)
