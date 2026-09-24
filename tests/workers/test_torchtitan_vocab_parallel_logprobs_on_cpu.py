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
"""Checks the TorchTitan vocab-parallel log-prob/entropy autograd functions against a dense reference.

Runs TP ranks as CPU processes over gloo, including an uneven vocab split (torch.chunk semantics).
"""

import importlib.util
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_TPU_UTILS_PATH = Path(__file__).resolve().parents[2] / "verl/workers/engine/torchtitan/tpu_utils.py"


def _load_tpu_utils():
    # Load by path: the `verl.workers.engine.torchtitan` package imports torchtitan.
    spec = importlib.util.spec_from_file_location("_torchtitan_tpu_utils_under_test", _TPU_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _reference(logits, labels, temperature, grad_lp, grad_ent):
    logits = logits.clone().requires_grad_(True)
    logp = torch.log_softmax(logits.float() / temperature, dim=-1)
    log_probs = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    entropy = -(logp.exp() * logp).sum(-1)
    ((log_probs * grad_lp).sum() + (entropy * grad_ent).sum()).backward()
    return log_probs.detach(), entropy.detach(), logits.grad


def _worker(rank, world_size, port, vocab, use_dtensor, result_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        tpu_utils = _load_tpu_utils()
        torch.manual_seed(0)
        seq, temperature = 7, 0.7
        full_logits = torch.randn(seq, vocab) * 3
        labels = torch.randint(0, vocab, (seq,))
        grad_lp, grad_ent = torch.randn(seq), torch.randn(seq)
        ref_lp, ref_ent, ref_grad = _reference(full_logits, labels, temperature, grad_lp, grad_ent)

        local = torch.chunk(full_logits, world_size, dim=-1)[rank].clone().requires_grad_(True)
        if use_dtensor:
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.tensor import DTensor, Shard

            mesh = init_device_mesh("cpu", (world_size,))
            logits_in = DTensor.from_local(local, mesh, [Shard(-1)], shape=full_logits.shape, stride=(vocab, 1))
            kwargs = {}
        else:
            logits_in = local
            kwargs = {"tp_group": dist.group.WORLD, "global_vocab": vocab}

        log_probs = tpu_utils.vocab_parallel_logprobs_from_logits(logits_in, labels, temperature, **kwargs)
        entropy = tpu_utils.vocab_parallel_entropy_from_logits(logits_in, temperature, **kwargs)
        ((log_probs * grad_lp).sum() + (entropy * grad_ent).sum()).backward()

        torch.testing.assert_close(log_probs.detach(), ref_lp, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(entropy.detach(), ref_ent, rtol=1e-5, atol=1e-5)
        ref_local_grad = torch.chunk(ref_grad, world_size, dim=-1)[rank]
        torch.testing.assert_close(local.grad, ref_local_grad, rtol=1e-5, atol=1e-5)
        assert log_probs.dtype == torch.float32
        result_queue.put((rank, None))
    except Exception as e:  # noqa: BLE001 - surfaced to the parent process
        result_queue.put((rank, repr(e)))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("vocab", [12, 11])  # even and uneven torch.chunk splits
@pytest.mark.parametrize("use_dtensor", [False, True])
def test_vocab_parallel_logprobs_and_entropy_match_dense(vocab, use_dtensor):
    world_size = 2
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    mp.start_processes(
        _worker,
        args=(world_size, _free_port(), vocab, use_dtensor, result_queue),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )
    errors = [result_queue.get() for _ in range(world_size)]
    assert all(err is None for _, err in errors), errors
