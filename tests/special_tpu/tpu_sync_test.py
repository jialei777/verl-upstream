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
"""tpu-sync weight-sync correctness on TPU: real TorchTitan trainer -> tpu-sync -> fake rollout workers (no vLLM).

The TPU counterpart of tests/checkpoint_engine/test_correctness_on_{gpu,npu}.py. A TorchTitan trainer (configured as the
GRPO actor) on the first TPU slice sends its FSDP shards with verl-hardware-plugin's real ``TPUSyncCheckpointEngine``,
driver-side controller and tpu-sync transport (``CheckpointEngineManager.update_weights`` with ``backend=tpu-sync``).
Fake rollout replicas on the second slice run the real ``vLLMTPUSyncWorkerExtension.init_tpu_sync_on_worker`` (receive
buffers, TP sharding specs, controller registration) in single-chip processes, without vLLM. After every sync each
rollout rank compares the tensors it received bit for bit with its TP slice of the HF checkpoint (the trainer does not
train, so the weights must equal the checkpoint), and the engine's own parity checks (``verify_parity=all``) run as
well.

Cases: one TP4 replica, two TP2 replicas (one transfer to several replicas), and TP4 keeping the trainer's send
buffers between syncs; 3 syncs each. Submitted as a Ray job to a RayCluster with two TPU slices (``tpu-group-0`` /
``tpu-group-1``; TPU CI test ``tpu_sync``: 2 x v6e-4):

    python3 tests/special_tpu/tpu_sync_test.py
"""

import asyncio
import importlib
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from types import SimpleNamespace

import ray
import torch

import verl  # noqa: F401  (loads verl-hardware-plugin: the TPU platform, engine and tpu-sync backend)
from verl.checkpoint_engine import CheckpointEngineManager, CheckpointEngineRegistry
from verl.plugin.platform import get_platform
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.config import (
    CheckpointEngineConfig,
    HFModelConfig,
    TorchtitanEngineConfig,
    TorchtitanOptimizerConfig,
    TrainingWorkerConfig,
)
from verl.workers.engine_workers import TrainingWorker

BACKEND = "tpu-sync"
MODEL_PATH = os.environ.get("TPU_SYNC_TEST_MODEL_PATH", "/data/jialei/assets/hf/Qwen3-0.6B")
NUM_SYNCS = int(os.environ.get("TPU_SYNC_TEST_NUM_SYNCS", "3"))
# (num_replicas, tp_size, release_buffers_after_sync)
CASES = [(1, 4, True), (2, 2, True), (1, 4, False)]
# Same TP layout as vLLM (and verl-hardware-plugin's rollout/tpu_sync_worker_extension._sampler_sharding).
ROW_PARALLEL_SUFFIXES = (".o_proj.weight", ".down_proj.weight")
KV_PROJ_SUFFIXES = (".self_attn.k_proj.weight", ".self_attn.v_proj.weight")
PARITY_LOGGERS = (
    "verl_hardware_plugin.accelerators.tpu.weight_sync.tpu_sync_checkpoint_engine",
    "verl_hardware_plugin.accelerators.tpu.weight_sync.tpu_sync_parity",
)


def log(msg: str) -> None:
    print(f"[tpu-sync test] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------
# Trainer: TorchTitan engine (as the GRPO actor) + TPUSyncCheckpointEngine
# ---------------------------------------------------------------------------------------------


class TPUSyncTrainerWorker(TrainingWorker):
    """TrainingWorker that sends its weights with ``TPUSyncCheckpointEngine``, as ``ActorRolloutRefWorker`` does."""

    def __init__(self, config: TrainingWorkerConfig) -> None:
        super().__init__(config)
        self.checkpoint_engine = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_tpu_sync(self, **engine_kwargs):
        """(Re)create the engine, so that every case registers with its own controller."""
        if self.checkpoint_engine is not None:
            self.checkpoint_engine.finalize()
        self.checkpoint_engine = CheckpointEngineRegistry.new(
            BACKEND, bucket_size=0, is_master=torch.distributed.get_rank() == 0, **engine_kwargs
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None, mode: str = "auto"):
        # Same as ActorRolloutRefWorker.update_weights for a disaggregated checkpoint engine.
        per_tensor_param, _ = self.engine.get_per_tensor_param()
        return await self.checkpoint_engine.send_weights(per_tensor_param, global_steps=global_steps) or {}

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        # Same as ActorRolloutRefWorker.execute_checkpoint_engine (the driver calls release_sync_buffers with it).
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)


# ---------------------------------------------------------------------------------------------
# Fake rollout worker: one single-chip process per TP rank, running the real sampler-side tpu-sync code
# ---------------------------------------------------------------------------------------------


def _pin_single_chip_env() -> None:
    """Restrict this process to the one TPU chip Ray assigned to it (as tests/special_tpu/tpu_smoke_test.py does)."""
    chip_ids = ray.get_runtime_context().get_accelerator_ids().get("TPU", [])
    assert len(chip_ids) == 1, f"expected exactly one assigned TPU chip, got {chip_ids}"
    port = 8471 + int(chip_ids[0])
    os.environ.update(
        {
            "TPU_VISIBLE_CHIPS": str(chip_ids[0]),
            "TPU_VISIBLE_DEVICES": str(chip_ids[0]),
            "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
            "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,1,1",
            "TPU_HOST_BOUNDS": "1,1,1",
            "TPU_PROCESS_BOUNDS": "1,1,1",
            "TPU_WORKER_ID": "0",
            "TPU_WORKER_HOSTNAMES": "localhost",
            "CLOUD_TPU_TASK_ID": "0",
            "TORCH_TPU_TOPOLOGY": "1,1,1",
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": f"localhost:{port}",
            "TPU_PROCESS_ADDRESSES": f"localhost:{port}",
            "TPU_PROCESS_PORT": str(port),
        }
    )


def expected_tp_slice(name: str, full: torch.Tensor, tp_size: int, tp_rank: int, num_kv_heads: int) -> torch.Tensor:
    """The slice of an HF tensor that vLLM's TP rank ``tp_rank`` holds, i.e. what tpu-sync must deliver to it."""
    if full.dim() != 2:
        return full
    if name.endswith(ROW_PARALLEL_SUFFIXES):
        return full.chunk(tp_size, dim=1)[tp_rank]
    if num_kv_heads < tp_size and name.endswith(KV_PROJ_SUFFIXES):
        return full.chunk(num_kv_heads, dim=0)[tp_rank // (tp_size // num_kv_heads)]
    if full.shape[0] % tp_size == 0:
        return full.chunk(tp_size, dim=0)[tp_rank]
    return full


@ray.remote(num_cpus=1)
class FakeRolloutWorker:
    """Stands in for one vLLM TP worker of a rollout replica.

    ``init_tpu_sync_on_worker`` and ``get_model_weights_stats`` run the real ``vLLMTPUSyncWorkerExtension`` code on
    this object. ``install_tpu_sync_weights`` replaces only the copy into vLLM's parameters: it runs the H2D and then
    compares every received tensor bit for bit with this rank's slice of the checkpoint.
    """

    def __init__(self, tp_rank: int, tp_size: int, model_path: str) -> None:
        _pin_single_chip_env()  # before anything initializes the TPU runtime
        import torch_tpu  # noqa: F401
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_path)
        self.rank = tp_rank
        self.tp_size = tp_size
        self.model_path = model_path
        self.num_kv_heads = hf_config.num_key_value_heads
        self.tie_word_embeddings = bool(getattr(hf_config, "tie_word_embeddings", False))
        # What vLLMTPUSyncWorkerExtension reads from the vLLM worker.
        self.model_runner = SimpleNamespace(model=torch.nn.Module())
        self.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
            model_config=SimpleNamespace(get_total_num_kv_heads=lambda: self.num_kv_heads, hf_text_config=None),
        )
        self._checkpoint = None
        self.last_exact = None

    @staticmethod
    def _extension():
        from verl_hardware_plugin.accelerators.tpu.rollout.tpu_sync_worker_extension import (
            vLLMTPUSyncWorkerExtension,
        )

        return vLLMTPUSyncWorkerExtension

    # The extension methods that init_tpu_sync_on_worker calls on ``self``.
    def _get_vllm_model(self):
        return self._extension()._get_vllm_model(self)

    def _wait_for_tpu_sync_rendezvous(self, rank_val, rendezvous_name=None):
        return self._extension()._wait_for_tpu_sync_rendezvous(self, rank_val, rendezvous_name)

    def init_tpu_sync_on_worker(self, **kwargs) -> bool:
        return self._extension().init_tpu_sync_on_worker(self, **kwargs)

    def get_model_weights_stats(self) -> dict:
        return self._extension().get_model_weights_stats(self)

    def _checkpoint_tensors(self) -> dict:
        """Every tensor of the HF checkpoint, full (loaded once)."""
        if self._checkpoint is None:
            from safetensors import safe_open

            index = os.path.join(self.model_path, "model.safetensors.index.json")
            if os.path.exists(index):
                with open(index) as f:
                    files = sorted(set(json.load(f)["weight_map"].values()))
            else:
                files = ["model.safetensors"]
            tensors = {}
            for file in files:
                with safe_open(os.path.join(self.model_path, file), framework="pt") as f:
                    for name in f.keys():
                        tensors[name] = f.get_tensor(name)
            if self.tie_word_embeddings and "lm_head.weight" not in tensors:
                # A tied checkpoint stores only the embedding; the trainer may still send lm_head.
                tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]
            self._checkpoint = tensors
        return self._checkpoint

    def install_tpu_sync_weights(self, exact_parity: bool = False) -> dict:
        from verl_hardware_plugin.accelerators.tpu.weight_sync import tpu_sync_checkpoint_engine as tpu_sync_engine
        from verl_hardware_plugin.accelerators.tpu.weight_sync.tpu_sync_parity import ExactParityCheck

        t0 = time.perf_counter()
        self._tpu_sync_ws.h2d()
        tpu_sync_engine.tpu_synchronize()
        t_h2d = time.perf_counter() - t0

        checkpoint = self._checkpoint_tensors()
        parity = ExactParityCheck()
        received = self._tpu_sync_staging
        for name, tensor in received.items():
            if name in checkpoint:
                expected = expected_tp_slice(name, checkpoint[name], self.tp_size, self.rank, self.num_kv_heads)
                parity.check(name, expected, tensor.cpu())
        # Tensors the trainer sent that are not in the checkpoint, or checkpoint tensors it did not send (a tied
        # lm_head may be left out).
        missing = [
            n for n in checkpoint if n not in received and not (self.tie_word_embeddings and n == "lm_head.weight")
        ]
        unresolved = sorted([n for n in received if n not in checkpoint] + missing)
        self.last_exact = parity.result(self.rank, unresolved)
        return {"total": time.perf_counter() - t0, "h2d": t_h2d, "exact_parity": self.last_exact}

    def last_exact_result(self) -> dict:
        return {"exact_parity": self.last_exact}


class _AsyncRemote:
    """``.remote(**kwargs)`` returning an awaitable, like a Ray actor method."""

    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class FakeReplica:
    """The parts of a ``RolloutReplica`` that ``CheckpointEngineManager`` and ``update_tpu_sync_weights`` use."""

    def __init__(self, replica_rank: int, workers: list) -> None:
        self.replica_rank = replica_rank
        self.workers = workers
        self.world_size = len(workers)
        self.global_steps = None
        self.server_handle = SimpleNamespace(
            collective_rpc=_AsyncRemote(self._collective_rpc), set_global_steps=_AsyncRemote(self._set_global_steps)
        )

    async def _collective_rpc(self, method: str, kwargs: dict = None):
        return await asyncio.gather(*[getattr(w, method).remote(**(kwargs or {})) for w in self.workers])

    async def _set_global_steps(self, global_steps):
        self.global_steps = global_steps

    async def abort_all_requests(self, reject_request: bool = False):
        pass

    async def resume_generation(self):
        pass


# ---------------------------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------------------------


@dataclass
class TPUSlice:
    name: str  # the slice's Ray resource, tpu-group-<n>
    hosts: list
    chips_per_host: int

    @property
    def chips(self) -> int:
        return len(self.hosts) * self.chips_per_host


def discover_tpu_slices() -> list:
    """The cluster's TPU slices: KubeRay advertises one ``tpu-group-<n>`` resource per slice on its hosts."""
    slices = {}
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        resources = node.get("Resources", {})
        for res in resources:
            if res.startswith("tpu-group-"):
                tpu_slice = slices.setdefault(res, TPUSlice(res, [], int(resources.get("TPU", 0))))
                tpu_slice.hosts.append(node["NodeManagerAddress"])
    return [slices[name] for name in sorted(slices)]


def create_trainer(trainer_slice: TPUSlice) -> RayWorkerGroup:
    model_config = HFModelConfig(path=MODEL_PATH, use_remove_padding=True)
    engine_config = TorchtitanEngineConfig(
        # Configured as the GRPO actor (fp32 parameters and an optimizer), whose weights tpu-sync sends: a
        # forward_only engine keeps bf16 parameters, which the weight export would hand over as views of their storage.
        forward_only=False,
        data_parallel_shard_size=trainer_slice.chips,
        use_torch_compile=False,
        attn_type="varlen",
        activation_checkpoint="none",
        # SimpleFSDP, as the TPU GRPO recipes (verl-hardware-plugin adds engine_kwargs to TorchtitanEngineConfig).
        engine_kwargs={"tpu": {"use_simple_fsdp": True}},
    )
    optimizer_config = TorchtitanOptimizerConfig(lr=1e-6, total_training_steps=1)
    config = TrainingWorkerConfig(
        model_type="language_model",
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
    )
    pool = RayResourcePool(
        process_on_nodes=[trainer_slice.chips_per_host] * len(trainer_slice.hosts),
        use_gpu=True,
        name_prefix="tpu_sync_test_trainer",
        accelerator_type=trainer_slice.name,
    )
    wg = RayWorkerGroup(
        resource_pool=pool,
        ray_cls_with_init=RayClassWithInitArgs(cls=ray.remote(TPUSyncTrainerWorker), config=config),
        device_name="tpu",
    )
    wg.reset()  # load the HF checkpoint into the TorchTitan engine
    return wg


def create_replicas(sampler_slice: TPUSlice, num_replicas: int, tp_size: int) -> list:
    assert num_replicas * tp_size <= sampler_slice.chips, f"{num_replicas}x TP{tp_size} > {sampler_slice.chips} chips"
    resources = {"TPU": 1, sampler_slice.name: 0.001}
    return [
        FakeReplica(
            i, [FakeRolloutWorker.options(resources=resources).remote(r, tp_size, MODEL_PATH) for r in range(tp_size)]
        )
        for i in range(num_replicas)
    ]


def run_case(trainer_wg, sampler_slice: TPUSlice, num_replicas: int, tp_size: int, release_buffers: bool) -> None:
    from verl_hardware_plugin.accelerators.tpu.weight_sync.tpu_sync_parity import report_exact_parity

    engine_kwargs = {"verify_parity": "all", "release_buffers_after_sync": release_buffers}
    trainer_wg.reset_tpu_sync(**engine_kwargs)
    replicas = create_replicas(sampler_slice, num_replicas, tp_size)
    config = CheckpointEngineConfig(backend=BACKEND, engine_kwargs={BACKEND: engine_kwargs})
    name = f"{num_replicas}xTP{tp_size} release={release_buffers}"
    labels = [""] if num_replicas == 1 else [f" | replica {r.replica_rank}" for r in replicas]
    try:
        manager = CheckpointEngineManager(config=config, actor_wg=trainer_wg, replicas=replicas)
        for step in range(NUM_SYNCS):
            # The engine runs its exact check at step 0 and its norm check at every step.
            metrics = manager.update_weights(global_steps=step)
            assert metrics and metrics["timing_s/tpu-sync/total_sync"] > 0, metrics
            assert all(r.global_steps == step for r in replicas), [r.global_steps for r in replicas]
            # Every sync, every rank of every replica: the received tensors equal the checkpoint slice, bit for bit.
            results = [ray.get([w.last_exact_result.remote() for w in r.workers]) for r in replicas]
            if not report_exact_parity(step, results, labels):  # raises TPUSyncParityError with fail-on-mismatch
                raise AssertionError(f"{name} step {step}: received tensors differ from the checkpoint")
            log(
                f"{name} step {step}: bit-exact, total_sync={metrics['timing_s/tpu-sync/total_sync']:.3f}s "
                f"p2p={metrics['timing_s/tpu-sync/p2p_transfer']:.3f}s"
            )
    finally:
        for r in replicas:
            for w in r.workers:
                ray.kill(w)


def main() -> int:
    # Show the engine's [TPU-SYNC PARITY ...] / [TPU-SYNC TELEMETRY ...] lines in the job log (the plugin's loggers
    # default to VERL_LOGGING_LEVEL, WARN; the CI also sets VERL_LOGGING_LEVEL=INFO for the workers).
    logging.basicConfig(format="%(levelname)s:%(asctime)s:%(name)s: %(message)s")
    for module in PARITY_LOGGERS:
        importlib.import_module(module)  # first: the modules set their logger level from the environment on import
        logging.getLogger(module).setLevel(logging.INFO)
    # The TPU platform's runtime_env (patch_ray_worker, the TPU environment), as verl's trainers use.
    ray.init(**get_platform().get_ray_init_kwargs())
    slices = discover_tpu_slices()
    log(f"TPU slices: {[(s.name, s.hosts, s.chips_per_host) for s in slices]}")
    if len(slices) < 2:
        log(f"FAILED: needs two TPU slices (trainer + rollout), found {len(slices)}")
        return 1
    trainer_slice, sampler_slice = slices[0], slices[1]
    t0 = time.perf_counter()
    trainer_wg = create_trainer(trainer_slice)
    log(f"trainer ready on {trainer_slice.name} ({trainer_slice.chips} chips) in {time.perf_counter() - t0:.1f}s")

    cases = [c for c in CASES if c[0] * c[1] <= sampler_slice.chips]
    if not cases:
        log(f"FAILED: no case fits the {sampler_slice.chips} rollout chips")
        return 1
    failed = []
    for num_replicas, tp_size, release_buffers in cases:
        name = f"{num_replicas}xTP{tp_size}-release={release_buffers}"
        try:
            run_case(trainer_wg, sampler_slice, num_replicas, tp_size, release_buffers)
            log(f"case {name}: PASSED")
        except Exception:
            traceback.print_exc()
            log(f"case {name}: FAILED")
            failed.append(name)
    if failed:
        log(f"FAILED: {len(failed)}/{len(cases)} cases: {failed}")
        return 1
    log(f"PASSED: {len(cases)}/{len(cases)} cases, {NUM_SYNCS} syncs each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
