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

import asyncio
import hashlib
import logging
import os
import sys
import time
from typing import Any, Generator, List, Optional

if "/tmp/tpu-sync" not in sys.path and os.path.exists("/tmp/tpu-sync"):
    sys.path.insert(0, "/tmp/tpu-sync")

import ray
import torch
import tpu_sync

from verl.checkpoint_engine.base import (
    CheckpointEngine,
    CheckpointEngineRegistry,
)

logger = logging.getLogger(__name__)


def _compute_tensor_checksum(tensors: List[torch.Tensor]) -> dict:
    """Compute deterministic SHA-256 hash and L1/L2 norms across tensors."""
    hasher = hashlib.sha256()
    total_numel = 0
    total_l1 = 0.0
    total_l2_sq = 0.0
    for p in tensors:
        p_cpu = p.detach().cpu().contiguous()
        hasher.update(p_cpu.flatten().view(torch.uint8).numpy().tobytes())
        total_numel += p_cpu.numel()
        total_l1 += float(p_cpu.float().abs().sum().item())
        total_l2_sq += float(p_cpu.float().pow(2).sum().item())

    return {
        "sha256": hasher.hexdigest(),
        "total_numel": total_numel,
        "num_tensors": len(tensors),
        "l1_norm": total_l1,
        "l2_norm": float(total_l2_sq**0.5),
    }


def _create_torch_weight_synchronizer(
    device_tensors: List[List[torch.Tensor]],
    local_port: int = 0,
    parallelism: int = 8,
    listener_port: int = 0,
    bind_ip: str = "127.0.0.1",
):
    """Creates an instance of WeightSynchronizer using tpu_sync."""
    from tpu_sync.api.torch.weight_synchronizer import WeightSynchronizer

    return WeightSynchronizer(
        device_tensors,
        local_port=local_port,
        parallelism=parallelism,
        listener_port=listener_port,
        bind_ip=bind_ip,
        unsafe_skip_buffer_lock=True,
        auto_h2d=False,
    )


@CheckpointEngineRegistry.register("raiden")
class RaidenCheckpointEngine(CheckpointEngine):
    """P2P Weight Synchronizer Checkpoint Engine for TPUs using Google Raiden (tpu-sync)."""

    def __init__(self, bucket_size: int = 0, is_master: bool = False, **kwargs) -> None:
        self.is_master = is_master
        self.bucket_size = bucket_size
        self.backend = "raiden"
        self._trainer_raiden_ws = None
        self._trainer_chunks = []
        self.rank = int(os.environ.get("RANK", "0"))

        from verl.checkpoint_engine.tpu_weight_registry import get_tpu_weight_registry

        self.registry = get_tpu_weight_registry()

    def prepare(self) -> dict:
        return {}

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]):
        return {}, {}

    def init_process_group(self, **kwargs):
        pass

    def finalize(self):
        if self._trainer_raiden_ws is not None:
            try:
                self._trainer_raiden_ws.close()
            except Exception:
                pass
            self._trainer_raiden_ws = None

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        dst_peers: Optional[List[str]] = None,
        global_steps: Optional[int] = None,
        compute_checksum: bool = True,
        **kwargs,
    ):
        """Register weights with RaidenController and prepare for coordinated P2P network transfer."""
        step_key = global_steps if global_steps is not None else 0
        logger.info(f"@@@ RaidenCheckpointEngine: [Step {step_key}] Start send_weights...")

        if hasattr(torch, "tpu") and hasattr(torch.tpu, "synchronize"):
            torch.tpu.synchronize()

        weight_list = list(weights)

        # Handle tied word embeddings
        has_embed = any("embed_tokens" in k or "tok_embeddings" in k for k, _ in weight_list)
        if has_embed:
            filtered_weight_list = [
                (k, v) for k, v in weight_list
                if not (k == "lm_head.weight" or k.endswith(".lm_head.weight"))
            ]
        else:
            filtered_weight_list = weight_list

        sorted_weights = sorted(filtered_weight_list, key=lambda x: x[0])

        device = torch.device("tpu:0" if hasattr(torch, "tpu") else "tpu")
        valid_weights = []
        for name, p in sorted_weights:
            if p is None:
                continue
            t = p.data if hasattr(p, "data") else p
            if hasattr(t, "to_local"):
                t = t.to_local()
            if hasattr(t, "data"):
                t = t.data
            if not isinstance(t, torch.Tensor):
                continue
            if t.numel() == 0 or getattr(t, "is_meta", False):
                continue
            if not (hasattr(t, "device") and str(t.device).startswith("tpu")):
                try:
                    t = t.to(device)
                except Exception as e:
                    logger.warning(f"Could not move {name} to TPU: {e}")
                    continue
            if not t.is_contiguous():
                t = t.contiguous()
            valid_weights.append((name, t))

        try:
            from torch_tpu._internal import sync as torch_tpu_sync
            torch_tpu_sync.synchronize(wait=True)
        except Exception as e:
            logger.warning(f"Could not synchronize via torch_tpu: {e}")

        tensors = [t for _, t in valid_weights]

        if not hasattr(self, "_trainer_raiden_ws") or self._trainer_raiden_ws is None:
            from tpu_sync.rpc import raiden_service_pb2

            bind_ip = ray.util.get_node_ip_address().strip("[]")
            listener_port = 11000 + self.rank

            self._trainer_tensors = tensors
            device_tensors = [[p] for p in tensors]

            logger.info(
                f"Trainer Rank {self.rank}: binding {len(device_tensors)} tensors to WeightSynchronizer "
                f"(sample tensor: name={valid_weights[0][0]}, device={valid_weights[0][1].device}, dtype={valid_weights[0][1].dtype}, "
                f"total numel={sum(t.numel() for t in tensors)})"
            )

            self._trainer_raiden_ws = _create_torch_weight_synchronizer(
                device_tensors,
                local_port=0,
                listener_port=listener_port,
                parallelism=8,
                bind_ip=bind_ip,
            )

            # Build variable metadata protos for each dynamic tensor
            variable_protos = []
            for idx, (name, p) in enumerate(valid_weights):
                shape = list(p.shape)
                itemsize = p.element_size()
                layout = list(range(len(shape) - 1, -1, -1))
                variable_protos.append(
                    raiden_service_pb2.VariableMetadataProto(
                        name=name,
                        shape=shape,
                        mesh_shape=[1] * len(shape),
                        layout=layout,
                        item_size=itemsize,
                        layer_idx=idx,
                    )
                )

            controller_addr = None
            if self.registry is not None:
                for _ in range(30):
                    try:
                        controller_addr = ray.get(self.registry.get_controller_address.remote())
                        if controller_addr:
                            break
                    except Exception:
                        pass
                    import time
                    time.sleep(0.5)

            if controller_addr and ":" in controller_addr:
                from tpu_sync.rpc import raiden_controller
                try:
                    ctrl_client = raiden_controller.RaidenControllerClientFacade(controller_addr)
                    unit_id = raiden_controller.RaidenId("trainer", str(self.rank), "weights")
                    ctrl_client.register_work_unit(
                        unit_id,
                        [f"{bind_ip}:{self._trainer_raiden_ws.local_port}"],
                        f"{bind_ip}:{self._trainer_raiden_ws.listener_port}",
                        mesh_shape=[1, 1],
                        variables=variable_protos,
                        mesh_axes=["fsdp", "tp"],
                    )
                    logger.info(
                        f"Trainer Rank {self.rank} bound {len(tensors)} dynamic tensors and registered directly with RaidenController ({controller_addr}): "
                        f"data_port={self._trainer_raiden_ws.local_port}, listener_port={self._trainer_raiden_ws.listener_port}"
                    )
                except Exception as reg_err:
                    logger.error(f"Trainer Rank {self.rank} failed to register with RaidenController ({controller_addr}): {reg_err}")
                    raise
            else:
                raise RuntimeError(f"Trainer Rank {self.rank}: No RaidenController address found in TPUWeightRegistry after timeout")
        else:
            # In-place parameter updates if new tensor objects
            for old_p, new_p in zip(self._trainer_tensors, tensors):
                if old_p.data_ptr() != new_p.data_ptr():
                    old_p.copy_(new_p)

        # Stage weights to host buffer via D2H DMA
        if hasattr(self._trainer_raiden_ws, "d2h"):
            self._trainer_raiden_ws.d2h()
        elif hasattr(self._trainer_raiden_ws, "D2h"):
            self._trainer_raiden_ws.D2h()

        # Compute deterministic checksum & norms across all sent tensors in background
        if compute_checksum and self.registry is not None and self.is_master:
            try:
                cpu_tensors = [p.detach().cpu().contiguous() for p in tensors]
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, self._record_checksum_bg, step_key, cpu_tensors)
            except Exception as sched_err:
                logger.warning(f"Failed to schedule checksum recording: {sched_err}")

    def _record_checksum_bg(self, step_key: int, cpu_tensors: List[torch.Tensor]) -> None:
        """Background worker thread to calculate SHA256 checksum and post to registry."""
        try:
            trainer_checksum = _compute_tensor_checksum(cpu_tensors)
            ray.get(self.registry.set_checksum.remote(step_key, trainer_checksum))
            logger.info(
                f"[RAIDEN PARITY] Successfully stored trainer checksum for step {step_key}: "
                f"{trainer_checksum['sha256']}"
            )
        except Exception as e:
            logger.warning(f"Failed to record trainer checksum in TPUWeightRegistry: {e}")

    @torch.no_grad()
    def receive_weights(self, global_steps: Optional[int] = None, **kwargs):
        return None


async def update_raiden_weights(manager, global_steps: Optional[int] = None) -> dict:
    """Orchestrator coordination for Raiden TPU P2P weight synchronization via central RaidenController."""
    t_abort_start = time.perf_counter()
    if global_steps and global_steps > 0:
        try:
            await manager.abort_replicas()
        except Exception as e:
            logger.warning(f"Failed to abort replicas at step {global_steps}: {e}")
    t_abort = time.perf_counter() - t_abort_start

    t_total_start = time.perf_counter()

    # 1. Initialize and register Sampler rollout replicas with central RaidenController
    t_init_sampler_start = time.perf_counter()
    sampler_init_futures = [
        replica.server_handle.collective_rpc.remote(method="init_raiden_sync_on_worker")
        for replica in manager.replicas
    ]
    await asyncio.gather(*sampler_init_futures)
    t_init_sampler = time.perf_counter() - t_init_sampler_start

    # 2. Trigger Trainer ranks to register their tensors with central RaidenController
    t_init_trainer_start = time.perf_counter()
    actor_refs = manager.actor_wg.update_weights(global_steps=global_steps, mode="raiden")
    if actor_refs is not None:
        await asyncio.to_thread(ray.get, actor_refs)
    t_init_trainer = time.perf_counter() - t_init_trainer_start

    # 3. Explicit Registration Barrier on Central RaidenController
    t_barrier_start = time.perf_counter()
    num_rollout_replicas = len(manager.replicas)
    sampler_replica_ids = [str(i) for i in range(num_rollout_replicas)]
    trainer_replica_ids = [str(i) for i in range(manager.actor_wg.world_size)]

    from tpu_sync.api.common import RaidenId
    from tpu_sync.rpc.raiden_controller import RaidenMemoryType

    src_units = [
        RaidenId(job_name="trainer", job_replica_id=r_id, data_name="weights")
        for r_id in trainer_replica_ids
    ]
    dst_units = [
        RaidenId(job_name="sampler", job_replica_id=r_id, data_name="weights")
        for r_id in sampler_replica_ids
    ]

    if hasattr(manager, "raiden_controller") and manager.raiden_controller is not None:
        barrier_timeout = 60.0
        while True:
            with manager.raiden_controller._lock:
                registered = set(manager.raiden_controller._registered_shards.keys())
            src_registered = all(u in registered for u in src_units)
            dst_registered = all(u in registered for u in dst_units)
            if src_registered and dst_registered:
                logger.info(
                    f"[RAIDEN CONTROLLER] All {len(src_units)} Trainer and {len(dst_units)} Sampler units verified and registered."
                )
                break
            if time.perf_counter() - t_barrier_start > barrier_timeout:
                missing_src = [u for u in src_units if u not in registered]
                missing_dst = [u for u in dst_units if u not in registered]
                raise RuntimeError(
                    f"Timeout ({barrier_timeout}s) waiting for workers to register with RaidenController! "
                    f"Missing Trainer: {missing_src}, Missing Sampler: {missing_dst}"
                )
            await asyncio.sleep(0.2)

        # 4. Trigger coordinated P2P network transfers via central RaidenController
        t_transfer_start = time.perf_counter()
        transfer_future = manager.raiden_controller.start_transfer(
            src_units=src_units,
            dst_units=dst_units,
            dst_mem_type=RaidenMemoryType.DRAM,
            use_block_chunks=True,
            is_sender=True,
            expected_block_count=0,
            parallelism=8,
            req_id=f"verl_step_{global_steps or 0}",
        )
        await transfer_future.wait()
        t_transfer = time.perf_counter() - t_transfer_start
    else:
        logger.error("No raiden_controller found on CheckpointEngineManager.")
        t_transfer = 0.0

    # 4. Sampler replicas install received weights to TPU HBM via H2D DMA
    t_install_start = time.perf_counter()
    install_futures = [
        replica.server_handle.collective_rpc.remote(method="install_raiden_weights")
        for replica in manager.replicas
    ]
    await asyncio.gather(*install_futures)
    t_install = time.perf_counter() - t_install_start

    t_total = time.perf_counter() - t_total_start

    logger.info(
        f"[RAIDEN TELEMETRY | Orchestrator] Step {global_steps} Completed in {t_total:.4f}s:\n"
        f"  * Sampler Quiesce/Pause  : {t_abort:.4f}s\n"
        f"  * Sampler Raiden Init    : {t_init_sampler:.4f}s\n"
        f"  * Trainer Raiden Init    : {t_init_trainer:.4f}s\n"
        f"  * RaidenController P2P   : {t_transfer:.4f}s\n"
        f"  * Sampler H2D DMA        : {t_install:.4f}s\n"
        f"  * Total End-to-End Sync  : {t_total:.4f}s"
    )

    # 5. Resume generation immediately
    await manager.resume_generation_replicas()

    # 6. Parity Verification in background
    try:
        asyncio.create_task(_verify_parity_async(manager, global_steps))
    except Exception as e:
        logger.warning(f"Failed to launch background parity verification: {e}")

    return {}


async def _verify_parity_async(manager, global_steps: Optional[int] = None) -> None:
    """Async background task for cryptographic checksum and norm parity verification."""
    step_key = global_steps if global_steps is not None else 0
    try:
        registry = ray.get_actor("TPUWeightRegistry", namespace="verl")
        trainer_checksum = None
        for _ in range(20):
            trainer_checksum = await registry.get_checksum.remote(step_key)
            if trainer_checksum is not None:
                break
            await asyncio.sleep(0.5)

        sampler_futures = [
            replica.server_handle.collective_rpc.remote(method="get_model_weights_checksum")
            for replica in manager.replicas
        ]
        sampler_checksums = await asyncio.gather(*sampler_futures)

        if not trainer_checksum or not sampler_checksums:
            logger.warning(
                f"[RAIDEN PARITY CHECK | Step {step_key}] Incomplete data: "
                f"trainer_checksum={trainer_checksum}, sampler_checksums={sampler_checksums}"
            )
            return

        def _flatten_checksums(items):
            res = []
            if isinstance(items, (list, tuple)):
                for it in items:
                    res.extend(_flatten_checksums(it))
            elif items:
                res.append(items)
            return res

        flat_sampler_checksums = _flatten_checksums(sampler_checksums)
        for i, s_meta in enumerate(flat_sampler_checksums):
            if not isinstance(s_meta, dict):
                continue
            t_sha = trainer_checksum.get("sha256")
            s_sha = s_meta.get("sha256")
            if t_sha == s_sha:
                logger.info(
                    f"[RAIDEN PARITY VERIFIED | Step {step_key}] Replica {i}: 100% BIT-FOR-BIT WEIGHT MATCH!\n"
                    f"  * Checksum (SHA-256) : {s_sha}\n"
                    f"  * Total Parameters   : {s_meta.get('total_numel')} across {s_meta.get('num_tensors')} tensors\n"
                    f"  * L1 Norm            : {s_meta.get('l1_norm'):.6f}\n"
                    f"  * L2 Norm            : {s_meta.get('l2_norm'):.6f}"
                )
            else:
                logger.error(
                    f"[RAIDEN PARITY MISMATCH | Step {step_key}] Replica {i} weights do NOT match trainer!\n"
                    f"  * Trainer Checksum : {t_sha} (L1={trainer_checksum.get('l1_norm'):.6f}, L2={trainer_checksum.get('l2_norm'):.6f})\n"
                    f"  * Sampler Checksum : {s_sha} (L1={s_meta.get('l1_norm'):.6f}, L2={s_meta.get('l2_norm'):.6f})"
                )
    except Exception as e:
        logger.warning(f"Error during parity verification for step {step_key}: {e}")
