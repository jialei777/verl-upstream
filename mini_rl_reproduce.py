#!/usr/bin/env python3
"""
Mini RL Reproduction Script for TPU-Sync Weight Synchronization with RaidenController.

Architecture:
  - 1 Ray Actor for Trainer (1 TPU chip).
  - 1 Ray Actor for Sampler (1 TPU chip).
  - Main Driver acts as the centralized Orchestrator:
      1. Starts `RaidenControllerServer` and `RaidenController`.
      2. Initializes `TrainerActor` and `SamplerActor` on TPU.
      3. Actors register with `RaidenControllerClientFacade` / `RaidenController`.
      4. Orchestrator triggers `controller.start_transfer(...)`.
      5. Verifies weights are updated and match checksums.
"""

import asyncio
import hashlib
import logging
import os
import socket
import sys
import time
from typing import Dict, List, Tuple

import ray

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MiniRL")


def configure_tpu_environment(actor_name: str, fallback_chip: int = 0) -> int:
    """Configures TPU_VISIBLE_CHIPS and TPU_PROCESS_PORT per actor to avoid libtpu collisions."""
    try:
        ctx = ray.get_runtime_context()
        accelerator_ids = ctx.get_accelerator_ids()
        tpu_ids = accelerator_ids.get("TPU", [])
        if tpu_ids:
            chip_id = int(tpu_ids[0])
        else:
            chip_id = fallback_chip
    except Exception:
        chip_id = fallback_chip

    os.environ["TPU_VISIBLE_CHIPS"] = str(chip_id)
    os.environ["TPU_PROCESS_PORT"] = str(8471 + chip_id)
    os.environ["CLOUD_TPU_TASK_ID"] = "0"
    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
    os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
    logger.info(f"[{actor_name}] Configured TPU env: TPU_VISIBLE_CHIPS={chip_id}, TPU_PROCESS_PORT={8471 + chip_id}")
    return chip_id


def resolve_local_ip() -> str:
    """Resolves local IP reachable by cluster peers."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return ray.util.get_node_ip_address().strip("[]")


def compute_model_checksum(model) -> Dict[str, any]:
    """Computes SHA256 and L1/L2 norms of all parameters in a PyTorch model on TPU."""
    import torch
    hasher = hashlib.sha256()
    total_l1 = 0.0
    total_l2_sq = 0.0
    total_numel = 0

    for name, p in sorted(model.named_parameters(), key=lambda x: x[0]):
        p_data = p.data
        if hasattr(p_data, "to_local"):
            p_data = p_data.to_local()
        p_cpu = p_data.detach().cpu().contiguous()
        hasher.update(p_cpu.flatten().view(torch.uint8).numpy().tobytes())
        total_numel += p_cpu.numel()
        total_l1 += float(p_cpu.float().abs().sum().item())
        total_l2_sq += float(p_cpu.float().pow(2).sum().item())

    return {
        "sha256": hasher.hexdigest(),
        "total_numel": total_numel,
        "l1_norm": total_l1,
        "l2_norm": float(total_l2_sq**0.5),
    }


def create_model():
    import torch
    import torch.nn as nn

    class MiniModel(nn.Module):
        def __init__(self, in_dim: int = 512, hidden_dim: int = 1024, out_dim: int = 512):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, out_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

    return MiniModel()


@ray.remote(resources={"TPU": 1})
class SamplerActor:
    """Ray actor representing the Rollout / Inference Sampler worker on 1 TPU chip."""

    def __init__(self, actor_id: int = 0):
        self.actor_id = actor_id
        self.chip_id = configure_tpu_environment(f"Sampler {self.actor_id}", fallback_chip=0)

        import torch
        self.device = torch.device("tpu:0" if hasattr(torch, "tpu") else "tpu")
        logger.info(f"[Sampler {self.actor_id}] Initializing on device {self.device} (chip {self.chip_id})...")

        # Initialize model on TPU with zeroed weights
        self.model = create_model().to(self.device)
        with torch.no_grad():
            for p in self.model.parameters():
                p.zero_()

        self._synchronize_tpu()
        self.initial_checksum = compute_model_checksum(self.model)
        logger.info(f"[Sampler {self.actor_id}] Model initialized. Checksum: {self.initial_checksum['sha256'][:16]}")

        self.ws = None
        self.node_ip = resolve_local_ip()

    def _synchronize_tpu(self):
        import torch
        try:
            from torch_tpu._internal import sync as torch_tpu_sync
            torch_tpu_sync.synchronize(wait=True)
        except Exception:
            if hasattr(torch, "tpu") and hasattr(torch.tpu, "synchronize"):
                torch.tpu.synchronize()

    def register_with_controller(self, controller_address: str, parallelism: int = 4):
        """Initializes WeightSynchronizer listener and registers with RaidenController."""
        from tpu_sync.api.common import RaidenId
        from tpu_sync.rpc import raiden_controller, raiden_service_pb2

        try:
            from tpu_sync.api.torch.weight_synchronizer import WeightSynchronizer
        except Exception as e:
            logger.error(f"[Sampler {self.actor_id}] FAILED to import WeightSynchronizer: {e}", exc_info=True)
            raise

        self._synchronize_tpu()

        sorted_params = [p for _, p in sorted(self.model.named_parameters(), key=lambda x: x[0])]
        device_tensors = [[p] for p in sorted_params]

        logger.info(
            f"[Sampler {self.actor_id}] Creating WeightSynchronizer listener on {self.node_ip} "
            f"with {len(device_tensors)} tensors..."
        )

        self.ws = WeightSynchronizer(
            device_tensors=device_tensors,
            local_port=0,
            listener_port=0,
            parallelism=parallelism,
            bind_ip=self.node_ip,
            unsafe_skip_buffer_lock=True,
            auto_h2d=True,
        )

        # Build variable metadata protos
        variable_protos = []
        for idx, (name, p) in enumerate(sorted(self.model.named_parameters(), key=lambda x: x[0])):
            variable_protos.append(
                raiden_service_pb2.VariableMetadataProto(
                    name=name,
                    shape=list(p.shape),
                    mesh_shape=[1, 1],
                    layout=list(range(len(p.shape) - 1, -1, -1)),
                    item_size=p.element_size(),
                    layer_idx=idx,
                    sharding_spec=[],
                )
            )

        unit_id = RaidenId("sampler", str(self.actor_id), "mini_model_weights")
        ctrl_client = raiden_controller.RaidenControllerClientFacade(controller_address)
        ctrl_client.register_work_unit(
            unit_id,
            [f"{self.node_ip}:{self.ws.local_port}"],
            f"{self.node_ip}:{self.ws.listener_port}",
            mesh_shape=[1, 1],
            variables=variable_protos,
            mesh_axes=[],
        )
        logger.info(f"[Sampler {self.actor_id}] Registered work unit {unit_id} successfully with controller at {controller_address}")
        return unit_id

    def get_current_checksum(self) -> Dict[str, any]:
        """Returns the current model checksum."""
        self._synchronize_tpu()
        return compute_model_checksum(self.model)

    def verify_against_checksum(self, expected_checksum: str, timeout_sec: float = 30.0) -> bool:
        """Polls until weights are updated in HBM matching the expected checksum."""
        t_start = time.time()
        while time.time() - t_start < timeout_sec:
            self._synchronize_tpu()
            curr = compute_model_checksum(self.model)
            if curr["sha256"] == expected_checksum:
                logger.info(
                    f"[Sampler {self.actor_id}] [PASS] Checksum matched! ({curr['sha256'][:16]}) "
                    f"in {time.time() - t_start:.3f}s"
                )
                return True
            time.sleep(0.1)

        curr = compute_model_checksum(self.model)
        logger.error(
            f"[Sampler {self.actor_id}] [FAIL] Timeout waiting for weight sync! "
            f"Expected {expected_checksum[:16]}, Current {curr['sha256'][:16]}"
        )
        return False


@ray.remote(resources={"TPU": 1})
class TrainerActor:
    """Ray actor representing the Training worker on 1 TPU chip."""

    def __init__(self, actor_id: int = 0):
        self.actor_id = actor_id
        self.chip_id = configure_tpu_environment(f"Trainer {self.actor_id}", fallback_chip=1)

        import torch
        import torch.nn as nn
        self.device = torch.device("tpu:0" if hasattr(torch, "tpu") else "tpu")
        logger.info(f"[Trainer {self.actor_id}] Initializing on device {self.device} (chip {self.chip_id})...")

        # Initialize model on TPU with random weights
        self.model = create_model().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self._synchronize_tpu()

        self.ws = None
        self.node_ip = resolve_local_ip()
        logger.info(f"[Trainer {self.actor_id}] Model initialized on {self.node_ip}.")

    def _synchronize_tpu(self):
        import torch
        try:
            from torch_tpu._internal import sync as torch_tpu_sync
            torch_tpu_sync.synchronize(wait=True)
        except Exception:
            if hasattr(torch, "tpu") and hasattr(torch.tpu, "synchronize"):
                torch.tpu.synchronize()

    def step_training(self, num_steps: int = 3) -> Dict[str, any]:
        """Runs a minimal forward/backward/optimizer step to update model parameters."""
        import torch
        import torch.nn as nn
        logger.info(f"[Trainer {self.actor_id}] Running {num_steps} training steps on TPU...")
        self.model.train()

        for step in range(num_steps):
            self.optimizer.zero_grad()
            dummy_input = torch.randn(16, 512, device=self.device)
            target = torch.randn(16, 512, device=self.device)
            output = self.model(dummy_input)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            self.optimizer.step()
            self._synchronize_tpu()
            logger.info(f"[Trainer {self.actor_id}] Step {step + 1}/{num_steps} Loss: {loss.item():.4f}")

        checksum = compute_model_checksum(self.model)
        logger.info(f"[Trainer {self.actor_id}] Training completed. Updated Checksum: {checksum['sha256'][:16]}")
        return checksum

    def register_with_controller(self, controller_address: str, parallelism: int = 4):
        """Initializes WeightSynchronizer sender and registers with RaidenController."""
        from tpu_sync.api.common import RaidenId
        from tpu_sync.rpc import raiden_controller, raiden_service_pb2

        try:
            from tpu_sync.api.torch.weight_synchronizer import WeightSynchronizer
        except Exception as e:
            logger.error(f"[Trainer {self.actor_id}] FAILED to import WeightSynchronizer: {e}", exc_info=True)
            raise

        self._synchronize_tpu()

        sorted_params = [p for _, p in sorted(self.model.named_parameters(), key=lambda x: x[0])]
        device_tensors = [[p] for p in sorted_params]

        logger.info(
            f"[Trainer {self.actor_id}] Creating WeightSynchronizer sender on {self.node_ip} "
            f"with {len(device_tensors)} tensors..."
        )

        self.ws = WeightSynchronizer(
            device_tensors=device_tensors,
            local_port=0,
            parallelism=parallelism,
            bind_ip=self.node_ip,
            unsafe_skip_buffer_lock=True,
            auto_h2d=False,
        )

        # Build variable metadata protos
        variable_protos = []
        for idx, (name, p) in enumerate(sorted(self.model.named_parameters(), key=lambda x: x[0])):
            variable_protos.append(
                raiden_service_pb2.VariableMetadataProto(
                    name=name,
                    shape=list(p.shape),
                    mesh_shape=[1, 1],
                    layout=list(range(len(p.shape) - 1, -1, -1)),
                    item_size=p.element_size(),
                    layer_idx=idx,
                    sharding_spec=[],
                )
            )

        unit_id = RaidenId("trainer", str(self.actor_id), "mini_model_weights")
        ctrl_client = raiden_controller.RaidenControllerClientFacade(controller_address)
        ctrl_client.register_work_unit(
            unit_id,
            [f"{self.node_ip}:{self.ws.local_port}"],
            f"{self.node_ip}:{self.ws.local_port}",
            mesh_shape=[1, 1],
            variables=variable_protos,
            mesh_axes=[],
        )
        logger.info(f"[Trainer {self.actor_id}] Registered work unit {unit_id} successfully with controller at {controller_address}")
        return unit_id

    def trigger_d2h(self):
        """Triggers D2H DMA copy on Trainer."""
        if self.ws is not None:
            self._synchronize_tpu()
            self.ws.d2h()


def main():
    print("=" * 80)
    print("MINI RL TPU-SYNC WEIGHT SYNCHRONIZATION WITH RAIDENCONTROLLER")
    print("=" * 80)

    # Initialize Ray connection
    ray_address = os.environ.get("RAY_ADDRESS", "auto")
    print(f"Connecting to Ray at: {ray_address}")
    try:
        ray.init(address=ray_address, ignore_reinit_error=True)
    except Exception as e:
        print(f"Direct connection failed ({e}), initializing local Ray instance...")
        ray.init(ignore_reinit_error=True)

    available_resources = ray.available_resources()
    print(f"Ray Cluster Resources: {available_resources}")
    tpu_count = available_resources.get("TPU", 0)
    print(f"Available TPUs: {tpu_count}")

    if tpu_count < 2:
        print(f"[WARNING] Cluster reports {tpu_count} TPUs. Need at least 2 TPUs for 1-Trainer + 1-Sampler test.")

    try:
        # Step 1: Start Centralized RaidenController in Orchestrator (Driver)
        print("\n--- Step 1: Starting Centralized RaidenController Server on Orchestrator ---")
        from tpu_sync.api.common import RaidenId
        from tpu_sync.rpc import raiden_controller

        driver_ip = resolve_local_ip()
        controller_port = 10019
        controller_address = f"{driver_ip}:{controller_port}"
        print(f"Starting RaidenController at {controller_address}...")

        worker_rpc_client = raiden_controller.WeightSyncWorkerRpcClient()
        controller = raiden_controller.RaidenController(
            port=controller_port,
            worker_rpc_client=worker_rpc_client,
        )
        controller_server = raiden_controller.RaidenControllerServer(controller)
        controller_server.start()
        print(f"RaidenControllerServer started and listening on {controller_address}.")

        # Step 2: Instantiating Actors
        print("\n--- Step 2: Instantiating Trainer and Sampler Actors on 1 TPU Chip each ---")
        sampler = SamplerActor.remote(actor_id=0)
        trainer = TrainerActor.remote(actor_id=0)

        # Step 3: Register Actors with RaidenController
        print("\n--- Step 3: Registering Trainer & Sampler Work Units with RaidenController ---")
        sampler_unit_id = ray.get(sampler.register_with_controller.remote(controller_address=controller_address, parallelism=4))
        trainer_unit_id = ray.get(trainer.register_with_controller.remote(controller_address=controller_address, parallelism=4))

        src_units = [trainer_unit_id]
        dst_units = [sampler_unit_id]

        print(f"Source Units: {src_units}")
        print(f"Destination Units: {dst_units}")

        # Wait for registration
        print("Waiting for workers to be registered with controller...")
        t0 = time.time()
        while True:
            registered = set(controller._registered_shards.keys())
            if all(u in registered for u in src_units) and all(u in registered for u in dst_units):
                print("All workers registered with RaidenController!")
                break
            if time.time() - t0 > 30.0:
                raise TimeoutError("Timeout waiting for workers to register with RaidenController!")
            time.sleep(0.5)

        initial_sampler_checksum = ray.get(sampler.get_current_checksum.remote())
        print(f"Sampler Initial Checksum: {initial_sampler_checksum['sha256'][:16]} (L1={initial_sampler_checksum['l1_norm']:.4f})")

        # Step 4: Step Training on Trainer
        print("\n--- Step 4: Executing Mini Training Steps on Trainer ---")
        trainer_checksum = ray.get(trainer.step_training.remote(num_steps=3))
        print(f"Trainer Trained Checksum: {trainer_checksum['sha256'][:16]} (L1={trainer_checksum['l1_norm']:.4f})")

        assert initial_sampler_checksum["sha256"] != trainer_checksum["sha256"], "Initial weights should differ!"

        # Trigger D2H on Trainer
        print("\n--- Step 5: Triggering Trainer D2H DMA Copy ---")
        ray.get(trainer.trigger_d2h.remote())

        # Step 6: Orchestrator calls controller.start_transfer()
        print("\n--- Step 6: Orchestrator Invoking controller.start_transfer() ---")
        t_transfer_start = time.perf_counter()
        future = controller.start_transfer(
            src_units=src_units,
            dst_units=dst_units,
            dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
            use_block_chunks=True,
            is_sender=True,
            expected_block_count=0,
            uuid=9999,
            req_id="mini_rl_transfer",
            group_size=1,
        )

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(future.wait())
        finally:
            loop.close()

        t_transfer_elapsed = time.perf_counter() - t_transfer_start
        print(f"controller.start_transfer() completed in {t_transfer_elapsed * 1000:.2f} ms")

        # Step 7: Verifying Checksum on Sampler
        print("\n--- Step 7: Verifying Weight Synchronization on Sampler ---")
        success = ray.get(sampler.verify_against_checksum.remote(expected_checksum=trainer_checksum["sha256"], timeout_sec=15.0))

        final_sampler_checksum = ray.get(sampler.get_current_checksum.remote())
        print(f"Sampler Final Checksum: {final_sampler_checksum['sha256'][:16]} (L1={final_sampler_checksum['l1_norm']:.4f})")

        print("\n" + "=" * 80)
        if success and final_sampler_checksum["sha256"] == trainer_checksum["sha256"]:
            print(">>> [SUCCESS] Mini RL Weight Synchronization with RaidenController PASSED!")
        else:
            print(">>> [FAILURE] Mini RL Weight Synchronization FAILED! Checksums do not match.")
        print("=" * 80)

    except Exception as e:
        import traceback
        print("\n" + "!" * 80)
        print(f">>> [REPRODUCED ERROR / EXCEPTION CAUGHT]:\n{traceback.format_exc()}")
        print("!" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
