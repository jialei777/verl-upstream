#!/usr/bin/env python3
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Submit Qwen3-4B-Base on v6e using this checkout and the hardware plugin."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
TOOLS = ROOT
CONTROL = ROOT

# Edit cluster and training settings here, or pass --config for an external JSON override.
DEFAULT_CONFIG = {
    "image": (
        "gcr.io/cloud-tpu-shared-capacity/verl-kueue-lixali@"
        "sha256:2da83c0d7e3078fa75e39d4b365f1941d14ac7533c21448f59a96b124d8f03a2"
    ),
    "data_manifest_uri": "gs://ubench-logs/lixali/verl-ray-kueue/assets/qwen3-4b-sha328a91d31223/data-manifest.json",
    "artifact_gcs_uri": "gs://ubench-logs/lixali/verl-ray-kueue/results",
    "backup_gcs_uri": "gs://lixali-tpu-storage/grpo/tpu-v6e",
    "artifact_upload_interval_seconds": 60,
    "run_id_stem": "qwen3-4b-base-grpo-v6e-250steps-32trainer32sampler-ppo-clip02-dual3-kl1e-3-no-prefix-seed1-plugin",
    "name_stem": "verl-qwen3-base-plugin",
    "source_snapshot": "source-snapshot",
    "manifest_file": "jobset.yaml",
    "artifact_layout": "{run_id}/backup-auto/pods/{worker_id}",
    "context": "gke_tpu-prod-env-one-vm_southamerica-west1_bodaborg-v6e-256-lcscld-c",
    "resources": {
        "hosts_per_role": 8,
        "physical_chips_per_host": 4,
        "logical_devices_per_host": 4,
        "rollout_tensor_parallel_size": 1,
        "actor_data_parallel_shard_size": 32,
        "actor_topology": "4x8",
        "rollout_topology": "2x2",
        "head_cpu": "12",
        "head_memory": "48Gi",
        "worker_cpu": "48",
        "worker_memory": "200Gi",
        "ephemeral_storage_request": "40Gi",
        "ephemeral_storage_limit": "150Gi",
        "active_deadline_seconds": 0,
        "start_timeout_seconds": 0,
        "worker_ack_timeout_seconds": 1800,
        "accelerator": "tpu-v6e-slice",
        "runtime_accelerator": "v6e",
        "worker_memory_limit": "400Gi",
        "worker_ephemeral_storage_request": "20Gi",
        "worker_ephemeral_storage_limit": "35Gi",
    },
    "training": {
        "steps": 250,
        "prompts_per_step": 128,
        "rollouts_per_prompt": 16,
        "responses_per_step": 2048,
        "micro_batch_size_per_gpu": 4,
        "validation": True,
        "validation_samples": 1849,
        "test_freq": 20,
        "val_before_train": True,
        "save_freq": -1,
        "checkpoints": False,
        "lr": 2e-06,
        "lr_warmup_steps": 10,
        "lr_decay_type": "cosine",
        "max_prompt_length": 512,
        "max_response_length": 2048,
        "seed": 1,
        "resume_mode": "disable",
        "prefix_caching": False,
        "bypass_mode": True,
        "thinking": True,
        "model_path": "/data/jialei/assets/hf/Qwen3-4B-Base",
        "extra_overrides": [
            "actor_rollout_ref.actor.use_torch_compile=False",
            "actor_rollout_ref.actor.torchtitan.use_torch_compile=False",
            "actor_rollout_ref.ref.torchtitan.use_torch_compile=False",
            "actor_rollout_ref.actor.torchtitan.spmd_backend=default",
            "actor_rollout_ref.ref.torchtitan.spmd_backend=default",
            "actor_rollout_ref.actor.torchtitan.max_seq_len=3072",
            "actor_rollout_ref.ref.torchtitan.max_seq_len=3072",
            "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
            "algorithm.use_kl_in_reward=False",
            "algorithm.rollout_correction.loss_type=ppo_clip",
            "algorithm.rollout_correction.rollout_is=token",
            "algorithm.rollout_correction.rollout_is_threshold=3.0",
            "algorithm.rollout_correction.rollout_is_batch_normalize=False",
            "algorithm.rollout_correction.rollout_rs=null",
            "actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode",
            "+actor_rollout_ref.actor.policy_loss.rollout_correction=${algorithm.rollout_correction}",
            "actor_rollout_ref.actor.use_kl_loss=True",
            "actor_rollout_ref.actor.kl_loss_coef=0.001",
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
            "actor_rollout_ref.rollout.calculate_log_probs=True",
            "actor_rollout_ref.rollout.checkpoint_engine.backend=tpu",
            "+trainer.log_response_token_probs=True",
            "actor_rollout_ref.actor.clip_ratio=0.2",
            "actor_rollout_ref.actor.clip_ratio_low=0.2",
            "actor_rollout_ref.actor.clip_ratio_high=0.2",
            "actor_rollout_ref.actor.clip_ratio_c=3.0",
            "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
            "data.val_files=/data/jialei/data/validation/gsm8k_math500_aime2024.parquet",
            "data.val_max_samples=-1",
            "data.custom_cls.path=/opt/run/source/recipe_validation_dataset.py",
            "data.custom_cls.name=FullValidationDataset",
            "actor_rollout_ref.rollout.prompt_length=1024",
            "actor_rollout_ref.rollout.max_model_len=3072",
            "actor_rollout_ref.rollout.max_num_batched_tokens=3072",
            "actor_rollout_ref.rollout.val_kwargs.do_sample=False",
            "actor_rollout_ref.rollout.val_kwargs.temperature=0.0",
            "actor_rollout_ref.rollout.val_kwargs.n=1",
            "custom_reward_function.path=/opt/run/source/recipe_reward_score.py",
            "custom_reward_function.name=compute_score",
        ],
    },
    "expected_worker_ids": [
        "head",
        "actor-0",
        "actor-1",
        "actor-2",
        "actor-3",
        "actor-4",
        "actor-5",
        "actor-6",
        "actor-7",
        "rollout-0",
        "rollout-1",
        "rollout-2",
        "rollout-3",
        "rollout-4",
        "rollout-5",
        "rollout-6",
        "rollout-7",
    ],
    "kubeconfig": "/home/lixali_google_com/gutianyu-rl/kueue-verl-ray/kubeconfig-v6e",
    "namespace": "default",
    "queue": "multislice-queue",
    "service_account": "default",
    "priority_class": "medium",
    "topology_aware_scheduling": False,
    "startup_wait_policy": "Wait without a startup deadline for eight trainer hosts and eight generator hosts "
    "(four physical TPU chips per host). No automatic retries.",
    "tpu_preflight": {
        "enabled": True,
        "file": "tpu-preflight.py",
        "timeout_seconds": 1800,
        "purpose": "Validate actual32chiptrainer and32independentTP1samplerchips, including "
        "user-updatedsamplerpatchactivation, beforetraining.",
    },
    "actor_memory": {"activation_checkpoint": "full", "reshard_after_forward": "always"},
    "training_script": "examples/tpu/grpo/run_qwen3_4b_torchtitan.sh",
    "status": "local_plugin_recipe",
    "model": {"repo": "Qwen/Qwen3-4B-Base", "revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539"},
    "validation": {
        "gsm8k": 1319,
        "math500": 500,
        "aime2024": 30,
        "total": 1849,
        "manifest": "assets/qwen3-4b-base-validation/validation_manifest.json",
        "prompt_limit": 1024,
        "generation": "greedy",
        "initial_step": 0,
        "every_steps": 20,
        "final_step": 250,
    },
    "training_prompt_limit": 512,
    "runtime_overlay": "qwen3_4b_base_250steps_runtime.py",
    "logging": {
        "response_min_max": True,
        "response_mean": True,
        "response_token_count": True,
        "raw_token_arrays": False,
        "trainer_and_generator": True,
        "validation_generator": True,
    },
    "model_cache_gcs_uri": "gs://lixali-tpu-storage/models/Qwen3-4B-Base/906bfd4b4dc7f14ee4320094d8b41684abff8539",
    "model_cache_reuse": True,
}

# Immutable validation data verified against the working recipe and its private cloud backup.
VALIDATION_INPUT = {
    "source": "gs://lixali-tpu-storage/grpo/tpu-v6e/qwen3-4b-base-grpo-v6e-250steps-32trainer32sampler-ppo-clip02-dual3-kl1e-3-no-prefix-seed1-20261007-115648-f776b0/backup-auto/inputs/jialei/data/validation/gsm8k_math500_aime2024.parquet",
    "destination": "/data/jialei/data/validation/gsm8k_math500_aime2024.parquet",
    "generation": "1791374250149415",
    "bytes": 753286,
    "sha256": "5899a3605cbd6f2326bd692e3c9de6a99c2ea6bb50715fd03e9056aa12800c52",
    "md5Hash": "AU7AfS+D5wYVCdLhnRbXig==",
    "crc32c": "64ZDiw==",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def run(command, cwd=None):
    subprocess.run(command, cwd=cwd, check=True)


def precision_file(plugin):
    for name in ("tpu_vllm_precision.py", "tpu_vllm_patches.py"):
        relative = "verl_hardware_plugin/rollout/" + name
        path = plugin / relative
        if path.is_file():
            functions = {node.name for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
            if {"patch_tpu_logprobs", "patch_tpu_sampler"}.issubset(functions):
                return relative
    raise ValueError("The hardware-plugin checkout must include the local FP32 precision patches.")


def precision_function_hashes(path):
    contents = path.read_text()
    return {
        node.name: hashlib.sha256(ast.get_source_segment(contents, node).encode()).hexdigest()
        for node in ast.parse(contents).body
        if isinstance(node, ast.FunctionDef) and node.name in {"patch_tpu_logprobs", "patch_tpu_sampler"}
    }


def snapshot_source(source, plugin, destination, cfg, launcher):
    """Snapshot both working trees, including local precision fixes."""
    names = (
        subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=source)
        .decode()
        .split("\0")
    )
    excludes = {
        ".git",
        ".codex",
        ".venv",
        "venv",
        "logs",
        "ray_logs",
        "tensorboard_log",
        "checkpoints",
        "__pycache__",
        "logs_compare_TPU_vs_GPU",
        ".mypy_cache",
    }
    records = {}
    destination.mkdir()
    for name in sorted(set(names)):
        rel = Path(name)
        if (
            not name
            or any(part in excludes for part in rel.parts)
            or rel.suffix in {".log", ".pt", ".bin", ".pyc", ".safetensors"}
        ):
            continue
        path = source / rel
        if not path.is_file():
            continue
        if path.is_symlink() and not path.resolve().is_relative_to(source):
            raise ValueError(f"Refusing external source symlink: {rel}")
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        records[name] = digest(path)
    training_script = Path(cfg["training_script"])
    if training_script.is_absolute() or ".." in training_script.parts:
        raise ValueError("Training script must be relative to the staged working tree.")
    staged_training = destination / training_script
    generated_training = not staged_training.is_file()
    if generated_training:
        staged_training.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TOOLS / "run_qwen3_4b_torchtitan.sh", staged_training)
    plugin_records = {}
    plugin_names = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "verl_hardware_plugin"],
            cwd=plugin,
        )
        .decode()
        .split("\0")
    )
    for name in sorted(set(plugin_names)):
        if not name or "__pycache__" in Path(name).parts or Path(name).suffix == ".pyc":
            continue
        path = plugin / name
        if not path.is_file():
            continue
        if path.is_symlink() and not path.resolve().is_relative_to(plugin):
            raise ValueError(f"Refusing external plugin symlink: {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        plugin_records[name] = digest(path)
    sampler = precision_file(plugin)
    original_precision = precision_function_hashes(plugin / sampler)
    # Adapt runtime placement only in the isolated snapshot.
    run([sys.executable, "-B", str(TOOLS / "overlay_plugin_v6e.py"), str(destination)])
    # Add detached response logging; the installer proves all original source
    # lines are recoverable byte for byte by removing only the new statements.
    run([sys.executable, "-B", str(TOOLS / "recipe_response_logging.py"), "--install", str(destination)])
    modified = [name for name, sha in (records | plugin_records).items() if digest(destination / name) != sha]
    logging_files = {
        "verl/workers/utils/losses.py",
        "verl/workers/engine_workers.py",
        "verl/trainer/ppo/v1/trainer_base.py",
    }
    allowed = {
        "verl_hardware_plugin/platforms/platform_tpu.py",
        "verl_hardware_plugin/rollout/tpu_vllm.py",
        "verl_hardware_plugin/rollout/tpu_vllm_patches.py",
    } | logging_files
    if set(modified) - allowed:
        raise ValueError(f"Unexpected computation/source modifications: {set(modified) - allowed}")
    if precision_function_hashes(destination / sampler) != original_precision:
        raise ValueError("Runtime placement changed the FP32 normalization or sampling implementation.")
    precision_files = {
        "verl/utils/torch_functional.py": source,
        "verl/workers/engine/torchtitan/transformer_impl.py": source,
        sampler: plugin,
        "verl_hardware_plugin/rollout/tpu_vllm_patches.py": plugin,
    }
    cfg["required_source_files"] = {
        name: {
            "sha256": digest(destination / name),
            "original_sha256": digest(root / name),
            "source": str(root / name),
            "reason": "Verify the staged precision source; its numerical functions are preserved unchanged.",
        }
        for name, root in precision_files.items()
    }
    for name, requirement in cfg["required_source_files"].items():
        if digest(destination / name) != requirement["sha256"]:
            raise ValueError(f"Runtime placement changed a precision fix: {name}")
    shutil.copy2(TOOLS / "recipe_validation_dataset.py", destination / "recipe_validation_dataset.py")
    shutil.copy2(TOOLS / "recipe_reward_score.py", destination / "recipe_reward_score.py")
    shutil.copy2(TOOLS / "source_runtime_check.py", destination / "source-runtime-check.py")
    shutil.copy2(CONTROL / "tpu_preflight.py", destination / "tpu-preflight.py")
    shutil.copy2(CONTROL / "runtime-env.yaml", destination / "runtime-env-kueue.yaml")
    entry = (CONTROL / "train.sh").read_text()
    check = (
        "python3 source-runtime-check.py "
        f"--model-path {cfg['training']['model_path']} "
        "--validation-file /data/jialei/data/validation/gsm8k_math500_aime2024.parquet "
        '--output "/tmp/verl_metrics/${RUN_ID}/source-runtime-check.json"\n'
    )
    entry = entry.replace('python3 - "$@"', check + 'python3 - "$@"', 1)
    (destination / "train-kueue.sh").write_text(entry)
    (destination / "train-kueue.sh").chmod(0o755)
    shutil.copy2(destination / "train-kueue.sh", launcher / "train.sh")
    write_json(destination / "run-training.json", cfg)
    provenance = {
        "source": str(source),
        "plugin_source": str(plugin),
        "plugin_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=plugin, text=True).strip(),
        "plugin_git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=plugin, text=True),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True),
        "original_file_sha256": records,
        "modified_snapshot_files": modified,
        "original_plugin_file_sha256": plugin_records,
        "model_and_sampler_files_unmodified": sampler not in modified,
        "precision_implementation_unmodified": True,
        "precision_function_sha256": original_precision,
        "loss_equations_unmodified": True,
        "logging_only_snapshot_files": sorted(logging_files),
        "training_script": cfg["training_script"],
        "training_script_sha256": digest(staged_training),
        "training_script_generated": generated_training,
        "sampler_sha256": plugin_records[sampler],
        "additional_recipe_files": [
            "recipe_validation_dataset.py",
            "recipe_reward_score.py",
            "recipe_response_logging.py",
            "recipe_v6e_runtime.py",
            "source-runtime-check.py",
            "train-kueue.sh",
            "tpu-preflight.py",
            "runtime-env-kueue.yaml",
            cfg["training_script"],
        ],
    }
    write_json(destination / "submission-source.json", provenance)
    # Verify copy checksums and leave both source checkouts untouched.
    changed_originals = [name for name, sha in records.items() if digest(source / name) != sha]
    if changed_originals:
        raise ValueError(f"Source changed during snapshot; retry: {changed_originals}")
    changed_plugin = [name for name, sha in plugin_records.items() if digest(plugin / name) != sha]
    if changed_plugin:
        raise ValueError(f"Plugin changed during snapshot; retry: {changed_plugin}")
    return provenance


def validate_configuration(snapshot, cfg):
    """Resolve the actual training command with Hydra, without Torch or Ray."""
    import shlex
    import tempfile
    from unittest.mock import patch

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    wrapper = (snapshot / "train-kueue.sh").read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    captured = {}

    def capture(file, argv, environment):
        captured.update(argv=argv, environment=environment)

    previous = Path.cwd()
    try:
        os.chdir(snapshot)
        with (
            patch.dict(os.environ, {"RUN_ID": cfg["run_id"]}),
            patch.object(os, "execvpe", capture),
            patch.object(sys, "argv", ["-"]),
        ):
            exec(compile(wrapper, "train-kueue", "exec"), {"__name__": "__main__"})
    finally:
        os.chdir(previous)
    with tempfile.TemporaryDirectory(prefix="base-recipe-command-check-") as temporary:
        tmp = Path(temporary)
        capture_py = tmp / "capture.py"
        capture_py.write_text(
            "import json,os,sys\nfrom pathlib import Path\n"
            'Path(os.environ["RECIPE_CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n'
        )
        fake = tmp / "python3"
        fake.write_text(
            "#!/usr/bin/env bash\nexec " + shlex.quote(sys.executable) + " " + shlex.quote(str(capture_py)) + ' "$@"\n'
        )
        fake.chmod(0o755)
        environment = dict(
            captured["environment"],
            PATH=str(tmp) + os.pathsep + os.environ["PATH"],
            RECIPE_CAPTURE=str(tmp / "argv.json"),
        )
        subprocess.run(captured["argv"], cwd=snapshot, env=environment, check=True, capture_output=True)
        argv = json.loads((tmp / "argv.json").read_text())
    assert argv[:2] == ["-m", "verl.trainer.main_ppo"]
    with initialize_config_dir(config_dir=str(snapshot / "verl/trainer/config"), version_base=None):
        resolved = OmegaConf.to_container(compose(config_name="ppo_trainer", overrides=argv[2:]), resolve=True)
    actor = resolved["actor_rollout_ref"]["actor"]
    rollout = resolved["actor_rollout_ref"]["rollout"]
    assert resolved["trainer"]["total_training_steps"] == cfg["training"]["steps"]
    assert resolved["trainer"]["val_before_train"] == cfg["training"]["val_before_train"]
    assert resolved["trainer"]["test_freq"] == cfg["training"]["test_freq"]
    assert actor["clip_ratio_low"] == actor["clip_ratio_high"] == 0.2 and actor["clip_ratio_c"] == 3.0
    assert actor["use_kl_loss"] and actor["kl_loss_coef"] == 0.001
    assert actor["policy_loss"]["rollout_correction"]["loss_type"] == "ppo_clip"
    assert rollout["tensor_model_parallel_size"] == 1 and rollout["prompt_length"] == 1024
    assert resolved["data"]["max_prompt_length"] == 512 and resolved["data"]["val_max_samples"] == -1
    assert resolved["data"]["seed"] == actor["data_loader_seed"] == actor["torchtitan"]["seed"] == rollout["seed"] == 1
    assert resolved["actor_rollout_ref"]["ref"]["torchtitan"]["seed"] == rollout["engine_kwargs"]["vllm"]["seed"] == 1
    write_json(snapshot.parent.parent / "resolved-config.json", resolved)
    write_json(snapshot.parent.parent / "training-command.json", argv)


def main():
    global TOOLS, CONTROL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional complete JSON configuration.")
    parser.add_argument("--steps", type=int, help="Override the training-step count; short runs use no LR warmup.")
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip benchmark evaluation for a training smoke test."
    )
    parser.add_argument(
        "--source", type=Path, default=REPO_ROOT, help="VERL working tree to snapshot (default: this checkout)."
    )
    parser.add_argument(
        "--plugin-source",
        type=Path,
        default=Path(os.environ.get("PLUGIN_REPO", str(REPO_ROOT.parent / "verl-hardware-plugin"))),
        help="Patched hardware-plugin checkout (default: PLUGIN_REPO or sibling verl-hardware-plugin).",
    )
    parser.add_argument("--output-dir", type=Path, help="New workspace directory for this submission.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--render-only", action="store_true", help="Local source/config checks; no cloud calls or job submission."
    )
    mode.add_argument(
        "--prepare-only", action="store_true", help="Stage and verify cloud inputs without submitting training."
    )
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text()) if args.config else copy.deepcopy(DEFAULT_CONFIG)
    if args.steps is not None:
        if args.steps <= 0:
            parser.error("--steps must be positive.")
        cfg["training"]["steps"] = args.steps
        cfg["validation"]["final_step"] = args.steps
        cfg["run_id_stem"] = cfg["run_id_stem"].replace("250steps", f"{args.steps}steps")
        if args.steps <= cfg["training"]["lr_warmup_steps"]:
            cfg["training"]["lr_warmup_steps"] = 0
            cfg["training"]["extra_overrides"] = [
                "actor_rollout_ref.actor.optim.lr_warmup_steps=0"
                if value.startswith("actor_rollout_ref.actor.optim.lr_warmup_steps=")
                else value
                for value in cfg["training"]["extra_overrides"]
            ]
    if args.skip_validation:
        cfg["training"].update(validation=False, val_before_train=False, test_freq=-1)
    source = args.source.expanduser().resolve()
    plugin = args.plugin_source.expanduser().resolve()
    cfg["source"] = str(source)
    cfg["plugin_source"] = str(plugin)
    cfg["validation"]["cloud_input"] = copy.deepcopy(cfg["validation"].get("cloud_input", VALIDATION_INPUT))
    try:
        precision_file(plugin)
    except ValueError as error:
        parser.error(str(error))
    if not (source / "verl/trainer/main_ppo.py").is_file():
        parser.error("The requested VERL working tree must exist.")
    if not (source / cfg["training_script"]).is_file() and cfg["training_script"] != DEFAULT_CONFIG["training_script"]:
        parser.error("The custom training script must exist in the requested working tree.")
    import yaml

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    cfg["name"] = cfg["name_stem"] + "-" + stamp[4:] + "-" + suffix
    if len(cfg["name"]) > 43:
        parser.error("JobSet name must be at most 43 characters to leave room for generated rollout pod names.")
    cfg["run_id"] = cfg["run_id_stem"] + "-" + stamp + "-" + suffix
    cfg["prefix"] = "gs://ubench-logs/lixali/verl-ray-kueue/" + cfg["run_id"]
    cfg["data_manifest_uri"] = cfg["prefix"] + "/data-manifest.json"
    cfg["source_snapshot"] = "source-snapshot"
    cfg["manifest_file"] = "jobset.yaml"
    old_kubeconfig = cfg.pop("kubeconfig", None)
    if not os.environ.get("KUBECONFIG") and old_kubeconfig and Path(old_kubeconfig).is_file():
        os.environ["KUBECONFIG"] = old_kubeconfig
    root = (args.output_dir or REPO_ROOT.parent / "verl-launches" / cfg["name"]).expanduser().resolve()
    if root.is_relative_to(source) or root.is_relative_to(plugin):
        parser.error("Submission files must be outside the VERL and hardware-plugin checkouts.")
    root.mkdir(parents=True, exist_ok=False)
    launcher = root / "kueue-verl-ray"
    launcher.mkdir()
    from qwen3_4b_base_250steps_runtime import materialize

    TOOLS = root / "runtime"
    materialize(TOOLS)
    CONTROL = TOOLS / "launcher"
    sys.path.insert(0, str(TOOLS))
    from prepare_qwen3_4b_base_model import MODEL_REPO, MODEL_REVISION
    from shared_qwen3_4b_base_model import MODEL_CACHE_PREFIX, ensure_model_cache

    if cfg["model"] != {"repo": MODEL_REPO, "revision": MODEL_REVISION}:
        parser.error("The model configuration must match the immutable Base model manifest.")
    validation_proof_path = TOOLS / cfg["validation"]["manifest"]
    validation_proof = json.loads(validation_proof_path.read_text())
    validation_input = copy.deepcopy(cfg["validation"].get("cloud_input", VALIDATION_INPUT))
    if (
        not validation_input
        or validation_input["sha256"] != validation_proof["combined"]["sha256"]
        or validation_proof["combined"]["rows"] != 1849
    ):
        raise ValueError("Validation cloud input does not match its verified 1849-row manifest.")
    cfg["validation"]["cloud_input"] = validation_input

    if not validation_proof["token_length_audit"]["all_rows_fit"]:
        raise ValueError("Validation prompts exceed the configured limit.")
    for name in (
        "submit_with_backup.py",
        "artifact_mirror.py",
        "build_manifest.py",
        "bootstrap.py",
        "runtime-env.yaml",
        "tpu_preflight.py",
    ):
        shutil.copy2(CONTROL / name, launcher / name)
    for name in (
        "overlay_plugin_v6e.py",
        "source_runtime_check.py",
        "recipe_validation_dataset.py",
        "recipe_reward_score.py",
        "recipe_response_logging.py",
    ):
        shutil.copy2(TOOLS / name, launcher / name)
    snapshot = launcher / cfg["source_snapshot"]
    provenance = snapshot_source(source, plugin, snapshot, cfg, launcher)
    validate_configuration(snapshot, cfg)
    config_path = launcher / "run.json"
    write_json(config_path, cfg)
    bundle = launcher / "source-snapshot.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for path in sorted(snapshot.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                tar.add(path, arcname=path.relative_to(snapshot).as_posix(), recursive=False)
    bundle_sha = digest(bundle)
    (launcher / "source-snapshot.tar.gz.sha256").write_text(bundle_sha + "\n")
    run([sys.executable, str(launcher / "build_manifest.py"), "--config", str(config_path)], cwd=root)
    # Rename the original 17-pod disruption protection; never retain its old owner UID.
    protection = {
        "apiVersion": "policy/v1",
        "kind": "PodDisruptionBudget",
        "metadata": {
            "name": cfg["name"] + "-protect",
            "namespace": cfg["namespace"],
            "labels": {"verl-run": cfg["name"]},
        },
        "spec": {
            "minAvailable": 17,
            "unhealthyPodEvictionPolicy": "AlwaysAllow",
            "selector": {"matchLabels": {"verl-run": cfg["name"]}},
        },
    }
    protection_path = root / "disruption-protection.yaml"
    protection_path.write_text(yaml.safe_dump(protection, sort_keys=False))
    proof = {
        "jobset": cfg["name"],
        "run_id": cfg["run_id"],
        "source": str(source),
        "plugin_source": str(plugin),
        "git_commit": provenance["git_commit"],
        "source_sha256": bundle_sha,
        "source_model_and_sampler_files_unmodified": provenance["model_and_sampler_files_unmodified"],
        "precision_implementation_unmodified": True,
        "loss_equations_unmodified": True,
        "model": cfg["model"],
        "response_log_probs": "per-response trainer/generator min,max,mean and token counts",
        "validation": cfg["validation"],
        "seed": 1,
        "configuration_verified": True,
        "mode": "render-only" if args.render_only else "prepare-only" if args.prepare_only else "submit",
    }
    write_json(root / "recipe-verification.json", proof)
    print(f"Local source/configuration checks passed. Files: {root}", flush=True)
    if args.render_only:
        print("No model weights downloaded, cloud writes, or job submission.")
        return
    kubectl = ["kubectl", "--context", cfg["context"], "--namespace", cfg["namespace"]]
    run(kubectl + ["get", "localqueue", cfg["queue"]])
    # Catch admission/schema errors before cloud input preparation.
    run(kubectl + ["apply", "--dry-run=server", "-f", str(root / cfg["manifest_file"])])
    sys.path.insert(0, str(launcher))
    from artifact_mirror import CloudStorage

    cloud = CloudStorage()
    cloud.preflight(cfg["backup_gcs_uri"].rstrip("/") + "/" + cfg["run_id"] + "/backup-auto/control")
    model_cache = ensure_model_cache(cloud, prefix=cfg.get("model_cache_gcs_uri", MODEL_CACHE_PREFIX), reuse_only=True)
    model_cache_record = root / "model-cache.json"
    write_json(model_cache_record, model_cache)
    # Run-specific objects prevent concurrent submissions from overwriting the
    # exact generations that the original infrastructure helper will archive.
    asset_prefix = cfg["prefix"] + "/inputs"
    entries = []
    for item in model_cache["files"]:
        entries.append(
            {
                "source": item["source"],
                "destination": cfg["training"]["model_path"] + "/" + item["name"],
                "generation": item["generation"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
        )
    # Reuse the exact pinned GSM8K training input; validation combines all three sources.
    old_inputs = json.loads((TOOLS / "original-gsm8k-inputs.json").read_text())
    for item in old_inputs:
        if not item["destination"].endswith("/gsm8k/train.parquet"):
            continue
        expected = cloud.stat(item["source"], generation=item["generation"])
        staged_uri = asset_prefix + "/data/gsm8k-train.parquet"
        staged = cloud.copy(item["source"], staged_uri, expected=expected)
        entries.append({**item, "source": staged_uri, "generation": staged["generation"]})
    validation_metadata = cloud.stat(validation_input["source"], generation=validation_input["generation"])
    if int(validation_metadata["size"]) != validation_input["bytes"] or str(validation_metadata["generation"]) != str(
        validation_input["generation"]
    ):
        raise ValueError("The pinned validation cloud object has an unexpected size or generation.")
    meta = cloud.copy(
        validation_input["source"], asset_prefix + "/data/validation.parquet", expected=validation_metadata
    )
    entries.append(
        {
            "source": asset_prefix + "/data/validation.parquet",
            "destination": "/data/jialei/data/validation/gsm8k_math500_aime2024.parquet",
            "generation": meta["generation"],
            "bytes": validation_input["bytes"],
            "sha256": validation_input["sha256"],
        }
    )
    data_manifest = root / "data-manifest.json"
    write_json(data_manifest, entries)
    cloud.put_file(data_manifest, cfg["data_manifest_uri"])
    cloud.put_file(validation_proof_path, cfg["prefix"] + "/validation-provenance.json")
    backup_control = cfg["backup_gcs_uri"].rstrip("/") + "/" + cfg["run_id"] + "/backup-auto/control/"
    review_files = [
        Path(__file__),
        ROOT / "submit_qwen3_4b_base_250steps.sh",
        config_path,
        ROOT / "qwen3_4b_base_250steps_runtime.py",
        root / "recipe-verification.json",
        root / "resolved-config.json",
        root / "training-command.json",
        data_manifest,
        protection_path,
        snapshot / "submission-source.json",
        validation_proof_path,
        model_cache_record,
    ]
    review_files.extend(
        TOOLS / name
        for name in (
            "overlay_plugin_v6e.py",
            "recipe_response_logging.py",
            "recipe_reward_score.py",
            "recipe_validation_dataset.py",
            "source_runtime_check.py",
            "prepare_qwen3_4b_base_model.py",
            "shared_qwen3_4b_base_model.py",
        )
    )
    if args.config:
        requested_config = root / "requested-config.json"
        shutil.copy2(args.config.resolve(), requested_config)
        review_files.append(requested_config)
    for path in review_files:
        cloud.put_file(path, backup_control + path.name)
    # Original submission helper handles input archiving, the keyless backup mirror,
    # server dry-run and final admission. Omit --fresh to retain this prepared snapshot.
    command = [sys.executable, str(launcher / "submit_with_backup.py"), "--config", str(config_path)]
    if args.prepare_only:
        command.append("--prepare-only")
    run(command, cwd=root)
    if not args.prepare_only:
        job = json.loads(subprocess.check_output(kubectl + ["get", "jobset", cfg["name"], "-o", "json"], text=True))
        protection["metadata"]["ownerReferences"] = [
            {
                "apiVersion": job["apiVersion"],
                "kind": "JobSet",
                "name": cfg["name"],
                "uid": job["metadata"]["uid"],
                "controller": False,
                "blockOwnerDeletion": False,
            }
        ]
        protection_path.write_text(yaml.safe_dump(protection, sort_keys=False))
        run(kubectl + ["apply", "-f", str(protection_path)])
        print(
            "Submitted "
            + cfg["name"]
            + "; backup: "
            + cfg["backup_gcs_uri"].rstrip("/")
            + "/"
            + cfg["run_id"]
            + "/backup-auto/"
        )


if __name__ == "__main__":
    main()
