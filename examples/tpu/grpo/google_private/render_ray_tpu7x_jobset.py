#!/usr/bin/env python3
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Renders a Ray-on-JobSet manifest for verl GRPO on GKE TPU v7x clusters without KubeRay.

    python3 render_ray_tpu7x_jobset.py --name NAME --image IMAGE --bucket gs://BUCKET/DIR --model Qwen3.5-35B-A3B \\
        --trainer 2x2x1:1 --rollout 2x2x1:1 [--namespace NS] [--reservation R] [--queue Q] [--priority P] \\
        | kubectl apply -f -

The JobSet has a CPU Ray head pod and one TPU Job per slice (trainer, rollout). Every pod downloads
``<bucket>/assets/hf/<model>`` (the head pod only the config and tokenizer) and ``<bucket>/data/gsm8k`` (the
bucket layout of examples/tpu/grpo/README.md, step 3) to ``/data``,
then starts Ray; each TPU host joins with 8 ``TPU`` units (one per TensorCore) and its slice's ``tpu-group-<i>``
resource.
"""

import argparse
import json

# Downloads the model (FULL_MODEL=0: config and tokenizer only) and GSM8K from BUCKET_PREFIX to /data.
_DOWNLOAD_SCRIPT = r"""mkdir -p /data/assets/hf /data/data
python3 - <<'PYEOF'
import os
from google.cloud import storage
from google.cloud.storage import transfer_manager

bucket_name, _, prefix = os.environ["BUCKET_PREFIX"][len("gs://") :].partition("/")
bucket = storage.Client().bucket(bucket_name)
model, full = os.environ["MODEL_NAME"], os.environ.get("FULL_MODEL", "1") == "1"


def fetch(src_prefix, dst_dir, skip_weights=False):
    blobs = bucket.list_blobs(prefix=src_prefix + "/")
    names = [b.name[len(src_prefix) + 1 :] for b in blobs if not (skip_weights and b.name.endswith(".safetensors"))]
    for n in names:
        os.makedirs(os.path.dirname(os.path.join(dst_dir, n)) or dst_dir, exist_ok=True)
    results = transfer_manager.download_many_to_path(
        bucket, names, destination_directory=dst_dir, blob_name_prefix=src_prefix + "/", max_workers=32,
        worker_type="thread",
    )
    for r in results:
        if isinstance(r, Exception):
            raise r
    print("downloaded", len(names), "files to", dst_dir, flush=True)


fetch(f"{prefix}/assets/hf/{model}", f"/data/assets/hf/{model}", skip_weights=not full)
fetch(f"{prefix}/data/gsm8k", "/data/data/gsm8k")
PYEOF
"""

HEAD_SCRIPT = (
    "set -ex\nexport FULL_MODEL=0\n"
    + _DOWNLOAD_SCRIPT
    + "ray start --head --port=6379 --dashboard-host=0.0.0.0 --num-cpus=16 --num-gpus=0 --block\n"
)

WORKER_SCRIPT = (
    "set -ex\n"
    + _DOWNLOAD_SCRIPT
    + r"""until ray health-check --address "${RAY_HEAD}:6379" >/dev/null 2>&1; do sleep 5; done
# v7x: one Ray TPU unit per TensorCore (8 per host); Ray checks the count against TPU_VISIBLE_CHIPS.
export TPU_VISIBLE_CHIPS=0,1,2,3,4,5,6,7
# GKE treats every TPU Job of the JobSet as a slice of one multislice job; trainer and rollout are separate meshes.
unset $(env | grep -o "^MEGASCALE_[A-Z_]*")
ray start --address="${RAY_HEAD}:6379" --resources="{\"TPU\": 8, \"${SLICE_GROUP}\": 1}" --block
"""
)

# The XLA flags the Qwen3.5 GatedDeltaNet backward needs on TPU v7x (see run_qwen3_5_torchtitan.sh).
_LIBTPU_INIT_ARGS = (
    "--xla_tpu_use_enhanced_launch_barrier=false "
    "--xla_tpu_enable_bitcast_reduce_fusion_graduation=false "
    "--xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false"
)


def _pod_spec(args, **spec):
    """Pod spec fields shared by the head and TPU pods: priority class (if any) plus ``spec``."""
    if args.priority:
        spec["priorityClassName"] = args.priority
    return {"restartPolicy": "Never", **spec}


def tpu_job(name, topology, hosts, group, args):
    node_selector = {
        "cloud.google.com/gke-tpu-accelerator": "tpu7x",
        "cloud.google.com/gke-tpu-topology": topology,
    }
    tolerations = [{"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"}]
    if args.reservation:
        node_selector["cloud.google.com/reservation-name"] = args.reservation
        tolerations.append(
            {
                "key": "cloud.google.com/reservation-name",
                "operator": "Equal",
                "value": args.reservation,
                "effect": "NoSchedule",
            }
        )
    container = {
        "name": "ray-worker",
        "image": args.image,
        "imagePullPolicy": "Always",
        "securityContext": {"privileged": True},
        "env": [
            {"name": "BUCKET_PREFIX", "value": args.bucket},
            {"name": "MODEL_NAME", "value": args.model},
            {"name": "SLICE_GROUP", "value": group},
            {"name": "RAY_HEAD", "value": f"{args.name}-head-0-0.{args.name}"},
            {"name": "TPU_ACCELERATOR_TYPE", "value": "tpu7x"},
            {"name": "LIBTPU_INIT_ARGS", "value": _LIBTPU_INIT_ARGS},
            {"name": "RAY_memory_monitor_refresh_ms", "value": "0"},
            {"name": "RAY_memory_usage_threshold", "value": "0.99"},
        ],
        "command": ["bash", "-c", WORKER_SCRIPT],
        "resources": {
            "requests": {"google.com/tpu": "4", "cpu": "64", "memory": "600Gi"},
            "limits": {"google.com/tpu": "4", "memory": "600Gi"},
        },
        "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}, {"name": "data", "mountPath": "/data"}],
    }
    # A multi-host slice is one node pool: keep its pods together. Single-host slices share node pools (several
    # 2x2x1 hosts per pool), so they must not claim a whole pool.
    annotations = {"alpha.jobset.sigs.k8s.io/exclusive-topology": "cloud.google.com/gke-nodepool"} if hosts > 1 else {}
    pod_spec = _pod_spec(
        args,
        nodeSelector=node_selector,
        tolerations=tolerations,
        containers=[container],
        volumes=[
            {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "200Gi"}},
            {"name": "data", "emptyDir": {}},
        ],
    )
    return {
        "name": name,
        "replicas": 1,
        "template": {
            "spec": {
                "backoffLimit": 0,
                "completionMode": "Indexed",
                "completions": hosts,
                "parallelism": hosts,
                "template": {"metadata": {"annotations": annotations}, "spec": pod_spec},
            }
        },
    }


def head_job(args):
    container = {
        "name": "ray-head",
        "image": args.image,
        "imagePullPolicy": "Always",
        "env": [
            {"name": "BUCKET_PREFIX", "value": args.bucket},
            {"name": "MODEL_NAME", "value": args.model},
            {"name": "TPU_ACCELERATOR_TYPE", "value": "tpu7x"},
            {"name": "RAY_memory_monitor_refresh_ms", "value": "0"},
        ],
        "command": ["bash", "-c", HEAD_SCRIPT],
        "resources": {"requests": {"cpu": "16", "memory": "64Gi"}, "limits": {"memory": "64Gi"}},
        "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
    }
    pod_spec = _pod_spec(
        args,
        nodeSelector=dict(kv.split("=", 1) for kv in args.head_node_selector),
        tolerations=[
            {"key": key, "operator": "Equal", "value": value, "effect": "NoSchedule"}
            for key, value in (kv.split("=", 1) for kv in args.head_toleration)
        ],
        containers=[container],
        volumes=[{"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"}}],
    )
    return {
        "name": "head",
        "replicas": 1,
        "template": {"spec": {"backoffLimit": 0, "completions": 1, "parallelism": 1, "template": {"spec": pod_spec}}},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="JobSet name (also the Ray head's DNS prefix)")
    p.add_argument("--namespace", default="default")
    p.add_argument("--image", required=True, help="Image built from examples/tpu/grpo/Dockerfile")
    p.add_argument("--bucket", required=True, help="gs://BUCKET[/DIR] holding assets/hf/<model> and data/gsm8k")
    p.add_argument("--model", required=True, help="Directory under <bucket>/assets/hf, e.g. Qwen3.5-35B-A3B")
    p.add_argument("--trainer", default="2x2x1:1", help="Trainer slice as <topology>:<hosts>")
    p.add_argument("--rollout", default="2x2x1:1", help="Rollout slice as <topology>:<hosts>")
    p.add_argument("--reservation", default="", help="GCE reservation of the TPU nodes (optional)")
    p.add_argument("--queue", default="", help="Kueue LocalQueue (optional)")
    p.add_argument("--priority", default="", help="PriorityClass, also used as the Kueue priority class (optional)")
    p.add_argument(
        "--head-node-selector",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Node selector of the CPU head pod (repeatable)",
    )
    p.add_argument(
        "--head-toleration",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="NoSchedule toleration of the CPU head pod (repeatable)",
    )
    args = p.parse_args()

    labels = {}
    if args.queue:
        labels["kueue.x-k8s.io/queue-name"] = args.queue
    if args.queue and args.priority:
        labels["kueue.x-k8s.io/priority-class"] = args.priority
    t_topo, t_hosts = args.trainer.split(":")
    r_topo, r_hosts = args.rollout.split(":")
    jobset = {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {"name": args.name, "namespace": args.namespace, "labels": labels},
        "spec": {
            "failurePolicy": {"maxRestarts": 0},
            "network": {"enableDNSHostnames": True, "publishNotReadyAddresses": True},
            "replicatedJobs": [
                head_job(args),
                tpu_job("trainer", t_topo, int(t_hosts), "tpu-group-0", args),
                tpu_job("rollout", r_topo, int(r_hosts), "tpu-group-1", args),
            ],
        },
    }
    print(json.dumps(jobset, indent=1))


if __name__ == "__main__":
    main()
