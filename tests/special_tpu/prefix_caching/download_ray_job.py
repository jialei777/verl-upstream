"""Download a finished prefix-cache probe and export its prompt/response records."""

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import urlopen

from prefix_cache_log import parse_log

# Run inside the execution pod using only the standard library. Nothing is
# written on the pod: the archive is streamed to the local collector.
REMOTE_COLLECT = r"""
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

job, config = json.loads(sys.argv[1]), json.loads(sys.argv[2])
submission = job["submission_id"]
sessions = {p.parent.parent.resolve() for p in Path("/tmp/ray").glob(
    "session_*/logs/job-driver-" + submission + ".log"
)}
if len(sessions) != 1:
    raise RuntimeError(f"Expected one retained session for {submission}, found {sessions}")
session = sessions.pop()
driver = session / "logs" / f"job-driver-{submission}.log"
files = [(driver, "ray_logs/rollout/" + driver.name)]
warnings = []
supervisor = session / "logs/jobs" / f"supervisor-{submission}.log"
if supervisor.is_file():
    files.append((supervisor, "ray_logs/rollout/jobs/" + supervisor.name))
else:
    warnings.append("The job supervisor log is no longer available.")

package = Path(job["runtime_env"]["working_dir"]).name.removesuffix(".zip")
working_dir = session / "runtime_resources/working_dir_files" / package
for name in ("hybrid_pool_e2e.py", "hybrid_pool_gsm8k.py", config["launcher"]):
    source = working_dir / name
    if source.is_file():
        files.append((source, "launch/" + name))
    else:
        warnings.append(f"Submitted source no longer available: {source}")

mode = "on" if config["prefix_cache"] else "off"
result_dir = job["runtime_env"].get("env_vars", {}).get("E2E_GSM8K_RESULT_DIR", "/tmp/hybrid_pool_gsm8k")
result_dir = Path(result_dir)
if not result_dir.is_absolute():
    result_dir = working_dir / result_dir
probe_log = result_dir / f"prefix_{mode}.log"
driver_bytes = driver.read_bytes()
snapshots = {driver: (driver_bytes, driver.stat().st_mtime)}
if probe_log.is_file():
    payload = probe_log.read_bytes()
    mtime = probe_log.stat().st_mtime
    if payload and job["start_time"] / 1000 - 2 <= mtime <= job["end_time"] / 1000 + 5 \
            and driver_bytes.endswith(payload):
        files.append((probe_log, f"prefix_{mode}.log"))
        snapshots[probe_log] = (payload, mtime)
    else:
        warnings.append("The shared probe log differs from this job; it may have been overwritten. Use ray-job.log.")
else:
    warnings.append("The shared probe log is no longer available. Use ray-job.log.")

manifest = {"session": str(session), "working_dir": str(working_dir), "files": [], "warnings": warnings}
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
    for source, target in files:
        if source in snapshots:
            payload, mtime = snapshots[source]
        else:
            payload, mtime = source.read_bytes(), source.stat().st_mtime
        info = tarfile.TarInfo(target)
        info.size = len(payload)
        info.mtime = mtime
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(payload))
        manifest["files"].append({"archive_path": target, "source_path": str(source),
                                  "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    payload = (json.dumps(manifest, indent=2) + "\n").encode()
    info = tarfile.TarInfo("download-manifest-rollout.json")
    info.size = len(payload)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))
"""


def get_json(url):
    """Read a JSON response from the Ray dashboard."""
    with urlopen(url, timeout=60) as response:
        return json.load(response)


def write_json(path, value):
    """Write a readable JSON artifact."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def kubectl_json(command, *args):
    """Run a read-only kubectl command and decode its JSON output."""
    completed = subprocess.run([*command, *args, "-o", "json"], check=True, capture_output=True, timeout=60)
    return json.loads(completed.stdout)


def slug(value):
    """Convert a metadata value to one safe directory-name component."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value)).strip(".-") or "unknown"


def download_pod_files(command, pod, container, job, config, run):
    """Stream, extract, and verify artifacts from the execution pod."""
    archive_path = run / "downloaded-artifacts.tar.gz"
    partial = archive_path.with_suffix(".gz.partial")
    with partial.open("wb") as output:
        completed = subprocess.run(
            [
                *command,
                "exec",
                pod,
                "-c",
                container,
                "--",
                "python3",
                "-c",
                REMOTE_COLLECT,
                json.dumps(job),
                json.dumps(config),
            ],
            stdout=output,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    if completed.returncode:
        raise RuntimeError(f"Pod download failed: {completed.stderr.decode(errors='replace')}")
    with tarfile.open(partial, "r:gz") as archive:
        archive.extractall(run, filter="data")
    manifest = json.loads((run / "download-manifest-rollout.json").read_text())
    for item in manifest["files"]:
        payload = (run / item["archive_path"]).read_bytes()
        if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Downloaded artifact failed checksum verification: {item['archive_path']}")
    archived_log = run / "ray_logs/rollout" / f"job-driver-{job['submission_id']}.log"
    # The Ray API reads in text mode, normalizing progress-bar carriage returns.
    # Verify the same text while retaining the pod's original bytes and hash.
    if archived_log.read_text(encoding="utf-8") != (run / "ray-job.log").read_text(encoding="utf-8"):
        raise RuntimeError("The pod driver log differs from the downloaded Ray API log.")
    partial.replace(archive_path)
    return manifest


def main():
    """Collect one finished probe without submitting or modifying any Ray job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_id", help="Ray submission ID, for example raysubmit_DKk2dPpHaiKwSrYb")
    parser.add_argument("--address", default="http://127.0.0.1:23334", help="Ray dashboard URL")
    parser.add_argument("--context", help="Kubernetes context (default: current context)")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--container", help="Execution container (default: discover ray-worker or ray-head)")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "logs")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.submission_id):
        parser.error("submission_id must contain only letters, numbers, underscores, or hyphens")
    if sys.version_info < (3, 12):
        parser.error("Python 3.12 or later is required")

    endpoint = f"{args.address.rstrip('/')}/api/jobs/{quote(args.submission_id, safe='')}"
    job = get_json(endpoint)
    if job.get("submission_id") != args.submission_id:
        raise RuntimeError("The dashboard returned a different submission ID.")
    if job["status"] != "SUCCEEDED":
        raise RuntimeError(f"Job is {job['status']}; this exporter requires a successful, complete GSM8K probe.")
    logs = get_json(endpoint + "/logs")["logs"]
    config, records, summary = parse_log(logs)

    collected = datetime.now(timezone.utc)
    model = slug(config["model"].rsplit("/", 1)[-1].lower())
    mode = "on" if config["prefix_cache"] else "off"
    tp, dp = config.get("tensor_parallel_size", "unknown"), config.get("data_parallel_size", "unknown")
    job_id = job.get("job_id") or "no-ray-job-id"
    name = (
        f"prefix_cache_gsm8k_{model}_tp{tp}-dp{dp}-gen_prefix-{mode}_"
        f"{args.submission_id}_{slug(job_id)}_{collected:%Y%m%d-%H%M%S}"
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    run = args.output_root.resolve() / name
    run.mkdir()  # Refuse to overwrite a prior export with the same name.
    print(f"Collecting into {run}", flush=True)
    (run / "launch").mkdir()
    (run / "ray-job.log").write_text(logs, encoding="utf-8")
    write_json(run / "ray-job.json", job)
    write_json(run / "resolved-config.json", config)
    write_json(run / "summary.json", summary)
    write_json(run / "launch/runtime-env.json", job["runtime_env"])
    (run / "launch/entrypoint.sh").write_text("#!/usr/bin/env bash\n" + job["entrypoint"] + "\n", encoding="utf-8")
    write_json(
        run / "launch/launch.json",
        {
            "submission_id": args.submission_id,
            "ray_job_id": job.get("job_id"),
            "entrypoint": job["entrypoint"],
            "start_time_ms": job["start_time"],
            "end_time_ms": job["end_time"],
        },
    )
    generations = run / "generations/rollout"
    generations.mkdir(parents=True)
    for step in sorted({record["step"] for record in records}):
        with (generations / f"{step}.jsonl").open("w", encoding="utf-8") as output:
            for record in records:
                if record["step"] == step:
                    record["uid"] = f"{args.submission_id}:{step}:{record['prompt_row']}:{record['sample_index']}"
                    record["submission_id"] = args.submission_id
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")

    context = (
        args.context
        or subprocess.check_output(
            ["kubectl", "config", "current-context"],
            text=True,
            timeout=30,
        ).strip()
    )
    command = ["kubectl", "--context", context, "--namespace", args.namespace, "--request-timeout=30s"]
    driver_ip = urlparse(job.get("driver_agent_http_address") or "").hostname
    pods = kubectl_json(command, "get", "pods")["items"]
    candidates = [pod for pod in pods if driver_ip and pod.get("status", {}).get("podIP") == driver_ip]
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot find the execution pod at {driver_ip}. Job log and JSONL saved in {run}")
    pod = candidates[0]
    pod_name = pod["metadata"]["name"]
    containers = [c["name"] for c in pod["spec"]["containers"] if c["name"] in {"ray-worker", "ray-head"}]
    if not args.container and len(containers) != 1:
        raise RuntimeError("Cannot identify the Ray container; specify --container.")
    container = args.container or containers[0]
    container_spec = next(c for c in pod["spec"]["containers"] if c["name"] == container)
    container_status = next(
        (c for c in pod["status"].get("containerStatuses", []) if c["name"] == container),
        {},
    )
    write_json(
        run / "collection-source.json",
        {
            "collected_at_utc": collected.isoformat(),
            "ray_address": args.address,
            "context": context,
            "namespace": args.namespace,
            "pod": pod_name,
            "pod_uid": pod["metadata"]["uid"],
            "pod_ip": driver_ip,
            "node": pod["spec"].get("nodeName"),
            "container": container,
            "container_image": container_spec["image"],
            "container_image_id": container_status.get("imageID"),
            "scope": "This probe's Ray driver/supervisor logs, retained submitted scripts and matching probe log.",
            "jsonl_source": "Reconstructed from PROMPT/OUTPUT records in ray-job.log; input is the logged user prompt.",
        },
    )
    launcher_args = shlex.split(job["entrypoint"])
    launchers = [Path(arg).name for arg in launcher_args if arg.endswith(".sh")]
    if len(launchers) != 1 or launchers[0] not in {
        "run_hybrid_pool_gsm8k_prefix_cache.sh",
        "run_hybrid_pool_gsm8k_prefix_on.sh",
    }:
        raise RuntimeError(f"Unrecognized probe launcher: {job['entrypoint']}")
    download = download_pod_files(command, pod_name, container, job, {**config, "launcher": launchers[0]}, run)
    probe_log = run / f"prefix_{mode}.log"
    (run / f"{name}.log").write_bytes((probe_log if probe_log.is_file() else run / "ray-job.log").read_bytes())
    write_json(
        run / "download-manifest.json",
        {
            "source_pod": pod_name,
            "archive": "downloaded-artifacts.tar.gz",
            **download,
        },
    )
    files = []
    for path in sorted(run.rglob("*")):
        if path.is_file():
            payload = path.read_bytes()
            files.append(
                {
                    "path": str(path.relative_to(run)),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    write_json(
        run / f"{name}_manifest.json",
        {
            "submission_id": args.submission_id,
            "collected_at_utc": collected.isoformat(),
            "files": files,
            "summary": summary,
            "warnings": download["warnings"],
        },
    )
    for warning in download["warnings"]:
        print(f"Warning: {warning}", file=sys.stderr)
    print(f"Saved {len(records)} prompt/response records to {generations}", flush=True)
    print(f"Artifacts: {run}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError, tarfile.TarError) as error:
        sys.exit(f"Error: {error}")
