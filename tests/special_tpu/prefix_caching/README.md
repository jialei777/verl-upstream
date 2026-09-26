# TPU prefix-caching issue

This manual, standalone vLLM-TPU test can run with prefix caching
**enabled or disabled** to help investigate incorrect generated responses.

The three test scripts are adapted from `vllm-torchtpu/examples/`. Keep them together:
the shell launcher calls `hybrid_pool_e2e.py`, which imports
`hybrid_pool_gsm8k.py` for this test.

## Requirements

- A TPU host with sufficient devices for the selected tensor-parallel size
  (the launcher defaults to eight).
- A Ray cluster reachable at `http://127.0.0.1:23334`, with the `TPU` and
  `rollout-0` resources requested by the launch command. Set up the dashboard
  connection or port forwarding before submitting the job.
- A compatible vLLM and vllm-torchtpu environment on the Ray execution node,
  including the Hugging Face `datasets` package. These scripts do not install
  the runtime.
- Access to download `Qwen/Qwen3-0.6B` and the `openai/gsm8k` dataset, or cached
  copies available to the runtime.

## Container image

Use the following container image for the Ray pods running this test:

```text
us-west2-docker.pkg.dev/tpu-pytorch/raycluster/verl-tpu@sha256:2da83c0d7e3078fa75e39d4b365f1941d14ac7533c21448f59a96b124d8f03a2
```

[View the image in Artifact Registry (Google-internal)](https://pantheon.corp.google.com/artifacts/docker/tpu-pytorch/us-west2/raycluster/verl-tpu/sha256:2da83c0d7e3078fa75e39d4b365f1941d14ac7533c21448f59a96b124d8f03a2?e=13802955&project=tpu-pytorch).

Configure the Ray cluster to use this image before submitting the job.

## Run

Set `CTX` to the Kubernetes context for the target cluster, then forward the
Ray dashboard port in one terminal. For the example cluster:

```bash
export CTX=gke_tpu-pytorch_europe-west4-a_lixali-tpu-cluster-16

kubectl --context "$CTX" -n default port-forward \
  svc/ray-lixali-tpu-16-head-svc 23334:8265
```

Keep the port-forward command running. In a second terminal, submit the job
from the verl checkout. The working directory contains all three test scripts
and is uploaded to Ray:

```bash
cd ~/verl-upstream-top-of-main-branch-original

ray job submit \
  --address http://127.0.0.1:23334 \
  --working-dir tests/special_tpu/prefix_caching \
  --runtime-env-json '{"excludes":["logs/","__pycache__/"]}' \
  --entrypoint-resources '{"TPU":8,"rollout-0":1}' \
  -- bash run_hybrid_pool_gsm8k_prefix_cache.sh --prefix-cache on
```

The `excludes` setting keeps downloaded artifacts out of future job uploads.
To run with prefix caching disabled, submit the same command with
`--prefix-cache off`. Omitting the option keeps prefix caching enabled.
Logs are written on the Ray execution node to
`/tmp/hybrid_pool_gsm8k/prefix_on.log` or
`/tmp/hybrid_pool_gsm8k/prefix_off.log`, according to the selected mode.

The job requests eight TPU resources and one `rollout-0` resource. The eight-TPU
reservation matches the launcher's default `E2E_TP=8`.
The launcher explicitly sets `VLLM_DP_SIZE=1`, keeping all requests in one
data-parallel replica in both prefix-cache modes.

Pass environment overrides to the remote job using `--runtime-env-json`. For
example, replace that argument with the following to change the log directory:

```bash
--runtime-env-json '{"excludes":["logs/","__pycache__/"],"env_vars":{"E2E_GSM8K_RESULT_DIR":"/tmp/my-prefix-test"}}'
```

Set `PYTHON` in the same `env_vars` object to select a different interpreter;
that interpreter path must exist on the Ray execution node.

## Download a finished job

Keep the dashboard port forwarding running. The collector needs local `kubectl`
access to the same cluster and Python 3.12 or later. It uses the Python standard
library, so a local Ray, vLLM, or datasets installation is not required for
downloading. `uv` can select the Python interpreter.

Run this from the checkout to collect the completed example job:

```bash
cd ~/verl-upstream-top-of-main-branch-original

uv run --no-project --python 3.12 \
  tests/special_tpu/prefix_caching/download_ray_job.py \
  raysubmit_DKk2dPpHaiKwSrYb \
  --address http://127.0.0.1:23334 \
  --context gke_tpu-pytorch_europe-west4-a_lixali-tpu-cluster-16 \
  --namespace default
```

Replace the submission ID for another successful, complete prefix-cache test. The collector
finds the execution pod from Ray's job metadata, downloads the artifacts, verifies
their SHA-256 checksums, and prints the local run directory. It creates `logs/`
automatically; `--output-root /another/path` changes the destination. It does not
submit a new job or change the existing job.

The folder follows the model, topology, submission ID, job ID, and UTC export-time
convention used by the TPU training logs, with a test label and cache mode:

```text
logs/
└── prefix_cache_gsm8k_<model>_tp<TP>-dp<DP>-gen_prefix-<on|off>_<submission>_<ray-job-id>_<YYYYMMDD-HHMMSS>/
    ├── ray-job.json
    ├── ray-job.log
    ├── <run-folder>.log
    ├── prefix_on.log                 # or prefix_off.log, when still available and matching this job
    ├── resolved-config.json
    ├── summary.json
    ├── collection-source.json
    ├── downloaded-artifacts.tar.gz
    ├── download-manifest.json
    ├── download-manifest-rollout.json
    ├── <run-folder>_manifest.json
    ├── launch/
    │   ├── entrypoint.sh
    │   ├── launch.json
    │   ├── runtime-env.json
    │   ├── run_hybrid_pool_gsm8k_prefix_cache.sh
    │   ├── hybrid_pool_e2e.py
    │   └── hybrid_pool_gsm8k.py
    ├── ray_logs/rollout/
    │   ├── job-driver-<submission>.log
    │   └── jobs/supervisor-<submission>.log
    └── generations/rollout/
        └── 0.jsonl
```

`no-ray-job-id` is used when Ray reports a null driver job ID, as it does for this
standalone example. TP and DP come from the engine log; no trainer topology is
included. The time suffix is the UTC download time. Each invocation creates a new
export. Generated artifacts are ignored by Git. No plots or Markdown reports are
created in the run folder.

`generations/rollout/0.jsonl` contains one record per response: **32 records for
eight prompts with four responses each**. Each record repeats its prompt in
`input` and stores the complete logged response in `output`. It also includes
`step` (test round, currently zero), `uid` (unique to each response),
`prompt_row`, `sample_index`, `prompt_tokens`, `cached_tokens`, `finish_reason`,
and `submission_id`. Newlines in responses are escaped inside each JSON line.
The collector reconstructs these records from the existing `PROMPT` and `OUTPUT`
log entries, so it also works for jobs completed before the collector was added.
It checks that all prompt/response records are present and the cache-token totals
agree with the final `GSM8K_DONE` line.

`input` is the logged user prompt, including the answer-format instruction; it
does not include the tokenizer's chat-template tokens. The test does not record
reference answers, reward scores, response token IDs, or token log probabilities,
so those fields are not invented in the export. `summary.json` contains response
counts, finish reasons, and the total reported cached tokens.

The archive contains this job's driver/supervisor logs and retained submitted
scripts. These files must still exist on the execution pod for a full collection.
The shared `/tmp/hybrid_pool_gsm8k/prefix_on.log` or `prefix_off.log` can be
overwritten by another run; the collector includes it only when it matches this
job. Missing submitted scripts or a missing/mismatched shared log are reported
in the download manifest. `ray-job.log` remains the source for the JSONL export.

The downloader uses the [Ray Jobs REST API](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/rest.html)
for job metadata and output, and `kubectl exec` to read the retained pod files.
