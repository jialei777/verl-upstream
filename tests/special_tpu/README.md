# TPU CI

End-to-end tests of verl on Cloud TPU v6e, built on the TPU examples in [`examples/tpu`](../../examples/tpu). TPU support comes from [`verl-hardware-plugin`](https://github.com/verl-project/verl-hardware-plugin); every test runs the verl and plugin commits under test (HEAD of both by default, see [Code under test](#code-under-test)), an image built from that verl's [`examples/tpu/grpo/Dockerfile`](../../examples/tpu/grpo/Dockerfile), and the commands of [`examples/tpu/grpo/README.md`](../../examples/tpu/grpo/README.md).

## Workflows

Like the NVIDIA, Ascend and AMD CI, there is one workflow per test category, named after the GPU workflow it mirrors with a `_tpu` suffix:

| Workflow | Mirrors | Test (`run_tpu_e2e_ci.sh <test>`) | TPU | Trigger |
|---|---|---|---|---|
| [`tpu_unit_tests.yml`](../../.github/workflows/tpu_unit_tests.yml) | `gpu_unit_tests.yml`, `npu_unit_tests.yml` | `tpu_platform`: [`tpu_smoke_test.py`](tpu_smoke_test.py) — `torch_tpu` ops and autograd, the same MLP on TPU and CPU, and that `import verl` loads the plugin, selects the `tpu` device and registers the TorchTitan TPU engine and the `tpu-sync` / `ray-plasma-store` weight sync backends | 1 x v6e-1 | push / PR |
| [`e2e_sft_llm_tpu.yml`](../../.github/workflows/e2e_sft_llm_tpu.yml) | `e2e_sft_llm.yml`, `e2e_sft_llm_ascend.yml` | `sft`: [`examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh`](../../examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh) with `SMOKE_TEST=1` (8 steps), once with torchrun (`verl.trainer.sft_trainer`, the default) and once with `SFT_TRAINER_MODE=ray` (`verl.trainer.sft_trainer_ray`) | 1 x v6e-4 | push / PR |
| [`e2e_v1_separate_async_tpu.yml`](../../.github/workflows/e2e_v1_separate_async_tpu.yml) | `e2e_v1_separate_async.yml`, `e2e_v1_separate_async_ascend.yml`, `e2e_ppo_trainer_torchtitan_vllm.yml` | `grpo`: [`examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh`](../../examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh) — 5-step Qwen3-0.6B GRPO on 2 chips, TorchTitan trainer on one v6e-1 slice, vLLM rollout on the other, `tpu-sync` weight sync with its parity checks on (`verify_parity=all`) | 2 x v6e-1 | push / PR |
| [`nightly_tpu.yml`](../../.github/workflows/nightly_tpu.yml) | `nightly_ascend.yml` | `grpo_4b`: [`examples/tpu/grpo/run_qwen3_4b_torchtitan.sh`](../../examples/tpu/grpo/run_qwen3_4b_torchtitan.sh) for 20 steps (`GRPO_4B_STEPS`) with validation before and after and `verify_parity=all` | 2 x v6e-4 | schedule / manual only |
| [`checkpoint_engine_tpu.yml`](../../.github/workflows/checkpoint_engine_tpu.yml) | the checkpoint engine tests of `tests/checkpoint_engine` | `tpu_sync`: [`tpu_sync_test.py`](tpu_sync_test.py) — `tpu-sync` weight sync without vLLM: a TorchTitan trainer (Qwen3-0.6B, GRPO actor configuration) on one v6e-4 slice sends its weights to fake single-chip rollout workers on the other slice, which run the plugin's sampler-side code and compare every received tensor bit for bit with its tensor-parallel slice of the checkpoint (1 x TP4, 2 x TP2, TP4 keeping the send buffers; 3 syncs each) | 2 x v6e-4 | push / PR (weight sync, engine and rollout sources) |

Push and PR triggers of the first three cover `main`, `main-ci` and `v0.*` with path filters (verl Python sources and configs, the matching `examples/tpu` directory and `tests/special_tpu`, no Markdown); `checkpoint_engine_tpu.yml` uses the same branches with narrower paths (`verl/checkpoint_engine`, `verl/workers/{engine,rollout}`, `verl/workers/engine_workers.py`, `verl/single_controller`, the Dockerfile and `tests/special_tpu`). The 4B run never starts on a push or PR. Every workflow can also be started by hand (`workflow_dispatch`), optionally with another `verl-hardware-plugin` repository and commit. GitHub only runs `schedule` triggers from the default branch, and only offers `workflow_dispatch` for workflows that are on the default branch or have already run on the branch.

[`verify_tpu_e2e_log.py`](verify_tpu_e2e_log.py) checks each job log: all steps ran; losses, gradient norms and rewards are finite; SFT train and validation losses go down; GRPO gradients flow, at least one rollout earns a reward, validation runs, and the rollout (vLLM) and trainer log-probs of the sampled tokens agree (`rollout_corr/kl` < 0.02, `rollout_corr/ppl_ratio` within 5% of 1, which catches a broken weight sync), and the `tpu-sync` parity checks passed: the step-0 sync matched the checkpoint vLLM loaded bit for bit (`0 mismatched, 0 unresolved`), the weight norms of every later sync were `VERIFIED`, nothing logged `MISMATCH`, and the sync timing was reported (the plugin logs these at INFO, so the GRPO jobs run with `VERL_LOGGING_LEVEL=INFO`); the 4B run must raise GSM8K validation accuracy to at least 0.45. For `tpu_sync` it requires `PASSED: <n>/<n> cases`, a `bit-exact` line for every sync and no `MISMATCH`.

## How a TPU job works

The jobs run on self-hosted runners that live in a GKE cluster with TPU v6e node pools. Every workflow has two stages:

- `build_image` runs [`run_tpu_e2e_ci.sh build_image`](run_tpu_e2e_ci.sh): it resolves the verl and plugin commits under test, builds the image of that verl's [`examples/tpu/grpo/Dockerfile`](../../examples/tpu/grpo/Dockerfile), or finds it in Artifact Registry if that Dockerfile was built before (see [Image](#image)), and passes the image and both commit ids on as job outputs;
- the test job (`needs: build_image`) gets them as `TPU_CI_IMAGE`, `VERL_COMMIT` and `VERL_HARDWARE_PLUGIN_COMMIT` and calls [`run_tpu_e2e_ci.sh <test>`](run_tpu_e2e_ci.sh), which:

1. fetches the verl and plugin commits under test (see [Code under test](#code-under-test)), or uses the local checkouts in `VERL_DIR` / `VERL_HARDWARE_PLUGIN_DIR`;
2. creates a RayCluster for this test from [`gke/raycluster-ci.yaml`](gke/raycluster-ci.yaml) (same pods as [`examples/tpu/grpo/ray-tpu-v6e-2slice.yaml`](../../examples/tpu/grpo/ray-tpu-v6e-2slice.yaml)), queued through Kueue: it stays suspended until its queue has TPU quota, then all its pods start at once;
3. submits the test with `ray job submit --working-dir <checkout> --runtime-env-json <the README's RUNTIME_ENV>`, passing the example's settings as environment variables in the entrypoint, polls it, saves the log and runs the verifier. The torchrun SFT run is the exception: it copies the checkout and the plugin into the TPU worker container and runs the script there with `kubectl exec` (a Ray driver that reserves all chips of a host gets Ray's host-default TPU environment, which does not describe a 2x2 subslice);
4. deletes the RayCluster on exit (pass, failure, timeout or cancellation). The workflow repeats the deletion in an `if: always()` step, and every run deletes CI RayClusters older than 3 hours.

The TPU node pools are spot VMs. When a test fails and one of the TPU nodes it ran on is gone or `NotReady`, the script reports a preemption and re-runs the test once on a new RayCluster (`PREEMPTION_RETRIES`, default 1).

| Shape | TPU workers | Kueue LocalQueue | Quota |
|---|---|---|---|
| 1 x v6e-1 | a whole `ct6e-standard-1t` (1x1) node; the pool scales from zero | `verl-tpu-ci-1chip` | 1 chip, borrows from 2chip |
| 2 x v6e-1 | two such nodes, one Ray TPU slice each (`tpu-group-0` trainer, `tpu-group-1` rollout) | `verl-tpu-ci-2chip` | 2 chips, borrows from 1chip |
| 1 x v6e-4 | one host of the 2x4 (v6e-8) node pools, as a 2x2 subslice (`cloud.google.com/gke-tpu-slice-topology: 2x2`) | `verl-tpu-ci-4chip` | 8 chips, borrows from 8chip |
| 2 x v6e-4 | two such hosts, one Ray TPU slice each (`tpu-group-0` trainer, `tpu-group-1` rollout) | `verl-tpu-ci-8chip` | 8 chips, borrows from 4chip |

A whole host is the smallest multi-chip unit: GKE only admits whole-host TPU requests on 2x4 nodes, and libtpu cannot open an ICI session on 2 chips of a host; a 2-chip test therefore uses two 1-chip nodes, one slice each. Concurrent runs (any branch or PR) wait in their queue in FIFO order instead of competing for hosts. The queues ([`gke/kueue-tpu-ci.yaml`](gke/kueue-tpu-ci.yaml)) are shared with every branch that runs the TPU CI on the cluster; CI only creates them when they are missing.

## Infrastructure

| | |
|---|---|
| GKE cluster | `jialeic-ci-v6e8-2s-spot` (project `tpu-pytorch`, region `us-central2`, zone `us-central2-b`), with the Ray operator, Cloud Storage FUSE, Workload Identity and Kueue (`ray.io/raycluster` integration) |
| TPU node pools | 2 x v6e-8 (`ct6e-standard-4t`, `--tpu-topology 2x4`, 2 nodes each, spot) and `verl-ci-v6e-1t` (`ct6e-standard-1t`, spot, autoscaling 0-3: one node for the 1-chip tier and two for the 2-chip tier) |
| Runners | [`gke/github-runner-deployment.yaml`](gke/github-runner-deployment.yaml): 6 runner pods on the CPU `default-pool`, service account `ray-ksa` (creates the RayClusters in-cluster). The workflows run on the label in the repository variable `TPU_CI_RUNNER` and are skipped where it is not set |
| Image | Built by CI from [`examples/tpu/grpo/Dockerfile`](../../examples/tpu/grpo/Dockerfile) into `TPU_CI_IMAGE_REPO` (see [Image](#image)); the build Job runs on `default-pool` as `ray-ksa`, whose Google service account can write to that repository |
| Code under test | `VERL_REPO` / `VERL_COMMIT` and `VERL_HARDWARE_PLUGIN_REPO` / `VERL_HARDWARE_PLUGIN_COMMIT` in [`gke/tpu-ci.env`](gke/tpu-ci.env) (HEAD of both by default) |
| Models and data | GCS bucket `TPU_CI_BUCKET`, mounted at `/data`; the scripts get `RAY_DATA_HOME=TPU_CI_DATA_HOME` (models in `assets/hf/Qwen3-{0.6B,4B}`, data in `data/gsm8k` and `data/gsm8k_sft`, see step 3 of [`examples/tpu/grpo/README.md`](../../examples/tpu/grpo/README.md)) |

> [!WARNING]
> Do not create long-lived RayClusters on the TPU node pools of the CI cluster: Kueue does not see them, so CI RayClusters get admitted but their TPU pods stay `Pending`.

## Image

The Ray pods run the image of the verl under test's [`examples/tpu/grpo/Dockerfile`](../../examples/tpu/grpo/Dockerfile), so a Dockerfile change is tested together with the code that needs it. `run_tpu_e2e_ci.sh` tags the image `TPU_CI_IMAGE_REPO:df-<first 16 hex digits of the Dockerfile's sha256>` and asks the registry for that tag:

- if it is there (the Dockerfile has not changed since an earlier run), the test uses it right away;
- otherwise the `build_image` stage builds it with the Kubernetes Job [`gke/image-build-job.yaml`](gke/image-build-job.yaml): daemonless BuildKit (`TPU_CI_BUILDKIT_IMAGE`, the builder of `docker buildx build`) on `default-pool`, with the Dockerfile as the whole build context (every source comes from a public URL) and the build state on a per-pod persistent disk. The build takes about 25 minutes (`IMAGE_BUILD_TIMEOUT_MINS` caps it); the jobs that start at the same time for the same Dockerfile wait for the one build. A failed build is retried by the next run.

The stage hands the image on by digest (`<repo>:df-<hash>@sha256:...`), so every test runs exactly that image even if a node caches another image under the same tag. A test run by hand does the same before its test. Set `TPU_CI_IMAGE` (environment or [`gke/tpu-ci.env`](gke/tpu-ci.env)) to skip the build and run a prebuilt image.

## Code under test

By default CI tests HEAD of both repositories: [verl-project/verl](https://github.com/verl-project/verl) `main` and [verl-project/verl-hardware-plugin](https://github.com/verl-project/verl-hardware-plugin) `main`. This branch only provides the CI itself (`tests/special_tpu` and the workflows), which is copied over the fetched verl tree. To test other work, for example a verl change and the plugin change it needs, point either or both at a fork and a commit:

| Setting | Default | Where to set it |
|---|---|---|
| `VERL_REPO`, `VERL_COMMIT` | `https://github.com/verl-project/verl.git`, `main` | [`gke/tpu-ci.env`](gke/tpu-ci.env) (e.g. in a PR to this branch), the environment, or the `verl_repo` / `verl_commit` inputs of a manual run |
| `VERL_HARDWARE_PLUGIN_REPO`, `VERL_HARDWARE_PLUGIN_COMMIT` | `https://github.com/verl-project/verl-hardware-plugin.git`, `main` | the same, inputs `plugin_repo` / `plugin_commit` |

```bash
VERL_REPO="${VERL_REPO:-https://github.com/<user>/verl.git}"
VERL_COMMIT="${VERL_COMMIT:-<full commit id, branch or tag>}"
VERL_HARDWARE_PLUGIN_REPO="${VERL_HARDWARE_PLUGIN_REPO:-https://github.com/<user>/verl-hardware-plugin.git}"
VERL_HARDWARE_PLUGIN_COMMIT="${VERL_HARDWARE_PLUGIN_COMMIT:-<full commit id, branch or tag>}"
```

A commit may be a full 40-character id (GitHub does not serve abbreviated ids), a branch or a tag. The `build_image` stage resolves both to commit ids, builds the image from that verl tree's `examples/tpu/grpo/Dockerfile`, and hands the image and both commit ids to the test job, so a branch that moves during the run does not change what is tested. The job logs start with both commits. The script prints a note when the plugin commit differs from the one `examples/tpu/grpo/README.md` checks out.

## Run a test by hand

From a machine with `gcloud` access to the cluster and the Ray CLI (`pip install "ray[default]==2.56.0"`), at the root of the checkout:

```bash
bash tests/special_tpu/run_tpu_e2e_ci.sh tpu_platform   # or sft / grpo / grpo_4b / tpu_sync
# other commits, e.g. from forks
VERL_REPO=https://github.com/<user>/verl.git VERL_COMMIT=<commit, branch or tag> \
VERL_HARDWARE_PLUGIN_REPO=https://github.com/<user>/verl-hardware-plugin.git VERL_HARDWARE_PLUGIN_COMMIT=<commit, branch or tag> \
    bash tests/special_tpu/run_tpu_e2e_ci.sh grpo
# local checkouts (e.g. uncommitted changes)
VERL_DIR=../verl VERL_HARDWARE_PLUGIN_DIR=../verl-hardware-plugin bash tests/special_tpu/run_tpu_e2e_ci.sh grpo
```

The script fetches its own kubeconfig, so it does not change your current `kubectl` context. Logs are written to `${TMPDIR:-/tmp}/verl-tpu-ci-<run id>/`.

## Maintenance

- **New image**: change `examples/tpu/grpo/Dockerfile` in verl; the next CI run that tests that verl builds and pushes it (see [Image](#image)).
- **Testing a change**: set `VERL_REPO` / `VERL_COMMIT` and/or `VERL_HARDWARE_PLUGIN_REPO` / `VERL_HARDWARE_PLUGIN_COMMIT`, see [Code under test](#code-under-test).
- **New test**: add a `test_<name>` function and its RayCluster shape in `run_test` in `run_tpu_e2e_ci.sh`, a verifier in `verify_tpu_e2e_log.py`, and a workflow (or a step in an existing one).
- **Running the TPU CI from a fork** (until it runs upstream): keep these files on a `main-ci` branch. Pushes to it test HEAD of verl and the plugin; to test a change before sending it upstream, open a PR against `main-ci` that sets the repositories and commits in `gke/tpu-ci.env`, or start the workflows by hand with them.
