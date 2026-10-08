#!/usr/bin/env bash
set -euo pipefail

# Submit Qwen3-4B-Base GRPO to the original v6e Kueue cluster.
# 250 steps; 128 prompts x 16 rollouts; 32 trainer + 32 generator chips.
# Seed 1 for data, trainer, reference, and generator; rollout TP1; prefix off.
# PPO clipping +/-0.2; dual clipping 3.0; KL coefficient 0.001 (low_var_kl).
# Validate GSM8K (1319), MATH-500 (500), frozen OpenMathInstruct-2 (1000) at step0, every20,
# and the final step250. Checkpoint saving and automatic resume stay disabled.
# Snapshot this VERL checkout and the sibling patched hardware plugin.
#
# Usage from this qwen3_4b_base_kueue_cluster_run directory:
#   bash submit_qwen3_4b_base_250steps.sh
# Local configuration/source checks, without cloud writes or submitting:
#   bash submit_qwen3_4b_base_250steps.sh --render-only
# Cloud input preparation without submitting:
#   bash submit_qwen3_4b_base_250steps.sh --prepare-only
#
# Requires the existing v6e kubeconfig, kubectl/GKE authentication, a GCE
# submission VM with storage permissions, Python3/PyYAML/pyarrow/transformers.
# The model is reused directly from the verified shared lixali GCS cache.
# Existing cached weights are not downloaded or uploaded by the launch VM.

recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
recipe_python="${RECIPE_PYTHON:-python3}"
if [[ -z "${RECIPE_PYTHON:-}" && -x /home/lixali_google_com/venvs/verl-ray-submit/bin/python ]]; then
    recipe_python=/home/lixali_google_com/venvs/verl-ray-submit/bin/python
fi
exec "${recipe_python}" "${recipe_dir}/submit_qwen3_4b_base_250steps.py" "$@"
