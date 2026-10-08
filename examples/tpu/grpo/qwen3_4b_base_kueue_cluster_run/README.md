# Qwen3-4B-Base GRPO — 64 TPUs

```bash
cd ~/verl-upstream-using-hardware-plugin-pr-50/examples/tpu/grpo/qwen3_4b_base_kueue_cluster_run
bash submit_qwen3_4b_base_250steps.sh
```

Default: 250 steps. For a 2-step test, append `--steps 2 --skip-validation`.
For a local check without submission, append `--render-only`.
Validation: GSM8K (1,319), MATH-500 (500), and 1,000 frozen OpenMathInstruct-2 `train_1M` questions; the selection stays fixed across runs.

W&B metrics are enabled under `verl_tpu_grpo`. The launcher reuses `wandb login` or `WANDB_API_KEY` through a Kubernetes Secret. Set `WANDB_ENTITY`/`WANDB_PROJECT` to choose the account/project, or `WANDB_MODE=offline` for offline logging.
