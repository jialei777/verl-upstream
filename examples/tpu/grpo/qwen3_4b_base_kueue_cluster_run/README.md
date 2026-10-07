# Qwen3-4B-Base GRPO — 64 TPUs

```bash
cd ~/verl-upstream-using-hardware-plugin-pr-50/examples/tpu/grpo/qwen3_4b_base_kueue_cluster_run
bash submit_qwen3_4b_base_250steps.sh
```

Default: 250 steps. For a 2-step test, append `--steps 2 --skip-validation`.
For a local check without submission, append `--render-only`.
