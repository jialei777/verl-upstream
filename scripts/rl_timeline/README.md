# RL timeline dashboard

Shows over wall-clock time what the trainer loop and every rollout replica (sampler slice) of a verl run are doing:
when the trainer waits for rollout (`gen`), computes (`old_log_prob`, `ref`, `adv`, `update_actor`), syncs weights
(`update_weights` and its `sync.*` phases) or validates (`testing`), and when each replica is generating, still
generating behind a closed gate while a weight sync waits for it, paused, or idle, with one bar per request.

## Record

Run the job with `VERL_TIMELINE=1` in its runtime env:

```bash
ray job submit ... --runtime-env-json '{"env_vars": {"VERL_TIMELINE": "1", ...}}' -- bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh
```

`marked_timer` (trainer loop) and `vLLMHttpServer` (requests, gate drain, engine pause) then print one
`VERL_TIMELINE{...}` line per finished span (see `verl/utils/timeline.py`); Ray forwards them into the job log.
Weight-sync phases are taken from the `timing_s/tpu-sync/*` step metrics the trainer already logs.

## View

```bash
ray job logs --address http://localhost:8265 <submission-id> > job.log
```

- Open `index.html` and drop `job.log` on it (works offline; older logs without timeline lines give an approximate
  trainer-only view rebuilt from `timing_s/*`).
- Or build one sharable file: `python3 scripts/rl_timeline/make_report.py job.log -o report.html --summary`.

Mouse wheel zooms, drag pans, drag on the time axis zooms to a range, double-click fits, click a step label (or a row
of the step table) to zoom to one step. The summary is computed for the visible window. **Compare…** loads a second
log or report and shows the steady-state phase means of both runs; **Export Perfetto** writes a Chrome trace.
