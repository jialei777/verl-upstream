---
name: monitor-tpu-job
description: Trigger this skill when the user wants to check the status of an ongoing or completed TPU job, set up port forwarding if needed, inspect logs for errors, and report current training step progress.
user_invocable: true
---

# TPU RL Job Monitoring Agent Skill

This skill provides step-by-step instructions for an agent to check TPU cluster job status, ensure port forwarding is active without disrupting running pods, inspect log files for errors, and report current training step progress.

> [!CAUTION]
> **DO NOT RESET CLUSTER:** When monitoring a job, **NEVER** run `kubectl delete pod` or kill active processes unless explicitly requested by the user.

---

## 🤖 Agent Execution Workflow

### Step 1: Non-Destructive Port Forwarding Check
Before checking Ray job APIs, verify if port forwarding on port `23333` is already active:

```bash
curl -s http://localhost:23333/api/version || true
```

* **If active** (returns JSON with `"ray_version"`): Proceed directly to Step 2.
* **If inactive / connection refused**: Start background port forwarding **without resetting pods**:
  ```bash
  kubectl port-forward svc/ray-tpu-v6e-cluster-head-svc 23333:8265
  ```

---

### Step 2: Locate Active / Latest Log File
Locate the target log file in `logs/`:
* Check if `$LOG_FILE` is set in the environment: `echo "$LOG_FILE"`
* Otherwise, pick the most recently modified log file in `logs/`:
  ```bash
  ls -t logs/*.log 2>/dev/null | head -n 1
  ```

---

### Step 3: Check Cluster & Ray Job Status
Check the status of submitted Ray jobs via the HTTP REST API:

```bash
curl -s http://localhost:23333/api/jobs/ | python3 -m json.tool || true
```

Also check cluster pod health:
```bash
kubectl get pods -l ray.io/cluster=ray-tpu-v6e-cluster
```

---

### Step 4: Scan Log File for Errors & Step Progress

1. **Scan for Errors / Exceptions**:
   Search the log for fatal runtime errors, CUDA/TPU OOMs, or XLA compilation failures:
   ```bash
   grep -Ei "error|exception|traceback|fatal|failed" "$LOG_FILE" | tail -n 25
   ```

2. **Extract Training Step & Reward Progress**:
   Search for step markers, loss, accuracy, and reward metrics:
   ```bash
   grep -Ei "step:|val-core|val-aux|reward" "$LOG_FILE" | tail -n 25
   ```

3. **Check Log Activity**:
   Verify that the log file is actively being written to (line count increasing):
   ```bash
   wc -l "$LOG_FILE"
   ```

---

### Step 5: Report Execution Summary to User
Provide a concise summary containing:
1. **Job Status**: Running, Succeeded, or Errored.
2. **Current Progress**: The latest step number reached and key reward/accuracy metrics.
3. **Cluster Health**: Pod readiness and Ray job status.
4. **Error Findings**: Any warnings or tracebacks detected in the logs.

---

### Step 6: Continuous 1-Minute Periodic Monitoring Protocol
If the job is still actively running:
1. **Schedule Next Check**: Use the `schedule` tool to set a 1-minute (60 seconds) timer for continuous monitoring:
   ```json
   {
     "DurationSeconds": "60",
     "Prompt": "Check TPU job log line count growth, scan for errors, and report latest training step progress.",
     "TimerCondition": "never"
   }
   ```
2. **Periodic Check Checklist (Every 2 Minutes)**:
   * Run `wc -l "$LOG_FILE"` to verify log line count is continuously growing.
   * Run `grep -Ei "step:|reward|val-core" "$LOG_FILE" | tail -n 10` to track training step progression.
   * Run `grep -Ei "error|exception|traceback" "$LOG_FILE" | tail -n 10` to catch any runtime failures immediately.
3. **Stop Condition**: Once the job reaches its target total steps or exits with a final error, stop setting new timers and present the final summary report.

---

## 📚 Debugging References

For deep debugging, refer to the following source code repositories:
1. [torchtpu-vllm](https://github.com/google-pytorch/torchtpu-vllm): Source code of inference used for generator.
2. [torch_tpu](https://github.com/google-pytorch/torch_tpu): Source code of torchtpu.
3. [xla](https://github.com/openxla/xla): Source code of XLA.

