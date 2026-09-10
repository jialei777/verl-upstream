---
name: gke-validate
description: Trigger this skill when the user wants to validate TPU RL training changes on the GKE Ray cluster, ensure port forwarding is active, and submit/monitor a GRPO RL training job. Supports fast re-use and optional cluster resets.
---

# TPU RL Job Submission & Monitoring Agent Skill

This skill provides direct, step-by-step instructions for an agent to establish local port connections, submit PyTorch/XLA RL jobs (such as GRPO) to the active GKE Ray cluster, and monitor progress efficiently.

* **Cluster Name**: `alekseyv-tpu-v6e8-spot-xpk`
* **GCP Project**: `tpu-pytorch`
* **Region**: `us-central2`
* **Kubeconfig Path**: `/tmp/alekseyv-kubeconfig`
* **Ray Head Service**: `svc/ray-tpu-v6e-cluster-head-svc`

---

## 🤖 Agent Execution Workflow

### Step 1: Pre-Submission Port Cleanup & Check
Before starting any Ray submissions or local port connections, ensure port `23333` is clear or check if port-forwarding is already active:

```bash
# Check if port forwarding is already active and healthy
if ! curl -s http://localhost:23333/api/version >/dev/null 2>&1; then
    fuser -k 23333/tcp || true
    kubectl --kubeconfig=/tmp/alekseyv-kubeconfig port-forward svc/ray-tpu-v6e-cluster-head-svc 23333:8265 >/dev/null 2>&1 &
    sleep 3
fi
```

---

### Step 2: Cluster Health Check (Fast Path vs. Cold Reset)

> [!TIP]
> **Avoid Unnecessary Cold Starts:** 
> Deleting pods triggers a **~16.8 minute cold-start XLA re-compilation** (Torchtitan ~7m + vLLM ~9.5m). 
> **Always prefer reusing existing healthy pods** unless the cluster is in an `Error`, `CrashLoopBackOff`, or TPU hardware deadlock state.

1. **Inspect Cluster Pod Health**:
   ```bash
   kubectl --kubeconfig=/tmp/alekseyv-kubeconfig get pods -l ray.io/cluster=ray-tpu-v6e-cluster
   ```
   * **If all pods are `Running` and `Ready` (e.g. `2/2`)**: Proceed directly to **Step 3** (Fast Path).
   * **If pods are in an `Error`/deadlocked state or a clean reset is explicitly requested**: Execute a cluster restart:
     ```bash
     kubectl --kubeconfig=/tmp/alekseyv-kubeconfig delete pod -l ray.io/cluster=ray-tpu-v6e-cluster
     ```
     Wait approximately 30–60 seconds until all pods return to `Running` and `Ready` (2/2) status.

---

### Step 3: Verify Local Ray Client Environment
Ensure the local Ray CLI with submission dependencies is in `$PATH`:

```bash
export PATH="/usr/local/google/home/wenjung/.local/bin:$PATH"
export KUBECONFIG="/tmp/alekseyv-kubeconfig"
export RAY_ADDRESS="http://localhost:23333"
```

---

### Step 4: Programmatic Ray Job Submission
Submit the Ray job to `http://localhost:23333` using `run.sh` and redirect output to a structured log file:

```bash
# Create the logs directory
mkdir -p logs/

# Export the active log file path
export LOG_FILE="logs/grpo_v1_run_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"
echo "$LOG_FILE" > .current_log_file

# Submit GRPO RL training job in background
bash run.sh > "$LOG_FILE" 2>&1 &
```

---

### Step 5: Anti-Hang Monitoring & Verification Protocol

1. **Verify Log Updates**:
   Track log line counts to confirm steady progress:
   ```bash
   wc -l "$LOG_FILE"
   ```

2. **Stream / Inspect Live Ray Job Logs**:
   ```bash
   # Extract the Ray Submission ID from $LOG_FILE or API
   SUBMISSION_ID=$(curl -s http://localhost:23333/api/jobs/ | python3 -c "import sys, json; print(json.load(sys.stdin)[0]['submission_id'])" 2>/dev/null)
   
   # Fetch recent log output
   ray job logs --address http://localhost:23333 "$SUBMISSION_ID" | tail -n 40
   ```

3. **Check Current Step Progress & Metrics**:
   Filter the active log for step numbers, validation progress, and metrics:
   ```bash
   ray job logs --address http://localhost:23333 "$SUBMISSION_ID" | grep -Ei "step:|loss|reward|val-core|TPU weight sync" | tail -n 25
   ```

> [!IMPORTANT]
> **Periodic Monitoring Checklist:**
> * **Every 1 Minute:** Actively check on the submitted job.
> * **Verify Output Growth:** Confirm that `wc -l` or `ray job logs` line count is steadily growing.
> * **Track Steps:** Check for `step: 0`, `step: 1`, `TPU weight sync completed in Xs`.
> * **Timeout Rule:** If the job hangs with zero log updates for more than 5 minutes during training, inspect worker pod logs with `kubectl logs`.