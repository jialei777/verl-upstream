---
name: gke-validate
description: Trigger this skill when the user wants to clean up stale locks, configure local port forwarding on port 23333, and submit a new GRPO RL training job to the cluster. For checking existing job progress without resetting the cluster, use monitor-tpu-job instead.
---

# TPU RL Job Submission & Monitoring Agent Skill

This skill provides direct, step-by-step instructions for an agent to cleanly establish local port connections, submit PyTorch/XLA RL jobs (such as GRPO) to the active GKE Ray cluster, and monitor progress. 

The cluster to use is "alekseyv-tpu-v6e8-spot-xpk", the project is tpu-pytorch, and the region is us-central2, the kubeconfig is located at "/tmp/jialei-kubeconfig"

## 🤖 Agent Execution Workflow

### Step 1: Pre-Submission Port Cleanup
Before starting any Ray submissions or local port connections, cleanly terminate any zombie processes on port `23333` to prevent port-already-bound or connection-refused errors:
```bash
fuser -k 23333/tcp || true
```
---

### Step 2: Reset / Restart GKE TPU Cluster Pods
To ensure that the Ray cluster is in a completely healthy and clean state, and to clear any lingering TPU memory locks or zombie processes, restart all pods in the Ray cluster:

```bash
kubectl delete pod -l ray.io/cluster=ray-tpu-v6e-cluster
```

> [!IMPORTANT]
> **Wait for Recovery:** After deleting the pods, wait approximately 30–60 seconds for GKE to re-schedule and successfully spin up the new TPU head and worker pods. You can check the status via:
> ```bash
> kubectl get pods -l ray.io/cluster=ray-tpu-v6e-cluster
> ```
> Wait until all pods (head and workers) show `Running` and `Ready` (e.g., `2/2`) status before proceeding to Step 3.

---

### Step 3: Establish Background Port Forwarding
Set up background port-forwarding from the local dev VM to the GKE Ray TPU head service on port `23333` as an asynchronous background task:
```bash
kubectl port-forward svc/ray-tpu-v6e-cluster-head-svc 23333:8265
```

---

### Step 4: Programmatic Ray Job Submission
Submit the Ray job to `http://localhost:23333`. Specify the following `excludes`, environment variables, and output redirects. 

Save `LOG_FILE` to your environment by exporting it so that all subsequent tracking and plotting steps reference the exact active file:

```bash
# Create the logs directory
mkdir -p logs/

# Export the active log file path
export LOG_FILE="logs/grpo_v1_run_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"

source /mnt/pd/daily/verl/bin/activate

# Submit the Ray Job (WandB API key is passed securely from your environment variable)
export RAY_ADDRESS="http://localhost:23333"

# Submit GRPO RL training job
bash run.sh > "$LOG_FILE" 2>&1 &
```

---

### Step 5: Anti-Hang Monitoring & Verification
Monitor the exported `$LOG_FILE` continuously to ensure progress and prevent silent hangs (such as XLA device locking or compilation jams):

1. **Verify Log Updates**:
   Track log line counts to ensure they are increasing:
   ```bash
   wc -l "$LOG_FILE"
   ```
2. **Monitor Step Progression**:
   Filter the active log for step numbers, validation progress, and metrics:
   ```bash
   grep -Ei "step|loss|reward" "$LOG_FILE" | tail -n 20
   ```
3. **Trace console updates**:
   ```bash
   tail -f "$LOG_FILE"
   ```

> [!IMPORTANT]
> **Periodic Monitoring Protocol:** 
> * **Every 1 Minutes:** Actively check on the submitted job every 1 minutes.
> * **Verify Log Updates:** Run `wc -l "$LOG_FILE"` to confirm that the log line count is continuously and steadily increasing.
> * **Check Current Step Progress:** Run `grep "Progress" "$LOG_FILE"` or search the logs for the keyword `"Progress"` to immediately identify which training step or validation iteration the job is currently executing.
> * **Timeout Session:** If the job has been running for over 10 minutes, stop it.
> * **Wait For Review:** If any failure happens, find the rootcause and propose the fix in production code, you must wait for the reviewer's approval before executing the next step. But if just adding debugging logs or excute test scrips, you can execute it directly without any supervision.

> [!TIP]
> **Detecting Hangs:** If the line count from `wc -l` remains static over a 30–60 second window during training, or if no new `step:` outputs appear after several minutes, check the worker pod logs using `kubectl logs` to diagnose possible TPU synchronization or communication issues.

Continue to monitor the step progression (ensuring validation rewards show monotonically increasing values, e.g., from step 5 $\to$ 10 $\to$ 15).
