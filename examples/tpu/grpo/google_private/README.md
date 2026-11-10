# Google-internal helpers for the TPU GRPO examples

These helpers are tracked on this fork only, for Google's own clusters. They are not part of upstream verl.

| File | |
|---|---|
| [`convert_qwen3_5_text_only.py`](convert_qwen3_5_text_only.py) | Converts a multimodal Qwen3.5 Hugging Face checkpoint (`model.language_model.*` + vision encoder) into the text-only causal-LM checkpoint (`Qwen3_5[Moe]ForCausalLM`) that `run_qwen3_5_torchtitan.sh` needs |
| [`render_ray_tpu7x_jobset.py`](render_ray_tpu7x_jobset.py) | Renders a Ray-on-JobSet manifest (CPU head + trainer TPU slice + rollout TPU slice) for GKE TPU v7x clusters without KubeRay. Every pod copies the model and GSM8K from the bucket to `/data` before it starts Ray. |

```bash
hf download Qwen/Qwen3.5-35B-A3B --local-dir /tmp/Qwen3.5-35B-A3B-raw
python3 examples/tpu/grpo/google_private/convert_qwen3_5_text_only.py /tmp/Qwen3.5-35B-A3B-raw /tmp/Qwen3.5-35B-A3B
gcloud storage cp -r /tmp/Qwen3.5-35B-A3B gs://<BUCKET>/assets/hf/

# Add --namespace, --reservation, --queue / --priority (Kueue), --head-node-selector / --head-toleration as needed.
python3 examples/tpu/grpo/google_private/render_ray_tpu7x_jobset.py --name verl-qwen35 --image <IMAGE> \
  --bucket gs://<BUCKET> --model Qwen3.5-35B-A3B --trainer 2x2x1:1 --rollout 2x2x1:1 | kubectl apply -f -
```

Then forward the head pod's dashboard (`kubectl port-forward pod/<verl-qwen35-head-0-0-...> 23333:8265`) and submit `examples/tpu/grpo/run_qwen3_5_torchtitan.sh` as described in `examples/tpu/grpo/README.md`.
