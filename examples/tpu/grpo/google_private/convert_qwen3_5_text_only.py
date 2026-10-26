# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Converts a multimodal Qwen3.5 HF checkpoint to a text-only causal-LM checkpoint.

Keeps ``model.language_model.*`` (renamed to ``model.*``) and ``lm_head.weight``; drops the vision
tower (``model.visual.*``) and the MTP head (``mtp.*``). The config becomes the text config with
``architectures=[Qwen3_5[Moe]ForCausalLM]`` so verl, torchtitan's ``Qwen35StateDictAdapter`` (text-only
naming) and vLLM's ``Qwen3_5[Moe]ForCausalLM`` all see a plain language model.

Usage: python convert_qwen3_5_text_only.py <src_dir> <dst_dir>
"""

import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

_DROP_FILES = {"preprocessor_config.json", "video_preprocessor_config.json", "model.safetensors.index.json"}


def main(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    text = dict(cfg["text_config"])
    is_moe = cfg["model_type"] == "qwen3_5_moe"
    text["architectures"] = ["Qwen3_5MoeForCausalLM" if is_moe else "Qwen3_5ForCausalLM"]
    text["model_type"] = "qwen3_5_moe_text" if is_moe else "qwen3_5_text"
    # TorchTitan's TPU Qwen3.5 configs disable weight tying and always load an explicit lm_head.weight;
    # materialize lm_head.weight from model.embed_tokens.weight when the source checkpoint ties them.
    text["tie_word_embeddings"] = False
    text.setdefault("torch_dtype", "bfloat16")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if key not in text and key in cfg:
            text[key] = cfg[key]
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(text, f, indent=2)

    for name in os.listdir(src):
        if name.endswith(".safetensors") or name in _DROP_FILES or name == "config.json":
            continue
        path = os.path.join(src, name)
        if os.path.isfile(path):
            shutil.copy(path, os.path.join(dst, name))

    weight_map = {}
    total_size = 0
    embed_tensor = None
    embed_shard = None
    for name in sorted(n for n in os.listdir(src) if n.endswith(".safetensors")):
        out = {}
        with safe_open(os.path.join(src, name), framework="pt") as f:
            for key in f.keys():
                if key.startswith("model.language_model."):
                    new_key = "model." + key[len("model.language_model.") :]
                elif key == "lm_head.weight":
                    new_key = key
                else:
                    continue
                tensor = f.get_tensor(key)
                out[new_key] = tensor
                if new_key == "model.embed_tokens.weight":
                    embed_tensor = tensor
                    embed_shard = name
        if not out:
            continue
        save_file(out, os.path.join(dst, name), metadata={"format": "pt"})
        for k, v in out.items():
            weight_map[k] = name
            total_size += v.numel() * v.element_size()
        print(f"{name}: {len(out)} tensors", flush=True)
    if "lm_head.weight" not in weight_map and embed_tensor is not None and embed_shard is not None:
        shard_path = os.path.join(dst, embed_shard)
        out = {}
        with safe_open(shard_path, framework="pt") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
        out["lm_head.weight"] = embed_tensor.clone()
        save_file(out, shard_path, metadata={"format": "pt"})
        weight_map["lm_head.weight"] = embed_shard
        total_size += embed_tensor.numel() * embed_tensor.element_size()
        print(f"{embed_shard}: materialized lm_head.weight from model.embed_tokens.weight", flush=True)
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)
    print(f"wrote {len(weight_map)} tensors ({total_size / 2**30:.2f} GiB) to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
