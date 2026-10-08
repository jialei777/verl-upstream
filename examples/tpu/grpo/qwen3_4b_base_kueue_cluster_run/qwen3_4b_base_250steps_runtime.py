# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Readable runtime sources for the Qwen3-4B Base TPU launcher.

The launcher materializes these templates into a fresh external run workspace.
Importing this module starts no services and imports no training dependencies.
Helper sources retain their normal layout under that workspace for inspection.
"""

from pathlib import Path


def tpu_plugin_layout(root: str | Path | None = None) -> dict[str, str]:
    """Resolve source paths and imports for either hardware-plugin TPU layout."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    for directory, platform in (
        ("verl_hardware_plugin/accelerators/tpu", "platform_tpu.py"),
        ("verl_hardware_plugin", "platforms/platform_tpu.py"),
    ):
        paths = {
            name: str(Path(directory) / relative)
            for name, relative in (
                ("platform", platform),
                ("rollout", "rollout/tpu_vllm.py"),
                ("patches", "rollout/tpu_vllm_patches.py"),
                ("precision", "rollout/tpu_vllm_precision.py"),
                ("engine", "engines/torchtitan_tpu.py"),
                ("checkpoint", "engines/tpu_checkpoint_engine.py"),
            )
        }
        if all((root / paths[name]).is_file() for name in ("platform", "rollout", "patches")):
            return paths | {name + "_module": path[:-3].replace("/", ".") for name, path in paths.items()}
    raise ValueError("The hardware-plugin checkout has no supported TPU source layout.")


# Each literal contains an ordinary source file. Adjacent literals preserve any
# triple-quote delimiters inside that source without encoding or archiving it.
TEMPLATES: dict[str, str] = {
    "recipe_tpu_plugin_layout.py": (
        r'''# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Resolve legacy and accelerator-specific TPU plugin source paths."""

from pathlib import Path


def tpu_plugin_layout(root: str | Path | None = None) -> dict[str, str]:
    """Resolve source paths and imports for either hardware-plugin TPU layout."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    for directory, platform in (
        ("verl_hardware_plugin/accelerators/tpu", "platform_tpu.py"),
        ("verl_hardware_plugin", "platforms/platform_tpu.py"),
    ):
        paths = {
            name: str(Path(directory) / relative)
            for name, relative in (
                ("platform", platform),
                ("rollout", "rollout/tpu_vllm.py"),
                ("patches", "rollout/tpu_vllm_patches.py"),
                ("precision", "rollout/tpu_vllm_precision.py"),
                ("engine", "engines/torchtitan_tpu.py"),
                ("checkpoint", "engines/tpu_checkpoint_engine.py"),
            )
        }
        if all((root / paths[name]).is_file() for name in ("platform", "rollout", "patches")):
            return paths | {name + "_module": path[:-3].replace("/", ".") for name, path in paths.items()}
    raise ValueError("The hardware-plugin checkout has no supported TPU source layout.")
'''
    ),
    "assets/qwen3-4b-base-validation/validation_manifest.json": (
        r"""{
  "format_version": 1,
  "profile_id": "gsm8k-math500-openmathinstruct2-1000-v1",
  "frozen": true,
  "source_inputs": [
    {
      "file": "source_inputs/train.parquet",
      "source": "gs://lixali-tpu-storage/grpo/tpu-v6e/qwen3-4b-base-grpo-v6e-250steps-32trainer32sampler-ppo-clip02-dual3-kl1e-3-no-prefix-seed1-20261007-115648-f776b0/backup-auto/inputs/jialei/data/gsm8k/train.parquet",
      "generation": "1791374247949787",
      "bytes": 6555601,
      "sha256": "89cd3cb8d28e5274e7f0bf71ff541ea5654ac9e30589ac4b5d19c3f783a3858c"
    },
    {
      "file": "source_inputs/original-validation.parquet",
      "source": "gs://lixali-tpu-storage/grpo/tpu-v6e/qwen3-4b-base-grpo-v6e-250steps-32trainer32sampler-ppo-clip02-dual3-kl1e-3-no-prefix-seed1-20261007-115648-f776b0/backup-auto/inputs/jialei/data/validation/gsm8k_math500_aime2024.parquet",
      "generation": "1791374250149415",
      "bytes": 753286,
      "sha256": "5899a3605cbd6f2326bd692e3c9de6a99c2ea6bb50715fd03e9056aa12800c52"
    }
  ],
  "upstream_metadata": {
    "file": "upstream_metadata.json",
    "bytes": 16007,
    "sha256": "5c3b8d6e5db1ef114a8c56f019307f0254f434ba483523b12a5b47bd9b2d9ddd",
    "declared_shard_sha256": [
      {
        "file": "data/train_1M-00000-of-00003.parquet",
        "bytes": 212931526,
        "sha256": "93700f39cdc87994f2c9a6ad62e1d62e09467793f57d376cfab722757ff26e1c"
      },
      {
        "file": "data/train_1M-00001-of-00003.parquet",
        "bytes": 213248631,
        "sha256": "c208b3cda82903d9923cb0b1286b30ce9de0d2e56641cac32435b171be36b912"
      },
      {
        "file": "data/train_1M-00002-of-00003.parquet",
        "bytes": 212872872,
        "sha256": "1a42cf7139b60ece5e14623626e1a0b9cdaadd5460704183395780ce26849e51"
      }
    ]
  },
  "benchmarks": {
    "gsm8k": {
      "rows": 1319,
      "preserved_original_fields": true
    },
    "math500": {
      "rows": 500,
      "preserved_original_fields": true
    },
    "openmathinstruct2": {
      "rows": 1000,
      "selection": {
        "count": 1000,
        "seed": 0,
        "dataset": "nvidia/OpenMathInstruct-2",
        "revision": "469216e3f46f4dacf476b382e192485ea51a143e",
        "config": "default",
        "split": "train_1M",
        "manifest_file": "openmathinstruct2_1000_selection.json",
        "manifest_sha256": "f9ace2ecc95250f662c64e7b357c4acb4fbb59b34db44891bfb8c6fd93bb4e7c",
        "manifest_uri": "gs://ubench-logs/lixali/verl-ray-kueue/assets/qwen3-4b-gsm8k-math500-openmath1000-v1/provenance/openmathinstruct2_1000_selection.json",
        "ordered_problem_hashes_sha256": "d9fd61249cc43094d9572850325e3b89a1d1a578a31c48c2d8d66a7b3287a2c4",
        "considered_rows": 1010,
        "exclusions": {
          "exact_training_or_original_validation_match": 8,
          "prompt_exceeds_1024_qwen_tokens": 1,
          "duplicate_selected_question": 1
        },
        "independent_of_training_seed": true,
        "no_exact_training_or_original_validation_overlap": true,
        "manifest_generation": "1791442030477217"
      },
      "labels": "expected_answer (original or majority-vote); custom validation from a training split"
    }
  },
  "combined": {
    "file": "gsm8k_math500_openmathinstruct2_1000.parquet",
    "bytes": 729951,
    "sha256": "27b4b6f17171771cf41a3045485d4afb5c2c5fd747abdb84b9cafb655c1e45c9",
    "rows": 2819,
    "counts_by_data_source": {
      "openai/gsm8k": 1319,
      "HuggingFaceH4/MATH-500": 500,
      "nvidia/OpenMathInstruct-2": 1000
    }
  },
  "openmathinstruct2": {
    "file": "openmathinstruct2_1000.parquet",
    "bytes": 199654,
    "sha256": "2fddb6ad0732bcb4cd5f3646df1780197c0038cc72b5b5a9fd39954ef776f3d8",
    "rows": 1000
  },
  "token_length_audit": {
    "tokenizer": {
      "repo": "Qwen/Qwen3-4B-Base",
      "revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
      "files": [
        {
          "file": "config.json",
          "bytes": 727,
          "sha256": "304b2545a258d35620f1d4bf46940c0471d9baa00715ff8e77f84c2fca5057c1"
        },
        {
          "file": "tokenizer_config.json",
          "bytes": 9678,
          "sha256": "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5"
        },
        {
          "file": "tokenizer.json",
          "bytes": 7031645,
          "sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
        },
        {
          "file": "vocab.json",
          "bytes": 2776833,
          "sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
        },
        {
          "file": "merges.txt",
          "bytes": 1671853,
          "sha256": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"
        }
      ],
      "math_scorer_sha256": "085eb67055c6976fea529d5ed9c8f0c63a989e85295d532db0d41fde8db50974"
    },
    "max_prompt_length": 1024,
    "add_generation_prompt": true,
    "enable_thinking": true,
    "all_rows_fit": true,
    "rows": 2819,
    "max_prompt_tokens": 824,
    "by_data_source": {
      "openai/gsm8k": {
        "rows": 1319,
        "min_prompt_tokens": 46,
        "max_prompt_tokens": 211,
        "mean_prompt_tokens": 84.30780894617135
      },
      "HuggingFaceH4/MATH-500": {
        "rows": 500,
        "min_prompt_tokens": 31,
        "max_prompt_tokens": 814,
        "mean_prompt_tokens": 93.94
      },
      "nvidia/OpenMathInstruct-2": {
        "rows": 1000,
        "min_prompt_tokens": 34,
        "max_prompt_tokens": 824,
        "mean_prompt_tokens": 89.826
      }
    }
  },
  "grader_audit": {
    "scorer": "verl.utils.reward_score.math_reward.compute_score",
    "scorer_sha256": "085eb67055c6976fea529d5ed9c8f0c63a989e85295d532db0d41fde8db50974",
    "correct_boxed_passed": 1000,
    "wrong_boxed_passed": 1000,
    "unboxed_first_answer_score": 0.0,
    "all_selected_labels_roundtrip_gradeable": true
  },
  "preserved_rows_canonical_sha256": "2feadafad4a8b3f5ead4ad81c4d7f5f05ebba38328550721bd9208762f834e8b",
  "builder_sha256": "563c91afc45f883f8edc2efc5de500dcc0431f6f4edeeec80e2c2e78c473b347",
  "artifacts": [
    {
      "file": "gsm8k_math500_openmathinstruct2_1000.parquet",
      "bytes": 729951,
      "sha256": "27b4b6f17171771cf41a3045485d4afb5c2c5fd747abdb84b9cafb655c1e45c9"
    },
    {
      "file": "openmathinstruct2_1000.parquet",
      "bytes": 199654,
      "sha256": "2fddb6ad0732bcb4cd5f3646df1780197c0038cc72b5b5a9fd39954ef776f3d8"
    },
    {
      "file": "openmathinstruct2_1000_selection.json",
      "bytes": 666761,
      "sha256": "f9ace2ecc95250f662c64e7b357c4acb4fbb59b34db44891bfb8c6fd93bb4e7c"
    },
    {
      "file": "upstream_metadata.json",
      "bytes": 16007,
      "sha256": "5c3b8d6e5db1ef114a8c56f019307f0254f434ba483523b12a5b47bd9b2d9ddd"
    },
    {
      "file": "upstream_range_audit.json",
      "bytes": 624504,
      "sha256": "c793248f218ebf4554394716fc0d8ae3e2b8250aa734a74d3f624ad4f2e29e95"
    },
    {
      "file": "token_length_audit.json",
      "bytes": 1688,
      "sha256": "f51d7df92cc89b91511186cee5793bd1a4a5ba85af8d380dcc3c184b58e78ba3"
    },
    {
      "file": "grader_audit.json",
      "bytes": 305,
      "sha256": "2408dace71aa36a81fb8d1e4ebd917fbe2c43f3b87591f4e3aed779f7148a789"
    },
    {
      "file": "source_inputs/train.parquet",
      "source": "gs://lixali-tpu-storage/grpo/tpu-v6e/qwen3-4b-base-grpo-v6e-250steps-32trainer32sampler-ppo-clip02-dual3-kl1e-3-no-prefix-seed1-20261007-115648-f776b0/backup-auto/inputs/jialei/data/gsm8k/train.parquet",
      "generation": "1791374247949787",
      "bytes": 6555601,
      "sha256": "89cd3cb8d28e5274e7f0bf71ff541ea5654ac9e30589ac4b5d19c3f783a3858c"
    },
    {
      "file": "source_inputs/original-validation.parquet",
      "source": "gs://lixali-tpu-storage/grpo/tpu-v6e/qwen3-4b-base-grpo-v6e-250steps-32trainer32sampler-ppo-clip02-dual3-kl1e-3-no-prefix-seed1-20261007-115648-f776b0/backup-auto/inputs/jialei/data/validation/gsm8k_math500_aime2024.parquet",
      "generation": "1791374250149415",
      "bytes": 753286,
      "sha256": "5899a3605cbd6f2326bd692e3c9de6a99c2ea6bb50715fd03e9056aa12800c52"
    }
  ],
  "full_manifest": {
    "generation": "1791442028066980",
    "bytes": 26041,
    "sha256": "afabfceb1f8001b21f65c2369ace73b7896069ace0f44ed216c98e883d593ac1",
    "md5Hash": "TaY3UNEYOgj8CJnJRJcgnA==",
    "crc32c": "1xC1mw==",
    "uri": "gs://ubench-logs/lixali/verl-ray-kueue/assets/qwen3-4b-gsm8k-math500-openmath1000-v1/provenance/validation_manifest.json"
  },
  "cloud_input": {
    "source": "gs://ubench-logs/lixali/verl-ray-kueue/assets/qwen3-4b-gsm8k-math500-openmath1000-v1/inputs/data/gsm8k_math500_openmathinstruct2_1000.parquet",
    "generation": "1791442026168356",
    "bytes": 729951,
    "sha256": "27b4b6f17171771cf41a3045485d4afb5c2c5fd747abdb84b9cafb655c1e45c9",
    "md5Hash": "7RgynKzmcLTB9OMNbRC6MA==",
    "crc32c": "KuMwrQ=="
  }
}
"""
    ),
    "launcher/artifact_mirror.py": (
        r'''#!/usr/bin/env python3
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Mirror one run's staged artifacts using this submission VM's identity.

No credentials are written to disk or forwarded to the shared TPU cluster.
CloudStorage is also used by the submission helper to archive control files.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import fcntl
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

DEFAULT_CONTEXT = "gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap"
METADATA = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/"
API = "https://storage.googleapis.com/storage/v1"
UPLOAD_API = "https://storage.googleapis.com/upload/storage/v1"


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def split_uri(uri):
    parsed = urlsplit(uri)
    if parsed.scheme != "gs" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("Expected a gs://bucket/object URI without query parameters")
    return parsed.netloc, parsed.path.lstrip("/")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def expected_worker_roles(config):
    """Resolve the complete artifact owner set, retaining old one-host names."""
    explicit = config.get("expected_worker_ids")
    if explicit is None:
        hosts = config.get("resources", {}).get("hosts_per_role", 1)
        if isinstance(hosts, dict):
            counts = {role: hosts.get(role, 1) for role in ("actor", "rollout")}
        else:
            counts = {"actor": hosts, "rollout": hosts}
        identities = {"head": "head"}
        for role, count in counts.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(f"{role} hosts_per_role must be a positive integer")
            identities.update({role if count == 1 else f"{role}-{index}": role for index in range(count)})
        return identities
    if (
        not isinstance(explicit, list)
        or not explicit
        or not all(isinstance(item, str) for item in explicit)
        or len(explicit) != len(set(explicit))
    ):
        raise ValueError("expected_worker_ids must be a nonempty list of unique worker names")
    identities = {}
    for worker_id in explicit:
        if not isinstance(worker_id, str) or not re.fullmatch(r"head|(?:actor|rollout)(?:-[0-9]+)?", worker_id):
            raise ValueError("Unexpected worker identity in expected_worker_ids")
        identities[worker_id] = worker_id.split("-", 1)[0]
    if "head" not in identities or set(identities.values()) != {"head", "actor", "rollout"}:
        raise ValueError("expected_worker_ids must contain the head, actor, and rollout roles")
    if "hosts_per_role" in config.get("resources", {}):
        inferred = expected_worker_roles({"resources": config["resources"]})
        if identities != inferred:
            raise ValueError("expected_worker_ids do not match resources.hosts_per_role")
    return identities


class StorageError(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(f"Cloud Storage HTTP {status}: {message}")


class CloudStorage:
    """Small JSON API client authenticated solely by VM metadata credentials."""

    def __init__(self, timeout=120, retries=6):
        self.timeout, self.retries = timeout, retries
        self._token, self._expires, self._identity = None, 0.0, None

    def _metadata(self, endpoint):
        request = Request(METADATA + endpoint, headers={"Metadata-Flavor": "Google"})
        for attempt in range(self.retries):
            try:
                with urlopen(request, timeout=10) as response:
                    return response.read()
            except (HTTPError, URLError, TimeoutError, OSError):
                if attempt + 1 == self.retries:
                    raise RuntimeError("Submission VM metadata credentials are unavailable") from None
                time.sleep(min(2**attempt, 10))

    @property
    def identity(self):
        if self._identity is None:
            self._identity = self._metadata("email").decode().strip()
        return self._identity

    def _access_token(self):
        if self._token is None or time.monotonic() >= self._expires:
            result = json.loads(self._metadata("token"))
            self._token = result["access_token"]
            self._expires = time.monotonic() + max(1, int(result["expires_in"]) - 120)
        return self._token

    def _perform(self, url, *, method="GET", data=None, headers=None):
        # Restrict where credentials can be sent, including resumable-upload URLs.
        if urlsplit(url).scheme != "https" or urlsplit(url).hostname != "storage.googleapis.com":
            raise ValueError("Storage requests must use the Google Cloud Storage HTTPS endpoint")
        for attempt in range(self.retries):
            request = Request(
                url,
                data=data,
                method=method,
                headers={**(headers or {}), "Authorization": "Bearer " + self._access_token()},
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return response.read(), dict(response.headers), response.status
            except HTTPError as error:
                if error.code == 308:  # Resumable upload accepted this chunk.
                    return error.read(), dict(error.headers), error.code
                if error.code == 401:
                    self._token = None
                if error.code not in (401, 408, 429, 500, 502, 503, 504) or attempt + 1 == self.retries:
                    try:
                        message = json.loads(error.read()).get("error", {}).get("message", "request failed")
                    except (ValueError, UnicodeDecodeError):
                        message = "request failed"
                    # Do not include request URLs, headers, or credentials in errors.
                    raise StorageError(error.code, str(message)[:1000]) from None
            except (URLError, TimeoutError, OSError):
                if attempt + 1 == self.retries:
                    raise RuntimeError("Cloud Storage network request failed after retries") from None
            time.sleep(min(2**attempt, 30) + random.random())
        raise RuntimeError("Cloud Storage retries exhausted")

    def request(self, url, *, method="GET", data=None, headers=None):
        if isinstance(data, dict):
            data = json.dumps(data).encode()
            headers = {"Content-Type": "application/json", **(headers or {})}
        body, _, _ = self._perform(url, method=method, data=data, headers=headers)
        return json.loads(body) if body else {}

    def _object_url(self, uri):
        bucket, name = split_uri(uri)
        if not name:
            raise ValueError("An object URI must contain an object name")
        return f"{API}/b/{quote(bucket, safe='')}/o/{quote(name, safe='')}"

    def stat(self, uri, generation=None):
        url = self._object_url(uri)
        if generation is not None:
            url += "?" + urlencode({"generation": generation})
        return self.request(url)

    def get_bytes(self, uri, generation=None):
        query = {"alt": "media"}
        if generation is not None:
            query["generation"] = generation
        return self._perform(self._object_url(uri) + "?" + urlencode(query))[0]

    def get_json(self, uri, generation=None):
        return json.loads(self.get_bytes(uri, generation))

    def list(self, uri):
        bucket, prefix = split_uri(uri)
        query = {
            "prefix": prefix,
            "maxResults": 1000,
            "fields": "nextPageToken,items(bucket,name,generation,size,crc32c,md5Hash,updated,metadata)",
        }
        items = []
        while True:
            page = self.request(f"{API}/b/{quote(bucket, safe='')}/o?" + urlencode(query))
            items.extend(page.get("items", []))
            if not page.get("nextPageToken"):
                return sorted(items, key=lambda item: item["name"])
            query["pageToken"] = page["nextPageToken"]

    @staticmethod
    def verify(source, destination):
        for field in ("size", "crc32c", "md5Hash"):
            if field in source and source[field] != destination.get(field):
                raise RuntimeError(f"Artifact verification failed: {field} differs")
        if "size" not in destination or "crc32c" not in destination:
            raise RuntimeError("Artifact verification failed: missing destination size or CRC32C")
        return destination

    def uploadbytes(
        self,
        payload,
        destination_gs_uri,
        *,
        content_type="application/octet-stream",
        if_generation_match=None,
        metadata=None,
    ):
        if not isinstance(payload, bytes):
            raise TypeError("uploadbytes expects bytes")
        bucket, name = split_uri(destination_gs_uri)
        if not name:
            raise ValueError("An upload destination must contain an object name")
        query = {"uploadType": "media", "name": name}
        if if_generation_match is not None:
            query["ifGenerationMatch"] = str(if_generation_match)
        uploaded = self.request(
            f"{UPLOAD_API}/b/{quote(bucket, safe='')}/o?" + urlencode(query),
            method="POST",
            data=payload,
            headers={"Content-Type": content_type},
        )
        expected = {"size": str(len(payload)), "md5Hash": base64.b64encode(hashlib.md5(payload).digest()).decode()}
        self.verify(expected, uploaded)
        if metadata:
            uploaded = self.request(
                self._object_url(destination_gs_uri) + "?" + urlencode({"ifGenerationMatch": uploaded["generation"]}),
                method="PATCH",
                data={"metadata": metadata},
            )
        return uploaded

    def put_file(self, localpath, destination_gs_uri):
        path = Path(localpath)
        before = path.stat()
        chunk_size = 8 * 1024**2
        if before.st_size <= chunk_size:
            uploaded = self.uploadbytes(path.read_bytes(), destination_gs_uri)
        else:
            bucket, name = split_uri(destination_gs_uri)
            if not name:
                raise ValueError("An upload destination must contain an object name")
            url = f"{UPLOAD_API}/b/{quote(bucket, safe='')}/o?" + urlencode({"uploadType": "resumable", "name": name})
            _, headers, _ = self._perform(
                url,
                method="POST",
                data=b"{}",
                headers={
                    "Content-Type": "application/json",
                    "X-Upload-Content-Length": str(before.st_size),
                    "X-Upload-Content-Type": "application/octet-stream",
                },
            )
            session = next(value for key, value in headers.items() if key.lower() == "location")
            checksum, offset, uploaded = hashlib.md5(), 0, None
            with path.open("rb") as stream:
                while offset < before.st_size:
                    chunk = stream.read(min(chunk_size, before.st_size - offset))
                    if not chunk:
                        raise RuntimeError("Local artifact changed during upload")
                    checksum.update(chunk)
                    end = offset + len(chunk) - 1
                    body, response_headers, status = self._perform(
                        session,
                        method="PUT",
                        data=chunk,
                        headers={
                            "Content-Type": "application/octet-stream",
                            "Content-Range": f"bytes {offset}-{end}/{before.st_size}",
                        },
                    )
                    if status == 308:
                        accepted = next(
                            (value for key, value in response_headers.items() if key.lower() == "range"), ""
                        )
                        if accepted != f"bytes=0-{end}":
                            raise RuntimeError("Resumable artifact upload returned an unexpected offset")
                    else:
                        uploaded = json.loads(body)
                    offset = end + 1
            if uploaded is None:
                raise RuntimeError("Resumable artifact upload did not complete")
            self.verify(
                {"size": str(before.st_size), "md5Hash": base64.b64encode(checksum.digest()).decode()}, uploaded
            )
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Local artifact changed during upload")
        return uploaded

    def copy(self, source_gs_uri, destination_gs_uri, *, expected=None):
        source = expected if expected is not None else self.stat(source_gs_uri)
        source_bucket, source_name = split_uri(source_gs_uri)
        destination_bucket, destination_name = split_uri(destination_gs_uri)
        if not source_name or not destination_name or source_gs_uri == destination_gs_uri:
            raise ValueError("Copy requires two distinct object URIs")
        try:
            existing = self.stat(destination_gs_uri)
        except StorageError as error:
            if error.status != 404:
                raise
        else:
            provenance = existing.get("metadata", {})
            if (
                provenance.get("artifact_mirror_source_generation") == source["generation"]
                and provenance.get("artifact_mirror_source_uri") == source_gs_uri
                and all(existing.get(key) == source.get(key) for key in ("size", "crc32c"))
            ):
                return self.verify(source, existing)
        url = (
            f"{API}/b/{quote(source_bucket, safe='')}/o/{quote(source_name, safe='')}/rewriteTo/"
            f"b/{quote(destination_bucket, safe='')}/o/{quote(destination_name, safe='')}"
        )
        query = {"sourceGeneration": source["generation"], "ifSourceGenerationMatch": source["generation"]}
        metadata = {
            **source.get("metadata", {}),
            "artifact_mirror_source_generation": source["generation"],
            "artifact_mirror_source_uri": source_gs_uri,
        }
        while True:
            result = self.request(url + "?" + urlencode(query), method="POST", data={"metadata": metadata})
            if result.get("done"):
                return self.verify(source, result["resource"])
            if not result.get("rewriteToken"):
                raise RuntimeError("Cloud Storage rewrite did not return a continuation token")
            query["rewriteToken"] = result["rewriteToken"]

    def copy_uri(self, source_gs_uri, destination_gs_uri):
        return self.copy(source_gs_uri, destination_gs_uri)

    def preflight(self, uri_prefix):
        uri = uri_prefix.rstrip("/") + "/_mirror/preflight-" + uuid.uuid4().hex + ".json"
        proof = {"identity": self.identity, "verified_utc": utcnow(), "nonce": uuid.uuid4().hex}
        payload = json.dumps(proof, sort_keys=True).encode()
        uploaded = self.uploadbytes(payload, uri, content_type="application/json", if_generation_match=0)
        try:
            if self.get_bytes(uri, uploaded["generation"]) != payload:
                raise RuntimeError("Private artifact destination write/read preflight failed")
        finally:
            self.request(
                self._object_url(uri) + "?" + urlencode({"ifGenerationMatch": uploaded["generation"]}), method="DELETE"
            )
        return {
            "identity": self.identity,
            "verified_utc": proof["verified_utc"],
            "destination": uri_prefix,
            "write_read_verified": True,
            "probe_removed": True,
        }


class ArtifactMirror:
    def __init__(self, config, state_dir, client=None, context=None, namespace=None):
        self.config, self.client = config, client or CloudStorage()
        self.name, self.run_id = config["name"], config["run_id"]
        self.expected_workers = expected_worker_roles(config)
        if not re.fullmatch(r"[a-z0-9.-]+", self.name) or not re.fullmatch(r"[A-Za-z0-9_.-]+", self.run_id):
            raise ValueError("Run and JobSet identifiers must be single safe path components")
        if self.name in (".", "..") or self.run_id in (".", ".."):
            raise ValueError("Run identifiers cannot be relative path components")
        self.source = config["artifact_gcs_uri"].rstrip("/") + "/" + self.run_id + "/backup-auto/"
        self.destination = config["backup_gcs_uri"].rstrip("/") + "/" + self.run_id + "/backup-auto/"
        self.source_bucket, self.source_name = split_uri(self.source)
        self.destination_bucket, self.destination_name = split_uri(self.destination)
        if self.source == self.destination:
            raise ValueError("Staging and private backup destinations must differ")
        self.context = context or config.get("context", config.get("kubectl_context", DEFAULT_CONTEXT))
        self.namespace = namespace or config.get("namespace", "default")
        self.kubeconfig = config.get("kubeconfig")
        if self.kubeconfig:
            path = Path(self.kubeconfig).resolve()
            if not path.is_file():
                raise ValueError("Run kubeconfig must exist")
            self.kubeconfig = str(path)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "state.json"
        self.state = {
            "version": 1,
            "name": self.name,
            "run_id": self.run_id,
            "source": self.source,
            "destination": self.destination,
            "expected_worker_ids": list(self.expected_workers),
            "objects": {},
            "phase": "starting",
        }
        if self.state_path.exists():
            previous = json.loads(self.state_path.read_text())
            if any(previous.get(key) != self.state[key] for key in ("name", "run_id", "source", "destination")):
                raise ValueError("Backup state belongs to a different run or destination")
            previous_workers = previous.get("expected_worker_ids", ["head", "actor", "rollout"])
            if set(previous_workers) != set(self.expected_workers):
                raise ValueError("Backup state belongs to a different set of workers")
            self.state = previous
            self.state["expected_worker_ids"] = list(self.expected_workers)
        self.lock = (self.state_dir / ".lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another artifact mirror already owns this run") from None
        self.stop_requested = False

    def save(self, phase=None):
        if phase:
            self.state["phase"] = phase
        self.state["updated_utc"] = utcnow()
        write_json(self.state_path, self.state)

    def initialize(self):
        (self.state_dir / "ready.json").unlink(missing_ok=True)
        proof = self.client.preflight(self.destination)
        self.state["preflight"] = proof
        self.save("ready")
        write_json(
            self.state_dir / "ready.json", {**proof, "name": self.name, "run_id": self.run_id, "pid": os.getpid()}
        )
        (self.state_dir / "pid").write_text(str(os.getpid()) + "\n")
        print(f"BACKUP ready: {self.destination} using {self.client.identity}", flush=True)

    @staticmethod
    def signature(objects):
        return [(item["name"], item["generation"], item.get("size"), item.get("crc32c")) for item in objects]

    def copy_current(self, source, destination_uri):
        """Refresh a listed generation only when an overwrite invalidated it."""
        source_uri = "gs://" + self.source_bucket + "/" + source["name"]
        for attempt in range(4):
            try:
                destination = self.client.copy(source_uri, destination_uri, expected=source)
                return source, destination
            except StorageError as error:
                if error.status not in (404, 412) or attempt == 3:
                    raise
                refreshed = self.client.stat(source_uri)
                if refreshed["generation"] == source["generation"]:
                    # A missing immutable generation is not evidence of an overwrite.
                    raise
                source = refreshed
        raise RuntimeError("Artifact generation retries exhausted")

    def synchronize(self):
        sources = self.client.list(self.source)
        destinations = {item["name"]: item for item in self.client.list(self.destination)}
        copied, pending = 0, []
        for index, source in enumerate(sources):
            if self.stop_requested:
                self.save()
                raise RuntimeError("Artifact mirror stopped during synchronization")
            relative = source["name"][len(self.source_name) :]
            if not relative or relative.startswith("/") or ".." in relative.split("/"):
                raise RuntimeError("Staging artifact has an unsafe relative name")
            target_uri = self.destination + relative
            destination = destinations.get(self.destination_name + relative)
            matching = destination and all(destination.get(key) == source.get(key) for key in ("size", "crc32c"))
            matching = (
                matching
                and destination.get("metadata", {}).get("artifact_mirror_source_generation") == source["generation"]
            )
            source_uri = "gs://" + self.source_bucket + "/" + source["name"]
            matching = matching and destination.get("metadata", {}).get("artifact_mirror_source_uri") == source_uri
            if matching:
                self.client.verify(source, destination)
            else:
                try:
                    source, destination = self.copy_current(source, target_uri)
                except StorageError as error:
                    if error.status not in (404, 412):
                        raise
                    # Keep copying other artifacts; a changing log must not starve
                    # the remaining workers. This snapshot cannot be finalized.
                    pending.append(relative)
                    continue
                sources[index] = source
                copied += 1
            self.state["objects"][relative] = {
                "source_generation": source["generation"],
                "destination_generation": destination["generation"],
                "size": destination["size"],
                "crc32c": destination["crc32c"],
                "md5Hash": destination.get("md5Hash"),
            }
            if copied and copied % 20 == 0:
                self.save()
        self.state["last_sync"] = {
            "utc": utcnow(),
            "source_objects": len(sources),
            "copied_objects": copied,
            "pending_objects": pending,
        }
        self.save()
        if copied:
            print(f"BACKUP copied and verified {copied} objects; staging contains {len(sources)}", flush=True)
        if pending:
            raise RuntimeError(
                f"Artifact snapshot incomplete: {len(pending)} source generations changed or disappeared"
            )
        return sources

    def jobset(self):
        command = [
            "kubectl",
            "--context",
            self.context,
            "--namespace",
            self.namespace,
            "--request-timeout=30s",
            "get",
            "jobset",
            self.name,
            "-o",
            "json",
        ]
        if self.kubeconfig:
            command[1:1] = ["--kubeconfig", self.kubeconfig]
        result = subprocess.run(command, capture_output=True, text=True, timeout=40)
        if result.returncode:
            if "NotFound" in result.stderr:
                return None
            raise RuntimeError("Cannot read JobSet status in the configured Kubernetes context")
        jobset = json.loads(result.stdout)
        return {
            "name": self.name,
            "namespace": self.namespace,
            "uid": jobset["metadata"]["uid"],
            "status": jobset.get("status", {}),
        }

    @staticmethod
    def terminal(jobset):
        return jobset is not None and any(
            condition.get("status") == "True" and condition.get("type") in ("Completed", "Failed")
            for condition in jobset["status"].get("conditions", [])
        )

    def role_completions(self, sources):
        indexed = {item["name"][len(self.source_name) :]: item for item in sources}
        markers = {}
        legacy = set(self.expected_workers) == {"head", "actor", "rollout"}
        for worker_id, role in self.expected_workers.items():
            relative = f"pods/{worker_id}/metadata/backup-complete.json"
            source = indexed.get(relative)
            if source is None:
                raise RuntimeError(f"Final artifact marker missing for {worker_id}")
            marker = self.client.get_json(self.source + relative, source["generation"])
            identity = marker.get("worker_id", worker_id if legacy else None)
            if (
                marker.get("run_id") != self.run_id
                or marker.get("role") != role
                or identity != worker_id
                or marker.get("backup_verified") is not True
            ):
                raise RuntimeError(f"Final artifact marker invalid for {worker_id}")
            markers[worker_id] = marker
        return markers

    def finalize(self, jobset):
        if not self.terminal(jobset):
            raise RuntimeError("Final artifact backup requires a terminal JobSet")
        for _ in range(5):
            sources = self.synchronize()
            refreshed = self.client.list(self.source)
            # A long copy can outlive the head's final upload. Check that the
            # inventory is current before judging whether its markers exist.
            if self.signature(sources) != self.signature(refreshed):
                continue
            markers = self.role_completions(sources)
            break
        else:
            raise RuntimeError("Staged artifacts continued changing during final verification")
        # Publish a completion manifest only after all roles and the JobSet finish.
        inventory = {}
        destinations = {item["name"]: item for item in self.client.list(self.destination)}
        for source in sources:
            relative = source["name"][len(self.source_name) :]
            destination = destinations.get(self.destination_name + relative)
            if destination is None:
                raise RuntimeError(f"Final artifact missing from private backup: {relative}")
            self.client.verify(source, destination)
            provenance = destination.get("metadata", {})
            if (
                provenance.get("artifact_mirror_source_generation") != source["generation"]
                or provenance.get("artifact_mirror_source_uri") != self.source + relative
            ):
                raise RuntimeError(f"Final artifact source generation differs: {relative}")
        for item in destinations.values():
            relative = item["name"][len(self.destination_name) :]
            if relative == "backup-manifest.json" or relative.startswith("_mirror/preflight-"):
                continue
            inventory[relative] = {
                key: item[key] for key in ("generation", "size", "crc32c", "md5Hash", "metadata") if key in item
            }
        completed_state = {
            **self.state,
            "jobset": jobset,
            "role_completions": markers,
            "completed_utc": utcnow(),
            "updated_utc": utcnow(),
            "phase": "completed",
        }
        completed_state.pop("last_error", None)
        manifest = {
            **completed_state,
            "backup_verified": True,
            "inventory": inventory,
            "object_count": len(inventory),
            "staged_object_count": len(sources),
            "total_bytes": sum(int(item["size"]) for item in inventory.values()),
        }
        self.client.uploadbytes(
            json.dumps(manifest, indent=2, sort_keys=True).encode(),
            self.destination + "backup-manifest.json",
            content_type="application/json",
        )
        self.state = completed_state
        self.save("completed")
        write_json(self.state_dir / "complete.json", manifest)
        print(f"BACKUP complete: verified {len(inventory)} objects at {self.destination}", flush=True)

    def follow(self, interval=60, final_timeout=300):
        terminal_since = None
        while not self.stop_requested:
            jobset = None
            try:
                jobset = self.jobset()
                self.state["jobset"] = jobset
                self.save()
                if self.terminal(jobset):
                    terminal_since = terminal_since or time.monotonic()
                    self.finalize(jobset)
                    return
                self.synchronize()
                self.save("running" if jobset else "waiting_for_jobset")
                self.state.pop("last_error", None)
            except (StorageError, RuntimeError, TimeoutError, subprocess.TimeoutExpired) as error:
                self.state["last_error"] = str(error)
                self.save("retrying")
                print(f"BACKUP retrying: {error}", flush=True)
                if jobset is not None and self.terminal(jobset):
                    terminal_since = terminal_since or time.monotonic()
                if terminal_since is not None and time.monotonic() - terminal_since >= final_timeout:
                    self.save("incomplete")
                    raise RuntimeError("Final artifact backup could not be verified before its deadline") from error
            deadline = time.monotonic() + interval
            while not self.stop_requested and time.monotonic() < deadline:
                time.sleep(min(1, deadline - time.monotonic()))
        self.save("stopped")
        raise RuntimeError("Artifact mirror stopped before final backup verification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Verify destination and mirror the current artifact snapshot")
    mode.add_argument(
        "--follow", action="store_true", help="Mirror until JobSet terminal state and verified final backup"
    )
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--context")
    parser.add_argument("--namespace")
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--final-timeout", type=float, default=300)
    args = parser.parse_args()
    if args.interval <= 0 or args.final_timeout <= 0:
        parser.error("Backup intervals and deadlines must be positive")
    config = json.loads(args.config.read_text())
    state_dir = args.state_dir or Path(__file__).resolve().parent / "backup-state" / config["name"]
    mirror = ArtifactMirror(config, state_dir, context=args.context, namespace=args.namespace)
    signal.signal(signal.SIGTERM, lambda *_: setattr(mirror, "stop_requested", True))
    signal.signal(signal.SIGINT, lambda *_: setattr(mirror, "stop_requested", True))
    try:
        mirror.initialize()
        if args.once:
            mirror.synchronize()
            mirror.save("snapshot_verified")
        else:
            mirror.follow(args.interval, args.final_timeout)
        return 0
    except Exception as error:
        mirror.state["last_error"] = str(error)
        mirror.save("incomplete")
        print(f"BACKUP failed: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        (mirror.state_dir / "pid").unlink(missing_ok=True)
        (mirror.state_dir / "ready.json").unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
'''
    ),
    "launcher/bootstrap.py": (
        r'''#!/usr/bin/env python3
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Run a manually managed Ray head/TPU worker within a Kueue JobSet.

Required: ROLE=head|actor|rollout, RUN_ID, HEAD_HOST, CODE_BUNDLE_URI.
The source bundle contains ENTRYPOINT_FILE (default train.sh). All pods must
mount the same /data bucket when ARTIFACT_MOUNT is used. This script creates no
Kubernetes resources and does not change the supplied training command.
"""

import glob
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROLE = os.environ["ROLE"]
RUN_ID = os.environ["RUN_ID"]
HEAD_HOST = os.environ["HEAD_HOST"]


def worker_identities(actor_count=1, rollout_count=1):
    """Keep legacy single-host names and identify every multi-host worker."""
    identities = {}
    for role, count in (("actor", actor_count), ("rollout", rollout_count)):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{role} worker count must be a positive integer")
        identities.update({role if count == 1 else f"{role}-{index}": role for index in range(count)})
    return identities


ACTOR_WORKER_COUNT = int(os.environ.get("ACTOR_WORKER_COUNT", "1"))
ROLLOUT_WORKER_COUNT = int(os.environ.get("ROLLOUT_WORKER_COUNT", "1"))
LOGICAL_TPU_DEVICES = int(os.environ.get("LOGICAL_TPU_DEVICES", "8"))
if LOGICAL_TPU_DEVICES < 1:
    raise ValueError("LOGICAL_TPU_DEVICES must be a positive integer")
EXPECTED_WORKERS = worker_identities(ACTOR_WORKER_COUNT, ROLLOUT_WORKER_COUNT)
WORKER_INDEX = int(os.environ.get("WORKER_INDEX", "0"))
WORKER_ID = (
    "head"
    if ROLE == "head"
    else ROLE
    if (ACTOR_WORKER_COUNT if ROLE == "actor" else ROLLOUT_WORKER_COUNT) == 1
    else f"{ROLE}-{WORKER_INDEX}"
)
if ROLE != "head" and (
    WORKER_INDEX < 0
    or WORKER_INDEX >= (ACTOR_WORKER_COUNT if ROLE == "actor" else ROLLOUT_WORKER_COUNT)
    or WORKER_ID not in EXPECTED_WORKERS
):
    raise ValueError("WORKER_INDEX must identify a configured worker for this role")
WORKDIR = Path(os.environ.get("WORKDIR", "/opt/run/source"))
STATUS_PORT = int(os.environ.get("STATUS_PORT", "8766"))
GCS_PORT = int(os.environ.get("RAY_GCS_PORT", "6379"))
DASHBOARD_PORT = int(os.environ.get("RAY_DASHBOARD_PORT", "8265"))
CLIENT_PORT = int(os.environ.get("RAY_CLIENT_PORT", "20001"))
STATUS_URL = f"http://{HEAD_HOST}:{STATUS_PORT}"
START_TIMEOUT = int(os.environ.get("START_TIMEOUT_SECONDS", "900"))
if START_TIMEOUT < 0:
    raise ValueError("START_TIMEOUT_SECONDS must be zero (unlimited) or positive")
STOP = threading.Event()
STATE_LOCK = threading.Lock()
STATE = {
    "run_id": RUN_ID,
    "phase": "starting",
    "exit_code": None,
    "startup_timeout_seconds": START_TIMEOUT,
    "expected_workers": sorted(EXPECTED_WORKERS),
    "worker_registered": {},
    "worker_failures": {},
    "worker_done": [],
}
SOURCE_BUNDLE_PATH = None
ARTIFACT_MANAGER = None
STORAGE_DOWNLOAD_CHUNK_BYTES = 32 * 1024 * 1024
STORAGE_RANGE_MIN_BYTES = 64 * 1024 * 1024


def startup_deadline():
    """Zero disables the startup clock while Kubernetes provisions TPU hosts."""
    return time.monotonic() + START_TIMEOUT if START_TIMEOUT else None


def startup_expired(deadline):
    return deadline is not None and time.monotonic() > deadline


def close_ray_driver():
    """Stop this process's CoreWorker before verifying its final log files."""
    ray_driver = sys.modules.get("ray")
    if ray_driver is not None and ray_driver.is_initialized():
        ray_driver.shutdown()


class ArtifactChangedDuringUpload(RuntimeError):
    """The final upload must retry after an artifact writer becomes quiet."""


def artifact_layout():
    template = os.environ.get("ARTIFACT_LAYOUT", "{run_id}/backup-auto/pods/{worker_id}")
    if WORKER_ID != ROLE and "{worker_id}" not in template:
        raise ValueError("Multi-host ARTIFACT_LAYOUT must include {worker_id}")
    layout = template.format(run_id=RUN_ID, role=ROLE, worker_id=WORKER_ID, worker_index=WORKER_INDEX).rstrip("/")
    if not layout or Path(layout).is_absolute() or ".." in Path(layout).parts:
        raise ValueError("ARTIFACT_LAYOUT must be a relative object prefix")
    return layout


def artifact_local_root():
    return Path(os.environ.get("ARTIFACT_LOCAL_ROOT", f"/tmp/bootstrap/{RUN_ID}/{WORKER_ID}"))


class LogTee:
    """Capture this process's stdout/stderr without dumping its environment."""

    def __init__(self, stream, destination, lock):
        self.stream, self.destination, self.lock = stream, destination, lock

    def write(self, message):
        with self.lock:
            self.stream.write(message)
            self.destination.write(message)
            self.destination.flush()
        return len(message)

    def flush(self):
        with self.lock:
            self.stream.flush()
            self.destination.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def initialize_logging():
    root = artifact_local_root()
    root.mkdir(parents=True, exist_ok=True)
    output = (root / "bootstrap.log").open("a", buffering=1)
    lock = threading.Lock()
    sys.stdout = LogTee(sys.stdout, output, lock)
    sys.stderr = LogTee(sys.stderr, output, lock)


def log(message):
    print(f"BOOTSTRAP [{WORKER_ID}] {message}", flush=True)


def command(argv, **kwargs):
    log("exec " + " ".join(argv))
    return subprocess.run(argv, check=True, **kwargs)


def storage_download(uri, target, generation=None):
    if generation is not None:
        generation = int(generation)
        if generation < 1:
            raise ValueError("Cloud Storage generation must be positive")
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    try:
        from google.cloud import storage
    except ImportError:
        if generation is not None:
            raise RuntimeError("Generation-pinned downloads require google-cloud-storage") from None
        import gcsfs

        gcsfs.GCSFileSystem(token="google_default").get(uri, str(target))
    else:
        import base64
        import google_crc32c
        from google.cloud.storage.retry import DEFAULT_RETRY
        from requests.exceptions import ChunkedEncodingError

        bucket_name, blob_name = uri[5:].split("/", 1)
        bucket = storage.Client().bucket(bucket_name)
        blob = bucket.blob(blob_name, generation=generation)
        retry = DEFAULT_RETRY.with_deadline(600)
        blob.reload(if_generation_match=generation, timeout=(20, 120), retry=retry)
        generation = int(blob.generation)
        size = int(blob.size)
        if size < STORAGE_RANGE_MIN_BYTES:
            blob.download_to_filename(
                str(target), if_generation_match=generation, timeout=(20, 120), retry=retry
            )
            return

        if not blob.crc32c:
            raise RuntimeError("Large download requires the pinned object's CRC32C checksum")
        expected_crc = blob.crc32c
        partial = Path(target).with_name(Path(target).name + ".download-part")
        checksum = google_crc32c.Checksum()
        try:
            with partial.open("wb") as destination:
                for offset in range(0, size, STORAGE_DOWNLOAD_CHUNK_BYTES):
                    end = min(size, offset + STORAGE_DOWNLOAD_CHUNK_BYTES) - 1
                    expected_bytes = end - offset + 1
                    def read_range():
                        # Each retry gets a fresh SDK byte buffer, never partial bytes from a prior response.
                        data = blob.download_as_bytes(
                            start=offset, end=end, raw_download=True, checksum=None,
                            if_generation_match=generation, timeout=(20, 120), retry=None,
                        )
                        if len(data) != expected_bytes:
                            raise ChunkedEncodingError(
                                f"Model range {offset}-{end}: {len(data)}/{expected_bytes} bytes"
                            )
                        return data

                    data = retry(read_range)()
                    destination.write(data)
                    checksum.update(data)
                    if end + 1 == size or (offset // STORAGE_DOWNLOAD_CHUNK_BYTES + 1) % 16 == 0:
                        log(f"Downloaded {Path(target).name}: {end + 1}/{size} bytes")
            if partial.stat().st_size != size:
                raise RuntimeError("Large download byte count differs from pinned object metadata")
            if base64.b64encode(checksum.digest()).decode("ascii") != expected_crc:
                raise RuntimeError("Large download CRC32C differs from pinned object metadata")
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)


def initialize_data():
    manifest = os.environ.get("DATA_MANIFEST_URI")
    prefix = os.environ.get("DATA_GCS_PREFIX")
    if manifest:
        local = Path(tempfile.gettempdir()) / f"{RUN_ID}-data-manifest.json"
        storage_download(manifest, local)
        entries = json.loads(local.read_text())
        for entry in entries:
            destination = Path(entry["destination"])
            if not destination.is_absolute() or not destination.resolve().is_relative_to(Path("/data").resolve()):
                raise ValueError(f"Data/model destination must be under /data: {destination}")
            storage_download(entry["source"], destination, generation=entry.get("generation"))
            for field in ("size", "bytes"):
                if field in entry and destination.stat().st_size != int(entry[field]):
                    raise ValueError(f"Data/model byte count mismatch: {destination}")
            if entry.get("sha256") is not None:
                digest = hashlib.sha256()
                with destination.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != entry["sha256"]:
                    raise ValueError(f"Data/model SHA256 mismatch: {destination}")
        log(f"downloaded {len(entries)} data/model files from manifest")
    elif prefix:
        base = prefix.rstrip("/") + "/"
        root = Path(os.environ.get("DATA_LOCAL_ROOT", "/data"))
        try:
            from google.cloud import storage
        except ImportError:
            import gcsfs

            fs = gcsfs.GCSFileSystem(token="google_default")
            files = ["gs://" + path for path in fs.find(base)]
        else:
            bucket, name = base[5:].split("/", 1)
            files = [
                f"gs://{bucket}/{blob.name}"
                for blob in storage.Client().list_blobs(bucket, prefix=name)
                if not blob.name.endswith("/")
            ]
        count = 0
        for uri in files:
            relative = Path(uri[len(base) :])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid data object path")
            storage_download(uri, root / relative)
            count += 1
        if not count:
            raise RuntimeError(f"No staged data objects at {base}")
        log(f"downloaded {count} staged data/model files under {root}")


def initialize_source():
    global SOURCE_BUNDLE_PATH
    uri = os.environ["CODE_BUNDLE_URI"]
    WORKDIR.mkdir(parents=True, exist_ok=True)
    if uri.startswith("gs://"):
        bundle = Path(tempfile.gettempdir()) / f"{RUN_ID}-source.tar.gz"
        storage_download(uri, bundle)
    else:
        bundle = Path(uri)
    expected_hash = os.environ["CODE_BUNDLE_SHA256"]
    digest = hashlib.sha256()
    with bundle.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError("Source bundle SHA256 mismatch")
    SOURCE_BUNDLE_PATH = bundle
    with tarfile.open(bundle, "r:*") as archive:
        # Bundles are generated locally; reject any unexpected outside paths.
        base = WORKDIR.resolve()
        for member in archive.getmembers():
            path = (base / member.name).resolve()
            if not path.is_relative_to(base) or member.issym() or member.islnk():
                raise ValueError(f"Unsafe archive member: {member.name}")
        archive.extractall(WORKDIR)
    os.chdir(WORKDIR)
    paths = [str(WORKDIR)]
    titan = os.environ.get("TORCHTITAN_DIR")
    if titan:
        paths.append(titan)
    if (WORKDIR / "torchtitan").is_dir():
        paths.append(str(WORKDIR / "torchtitan"))
    paths.extend(filter(None, os.environ.get("PYTHONPATH", "").split(":")))
    os.environ["PYTHONPATH"] = ":".join(dict.fromkeys(paths))
    if not (WORKDIR / "verl_hardware_plugin/__init__.py").is_file():
        raise RuntimeError("The source bundle must include verl_hardware_plugin at its top level")
    os.environ.setdefault("VERL_PLATFORM", "tpu")
    os.environ["VERL_USE_EXTERNAL_MODULES"] = "verl_hardware_plugin"
    os.environ.setdefault("TPU_ACCELERATOR_TYPE", "v6e")
    os.environ["RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS"] = "1"
    log(f"source ready: {WORKDIR}")


def initialize_artifacts():
    mount = os.environ.get("ARTIFACT_MOUNT")
    if mount:
        root = Path(mount) / artifact_layout()
        root.mkdir(parents=True, exist_ok=True)
        dump = root / "verl_dump"
        dump.mkdir(exist_ok=True)
        local = Path("/tmp/verl_dump")
        if local.is_symlink():
            local.unlink()
        elif local.exists():
            shutil.copytree(local, dump, dirs_exist_ok=True)
            shutil.rmtree(local)
        local.symlink_to(dump, target_is_directory=True)
        os.environ.setdefault("TENSORBOARD_DIR", str(root / "tensorboard"))
    else:
        (Path("/tmp/verl_dump") / RUN_ID).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TENSORBOARD_DIR", f"/tmp/tensorboard/{RUN_ID}/{WORKER_ID}")
    Path(os.environ["TENSORBOARD_DIR"]).mkdir(parents=True, exist_ok=True)


def credential_content(name, content=None):
    """Recognize credential files, including service-account keys under any name."""
    path = Path(name)
    lower = path.name.lower()
    if (
        ".git" in path.parts
        or lower
        in {
            "credentials.json",
            "application_default_credentials.json",
            "service-account.json",
            "service_account.json",
            "sa-key.json",
            "id_rsa",
            "id_ed25519",
            ".env",
            "token.json",
            "token",
            "credentials",
            "credentials.db",
            "access_tokens.db",
            ".netrc",
            ".git-credentials",
        }
        or lower.endswith((".pem", ".key", ".p12", ".pfx"))
    ):
        return True
    if content and path.suffix.lower() in (".json", ".yaml", ".yml"):
        if path.suffix.lower() == ".json":
            try:
                data = json.loads(content)
            except (ValueError, UnicodeDecodeError):
                return False
        else:
            import yaml

            try:
                data = yaml.safe_load(content)
            except (yaml.YAMLError, UnicodeDecodeError):
                return False

        def contains_secret(value):
            if isinstance(value, dict):
                return any(
                    (
                        str(key).lower()
                        in {
                            "private_key",
                            "refresh_token",
                            "client_secret",
                            "access_token",
                            "wandb_api_key",
                            "hf_token",
                            "hugging_face_hub_token",
                        }
                        and item
                    )
                    or contains_secret(item)
                    for key, item in value.items()
                )
            if isinstance(value, list):
                return any(contains_secret(item) for item in value)
            return False

        return bool(contains_secret(data))
    return False


def credential_file(path):
    content = None
    if path.suffix.lower() in (".json", ".yaml", ".yml") and path.stat().st_size <= 4 * 1024**2:
        content = path.read_bytes()
    return credential_content(str(path), content)


def source_bundle_safe():
    if SOURCE_BUNDLE_PATH is None:
        return False
    with tarfile.open(SOURCE_BUNDLE_PATH, "r:*") as archive:
        for member in archive.getmembers():
            content = None
            if (
                member.isfile()
                and member.name.lower().endswith((".json", ".yaml", ".yml"))
                and member.size <= 4 * 1024**2
            ):
                content = archive.extractfile(member).read()
            if credential_content(member.name, content):
                return False
    return True


def artifact_targets():
    # Ray tasks can write the head's TensorBoard directory on either TPU pod.
    tensorboard = Path("/tmp/tensorboard") / RUN_ID
    if not tensorboard.exists():
        tensorboard = Path(os.environ.get("TENSORBOARD_DIR", "/tmp/tensorboard"))
    targets = {
        "verl_dump": Path("/tmp/verl_dump") / RUN_ID,
        "tensorboard": tensorboard,
        "ray_logs": Path("/tmp/ray/session_latest/logs"),
        "metadata": artifact_local_root(),
        "checkpoints": Path("/tmp/verl_checkpoints") / RUN_ID,
        "checkpoints/workdir": WORKDIR / "checkpoints",
        "configs/hydra": Path("/tmp/verl_hydra") / RUN_ID,
        "configs/workdir_hydra": WORKDIR / ".hydra",
        "configs/outputs": WORKDIR / "outputs",
        "metrics": Path(os.environ.get("VERL_FILE_LOGGER_ROOT", f"/tmp/verl_metrics/{RUN_ID}")),
        "source/launcher.sh": WORKDIR / os.environ.get("ENTRYPOINT_FILE", "train.sh"),
        "source/runtime_env.yaml": WORKDIR / os.environ.get("RUNTIME_ENV_FILE", "verl/trainer/runtime_env.yaml"),
    }
    if SOURCE_BUNDLE_PATH is not None and source_bundle_safe():
        targets["source/source.tar.gz"] = SOURCE_BUNDLE_PATH
    extra = json.loads(os.environ.get("ARTIFACT_EXTRA_DIRS", "{}"))
    if not isinstance(extra, dict):
        raise ValueError("ARTIFACT_EXTRA_DIRS must be a JSON object mapping labels to paths")
    for label, path in extra.items():
        if not label or Path(label).is_absolute() or ".." in Path(label).parts:
            raise ValueError("Artifact labels must be relative paths")
        targets[label] = Path(path)
    return targets


class ArtifactStore:
    """Storage adapter; credential selection remains the existing ADC behavior."""

    def __init__(self):
        self.root = None
        self.bucket = None
        self.fs = None
        mount = os.environ.get("ARTIFACT_MOUNT")
        uri = os.environ.get("ARTIFACT_GCS_URI")
        if os.environ.get("ARTIFACT_UPLOAD_ENDPOINT"):
            raise ValueError("HTTP artifact uploader endpoint is not configured in this build")
        if mount:
            self.root = Path(mount) / artifact_layout()
            self.root.mkdir(parents=True, exist_ok=True)
        elif uri:
            if not uri.startswith("gs://"):
                raise ValueError("ARTIFACT_GCS_URI must start with gs://")
            self.bucket_name, _, prefix = uri[5:].rstrip("/").partition("/")
            self.prefix = "/".join(filter(None, (prefix, artifact_layout())))
            try:
                from google.cloud import storage
            except ImportError:
                import gcsfs

                self.fs = gcsfs.GCSFileSystem(token="google_default")
            else:
                self.bucket = storage.Client().bucket(self.bucket_name)
        else:
            raise ValueError("An artifact destination is required before training")

    def upload(self, path, relative):
        if self.root is not None:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.resolve() != path.resolve():
                shutil.copy2(path, target)
        elif self.bucket is not None:
            self.bucket.blob(f"{self.prefix}/{relative}").upload_from_filename(str(path), timeout=120)
        else:
            self.fs.put(str(path), f"gs://{self.bucket_name}/{self.prefix}/{relative}")

    def read(self, relative):
        if self.root is not None:
            return (self.root / relative).read_bytes()
        if self.bucket is not None:
            return self.bucket.blob(f"{self.prefix}/{relative}").download_as_bytes(timeout=60)
        return self.fs.cat_file(f"gs://{self.bucket_name}/{self.prefix}/{relative}")


class ArtifactManager:
    def __init__(self, store=None):
        self.store = store if store is not None else ArtifactStore()
        self.interval = float(os.environ.get("ARTIFACT_UPLOAD_INTERVAL_SECONDS", "60"))
        if not math.isfinite(self.interval) or self.interval <= 0:
            raise ValueError("ARTIFACT_UPLOAD_INTERVAL_SECONDS must be positive")
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.uploaded = {}
        self.thread = None

    def write_metadata(self, phase, exit_code=None):
        root = artifact_local_root()
        root.mkdir(parents=True, exist_ok=True)
        metadata = {
            "run_id": RUN_ID,
            "role": ROLE,
            "worker_id": WORKER_ID,
            "phase": phase,
            "exit_code": exit_code,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "code_bundle_sha256": os.environ.get("CODE_BUNDLE_SHA256"),
            "source_bundle_backup_included": source_bundle_safe(),
            "entrypoint": os.environ.get("ENTRYPOINT_FILE", "train.sh"),
            "artifact_layout": artifact_layout(),
            "tpu_accelerator_type": os.environ.get("TPU_ACCELERATOR_TYPE", "v6e"),
            "logical_tpu_devices_per_host": LOGICAL_TPU_DEVICES,
        }
        (root / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def verify_write(self):
        root = artifact_local_root()
        root.mkdir(parents=True, exist_ok=True)
        probe = root / "write-check.json"
        probe.write_text(
            json.dumps({"run_id": RUN_ID, "role": ROLE, "worker_id": WORKER_ID, "check": time.time_ns()}) + "\n"
        )
        self.store.upload(probe, "metadata/write-check.json")
        if self.store.read("metadata/write-check.json") != probe.read_bytes():
            raise RuntimeError("Artifact write/read verification failed")
        log("artifact destination write/read verified before training")

    def flush(self, strict=True):
        errors = []
        changing = []
        count = 0
        with self.lock:
            for label, source in artifact_targets().items():
                if not source.exists():
                    continue
                files = [source] if source.is_file() else source.rglob("*")
                for path in files:
                    if not strict and self.stop_event.is_set():
                        return count
                    before = None
                    try:
                        if not path.is_file() or path.is_symlink() or credential_file(path):
                            continue
                        relative = label if source.is_file() else f"{label}/{path.relative_to(source)}"
                        # A completion marker is published only by finish after
                        # every final upload succeeds, never by directory scans.
                        if relative == "metadata/backup-complete.json":
                            continue
                        before = path.stat()
                        stamp = (str(path.resolve()), before.st_size, before.st_mtime_ns)
                        if self.uploaded.get(relative) == stamp:
                            continue
                        self.store.upload(path, relative)
                        after = path.stat()
                        if (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns):
                            self.uploaded[relative] = stamp
                        elif strict:
                            changing.append(relative)
                        count += 1
                    except Exception as error:
                        changed = isinstance(error, FileNotFoundError)
                        if strict and before is not None and not changed:
                            try:
                                after = path.stat()
                                changed = (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns)
                            except FileNotFoundError:
                                changed = True
                        if strict and changed:
                            changing.append(str(path))
                        else:
                            errors.append(f"{label}: {type(error).__name__}")
            if self.store.root is not None:
                os.sync()
        if errors:
            raise RuntimeError(f"Artifact backup incomplete ({len(errors)} files): " + "; ".join(errors[:3]))
        if changing:
            raise ArtifactChangedDuringUpload("Artifacts changed during final upload: " + ", ".join(changing[:3]))
        return count

    def file_stamps(self):
        """Inspect the same eligible artifacts as flush, including newly created files."""
        stamps = {}
        for label, source in artifact_targets().items():
            if not source.exists():
                continue
            files = [source] if source.is_file() else source.rglob("*")
            for path in files:
                try:
                    if not path.is_file() or path.is_symlink() or credential_file(path):
                        continue
                    relative = label if source.is_file() else f"{label}/{path.relative_to(source)}"
                    if relative == "metadata/backup-complete.json":
                        continue
                    status = path.stat()
                    stamps[relative] = (str(path.resolve()), status.st_size, status.st_mtime_ns)
                except FileNotFoundError as error:
                    raise ArtifactChangedDuringUpload("Artifact files changed during final verification") from error
        return stamps

    def flush_final_stable(self):
        timeout = float(os.environ.get("ARTIFACT_FINAL_FLUSH_TIMEOUT_SECONDS", "240"))
        quiet = float(os.environ.get("ARTIFACT_FINAL_QUIET_SECONDS", "2"))
        if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(quiet) or quiet <= 0:
            raise ValueError("Final artifact flush timeout and quiet interval must be positive")
        deadline = time.monotonic() + timeout
        total = 0
        while True:
            try:
                total += self.flush(strict=True)
                confirmed = self.file_stamps()
                if any(self.uploaded.get(relative) != stamp for relative, stamp in confirmed.items()):
                    raise ArtifactChangedDuringUpload("Artifacts changed after the final upload scan")
                if deadline - time.monotonic() < quiet:
                    raise TimeoutError("Final backup could not complete its quiet verification before the deadline")
                time.sleep(quiet)
                if self.file_stamps() == confirmed:
                    return total
            except ArtifactChangedDuringUpload:
                time.sleep(min(quiet, max(0, deadline - time.monotonic())))
            if time.monotonic() >= deadline:
                raise TimeoutError("Artifacts continued changing before final backup verification")

    def start(self):
        def backup_loop():
            while not self.stop_event.wait(self.interval):
                try:
                    self.write_metadata("running")
                    self.flush(strict=False)
                except Exception as error:
                    # Emit one compact notice per interval; final flush is strict.
                    log(f"periodic artifact backup will retry: {error}")

        self.thread = threading.Thread(target=backup_loop, name="artifact-backup", daemon=True)
        self.thread.start()

    def finish(self, exit_code):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=float(os.environ.get("ARTIFACT_BACKUP_JOIN_TIMEOUT_SECONDS", "240")))
            if self.thread.is_alive():
                raise TimeoutError("Periodic artifact upload did not stop before final flush")
        self.write_metadata("flushing", exit_code)
        count = self.flush_final_stable()
        self.write_metadata("completed", exit_code)
        root = artifact_local_root()
        self.store.upload(root / "run.json", "metadata/run.json")
        log(f"final artifact backup verified: {count} changed files uploaded; publishing role completion marker")
        if (root / "bootstrap.log").is_file():
            self.store.upload(root / "bootstrap.log", "metadata/bootstrap.log")
        marker = root / "backup-complete.json"
        marker.write_text(
            json.dumps(
                {
                    "run_id": RUN_ID,
                    "role": ROLE,
                    "worker_id": WORKER_ID,
                    "exit_code": exit_code,
                    "backup_verified": True,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            + "\n"
        )
        self.store.upload(marker, "metadata/backup-complete.json")


def upload_artifacts():
    manager = ARTIFACT_MANAGER if ARTIFACT_MANAGER is not None else ArtifactManager()
    return manager.flush(strict=True)


def http_json(path="/status", payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(STATUS_URL + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=10) as response:
        return json.load(response)


def record_worker_update(path, payload):
    """Validate worker identity before changing the head's coordination state."""
    if path not in ("/register", "/failure", "/done"):
        return 404, {"error": "unknown worker status endpoint"}
    role = payload.get("role")
    worker_id = payload.get("worker_id", role)
    if (
        payload.get("run_id") != RUN_ID
        or role not in ("actor", "rollout")
        or not isinstance(worker_id, str)
        or EXPECTED_WORKERS.get(worker_id) != role
    ):
        return 400, {"error": "wrong run, role, or worker identity"}
    with STATE_LOCK:
        if path == "/register":
            STATE["worker_registered"][worker_id] = {"role": role, "node_ip": payload.get("node_ip")}
        elif path == "/failure":
            STATE["worker_failures"][worker_id] = payload.get("message", "worker failed")
        elif path == "/done":
            if worker_id not in STATE["worker_done"]:
                STATE["worker_done"].append(worker_id)
            if payload.get("exit_code", 0):
                STATE["worker_failures"][worker_id] = f"Worker exit code {payload['exit_code']}"
        return 200, json.loads(json.dumps(STATE))


def registered_worker_nodes(nodes):
    """Count alive nodes exposing the configured TPU devices in each role's pool."""
    counts = {"actor": 0, "rollout": 0}
    for node in nodes:
        if not node.get("Alive") or node.get("Resources", {}).get("TPU", 0) != LOGICAL_TPU_DEVICES:
            continue
        for role, group in (("actor", "tpu-group-0"), ("rollout", "tpu-group-1")):
            if node.get("Resources", {}).get(group, 0) >= 1:
                counts[role] += 1
    return counts


def workers_ready(nodes):
    counts = registered_worker_nodes(nodes)
    with STATE_LOCK:
        registered = set(STATE["worker_registered"])
    return counts == {"actor": ACTOR_WORKER_COUNT, "rollout": ROLLOUT_WORKER_COUNT} and registered == set(
        EXPECTED_WORKERS
    )


class StatusHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def respond(self, code, value):
        data = json.dumps(value).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        with STATE_LOCK:
            self.respond(200, STATE)

    def do_POST(self):
        try:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not isinstance(payload, dict):
                raise ValueError("worker status must be a JSON object")
        except (ValueError, TypeError, KeyError):
            self.respond(400, {"error": "invalid worker status payload"})
            return
        code, state = record_worker_update(self.path, payload)
        self.respond(code, state)


def ray_port_flags():
    """Validate fixed ports and keep Ray workers outside the control ports."""

    def port(name, default=None, allow_random=False):
        raw = os.environ.get(name, str(default) if default is not None else None)
        if raw is None:
            return None
        if not str(raw).isdigit():
            raise ValueError(f"{name} must be a numeric port, got {raw!r}")
        value = int(raw)
        if not (0 if allow_random else 1) <= value <= 65535:
            raise ValueError(f"{name} must be between {0 if allow_random else 1} and 65535")
        return value

    minimum = port("RAY_MIN_WORKER_PORT", 22000)
    maximum = port("RAY_MAX_WORKER_PORT", 22999)
    if minimum > maximum:
        raise ValueError("RAY_MIN_WORKER_PORT must not exceed RAY_MAX_WORKER_PORT")
    fixed = {
        "RAY_GCS_PORT": GCS_PORT,
        "RAY_DASHBOARD_PORT": DASHBOARD_PORT,
        "STATUS_PORT": STATUS_PORT,
        "RAY_CLIENT_PORT": CLIENT_PORT,
    }
    # The dashboard agent otherwise uses a fixed 52365 port on every node.
    # Zero lets Ray select a free port, which also supports shared CPU hosts.
    agents = [
        ("RAY_DASHBOARD_AGENT_LISTEN_PORT", "dashboard-agent-listen-port", 0),
        ("RAY_DASHBOARD_AGENT_GRPC_PORT", "dashboard-agent-grpc-port", None),
        ("RAY_RUNTIME_ENV_AGENT_PORT", "runtime-env-agent-port", None),
        ("RAY_METRICS_EXPORT_PORT", "metrics-export-port", None),
    ]
    flags = [f"--min-worker-port={minimum}", f"--max-worker-port={maximum}"]
    for name, option, default in agents:
        value = port(name, default, allow_random=True)
        if value is not None:
            flags.append(f"--{option}={value}")
            if value:
                fixed[name] = value
    occupied = {}
    for name, value in fixed.items():
        if not 1 <= value <= 65535:
            raise ValueError(f"{name} must be between 1 and 65535")
        if minimum <= value <= maximum:
            raise ValueError(f"Ray worker range {minimum}..{maximum} overlaps {name}={value}")
        if value in occupied:
            raise ValueError(f"{name}={value} duplicates {occupied[value]}")
        occupied[value] = name
    return flags


def start_ray(head=False):
    isolate_v6e_topology()
    port_flags = ray_port_flags()
    cpus = os.environ.get("RAY_NUM_CPUS", "10" if head else "48")
    memory = os.environ.get("RAY_OBJECT_STORE_BYTES", str((4 if head else 8) * 1024**3))
    args = [
        "ray",
        "start",
        f"--num-cpus={cpus}",
        f"--object-store-memory={memory}",
        "--disable-usage-stats",
        *port_flags,
    ]
    if os.environ.get("POD_IP"):
        args += [f"--node-ip-address={os.environ['POD_IP']}"]
    if head:
        args += [
            "--head",
            f"--port={GCS_PORT}",
            "--dashboard-host=0.0.0.0",
            f"--dashboard-port={DASHBOARD_PORT}",
            f"--ray-client-server-port={CLIENT_PORT}",
        ]
    else:
        group = 0 if ROLE == "actor" else 1
        resources = {"TPU": LOGICAL_TPU_DEVICES, f"tpu-group-{group}": 1}
        args += [f"--address={HEAD_HOST}:{GCS_PORT}", "--resources=" + json.dumps(resources)]
    command(args)


def isolate_v6e_topology():
    """Each VERL role supplies its own TPU mesh, independently of JobSet slices."""
    if "v6e" not in os.environ.get("TPU_ACCELERATOR_TYPE", "").lower():
        return
    names = sorted(name for name in os.environ if name.startswith("MEGASCALE_"))
    for name in names:
        os.environ.pop(name, None)
    if names:
        log("removed JobSet multislice environment before starting Ray: " + ", ".join(names))


def require_vfio():
    devices = sorted(path for path in glob.glob("/dev/vfio/*") if Path(path).name.isdigit())
    log(f"VFIO preflight: {len(devices)} device nodes; expected {LOGICAL_TPU_DEVICES} logical TPU devices")
    if len(devices) != LOGICAL_TPU_DEVICES:
        accelerator = os.environ.get("TPU_ACCELERATOR_TYPE", "v6e")
        raise RuntimeError(f"Expected {LOGICAL_TPU_DEVICES} VFIO devices on a {accelerator} VM, found {len(devices)}")


def run_worker():
    require_vfio()
    deadline = startup_deadline()
    waiting_since = time.monotonic()
    last_progress = waiting_since - 60
    while not STOP.is_set():
        try:
            status = http_json()
            if status["phase"] in ("running", "starting"):
                break
            code = status.get("exit_code")
            return int(code) if code is not None else 1
        except (URLError, OSError):
            if startup_expired(deadline):
                raise RuntimeError("Ray head did not become reachable") from None
            now = time.monotonic()
            if now - last_progress >= 60:
                limit = "unlimited startup wait" if deadline is None else f"startup limit {START_TIMEOUT}s"
                log(f"waiting for Ray head ({int(now - waiting_since)}s elapsed; {limit})")
                last_progress = now
            time.sleep(3)
    if STOP.is_set():
        return 143
    start_ray()
    last_head_seen = time.monotonic()
    import ray

    ray.init(address=f"{HEAD_HOST}:{GCS_PORT}", ignore_reinit_error=True)
    try:
        http_json(
            "/register", {"run_id": RUN_ID, "role": ROLE, "worker_id": WORKER_ID, "node_ip": os.environ.get("POD_IP")}
        )
        while not STOP.is_set():
            try:
                status = http_json()
                last_head_seen = time.monotonic()
                if status["phase"] in ("finishing", "completed"):
                    return int(status["exit_code"])
            except (URLError, OSError):
                if time.monotonic() - last_head_seen > 120:
                    raise RuntimeError("Lost contact with Ray head for 120 seconds") from None
            alive = [node for node in ray.nodes() if node.get("Alive")]
            own_ip = os.environ.get("POD_IP")
            if own_ip and not any(node.get("NodeManagerAddress") == own_ip for node in alive):
                raise RuntimeError("Worker disappeared from Ray cluster")
            time.sleep(5)
        return 143
    finally:
        ray.shutdown()


def runtime_env():
    env = {}
    path = WORKDIR / os.environ.get("RUNTIME_ENV_FILE", "verl/trainer/runtime_env.yaml")
    if path.is_file():
        import yaml

        env = yaml.safe_load(path.read_text()) or {}
    # Every pod already extracts and verifies this same source bundle. Ray's
    # concurrent runtime-env setups can delete/recreate a packaged working_dir
    # while another worker is importing Torch, causing an MKL loader fatal.
    # Keep the pod's stable source directory and absolute PYTHONPATH instead.
    env.pop("working_dir", None)
    vars_ = env.setdefault("env_vars", {})
    keys = [
        "PYTHONPATH",
        "VERL_PLATFORM",
        "VERL_USE_EXTERNAL_MODULES",
        "TENSORBOARD_DIR",
        "VERL_FILE_LOGGER_ROOT",
        "RUN_ID",
        "TPU_ACCELERATOR_TYPE",
        "LOGICAL_TPU_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "WANDB_MODE",
        "WANDB_ENTITY",
        "WANDB_BASE_URL",
    ]
    for key in keys:
        if key in os.environ:
            vars_[key] = os.environ[key]
    return env


def run_head():
    server = ThreadingHTTPServer(("0.0.0.0", STATUS_PORT), StatusHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    start_ray(head=True)
    import ray
    from ray.job_submission import JobSubmissionClient

    ray.init(address=f"127.0.0.1:{GCS_PORT}")
    deadline = startup_deadline()
    waiting_since = time.monotonic()
    last_progress = waiting_since - 60
    while not STOP.is_set():
        nodes = ray.nodes()
        with STATE_LOCK:
            if STATE["worker_failures"]:
                raise RuntimeError(f"Worker preflight failed: {STATE['worker_failures']}")
        if workers_ready(nodes):
            log(
                f"all {len(EXPECTED_WORKERS)} TPU workers registered: "
                f"{ACTOR_WORKER_COUNT} actor and {ROLLOUT_WORKER_COUNT} rollout hosts"
            )
            break
        with STATE_LOCK:
            missing = sorted(set(EXPECTED_WORKERS) - set(STATE["worker_registered"]))
        if startup_expired(deadline):
            raise RuntimeError(
                f"Expected TPU workers did not register; missing={missing}, "
                f"alive role nodes={registered_worker_nodes(nodes)}"
            )
        now = time.monotonic()
        if now - last_progress >= 60:
            limit = "unlimited startup wait" if deadline is None else f"startup limit {START_TIMEOUT}s"
            counts = registered_worker_nodes(nodes)
            log(
                f"waiting for TPU worker registration ({int(now - waiting_since)}s elapsed; {limit}); "
                f"actor hosts {counts['actor']}/{ACTOR_WORKER_COUNT}, "
                f"rollout hosts {counts['rollout']}/{ROLLOUT_WORKER_COUNT}; missing={missing}"
            )
            last_progress = now
        time.sleep(3)
    if STOP.is_set():
        return 143
    ray.shutdown()
    client = JobSubmissionClient(f"http://127.0.0.1:{DASHBOARD_PORT}")
    preflight_file = os.environ.get("TPU_PREFLIGHT_FILE")
    if preflight_file:
        import shlex

        preflight_path = WORKDIR / preflight_file
        if not preflight_path.is_file() or not preflight_path.resolve().is_relative_to(WORKDIR.resolve()):
            raise RuntimeError("TPU preflight must exist inside the isolated source bundle")
        timeout = int(os.environ.get("TPU_PREFLIGHT_TIMEOUT_SECONDS", "900"))
        if timeout <= 0:
            raise ValueError("TPU preflight timeout must be positive")
        proof = artifact_local_root() / "tpu-preflight-result.json"
        proof.unlink(missing_ok=True)
        preflight_env = runtime_env()
        preflight_env.setdefault("env_vars", {})["TPU_PREFLIGHT_RESULT_PATH"] = str(proof)
        preflight_id = client.submit_job(
            entrypoint="cd "
            + shlex.quote(str(WORKDIR))
            + " && python3 "
            + shlex.quote(str(preflight_path))
            + " --timeout "
            + str(max(1, timeout - 45)),
            submission_id=RUN_ID + "-tpu-preflight",
            runtime_env=preflight_env,
        )
        with STATE_LOCK:
            STATE.update(phase="preflight", preflight_job_id=preflight_id)
        log(f"real TPU preflight submitted: {preflight_id}")
        preflight_deadline = time.monotonic() + timeout
        preflight_offset = 0
        while True:
            status = client.get_job_status(preflight_id)
            logs = client.get_job_logs(preflight_id)
            if len(logs) > preflight_offset:
                print(logs[preflight_offset:], end="", flush=True)
                preflight_offset = len(logs)
            with STATE_LOCK:
                failures = dict(STATE["worker_failures"])
            if STOP.is_set() or failures or time.monotonic() >= preflight_deadline:
                client.stop_job(preflight_id)
                raise RuntimeError(f"TPU preflight interrupted or exceeded {timeout}s: {failures}")
            if str(status) in ("SUCCEEDED", "JobStatus.SUCCEEDED"):
                break
            if str(status) in ("FAILED", "STOPPED", "JobStatus.FAILED", "JobStatus.STOPPED"):
                raise RuntimeError(f"Real TPU preflight failed: {status}")
            time.sleep(3)
        if not proof.is_file():
            raise RuntimeError("TPU preflight did not publish its result")
        preflight_result = json.loads(proof.read_text())
        if preflight_result.get("passed") is not True or preflight_result.get("run_id") != RUN_ID:
            raise RuntimeError("TPU preflight did not verify this run")
        log("real TPU preflight passed; starting the requested training job")
        with STATE_LOCK:
            STATE.update(phase="starting", preflight_passed=True)
    entrypoint = WORKDIR / os.environ.get("ENTRYPOINT_FILE", "train.sh")
    if not entrypoint.is_file():
        raise RuntimeError(f"Training entrypoint missing: {entrypoint}")
    with STATE_LOCK:
        failures = dict(STATE["worker_failures"])
    if STOP.is_set() or failures:
        raise RuntimeError(f"Training submission cancelled after startup verification: {failures}")
    # Run from the verified pod-local source instead of a Ray-managed copy.
    import shlex

    job_id = client.submit_job(
        entrypoint="cd " + shlex.quote(str(WORKDIR)) + " && bash " + shlex.quote(str(entrypoint)),
        submission_id=RUN_ID,
        runtime_env=runtime_env(),
    )
    log(f"Ray job submitted: {job_id}")
    with STATE_LOCK:
        STATE.update(phase="running", job_id=job_id)
    result = 1
    log_offset = 0
    while True:
        status = client.get_job_status(job_id)
        try:
            logs = client.get_job_logs(job_id)
            if len(logs) > log_offset:
                print(logs[log_offset:], end="", flush=True)
                log_offset = len(logs)
        except Exception as error:
            log(f"Job log polling failed: {error}")
        with STATE_LOCK:
            failed = bool(STATE["worker_failures"])
        if STOP.is_set() or failed:
            client.stop_job(job_id)
            result = 143 if STOP.is_set() else 1
            break
        if str(status) in ("SUCCEEDED", "JobStatus.SUCCEEDED"):
            result = 0
            break
        if str(status) in ("FAILED", "STOPPED", "JobStatus.FAILED", "JobStatus.STOPPED"):
            result = 1
            break
        time.sleep(5)
    log(f"Ray job terminal status: {status}")
    try:
        print(client.get_job_logs(job_id)[log_offset:], flush=True)
    except Exception as error:
        log(f"Could not retrieve final Ray job log: {error}")
    return result


def main():
    global ARTIFACT_MANAGER
    if ROLE not in ("head", "actor", "rollout"):
        raise ValueError("ROLE must be head, actor, or rollout")
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    exit_code = 1
    initialize_logging()
    try:
        initialize_source()
        initialize_data()
        initialize_artifacts()
        ARTIFACT_MANAGER = ArtifactManager()
        ARTIFACT_MANAGER.verify_write()
        ARTIFACT_MANAGER.write_metadata("starting")
        ARTIFACT_MANAGER.flush(strict=True)
        ARTIFACT_MANAGER.start()
        exit_code = run_head() if ROLE == "head" else run_worker()
    except Exception as error:
        log(f"FAILED: {type(error).__name__}: {error}")
        if ROLE != "head":
            try:
                http_json("/failure", {"run_id": RUN_ID, "role": ROLE, "worker_id": WORKER_ID, "message": str(error)})
            except Exception:
                pass
    finally:
        if ROLE == "head":
            with STATE_LOCK:
                # Let workers stop Ray and finish backups while the head's
                # HTTP coordination process remains alive to receive their acks.
                STATE.update(phase="finishing", exit_code=exit_code)
        if ROLE == "head":
            deadline = time.monotonic() + int(os.environ.get("WORKER_ACK_TIMEOUT_SECONDS", "90"))
            while time.monotonic() < deadline:
                with STATE_LOCK:
                    if set(STATE["worker_done"]) == set(EXPECTED_WORKERS):
                        break
                time.sleep(2)
            with STATE_LOCK:
                if STATE["worker_failures"]:
                    exit_code = exit_code or 1
                if set(STATE["worker_done"]) != set(EXPECTED_WORKERS):
                    missing = sorted(set(EXPECTED_WORKERS) - set(STATE["worker_done"]))
                    log(f"Worker artifact acknowledgments incomplete at cleanup deadline: {missing}")
                    exit_code = exit_code or 1
                STATE["exit_code"] = exit_code
        try:
            close_ray_driver()
        except Exception as error:
            log(f"Ray driver cleanup failed: {error}")
            exit_code = exit_code or 1
        try:
            subprocess.run(["ray", "stop", "--force"], timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            log(f"Ray cleanup failed: {error}")
            exit_code = exit_code or 1
        try:
            if ARTIFACT_MANAGER is not None:
                ARTIFACT_MANAGER.finish(exit_code)
            else:
                # Preserve early-failure diagnostics if a destination is usable.
                manager = ArtifactManager()
                manager.finish(exit_code)
        except Exception as error:
            log(f"Artifact retention failed: {error}")
            exit_code = exit_code or 1
        if ROLE == "head":
            with STATE_LOCK:
                STATE.update(phase="completed", exit_code=exit_code)
        else:
            try:
                http_json("/done", {"run_id": RUN_ID, "role": ROLE, "worker_id": WORKER_ID, "exit_code": exit_code})
            except Exception:
                pass
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
'''
    ),
    "launcher/build_manifest.py": (
        r'''#!/usr/bin/env python3
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Render the namespace-scoped Ray JobSet using the prepared run artifacts."""

import argparse
import json
from pathlib import Path

import yaml


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "next-run.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    name, run_id, prefix = (config[key] for key in ("name", "run_id", "prefix"))
    source_snapshot = config.get("source_snapshot", "source-snapshot")
    sizing = config.get("resources", {})
    hosts = int(sizing.get("hosts_per_role", 1))
    physical_chips = int(sizing.get("physical_chips_per_host", 4))
    logical_devices = int(sizing.get("logical_devices_per_host", 8))
    accelerator = sizing.get("accelerator", "tpu-v6e-slice")
    runtime_accelerator = sizing.get("runtime_accelerator", "v6e")
    namespace = config.get("namespace", "default")
    if hosts < 1 or physical_chips < 1 or logical_devices < 1:
        raise ValueError("Worker host and TPU counts must be positive")
    if accelerator == "tpu7x" and (physical_chips, logical_devices) != (4, 8):
        raise ValueError("TPU7x requires four physical chips/eight logical devices per worker")
    if runtime_accelerator == "v6e" and (physical_chips != logical_devices or physical_chips not in (4, 8)):
        raise ValueError("V6e requires one logical device per physical chip on four- or eight-chip hosts")
    active_deadline = int(sizing.get("active_deadline_seconds", 172800))
    if active_deadline < 0:
        raise ValueError("The active deadline must be nonnegative; zero disables it")
    ports = {"gcs": 16379, "dashboard": 18265, "status": 18766}
    labels = {"app.kubernetes.io/name": "verl-kueue-ray", "verl-run": name}
    environment = {
        "RUN_ID": run_id,
        "HEAD_HOST": name + "-head." + namespace + ".svc.cluster.local",
        "CODE_BUNDLE_URI": prefix + "/source-snapshot.tar.gz",
        "CODE_BUNDLE_SHA256": (root / (source_snapshot + ".tar.gz.sha256")).read_text().strip(),
        "DATA_MANIFEST_URI": config.get("data_manifest_uri", prefix + "/data-manifest.json"),
        "ARTIFACT_GCS_URI": config.get("artifact_gcs_uri", "gs://ubench-logs/lixali/verl-ray-kueue/results"),
        "ARTIFACT_LAYOUT": config.get("artifact_layout", "{run_id}/backup-auto/pods/{role}"),
        "ARTIFACT_UPLOAD_INTERVAL_SECONDS": str(config.get("artifact_upload_interval_seconds", 60)),
        "ACTOR_WORKER_COUNT": str(hosts),
        "ROLLOUT_WORKER_COUNT": str(hosts),
        "VERL_FILE_LOGGER_ROOT": "/tmp/verl_metrics/" + run_id,
        "ENTRYPOINT_FILE": "train-kueue.sh",
        "RUNTIME_ENV_FILE": "runtime-env-kueue.yaml",
        "TPU_ACCELERATOR_TYPE": runtime_accelerator,
        "LOGICAL_TPU_DEVICES": str(logical_devices),
        "TORCHTITAN_DIR": "/torchtitan",
        "PYTHONUNBUFFERED": "1",
        "VERL_PLATFORM": "tpu",
        "VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin",
        "RAY_memory_monitor_refresh_ms": "0",
        "RAY_memory_usage_threshold": "0.99",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
        "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
        "SMOKE_TEST": "0",
        "RAY_GCS_PORT": str(ports["gcs"]),
        "RAY_DASHBOARD_PORT": str(ports["dashboard"]),
        "STATUS_PORT": str(ports["status"]),
        "RAY_CLIENT_PORT": "20001",
        "RAY_MIN_WORKER_PORT": "22000",
        "RAY_MAX_WORKER_PORT": "22999",
        "RAY_DASHBOARD_AGENT_LISTEN_PORT": "0",
        "START_TIMEOUT_SECONDS": str(sizing.get("start_timeout_seconds", 900)),
        "WORKER_ACK_TIMEOUT_SECONDS": str(sizing.get("worker_ack_timeout_seconds", 900)),
    }
    wandb_settings = config.get("wandb", {})
    wandb_mode = wandb_settings.get("mode", "online")
    environment["WANDB_MODE"] = wandb_mode
    environment["WANDB_DISABLED"] = "false"
    for field, variable in (("entity", "WANDB_ENTITY"), ("base_url", "WANDB_BASE_URL")):
        if wandb_settings.get(field):
            environment[variable] = str(wandb_settings[field])
    if runtime_accelerator == "v6e":
        # GKE treats JobSet replicated jobs as MegaScale slices by default.
        # These roles are one trainer slice and independent sampler hosts.
        # Explicit empty values prevent injection of a three-slice rendezvous.
        environment.update({"MEGASCALE_NUM_SLICES": "", "MEGASCALE_SLICE_ID": "", "MEGASCALE_COORDINATOR_ADDRESS": ""})
    preflight = config.get("tpu_preflight", {})
    if preflight.get("enabled"):
        environment["TPU_PREFLIGHT_FILE"] = preflight.get("file", "tpu-preflight.py")
        environment["TPU_PREFLIGHT_TIMEOUT_SECONDS"] = str(preflight.get("timeout_seconds", 900))
    documents = [
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": name + "-bootstrap", "namespace": namespace, "labels": labels},
            "data": {"bootstrap.py": (root / "bootstrap.py").read_text()},
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": name + "-head", "namespace": namespace, "labels": labels},
            "spec": {
                "selector": dict(labels, **{"ray-role": "head"}),
                "ports": [{"name": key, "port": value, "targetPort": value} for key, value in ports.items()],
            },
        },
    ]
    replicated_jobs = []
    for role in ("head", "actor", "rollout"):
        is_head = role == "head"
        cpu, memory = (
            (str(sizing.get("head_cpu", "12")), sizing.get("head_memory", "48Gi"))
            if is_head
            else (str(sizing.get("worker_cpu", "48")), sizing.get("worker_memory", "400Gi"))
        )
        storage_role = "head" if is_head else "worker"
        storage_request = sizing.get(
            storage_role + "_ephemeral_storage_request", sizing.get("ephemeral_storage_request", "20Gi")
        )
        storage_limit = sizing.get(
            storage_role + "_ephemeral_storage_limit", sizing.get("ephemeral_storage_limit", storage_request)
        )
        resources = {"cpu": cpu, "memory": memory, "ephemeral-storage": storage_request}
        if not is_head:
            resources["google.com/tpu"] = str(physical_chips)
        limits = dict(resources, **{"ephemeral-storage": storage_limit})
        limits["memory"] = sizing.get("head_memory_limit" if is_head else "worker_memory_limit", memory)
        env = dict(
            environment, ROLE=role, RAY_NUM_CPUS=cpu, RAY_OBJECT_STORE_BYTES=str((8 if is_head else 32) * 1024**3)
        )
        container = {
            "name": "ray-" + role,
            "image": config["image"],
            "imagePullPolicy": "IfNotPresent",
            "command": ["python3", "-u", "/bootstrap/bootstrap.py"],
            "env": [{"name": key, "value": value} for key, value in env.items()]
            + [{"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}}]
            + (
                [{"name": "WORKER_INDEX", "value": "0"}]
                if is_head or hosts == 1
                else [
                    {
                        "name": "WORKER_INDEX",
                        "valueFrom": {
                            "fieldRef": {
                                "fieldPath": "metadata.annotations['batch.kubernetes.io/job-completion-index']"
                            }
                        },
                    }
                ]
            ),
            "resources": {"requests": dict(resources), "limits": limits},
            "volumeMounts": [
                {"name": "bootstrap", "mountPath": "/bootstrap", "readOnly": True},
                {"name": "tmp", "mountPath": "/tmp"},
                {"name": "shm", "mountPath": "/dev/shm"},
                {"name": "run", "mountPath": "/opt/run"},
                {"name": "data", "mountPath": "/data"},
            ],
        }
        if "wandb" in config.get("training", {}).get("logger", ["wandb"]) and wandb_mode == "online":
            # Credentials stay in Kubernetes, outside source/config archives and Ray's printed env_vars.
            container["env"].append({
                "name": "WANDB_API_KEY",
                "valueFrom": {"secretKeyRef": {
                    "name": wandb_settings.get("api_key_secret", "verl-wandb-lixali"),
                    "key": wandb_settings.get("api_key_secret_key", "api-key"),
                }},
            })
        selector = (
            sizing.get("head_node_selector", {"cloud.google.com/gke-nodepool": "cpu-np"})
            if is_head
            else {
                "cloud.google.com/gke-tpu-accelerator": accelerator,
                "cloud.google.com/gke-tpu-topology": sizing.get(
                    "actor_topology" if role == "actor" else "rollout_topology", "2x2x1"
                ),
            }
        )
        if role == "actor" and hosts > 1:
            policy = sizing.get(
                "actor_placement_policy", "tpu7x-64-2x4x4-placement-policy" if accelerator == "tpu7x" else None
            )
            if policy:
                selector["cloud.google.com/placement-policy-name"] = policy
        pod_spec = {
            "restartPolicy": "Never",
            "serviceAccountName": config.get("service_account", "gutianyu-ksa"),
            "hostNetwork": True,
            "dnsPolicy": "ClusterFirstWithHostNet",
            "priorityClassName": config.get("priority_class", "medium"),
            "nodeSelector": selector,
            "terminationGracePeriodSeconds": 180,
            "containers": [container],
            "volumes": [
                {"name": "bootstrap", "configMap": {"name": name + "-bootstrap"}},
                {"name": "tmp", "emptyDir": {}},
                {"name": "run", "emptyDir": {}},
                {"name": "data", "emptyDir": {}},
                {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi" if is_head else "64Gi"}},
            ],
        }
        if not is_head:
            pod_spec["tolerations"] = [{"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"}]
        if role == "actor" and hosts > 1 and not config.get("topology_aware_scheduling", True):
            pod_spec["affinity"] = {
                "podAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": [
                        {
                            "labelSelector": {"matchLabels": dict(labels, **{"ray-role": "actor"})},
                            "topologyKey": "cloud.google.com/gke-nodepool",
                        }
                    ]
                }
            }
        pod_metadata = {"labels": dict(labels, **{"ray-role": role})}
        if not is_head and config.get("topology_aware_scheduling", True):
            # This cluster injects the main unconstrained annotation. A single
            # required eight-pod slice keeps the trainer in one connected pool.
            pod_metadata["annotations"] = (
                {
                    "kueue.x-k8s.io/podset-unconstrained-topology": "true",
                    "kueue.x-k8s.io/podset-slice-required-topology": "cloud.google.com/gke-nodepool",
                    "kueue.x-k8s.io/podset-slice-size": str(hosts),
                    "kueue.x-k8s.io/podset-pod-index-label": "batch.kubernetes.io/job-completion-index",
                }
                if role == "actor" and hosts > 1
                else {"kueue.x-k8s.io/podset-unconstrained-topology": "true"}
            )
        worker_count = 1 if is_head else hosts
        replicated_jobs.append(
            {
                "name": role,
                "replicas": 1,
                "template": {
                    "spec": {
                        "parallelism": worker_count,
                        "completions": worker_count,
                        "backoffLimit": 0,
                        **({"completionMode": "Indexed"} if worker_count > 1 else {}),
                        **({"activeDeadlineSeconds": active_deadline} if active_deadline else {}),
                        "template": {"metadata": pod_metadata, "spec": pod_spec},
                    }
                },
            }
        )
    documents.append(
        {
            "apiVersion": "jobset.x-k8s.io/v1alpha2",
            "kind": "JobSet",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": dict(labels, **{"kueue.x-k8s.io/queue-name": config.get("queue", "multislice-queue")}),
            },
            "spec": {
                "suspend": True,
                "failurePolicy": {"maxRestarts": 0},
                "successPolicy": {"operator": "All", "targetReplicatedJobs": ["head", "actor", "rollout"]},
                "replicatedJobs": replicated_jobs,
            },
        }
    )
    output = (root.parent / config.get("manifest_file", "verl-ray-grpo-kueue.yaml")).resolve()
    if not output.is_relative_to(root.parent):
        raise ValueError("The rendered manifest must remain inside the recipe directory")
    output.write_text(
        "# VERL Ray training admitted atomically by Kueue as a JobSet.\n"
        "# Submit with: python3 launcher/submit_with_backup.py --config "
        f"{args.config.resolve().relative_to(root.parent)}\n"
        "# This starts the keyless backup mirror to gs://lixali-tpu-storage.\n"
        f"# Trainer: {hosts * physical_chips} physical {runtime_accelerator} chips; "
        f"sampler: {hosts * physical_chips} chips.\n"
        + (
            "# TPU provisioning wait has no startup or Kubernetes Job deadline.\n"
            if not active_deadline and int(sizing.get("start_timeout_seconds", 900)) == 0
            else ""
        )
        + "# Original VERL repository is read-only; this run uses an isolated source snapshot.\n"
        + yaml.safe_dump_all(documents, sort_keys=False)
    )
    print(output)


if __name__ == "__main__":
    main()
'''
    ),
    "launcher/runtime-env.yaml": (
        r"""excludes:
  - .git
  - .codex
  - .venv
  - venv
  - logs
  - tensorboard_log
  - checkpoints
  - '*.log'
  - '*.pt'
  - '*.bin'
  - __pycache__
env_vars:
  PYTHONPATH: '.'
  PYTHONUNBUFFERED: '1'
  VERL_PLATFORM: tpu
  VERL_USE_EXTERNAL_MODULES: verl_hardware_plugin
  RAY_memory_monitor_refresh_ms: '0'
  RAY_memory_usage_threshold: '0.99'
  RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS: '1'
  RAY_OVERRIDE_JOB_RUNTIME_ENV: '1'
  SMOKE_TEST: '0'
  TPU_ACCELERATOR_TYPE: v6e
  VERL_FILE_LOGGER_ROOT: /tmp/verl_metrics
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""
    ),
    "launcher/submit_with_backup.py": (
        r'''#!/usr/bin/env python3
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Prepare a Ray JobSet and start verified, keyless backups before submission."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from artifact_mirror import CloudStorage

ROOT = Path(__file__).resolve().parent
CONTEXT = "gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap"
REPREPARE = "Rerun submit_qwen3_4b_base_250steps.sh to prepare a fresh core and hardware-plugin snapshot."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "next-run.json")
    parser.add_argument(
        "--prepare-only", action="store_true", help="Verify storage and stage inputs without submitting training"
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Unsupported here; rerun the outer recipe launcher instead"
    )
    parser.add_argument(
        "--wait", action="store_true", help="Wait for the backup mirror to finish its final verification"
    )
    args = parser.parse_args()
    if args.fresh:
        raise SystemExit(REPREPARE)
    config_path = args.config.resolve()
    if not config_path.is_relative_to(ROOT.parent):
        raise SystemExit("Run configuration must be inside this submission directory")
    cfg = json.loads(config_path.read_text())
    staged_source = (ROOT / cfg["source_snapshot"]).resolve()
    if not staged_source.is_relative_to(ROOT):
        raise SystemExit("Prepared source snapshot must be inside this launcher's submission directory")
    for name in (
        "submission-source.json",
        "run-training.json",
        "verl/trainer/main_ppo.py",
        "verl_hardware_plugin/__init__.py",
    ):
        if not (staged_source / name).is_file():
            raise SystemExit(f"Prepared source snapshot lacks {name}. {REPREPARE}")
    packaged_config = json.loads((staged_source / "run-training.json").read_text())
    for key in (
        "training",
        "wandb",
        "resources",
        "geometry",
        "actor_memory",
        "tpu_preflight",
        "training_script",
        "required_source_files",
    ):
        if packaged_config.get(key) != cfg.get(key):
            raise SystemExit(f"Packaged {key} differs from the run configuration. {REPREPARE}")
    context = cfg.get("context", CONTEXT)
    kubectl = ["kubectl", "--context", context, "--namespace", cfg.get("namespace", "default")]
    if cfg.get("kubeconfig"):
        kubeconfig = Path(cfg["kubeconfig"]).resolve()
        if not kubeconfig.is_file():
            raise SystemExit("Run kubeconfig must exist")
        kubectl += ["--kubeconfig", str(kubeconfig)]
    existing = subprocess.run(kubectl + ["get", "jobset", cfg["name"], "-o", "json"], capture_output=True, text=True)
    if existing.returncode == 0:
        raise SystemExit(f"This JobSet already exists. {REPREPARE}")
    if "NotFound" not in existing.stderr:
        raise SystemExit("Cannot check the target cluster: " + existing.stderr.strip())

    cloud = CloudStorage()
    target = cfg["backup_gcs_uri"].rstrip("/") + "/" + cfg["run_id"] + "/backup-auto"
    cloud.preflight(target + "/control")
    print("Private bucket write/read access verified: " + target, flush=True)
    snapshot = ROOT / (cfg["source_snapshot"] + ".tar.gz")
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if digest != (ROOT / (cfg["source_snapshot"] + ".tar.gz.sha256")).read_text().strip():
        raise SystemExit("Local source archive SHA-256 mismatch")
    # The original repository is never imported or written by this helper.
    cloud.put_file(snapshot, cfg["prefix"] + "/source-snapshot.tar.gz")
    cloud.put_file(ROOT / (cfg["source_snapshot"] + ".tar.gz.sha256"), cfg["prefix"] + "/source-snapshot.tar.gz.sha256")
    cloud.copy_uri(cfg["prefix"] + "/source-snapshot.tar.gz", target + "/control/source-snapshot.tar.gz")
    manifest = cloud.get_json(cfg["data_manifest_uri"])
    archived_inputs = []
    for entry in manifest:
        relative = Path(entry["destination"]).relative_to("/data").as_posix()
        destination = target + "/inputs/" + relative
        source_object_metadata = cloud.stat(entry["source"], generation=entry.get("generation"))
        metadata = cloud.copy(entry["source"], destination, expected=source_object_metadata)
        print("Archived input: " + relative + " (" + metadata["size"] + " bytes)", flush=True)
        archived_inputs.append(
            {
                "source": entry["source"],
                "destination": destination,
                "size": metadata["size"],
                "crc32c": metadata.get("crc32c"),
                "generation": metadata["generation"],
            }
        )
    cloud.uploadbytes(
        json.dumps(archived_inputs, indent=2).encode(),
        target + "/control/archived-inputs.json",
        content_type="application/json",
    )
    subprocess.run([sys.executable, str(ROOT / "build_manifest.py"), "--config", str(config_path)], check=True)
    rendered = ROOT.parent / cfg.get("manifest_file", "verl-ray-grpo-kueue.yaml")
    if not rendered.resolve().is_relative_to(ROOT.parent):
        raise SystemExit("Rendered manifest must be inside this submission directory")
    subprocess.run(kubectl + ["apply", "--dry-run=server", "-f", str(rendered)], check=True)
    control_files = [config_path, rendered] + [
        ROOT / name
        for name in (
            "bootstrap.py",
            "train.sh",
            "runtime-env.yaml",
            "build_manifest.py",
            "artifact_mirror.py",
            "submit_with_backup.py",
        )
    ]
    if cfg.get("tpu_preflight", {}).get("enabled"):
        control_files.append(ROOT / "tpu_preflight.py")
    for path in control_files:
        cloud.put_file(path, target + "/control/" + path.name)
    proof = {
        "run_id": cfg["run_id"],
        "jobset": cfg["name"],
        "backup_uri": target,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "source_sha256": digest,
        "archived_inputs": len(archived_inputs),
    }
    cloud.uploadbytes(
        json.dumps(proof, indent=2).encode(), target + "/control/prepared.json", content_type="application/json"
    )
    print("Storage verified; source, launcher and model/data inputs archived in " + target, flush=True)
    if args.prepare_only:
        print("Prepared only; no training job was submitted.")
        return

    state_dir = ROOT / "backup-state" / cfg["name"]
    state_dir.mkdir(parents=True, exist_ok=True)
    log_path = state_dir / "mirror.log"
    with log_path.open("ab") as output:
        mirror = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(ROOT / "artifact_mirror.py"),
                "--config",
                str(config_path),
                "--follow",
                "--state-dir",
                str(state_dir),
            ],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=ROOT,
        )
    (state_dir / "mirror.pid").write_text(str(mirror.pid) + "\n")
    deadline = time.monotonic() + 60
    ready = state_dir / "ready.json"
    while not ready.exists():
        if mirror.poll() is not None:
            raise SystemExit("Backup mirror failed before submission; see " + str(log_path))
        if time.monotonic() >= deadline:
            mirror.terminate()
            raise SystemExit("Backup mirror did not become ready; see " + str(log_path))
        time.sleep(1)
    try:
        subprocess.run(kubectl + ["apply", "-f", str(rendered)], check=True)
    except BaseException:
        os.killpg(mirror.pid, signal.SIGTERM)
        raise
    proof["submitted"] = True
    cloud.uploadbytes(
        json.dumps(proof, indent=2).encode(), target + "/control/prepared.json", content_type="application/json"
    )
    print("Submitted " + cfg["name"] + "; automatic backup log: " + str(log_path), flush=True)
    if args.wait:
        exit_code = mirror.wait()
        if exit_code == 0:
            complete = json.loads((state_dir / "complete.json").read_text())
            conditions = complete.get("jobset", {}).get("status", {}).get("conditions", [])
            if any(item.get("type") == "Failed" and item.get("status") == "True" for item in conditions):
                exit_code = 1
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
'''
    ),
    "launcher/tpu_preflight.py": (
        r'''#!/usr/bin/env python3
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Gate training on real TPU collectives and compiled backward operations.

Run inside the isolated VERL bundle, after all TPU Ray nodes have registered.
The trainer uses the staged PlatformTPU environment for all 32 ranks. Samplers
then use either eight four-rank groups or 32 independent one-chip groups,
matching the configured rollout tensor parallelism. No model, training step,
or successful hardware result is simulated by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

os.environ["VERL_USE_EXTERNAL_MODULES"] = "verl_hardware_plugin"


def normalize_required_source_files(requirements):
    if not isinstance(requirements, dict):
        raise ValueError("required_source_files must be a mapping")
    return {
        name: requirement["sha256"] if isinstance(requirement, dict) else requirement
        for name, requirement in requirements.items()
    }


def verify_required_sampler_patch(required_source_files):
    """Verify submitted source and normal vLLM patch activation, without sampling.

    Ray actors inherit the job's bundled working directory. Requiring the actual
    imported module to match that file prevents an image-installed copy from
    satisfying the gate. Configurations without source requirements retain the
    existing hardware-only preflight.
    """
    if not required_source_files:
        return None

    import hashlib
    import importlib
    import inspect
    import re

    if not isinstance(required_source_files, dict):
        raise ValueError("required_source_files must map bundled relative paths to SHA256 values")
    from recipe_tpu_plugin_layout import tpu_plugin_layout

    source_root = Path.cwd().resolve()
    layout = tpu_plugin_layout(source_root)
    entrypoint_relative = layout["patches"]
    separated_relative = layout["precision"]
    patch_relative = separated_relative if (source_root / separated_relative).is_file() else entrypoint_relative
    if not {patch_relative, entrypoint_relative}.issubset(required_source_files):
        raise ValueError("The sampler gate requires SHA256s for the submitted TPU precision source and entrypoint")
    checked = []
    for relative, expected in required_source_files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Required source must be a bundled relative path: {relative}")
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None:
            raise ValueError(f"Invalid required source SHA256 for {relative}")
        source = (source_root / relative_path).resolve(strict=True)
        if not source.is_relative_to(source_root):
            raise RuntimeError(f"Required source resolves outside the submitted bundle: {source}")
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != expected.lower():
            raise RuntimeError(f"Required sampler source SHA256 mismatch for {relative}: {actual} != {expected}")
        checked.append({"relative_path": relative, "file": str(source), "sha256": actual})

    # TPUvLLMHttpServer installs this entrypoint before constructing the engine;
    # apply the same entrypoint here without constructing a model or sampling.
    entrypoint = importlib.import_module(layout["patches_module"])
    entrypoint_file = Path(entrypoint.__file__).resolve(strict=True)
    if entrypoint_file != (source_root / entrypoint_relative).resolve(strict=True):
        raise RuntimeError(f"TPU patch entrypoint imported outside the submitted bundle: {entrypoint_file}")
    entrypoint.patch_vllm_for_tpu()
    if entrypoint._PATCHES_APPLIED is not True:
        raise RuntimeError("The submitted patch_vllm_for_tpu entrypoint did not activate")
    patch_module = (
        importlib.import_module(layout["precision_module"])
        if patch_relative == separated_relative
        else entrypoint
    )
    patch_file = Path(patch_module.__file__).resolve(strict=True)
    expected_file = (source_root / patch_relative).resolve(strict=True)
    if patch_file != expected_file:
        raise RuntimeError(f"Sampler patch imported from {patch_file}, expected submitted source {expected_file}")
    patch_sha = hashlib.sha256(patch_file.read_bytes()).hexdigest()
    if patch_sha != required_source_files[patch_relative].lower():
        raise RuntimeError("Imported sampler patch changed after the bundled source hash check")

    runner_module = importlib.import_module("vllm_torchtpu.runner.tpu_runner")
    runner = runner_module.TPUModelRunner
    flag = getattr(runner, "_verl_fp32_sampler_patched", False)
    if flag is not True:
        raise RuntimeError(
            "Submitted sampler patch did not activate: TPUModelRunner._verl_fp32_sampler_patched is not True"
        )
    signature = inspect.signature(runner.sample_from_logits, follow_wrapped=False)
    expected_parameters = ["self", "logits", "temperatures", "u", "top_k", "top_p", "all_greedy"]
    if list(signature.parameters) != expected_parameters:
        raise RuntimeError(f"Activated sampler wrapper has unexpected parameters: {signature}")
    wrapper_file = Path(inspect.getsourcefile(runner.sample_from_logits)).resolve(strict=True)
    if wrapper_file != expected_file:
        raise RuntimeError(f"Activated sampler wrapper comes from {wrapper_file}, expected {expected_file}")
    logprobs_flag = getattr(runner, "_verl_fp32_logprobs_patched", False)
    if logprobs_flag is not True:
        raise RuntimeError("The submitted TPU FP32 log-probability patch did not activate")
    return {
        "patchfile": str(patch_file),
        "sha256": patch_sha,
        "flag": flag,
        "entrypoint_file": str(entrypoint_file),
        "sampler_wrapper_file": str(wrapper_file),
        "logprobs_flag": logprobs_flag,
        "signature": str(signature),
        "runner_module_file": str(runner_module.__file__),
        "source_root": str(source_root),
        "source_files": checked,
        "activation": layout["patches_module"] + ".patch_vllm_for_tpu()",
    }


class ResourcePool:
    def __init__(self, groups, start=None):
        self.groups = groups
        if start is not None:
            self.start_bundle_index = start

    def get_placement_groups(self, **_):
        return self.groups


def singleton_sampler_environment(env, accelerator_ids):
    """Scope a TP1 native program to its actual Ray-assigned physical chip."""
    assigned = accelerator_ids.get("TPU", [])
    if len(assigned) != 1:
        raise RuntimeError(f"TP1 sampler probe requires exactly one assigned TPU chip, got {assigned}")
    actual_chip = int(assigned[0])
    host_devices = int(env.get("TPU_PHYSICAL_DEVICES_PER_HOST", "4"))
    if not 0 <= actual_chip < host_devices:
        raise RuntimeError(f"TP1 sampler received an invalid physical chip ID: {actual_chip}")
    output = dict(env)
    planned_chip = int(output["TPU_VISIBLE_CHIPS"])
    endpoints = output["TORCH_TPU_SLICEBUILDER_ADDRESSES"].split(",")
    if len(endpoints) != 1:
        raise RuntimeError("TP1 sampler requires exactly one native SliceBuilder endpoint")
    native_host, hinted_port = endpoints[0].rsplit(":", 1)
    base_port = int(hinted_port) - planned_chip
    native_port = base_port + actual_chip
    master_port = 29651 + actual_chip
    output.update(
        {
            "TPU_VISIBLE_CHIPS": str(actual_chip),
            "TPU_VISIBLE_DEVICES": str(actual_chip),
            "TPU_PROCESS_PORT": str(native_port),
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": f"{native_host}:{native_port}",
            "TPU_PROCESS_ADDRESSES": f"{native_host}:{native_port}",
            "MASTER_PORT": str(master_port),
            "DIST_INIT_METHOD": f"tcp://{output['MASTER_ADDR']}:{master_port}",
        }
    )
    return output, planned_chip


class TPUProbe:
    def run(self, env, role, replica, rank, world_size, init_timeout, required_source_files=None):
        import os
        import time

        started = time.monotonic()
        inherited = sorted(key for key in os.environ if key.startswith("MEGASCALE_"))
        for key in inherited:
            os.environ.pop(key, None)
        assigned_tpu_ids = None
        planned_chip_slot = None
        if role == "sampler" and world_size == 1:
            # Read Ray's allocation while chip visibility still covers the host.
            # The native selector must be installed before the workaround's
            # Torch import can autoload its TPU backend. Native one-rank programs
            # otherwise all select device zero, regardless of TPU_VISIBLE_CHIPS.
            import ray

            assigned_tpu_ids = ray.get_runtime_context().get_accelerator_ids()
            env, planned_chip_slot = singleton_sampler_environment(env, assigned_tpu_ids)
            os.environ.update(env)
        # Preserve the proven trainer import order. For TP1 samplers the native
        # selector is already scoped to the actual allocation before this import.
        import importlib
        from recipe_tpu_plugin_layout import tpu_plugin_layout

        importlib.import_module(tpu_plugin_layout()["platform_module"]).patch_ray_worker()
        os.environ.update(env)
        if any(key.startswith("MEGASCALE_") for key in os.environ):
            raise RuntimeError("GKE Multislice settings survived the worker cleanup")
        # Import after installing this rank's endpoint/visible-chip environment.
        import torch
        import torch.distributed as dist
        import torch_tpu  # noqa: F401 -- registers TPU device and collectives

        from verl.plugin.platform import get_platform
        from verl.utils.distributed import initialize_global_process_group_ray

        platform = get_platform()
        if platform.device_name != "tpu":
            raise RuntimeError(f"Hardware probe selected {platform.device_name}, expected TPU")
        initialize_global_process_group_ray(timeout_second=init_timeout)
        try:
            if dist.get_world_size() != world_size or dist.get_rank() != rank:
                raise RuntimeError("The distributed process group has incorrect rank geometry")
            sampler_patch = verify_required_sampler_patch(required_source_files) if role == "sampler" else None
            device = torch.device("tpu:0")
            value = torch.tensor([float(rank)], dtype=torch.float32, device=device)
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            actual = float(value.item())
            expected = float(world_size * (world_size - 1) // 2)
            if actual != expected:
                raise RuntimeError(f"TPU all-reduce returned {actual}, expected {expected}")

            # Exercise the same TorchTPU compilation backend used by training.
            square_sum = torch.compile(lambda data: (data * data).sum(), backend="tpu", fullgraph=True, dynamic=False)
            data = torch.arange(1, 17, dtype=torch.float32, device=device).requires_grad_()
            loss = square_sum(data)
            loss.backward()
            platform.synchronize()
            loss_value = float(loss.detach().item())
            gradient = data.grad.detach().cpu()
            expected_gradient = torch.arange(1, 17, dtype=torch.float32) * 2
            if loss_value != 1496.0 or not torch.equal(gradient, expected_gradient):
                raise RuntimeError("Compiled TPU forward/backward produced an incorrect result")
            dist.barrier()
            import ray

            proof = {
                "role": role,
                "replica": replica,
                "rank": rank,
                "world_size": world_size,
                "pid": os.getpid(),
                "node_id": ray.get_runtime_context().get_node_id(),
                "node_ip": ray.util.get_node_ip_address(),
                "backend": platform.communication_backend_name(),
                "visible_chip": os.environ.get("TPU_VISIBLE_CHIPS"),
                "native_visible_devices": os.environ.get("TPU_VISIBLE_DEVICES"),
                "physical_task_id": os.environ.get("TPU_WORKER_ID"),
                "cloud_task_id": os.environ.get("CLOUD_TPU_TASK_ID"),
                "topology": os.environ.get("TORCH_TPU_TOPOLOGY"),
                "slice_endpoints": os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES"),
                "native_process_port": os.environ.get("TPU_PROCESS_PORT"),
                "master_port": os.environ.get("MASTER_PORT"),
                "megascale_keys": [],
                "removed_megascale_keys": inherited,
                "all_reduce_actual": actual,
                "all_reduce_expected": expected,
                "compiled_loss": loss_value,
                "compiled_backward_passed": True,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            if sampler_patch is not None:
                proof["sampler_patch"] = sampler_patch
            if assigned_tpu_ids is not None:
                proof["assigned_tpu_ids"] = assigned_tpu_ids.get("TPU", [])
                proof["planned_chip_slot"] = planned_chip_slot
            return proof
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("run-training.json"))
    # Leave enough time for actor/placement-group cleanup and proof upload
    # before the bootstrap gate reaches its own hard deadline.
    gate_timeout = int(os.environ.get("TPU_PREFLIGHT_TIMEOUT_SECONDS", "1800"))
    parser.add_argument("--timeout", type=int, default=max(1, gate_timeout - 60))
    args = parser.parse_args()
    if args.timeout <= 0:
        raise ValueError("Hardware preflight timeout must be positive")
    config = json.loads(args.config.read_text())
    required_source_files = normalize_required_source_files(config.get("required_source_files", {}))
    sizing = config["resources"]
    hosts = int(sizing["hosts_per_role"])
    devices = int(sizing["logical_devices_per_host"])
    if hosts != 8 or devices != 4 or int(sizing["physical_chips_per_host"]) != 4:
        raise ValueError("This hardware preflight requires eight four-chip hosts per role")
    sampler_tp = int(sizing.get("rollout_tensor_parallel_size", devices))
    if sampler_tp not in (1, devices):
        raise ValueError("Sampler hardware preflight supports TP1 or one full four-chip host")
    sampler_replicas = hosts * devices // sampler_tp
    result_path = Path(os.environ["TPU_PREFLIGHT_RESULT_PATH"])
    if not result_path.is_absolute():
        raise ValueError("Hardware preflight proof requires an absolute result path")
    run_id = os.environ["RUN_ID"]
    deadline = time.monotonic() + args.timeout
    result = {
        "passed": False,
        "run_id": run_id,
        "trainer_world_size": hosts * devices,
        "sampler_replica_count": sampler_replicas,
        "sampler_world_size": sampler_tp,
        "trainer": [],
        "samplers": [],
        "resources_released": False,
    }
    if required_source_files:
        result["required_source_files"] = required_source_files
    actors, groups = [], []
    failure = None
    import ray
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    from verl.plugin.platform import get_platform

    ray.init(address="auto", ignore_reinit_error=True)
    platform = get_platform()
    RemoteProbe = ray.remote(TPUProbe)

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("The TPU hardware preflight exceeded its deadline")
        return seconds

    def host_for(group):
        state = ray.util.placement_group_table(group)
        nodes = set(state["bundles_to_node_id"].values())
        if len(nodes) != 1:
            raise RuntimeError("A TPU host placement group spans multiple nodes")
        node_id = next(iter(nodes))
        node = next(node for node in ray.nodes() if node["Alive"] and node["NodeID"] == node_id)
        if int(node["Resources"].get("TPU", 0)) != devices:
            raise RuntimeError("Placement group is on an incorrectly sized TPU node")
        return node

    def create_role_groups(role, name, group_resource):
        role_groups = []
        for index in range(hosts):
            bundles = []
            for _ in range(devices):
                bundle = {"CPU": 1}
                platform.configure_placement_group_bundle(bundle, True, "TPU", name, group_resource)
                bundles.append(bundle)
            group = ray.util.placement_group(bundles, strategy="STRICT_PACK", name=f"preflight_{role}_{index}_{run_id}")
            groups.append(group)
            role_groups.append(group)
        ray.get([group.ready() for group in role_groups], timeout=min(remaining(), 120))
        nodes = [host_for(group) for group in role_groups]
        if len({node["NodeID"] for node in nodes}) != hosts:
            raise RuntimeError(f"The {role} preflight did not reserve eight distinct hosts")
        return role_groups, nodes

    def spawn_group(
        env_groups, scheduling_groups, role, replica, master, world_size, bundle_offset=0, master_port=None
    ):
        references = []
        local_world_size = min(devices, world_size)
        pool = ResourcePool(env_groups, None if role == "trainer" else replica * world_size)
        if master_port is None:
            master_port = 29650 if role == "trainer" else 29651
        for rank in range(world_size):
            host_index, local_rank = divmod(rank, local_world_size)
            env = platform.get_worker_env_vars(
                pool,
                rank,
                world_size,
                local_rank,
                local_world_size,
                "global_pool" if role == "trainer" else "rollout_pool",
                "TPU",
            )
            env.update(
                {
                    "RANK": str(rank),
                    "WORLD_SIZE": str(world_size),
                    "LOCAL_RANK": str(local_rank),
                    "MASTER_ADDR": master,
                    "MASTER_PORT": str(master_port),
                    "DIST_INIT_METHOD": f"tcp://{master}:{master_port}",
                }
            )
            runtime = dict(platform.get_ray_init_kwargs()["runtime_env"])
            # Ray requires actor runtime_env options to be JSON serializable.
            # Keep host-wide chip visibility during its startup resource lookup;
            # TPUProbe then installs rank-local visibility and the VERL bounds
            # guard before initializing any TPU tensors or collectives.
            runtime.pop("worker_process_setup_hook", None)
            runtime["env_vars"] = {
                key: value for key, value in env.items() if key not in ("TPU_VISIBLE_CHIPS", "TPU_VISIBLE_DEVICES")
            }
            runtime["env_vars"]["VERL_USE_EXTERNAL_MODULES"] = "verl_hardware_plugin"
            json.dumps(runtime)
            scheduling = PlacementGroupSchedulingStrategy(
                placement_group=scheduling_groups[host_index],
                placement_group_bundle_index=bundle_offset + local_rank,
                placement_group_capture_child_tasks=True,
            )
            actor = RemoteProbe.options(
                num_cpus=1 if role == "trainer" else 0,
                resources={"TPU": 1},
                scheduling_strategy=scheduling,
                runtime_env=runtime,
            ).remote()
            actors.append(actor)
            references.append(
                actor.run.remote(
                    env,
                    role,
                    replica,
                    rank,
                    world_size,
                    min(300, max(1, int(remaining()))),
                    required_source_files if role == "sampler" else None,
                )
            )
        return references

    def stop_phase(phase_actors, phase_groups):
        for actor in phase_actors:
            ray.kill(actor, no_restart=True)
        for group in phase_groups:
            ray.util.remove_placement_group(group)
        limit = time.monotonic() + min(remaining(), 30)
        while time.monotonic() < limit:
            if all(ray.util.placement_group_table(group).get("state") == "REMOVED" for group in phase_groups):
                return
            time.sleep(0.5)
        raise RuntimeError("TPU preflight placement groups were not released")

    try:
        trainer_groups, trainer_nodes = create_role_groups("trainer", "global_pool", "tpu-group-0")
        trainer_actors = len(actors)
        refs = spawn_group(
            trainer_groups, trainer_groups, "trainer", 0, trainer_nodes[0]["NodeManagerAddress"], hosts * devices
        )
        result["trainer"] = ray.get(refs, timeout=remaining())
        if len(result["trainer"]) != 32 or {entry["rank"] for entry in result["trainer"]} != set(range(32)):
            raise RuntimeError("Trainer preflight results do not cover every rank")
        for entry in result["trainer"]:
            host_index, local_rank = divmod(entry["rank"], devices)
            if (
                entry["node_id"] != trainer_nodes[host_index]["NodeID"]
                or entry["visible_chip"] != str(local_rank)
                or entry["physical_task_id"] != str(host_index)
                or entry["cloud_task_id"] != str(host_index)
            ):
                raise RuntimeError("Trainer hardware identity does not match its rank placement")
        if len({(entry["node_id"], entry["pid"]) for entry in result["trainer"]}) != hosts * devices:
            raise RuntimeError("Trainer ranks did not execute in 32 distinct processes")
        print("TPU_PREFLIGHT trainer all 32 ranks passed: all_reduce=496; compiled backward passed", flush=True)
        stop_phase(actors[trainer_actors:], trainer_groups)

        sampler_groups, sampler_nodes = create_role_groups("sampler", "rollout_pool", "tpu-group-1")
        sampler_actors = len(actors)
        refs = []
        # The plugin executor reuses the TPU reservations in the rollout pool.
        # Probe those same bundles, with one independent native group per TP1 replica.
        for replica in range(sampler_replicas):
            host_index, chip_slot = divmod(replica * sampler_tp, devices)
            node = sampler_nodes[host_index]
            refs.extend(
                spawn_group(
                    sampler_groups,
                    [sampler_groups[host_index]],
                    "sampler",
                    replica,
                    node["NodeManagerAddress"],
                    sampler_tp,
                    bundle_offset=chip_slot,
                    master_port=29651 + chip_slot,
                )
            )
        result["samplers"] = ray.get(refs, timeout=remaining())
        for replica in range(sampler_replicas):
            rows = [entry for entry in result["samplers"] if entry["replica"] == replica]
            if len(rows) != sampler_tp or {entry["rank"] for entry in rows} != set(range(sampler_tp)):
                raise RuntimeError(f"Sampler replica {replica} did not prove all {sampler_tp} ranks")
            if len({entry["node_id"] for entry in rows}) != 1:
                raise RuntimeError(f"Sampler replica {replica} crossed host boundaries")
            for entry in rows:
                host_index = replica * sampler_tp // devices
                if (
                    entry["node_id"] != sampler_nodes[host_index]["NodeID"]
                    or entry["world_size"] != sampler_tp
                    or entry["physical_task_id"] != "0"
                    or entry["cloud_task_id"] != "0"
                ):
                    raise RuntimeError(f"Sampler replica {replica} has incorrect hardware identity")
                if sampler_tp == devices and entry["visible_chip"] != str(entry["rank"]):
                    raise RuntimeError(f"TP4 sampler replica {replica} has an incorrect visible chip")
                if sampler_tp == 1:
                    chip = int(entry["visible_chip"])
                    native_port = 8070 + chip
                    if (
                        entry["assigned_tpu_ids"] != [str(chip)]
                        or entry["native_visible_devices"] != str(chip)
                        or entry["topology"] != "1,1,1"
                        or entry["slice_endpoints"]
                        != f"{sampler_nodes[host_index]['NodeManagerAddress']}:{native_port}"
                        or entry["native_process_port"] != str(native_port)
                        or entry["master_port"] != str(29651 + chip)
                    ):
                        raise RuntimeError(f"TP1 sampler replica {replica} did not prove isolated native geometry")
        if len(result["samplers"]) != hosts * devices:
            raise RuntimeError("Sampler hardware results do not cover all 32 processes")
        if len({(entry["node_id"], entry["visible_chip"]) for entry in result["samplers"]}) != hosts * devices:
            raise RuntimeError("Sampler hardware results do not cover 32 distinct physical chips")
        if len({(entry["node_id"], entry["pid"]) for entry in result["samplers"]}) != hosts * devices:
            raise RuntimeError("Sampler hardware results do not cover 32 distinct processes")
        if sampler_tp == 1:
            for node in sampler_nodes:
                if {entry["visible_chip"] for entry in result["samplers"] if entry["node_id"] == node["NodeID"]} != {
                    "0",
                    "1",
                    "2",
                    "3",
                }:
                    raise RuntimeError("A TP1 sampler host did not prove each of its four physical chips")
            print(
                "TPU_PREFLIGHT all 32 TP1 sampler replicas passed: "
                "one rank each, all_reduce=0; compiled backward passed",
                flush=True,
            )
        else:
            print(
                "TPU_PREFLIGHT all 8 sampler hosts passed: 4 ranks each, all_reduce=6; compiled backward passed",
                flush=True,
            )
        stop_phase(actors[sampler_actors:], sampler_groups)
    except BaseException as error:
        failure = error
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    finally:
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
        for group in groups:
            try:
                ray.util.remove_placement_group(group)
            except Exception:
                pass
        release_deadline = time.monotonic() + 30
        while time.monotonic() < release_deadline:
            released = all(ray.util.placement_group_table(group).get("state") == "REMOVED" for group in groups)
            free_tpus = float(ray.available_resources().get("TPU", 0))
            if released and free_tpus >= hosts * devices * 2:
                result["resources_released"] = True
                result["ray_tpus_available_after_cleanup"] = free_tpus
                break
            time.sleep(0.5)
        if not result["resources_released"] and failure is None:
            failure = RuntimeError("TPU hardware resources were not released before training")
            result["error"] = str(failure)
        result["passed"] = failure is None
        write_result(result_path, result)
        ray.shutdown()
    if failure is not None:
        raise failure
    print(
        "TPU_PREFLIGHT PASSED: real trainer and sampler collectives, compiled backward, resources released", flush=True
    )


if __name__ == "__main__":
    main()
'''
    ),
    "launcher/train.sh": (
        r"""#!/usr/bin/env bash
set -euo pipefail

# Execute the configured training script from the isolated source copy.
# The packaged configuration supplies only submission-specific overrides.
: "${RUN_ID:?RUN_ID must identify this run}"
python3 - "$@" <<'PY'
import json
import os
from pathlib import Path
import sys

configuration = Path("run-training.json")
cfg = json.loads(configuration.read_text()) if configuration.exists() else {}
training = cfg.get("training", {})
actor_memory = cfg.get("actor_memory", {})
resources = cfg.get("resources", cfg.get("geometry", {}))
steps = int(training.get("steps", 4))
short_run = steps <= 4
prompts = int(training.get("prompts_per_step", 8 if short_run else 128))
rollouts = int(training.get("rollouts_per_prompt", 4 if short_run else 16))
microbatch = int(training.get("micro_batch_size_per_gpu", 4 if short_run else 16))
hosts = int(resources.get("hosts_per_role", 1))
actor_hosts = int(resources.get("actor_hosts", hosts))
rollout_hosts = int(resources.get("rollout_hosts", hosts))
devices = int(resources.get("logical_devices_per_host", 8))
actor_devices = int(resources.get("actor_logical_devices_per_host", devices))
rollout_devices = int(resources.get("rollout_logical_devices_per_host", devices))
actor_shards = int(resources.get("actor_data_parallel_shard_size", actor_hosts * actor_devices))
rollout_parallelism = int(resources.get("rollout_tensor_parallel_size", rollout_devices))
validation = bool(training.get("validation", not short_run))
save_freq = int(training.get("save_freq", 4 if short_run else -1))
seed = int(training.get("seed", 42))
prompt_length = int(training.get("max_prompt_length", 512))
response_length = int(training.get("max_response_length", 2048))
validation_samples = int(training.get("validation_samples", 1319))
if min(steps, prompts, rollouts, microbatch, actor_hosts, rollout_hosts,
       actor_devices, rollout_devices, actor_shards, rollout_parallelism) <= 0:
    raise SystemExit("Training and resource counts must be positive")
if actor_shards != actor_hosts * actor_devices:
    raise SystemExit("Pure FSDP must cover every actor logical device")
if (rollout_hosts * rollout_devices) % rollout_parallelism:
    raise SystemExit("Rollout tensor parallel size must divide rollout logical devices")

env = dict(os.environ)
env.update({
    "SMOKE_TEST": "0",
    "TRAIN_BATCH_SIZE": str(prompts),
    "PPO_MINI_BATCH_SIZE": str(training.get("ppo_mini_batch_size", prompts)),
    "MICRO_BATCH_SIZE": str(microbatch),
    "ROLLOUT_N": str(rollouts),
    "MAX_RESPONSE_LEN": str(response_length),
    "TOTAL_TRAINING_STEPS": str(steps),
    "LR_WARMUP_STEPS": str(training.get("lr_warmup_steps", 0 if short_run else 10)),
    "VAL_BATCH_SIZE": str(validation_samples),
    "VAL_MAX_SAMPLES": str(validation_samples),
    "TEST_FREQ": str(training.get("test_freq", 25 if validation else -1)),
    "VAL_BEFORE_TRAIN": str(bool(training.get("val_before_train", False))),
    "TENSOR_PARALLEL_SIZE": "1",
    "DATA_PARALLEL_SHARD_SIZE": str(actor_shards),
})
run_id = env["RUN_ID"]
overrides = [
    f"trainer.experiment_name={run_id}",
    f"trainer.total_training_steps={steps}",
    f"data.max_prompt_length={prompt_length}",
    f"data.max_response_length={response_length}",
    f"actor_rollout_ref.actor.optim.lr={training.get('lr', 2e-6)}",
    f"actor_rollout_ref.actor.optim.decay_type={training.get('lr_decay_type', 'cosine')}",
    f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={microbatch}",
    f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={microbatch}",
    "actor_rollout_ref.actor.torchtitan.tensor_parallel_size=1",
    f"actor_rollout_ref.actor.torchtitan.data_parallel_shard_size={actor_shards}",
    f"trainer.nnodes={actor_hosts}",
    f"trainer.n_gpus_per_node={actor_devices}",
    f"actor_rollout_ref.rollout.nnodes={rollout_hosts}",
    f"actor_rollout_ref.rollout.n_gpus_per_node={rollout_devices}",
    f"rollout.nnodes={rollout_hosts}",
    f"rollout.n_gpus_per_node={rollout_devices}",
    f"actor_rollout_ref.rollout.tensor_model_parallel_size={rollout_parallelism}",
    "actor_rollout_ref.rollout.data_parallel_size=1",
    "algorithm.rollout_correction.bypass_mode=True",
    f"trainer.val_before_train={bool(training.get('val_before_train', False))}",
    f"trainer.save_freq={save_freq}",
    f"trainer.default_local_dir=/tmp/verl_checkpoints/{run_id}",
    "trainer.logger=" + json.dumps(training.get("logger", ["console", "tensorboard", "file", "wandb"])),
    "trainer.project_name=" + str(training.get("project_name", "verl_tpu_grpo")),
    f"hydra.run.dir=/tmp/verl_hydra/{run_id}",
    "trainer.resume_mode=disable",
    "data.dataloader_num_workers=0",
    f"data.seed={seed}",
    f"actor_rollout_ref.actor.data_loader_seed={seed}",
    f"actor_rollout_ref.actor.torchtitan.seed={seed}",
    f"actor_rollout_ref.ref.torchtitan.seed={seed}",
    f"actor_rollout_ref.rollout.seed={seed}",
    f"++actor_rollout_ref.rollout.engine_kwargs.vllm.seed={seed}",
    "+data.apply_chat_template_kwargs.enable_thinking=True",
    "actor_rollout_ref.rollout.enable_prefix_caching=False",
    f"trainer.rollout_data_dir=/tmp/verl_dump/{run_id}/rollout",
    f"trainer.validation_data_dir=/tmp/verl_dump/{run_id}/validation",
]
for key, allowed in (
    ("activation_checkpoint", {"none", "selective", "full"}),
    ("reshard_after_forward", {"default", "always", "never"}),
):
    if key in actor_memory:
        value = actor_memory[key]
        if value not in allowed:
            raise SystemExit(f"Unsupported actor memory setting {key}={value}")
        overrides.append(f"actor_rollout_ref.actor.torchtitan.{key}={value}")
if "model_path" in training:
    env["MODEL_PATH"] = str(training["model_path"])
    overrides.append(f"actor_rollout_ref.model.path={training['model_path']}")
extra_overrides = training.get("extra_overrides", [])
if not isinstance(extra_overrides, list) or not all(isinstance(value, str) for value in extra_overrides):
    raise SystemExit("training.extra_overrides must be a list of command-line strings")
overrides.extend(extra_overrides)
script = Path(cfg.get("training_script", "examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh"))
if script.is_absolute() or ".." in script.parts:
    raise SystemExit("Training script must be relative to the isolated source copy")
if not script.is_file():
    raise SystemExit(f"Original training script missing from isolated copy: {script}")
os.execvpe("bash", ["bash", str(script), *overrides, *sys.argv[1:]], env)
PY
"""
    ),
    "original-gsm8k-inputs.json": (
        r"""[
  {
    "source": "gs://ubench-logs/lixali/verl-ray-kueue/qwen3-grpo-tpu-200steps-reinforce-is3-no-prefix-seed42-20261002-183700/data/jialei/data/gsm8k/test.parquet",
    "destination": "/data/jialei/data/gsm8k/test.parquet",
    "generation": "1790966214215673",
    "bytes": 1183032,
    "sha256": "e801bce6b15925630ea9976dc14419e49735a90fd54d267911e6701ebdc0d489"
  },
  {
    "source": "gs://ubench-logs/lixali/verl-ray-kueue/qwen3-grpo-tpu-200steps-reinforce-is3-no-prefix-seed42-20261002-183700/data/jialei/data/gsm8k/train.parquet",
    "destination": "/data/jialei/data/gsm8k/train.parquet",
    "generation": "1790966214249561",
    "bytes": 6555601,
    "sha256": "89cd3cb8d28e5274e7f0bf71ff541ea5654ac9e30589ac4b5d19c3f783a3858c"
  }
]
"""
    ),
    "overlay_plugin_v6e.py": (
        r'''#!/usr/bin/env python3
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Apply the 32 trainer / 32 TP1 sampler recipe to an isolated plugin snapshot."""

from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

MARKER = "# Qwen3-4B-Base recipe: isolated v6e placement and physical chip binding."
RUNTIME_NAME = "recipe_v6e_runtime.py"

RUNTIME = r'''
        "'''"
        r'''# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Native v6e geometry and TP1 binding for this isolated submission only."""

import inspect
import os


def clear_multislice_environment():
    removed = sorted(name for name in os.environ if name.startswith("MEGASCALE_"))
    for name in removed:
        os.environ.pop(name, None)
    return removed


def slice_geometry(world_size, local_world_size, num_hosts):
    if (world_size, local_world_size, num_hosts) == (1, 1, 1):
        topology, host_bounds, chip_bounds = "1,1,1", "1,1,1", "1,1,1"
    elif (world_size, local_world_size, num_hosts) == (32, 4, 8):
        topology, host_bounds, chip_bounds = "4,8,1", "2,4,1", "2,2,1"
    else:
        raise ValueError("Recipe requires 32 trainer chips on eight hosts or one TP1 sampler chip")
    return {
        "TORCH_TPU_TOPOLOGY": topology,
        "TPU_HOST_BOUNDS": host_bounds,
        "TPU_CHIPS_PER_HOST_BOUNDS": chip_bounds,
        "CHIPS_PER_HOST": str(local_world_size),
        "TORCH_TPU_DEVICES_PER_HOST": str(local_world_size),
        "TPU_LOCAL_DEVICE_COUNT": str(local_world_size),
        "TPU_PHYSICAL_DEVICES_PER_HOST": "4",
        "TPU_NUM_HOSTS": str(num_hosts),
    }


def assigned_tpu_ids():
    import ray

    assignments = ray.get_runtime_context().worker.core_worker.resource_ids()
    chips = set()
    for name, entries in assignments.items():
        if name == "TPU" or (name.startswith("TPU_group_") and name[len("TPU_group_"):].isalnum()):
            for chip, _ in entries:
                if int(chip) != chip:
                    raise ValueError("Ray assigned a nonintegral physical TPU chip")
                chips.add(int(chip))
    return [str(chip) for chip in sorted(chips)]


def _bundle_locations(groups):
    import ray

    nodes = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes() if node.get("Alive")}
    locations = []
    for group in groups:
        spec = ray._private.state.state.placement_group_table(group.id)
        if spec.get("state") != "CREATED":
            raise ValueError("Recipe placement group is not ready")
        mapping = spec["bundles_to_node_id"]
        for index in range(len(group.bundle_specs)):
            if mapping.get(index) not in nodes:
                raise ValueError("Recipe placement group refers to a missing worker node")
            locations.append((group, index, nodes[mapping[index]]))
    return locations


def pool_environment(resource_pool, rank, world_size, local_rank, local_world_size, name_prefix, device_name):
    groups = resource_pool.get_placement_groups(device_name=device_name)
    locations = _bundle_locations(groups)
    start = getattr(resource_pool, "start_bundle_index", 0)
    selected = locations[start:start + world_size]
    if len(selected) != world_size:
        raise ValueError("Recipe resource subpool extends beyond its placement groups")
    is_sampler = world_size == 1 and "rollout" in name_prefix.lower()
    if is_sampler:
        ip = selected[0][2]
        chip = selected[0][1]
        if not 0 <= chip < 4:
            raise ValueError("TP1 recipe requires four-chip host placement groups")
        geometry = slice_geometry(1, 1, 1)
        addresses = f"{ip}:{8070 + chip}"
        task_id = 0
        hosts = [ip]
    else:
        ips = [entry[2] for entry in selected]
        hosts = list(dict.fromkeys(ips))
        geometry = slice_geometry(world_size, local_world_size, len(hosts))
        if any(ips.count(ip) != 4 for ip in hosts):
            raise ValueError("Trainer placement groups must cover four chips on each of eight hosts")
        addresses = ",".join(f"{ip}:{8471 + index % 4}" for index, ip in enumerate(ips))
        chip, task_id = local_rank, rank // 4
    return {
        **geometry,
        "TPU_VISIBLE_CHIPS": str(chip),
        "TPU_PROCESS_PORT": str((8070 if is_sampler else 8471) + chip),
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": addresses,
        "TPU_PROCESS_ADDRESSES": addresses,
        "TPU_WORKER_HOSTNAMES": ",".join(hosts),
        "NODE_RANK": str(task_id),
        "TPU_WORKER_ID": str(task_id),
        "CLOUD_TPU_TASK_ID": str(task_id),
    }


def worker_environment(worker_ips, base_port, physical_chips=None):
    if len(worker_ips) != 1 or physical_chips is None or len(physical_chips) != 1:
        raise ValueError("Recipe rollout requires a single Ray-assigned TP1 worker")
    chip = int(physical_chips[0])
    if not 0 <= chip < 4:
        raise ValueError("TP1 worker received an invalid physical TPU chip")
    ip = worker_ips[0]
    return [{
        **slice_geometry(1, 1, 1),
        "TPU_VISIBLE_DEVICES": str(chip),
        "TPU_VISIBLE_CHIPS": str(chip),
        "TPU_PROCESS_PORT": str(base_port + chip),
        "TPU_PROCESS_ADDRESSES": f"{ip}:{base_port + chip}",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": f"{ip}:{base_port + chip}",
        "TPU_WORKER_HOSTNAMES": ip,
        "NODE_RANK": "0", "TPU_WORKER_ID": "0", "CLOUD_TPU_TASK_ID": "0",
    }]


def single_chip_worker_class(wrapper_class):
    # Capture strings and geometry, never the native class or source helper functions.
    # Disable backend autoload at process startup, then bind before explicit native imports.
    wrapper_module, wrapper_name = wrapper_class.__module__, wrapper_class.__name__
    geometry = slice_geometry(1, 1, 1)

    class SingleChipRayWorker:
        def __init__(self, *args, **kwargs):
            import importlib
            import json
            import os
            import sys
            import ray

            context = ray.get_runtime_context()
            assignments = context.worker.core_worker.resource_ids()
            chips = set()
            for name, entries in assignments.items():
                if name == "TPU" or (name.startswith("TPU_group_") and name[len("TPU_group_"):].isalnum()):
                    for chip, _ in entries:
                        if int(chip) != chip:
                            raise ValueError("Ray assigned a nonintegral physical TPU chip")
                        chips.add(int(chip))
            if len(chips) != 1 or not 0 <= next(iter(chips)) < 4:
                raise ValueError("TP1 wrapper requires one valid physical TPU chip")
            chip = next(iter(chips))
            ip = ray.util.get_node_ip_address()
            torch_imported = "torch" in sys.modules
            torch_tpu_imported = "torch_tpu" in sys.modules
            tpu_module = getattr(sys.modules.get("torch"), "tpu", None)
            is_initialized = getattr(tpu_module, "is_initialized", None)
            if torch_tpu_imported and not callable(is_initialized):
                native_device = sys.modules.get("torch_tpu._internal.device._device_module")
                native_ops = sys.modules.get("torch_tpu._internal.device._device_ops_backend")
                if native_ops is None:
                    native_ops = getattr(native_device, "_device_ops_backend", None)
                is_initialized = getattr(native_ops, "_is_initialized", None)
                internal_imported = any(name.startswith("torch_tpu._internal") for name in tuple(sys.modules))
                # The pinned package's bare __init__ only installs its loader and library path.
                # Once internals are imported, require a native initialization-state query.
                if not callable(is_initialized) and (tpu_module is not None or internal_imported):
                    raise RuntimeError("Cannot verify whether the imported TorchTPU runtime is initialized")
            runtime_initialized = bool(is_initialized()) if callable(is_initialized) else False
            if runtime_initialized:
                raise RuntimeError("TP1 wrapper must bind its physical chip before initializing TorchTPU")
            for name in tuple(os.environ):
                if name.startswith("MEGASCALE_"):
                    os.environ.pop(name, None)
            os.environ.update(geometry)
            os.environ.update({
                "WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0", "NODE_RANK": "0",
                "TPU_WORKER_ID": "0", "CLOUD_TPU_TASK_ID": "0",
                "TPU_VISIBLE_DEVICES": str(chip), "TPU_VISIBLE_CHIPS": str(chip),
                "TPU_PROCESS_PORT": str(8070 + chip), "TPU_WORKER_HOSTNAMES": ip,
                "TORCH_TPU_SLICEBUILDER_ADDRESSES": f"{ip}:{8070 + chip}",
                "TPU_PROCESS_ADDRESSES": f"{ip}:{8070 + chip}",
            })
            print(json.dumps({"event": "v6e_tp1_binding_before_wrapper_import",
                              "physical_tpu_ids": [chip], "node_id": context.get_node_id(),
                              "torch_imported_before_binding": torch_imported,
                              "torch_tpu_imported_before_binding": torch_tpu_imported,
                              "tpu_runtime_initialized_before_binding": runtime_initialized,
                              "torch_device_backend_autoload": os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD"),
                              "tpu_visible_chips": os.environ["TPU_VISIBLE_CHIPS"]}), flush=True)
            # Bare torch_tpu import does not register torch.tpu when autoload is disabled.
            # Invoke the pinned image's backend entrypoint only after binding its chip.
            torch_module = importlib.import_module("torch")
            if not hasattr(torch_module, "tpu"):
                importlib.import_module("torch_tpu._loader").load()
            if not hasattr(torch_module, "tpu"):
                raise RuntimeError("TorchTPU backend registration failed after TP1 chip binding")
            from recipe_tpu_plugin_layout import tpu_plugin_layout

            patches = importlib.import_module(tpu_plugin_layout()["patches_module"])
            patches.patch_vllm_for_tpu()
            image_class = getattr(importlib.import_module(wrapper_module), wrapper_name)
            self._image_worker = image_class(*args, **kwargs)

        def recipe_get_physical_tpu_ids(self):
            import ray

            assignments = ray.get_runtime_context().worker.core_worker.resource_ids()
            chips = {int(chip) for name, entries in assignments.items()
                     if name == "TPU" or (name.startswith("TPU_group_") and name[len("TPU_group_"):].isalnum())
                     for chip, _ in entries}
            return sorted(chips)

    def make_delegate(method_name, is_async):
        if is_async:
            async def delegate(self, *args, **kwargs):
                return await getattr(self._image_worker, method_name)(*args, **kwargs)
        else:
            def delegate(self, *args, **kwargs):
                return getattr(self._image_worker, method_name)(*args, **kwargs)
        delegate.__name__ = method_name
        return delegate

    for name in dir(wrapper_class):
        if not name.startswith("_") and callable(getattr(wrapper_class, name)):
            delegate = make_delegate(name, inspect.iscoroutinefunction(getattr(wrapper_class, name)))
            setattr(SingleChipRayWorker, name, delegate)
    return SingleChipRayWorker
'''
        "'''"
        r'''


def replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one {label} anchor; found {source.count(old)}")
    return source.replace(old, new, 1)


def replace_function(source, name, replacement, class_name=None):
    tree = ast.parse(source)
    scope = tree.body
    if class_name:
        scope = next(node for node in scope if isinstance(node, ast.ClassDef) and node.name == class_name).body
    nodes = [node for node in scope if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(nodes) != 1:
        raise ValueError(f"Expected exactly one {class_name or ''}.{name}")
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    return "".join(lines[: node.lineno - 1]) + replacement.rstrip() + "\n" + "".join(lines[node.end_lineno :])


def precision_functions(source):
    """Fingerprint the two precision implementations without freezing placement glue."""
    functions = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in {"patch_tpu_logprobs", "patch_tpu_sampler"}:
            functions[node.name] = hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
    if len(functions) != 2:
        raise ValueError("Expected both FP32 rollout precision implementations")
    return functions


def precision_fingerprint(snapshot):
    from recipe_tpu_plugin_layout import tpu_plugin_layout

    layout = tpu_plugin_layout(snapshot)
    precision = snapshot / layout["precision"]
    if precision.is_file():
        return {"precision_module": hashlib.sha256(precision.read_bytes()).hexdigest()}
    patches = snapshot / layout["patches"]
    return precision_functions(patches.read_text())


def prepare(snapshot):
    from recipe_tpu_plugin_layout import tpu_plugin_layout

    layout = tpu_plugin_layout(snapshot)
    paths = [snapshot / layout[name] for name in ("platform", "rollout", "patches")]
    platform, rollout, patches = [path.read_text() for path in paths]
    original_precision = precision_fingerprint(snapshot)
    if any(MARKER in source for source in (platform, rollout, patches)):
        raise ValueError("Recipe overlay already applied; start from a fresh snapshot")
    platform = replace_once(
        platform,
        "import ray\n",
        "import ray\n\n"
        + MARKER
        + "\nfrom recipe_v6e_runtime import clear_multislice_environment, pool_environment\n"
        + "\nclear_multislice_environment()\n",
        "platform import",
    )
    platform = replace_function(
        platform,
        "get_worker_env_vars",
        """    def get_worker_env_vars(
        self, resource_pool, rank, world_size, local_rank, local_world_size, name_prefix, device_name,
    ):
        env_vars = {"VERL_PLATFORM": "tpu", **{name: "1" for name in self.ray_noset_envvars()}}
        env_vars.update(pool_environment(resource_pool, rank, world_size, local_rank,
                                         local_world_size, name_prefix, device_name))
        return env_vars
""",
        "PlatformTPU",
    )
    platform = replace_once(
        platform,
        '    os.environ["VERL_PLATFORM"] = "tpu"\n',
        '    clear_multislice_environment()\n    os.environ["VERL_PLATFORM"] = "tpu"\n',
        "worker cleanup",
    )
    rollout = replace_once(
        rollout,
        '        "VERL_TPU_PG_IDS": ",".join(pg.id.hex() for pg in pgs),\n',
        '        "VERL_TPU_PG_IDS": ",".join(pg.id.hex() for pg in pgs),\n'
        + '        "VERL_TPU_START_BUNDLE_INDEX": str(getattr(replica.resource_pool, "start_bundle_index", 0)),\n',
        "replica subpool offset",
    )
    rollout = replace_once(rollout, "import ray\n", "import ray\n\n" + MARKER + "\n", "rollout marker")
    patches = replace_once(
        patches,
        "import ray\n",
        "import ray\n\n"
        + MARKER
        + "\nfrom recipe_v6e_runtime import (\n"
        + "    clear_multislice_environment, single_chip_worker_class, worker_environment,\n"
        + ")\n\nclear_multislice_environment()\n",
        "executor helper imports",
    )
    patches = replace_once(
        patches,
        "    ][: self.parallel_config.world_size]\n",
        "    ]\n"
        + '    start = int(os.environ.get("VERL_TPU_START_BUNDLE_INDEX", "0"))\n'
        + "    bundles = bundles[start:start + self.parallel_config.world_size]\n"
        + "    if self.parallel_config.world_size != 1 or len(bundles) != 1:\n"
        + '        raise ValueError("Recipe requires one selected bundle per TP1 replica")\n'
        + "    RayWorkerWrapper = single_chip_worker_class(RayWorkerWrapper)\n"
        + "    ray_remote_kwargs = dict(ray_remote_kwargs)\n"
        + '    runtime_env = dict(ray_remote_kwargs.get("runtime_env") or {})\n'
        + '    runtime_env["env_vars"] = dict(runtime_env.get("env_vars") or {})\n'
        + '    runtime_env["worker_process_setup_hook"] = "builtins.dict"\n'
        + '    runtime_env["env_vars"]["RAY_RUNTIME_ENV_WORKER_PROCESS_SETUP_HOOK"] = "builtins.dict"\n'
        # Ray may import torch while deserializing actors; defer its native backend autoload.
        + '    runtime_env["env_vars"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"\n'
        + '    ray_remote_kwargs["runtime_env"] = runtime_env\n',
        "executor subpool selection",
    )
    patches = replace_once(
        patches,
        "    ips = ray.get([worker.get_node_ip.remote() for worker in self.workers])\n",
        "    ips = ray.get([worker.get_node_ip.remote() for worker in self.workers])\n"
        + "    physical_ids = ray.get([worker.recipe_get_physical_tpu_ids.remote() for worker in self.workers])\n"
        + "    if any(len(ids) != 1 for ids in physical_ids):\n"
        + '        raise ValueError("TP1 replica worker must own exactly one physical TPU chip")\n',
        "physical allocation lookup",
    )
    patches = replace_once(
        patches,
        "    ips = [ips[i] for i in order]\n",
        "    ips = [ips[i] for i in order]\n    physical_chips = [physical_ids[i][0] for i in order]\n",
        "physical allocation order",
    )
    patches = replace_once(
        patches,
        "        for tpu_env in _tpu_worker_envs(ips, vllm_torchtpu_envs.TORCH_TPU_BASE_PORT)\n",
        "        for tpu_env in worker_environment(ips, vllm_torchtpu_envs.TORCH_TPU_BASE_PORT, physical_chips)\n",
        "native TP1 environment",
    )
    patches = replace_once(
        patches,
        "        if name in os.environ\n    }\n",
        '        if name in os.environ and not name.startswith("MEGASCALE_")\n    }\n',
        "environment cleanup",
    )
    outputs = dict(zip(paths, (platform, rollout, patches), strict=True))
    outputs[snapshot / RUNTIME_NAME] = RUNTIME
    if "precision_module" not in original_precision and precision_functions(patches) != original_precision:
        raise RuntimeError("Recipe overlay changed the FP32 precision implementations")
    for path, text in outputs.items():
        compile(text, str(path), "exec")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--check", action="store_true", help="Validate anchors and syntax without writing.")
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    if (snapshot / ".git").exists():
        parser.error("Apply this overlay only to an isolated submission snapshot, never a git checkout")
    original = precision_fingerprint(snapshot)
    outputs = prepare(snapshot)
    if not args.check:
        for path, text in outputs.items():
            path.write_text(text)
    if precision_fingerprint(snapshot) != original:
        raise RuntimeError("Recipe overlay changed the FP32 precision implementation")
    print(f"Recipe v6e overlay {'validated' if args.check else 'applied'}: {len(outputs)} files")


if __name__ == "__main__":
    main()
'''
    ),
    "prepare_qwen3_4b_base_model.py": (
        r'''# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Pinned model-file manifest copied from the working Qwen3-4B-Base launcher."""

from pathlib import Path

MODEL_REPO = "Qwen/Qwen3-4B-Base"
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
MODEL_URL_ROOT = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}"

# Small-file SHA256 values were computed from immutable resolve URLs. Weight
# hashes are the official Hugging Face LFS SHA256 values; no weights were fetched
# while preparing this manifest.
MODEL_FILES = {
    "LICENSE": {
        "size_bytes": 11343,
        "sha256": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    },
    "config.json": {
        "size_bytes": 727,
        "sha256": "304b2545a258d35620f1d4bf46940c0471d9baa00715ff8e77f84c2fca5057c1",
    },
    "generation_config.json": {
        "size_bytes": 138,
        "sha256": "8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8",
    },
    "merges.txt": {
        "size_bytes": 1671853,
        "sha256": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    },
    "model-00001-of-00003.safetensors": {
        "size_bytes": 3957900840,
        "sha256": "4c807e2503d68ae373d508689d00a41f4b33f33c2536da97ab81a20caddc1241",
    },
    "model-00002-of-00003.safetensors": {
        "size_bytes": 3987450520,
        "sha256": "f4707585548b2fc75a6b1d732e8465c62040a8699903c32850781beeb9b27826",
    },
    "model-00003-of-00003.safetensors": {
        "size_bytes": 99630640,
        "sha256": "c7b1aa8fb672de2e00423c99876926022e50b18d4f0d140670788510a27f9965",
    },
    "model.safetensors.index.json": {
        "size_bytes": 32819,
        "sha256": "d6c42883a895dfef5b0080ed2116a1bcd764f558406b98923d675978a1abf29c",
    },
    "tokenizer.json": {
        "size_bytes": 7031645,
        "sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    },
    "tokenizer_config.json": {
        "size_bytes": 9678,
        "sha256": "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
    },
    "vocab.json": {
        "size_bytes": 2776833,
        "sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    },
}
for _filename, _record in MODEL_FILES.items():
    _record["url"] = f"{MODEL_URL_ROOT}/{_filename}"


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "assets/qwen3-4b-base-model"
'''
    ),
    "recipe_response_logging.py": (
        r'''#!/usr/bin/env python3
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Detached per-response trainer/generator log-probability artifacts.

The CLI installs additive logging hooks in a staged source copy. Hooks preserve
the original training equations and tensors and write JSONL under the recipe's
existing rollout/validation directories, which the artifact mirror uploads.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import os
import shutil
from pathlib import Path

_LOSS_FORWARD_COUNTER = itertools.count()


def response_record(trainer, generator, token_ids, mask, uid, context, *, save_raw=None):
    """Pure CPU/list conversion: masked padding never contributes extrema."""
    if not (len(generator) == len(mask)) or (trainer is not None and len(trainer) != len(mask)):
        raise ValueError("Log-probability/mask lengths are not aligned")
    positions = [i for i, keep in enumerate(mask) if keep]
    if positions and max(positions) >= len(token_ids):
        raise ValueError("Response token IDs do not cover valid log-probability positions")
    generator_values = [float(generator[i]) for i in positions]
    trainer_values = [float(trainer[i]) for i in positions] if trainer is not None else None
    key_parts = str(uid).rsplit("_", 2)
    response_indices = {}
    if len(key_parts) == 3 and key_parts[1].isdigit() and key_parts[2].isdigit():
        response_indices = {
            "prompt_uid": key_parts[0],
            "rollout_index": int(key_parts[1]),
            "output_index": int(key_parts[2]),
        }
    record = {
        **context,
        **response_indices,
        "run_id": os.environ.get("RUN_ID"),
        "uid": str(uid),
        "response_id": str(uid),
        "valid_response_tokens": len(positions),
        "trainer_log_prob_min": min(trainer_values) if trainer_values else None,
        "trainer_log_prob_max": max(trainer_values) if trainer_values else None,
        "trainer_log_prob_mean": sum(trainer_values) / len(trainer_values) if trainer_values else None,
        "generator_log_prob_min": min(generator_values) if generator_values else None,
        "generator_log_prob_max": max(generator_values) if generator_values else None,
        "generator_log_prob_mean": sum(generator_values) / len(generator_values) if generator_values else None,
    }
    if save_raw is None:
        save_raw = os.environ.get("RECIPE_SAVE_RAW_TOKEN_LOG_PROBS", "0") == "1"
    if save_raw:
        # Preserve the original artifact convention for existing plot parsers.
        record.update(
            {
                "response_token_positions": positions,
                "token_ids": [int(token_ids[i]) for i in positions],
                "log_prob": trainer_values,
                "old_log_prob": generator_values,
                "generator_log_prob": generator_values,
                "log_prob_minus_old_log_prob": [a - b for a, b in zip(trainer_values, generator_values, strict=True)]
                if trainer_values is not None
                else None,
            }
        )
    return record


def _append_records(directory, records, context):
    directory = Path(directory) / f"step_{context['step']:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    rank = int(context.get("rank", -1))
    path = directory / f"rank_{rank:05d}_pid_{os.getpid()}.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def dump_response_token_log_probs(data, log_prob, old_log_prob, response_mask):
    """Snapshot exact actor loss inputs without retaining autograd or TPU ops.

    With the recipe's bypass_mode=True, old_log_prob is generator log-probability.
    Subtraction/extrema/indexing run on CPU lists after detached tensor copies.
    """
    from verl.utils import tensordict_utils as tu

    directory = tu.get(data, "token_log_prob_dir")
    if not directory or not tu.get(data, "token_log_prob_is_writer", False):
        return
    # Every tensor is detached before transfer; no tensor computation is added
    # to the live loss graph and every model output is passed through unchanged.
    current = log_prob.detach().cpu().tolist()
    generator = old_log_prob.detach().cpu().tolist()
    masks = response_mask.detach().cpu().tolist()
    responses = [response.detach().cpu().tolist() for response in data["responses"].unbind()]
    uids = tu.get(data, "token_log_prob_uid")
    if not (len(current) == len(generator) == len(masks) == len(responses) == len(uids)):
        raise ValueError("Per-response logging lost row/UID alignment")
    context = {
        name: int(tu.get(data, "token_log_prob_" + name)) for name in ("step", "rank", "dp_rank", "epoch", "mini_batch")
    }
    context.update(
        stage="training",
        loss_forward=next(_LOSS_FORWARD_COUNTER),
        current_dtype=str(log_prob.dtype),
        old_dtype=str(old_log_prob.dtype),
    )
    records = [
        response_record(a, b, token_ids, mask, uid, context)
        for a, b, token_ids, mask, uid in zip(current, generator, responses, masks, uids, strict=True)
    ]
    _append_records(directory, records, context)


def dump_validation_token_log_probs(data, keys, directory, step):
    """Log generator probabilities for every validation response, including step 0."""
    generators = [value.detach().cpu().tolist() for value in data["rollout_log_probs"].unbind()]
    masks = [value.detach().cpu().tolist() for value in data["response_mask"].unbind()]
    responses = [value.detach().cpu().tolist() for value in data["responses"].unbind()]
    if not (len(generators) == len(masks) == len(responses) == len(keys)):
        raise ValueError("Validation response probability rows are not aligned")
    context = {"step": int(step), "rank": -1, "dp_rank": -1, "stage": "validation", "trainer_log_prob_available": False}
    records = [
        response_record(None, generator, token_ids, mask, uid, context)
        for generator, token_ids, mask, uid in zip(generators, responses, masks, keys, strict=True)
    ]
    _append_records(Path(directory) / "token_log_probs", records, context)


def install_logging(root):
    """Add only new logging statements; reversing them restores byte-identical source."""
    root = Path(root).resolve()
    if not root.is_dir() or any((directory / ".git").exists() for directory in (root, *root.parents)):
        raise ValueError("Logging hooks require an isolated source snapshot outside any Git checkout")
    patches = {
        "verl/workers/utils/losses.py": [
            (
                "from verl.workers.utils.padding import no_padding_2_padding\n",
                "from recipe_response_logging import dump_response_token_log_probs\n",
            ),
            (
                '    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) '
                'and old_log_probs from data."""\n',
                "    token_log_data = data  # Preserve response IDs before selecting loss fields.\n",
            ),
            (
                "    policy_loss_fn = get_policy_loss_fn(loss_mode)\n",
                "    dump_response_token_log_probs(token_log_data, log_prob, old_log_prob, response_mask)\n",
            ),
        ],
        "verl/workers/engine_workers.py": [
            (
                "            for batch_idx, mini_batch_td in enumerate(dataloader):\n"
                "                maybe_fix_3d_position_ids(mini_batch_td)\n",
                '                if tu.get(mini_batch_td, "token_log_prob_dir"):\n'
                "                    tu.assign_non_tensor(\n"
                "                        mini_batch_td,\n"
                "                        token_log_prob_rank=self.rank,\n"
                "                        token_log_prob_dp_rank=self.engine.get_data_parallel_rank(),\n"
                "                        token_log_prob_epoch=batch_idx // "
                "(batch_size_per_dp // mini_batch_size_per_gpu),\n"
                "                        token_log_prob_mini_batch=batch_idx % "
                "(batch_size_per_dp // mini_batch_size_per_gpu),\n"
                "                        token_log_prob_is_writer=self.engine.is_mp_src_rank_with_outputs(),\n"
                "                    )\n",
            ),
        ],
        "verl/trainer/ppo/v1/trainer_base.py": [
            (
                "from verl.checkpoint_engine import CheckpointEngineManager\n",
                "from recipe_response_logging import dump_validation_token_log_probs\n",
            ),
            (
                "    def _update_actor(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:\n"
                '        """Update the actor network."""\n',
                '        if self.config.trainer.get("log_response_token_probs", False):\n'
                '            dump_dir = self.config.trainer.get("rollout_data_dir")\n'
                "            if not dump_dir:\n"
                '                raise ValueError("log_response_token_probs requires trainer.rollout_data_dir")\n'
                '            if not self.config.algorithm.rollout_correction.get("bypass_mode", False):\n'
                '                raise ValueError("Generator comparison logging requires bypass_mode=True")\n'
                "            token_log_ids = TensorDict({}, batch_size=[len(batch)])\n"
                '            tu.assign_non_tensor_stack(token_log_ids, "token_log_prob_uid", list(batch.keys))\n'
                "            dump_meta = tq.kv_batch_put(keys=batch.keys, "
                "partition_id=batch.partition_id, fields=token_log_ids)\n"
                "            batch.fields = dump_meta.fields\n"
                "            batch.extra_info.update(\n"
                '                token_log_prob_dir=os.path.join(dump_dir, "token_log_probs"),\n'
                "                token_log_prob_step=self.global_steps,\n"
                "            )\n",
            ),
            (
                "            # 5. cleanup transfer queue\n",
                '            if self.config.trainer.get("log_response_token_probs", False):\n'
                "                probability_data = tq.kv_batch_get(\n"
                "                    keys=batch.keys, partition_id=batch.partition_id,\n"
                '                    select_fields=["responses", "response_mask", "rollout_log_probs"],\n'
                "                )\n"
                "                dump_validation_token_log_probs(\n"
                "                    probability_data, batch.keys, "
                "self.config.trainer.validation_data_dir, self.global_steps\n"
                "                )\n",
            ),
        ],
    }
    prepared = {}
    audit = []
    for name, entries in patches.items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Refusing staged source link: {name}")
        original = path.read_text()
        updated = original
        for anchor, addition in entries:
            if updated.count(anchor) != 1 or addition in updated:
                raise ValueError(f"Logging insertion anchor changed or already installed: {name}, {anchor!r}")
            updated = updated.replace(anchor, anchor + addition, 1)
        ast.parse(updated, filename=name)
        reversed_source = updated
        for anchor, addition in reversed(entries):
            reversed_source = reversed_source.replace(anchor + addition, anchor, 1)
        if reversed_source != original:
            raise ValueError(f"Logging installation modified original computation: {name}")
        prepared[path] = updated
        audit.append(
            {
                "path": name,
                "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "staged_sha256": hashlib.sha256(updated.encode()).hexdigest(),
                "original_computation_lines_preserved": True,
            }
        )
    for path, updated in prepared.items():
        path.write_text(updated)
    destination = root / "recipe_response_logging.py"
    if destination.resolve() != Path(__file__).resolve():
        shutil.copy2(__file__, destination)
    return {
        "additive_detached_response_logging": True,
        "modified_files": audit,
        "artifact_directories": ["rollout/token_log_probs", "validation/token_log_probs"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(install_logging(args.install), indent=2))


if __name__ == "__main__":
    main()
'''
    ),
    "recipe_reward_score.py": (
        r'''# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Uniform reward metadata for mixed validation, preserving native scores."""

from verl.utils.reward_score import default_compute_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source == "nvidia/OpenMathInstruct-2":
        from verl.utils.reward_score import math_reward

        result = math_reward.compute_score(solution_str, ground_truth)
    else:
        result = default_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    if isinstance(result, dict):
        return {"score": result["score"], "acc": result["acc"], "pred": result.get("pred", "")}
    return {"score": result, "acc": result, "pred": ""}
'''
    ),
    "recipe_validation_dataset.py": (
        r'''# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Keep the training prompt limit while admitting complete benchmark validation."""

from omegaconf import OmegaConf

from verl.utils.dataset.rl_dataset import RLHFDataset


def _paths(value):
    return [value] if isinstance(value, str) else list(value)


class FullValidationDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, config, processor=None, max_samples=-1):
        is_validation = _paths(data_files) == _paths(config.val_files)
        if is_validation:
            config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
            config.max_prompt_length = 1024
            expected_rows = int(config.get("validation_expected_rows", 2819))
            if expected_rows < 1:
                raise ValueError("validation_expected_rows must be positive")
            max_samples = -1
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )
        if is_validation and len(self) != expected_rows:
            raise ValueError(f"Expected all {expected_rows} validation problems, loaded {len(self)}")
'''
    ),
    "run_qwen3_4b_torchtitan.sh": (
        r"""#!/usr/bin/env bash
# GRPO | Qwen3-4B | GSM8K | TorchTitan Training (Full FSDP) & vLLM Rollout | TPU v6e-8 x2 Slices
# V1 PPOTrainer (Separate Async Overlap) with verl-hardware-plugin

set -xeuo pipefail

export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
export VERL_PLATFORM=tpu
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99

# JAX/XLA Launch Barrier Configuration
export LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false"

SMOKE_TEST="${SMOKE_TEST:-0}"

if [[ "${SMOKE_TEST}" == "1" ]]; then
    exp_name="${EXPERIMENT_NAME:-qwen3_4b_fast_smoke_test}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
    VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-8}"
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
    ROLLOUT_N="${ROLLOUT_N:-4}"
    MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-768}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-5}"
    TEST_FREQ="${TEST_FREQ:-2}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
else
    exp_name="${EXPERIMENT_NAME:-qwen3_4b_gsm8k_fsdp}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
    VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-128}"
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
    ROLLOUT_N="${ROLLOUT_N:-8}"
    MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
    TEST_FREQ="${TEST_FREQ:-10}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
fi

# Project details
project_name="${WANDB_PROJECT:-verl_tpu_grpo}"

# Paths
RAY_DATA_HOME="/data/jialei"
MODEL_PATH="${MODEL_PATH:-${RAY_DATA_HOME}/assets/hf/Qwen3-4B}"

TRAIN_FILE="${RAY_DATA_HOME}/data/gsm8k/train.parquet"
TEST_FILE="${RAY_DATA_HOME}/data/gsm8k/test.parquet"

# TPU 2-slice v6e-8 configurations
export NNODES_TRAINER=2       # 2 physical VM hosts for training slice
export N_CHIPS_TRAINER=4      # 4 TPU chips per training host

export NNODES_ROLLOUT=2       # 2 physical VM hosts for rollout slice
export N_CHIPS_ROLLOUT=4      # 4 TPU chips per rollout host

TOTAL_ROLLOUT_CHIPS=$((NNODES_ROLLOUT * N_CHIPS_ROLLOUT))
TOTAL_TRAINER_CHIPS=$((NNODES_TRAINER * N_CHIPS_TRAINER))

# Sequence budget
MAX_PROMPT_LEN=512
MAX_MODEL_LEN=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))

# Actor parallelism (default: Full FSDP across all 8 trainer chips: tp=1, dp_shard=8)
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DEFAULT_DP_SHARD=$((TOTAL_TRAINER_CHIPS / TENSOR_PARALLEL_SIZE))
DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-${DEFAULT_DP_SHARD}}"

# Off-policy correction (truncated importance sampling). Required, not optional: this pipeline
# runs verl's decoupled regime with one batch in flight, so rollouts come from theta_{t-1} while
# old_log_probs are recomputed under theta_t. Without IS weights the reward peaks around step 50
# and then collapses (grad_norm 0.4 -> 363, rollout/training logprob correlation 0.98 -> 0.28).
# See run_qwen3_0_6b_torchtitan.sh for the full derivation.
ROLLOUT_IS="${ROLLOUT_IS:-token}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"

# The hardware plugin provides TPU TorchTitan execution and the TPU checkpoint engine.
# The submission wrapper supplies the 250-step Base-model recipe and 64-chip geometry.

python3 -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches=1 \
    trainer.v1.separate_async.parameter_sync_step=1 \
    transfer_queue.enable=True \
    model_engine=torchtitan \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is="${ROLLOUT_IS}" \
    algorithm.rollout_correction.rollout_is_threshold="${ROLLOUT_IS_THRESHOLD}" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.val_max_samples="${VAL_MAX_SAMPLES}" \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESPONSE_LEN}" \
    +data.max_length=4096 \
    +data.max_token_len_per_gpu=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.pad_mode=no_padding \
    actor_rollout_ref.actor.strategy=torchtitan \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.torchtitan.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${DATA_PARALLEL_SHARD_SIZE}" \
    actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1 \
    actor_rollout_ref.actor.torchtitan.attn_type=varlen \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.no_enable_prefix_caching=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TOTAL_ROLLOUT_CHIPS}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=tpu \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.logger="['console','tensorboard','file','wandb']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.log_val_generations=4 \
    trainer.rollout_data_dir=/tmp/verl_dump/rollout \
    trainer.validation_data_dir=/tmp/verl_dump/validation \
    trainer.save_freq=-1 \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs=10 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.nnodes="${NNODES_TRAINER}" \
    trainer.n_gpus_per_node="${N_CHIPS_TRAINER}" \
    actor_rollout_ref.rollout.nnodes="${NNODES_ROLLOUT}" \
    actor_rollout_ref.rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" \
    +rollout.nnodes="${NNODES_ROLLOUT}" \
    +rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" "$@"
"""
    ),
    "shared_qwen3_4b_base_model.py": (
        r'''#!/usr/bin/env python3
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Populate and reuse immutable Qwen3-4B-Base objects in Cloud Storage.

The ready manifest is created only after all eleven model objects have been
verified. Existing objects are never overwritten. Authentication stays in the
submission VM's existing CloudStorage client; no credentials enter manifests.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from urllib.parse import quote, urlencode

from prepare_qwen3_4b_base_model import DEFAULT_OUTPUT_DIR, MODEL_FILES, MODEL_REPO, MODEL_REVISION

API = "https://storage.googleapis.com/storage/v1"
UPLOAD_API = "https://storage.googleapis.com/upload/storage/v1"
MODEL_CACHE_PREFIX = f"gs://lixali-tpu-storage/models/Qwen3-4B-Base/{MODEL_REVISION}"
CHUNK_BYTES = 128 * 1024**2
HASH_CHUNK_BYTES = 8 * 1024**2
MANIFEST_NAME = "model-cache-manifest.json"


def _status(error):
    return getattr(error, "status", None)


def _split(uri):
    if not uri.startswith("gs://") or "?" in uri or "#" in uri:
        raise ValueError("Model cache paths must be gs://bucket/object URIs")
    bucket, separator, name = uri[5:].partition("/")
    if not bucket or not separator or not name or ".." in name.split("/"):
        raise ValueError("Model cache paths must include an object name")
    return bucket, name


def _stat_or_none(cloud, uri):
    try:
        return cloud.stat(uri)
    except Exception as error:
        if _status(error) == 404:
            return None
        raise


def _verify_object(meta, expected, uri):
    if meta is None:
        raise RuntimeError(f"Missing model object: {uri}")
    if (
        int(meta.get("size", -1)) != expected["bytes"]
        or not meta.get("md5Hash")
        or meta["md5Hash"] != expected["md5Hash"]
    ):
        raise RuntimeError(f"Model object size/MD5 mismatch; refusing overwrite: {uri}")
    if not meta.get("generation") or int(meta["generation"]) < 1:
        raise RuntimeError(f"Model object lacks its immutable generation: {uri}")
    return meta


def _local_records(files):
    records = {}
    for name, original in MODEL_FILES.items():
        path = Path(files[name])
        before = path.stat()
        if before.st_size != original["size_bytes"]:
            raise RuntimeError(f"Local model size mismatch: {path}")
        sha, md5 = hashlib.sha256(), hashlib.md5()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
                sha.update(chunk)
                md5.update(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha.hexdigest() != original[
            "sha256"
        ]:
            raise RuntimeError(f"Local model SHA256 mismatch or file changed: {path}")
        records[name] = {
            "name": name,
            "path": path,
            "bytes": before.st_size,
            "sha256": original["sha256"],
            "md5Hash": base64.b64encode(md5.digest()).decode(),
        }
    return records


def _upload_create_only(cloud, path, destination, expected):
    bucket, name = _split(destination)
    metadata = {"model_repo": MODEL_REPO, "model_revision": MODEL_REVISION, "model_sha256": expected["sha256"]}
    if expected["bytes"] <= HASH_CHUNK_BYTES:
        try:
            meta = cloud.uploadbytes(Path(path).read_bytes(), destination, if_generation_match=0, metadata=metadata)
        except Exception as error:
            if _status(error) != 412:
                raise
            meta = cloud.stat(destination)
        return _verify_object(meta, expected, destination)
    before = Path(path).stat()
    url = f"{UPLOAD_API}/b/{quote(bucket, safe='')}/o?" + urlencode(
        {"uploadType": "resumable", "name": name, "ifGenerationMatch": "0"}
    )
    try:
        _, headers, _ = cloud._perform(
            url,
            method="POST",
            data=json.dumps({"metadata": metadata}).encode(),
            headers={
                "Content-Type": "application/json",
                "X-Upload-Content-Length": str(expected["bytes"]),
                "X-Upload-Content-Type": "application/octet-stream",
            },
        )
        session = next(value for key, value in headers.items() if key.lower() == "location")
        offset, uploaded = 0, None
        with Path(path).open("rb") as stream:
            while offset < expected["bytes"]:
                chunk = stream.read(min(CHUNK_BYTES, expected["bytes"] - offset))
                if not chunk:
                    raise RuntimeError("Local model changed during upload")
                end = offset + len(chunk) - 1
                body, response_headers, status = cloud._perform(
                    session,
                    method="PUT",
                    data=chunk,
                    headers={
                        "Content-Type": "application/octet-stream",
                        "Content-Range": f"bytes {offset}-{end}/{expected['bytes']}",
                    },
                )
                if status == 308:
                    accepted = next((value for key, value in response_headers.items() if key.lower() == "range"), "")
                    if accepted != f"bytes=0-{end}":
                        raise RuntimeError("Model upload returned an unexpected accepted offset")
                else:
                    uploaded = json.loads(body)
                offset = end + 1
                print(f"Uploading {Path(path).name}: {offset:,}/{expected['bytes']:,} bytes", flush=True)
        if uploaded is None:
            raise RuntimeError("Model upload did not finalize")
    except Exception as error:
        if _status(error) != 412:
            raise
        uploaded = cloud.stat(destination)
    after = Path(path).stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Local model changed during upload")
    return _verify_object(uploaded, expected, destination)


def _copy_create_only(cloud, source, destination, source_meta, expected):
    source_bucket, source_name = _split(source)
    destination_bucket, destination_name = _split(destination)
    _verify_object(source_meta, expected, source)
    url = (
        f"{API}/b/{quote(source_bucket, safe='')}/o/{quote(source_name, safe='')}/rewriteTo/"
        f"b/{quote(destination_bucket, safe='')}/o/{quote(destination_name, safe='')}"
    )
    query = {
        "sourceGeneration": source_meta["generation"],
        "ifSourceGenerationMatch": source_meta["generation"],
        "ifGenerationMatch": "0",
    }
    metadata = {
        **source_meta.get("metadata", {}),
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_sha256": expected["sha256"],
    }
    try:
        while True:
            result = cloud.request(url + "?" + urlencode(query), method="POST", data={"metadata": metadata})
            if result.get("done"):
                return _verify_object(result["resource"], expected, destination)
            if not result.get("rewriteToken"):
                raise RuntimeError("Model copy lacks its continuation token")
            query["rewriteToken"] = result["rewriteToken"]
    except Exception as error:
        if _status(error) != 412:
            raise
        return _verify_object(cloud.stat(destination), expected, destination)


def _manifest(prefix, entries):
    return {
        "format_version": 1,
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "prefix": prefix,
        "complete": True,
        "files": entries,
    }


def _verify_manifest(cloud, value, prefix):
    if (
        value.get("format_version") != 1
        or value.get("repo") != MODEL_REPO
        or value.get("revision") != MODEL_REVISION
        or value.get("prefix") != prefix
        or value.get("complete") is not True
    ):
        raise RuntimeError("Existing model cache ready manifest has incompatible provenance")
    entries = value.get("files", [])
    if len(entries) != len(MODEL_FILES) or {entry.get("name") for entry in entries} != set(MODEL_FILES):
        raise RuntimeError("Existing model cache manifest has incomplete or duplicate files")
    for entry in entries:
        name = entry["name"]
        uri = prefix + "/" + name
        if (
            entry.get("source") != uri
            or entry.get("bytes") != MODEL_FILES[name]["size_bytes"]
            or entry.get("sha256") != MODEL_FILES[name]["sha256"]
        ):
            raise RuntimeError(f"Existing model manifest differs from pinned model: {name}")
        _verify_object(cloud.stat(uri, generation=entry["generation"]), entry, uri)
        if cloud.stat(uri)["generation"] != entry["generation"]:
            raise RuntimeError(f"Immutable cache object was replaced: {uri}")
    return value


def _read_ready(cloud, prefix):
    uri = prefix + "/" + MANIFEST_NAME
    meta = _stat_or_none(cloud, uri)
    if meta is None:
        return None
    value = _verify_manifest(cloud, cloud.get_json(uri, generation=meta["generation"]), prefix)
    return {**value, "manifest_uri": uri, "manifest_generation": meta["generation"]}


def _publish_ready(cloud, prefix, entries):
    uri = prefix + "/" + MANIFEST_NAME
    value = _manifest(prefix, entries)
    try:
        meta = cloud.uploadbytes(
            json.dumps(value, indent=2, sort_keys=True).encode(),
            uri,
            content_type="application/json",
            if_generation_match=0,
        )
    except Exception as error:
        if _status(error) != 412:
            raise
        found = _read_ready(cloud, prefix)
        if _manifest(prefix, found["files"]) != value:
            raise RuntimeError("Concurrent model cache manifest differs from verified objects") from error
        return found
    return {**value, "manifest_uri": uri, "manifest_generation": meta["generation"]}


def ensure_model_cache(cloud, files=None, *, source_prefixes=(), prefix=MODEL_CACHE_PREFIX, reuse_only=False):
    """Return verified canonical entries, copying completed uploads when available.

    A complete cache is checked without reading local model bytes. An incomplete
    cache requires the already-downloaded local files, whose hashes are checked
    in one streaming pass. This helper never downloads weights from Hugging Face.
    """
    prefix = prefix.rstrip("/")
    _split(prefix + "/probe")
    ready = _read_ready(cloud, prefix)
    if ready is not None:
        print("Reusing verified model cache: " + prefix, flush=True)
        return ready
    if reuse_only:
        raise RuntimeError(
            f"The shared model cache is not complete: {prefix}. Prepare it before submitting this recipe."
        )
    files = files or {name: DEFAULT_OUTPUT_DIR / name for name in MODEL_FILES}
    records = _local_records(files)
    entries = []
    for name, expected in records.items():
        uri = prefix + "/" + name
        meta = _stat_or_none(cloud, uri)
        if meta is not None:
            _verify_object(meta, expected, uri)
            print("Reusing verified object: " + name, flush=True)
        else:
            for source_prefix in source_prefixes:
                source = source_prefix.rstrip("/") + "/" + name
                candidate = _stat_or_none(cloud, source)
                if candidate is not None:
                    print("Copying completed cloud object: " + name, flush=True)
                    meta = _copy_create_only(cloud, source, uri, candidate, expected)
                    break
            if meta is None:
                print("Uploading missing model object: " + name, flush=True)
                meta = _upload_create_only(cloud, expected["path"], uri, expected)
        entries.append(
            {
                "name": name,
                "source": uri,
                "generation": meta["generation"],
                "bytes": expected["bytes"],
                "sha256": expected["sha256"],
                "md5Hash": expected["md5Hash"],
            }
        )
    return _publish_ready(cloud, prefix, entries)


def ensure_worker_mirror(cloud, cache, prefix):
    """Create a shared readable mirror once, using only server-side cloud copies."""
    prefix = prefix.rstrip("/")
    ready = _read_ready(cloud, prefix)
    if ready is not None:
        return ready
    entries = []
    for entry in cache["files"]:
        uri = prefix + "/" + entry["name"]
        meta = _stat_or_none(cloud, uri)
        if meta is None:
            source_meta = cloud.stat(entry["source"], generation=entry["generation"])
            print("Creating shared worker model copy: " + entry["name"], flush=True)
            meta = _copy_create_only(cloud, entry["source"], uri, source_meta, entry)
        else:
            _verify_object(meta, entry, uri)
        entries.append({**entry, "source": uri, "generation": meta["generation"]})
    return _publish_ready(cloud, prefix, entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-prefix",
        action="append",
        default=[],
        help="Existing completed model-object prefix to promote without reuploading.",
    )
    parser.add_argument("--prefix", default=MODEL_CACHE_PREFIX)
    parser.add_argument("--worker-prefix", help="Optional shared worker-readable mirror prefix.")
    parser.add_argument("--verification-json", type=Path)
    parser.add_argument("--verify-only", action="store_true", help="Read-only cache verification.")
    args = parser.parse_args()
    from artifact_mirror import CloudStorage

    cloud = CloudStorage()
    if args.verify_only:
        cache = _read_ready(cloud, args.prefix.rstrip("/"))
        if cache is None:
            raise SystemExit("No complete model cache manifest exists.")
    else:
        files = {name: args.local_dir / name for name in MODEL_FILES}
        cache = ensure_model_cache(cloud, files, source_prefixes=args.source_prefix, prefix=args.prefix)
    proof = {"canonical": cache}
    if args.worker_prefix:
        mirror = (
            _read_ready(cloud, args.worker_prefix.rstrip("/"))
            if args.verify_only
            else ensure_worker_mirror(cloud, cache, args.worker_prefix)
        )
        if mirror is None:
            raise SystemExit("No complete worker mirror manifest exists.")
        proof["worker_mirror"] = mirror
    if args.verification_json:
        args.verification_json.write_text(json.dumps(proof, indent=2) + "\n")
    print(
        json.dumps(
            {
                "model": MODEL_REPO,
                "revision": MODEL_REVISION,
                "canonical": cache["prefix"],
                "files": len(cache["files"]),
                "bytes": sum(entry["bytes"] for entry in cache["files"]),
                "worker_mirror": proof.get("worker_mirror", {}).get("prefix"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
'''
    ),
    "source_runtime_check.py": (
        r'''#!/usr/bin/env python3
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Check the staged TPU recipe's imports and tokenizer before hardware preflight.

Run inside the pinned Ray head container after inputs/source are downloaded.
Checks CPU imports, public API availability and a local tokenizer/config only:
no Ray cluster connection, model loading, TPU allocation or package upgrades.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True


def run(args):
    source = args.source_dir.resolve()
    if not (source / "verl/trainer/main_ppo.py").is_file():
        raise RuntimeError(f"Missing staged VERL source: {source}")
    if not (source / "verl_hardware_plugin/__init__.py").is_file():
        raise RuntimeError(f"Missing staged hardware plugin: {source}")
    sys.path.insert(0, str(source))
    if Path("/torchtitan").is_dir():
        sys.path.insert(1, "/torchtitan")
    os.environ.setdefault("VERL_PLATFORM", "tpu")
    os.environ["VERL_USE_EXTERNAL_MODULES"] = "verl_hardware_plugin"
    report = {
        "passed": False,
        "source": str(source),
        "python": sys.version,
        "checks": [],
        "package_versions": {},
        "hardware_allocated": False,
    }
    for package in (
        "torch",
        "torch-tpu",
        "torch_tpu",
        "torchtitan",
        "vllm",
        "vllm-torchtpu",
        "transformers",
        "ray",
        "tensordict",
        "pyarrow",
        "math-verify",
        "mathruler",
        "TransferQueue",
        "verl-hardware-plugin",
    ):
        try:
            report["package_versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["package_versions"][package] = None

    def check(name, operation):
        print(f"[source-runtime-check] {name}", flush=True)
        value = operation()
        report["checks"].append({"name": name, "passed": True})
        return value

    try:
        # Import the actual source used by the recipe; version numbers alone
        # cannot guarantee image compatibility with the user's current tree.
        from recipe_tpu_plugin_layout import tpu_plugin_layout

        layout = tpu_plugin_layout(source)
        precision_modules = (layout["precision_module"],) if (source / layout["precision"]).is_file() else ()
        for name in (
            "verl.trainer.main_ppo",
            layout["engine_module"],
            layout["rollout_module"],
            layout["checkpoint_module"],
            layout["platform_module"],
            "verl.utils.dataset.rl_dataset",
            layout["patches_module"],
            *precision_modules,
            "verl.utils.reward_score.math_reward",
            "verl.utils.reward_score.math_dapo",
        ):
            imported = check("import " + name, lambda name=name: importlib.import_module(name))
            if name.startswith(("verl.", "verl_hardware_plugin.")):
                imported_path = Path(imported.__file__).resolve()
                if not imported_path.is_relative_to(source):
                    raise RuntimeError(f"Imported {name} from {imported_path}, expected {source}")
        engine = importlib.import_module("verl.checkpoint_engine")
        check("registered TPU weight-sync backend", lambda: engine.CheckpointEngineRegistry.get("tpu"))

        from transformers import AutoConfig, AutoTokenizer

        model_path = args.model_path.resolve()
        config = check(
            "local Qwen3-4B-Base model config",
            lambda: AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False),
        )
        if config.model_type != "qwen3":
            raise RuntimeError(f"Expected Qwen3 model, got model_type={config.model_type}")
        tokenizer = check(
            "local tokenizer",
            lambda: AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False),
        )
        if args.chat_template_file:
            tokenizer.chat_template = args.chat_template_file.read_text()
        if not tokenizer.chat_template:
            raise RuntimeError(
                "The Base tokenizer has no chat_template. Supply the recipe's explicit "
                "template with --chat-template-file; do not borrow different model weights."
            )
        check(
            "chat template and tokenize a prompt",
            lambda: tokenizer.apply_chat_template(
                [{"role": "user", "content": "What is 2 + 2?"}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            ),
        )
        report["model"] = {
            "path": str(model_path),
            "type": config.model_type,
            "architectures": config.architectures,
            "tokenizer": type(tokenizer).__name__,
        }
        if args.validation_file:
            import pyarrow.parquet as pq

            report["validation"] = []
            for file in args.validation_file:
                metadata = check("validation parquet " + str(file), lambda file=file: pq.read_metadata(file))
                if not {"prompt", "reward_model", "data_source"}.issubset(metadata.schema.names):
                    # Nested Arrow schema may flatten the top-level fields in
                    # Parquet; check its Arrow reconstruction instead.
                    arrow_names = set(pq.read_schema(file).names)
                    if not {"prompt", "reward_model", "data_source"}.issubset(arrow_names):
                        raise RuntimeError(f"Missing VERL validation columns: {file}")
                report["validation"].append({"path": str(file), "rows": metadata.num_rows})
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("/opt/run/source"))
    parser.add_argument("--model-path", type=Path, default=Path("/data/jialei/assets/hf/Qwen3-4B-Base"))
    parser.add_argument("--chat-template-file", type=Path)
    parser.add_argument("--validation-file", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
'''
    ),
}

_EXECUTABLES = {
    "run_qwen3_4b_torchtitan.sh",
}


def materialize(destination: str | Path) -> Path:
    """Write inspectable helper files into a new directory outside this repository.

    Existing directories are rejected so a prepared run cannot be overwritten.
    The returned path is absolute and contains the launcher helpers and training
    recipe; callers supply the run configuration separately.
    """
    destination = Path(destination).expanduser().resolve()
    source_repository = Path(__file__).resolve().parents[4]
    if destination.is_relative_to(source_repository):
        raise ValueError("Runtime helpers must be materialized outside the source repository")
    if destination.exists():
        raise FileExistsError(f"Runtime destination already exists: {destination}")
    for name in TEMPLATES:
        relative = Path(name)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(f"Runtime template must have a safe relative path: {name}")
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in TEMPLATES.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode("utf-8"))
        if name in _EXECUTABLES:
            target.chmod(0o755)
    return destination
