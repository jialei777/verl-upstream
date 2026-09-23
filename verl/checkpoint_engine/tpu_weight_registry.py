# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio

import ray

"""Lightweight CPU-only Ray actor registry for tracking TPU weight checkpoint references."""


@ray.remote(num_cpus=0)
class TPUWeightRegistry:
    """Ray actor for holding references to synchronized TPU model weights across steps."""

    def __init__(self):
        self.weights = {}
        self.buckets = {}
        self.bucket_ready_events = {}
        self.bucket_ack_counts = {}
        self.bucket_ack_events = {}
        self.bucket_num_receivers = {}

    def _get_ready_event(self, key: tuple[int, int]) -> asyncio.Event:
        if key not in self.bucket_ready_events:
            self.bucket_ready_events[key] = asyncio.Event()
        return self.bucket_ready_events[key]

    def _get_ack_event(self, key: tuple[int, int]) -> asyncio.Event:
        if key not in self.bucket_ack_events:
            self.bucket_ack_events[key] = asyncio.Event()
        return self.bucket_ack_events[key]

    async def set_bucket(
        self,
        step: int,
        bucket_idx: int,
        bucket_ref_list: list,
        bucket_meta: dict,
        is_last: bool,
        num_receivers: int = 1,
    ):
        key = (step, bucket_idx)
        self.buckets[key] = (bucket_ref_list, bucket_meta, is_last)
        self.bucket_num_receivers[key] = max(1, num_receivers)
        if self.bucket_ack_counts.get(key, 0) >= self.bucket_num_receivers[key]:
            self._get_ack_event(key).set()
        self._get_ready_event(key).set()

    async def get_bucket(self, step: int, bucket_idx: int):
        key = (step, bucket_idx)
        await self._get_ready_event(key).wait()
        return self.buckets.get(key, None)

    async def ack_bucket(self, step: int, bucket_idx: int):
        key = (step, bucket_idx)
        count = self.bucket_ack_counts.get(key, 0) + 1
        self.bucket_ack_counts[key] = count
        target = self.bucket_num_receivers.get(key, 1)
        if count >= target:
            self.buckets.pop(key, None)
            self._get_ack_event(key).set()

    async def wait_bucket_acks(self, step: int, bucket_idx: int):
        key = (step, bucket_idx)
        await self._get_ack_event(key).wait()
        self.buckets.pop(key, None)
        self.bucket_ready_events.pop(key, None)
        self.bucket_ack_events.pop(key, None)
        self.bucket_ack_counts.pop(key, None)
        self.bucket_num_receivers.pop(key, None)

    def clear_bucket(self, bucket_idx: int):
        self.buckets.pop(bucket_idx, None)

    def set_weights(self, step, ref):
        self.weights[step] = ref
        for old_step in [s for s in self.weights if s != step]:
            old_ref = self.weights[old_step]
            if isinstance(old_ref, str):
                try:
                    import os

                    if os.path.exists(old_ref):
                        os.remove(old_ref)
                except Exception:
                    pass
            del self.weights[old_step]

    def get_weights(self, step):
        return self.weights.get(step, None)

    def clear(self):
        """Drops every cached entry. Used to reset state left by a previous job."""
        self.weights.clear()
        self.buckets.clear()
        self.bucket_ready_events.clear()
        self.bucket_ack_counts.clear()
        self.bucket_ack_events.clear()
        self.bucket_num_receivers.clear()
