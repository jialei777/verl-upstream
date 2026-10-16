# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Google LLC
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
"""Timeline prints for RL performance analysis.

With ``VERL_TIMELINE=1`` in the job's runtime env, the trainer loop (``marked_timer``) and the vLLM rollout servers
print one line per finished span::

    VERL_TIMELINE{"name":"update_actor","group":"trainer","lane":"loop","ts":1791857954.85,"dur":3.87}

Ray forwards these lines into the job log; ``scripts/rl_timeline/index.html`` reads the log and draws the timeline.
"""

import json
import os
import sys

MARKER = "VERL_TIMELINE"


def enabled() -> bool:
    """Whether ``VERL_TIMELINE`` is set to a true value in this process."""
    return os.environ.get("VERL_TIMELINE", "0").lower() not in ("", "0", "false", "no", "off")


def emit(name: str, ts: float, dur: float, group: str = "trainer", lane: str = "loop", **args) -> None:
    """Print one finished span that started at epoch time ``ts`` and lasted ``dur`` seconds."""
    if not enabled():
        return
    event = {"name": name, "group": group, "lane": lane, "ts": round(ts, 6), "dur": round(dur, 6)}
    if args:
        event["args"] = args
    # One whitespace-free token that contains digits: Ray's log deduplication ignores such words, so it never folds
    # the lines of different workers into "[repeated Nx across cluster]".
    sys.stdout.write(MARKER + json.dumps(event, separators=(",", ":")).replace(" ", "\\u0020") + "\n")
    sys.stdout.flush()
