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
"""Turn a Ray job log of a ``VERL_TIMELINE=1`` run into a self-contained timeline dashboard (one HTML file).

The events (``VERL_TIMELINE{...}`` lines, see ``verl/utils/timeline.py``) and the per-step ``timing_s/*`` metrics are
extracted from the log and embedded into a copy of ``index.html``, so the report opens offline and can be shared as a
single file (e.g. on x20). Logs without timeline lines still produce an approximate trainer-only view.

Examples::

    python3 scripts/rl_timeline/make_report.py job.log -o report.html
    # fetch the log from a Ray dashboard first (e.g. a kubectl port-forward of the head service):
    python3 scripts/rl_timeline/make_report.py --ray-address http://localhost:8265 --job <submission-id> -o report.html
    # print a per-step text summary as well:
    python3 scripts/rl_timeline/make_report.py job.log -o report.html --summary
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from typing import Optional

MARKER = "VERL_TIMELINE"
_STEP_RE = re.compile(r"(?:^|\s)step:(\d+)\s+-\s+(.*)$")
_EMBED_TAG = '<script id="tl-embedded" type="application/json"></script>'


def parse_log_lines(lines: Iterable[str]) -> tuple[list[dict], dict[int, dict[str, float]], int]:
    """Return (events, per-step metrics, number of unparsable timeline lines) from an iterable of log lines."""
    events, metrics, bad = [], defaultdict(dict), 0
    pending = []  # trainer-loop spans printed since the last "step:N" metrics line
    for line in lines:
        i = line.find(MARKER + "{")
        if i >= 0:
            j = line.rfind("}")
            try:
                event = json.loads(line[i + len(MARKER) : j + 1])
            except ValueError:
                bad += 1
                continue
            events.append(event)
            if event.get("group", "trainer") == "trainer" and event.get("step") is None:
                pending.append(event)
            continue
        k = line.find("step:")
        m = _STEP_RE.search(line[max(0, k - 1) :]) if k >= 0 else None
        if not m:
            continue
        step = int(m.group(1))
        for event in pending:  # the trainer logs a step's metrics right after the step's spans
            event["step"] = step
        pending = []
        if "timing_s/" not in line:  # e.g. the step-0 validation line
            continue
        rec = metrics[step]
        for part in m.group(2).split(" - "):
            key, sep, val = part.partition(":")
            if not sep:
                continue
            try:
                rec[key.strip()] = float(val)
            except ValueError:
                pass
    return events, dict(metrics), bad


def fetch_ray_job_log(address: str, job: str) -> str:
    url = f"{address.rstrip('/')}/api/jobs/{job}/logs"
    with urllib.request.urlopen(url, timeout=300) as r:
        return json.loads(r.read().decode("utf-8", "replace"))["logs"]


def build_html(template: str, payload: dict) -> str:
    if _EMBED_TAG not in template:
        raise ValueError("index.html has no empty tl-embedded <script> tag to fill")
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return template.replace(_EMBED_TAG, f'<script id="tl-embedded" type="application/json">{data}</script>', 1)


def text_summary(events: list[dict], metrics: dict[int, dict[str, float]]) -> str:
    """Per-step phase table from the trainer-loop spans (or the metrics when there are no events)."""
    out = []
    loop = [e for e in events if e.get("ph", "X") == "X" and e.get("group", "trainer") == "trainer"]
    if loop:
        by_step = defaultdict(lambda: defaultdict(float))
        for e in loop:
            if e.get("step") is not None and e["name"] != "step":
                by_step[e["step"]][e["name"]] += e["dur"]
        steps = {e["step"]: e for e in loop if e["name"] == "step"}
        cols = ["gen", "old_log_prob", "ref", "adv", "update_actor", "update_weights", "testing"]
        out.append("step  " + "".join(f"{c[:14]:>15}" for c in ["step"] + cols))
        for s in sorted(steps):
            row = [steps[s]["dur"]] + [by_step[s].get(c, 0.0) for c in cols]
            out.append(f"{s:>4}  " + "".join(f"{v:15.2f}" for v in row))
        reqs = [e for e in events if e.get("name") == "request"]
        if reqs:
            out.append(f"{len(reqs)} rollout requests, {sum(r['args'].get('out', 0) for r in reqs)} generated tokens")
    elif metrics:
        out.append("no VERL_TIMELINE lines; timing_s/step per step: ")
        out.append(", ".join(f"{k}:{v.get('timing_s/step', float('nan')):.1f}" for k, v in sorted(metrics.items())))
    return "\n".join(out)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", help="Ray job log file ('-' for stdin)")
    ap.add_argument("--ray-address", help="Ray dashboard address to fetch the log from, e.g. http://localhost:8265")
    ap.add_argument("--job", help="Ray job submission id (with --ray-address)")
    ap.add_argument("-o", "--output", required=True, help="output HTML file")
    ap.add_argument("--template", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"))
    ap.add_argument("--title", help="source name shown in the dashboard (default: log file name / job id)")
    ap.add_argument("--save-log", help="also write the fetched log here (with --ray-address)")
    ap.add_argument("--home", help="optional link target for an 'all runs' button (e.g. ../index.html)")
    ap.add_argument("--summary", action="store_true", help="print a per-step text summary")
    args = ap.parse_args(argv)

    if args.ray_address:
        if not args.job:
            ap.error("--job is required with --ray-address")
        text = fetch_ray_job_log(args.ray_address, args.job)
        if args.save_log:
            with open(args.save_log, "w") as f:
                f.write(text)
        lines, source = text.splitlines(), args.job
    elif args.log:
        fh = sys.stdin if args.log == "-" else open(args.log, errors="replace")
        lines, source = fh, os.path.basename(args.log)
    else:
        ap.error("give a log file or --ray-address/--job")

    events, metrics, bad = parse_log_lines(lines)
    with open(args.template) as f:
        template = f.read()
    payload = {"source": args.title or source, "events": events, "metrics": metrics}
    if args.home:
        payload["home"] = args.home
    html = build_html(template, payload)
    with open(args.output, "w") as f:
        f.write(html)
    n_steps = len({e.get("step") for e in events if e.get("name") == "step"}) or len(metrics)
    print(
        f"wrote {args.output}: {len(events)} timeline events, {n_steps} steps, {len(metrics)} metric rows"
        + (f", {bad} unparsable lines" if bad else ""),
        file=sys.stderr,
    )
    if args.summary:
        print(text_summary(events, metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
