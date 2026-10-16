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
"""CPU tests for the VERL_TIMELINE prints (verl/utils/timeline.py) and the log reader in scripts/rl_timeline."""

import importlib.util
import json
import pathlib
import re

from verl.utils import timeline

_REPORT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "rl_timeline" / "make_report.py"
# Same canonicalisation as ray._private.ray_logging (words with digits are dropped before lines are compared).
_RAY_NUMBERS = re.compile(r"(\d+|0x[0-9a-fA-F]+)")


def _events(text: str) -> list[dict]:
    return [json.loads(line[line.find("{") :]) for line in text.splitlines() if timeline.MARKER in line]


def _load_make_report():
    spec = importlib.util.spec_from_file_location("make_report", _REPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_disabled_prints_nothing(monkeypatch, capsys):
    monkeypatch.delenv("VERL_TIMELINE", raising=False)
    timeline.emit("gen", 1000.0, 1.0)
    monkeypatch.setenv("VERL_TIMELINE", "0")
    timeline.emit("gen", 1000.0, 1.0)
    assert capsys.readouterr().out == ""


def test_line_is_one_dedup_safe_token(monkeypatch, capsys):
    monkeypatch.setenv("VERL_TIMELINE", "1")
    timeline.emit("update weights", 1000.0, 2.5, group="rollout_replica_1", lane="server", note="a b\tc")
    (line,) = capsys.readouterr().out.splitlines()
    assert line.startswith(timeline.MARKER)
    # One whitespace-free token with digits: Ray never folds such lines into "[repeated Nx across cluster]".
    assert len(line.split()) == 1
    assert " ".join(x for x in line.split() if not _RAY_NUMBERS.search(x)) == ""
    (ev,) = _events(line)
    assert ev == {
        "name": "update weights",
        "group": "rollout_replica_1",
        "lane": "server",
        "ts": 1000.0,
        "dur": 2.5,
        "args": {"note": "a b\tc"},
    }


def test_marked_timer_prints_a_trainer_span(monkeypatch, capsys):
    from verl.utils.profiler.performance import marked_timer, simple_timer

    monkeypatch.setenv("VERL_TIMELINE", "1")
    timing_raw = {}
    with marked_timer("old_log_prob", timing_raw):
        pass
    with simple_timer("generate_sequences", timing_raw):  # per-sample timer in agent loops: no prints
        pass
    (ev,) = _events(capsys.readouterr().out)
    assert (ev["name"], ev["group"], ev["lane"]) == ("old_log_prob", "trainer", "loop")
    assert abs(ev["dur"] - timing_raw["old_log_prob"]) < 0.01
    assert set(timing_raw) == {"old_log_prob", "generate_sequences"}


def test_report_reads_ray_job_log(monkeypatch, capsys):
    """make_report parses the lines as Ray forwards them and numbers trainer spans by the next "step:N" line."""
    monkeypatch.setenv("VERL_TIMELINE", "1")
    timeline.emit("testing", 990.0, 5.0)
    timeline.emit("gen", 1000.0, 2.5)
    timeline.emit("request", 1000.5, 1.0, group="rollout_replica_0", lane="requests", out=7)
    timeline.emit("step", 1000.0, 9.0)
    testing, gen, request, step = capsys.readouterr().out.splitlines()
    trainer, server = "\x1b[36m(TaskRunner pid=1)\x1b[0m ", "\x1b[36m(vLLMHttpServer pid=2, ip=10.0.0.2)\x1b[0m "
    log = [
        trainer + testing,
        trainer + "step:0 - val-core/gsm8k/acc/mean@1:0.34",  # validation before training: no timing_s
        trainer + gen,
        server + request,
        trainer + step,
        trainer + "step:1 - timing_s/gen:2.5 - timing_s/step:9.0 - critic/rewards/mean:0.25",
    ]

    make_report = _load_make_report()
    events, metrics, bad = make_report.parse_log_lines(log)
    assert bad == 0
    assert [(e["name"], e["group"], e.get("step")) for e in events] == [
        ("testing", "trainer", 0),
        ("gen", "trainer", 1),
        ("request", "rollout_replica_0", None),
        ("step", "trainer", 1),
    ]
    assert metrics == {1: {"timing_s/gen": 2.5, "timing_s/step": 9.0, "critic/rewards/mean": 0.25}}

    html = make_report.build_html(_REPORT.with_name("index.html").read_text(), {"events": events, "metrics": metrics})
    # The empty placeholder is filled exactly once (the dashboard's own JS mentions the tag in a regex).
    assert html.count('<script id="tl-embedded" type="application/json">{') == 1
    assert '<script id="tl-embedded" type="application/json"></script>' not in html
