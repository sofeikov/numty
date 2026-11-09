"""Trace (NUMTY_TRACE) and strict (NUMTY_STRICT) give-up visibility (issue #60)."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run_mypy(path: Path, cache: Path, extra_env: dict[str, str]) -> "subprocess.CompletedProcess[str]":
    env = {**os.environ, "PYTHONPATH": str(ROOT), **extra_env}
    return subprocess.run(
        [sys.executable, "-m", "mypy", "--cache-dir", str(cache), str(path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _load_trace(trace_path: Path, only_file: str) -> list[dict]:
    events = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Trace records absolute paths via ctx.api.path; match by suffix
        # so tmp dirs work.
        if str(ev.get("file", "")).endswith(only_file) or ev.get("file") == only_file:
            events.append(ev)
    return events


def test_trace_records_bail_and_unhooked(tmp_path: Path) -> None:
    src = tmp_path / "trace_repro.py"
    src.write_text(
        "import numpy as np\n"
        "a = np.zeros((2, 3))\n"
        "b = np.zeros((4, 5))\n"
        "reveal_type(np.kron(a, b))\n"
        "reveal_type(np.sqrt(8))\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    trace = tmp_path / "trace.jsonl"
    result = _run_mypy(src, cache, {"NUMTY_TRACE": str(trace)})
    assert result.returncode == 0, result.stdout
    assert trace.exists(), result.stdout
    events = _load_trace(trace, src.name)
    assert events, f"no trace events for {src.name}: {trace.read_text()[:2000]}"
    by_op = [(e["op"], e["outcome"], e["line"]) for e in events]
    # kron is unhooked (no solver), sqrt(8) bails (non-array operand).
    assert any(
        "kron" in op and outcome == "unhooked" for op, outcome, _ in by_op
    ), by_op
    assert any(
        outcome == "bail" for _, outcome, _ in by_op
    ), by_op
    # The unhooked kron line is the reveal line (line 4).
    assert any(
        "kron" in op and outcome == "unhooked" and line == 4
        for op, outcome, line in by_op
    ), by_op


def test_strict_flags_lost_shape(tmp_path: Path) -> None:
    src = tmp_path / "strict_repro.py"
    src.write_text(
        "from typing import TYPE_CHECKING\n"
        "import numpy as np\n"
        "from numty import F64\n"
        "if TYPE_CHECKING:\n"
        "    from typing import reveal_type\n"
        "else:\n"
        "    def reveal_type(x): return x\n"
        "def scores_scaled(q: F64[2, 5, 8], k: F64[2, 7, 8], mask: F64[5, 5]) -> F64[2, 5, 7]:\n"
        "    s = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(8)\n"
        "    reveal_type(s)\n"
        "    return s + mask\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    result = _run_mypy(src, cache, {"NUMTY_STRICT": "1"})
    assert result.returncode == 1, result.stdout
    assert "numty-unsolved" in result.stdout, result.stdout
    # The sqrt birth is on line 9 (the `s = ... / np.sqrt(8)` line).
    lines = [
        int(line.split(":")[1])
        for line in result.stdout.splitlines()
        if "numty-unsolved" in line
    ]
    assert 9 in lines, result.stdout


def test_no_strict_noise_by_default(tmp_path: Path) -> None:
    src = tmp_path / "quiet_repro.py"
    src.write_text(
        "import numpy as np\n"
        "reveal_type(np.sqrt(8))\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    result = _run_mypy(src, cache, {})
    assert result.returncode == 0, result.stdout
    assert "numty-unsolved" not in result.stdout
