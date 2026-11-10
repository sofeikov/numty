"""Exceptions inside the plugin are loud, never a silent pass (issue #164).

A hook that raises must not abort the run, but it must not look like a
clean result either: production reports [numty-internal] at the call
site and records a `crash` trace event, and NUMTY_DEBUG=1 (which
conftest.py sets for the whole suite) lets the exception propagate.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import numty_plugin

ROOT = Path(__file__).resolve().parent.parent

# A plugin that is numty with one solver helper broken, standing in for
# any latent bug: every `a + b` on arrays now raises inside its hook.
BROKEN_PLUGIN = """\
import numty_plugin
from numty_plugin import plugin


def _boom(*args, **kwargs):
    raise RuntimeError("seeded failure")


numty_plugin._broadcast_result = _boom
"""

SOURCE = """\
import numpy as np

a = np.zeros((2, 3)) + np.zeros((2, 3))
b = np.matmul(np.zeros((2, 3)), np.zeros((4, 5)))
"""


def _run(
    tmp_path: Path, debug: bool, config: str = ""
) -> tuple["subprocess.CompletedProcess[str]", Path]:
    (tmp_path / "broken_numty.py").write_text(BROKEN_PLUGIN, encoding="utf-8")
    (tmp_path / "mypy.ini").write_text(
        "[mypy]\nplugins = broken_numty\n" + config, encoding="utf-8"
    )
    (tmp_path / "repro.py").write_text(SOURCE, encoding="utf-8")
    trace = tmp_path / "trace.jsonl"
    env = {
        **os.environ,
        "PYTHONPATH": f"{tmp_path}{os.pathsep}{ROOT}",
        "NUMTY_TRACE": str(trace),
    }
    env.pop("NUMTY_DEBUG", None)
    if debug:
        env["NUMTY_DEBUG"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--cache-dir", str(tmp_path / "cache"), "repro.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result, trace


def test_hook_exception_is_reported_not_silent(tmp_path: Path) -> None:
    result, trace = _run(tmp_path, debug=False)
    out = result.stdout + result.stderr
    assert "INTERNAL ERROR" not in out, out
    internal = [line for line in result.stdout.splitlines() if "[numty-internal]" in line]
    # Both hooks on the `+` (signature and method) reach the broken helper.
    assert internal, out
    for line in internal:
        assert line.startswith("repro.py:3: error: numty internal error in "), out
        assert "RuntimeError: seeded failure" in line, out
        assert "result is unchecked" in line, out
    assert any("numpy.ndarray.__add__" in line for line in internal), out
    # The run carried on: the unrelated bug on the next line is still found.
    assert "repro.py:4: error: matmul inner dimensions disagree: 3 vs 4" in result.stdout, out
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    crashes = [e for e in events if e["outcome"] == "crash"]
    assert crashes and all(e["line"] == 3 for e in crashes), events
    assert "RuntimeError: seeded failure" in crashes[0]["reason"], events


def test_hook_exception_is_silenceable_by_code(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, debug=False, config="disable_error_code = numty-internal\n")
    assert "numty internal error" not in result.stdout, result.stdout
    assert "matmul inner dimensions disagree" in result.stdout, result.stdout


def test_debug_mode_propagates_the_exception(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, debug=True)
    out = result.stdout + result.stderr
    assert "INTERNAL ERROR" in out, out
    assert "[numty-internal]" not in out, out


def test_suite_runs_in_debug_mode() -> None:
    # conftest.py turns this on; without it a swallowed exception in any
    # fixture run would degrade to a passing test.
    assert os.environ.get("NUMTY_DEBUG") == "1"


def test_every_broad_except_goes_through_the_policy() -> None:
    lines = Path(numty_plugin.__file__).read_text(encoding="utf-8").splitlines()
    stray = []
    for i, line in enumerate(lines):
        if line.strip().startswith("except Exception"):
            nxt = lines[i + 1].strip()
            inside_helper = "reporting must never" in line
            if not nxt.startswith("_swallowed(") and not inside_helper:
                stray.append(f"{i + 1}: {line.strip()} -> {nxt}")
    assert not stray, "broad except without _swallowed():\n" + "\n".join(stray)


def test_swallowed_reraises_only_in_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NUMTY_DEBUG", "1")
    with pytest.raises(KeyError):
        try:
            raise KeyError("x")
        except Exception:
            numty_plugin._swallowed(None, "unit")
    monkeypatch.delenv("NUMTY_DEBUG")
    try:
        raise KeyError("x")
    except Exception:
        numty_plugin._swallowed(None, "unit")  # no context: trace-only, no raise
