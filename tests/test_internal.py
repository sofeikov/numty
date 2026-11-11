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
    # Every `except Exception:` must route through _swallowed() on the
    # very next line, except the reporting helpers themselves (those
    # lines carry "reporting must never"). Scans every .py under
    # numty_plugin/ so Phase 2 leaves (_*.py) are covered.
    # NOTE: the floor is 75 (was 76 on the monolith). Monolith line-scan
    # counted 76; Phase 2 removed a dead _swallowed-guarded try in
    # _forms_provably_ne (pure after the move) as part of the leaf
    # extraction, so 75 is the correct package floor. Do not raise to
    # 77 (grep -c also matches docstring prose).
    plugin_dir = ROOT / "numty_plugin"
    files = sorted(
        p for p in plugin_dir.rglob("*.py") if "__pycache__" not in p.parts
    )
    assert files, f"no .py files found under {plugin_dir} - glob broken?"
    assert plugin_dir / "__init__.py" in files, "scan must include __init__.py"
    stray = []
    total = 0
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        rel = path.relative_to(plugin_dir)
        for i, line in enumerate(lines):
            if line.strip().startswith("except Exception"):
                total += 1
                nxt = lines[i + 1].strip() if i + 1 < len(lines) else "<EOF>"
                inside_helper = "reporting must never" in line
                if not nxt.startswith("_swallowed(") and not inside_helper:
                    stray.append(f"{rel}:{i + 1}: {line.strip()} -> {nxt}")
    assert not stray, "broad except without _swallowed():\n" + "\n".join(stray)
    assert total >= 75, f"broad-except total dropped: {total} < 75"


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


SEEDED_TUPLE_PLUGIN = """\
import sys

from mypy.types import TupleType
from numty_plugin import plugin

_original = TupleType.copy_modified


def _copy_modified(self, *args, **kwargs):
    if sys._getframe(1).f_code.co_name == "_substitute_return_exprs":
        raise RuntimeError("seeded failure")
    return _original(self, *args, **kwargs)


TupleType.copy_modified = _copy_modified
"""

FORMULA_RETURN_SOURCE = """\
from typing import TypeVar, reveal_type

import numpy as np

from numty import I64

M = TypeVar("M", bound=int)
K = TypeVar("K", bound=int)


def make_offsets(lengths: I64[M]) -> I64["M+1"]:
    return np.concatenate([[0], np.cumsum(lengths)])


def caller(a: I64[K]) -> None:
    reveal_type(make_offsets(a))
"""


def test_return_substitution_survives_a_failed_tuple_copy(tmp_path: Path) -> None:
    # `_substitute_return_exprs` rebuilds the shape tuple by hand when
    # TupleType.copy_modified raises. That path used `shape.fallback`,
    # an attribute TupleType does not have (it is `partial_fallback`),
    # so the rebuild raised too and the call site lost its K+1 shape.
    # Found by ty (#213).
    (tmp_path / "seeded_numty.py").write_text(SEEDED_TUPLE_PLUGIN, encoding="utf-8")
    (tmp_path / "mypy.ini").write_text("[mypy]\nplugins = seeded_numty\n", encoding="utf-8")
    (tmp_path / "repro.py").write_text(FORMULA_RETURN_SOURCE, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": f"{tmp_path}{os.pathsep}{ROOT}"}
    env.pop("NUMTY_DEBUG", None)
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--cache-dir", str(tmp_path / "cache"), "repro.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = result.stdout + result.stderr
    assert "INTERNAL ERROR" not in out, out
    # The seeded failure is reported, and the hand-built tuple is used.
    assert "RuntimeError: seeded failure" in result.stdout, out
    assert "AttributeError" not in out, out
    assert 'Revealed type is "numpy.ndarray[tuple[K+1], ' in result.stdout, out
