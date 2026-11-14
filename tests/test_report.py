"""`python -m numty_report` give-up measurements (issue #164 step 3).

The report runs mypy with tracing on a fresh cache and prints
per-file solved/bailed/unhooked/crashed counts plus the top bail
reasons. Bails stay silent by default; the report is how they are
measured instead of errored on.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = "[mypy]\nplugins = numty_plugin\n"

MIXED_SOURCE = """\
import numpy as np

a = np.zeros((2, 3))
n = int("3")
b = np.zeros((n, 3))
reveal_type(np.isreal(a))
"""

BAIL_SOURCE = """\
import numpy as np


def stack_bare(columns: list[np.ndarray]) -> None:
    np.column_stack([c for c in columns])
"""

BROKEN_PLUGIN = """\
import numty_plugin
from numty_plugin import plugin


def _boom(*args, **kwargs):
    raise RuntimeError("seeded failure")


numty_plugin._broadcast_result = _boom
"""

BROKEN_SOURCE = """\
import numpy as np

a = np.zeros((2, 3)) + np.zeros((2, 3))
"""


def _run_report(
    workdir: Path, *args: str, extra_path: str = ""
) -> "subprocess.CompletedProcess[str]":
    pythonpath = str(ROOT) if not extra_path else f"{extra_path}{os.pathsep}{ROOT}"
    env = {**os.environ, "PYTHONPATH": pythonpath}
    return subprocess.run(
        [sys.executable, "-m", "numty_report", *args],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _write_project(workdir: Path, source: str, config: str = CONFIG) -> Path:
    (workdir / "mypy.ini").write_text(config, encoding="utf-8")
    src = workdir / "repro.py"
    src.write_text(source, encoding="utf-8")
    return src


def test_report_counts_solved_bailed_unhooked(tmp_path: Path) -> None:
    _write_project(tmp_path, MIXED_SOURCE)
    result = _run_report(tmp_path, "--top", "5", "repro.py")
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    assert "numty report:" in result.stdout, out
    assert "repro.py" in result.stdout, out
    row = next(
        line for line in result.stdout.splitlines() if line.startswith("repro.py ")
    )
    solved, bailed, unhooked, crashed, total = (int(x) for x in row.split()[1:6])
    assert solved >= 1, row
    # MIXED_SOURCE has no numpy bail (zeros solved, isreal unhooked;
    # non-numpy SupportsIndex bails are hidden); total is deduped sites.
    assert total == solved + bailed + unhooked + crashed, row
    assert total <= 5, row  # deduped: 2 zeros + isreal, not 18 events
    assert "top unhooked ops" in result.stdout, out
    assert "numpy.lib._type_check_impl.isreal" in result.stdout, out


def test_report_counts_numpy_bail(tmp_path: Path) -> None:
    _write_project(tmp_path, BAIL_SOURCE)
    result = _run_report(tmp_path, "--top", "5", "repro.py")
    out = result.stdout + result.stderr
    assert result.returncode == 1, out  # column_stack bare bails loudly
    assert "numty report:" in result.stdout, out
    row = next(
        line for line in result.stdout.splitlines() if line.startswith("repro.py ")
    )
    solved, bailed, unhooked, crashed, total = (int(x) for x in row.split()[1:6])
    assert bailed >= 1, row
    assert "top bail reasons" in result.stdout, out
    assert "column_stack" in result.stdout, out
    assert "items have no known shape" in result.stdout, out


def test_bails_stay_silent_by_default(tmp_path: Path) -> None:
    _write_project(tmp_path, MIXED_SOURCE)
    result = _run_report(tmp_path, "repro.py")
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    assert "numty-unsolved" not in out, out


def test_report_passes_mypy_exit_code_through(tmp_path: Path) -> None:
    _write_project(
        tmp_path,
        "import numpy as np\n"
        "np.matmul(np.zeros((2, 3)), np.zeros((4, 5)))\n",
    )
    result = _run_report(tmp_path, "repro.py")
    out = result.stdout + result.stderr
    assert result.returncode == 1, out
    assert "matmul inner dimensions disagree" in out, out
    assert "numty report:" in result.stdout, out


def test_report_uses_a_fresh_cache(tmp_path: Path) -> None:
    _write_project(tmp_path, MIXED_SOURCE)
    first = _run_report(tmp_path, "repro.py")
    second = _run_report(tmp_path, "repro.py")
    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    # A warm-cache trace would be empty (only re-checked modules emit);
    # the report always starts fresh, so both runs measure calls.
    assert "0 traced call(s)" not in second.stdout, second.stdout


def test_report_hides_stub_noise_unless_asked(tmp_path: Path) -> None:
    _write_project(tmp_path, MIXED_SOURCE)
    default = _run_report(tmp_path, "repro.py")
    assert "hidden" in default.stdout, default.stdout
    assert "numpy/__init__" not in default.stdout, default.stdout
    assert ".venv" not in default.stdout, default.stdout
    full = _run_report(tmp_path, "--all-files", "repro.py")
    assert "numpy" in full.stdout, full.stdout


def test_report_records_crashes(tmp_path: Path) -> None:
    (tmp_path / "broken_numty.py").write_text(BROKEN_PLUGIN, encoding="utf-8")
    (tmp_path / "mypy.ini").write_text(
        "[mypy]\nplugins = broken_numty\n", encoding="utf-8"
    )
    (tmp_path / "repro.py").write_text(BROKEN_SOURCE, encoding="utf-8")
    result = _run_report(tmp_path, "repro.py", extra_path=str(tmp_path))
    out = result.stdout + result.stderr
    assert "numty report:" in result.stdout, out
    row = next(
        line for line in result.stdout.splitlines() if line.startswith("repro.py ")
    )
    crashed = int(row.split()[4])
    assert crashed >= 1, row
    assert "RuntimeError: seeded failure" in result.stdout, out
