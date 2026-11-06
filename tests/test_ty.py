"""ty checks: what numty promises without the plugin (layers 1-2, FlexArray)."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

# ty is a pinned dev dependency; skip, not fail, where it is not installed.
pytest.importorskip("ty")

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"


def run_ty(*paths: Path) -> "subprocess.CompletedProcess[str]":
    # ty is pinned in the dev group; --python points it at this venv's numpy.
    return subprocess.run(
        [sys.executable, "-m", "ty", "check", "--output-format", "concise",
         "--python", sys.prefix, *map(str, paths)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


def error_lines(result: "subprocess.CompletedProcess[str]") -> list[tuple[str, int]]:
    found = re.findall(r"^(\S+?\.py):(\d+):\d+: error\[", result.stdout, re.M)
    return [(Path(name).name, int(line)) for name, line in found]


def test_quickstart_reports_only_the_demo_errors() -> None:
    result = run_ty(ROOT / "numty.py", ROOT / "demo.py")
    assert result.returncode == 1, result.stdout
    assert error_lines(result) == [("demo.py", 30), ("demo.py", 34)], result.stdout
    assert "Found 2 diagnostics" in result.stdout, result.stdout


def test_flexarray_keeps_rank_checks_under_ty() -> None:
    # The assignment alias `FlexArray = np.ndarray[S2, D2]` resolved to an
    # Unknown shape under ty, so both bugs passed (#36).
    fixture = FIXTURES / "flex_portable.py"
    bugs = [
        i for i, line in enumerate(fixture.read_text().splitlines(), 1)
        if "# BUG" in line
    ]
    result = run_ty(fixture)
    assert result.returncode == 1, result.stdout
    assert error_lines(result) == [("flex_portable.py", n) for n in bugs], result.stdout
