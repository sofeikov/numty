"""Static checks: each bad fixture must fail mypy with exactly one error."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"


def run_mypy(path: Path) -> "subprocess.CompletedProcess[str]":
    # PYTHONPATH (not MYPYPATH) is what lets mypy import the local plugin.
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run(
        [sys.executable, "-m", "mypy", str(path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def assert_single_error(result: "subprocess.CompletedProcess[str]", *fragments: str) -> None:
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 1, result.stdout
    for fragment in fragments:
        assert fragment in errors[0], result.stdout


def test_good_code_is_clean() -> None:
    result = run_mypy(FIXTURES / "static_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout


def test_rank_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_rank.py"),
        "tuple[int]",
        "tuple[Literal[2], Literal[3]]",
    )


def test_size_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_size.py"),
        "tuple[Literal[3], Literal[2]]",
        "tuple[Literal[2], Literal[3]]",
    )


def test_matmul_inner_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_matmul_inner.py"),
        "matmul inner dimensions disagree: 3 vs 2",
    )


def test_matmul_batch_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_matmul_batch.py"),
        "matmul batch dimensions disagree",
        "(5,)",
        "(4,)",
    )
