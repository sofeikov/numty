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


def assert_operator_mismatch(result: "subprocess.CompletedProcess[str]", *fragments: str) -> None:
    """Our solver message, plus mypy's generic operator error on top."""
    assert result.returncode == 1, result.stdout
    ours = [line for line in result.stdout.splitlines() if fragments[0] in line]
    assert len(ours) == 1, result.stdout
    for fragment in fragments[1:]:
        assert fragment in ours[0], result.stdout
    rest = [line for line in result.stdout.splitlines() if ": error:" in line and line not in ours]
    assert rest and all("Unsupported operand" in line for line in rest), result.stdout


def test_broadcast_mismatch() -> None:
    assert_operator_mismatch(
        run_mypy(FIXTURES / "bad_broadcast.py"),
        "broadcast shapes disagree at trailing axis 0",
        "3 vs 4",
        "(2, 3)",
        "(2, 4)",
    )


def test_good_path_precision_visible() -> None:
    result = run_mypy(FIXTURES / "static_ok.py")
    assert result.returncode == 0, result.stdout
    notes = [line for line in result.stdout.splitlines() if "Revealed type" in line]
    assert len(notes) == 3, result.stdout
    assert any("tuple[Literal[2], Literal[4]]" in line for line in notes), result.stdout
    assert any("tuple[Literal[2], Literal[3]]" in line for line in notes), result.stdout
