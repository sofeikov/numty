"""Static checks: each bad fixture must fail mypy with exactly one error."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
# A dedicated cache dir, cleared once per suite run: mypy does not reliably
# invalidate plugin results across plugin edits, so never reuse the default
# cache (or a previous run's) here. Within one run the plugin is fixed, so
# incremental caching across fixtures stays fast.
CACHE = ROOT / ".mypy-test-cache"


@pytest.fixture(scope="session", autouse=True)
def _fresh_mypy_cache() -> None:
    shutil.rmtree(CACHE, ignore_errors=True)
    yield
    shutil.rmtree(CACHE, ignore_errors=True)


def run_mypy(path: Path) -> "subprocess.CompletedProcess[str]":
    # PYTHONPATH (not MYPYPATH) is what lets mypy import the local plugin.
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run(
        [sys.executable, "-m", "mypy", "--cache-dir", str(CACHE), str(path)],
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
    assert len(notes) == 10, result.stdout
    assert any("tuple[Literal[2], Literal[4]]" in line for line in notes), result.stdout
    assert any("tuple[Literal[2], Literal[3]]" in line for line in notes), result.stdout
    assert any("tuple[Literal[3], Literal[2]]" in line for line in notes), result.stdout
    assert any("tuple[Literal[4], Literal[4]]" in line for line in notes), result.stdout
    assert any("tuple[Literal[2], Literal[1]]" in line for line in notes), result.stdout
    assert any('"numpy.float64"' in line for line in notes), result.stdout
    assert any("tuple[Literal[6]]" in line for line in notes), result.stdout


def test_concat_dim_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_concat_dim.py"),
        "concatenate dimension 1 differs: 3 vs 4",
    )


def test_stack_shape_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_stack_shape.py"),
        "stack shapes differ: (2, 3), (3, 2)",
    )


def test_arange_step_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_arange_step.py"),
        "arange step is zero",
    )


def test_eye_neg_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_eye_neg.py"),
        "eye dimensions must be non-negative, got (-1, -1)",
    )


def test_concat_axis_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_concat_axis.py"),
        "concatenate axis 5 is out of range for 2 dimensions",
    )


def test_index_range_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_index_range.py"),
        "index 5 is out of range for axis 0 with size 2",
    )


def test_index_too_many_mismatch() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_index_too_many.py"),
        "too many indices for 2 dimensions",
        "Invalid index type",
    )


def test_slice_step_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_slice_step.py"),
        "slice step is zero",
    )


def test_compare_broadcast_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_compare_broadcast.py"),
        "broadcast shapes disagree at trailing axis 0: 3 vs 2",
    )


def test_argmax_axis_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_argmax_axis.py"),
        "axis 9 is out of range for 2 dimensions",
    )


def test_einsum_label_conflict() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_einsum_labels.py"),
        "einsum label 'j' has conflicting sizes: 3 vs 5",
        '"einsum"',
    )


def test_einsum_rank_mismatch() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_einsum_rank.py"),
        "einsum subscript 'ijk' has 3 labels but operand has rank 2",
        '"einsum"',
    )


def test_solve_square_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_solve_square.py"),
        "linalg.solve requires a square (..., N, N) array, got shape (2, 3)",
    )


def test_solve_inner_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_solve_inner.py"),
        "linalg.solve inner dimensions disagree: 4 vs 5",
    )


def test_svd_rank_mismatch() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_svd_rank.py"),
        "linalg.svd requires at least 2 dimensions, got shape (5,)",
        '"svd"',
    )


def test_norm_axis_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_norm_axis.py"),
        "linalg.norm axis 9 is out of range for 2 dimensions",
    )


def assert_rank_mismatch(
    result: "subprocess.CompletedProcess[str]", ours: str, native: str, *extra: str
) -> None:
    """Our solver message, plus mypy's own overload error on top."""
    assert result.returncode == 1, result.stdout
    ours_lines = [line for line in result.stdout.splitlines() if ours in line]
    assert len(ours_lines) == 1, result.stdout
    for fragment in extra:
        assert fragment in ours_lines[0], result.stdout
    rest = [line for line in result.stdout.splitlines() if ": error:" in line and line not in ours_lines]
    assert rest and all(native in line for line in rest), result.stdout


def test_transpose_axes_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_transpose_axes.py"),
        "transpose axes (0, 0) are not a permutation",
    )


def test_squeeze_size_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_squeeze_size.py"),
        "squeeze axis 0 has size 2, not 1",
    )


def test_sum_axis_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_sum_axis.py"),
        "axis 5 is out of range for 2 dimensions",
    )


def test_expand_axis_mismatch() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_expand_axis.py"),
        "expand_dims axis 5 is out of range",
        '"expand_dims"',
    )


def test_folded_size_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_fold_size.py"),
        "tuple[Literal[4], Literal[3]]",
        "tuple[Literal[2], Literal[4]]",
    )


def test_folded_reshape_mismatch() -> None:
    assert_reshape_mismatch(
        run_mypy(FIXTURES / "bad_fold_reshape.py"),
        "reshape changes total size: 6 -> 12",
    )


def assert_reshape_mismatch(result: "subprocess.CompletedProcess[str]", *fragments: str) -> None:
    """Our solver message, plus mypy's own overload error on top."""
    assert result.returncode == 1, result.stdout
    ours = [line for line in result.stdout.splitlines() if fragments[0] in line]
    assert len(ours) == 1, result.stdout
    for fragment in fragments[1:]:
        assert fragment in ours[0], result.stdout
    rest = [line for line in result.stdout.splitlines() if ": error:" in line and line not in ours]
    assert rest and all('"reshape"' in line for line in rest), result.stdout


def test_reshape_size_mismatch() -> None:
    assert_reshape_mismatch(
        run_mypy(FIXTURES / "bad_reshape_size.py"),
        "reshape changes total size: 6 -> 8",
        "(2, 3)",
        "(2, 4)",
    )


def test_reshape_minus1_mismatch() -> None:
    assert_reshape_mismatch(
        run_mypy(FIXTURES / "bad_reshape_minus1.py"),
        "reshape cannot infer -1",
        "6",
        "4",
    )


def test_reshape_varargs_mismatch() -> None:
    # Varargs match the (size1, size2) overload cleanly, so only our
    # solver speaks up; the tuple form also draws mypy's overload blame.
    assert_single_error(
        run_mypy(FIXTURES / "bad_reshape_varargs.py"),
        "reshape changes total size: 8 -> 10",
    )


def test_reshape_two_minus1_mismatch() -> None:
    assert_reshape_mismatch(
        run_mypy(FIXTURES / "bad_reshape_two_minus1.py"),
        "reshape accepts only one -1 dimension",
    )
