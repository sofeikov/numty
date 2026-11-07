"""Static checks: each bad fixture must fail mypy with exactly one error."""
import os
import runpy
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


def run_mypy(*paths: Path, flags: tuple[str, ...] = ()) -> "subprocess.CompletedProcess[str]":
    # PYTHONPATH (not MYPYPATH) is what lets mypy import the local plugin.
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run(
        [sys.executable, "-m", "mypy", "--cache-dir", str(CACHE), *flags, *map(str, paths)],
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
    assert len(notes) == 39, result.stdout
    assert any(
        "tuple[Literal[2], Literal[2]]" in line and "|" in line for line in notes
    ), result.stdout
    assert any("tuple[Literal[5]]" in line for line in notes), result.stdout
    assert any("tuple[Literal[5], Literal[2]]" in line for line in notes), result.stdout
    assert any("tuple[Literal[2], Literal[2]]" in line for line in notes), result.stdout
    assert any(
        "tuple[Literal[2], Literal[3], Literal[2]]" in line for line in notes
    ), result.stdout
    assert any("tuple[Literal[2], Literal[3]]" in line for line in notes), result.stdout
    assert any("tuple[Literal[3]]" in line for line in notes), result.stdout
    assert any("tuple[Literal[2], Literal[4]]" in line for line in notes), result.stdout
    assert any("tuple[Literal[2], Literal[3]]" in line for line in notes), result.stdout
    assert any("tuple[Literal[3], Literal[2]]" in line for line in notes), result.stdout
    assert any("tuple[Literal[4], Literal[4]]" in line for line in notes), result.stdout
    assert any("tuple[Literal[2], Literal[1]]" in line for line in notes), result.stdout
    assert any('"numpy.float64"' in line for line in notes), result.stdout
    assert any("tuple[Literal[6]]" in line for line in notes), result.stdout
    assert any("tuple[Literal[2], Literal[4], Literal[6]]" in line for line in notes), result.stdout
    assert any("tuple[Literal[2], *Ts]" in line for line in notes), result.stdout
    assert any(
        "tuple[Literal[5], Literal[4], *Ts, Literal[5]]" in line for line in notes
    ), result.stdout
    assert any("tuple[*Ts, Literal[6]]" in line for line in notes), result.stdout
    assert any("tuple[Literal[5], *Ts, Literal[6]]" in line for line in notes), result.stdout
    assert any(
        "tuple[Literal[5], *Ts, Literal[1], Literal[7]]" in line for line in notes
    ), result.stdout
    assert any("tuple[*Ts, Literal[7]]" in line for line in notes), result.stdout
    assert any(
        "tuple[Literal[5], Literal[6], Literal[6], Literal[7], *Ts, Literal[8]]" in line
        for line in notes
    ), result.stdout


def test_matmul_ts_inner_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_matmul_ts_inner.py"),
        "matmul inner dimensions disagree: 5 vs 3",
    )


def test_matmul_ts_batch_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_matmul_ts_batch.py"),
        "matmul batch dimensions disagree",
        "(5, *Ts)",
        "(4, *Ts)",
    )


def test_reshape_ts_size_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_reshape_ts_size.py"),
        "reshape changes total size: 8 is not a multiple of known size 6",
        "(*Ts, 2, 3)",
        "(2, 4)",
    )


def test_index_ts_prefix_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_index_ts_prefix.py"),
        "index 99 is out of range for axis 0 with size 5",
        "((5, *Ts, 7))",
    )


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


def test_index_enum_range_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_index_enum_range.py"),
        "index 2 is out of range for axis 0 with size 2",
    )


def test_tail2_trailing_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_tail2_mismatch.py"),
        "tuple[Literal[2], Literal[3], Literal[3]]",
        "tuple[Literal[2], Literal[3], Literal[2]]",
    )


def test_tail2_range_mismatch() -> None:
    # The stubs echo their own [index] complaint on unpack shapes, so
    # this asserts the message rather than a single error.
    result = run_mypy(FIXTURES / "bad_tail2_range.py")
    assert result.returncode == 1, result.stdout
    assert "index 99 is out of range for axis -1 with size 2" in result.stdout, result.stdout
    assert "((*Ts, 2))" in result.stdout, result.stdout


def test_where_union_member_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_where_union_member.py"),
        "trailing axis 0: 3 vs 2",
        "(2, 3)",
        "(2, 2)",
    )


def test_union_genexp_member_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_union_genexp_member.py"),
        "trailing axis 0: 2 vs 3",
        "(2, 2)",
        "(2, 3)",
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


def test_where_broadcast_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_where_broadcast.py"),
        "where shapes disagree:",
    )


def test_raw_matmul_inner_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_matmul_raw.py"),
        "matmul inner dimensions disagree: 3 vs 5",
    )


def test_truediv_broadcast_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_truediv_broadcast.py"),
        "true_divide shapes disagree:",
    )


def test_take_rank_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_take_rank.py"),
        "must have the same number of dimensions",
    )


def test_take_nonaxis_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_take_mismatch.py"),
        "take_along_axis shapes mismatch at axis 0: 5 vs 4",
    )


def test_take_axis_out_of_range() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_take_axis.py"),
        "take_along_axis axis 5 is out of range for 2 dimensions",
    )


def test_logical_broadcast_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_logical_broadcast.py"),
        "broadcast shapes disagree at trailing axis 0",
        "3 vs 4",
        "(2, 3)",
        "(2, 4)",
    )


def test_any_axis_out_of_range() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_any_axis.py"),
        "axis 5 is out of range for 2 dimensions",
    )


def test_reshape_symbolic_size() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_reshape_sym_size.py"),
        "reshape changes total size: 20*N -> 21*N",
        "(N, 10, 2)",
        "(N, 21)",
    )


def test_reshape_symbolic_minus1() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_reshape_sym_minus1.py"),
        "reshape cannot infer -1",
        "20",
        "3",
    )


def test_min_axis_out_of_range() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_min_axis.py"),
        "axis 5 is out of range for 2 dimensions",
    )


def test_callsite_conflict() -> None:
    result = run_mypy(FIXTURES / "bad_callsite_conflict.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 3, result.stdout
    assert "shape conflict for 'N'" in errors[0], result.stdout
    assert "'a'" in errors[0] and "'b'" in errors[0], result.stdout
    assert "5" in errors[0] and "7" in errors[0], result.stdout
    assert "shape conflict for 'K'" in errors[1], result.stdout
    assert "7" in errors[1] and "9" in errors[1], result.stdout
    assert "shape conflict for 'N'" in errors[2], result.stdout
    assert "'a'" in errors[2] and "'b'" in errors[2], result.stdout


def test_fancy_dtype_mismatch() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_fancy_dtype.py"),
        "arrays used as indices must be of integer (or boolean) type",
        "Invalid index type",
    )


def test_symbolic_take_rank_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_sym_take_rank.py"),
        "must have the same number of dimensions",
    )


def test_len_unsized_object() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_len_scalar.py"),
        "len() of unsized object",
    )


def test_symbolic_too_many_indices() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_sym_too_many.py"),
        "too many indices for 2 dimensions",
        "Invalid index type",
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


def test_f64_ok_clean_and_runs() -> None:
    result = run_mypy(FIXTURES / "f64_ok.py")
    assert result.returncode == 0, result.stdout
    run = subprocess.run(
        [sys.executable, str(FIXTURES / "f64_ok.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == [
        "(5, 8)",
        "(3,)",
        "(4, 3)",
        "(5,)",
        "(5, 4)",
        "(2, 3)",
    ]


def test_storage_heads_ok_clean_and_runs() -> None:
    result = run_mypy(FIXTURES / "storage_heads_ok.py")
    assert result.returncode == 0, result.stdout
    run = subprocess.run(
        [sys.executable, str(FIXTURES / "storage_heads_ok.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == [
        "(5, 2)",
        "(4, 7)",
        "(6,)",
        "(32, 48, 3)",
        "(4, 5)",
        "(3, 23)",
    ]


def test_storage_head_dtype_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_storage_dtype.py"),
        'Argument 1 to "show"',
        "dtype[signedinteger[_8Bit]]",
        "dtype[unsignedinteger[_8Bit]]",
    )


def test_mixed_builders_keep_literals() -> None:
    # #34: one dynamic dim no longer erases the literal ones, so a stored
    # builder result fits a field pinned to 2 / 23. Ops that would move
    # the literal to the wrong axis widen it to unknown (Any) instead.
    result = run_mypy(FIXTURES / "mixed_builders_ok.py")
    assert result.returncode == 0, result.stdout
    notes = [
        line.split("Revealed type is ")[1].replace("numpy._typing._nbit_base.", "")
        for line in result.stdout.splitlines()
        if "Revealed type" in line
    ]
    mixed3 = '"numpy.ndarray[tuple[int, Literal[3]], numpy.dtype[numpy.float64]]"'
    widened = '"numpy.ndarray[tuple[int, Any], numpy.dtype[numpy.float64]]"'
    assert notes == [
        '"numpy.ndarray[tuple[Literal[0], int, Literal[23], Literal[2]],'
        ' numpy.dtype[numpy.float64]]"',
        mixed3,  # ones
        mixed3,  # empty
        mixed3,  # full
        '"numpy.ndarray[tuple[int, Literal[4]], numpy.dtype[numpy.float64]]"',
        mixed3,  # rng.normal(size=)
        '"numpy.ndarray[tuple[int, Literal[3]],'
        ' numpy.dtype[numpy.signedinteger[_64Bit]]]"',
        # A dim read from .shape stays symbolic (this crashed before).
        '"numpy.ndarray[tuple[N, Literal[3]], numpy.dtype[numpy.float64]]"',
        # nothing provable: the stubs' answer
        '"numpy.ndarray[tuple[int, int], numpy.dtype[numpy.float64]]"',
        mixed3,  # a * 2
        '"numpy.ndarray[tuple[int, Literal[3]],'
        ' numpy.dtype[numpy.floating[_32Bit]]]"',
        '"numpy.ndarray[tuple[Literal[3], int], numpy.dtype[numpy.float64]]"',  # .T
        widened,  # swapaxes
        widened,  # pad
        widened,  # diff
        widened,  # keepdims sum
    ], result.stdout
    run = subprocess.run(
        [sys.executable, str(FIXTURES / "mixed_builders_ok.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == ["(4, 7, 2) (4, 7, 23)"]


def test_mixed_widened_results_fit_literal_types() -> None:
    # Widened dims are unknown, not int: diff/pad/clip of a mixed array
    # still return or rebind as the input's tuple[int, Literal[3]] type.
    result = run_mypy(FIXTURES / "mixed_widen_ok.py")
    assert result.returncode == 0, result.stdout
    assert _run_fixture("mixed_widen_ok.py") == [
        "(4, 3) (3, 3) (6, 3)",
        "(4, 3, 3, 3, 6, 5)",
    ]


def test_mixed_rebinding_to_own_op_stays_accepted() -> None:
    # A name first bound to np.zeros((n, 3)) is tuple[int, Literal[3]];
    # rebinding it to a fitting op on itself was accepted when builders
    # answered tuple[int, int] and must stay so. reshape and dot/matmul
    # now solve to the literal; the stubs' int dims where the literal
    # was turn unknown.
    result = run_mypy(FIXTURES / "mixed_rebind_ok.py")
    assert result.returncode == 0, result.stdout
    notes = [
        line.split("Revealed type is ")[1]
        for line in result.stdout.splitlines()
        if "Revealed type" in line
    ]
    mixed3 = '"numpy.ndarray[tuple[int, Literal[3]], numpy.dtype[numpy.float64]]"'
    widened = '"numpy.ndarray[tuple[int, Any], numpy.dtype[numpy.float64]]"'
    # (n, 3) by (3, k): k sits where o has its 3, n where w has its 3.
    unknown = '"numpy.ndarray[tuple[Any, Any], numpy.dtype[numpy.float64]]"'
    assert notes == [mixed3] * 5 + [widened] * 2 + [unknown], result.stdout
    assert _run_fixture("mixed_rebind_ok.py") == [
        "[(5, 3), (5, 3), (5, 3), (5, 3), (5, 3), (5, 3), (5, 3), (5, 3)]",
        "[(5, 3), (3, 5), (5, 3, 4), (5, 3), (5, 3), (5, 3), (5, 3), (5, 3)]",
        "n3 65",
        "three_n 59",
        "n34 56",
    ]


def test_recursive_alias_operands_do_not_crash() -> None:
    # The mixed-shape guard walked recursive aliases forever
    # (RecursionError, an INTERNAL ERROR) on any numpy call.
    result = run_mypy(FIXTURES / "recursive_alias_ok.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _run_fixture("recursive_alias_ok.py") == ["False True"]


def test_mixed_builder_literal_conflict() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_mixed_builder.py"),
        'Argument 1 to "Positions"',
        "tuple[int, Literal[3]]",
        "tuple[int, Literal[2]]",
    )


def test_f64_callsite_conflict() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_f64_conflict.py"),
        "call-site shape conflict for 'hid'",
        "8 (argument 'x', axis 0) vs 2 (argument 'y', axis 0)",
    )


def test_f64_callsite_same_argument_names_axes() -> None:
    # One argument, two axes: the axes are what tell the sides apart.
    assert_single_error(
        run_mypy(FIXTURES / "bad_callsite_same_arg.py"),
        "call-site shape conflict for 'seq'",
        "4 (argument 'x', axis 0) vs 5 (argument 'x', axis 1)",
    )
    # numpy runs it anyway: the square annotation is what is violated.
    assert _run_fixture("bad_callsite_same_arg.py") == ["(4,)"]


def test_f64_return_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_f64_return.py"),
        "Incompatible return value",
        "Literal[3], Literal[5]",
        "Literal[3], Literal[4]",
    )


def test_f64_dtype_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_f64_dtype.py"),
        "Incompatible return value",
        "dtype[float64]",
        "dtype[signedinteger[_32Bit]]",
    )


def test_f64_tuple_spelling_rejected() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_f64_shape.py"),
        "F64 takes flat dims",
        "F64[N, M], not F64[tuple[N, M]]",
    )


def test_f64_empty_rejected() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_f64_empty.py"),
        "F64 takes at least one dim",
    )


def test_f64_undeclared_name() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_f64_undeclared.py"),
        'Name "batsh" is not defined',
    )


def test_f64_star_redirect() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_f64_star.py"),
        "F64 star syntax (*Ts) is not supported",
        "write Unpack[Ts] instead",
    )


def test_f64_unbounded_typevar() -> None:
    # numpy's own [type-var] error would blame its tuple[int, ...] bound.
    assert_single_error(
        run_mypy(FIXTURES / "bad_f64_unbounded.py"),
        'F64 dim "M" is a TypeVar without an int bound',
        'declare it as TypeVar("M", bound=int)',
    )


def test_f64_unbound_typevar_reported_once() -> None:
    # mypy already explains the unbound TypeVar; no second, wrong reason.
    result = run_mypy(FIXTURES / "bad_f64_unbound.py")
    assert_single_error(result, 'Type variable "numty.batch" is unbound')
    assert "must be a TypeVar" not in result.stdout, result.stdout


def test_f64_str_ok_clean_and_runs() -> None:
    result = run_mypy(FIXTURES / "f64_str_ok.py")
    assert result.returncode == 0, result.stdout
    run = subprocess.run(
        [sys.executable, str(FIXTURES / "f64_str_ok.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == [
        "(5, 3)",
        "(5, 6)",
        "(3,)",
        "(5,)",
        "(7,)",
    ]


def test_f64_str_callsite_conflict() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_f64_str_conflict.py"),
        "call-site shape conflict for 'hid'",
        "8 (argument 'x', axis 0) vs 2 (argument 'y', axis 0)",
    )


def test_f64_str_matmul_dim_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_f64_str_matmul.py"),
        "call-site shape conflict for 'dim2'",
        "4 (argument 'x', axis 1) vs 5 (argument 'y', axis 0)",
    )


def test_f64_str_undeclared_single_word() -> None:
    # Single-word specs are forward references: mypy resolves them before
    # any hook runs, so an undeclared one is mypy's own error, not a
    # minted dim. Multi-word specs never parse as types and mint freely.
    assert_single_error(
        run_mypy(FIXTURES / "bad_f64_str_undeclared.py"),
        'Name "batsh" is not defined',
    )


def test_f64_str_empty_rejected() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_f64_str_empty.py"),
        "F64 string dims must name at least one dim",
    )


def test_sklearn_nmf_clean_and_runs() -> None:
    pytest.importorskip("sklearn")
    result = run_mypy(FIXTURES / "sklearn_nmf.py")
    assert result.returncode == 0, result.stdout
    run = subprocess.run(
        [sys.executable, str(FIXTURES / "sklearn_nmf.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == ["(30, 5)"]


def test_sklearn_normalize_clean_and_runs() -> None:
    pytest.importorskip("sklearn")
    result = run_mypy(FIXTURES / "sklearn_normalize.py")
    assert result.returncode == 0, result.stdout
    run = subprocess.run(
        [sys.executable, str(FIXTURES / "sklearn_normalize.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == ["(25, 7)"]


def test_chained_block_clean_and_runs() -> None:
    result = run_mypy(FIXTURES / "chained_block.py")
    assert result.returncode == 0, result.stdout
    run = subprocess.run(
        [sys.executable, str(FIXTURES / "chained_block.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == ["(5, 7, 16)"]


def test_unknown_ok_clean_and_runs() -> None:
    result = run_mypy(FIXTURES / "unknown_ok.py")
    assert result.returncode == 0, result.stdout
    run = subprocess.run(
        [sys.executable, str(FIXTURES / "unknown_ok.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == ["ok"]


def test_unknown_rank_still_errors() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_unknown_rank.py"),
        "matmul inner dimensions disagree: 4 vs 5",
    )


def test_dynamic_concat_axis_oob() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_dynamic_concat_axis.py"),
        "concatenate axis 5 is out of range for 2 dimensions",
    )


def test_unknown_concrete_boundary() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_unknown_concrete.py"),
        "Incompatible types in assignment",
        "tuple[?U@13, M]",
        "tuple[Literal[3], M]",
    )


def _run_fixture(name: str) -> list[str]:
    run = subprocess.run(
        [sys.executable, str(FIXTURES / name)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    return run.stdout.strip().splitlines()


def test_unknown_names_distinct_and_readable() -> None:
    # Unrelated unknowns are named after the line that minted them, and
    # formulas over them never print the internal site key.
    result = run_mypy(FIXTURES / "unknown_names_ok.py")
    assert result.returncode == 0, result.stdout
    notes = [line for line in result.stdout.splitlines() if "Revealed type" in line]
    assert len(notes) == 2, result.stdout
    assert "tuple[?U@17, ?U@18]" in notes[0], result.stdout
    assert "tuple[?U@24+2]" in notes[1], result.stdout
    assert "_mask_" not in result.stdout and "__numty" not in result.stdout
    # numpy agrees the outer product is not square.
    assert _run_fixture("unknown_names_ok.py") == ["(4, 3)", "(5,)"]


def test_unknown_formula_in_error_is_readable() -> None:
    result = run_mypy(FIXTURES / "bad_unknown_pad_return.py")
    assert_single_error(
        result,
        "Incompatible return value",
        "tuple[?U@12+2, Literal[3]]",
        "tuple[N, Literal[3]]",
    )
    assert "_mask_" not in result.stdout, result.stdout
    # The annotation really is violated: 4 rows in, 6 out.
    assert _run_fixture("bad_unknown_pad_return.py") == ["(6, 3)"]


def test_sklearn_gnb_stack_dynamic_length() -> None:
    # Correct code hitting a solver gap (see #1 triage → #12): np.stack
    # over a loop-built list has no statically known length, so the hook
    # fails once at the call instead of poisoning downstream.
    pytest.importorskip("sklearn")
    result = run_mypy(FIXTURES / "sklearn_gnb.py")
    assert_single_error(
        result,
        "stack needs a sequence of known length",
    )
    run = subprocess.run(
        [sys.executable, str(FIXTURES / "sklearn_gnb.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == ["(40, 3)"]


def test_sklearn_ridge_clean_and_runs() -> None:
    # Optional corpus dependency: the annotated sklearn excerpt only runs
    # where sklearn is installed; elsewhere the dogfood is skipped, never red.
    pytest.importorskip("sklearn")
    result = run_mypy(FIXTURES / "sklearn_ridge.py")
    assert result.returncode == 0, result.stdout
    run = subprocess.run(
        [sys.executable, str(FIXTURES / "sklearn_ridge.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == ["(3, 8) (3,)"]


def test_f64_str_bad_token_rejected() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_f64_str_token.py"),
        'F64 string dim "3" is not a name',
        'space-separated names like "batch features"',
    )


def test_reduce15_ok_clean() -> None:
    result = run_mypy(FIXTURES / "reduce15_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout


def test_average_weights_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_average_weights.py"),
        "average weights length 5 does not match axis 1 size 6",
        "(128, 6)",
    )


def test_diff_axis_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_diff_axis.py"),
        "axis 2 is out of range for 2 dimensions",
        "(4, 3)",
    )


def test_diff_none_axis_rejected() -> None:
    # The stubs reject axis=None before our hook runs; lock that in.
    assert_single_error(
        run_mypy(FIXTURES / "bad_diff_none_axis.py"),
        'No overload variant of "diff" matches argument types',
    )


def test_diff_negative_order() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_diff_neg_n.py"),
        "np.diff order n=-1 must be non-negative",
    )


def test_cumsum_axis_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_cumsum_axis.py"),
        "axis 5 is out of range for 2 dimensions",
        "(2, 2)",
    )


def test_diff_size_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_diff_size.py"),
        "tuple[Literal[127], Literal[6]]",
        "tuple[Literal[126], Literal[6]]",
    )


def test_named_dims_clean_and_runs() -> None:
    result = run_mypy(FIXTURES / "named_dims.py")
    assert result.returncode == 0, result.stdout
    run = subprocess.run(
        [sys.executable, str(FIXTURES / "named_dims.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines() == ["(128,) (127, 6)"]


def test_named_dims_weights_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_named_dims_weights.py"),
        'Argument 2 to "profile" has incompatible type',
        "tuple[Literal[5]]",
        "tuple[Literal[6]]",
    )


def test_dot18_ok_clean() -> None:
    result = run_mypy(FIXTURES / "dot18_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout


def test_dot_nd_mismatch() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_dot_nd.py"),
        "dot contracted dimensions disagree: 3 vs 4",
        "incompatible type",
    )


def test_tensordot_size_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_tensordot_size.py"),
        "tensordot contracted dimensions disagree: 3 vs 4",
    )


def test_tensordot_axis_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_tensordot_axis.py"),
        "tensordot axis 5 is out of range for 2 dimensions",
        "(2, 3)",
    )


def test_inner_size_mismatch() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_inner_size.py"),
        "inner contracted dimensions disagree: 3 vs 4",
        "incompatible type",
    )


def test_vdot_size_mismatch() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_vdot_size.py"),
        "vdot operands have different sizes: 6 vs 4",
        "incompatible type",
    )


def test_take_oob_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_take_oob.py"),
        "axis 5 is out of range for 2 dimensions",
        "(2, 2)",
    )


def test_nan_ok_clean() -> None:
    result = run_mypy(FIXTURES / "nan_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout


def test_nan_axis_mismatch() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_nan_axis.py"),
        "axis 2 is out of range for 2 dimensions",
        "incompatible type",
    )


def test_atleast_ok_clean() -> None:
    result = run_mypy(FIXTURES / "atleast_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout


def test_stack_ok_clean() -> None:
    result = run_mypy(FIXTURES / "stack_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout


def test_hstack_rank_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_hstack_rank.py"),
        "hstack ranks differ: (3,), (2, 2)",
    )


def test_shapemath_ok_clean() -> None:
    result = run_mypy(FIXTURES / "shapemath_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout


def test_flip_axis_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_flip_axis.py"),
        "flip axis 5 is out of range for 2 dimensions",
        "(3, 4)",
    )


def test_fliplr_rank_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_fliplr_rank.py"),
        "fliplr requires at least 2 dimensions, got 1",
        "(8,)",
    )


def test_rot90_axis_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_rot90_axis.py"),
        "rot90 axis 5 is out of range for 2 dimensions",
        "(3, 4)",
    )


def test_gradient_axis_mismatch() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_gradient_axis.py"),
        "gradient axis 5 is out of range for 2 dimensions",
        '"gradient"',
    )


def test_gradient_axis_mismatch_dynamic_dims() -> None:
    # Gradient's axis checks need only the rank, so a plain int dim
    # (here from np.zeros((n, 3))) no longer skips them.
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_gradient_axis_dynamic.py"),
        "gradient axis 5 is out of range for 2 dimensions (shape (?, 3))",
        '"gradient"',
    )


def test_convolve_dim_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_convolve_dim.py"),
        "convolve only supports 1-dimensional arrays",
        "(3, 4)",
        "(8,)",
    )


def test_fft_n_mismatch() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_fft_n.py"),
        "fft n must be a positive number of data points, got 0",
        '"fft"',
    )


def test_rfft_axis_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_rfft_axis.py"),
        "rfft axis 3 is out of range for 1 dimensions",
        "(8,)",
    )


def test_fftn_len_mismatch() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_fftn_len.py"),
        "fftn Shape and axes have different lengths (1 vs 2)",
        '"fftn"',
    )


def test_fftn_size_mismatch() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_fftn_size.py"),
        "fftn s entries must be positive, got 0",
        '"fftn"',
    )


def test_fftn_long_mismatch() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_fftn_long.py"),
        "fftn s has 3 entries for 2 dimensions",
        '"fftn"',
    )


def test_dataprep_ok_clean() -> None:
    result = run_mypy(FIXTURES / "dataprep_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout


def test_tile_reps_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_tile_reps.py"),
        "tile reps must be non-negative, got [-1]",
    )


def test_repeat_neg_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_repeat_neg.py"),
        "repeat repeats must be non-negative, got -2",
    )


def test_repeat_axis_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_repeat_axis.py"),
        "repeat axis 5 is out of range for 2 dimensions",
        "(2, 3)",
    )


def test_pad_neg_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_pad_neg.py"),
        "pad pad_width entries must be non-negative, got (-1, -1)",
    )


def test_pad_len_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_pad_len.py"),
        "pad pad_width entries do not match 2 dimensions",
        "(2, 3)",
    )


def test_split_equal_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_split_equal.py"),
        "split cannot divide axis 0 of size 4 into 3 equal sections (shape (4,))",
    )


def test_split_sections_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_split_sections.py"),
        "split sections must be positive, got 0",
    )


def test_split_axis_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_split_axis.py"),
        "split axis 5 is out of range for 2 dimensions",
        "(2, 3)",
    )


def test_select_ok_clean() -> None:
    result = run_mypy(FIXTURES / "select_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout


def test_unique_axis_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_unique_axis.py"),
        "unique axis 5 is out of range for 2 dimensions",
        "(2, 3)",
    )


def test_nonzero_rank_mismatch() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_nonzero_rank.py"),
        "nonzero on 0d arrays is not allowed",
        '"nonzero"',
    )


def test_where_rank_mismatch() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_where_rank.py"),
        "where on 0d arrays is not allowed",
        '"where"',
    )


def test_symexpr_ok_clean() -> None:
    result = run_mypy(FIXTURES / "symexpr_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout


def test_symexpr_product_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_symexpr_product.py"),
        "reshape changes total size: 2*N -> 3*N",
        "(2*N,)",
        "(3, N)",
    )


def test_sliding_ok_clean() -> None:
    result = run_mypy(FIXTURES / "sliding_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout


def test_swv_shape_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_swv_shape.py"),
        "must provide window_shape for all 2 dimensions",
        "(10, 8)",
    )


def test_swv_size_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_swv_size.py"),
        "window shape cannot be larger than input array shape",
        "window 20 over dim 10",
    )


def test_swv_neg_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_swv_neg.py"),
        "`window_shape` cannot contain negative values (got -1)",
    )


def test_swv_len_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_swv_len.py"),
        "must provide matching length window_shape and axis",
        "2 window entries for 1 axes",
    )


def test_swv_axis_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_swv_axis.py"),
        "sliding_window_view axis 5 is out of range for 2 dimensions",
        "(10, 8)",
    )


def test_validations_ok_clean_and_runs() -> None:
    result = run_mypy(FIXTURES / "validations_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    run = subprocess.run(
        [sys.executable, str(FIXTURES / "validations_ok.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    # Every shape the checker solved (the annotations) is numpy's too.
    assert run.stdout.strip().splitlines() == [
        "(4, 3)",
        "(2, 4, 3)",
        "(2, 3, 4)",
        "(2, 3, 4)",
        "(2, 3)",
        "(3, 3)",
        "(6,)",
        "(3, 4)",
        "(12,)",
        "(3,)",
        "(3, 2)",
        "(3,)",
        "(1,)",
        "(1099511627776, 0)",
    ]


# Each fixture's f() is a call numpy always rejects; the static error
# must be backed by the runtime one.
NUMPY_REJECTS = [
    "bad_rot90_same_axes",
    "bad_flip_dup_axis",
    "bad_squeeze_dup_axis",
    "bad_gradient_dup_axis",
    "bad_roll_shift_len",
    "bad_repeat_list_len",
    "bad_repeat_flat_len",
    "bad_norm_ord_rank",
    "bad_norm_ord_fro",
    "bad_norm_fro_display",
    "bad_partition_kth",
    "bad_argpartition_kth",
    "bad_take_index_oob",
    "bad_take_flat_oob",
    "bad_matmul_scalar",
    "bad_matmul_0d",
    "bad_convolve_empty",
    "bad_correlate_empty",
    "bad_huge_dim",
    "bad_sum_dup_axis",
]


@pytest.mark.parametrize("name", NUMPY_REJECTS)
def test_numpy_rejects_at_runtime(name: str) -> None:
    f = runpy.run_path(str(FIXTURES / f"{name}.py"))["f"]
    with pytest.raises((ValueError, IndexError)):
        f()


def test_rot90_same_axes() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_rot90_same_axes.py"),
        "rot90 axes must be different (got (2, -1)",
        "(2, 3, 4)",
    )


def test_flip_duplicate_axis() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_flip_dup_axis.py"),
        "flip has duplicate axis -2 in (0, -2)",
        "(3, 4)",
    )


def test_squeeze_duplicate_axis() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_squeeze_dup_axis.py"),
        "squeeze has duplicate axis -2 in (1, -2)",
        "(2, 1, 3)",
    )


def test_gradient_duplicate_axis() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_gradient_dup_axis.py"),
        "gradient has duplicate axis 0 in (0, 0)",
        '"gradient"',
    )


def test_roll_shift_length_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_roll_shift_len.py"),
        "roll shift has 2 entries for 3 axes",
        "shift (1, 2), axis (0, 1, 2)",
    )


def test_repeat_list_length_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_repeat_list_len.py"),
        "repeat repeats has 3 entries for axis 0 of size 2",
        "(2, 3)",
    )


def test_repeat_flat_length_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_repeat_flat_len.py"),
        "repeat repeats has 2 entries for the flattened size 6",
    )


def test_norm_ord_rank_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_norm_ord_rank.py"),
        "linalg.norm with ord=2 reduces 3 dimensions",
        "(2, 3, 4)",
    )


def test_norm_fro_on_one_axis() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_norm_ord_fro.py"),
        "linalg.norm ord='fro' is a matrix norm and needs 2 axes, got 1",
        "(2, 3, 4)",
    )


def test_norm_fro_display_on_one_axis() -> None:
    # Spelled inline, "fro" is only a literal inside each overload
    # candidate. Once the hook rejects them all, mypy's fallback infers
    # plain str, finds no erased target, and reports its own overload
    # error instead of ours. The call still fails, as numpy's does.
    assert_single_error(
        run_mypy(FIXTURES / "bad_norm_fro_display.py"),
        'No overload variant of "norm" matches argument types',
    )


def test_partition_kth_out_of_bounds() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_partition_kth.py"),
        "partition kth 4 is out of bounds for axis -1 of size 4",
    )


def test_argpartition_kth_out_of_bounds() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_argpartition_kth.py"),
        "argpartition kth -5 is out of bounds for axis 1 of size 4",
    )


def test_take_index_out_of_bounds() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_take_index_oob.py"),
        "take index 7 is out of bounds for axis 1 of size 4",
    )


def test_take_flat_index_out_of_bounds() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_take_flat_oob.py"),
        "take index 12 is out of bounds for the flattened size 12",
    )


def test_matmul_scalar_operand() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_matmul_scalar.py"),
        "matmul operand 1 has no dimensions (a scalar float)",
    )


def test_matmul_0d_operand() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_matmul_0d.py"),
        "matmul operand 0 has no dimensions (a 0-d array)",
    )


def test_convolve_empty_operand() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_convolve_empty.py"),
        "convolve first array argument cannot be empty",
        "(0,) and (3,)",
    )


def test_correlate_empty_operand() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_correlate_empty.py"),
        "correlate second array argument cannot be empty",
        "(3,) and (0,)",
    )


def test_huge_dim_rejected() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_huge_dim.py"),
        "dimension 1000000000000000000000000000000 exceeds the maximum "
        "allowed dimension 9223372036854775807",
    )


def test_sum_duplicate_axis() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_sum_dup_axis.py"),
        "duplicate axis -2",
    )


# Error codes (#66): each kind of numty diagnostic carries its own code, a
# sub-code of misc, so it can be silenced or selected on its own while old
# `ignore[misc]` / `disable_error_code = misc` setups keep working.
@pytest.mark.parametrize(
    ("fixture", "code"),
    [
        ("bad_matmul_inner.py", "numty-shape"),
        ("bad_concat_dim.py", "numty-shape"),
        ("bad_where_broadcast.py", "numty-broadcast"),
        ("bad_f64_conflict.py", "numty-callsite"),
        ("bad_min_axis.py", "numty-axis"),
        ("bad_index_range.py", "numty-axis"),
        ("bad_bail_code.py", "numty-bail"),
        ("bad_f64_empty.py", "numty-annotation"),
    ],
)
def test_error_code_tagged(fixture: str, code: str) -> None:
    assert_single_error(run_mypy(FIXTURES / fixture), f"  [{code}]")


def test_broadcast_operator_error_code() -> None:
    # The operator form adds mypy's own [operator] error; ours is tagged.
    assert_operator_mismatch(
        run_mypy(FIXTURES / "bad_broadcast.py"),
        "broadcast shapes disagree",
        "  [numty-broadcast]",
    )


def test_no_numty_error_falls_back_to_misc() -> None:
    # One run over every bad fixture: no diagnostic is left on the default
    # misc code (none of the fixtures trips a mypy-core misc error).
    result = run_mypy(*sorted(FIXTURES.glob("bad_*.py")))
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert errors, result.stdout
    assert not [line for line in errors if line.endswith("[misc]")], result.stdout


def test_bail_fixture_runs_cleanly() -> None:
    # Runtime oracle: the bail is on correct code.
    assert _run_fixture("bad_bail_code.py") == ["(2, 3)"]


def test_disable_bail_keeps_real_errors() -> None:
    flags = ("--disable-error-code", "numty-bail")
    result = run_mypy(FIXTURES / "bad_bail_code.py", flags=flags)
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout, result.stdout
    assert_single_error(
        run_mypy(FIXTURES / "bad_matmul_inner.py", flags=flags),
        "matmul inner dimensions disagree",
        "  [numty-shape]",
    )


def test_disable_misc_still_covers_numty_codes() -> None:
    # Backward compatibility: misc is the parent code of every numty code.
    for fixture in ("bad_bail_code.py", "bad_matmul_inner.py"):
        result = run_mypy(FIXTURES / fixture, flags=("--disable-error-code", "misc"))
        assert result.returncode == 0, result.stdout
        assert "error:" not in result.stdout, result.stdout


def test_targeted_ignores() -> None:
    # ignore[numty-bail] and ignore[misc] silence the bails, ignore[misc]
    # silences the shape error, ignore[numty-bail] does not.
    result = run_mypy(FIXTURES / "bad_ignore_codes.py")
    assert_single_error(result, ":14: error: matmul inner dimensions disagree", "[numty-shape]")
    assert 'Error code "numty-shape" not covered by "type: ignore[numty-bail]"' in (
        result.stdout
    ), result.stdout


def test_every_plugin_fail_site_passes_an_error_code() -> None:
    # Static guard: fixtures only reach some fail sites, so check them all
    # in the source. A new ctx.api.fail without code= would print [misc].
    import ast

    tree = ast.parse((ROOT / "numty_plugin.py").read_text())
    sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fail"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "api"
    ]
    assert len(sites) > 100, len(sites)
    missing = [n.lineno for n in sites if not any(k.arg == "code" for k in n.keywords)]
    assert not missing, f"ctx.api.fail without code= at numty_plugin.py lines {missing}"


def test_flexarray_portable_fixture_under_mypy() -> None:
    # Same fixture test_ty.py runs under ty: both checkers flag exactly
    # the `# BUG` lines, so the FlexArray spelling is safe in a dual gate.
    fixture = FIXTURES / "flex_portable.py"
    bugs = [
        i for i, line in enumerate(fixture.read_text().splitlines(), 1)
        if "# BUG" in line
    ]
    result = run_mypy(fixture, flags=("--warn-unused-ignores",))
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert [int(e.split(":")[1]) for e in errors] == bugs, result.stdout


def test_numpy_twins_keep_stub_types() -> None:
    # np.ma / np.emath / numpy.matlib share names with the ndarray API;
    # exact-fullname dispatch leaves their MaskedArray, complex and
    # matrix stub types alone (issue #59).
    result = run_mypy(FIXTURES / "numpy_twins_ok.py")
    assert result.returncode == 0, result.stdout
    assert _run_fixture("numpy_twins_ok.py") == [
        "[False, True, False] [1.0, 0.0, 3.0] 2",
        "MaskedArray (4, 3) False",
        "False 6",
        "(3, 2)",
        "complex128 complex128",
        "matrix (1, 3)",
    ]


def test_matlib_zeros_is_not_a_vector() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_matlib_vector.py"),
        "Incompatible types in assignment",
        'expression has type "matrix[tuple[int, int], dtype[float64]]"',
    )
