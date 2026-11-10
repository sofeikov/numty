"""Static checks: each bad fixture must fail mypy with exactly one error."""
import os
import re
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
# Under pytest-xdist each worker gets its own cache dir (via
# PYTEST_XDIST_WORKER) so parallel mypy runs do not thrash each other.
_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "")
CACHE = ROOT / (".mypy-test-cache" if not _WORKER else f".mypy-test-cache-{_WORKER}")

# Session batch: one mypy run over every fixture serves all default-flag
# single-file checks from memory. Per-file subprocesses cost ~0.1s of
# startup each (≈20s for the suite); one batched run is ~2-3s cold and
# ~0.08s warm. Flagged / multi-file runs still use subprocesses (with a
# per-flag cache dir so option changes do not invalidate the default cache).
_BATCH_RESULTS: dict[str, tuple[int, str]] = {}
_BATCH_READY = False


def _ensure_batch() -> None:
    global _BATCH_READY
    if _BATCH_READY:
        return
    _BATCH_READY = True
    files = sorted(FIXTURES.glob("*.py"))
    if not files:
        return
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--cache-dir", str(CACHE), *map(str, files)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    per_file: dict[str, list[str]] = {f.name: [] for f in files}
    for line in result.stdout.splitlines():
        # mypy prints paths relative to cwd (tests/fixtures/<name>:...).
        # Match by prefix to avoid false positives from message text.
        for f in files:
            if line.startswith(f"tests/fixtures/{f.name}:") or line.startswith(f"{f}:"):
                per_file[f.name].append(line)
                break
            # Absolute-path fallback (cwd variations).
            if line.startswith(f"{str(f)}:"):
                per_file[f.name].append(line)
                break
    for f in files:
        lines = per_file[f.name]
        n_errors = sum(1 for line in lines if ": error:" in line)
        if n_errors == 0:
            stdout = ("".join(line + "\n" for line in lines)
                      + "Success: no issues found in 1 source file\n")
            _BATCH_RESULTS[f.name] = (0, stdout)
        else:
            stdout = ("".join(line + "\n" for line in lines)
                      + f"Found {n_errors} error{'s' if n_errors != 1 else ''}"
                        f" in 1 file (checked 1 source file)\n")
            _BATCH_RESULTS[f.name] = (1, stdout)


@pytest.fixture(scope="session", autouse=True)
def _fresh_mypy_cache() -> None:
    global _BATCH_READY
    _BATCH_RESULTS.clear()
    _BATCH_READY = False
    shutil.rmtree(CACHE, ignore_errors=True)
    # Flag-specific caches (see run_mypy) live beside CACHE.
    for extra in CACHE.parent.glob(CACHE.name + "-flags-*"):
        shutil.rmtree(extra, ignore_errors=True)
    yield
    shutil.rmtree(CACHE, ignore_errors=True)
    for extra in CACHE.parent.glob(CACHE.name + "-flags-*"):
        shutil.rmtree(extra, ignore_errors=True)
    _BATCH_RESULTS.clear()
    _BATCH_READY = False


def _cache_for_flags(flags: tuple[str, ...]) -> Path:
    if not flags:
        return CACHE
    import hashlib

    digest = hashlib.md5(" ".join(flags).encode()).hexdigest()[:12]
    return CACHE.parent / f"{CACHE.name}-flags-{digest}"


def run_mypy(*paths: Path, flags: tuple[str, ...] = ()) -> "subprocess.CompletedProcess[str]":
    # Fast path: default-flag single-fixture checks are served from the
    # session batch (no new subprocess, no cache thrashing).
    if not flags and len(paths) == 1:
        try:
            single = Path(paths[0])
        except Exception:
            single = None  # type: ignore[assignment]
        if single is not None and single.parent == FIXTURES and single.suffix == ".py":
            _ensure_batch()
            hit = _BATCH_RESULTS.get(single.name)
            if hit is not None:
                returncode, stdout = hit
                return subprocess.CompletedProcess(
                    args=[], returncode=returncode, stdout=stdout, stderr=""
                )
    # PYTHONPATH (not MYPYPATH) is what lets mypy import the local plugin.
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    cache_dir = _cache_for_flags(flags)
    return subprocess.run(
        [sys.executable, "-m", "mypy", "--cache-dir", str(cache_dir), *flags, *map(str, paths)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def assert_single_error(
    result: "subprocess.CompletedProcess[str]",
    *fragments: str,
    line: "int | None" = None,
) -> None:
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 1, result.stdout
    for fragment in fragments:
        assert fragment in errors[0], result.stdout
    if line is not None:
        assert int(errors[0].split(":")[1]) == line, result.stdout


def reveals_by_line(result: "subprocess.CompletedProcess[str]") -> dict[int, str]:
    """Index mypy `Revealed type` notes by source line (single-file run).

    Annotation pins accept Any, so exact-shape claims need golden reveals
    compared per line (issue #71): unordered substring checks let swapped
    shapes pass.
    """
    out: dict[int, str] = {}
    for text in result.stdout.splitlines():
        if "Revealed type" not in text:
            continue
        lineno = int(text.split(":")[1])
        assert lineno not in out, result.stdout
        out[lineno] = text.split("Revealed type is", 1)[1].strip()
    return out


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


def test_unknown_broadcast_keeps_rank_and_dtype() -> None:
    # Issue #56: one unprovable axis used to dissolve the whole result
    # into Any; now the axis is a fresh unknown (or the literal, when
    # one side is a literal) and rank, known dims and dtype survive.
    result = run_mypy(FIXTURES / "unknown_broadcast_ok.py")
    assert result.returncode == 0, result.stdout
    notes = [
        re.sub(r"\?U@\d+[a-z]?", "?U", line.split('Revealed type is "', 1)[1])
        for line in result.stdout.splitlines()
        if "Revealed type" in line
    ]
    f64 = "numpy.dtype[numpy.float64]"
    assert notes == [
        f'numpy.ndarray[tuple[?U, Literal[3]], {f64}]"',
        f'numpy.ndarray[tuple[?U, Literal[3]], {f64}]"',
        f'numpy.ndarray[tuple[Literal[4], Literal[3]], {f64}]"',
        f'numpy.ndarray[tuple[P, D], {f64}]"',
        f'numpy.ndarray[tuple[?U, Literal[3]], {f64}]"',
    ], result.stdout
    assert _run_fixture("unknown_broadcast_ok.py") == [
        "(3, 3) float64",
        "(4, 3)",
        "(40, 7)",
    ]


def test_unknown_broadcast_downstream_error() -> None:
    # Issue #56: the known inner dims still check through the unknown,
    # so the 3-vs-4 matmul bug is caught instead of passing silently.
    assert_single_error(
        run_mypy(FIXTURES / "bad_unknown_broadcast_downstream.py"),
        "matmul inner dimensions disagree: 3 vs 4",
        line=15,
    )


def test_unknown_slice_keeps_sibling_dims() -> None:
    # Issue #57: a variable slice bound used to bail the whole getitem
    # to Any; now the sliced axis is a fresh unknown and the rest keeps
    # checking (a full-range reversal keeps the symbolic dim exactly).
    result = run_mypy(FIXTURES / "unknown_slice_ok.py")
    assert result.returncode == 0, result.stdout
    notes = [
        re.sub(r"\?U@\d+[a-z]?", "?U", line.split('Revealed type is "', 1)[1])
        for line in result.stdout.splitlines()
        if "Revealed type" in line
    ]
    f64 = "numpy.dtype[numpy.float64]"
    assert notes == [
        f'numpy.ndarray[tuple[?U, Literal[3]], {f64}]"',
        f'numpy.ndarray[tuple[?U, Literal[7]], {f64}]"',
        f'numpy.ndarray[tuple[Literal[5], ?U], {f64}]"',
        f'numpy.ndarray[tuple[Literal[5], Literal[3]], {f64}]"',
        f'numpy.ndarray[tuple[N, Literal[3]], {f64}]"',
        f'numpy.ndarray[tuple[N-1, Literal[3]], {f64}]"',
    ], result.stdout
    assert _run_fixture("unknown_slice_ok.py") == [
        "(32, 3)",
        "(4, 7)",
        "(5, 3)",
        "(5, 3)",
        "(6, 3)",
        "(5, 3)",
    ]


def test_unknown_slice_downstream_error() -> None:
    # Issue #57: the minibatch slice keeps the feature width, so the
    # 3-vs-7 matmul bug is caught instead of passing silently.
    assert_single_error(
        run_mypy(FIXTURES / "bad_unknown_slice_downstream.py"),
        "matmul inner dimensions disagree: 3 vs 7",
        line=10,
    )


def test_good_path_precision_visible() -> None:
    result = run_mypy(FIXTURES / "static_ok.py")
    assert result.returncode == 0, result.stdout
    notes = [line for line in result.stdout.splitlines() if "Revealed type" in line]
    assert len(notes) == 35, result.stdout
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


def test_reshape_ts_tail_mismatch() -> None:
    # Issue #48: reshape over *Ts claims the whole target shape, so a
    # (*Ts, 6) annotation on a `reshape(6)` body is a return-value error.
    assert_single_error(
        run_mypy(FIXTURES / "bad_ts_reshape_tail.py"),
        "tuple[Literal[6]]",
        "tuple[*Ts, Literal[6]]",
    )


def test_transpose_ts_mismatch() -> None:
    # Issue #48: no annotation can spell a reversed *Ts, so flipping
    # (*Ts, 2) into (2, *Ts) is a return-value error, not a silent pass.
    assert_single_error(
        run_mypy(FIXTURES / "bad_ts_transpose.py"),
        "tuple[*Ts, Literal[2]]",
        "tuple[Literal[2], *Ts]",
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


def test_everyday_ok() -> None:
    # Issue #62: ordinary numpy code checks clean — int bookkeeping on
    # .shape/.len, 0-d results as numpy scalars, size-1 einsum labels,
    # shadowed string-spec words, axis 0/-1 on 0-d arrays, plain NDArray
    # into a head, and literal slices on linear dims.
    result = run_mypy(FIXTURES / "everyday_ok.py")
    assert result.returncode == 0, result.stdout
    notes = {
        line: re.sub(r"\?U@\d+[a-z]?", "?U", shape)
        for line, shape in reveals_by_line(result).items()
    }
    assert notes == {
        43: '"float"',
        45: '"float"',
        47: '"float"',
        49: '"float"',
        51: '"numpy.bool[bool]"',
        55: '"numpy.ndarray[tuple[time, freq], numpy.dtype[numpy.float64]]"',
        56: '"numpy.ndarray[tuple[n, d], numpy.dtype[numpy.float64]]"',
        57: '"numpy.ndarray[tuple[batch, input], numpy.dtype[numpy.float64]]"',
        62: '"numpy.float64"',
        63: '"numpy.ndarray[tuple[Literal[1]], numpy.dtype[numpy.float64]]"',
        64: '"numpy.ndarray[tuple[Literal[2]], numpy.dtype[numpy.float64]]"',
        65: '"numpy.ndarray[tuple[()], numpy.dtype[numpy.float64]]"',
        85: '"numpy.ndarray[tuple[Literal[2], Literal[3]], numpy.dtype[numpy.float64]]"',
        86: '"numpy.ndarray[tuple[?U], numpy.dtype[numpy.float64]]"',
        87: '"numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]"',
    }, result.stdout
    assert _run_fixture("everyday_ok.py") == ["ok"]


def test_everyday_bad_still_fails() -> None:
    # The same idioms with real bugs: size clashes, diagonal collapse,
    # and out-of-range 0-d axes keep failing loudly.
    result = run_mypy(FIXTURES / "bad_everyday.py")
    assert result.returncode == 1, result.stdout
    lines = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert any(
        "einsum label 'j' has conflicting sizes: 2 vs 3" in line for line in lines
    ), result.stdout
    assert any(
        "einsum label 'i' has conflicting sizes: 1 vs 3" in line for line in lines
    ), result.stdout
    assert sum("out of range for 0 dimensions" in line for line in lines) == 3, (
        result.stdout
    )


def test_alias_int_dims_ok() -> None:
    # Issue #62: bare ints never survive an alias definition (mypy core
    # checks the right-hand side itself), so spell the sizes with
    # Literal; every alias form then checks and infers exactly.
    result = run_mypy(FIXTURES / "alias_int_dims_ok.py")
    assert result.returncode == 0, result.stdout
    assert reveals_by_line(result) == {
        27: '"numpy.ndarray[tuple[Literal[3], Literal[4]], numpy.dtype[numpy.float64]]"',
        28: '"numpy.ndarray[tuple[Literal[3], Literal[4]], numpy.dtype[numpy.float64]]"',
    }, result.stdout
    assert _run_fixture("alias_int_dims_ok.py") == ["ok"]


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


def test_issue58_kron_pinv_randn_percentile_ok() -> None:
    # Issue #58 slice: the four repro calls now solve precise shapes.
    result = run_mypy(FIXTURES / "issue58_kron_pinv_randn_percentile_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    assert notes[9] == (
        '"numpy.ndarray[tuple[Literal[8], Literal[15]], numpy.dtype[numpy.float64]]"'
    ), result.stdout
    assert notes[10] == (
        '"numpy.ndarray[tuple[Literal[4], Literal[3]], numpy.dtype[numpy.float64]]"'
    ), result.stdout
    assert notes[11] == (
        '"numpy.ndarray[tuple[Literal[3], Literal[4]], numpy.dtype[numpy.float64]]"'
    ), result.stdout
    assert notes[13] == (
        '"numpy.ndarray[tuple[Literal[3], Literal[6]], numpy.dtype[numpy.float64]]"'
    ), result.stdout


def test_kron_broadcast_mismatch() -> None:
    assert_operator_mismatch(
        run_mypy(FIXTURES / "bad_kron_broadcast.py"),
        "broadcast shapes disagree at trailing axis 0",
        "15 vs 4",
        "(8, 15)",
        "(4,)",
    )


def test_pinv_rank_mismatch() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_pinv_rank.py"),
        "linalg.pinv requires at least 2 dimensions, got shape (5,)",
        '"pinv"',
    )


def test_percentile_axis_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_percentile_axis.py"),
        "percentile axis 5 is out of range for 2 dimensions",
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


def test_arith_ok_stays_clean() -> None:
    # Issue #54: floordiv/mod/divmod/xor/shift/int-&/|/bool-flag keep shapes.
    result = run_mypy(FIXTURES / "arith_ok.py")
    assert result.returncode == 0, result.stdout
    notes = [line for line in result.stdout.splitlines() if "Revealed type" in line]
    assert len(notes) == 10, result.stdout
    assert sum("tuple[Literal[2], Literal[3]]" in n and "float64" in n for n in notes) == 3
    assert sum("signedinteger" in n for n in notes) == 5
    assert sum("numpy.bool" in n for n in notes) == 2
    assert _run_fixture("arith_ok.py") == ["ok"]


def test_arith_broadcast_mismatch() -> None:
    # Issue #54: the newly hooked operators fail with the binary wording.
    result = run_mypy(FIXTURES / "bad_arith_broadcast.py")
    assert result.returncode == 1, result.stdout
    ours = [
        line
        for line in result.stdout.splitlines()
        if "broadcast shapes disagree at trailing axis 0" in line
    ]
    assert len(ours) == 4, result.stdout
    for line in ours:
        assert "3 vs 5" in line and "(2, 3)" in line and "(5,)" in line, result.stdout
    rest = [
        line
        for line in result.stdout.splitlines()
        if ": error:" in line and line not in ours
    ]
    assert rest and all("Unsupported operand" in line for line in rest), result.stdout


def test_dtype_ok_stays_clean() -> None:
    # Issue #61: RNG, float ufuncs, truediv and int ops keep NumPy dtypes.
    result = run_mypy(FIXTURES / "dtype_ok.py")
    assert result.returncode == 0, result.stdout
    notes = [line for line in result.stdout.splitlines() if "Revealed type" in line]
    assert len(notes) == 11, result.stdout
    assert sum("_32Bit" in n for n in notes) == 4, result.stdout
    assert sum("_16Bit" in n for n in notes) == 3, result.stdout
    # Scalar-only calls are numpy scalars (issue #62), not 0-d arrays:
    # np.divide(7, 2) reveals exactly numpy.float64.
    assert sum('"numpy.float64"' in n and "tuple" not in n for n in notes) == 1, result.stdout
    assert _run_fixture("dtype_ok.py") == ["ok"]


def test_dtype_wrong_fails() -> None:
    # Issue #61: confident wrong dtypes are rejected; float16 linalg fails.
    result = run_mypy(FIXTURES / "bad_dtype.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 9, result.stdout
    assert "floating[_32Bit]" in errors[0] and "float64" in errors[0], result.stdout
    assert "floating[_16Bit]" in errors[1] and "float64" in errors[1], result.stdout
    assert "floating[_32Bit]" in errors[2] and "float64" in errors[2], result.stdout
    assert "float64" in errors[3] and "signedinteger[_64Bit]" in errors[3], result.stdout
    assert "float64" in errors[4] and "signedinteger[_64Bit]" in errors[4], result.stdout
    assert "unsupported in linalg.solve" in errors[5], result.stdout
    assert "float64" in errors[7] and "complex128" in errors[7], result.stdout
    assert "signedinteger[_64Bit]" in errors[8] and "signedinteger[_32Bit]" in errors[8], result.stdout


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


def test_callsite_conflict_in_comprehension() -> None:
    # Issue #44: the process-lifetime `_seen_conflicts` dedup consumed the
    # single allowed emission during mypy's throw-away inference pass
    # inside comprehensions / generator expressions / lambdas, so the pass
    # that counts stayed quiet. All seven calls must error, exactly once.
    result = run_mypy(FIXTURES / "bad_callsite_conflict_comp.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 7, result.stdout
    assert [int(line.split(":")[1]) for line in errors] == [12, 13, 14, 15, 16, 17, 18]
    for line in errors:
        assert "call-site shape conflict for 'features'" in line, result.stdout
        assert "[numty-callsite]" in line, result.stdout


def test_callsite_conflict_survives_dmypy_recheck(tmp_path: Path) -> None:
    # Issue #44: dmypy keeps one plugin instance across checks, so the
    # process-lifetime dedup hid the conflict after any recheck. Both the
    # cold check and the check after a comment-only edit must report it.
    dmypy = Path(sys.executable).parent / "dmypy"
    if not dmypy.exists():
        pytest.skip("dmypy is not installed")
    proj = tmp_path / "proj"
    proj.mkdir()
    src = proj / "repro.py"
    src.write_text(
        "import numpy as np\n"
        "from numty import F64, batch, features, hid\n"
        "\n"
        "\n"
        "def proj(x: F64[batch, features], w: F64[features, hid]) -> F64[batch, hid]:\n"
        "    return np.matmul(x, w)\n"
        "\n"
        "\n"
        "bad = proj(np.zeros((4, 3)), np.ones((2, 5)))\n"
        "# edit-marker v1\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT), "MYPYPATH": str(ROOT)}
    status = tmp_path / "dmypy.json"
    cache = tmp_path / "cache"
    base = [str(dmypy), "--status-file", str(status)]
    try:
        start = subprocess.run(
            [*base, "start", "--", "--config-file", str(ROOT / "mypy.ini"),
             "--cache-dir", str(cache)],
            cwd=proj, env=env, capture_output=True, text=True, timeout=300,
        )
        assert start.returncode == 0, start.stdout + start.stderr
        first = subprocess.run(
            [*base, "check", "repro.py"],
            cwd=proj, env=env, capture_output=True, text=True, timeout=300,
        )
        assert "call-site shape conflict for 'features'" in first.stdout, first.stdout
        src.write_text(src.read_text().replace("edit-marker v1", "edit-marker v2"))
        second = subprocess.run(
            [*base, "check", "repro.py"],
            cwd=proj, env=env, capture_output=True, text=True, timeout=300,
        )
        assert "call-site shape conflict for 'features'" in second.stdout, second.stdout
    finally:
        subprocess.run([*base, "stop"], cwd=proj, env=env,
                       capture_output=True, text=True, timeout=120)


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


def test_pad_time_kernel_clean_and_runs() -> None:
    # #33: a leading dim plus Unpack[Ts] in a head needs no type: ignore,
    # (n, target, *frames.shape[1:]) builds (int, int, *Ts) past numpy's
    # shape bound, `int` names the runtime counts, and the kernel runs on
    # 2-D and 3-D frames.
    result = run_mypy(FIXTURES / "pad_time_ok.py", flags=("--warn-unused-ignores",))
    assert result.returncode == 0, result.stdout
    assert "# type: ignore" not in (FIXTURES / "pad_time_ok.py").read_text()
    assert _run_fixture("pad_time_ok.py") == [
        "(2, 4, 2) (2, 4, 3, 2)",
        "[0.0, 0.0, 2.0, 4.0] [8.0, 10.0, 12.0, 14.0]",
        "(2, 3, 2)",
    ]


def test_pad_time_kernel_keeps_time_axis() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_pad_time_axis.py"),
        "Incompatible return value",
        "tuple[int, int, total, *Ts]",
        "tuple[int, int, *Ts]",
    )


def test_pad_time_kernel_swapped_frame_axes() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_pad_time_swap.py"),
        "tuple[Literal[8], Literal[3], Literal[2]]",
        "tuple[Literal[8], Literal[2], Literal[3]]",
    )


def test_gather_over_ts_prefix_solves() -> None:
    # A lone integer array on the known prefix of (total, *Ts) gathers in
    # place, so the wrong leading dim is caught instead of going Any.
    assert_single_error(
        run_mypy(FIXTURES / "bad_gather_ts.py"),
        "Incompatible return value",
        "tuple[K, *Ts]",
        "tuple[total, *Ts]",
    )


def test_mask_over_ts_prefix_counts_once() -> None:
    # A k-d mask consumes k prefix axes for one data-dependent count.
    assert_single_error(
        run_mypy(FIXTURES / "bad_mask_ts.py"),
        "Incompatible return value",
        "tuple[?U@12, *Ts]",
        "tuple[N, *Ts]",
    )


def test_mask_nd_collapses_to_rank_one() -> None:
    # Issue #49: an n-D mask consumes n axes for one count, so x[x > 0]
    # on (N, M) is rank 1 and (4, 5, 3) masked on the first two axes is
    # (?U, 3); numpy agrees on rank and trailing dims.
    result = run_mypy(FIXTURES / "mask_nd_ok.py")
    assert result.returncode == 0, result.stdout
    notes = [line for line in result.stdout.splitlines() if "Revealed type" in line]
    assert len(notes) == 2, result.stdout
    assert "tuple[?U@14]" in notes[0], result.stdout
    assert "tuple[?U@18, Literal[3]]" in notes[1], result.stdout
    assert _run_fixture("mask_nd_ok.py") == ["ok"]


def test_split_advanced_moves_to_front() -> None:
    # Issue #49: int/mask advanced indices split by a slice move their
    # block to the front: (5, 3) not (3, 5), (?U, 3) not (3, ?U).
    result = run_mypy(FIXTURES / "adv_sep_ok.py")
    assert result.returncode == 0, result.stdout
    notes = [line for line in result.stdout.splitlines() if "Revealed type" in line]
    assert len(notes) == 2, result.stdout
    assert "tuple[Literal[5], Literal[3]]" in notes[0], result.stdout
    assert "tuple[?U@13, Literal[3]]" in notes[1], result.stdout
    assert _run_fixture("adv_sep_ok.py") == ["ok"]


def test_mask_length_mismatch() -> None:
    # Issue #49: a Bool[6] mask on a size-4 axis fails with numpy's wording.
    assert_single_error(
        run_mypy(FIXTURES / "bad_mask_len.py"),
        "boolean index did not match indexed array along axis 0",
        "size of axis is 4 but size of corresponding boolean axis is 6",
    )


def test_mask_nd_axis_out_of_range() -> None:
    # Issue #49: the 2-D mask is rank 1, so mean(axis=1) is an axis error.
    assert_single_error(
        run_mypy(FIXTURES / "bad_mask_nd_axis.py"),
        "axis 1 is out of range for 1 dimensions",
        "(?U@8,)",
    )


def test_inplace_ok_stays_clean() -> None:
    # Issue #50: legal in-place updates keep their shape and stay clean.
    result = run_mypy(FIXTURES / "inplace_ok.py")
    assert result.returncode == 0, result.stdout
    notes = [line for line in result.stdout.splitlines() if "Revealed type" in line]
    assert len(notes) == 2, result.stdout
    assert "tuple[Literal[2], Literal[3]]" in notes[0], result.stdout
    assert "tuple[Literal[3], Literal[4]]" in notes[1], result.stdout
    assert _run_fixture("inplace_ok.py") == ["ok"]


def test_inplace_broadcast_mismatch() -> None:
    # Issue #50: in-place uses the binary broadcast wording.
    result = run_mypy(FIXTURES / "bad_inplace_broadcast.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 2, result.stdout
    assert "broadcast shapes disagree at trailing axis 0" in errors[0], result.stdout
    assert "(2, 3)" in errors[0] and "(4,)" in errors[0], result.stdout
    assert "broadcast shapes disagree at trailing axis 0" in errors[1], result.stdout
    assert "(3,)" in errors[1] and "(4,)" in errors[1], result.stdout


def test_inplace_grow_does_not_fit() -> None:
    # Issue #50: a broadcast that grows the target fails in place.
    result = run_mypy(FIXTURES / "bad_inplace_grow.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 3, result.stdout
    assert "in-place result (2, 3) does not fit target (3,)" in errors[0], result.stdout
    assert "in-place result (4, 3) does not fit target (1, 3)" in errors[1], result.stdout
    assert (
        "in-place result (batch, features) does not fit target (features,)"
        in errors[2]
    ), result.stdout


def test_setitem_ok_stays_clean() -> None:
    # Issue #51: legal indexed assignments keep their shape and stay clean.
    result = run_mypy(FIXTURES / "setitem_ok.py")
    assert result.returncode == 0, result.stdout
    notes = [line for line in result.stdout.splitlines() if "Revealed type" in line]
    assert len(notes) == 1, result.stdout
    assert "tuple[Literal[5], Literal[3]]" in notes[0], result.stdout
    assert _run_fixture("setitem_ok.py") == ["ok"]


def test_setitem_shape_mismatch() -> None:
    # Issue #51: values must broadcast one-directionally into the target.
    result = run_mypy(FIXTURES / "bad_setitem_shape.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 6, result.stdout
    assert "cannot assign value of shape (4,) into target of shape (5,)" in errors[0]
    assert "cannot assign value of shape (3, 3) into target of shape (2, 3)" in errors[1]
    assert "cannot assign value of shape (3,) into target of shape (4,)" in errors[2]
    assert "cannot assign value of shape (2, 1, 3) into target of shape (2, 3)" in errors[3]
    assert "cannot assign value of shape (2, 2) into target of shape" in errors[4]
    assert "cannot assign value of shape (4,) into target of shape (3,)" in errors[5]


def test_setitem_oob() -> None:
    # Issue #51: out-of-range and mask-length errors fire on writes too.
    result = run_mypy(FIXTURES / "bad_setitem_oob.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 4, result.stdout
    assert "index 5 is out of range for axis 0 with size 5" in errors[0]
    assert "index -6 is out of range for axis 0 with size 5" in errors[1]
    assert "index 3 is out of range for axis 1 with size 3" in errors[2]
    assert "boolean index did not match indexed array along axis 0" in errors[3]


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
        # A sampler's mixed size= stays with the stubs: its answer is
        # final, so a literal there could not be rebound to (n, d).
        '"numpy.ndarray[tuple[Any, ...], numpy.dtype[numpy.float64]]"',
        '"numpy.ndarray[tuple[Any, ...], numpy.dtype[numpy.signedinteger[_64Bit]]]"',
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
        "[(5, 3), (5, 3, 4), (5, 3, 4), (5, 3)]",
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


def test_callsite_conflict_methods() -> None:
    # Issue #53: instance methods, staticmethods, classmethods and
    # __call__ report the same conflict a free function reports.
    result = run_mypy(FIXTURES / "bad_callsite_method.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 4, result.stdout
    assert "call-site shape conflict for 'I'" in errors[0], result.stdout
    assert "3 (argument 'x', axis 1) vs 4 (argument 'w', axis 0)" in errors[0]
    for err in errors[1:]:
        assert "call-site shape conflict for 'N'" in err, result.stdout
        assert "3 (argument 'x', axis 0) vs 5 (argument 'y', axis 0)" in err


def test_callsite_conflict_flexible_formals() -> None:
    # Issue #53: Optional/PEP-604 bias, keyword-only args, *args,
    # Sequence/list elements and *Ts segments report conflicts too.
    result = run_mypy(FIXTURES / "bad_callsite_flexible.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 7, result.stdout
    assert "call-site shape conflict for 'M'" in errors[0], result.stdout
    assert "call-site shape conflict for 'M'" in errors[1], result.stdout
    for err in errors[2:6]:
        assert "call-site shape conflict for 'N'" in err, result.stdout
    assert "call-site shape conflict for 'Ts'" in errors[6], result.stdout
    assert "4 (argument 'x', axis 1) vs 5 (argument 'y', axis 1)" in errors[6]


def test_callsite_strspec_valid_calls_clean() -> None:
    # Issue #53: valid string-spec calls are clean in every position
    # that used to reject them, with exact revealed shapes.
    result = run_mypy(FIXTURES / "callsite_strspec_ok.py")
    assert result.returncode == 0, result.stdout
    assert reveals_by_line(result) == {
        47: '"numpy.ndarray[tuple[Literal[2], Literal[5]], numpy.dtype[numpy.float64]]"',
        48: '"numpy.ndarray[tuple[Literal[3], Literal[2]], numpy.dtype[numpy.float64]]"',
        49: '"numpy.ndarray[tuple[Literal[3], Literal[2]], numpy.dtype[numpy.float64]]"',
        50: '"None"',
        51: '"None"',
        52: '"None"',
        53: '"None"',
    }, result.stdout
    assert _run_fixture("callsite_strspec_ok.py") == ["ok"]


def test_rank_mismatch_note_names_declared_dims() -> None:
    # #35: a wrong-rank argument leaves the parameter's dim TypeVars
    # unsolved, so mypy's own error prints them as Never. That error is
    # kept as is; a note under it names the declared dims.
    result = run_mypy(FIXTURES / "bad_rank_dim_names.py")
    assert result.returncode == 1, result.stdout
    lines = [line.split(":", 1)[1] for line in result.stdout.splitlines()[:-1]]
    errors = [line for line in lines if ": error:" in line]
    assert len(errors) == 8, result.stdout
    assert all(line.endswith("  [arg-type]") for line in errors), result.stdout
    assert 'expected "ndarray[tuple[Never], dtype[floating[_32Bit]]]"' in errors[0]
    assert [line for line in lines if ": note:" in line] == [
        '25: note: "act" declares parameter "action" with shape (A,);'
        " the argument has shape (N, A) (rank 2, not 1)",
        '26: note: "act" declares parameter "action" with shape (A,);'
        " the argument has shape (N, A) (rank 2, not 1)",
        '27: note: "act" of "Policy" declares parameter "action" with shape (A,);'
        " the argument has shape (N, A) (rank 2, not 1)",
        '28: note: "head" declares parameter "x" with shape (3, K);'
        " the argument has shape (4,) (rank 1, not 2)",
        '29: note: "spec" declares parameter "x" with shape (rows, cols);'
        " the argument has shape (4,) (rank 1, not 2)",
        '30: note: "maybe" declares parameter "x" with shape (A,);'
        " the argument has shape (N, A) (rank 2, not 1)",
    ], result.stdout
    # Each note prints right under the error it explains.
    for i, line in enumerate(lines):
        if ": note:" in line:
            assert lines[i - 1].split(":")[0] == line.split(":")[0], result.stdout
            assert ": error:" in lines[i - 1], result.stdout
    # Same-rank literal mismatch: mypy already names Literal[3], no note.
    assert errors[-2].startswith("31: error:"), result.stdout
    assert 'expected "ndarray[tuple[Literal[3]]' in errors[-2], result.stdout
    # A Callable-typed callee has no name, and mypy prints tuple[A]: no note.
    assert errors[-1].startswith("33: error:"), result.stdout
    assert 'expected "ndarray[tuple[A]' in errors[-1], result.stdout


def test_rank_mismatch_note_follows_arg_type_ignores() -> None:
    # The note carries arg-type, so silencing the error silences it too.
    result = run_mypy(
        FIXTURES / "bad_rank_dim_names.py", flags=("--disable-error-code", "arg-type")
    )
    assert result.returncode == 0, result.stdout
    assert "note:" not in result.stdout, result.stdout


def test_rank_mismatch_note_skips_star_args_and_operators() -> None:
    # Only explicit calls with plain arguments get the note: a *pair
    # actual is not the array passed, and operators, indexing and `in`
    # report under [operator]/[index]/[assignment], whose ignores and
    # disables the note's arg-type would escape.
    fixture = FIXTURES / "rank_dim_names_ok.py"
    for flags in (
        ("--warn-unused-ignores",),
        tuple(
            arg
            for code in ("operator", "index", "assignment")
            for arg in ("--disable-error-code", code)
        ),
    ):
        result = run_mypy(fixture, flags=flags)
        assert result.returncode == 0, result.stdout
        assert result.stdout == "Success: no issues found in 1 source file\n", result.stdout


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


def test_stack_comprehension_solves() -> None:
    # #30: a dynamically built sequence whose items share one shape
    # solves instead of bailing; numpy is the runtime oracle.
    result = run_mypy(FIXTURES / "stack_comprehension_ok.py")
    assert result.returncode == 0, result.stdout
    notes = [
        re.sub(r"\?U@\d+[a-z]?", "?U", line.split('Revealed type is "', 1)[1])
        for line in result.stdout.splitlines()
        if "Revealed type" in line
    ]
    f32 = "numpy.dtype[numpy.floating[numpy._typing._nbit_base._32Bit]]"
    f64 = "numpy.dtype[numpy.float64]"
    expected = [
        ("?U, T, n_agents, Literal[2]", f32),
        ("T, n_agents, Literal[2], ?U", f32),
        ("T, ?U, n_agents, Literal[2]", f32),
        ("?U, Literal[3], Literal[2]", f32),
        ("?U, n_agents, Literal[2]", f32),
        ("?U, ?U, Literal[3]", f32),
        ("?U, Literal[3]", f32),
        ("?U, Literal[3]", f32),
        ("?U, Literal[3], Literal[2]", f64),
        ("T, ?U, Literal[2]", f32),
        ("?U, Literal[3]", f32),
        ("Literal[3], ?U", f32),
        ("Literal[3], ?U", f32),
        ("Literal[3], Literal[2], ?U", f32),
    ]
    assert len(notes) == len(expected), result.stdout
    for note, (dims, dtype) in zip(notes, expected):
        assert note == f"numpy.ndarray[tuple[{dims}], {dtype}]\"", result.stdout
    assert _run_fixture("stack_comprehension_ok.py") == [
        "(3, 5, 4, 2) float32",
        "(5, 4, 2, 3) float32",
        "(5, 3, 4, 2) float32",
        "(7, 3, 2) float32",
        "(6, 4, 2) float32",
        "(8, 2, 3) float32",
        "(16, 3) float32",
        "(2, 3) float32",
        "(7, 3, 2) float64",
        "(5, 12, 2) float32",
        "(16, 3) float32",
        "(3, 14) float32",
        "(3, 16) float32",
        "(3, 2, 7) float32",
    ]


def test_stack_comprehension_keeps_checking() -> None:
    # The solved stack keeps rank and item dims, so a provable axis and a
    # downstream inner-dim clash still fail; numpy raises on both.
    assert_single_error(
        run_mypy(FIXTURES / "bad_stack_comprehension_matmul.py"),
        "matmul inner dimensions disagree: 3 vs 4",
        "[numty-shape]",
    )
    assert _run_fixture("bad_stack_comprehension_matmul.py") == ["ValueError"]
    assert_single_error(
        run_mypy(FIXTURES / "bad_stack_comprehension_axis.py"),
        "stack axis 3 is out of range for 2 dimensions",
        "[numty-axis]",
    )
    assert _run_fixture("bad_stack_comprehension_axis.py") == ["AxisError"]


def test_unknown_concrete_boundary() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_unknown_concrete.py"),
        "Incompatible types in assignment",
        "tuple[?U@13, M]",
        "tuple[Literal[3], M]",
    )


_FIXTURE_RUN_CACHE: dict[str, list[str]] = {}


def _run_fixture(name: str) -> list[str]:
    hit = _FIXTURE_RUN_CACHE.get(name)
    if hit is not None:
        return list(hit)
    run = subprocess.run(
        [sys.executable, str(FIXTURES / name)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    out = run.stdout.strip().splitlines()
    _FIXTURE_RUN_CACHE[name] = out
    return list(out)


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
        'F64 string dim "foo!" is not a name or linear expression',
        'space-separated names like "batch features"',
    )


def test_reduce15_ok_clean() -> None:
    # Issue #32 soundness half: the symbolic diff answer is pinned by
    # golden reveal plus a runtime oracle.
    result = run_mypy(FIXTURES / "reduce15_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    i32 = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.signedinteger[numpy._typing._nbit_base._32Bit]]]"
    i64 = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.signedinteger[numpy._typing._nbit_base._64Bit]]]"
    f64 = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.float64]]"
    assert reveals_by_line(result) == {
        40: f'"{f64.format("N-1, M")}"',
        41: f'"{i32.format("Literal[3]")}"',
        42: f'"{i64.format("N")}"',
    }, result.stdout
    assert _run_fixture("reduce15_ok.py") == ["(4, 7)", "(3,)", "(5,)"]


def test_average_weights_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_average_weights.py"),
        "average weights length 5 does not match axis 1 size 6",
        "(128, 6)",
    )


def test_reduce_keepdims_ok_clean_and_runs() -> None:
    # Issue #47: keepdims keeps size-1 axes, argpartition keeps the input
    # shape, average(returned=True) is a shape-matched tuple, reductions
    # without axis reduce every axis, and lstsq stays 2-D. Golden reveals
    # plus a runtime oracle.
    result = run_mypy(FIXTURES / "reduce_keepdims_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    f64 = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.float64]]"
    iu = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.signedinteger[numpy._typing._nbit_base._32Bit | numpy._typing._nbit_base._64Bit]]]"
    notes = {
        line: re.sub(r"\?U@\d+[a-z]?", "?U", note)
        for line, note in reveals_by_line(result).items()
    }
    assert notes == {
        30: f'"{iu.format("Literal[5], Literal[1]")}"',
        31: f'"{iu.format("Literal[1], Literal[7]")}"',
        32: f'"{iu.format("Literal[5], Literal[1]")}"',
        34: f'"{iu.format("Literal[5], Literal[7]")}"',
        36: f'"{iu.format("Literal[5], Literal[7]")}"',
        39: f'"{f64.format("Literal[4]")}"',
        40: f'"{f64.format("Literal[4]")}"',
        42: f'"{f64.format("Literal[1], Literal[1]")}"',
        43: f'"{f64.format("Literal[1], Literal[1]")}"',
        44: f'"{f64.format("Literal[1], Literal[1]")}"',
        45: f'"numpy.ndarray[tuple[Literal[1], Literal[1]], numpy.dtype[numpy.bool[bool]]]"',
        46: f'"{iu.format("Literal[1], Literal[1]")}"',
        47: f'"{f64.format("Literal[1], Literal[1]")}"',
        48: f'"{f64.format("Literal[3]")}"',
    }, result.stdout
    assert _run_fixture("reduce_keepdims_ok.py") == [
        "(5, 1)",
        "(1, 7)",
        "(5, 1)",
        "(5, 7)",
        "(5, 3)",
        "(5, 7)",
        "(4,)",
        "(4,)",
        "(1, 1)",
        "(1, 1)",
        "(1, 1)",
        "(1, 1)",
        "(1, 1)",
        "(1, 1)",
        "(3,)",
    ]


def test_reduce_shapes_mismatch() -> None:
    # Issue #47: every wrong reduction/sort shape must fail loudly.
    result = run_mypy(FIXTURES / "bad_reduce_shapes.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 9, result.stdout
    assert errors[0].startswith("tests/fixtures/bad_reduce_shapes.py:14: error:"), result.stdout
    assert "tuple[Literal[1], Literal[1]]" in errors[0], result.stdout
    assert "tuple[Literal[3], Literal[4]]" in errors[0], result.stdout
    assert errors[1].startswith("tests/fixtures/bad_reduce_shapes.py:18: error:"), result.stdout
    assert "tuple[N, Literal[4]]" in errors[1], result.stdout
    assert errors[2].startswith("tests/fixtures/bad_reduce_shapes.py:23: error:"), result.stdout
    assert "tuple[Literal[5], Literal[3]]" in errors[2], result.stdout
    assert errors[3].startswith("tests/fixtures/bad_reduce_shapes.py:27: error:"), result.stdout
    assert '"sort" of "ndarray" does not return a value' in errors[3], result.stdout
    assert errors[4].startswith("tests/fixtures/bad_reduce_shapes.py:28: error:"), result.stdout
    assert errors[5].startswith("tests/fixtures/bad_reduce_shapes.py:32: error:"), result.stdout
    assert "average weights with shape (20,) need axis=" in errors[5], result.stdout
    assert errors[6].startswith("tests/fixtures/bad_reduce_shapes.py:36: error:"), result.stdout
    assert "linalg.lstsq requires a 2-dimensional array, got shape (2, 5, 3)" in errors[7], result.stdout


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


def test_diff_symbolic_mismatch() -> None:
    # Issue #32: np.diff on a symbolic axis is N-1, not N.
    assert_single_error(
        run_mypy(FIXTURES / "bad_diff_symbolic.py"),
        "tuple[N-1, M]",
        "tuple[N, M]",
    )


def test_concat_symbolic_ok_clean() -> None:
    # Issue #32 (precision half): symbolic concat lengths stay exact
    # (N+M, N+1, 2*N) and 1-D cumsum without axis keeps (N,).
    result = run_mypy(FIXTURES / "concat_symbolic_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    f64 = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.float64]]"
    i64 = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.signedinteger[numpy._typing._nbit_base._64Bit]]]"
    assert reveals_by_line(result) == {
        20: f'"{f64.format("M+N")}"',
        21: f'"{f64.format("N+1")}"',
        22: f'"{f64.format("2*N")}"',
        26: f'"{i64.format("N")}"',
        29: f'"{i64.format("N+1")}"',
    }, result.stdout
    assert _run_fixture("concat_symbolic_ok.py") == ["(12,)", "(6,)", "(10,)", "(5,)", "(6,)"]


def test_concat_symbolic_mismatch() -> None:
    # Issue #32: concat of (1,) and (N,) is N+1, not N.
    assert_single_error(
        run_mypy(FIXTURES / "bad_concat_symbolic.py"),
        "tuple[N+1]",
        "tuple[N]",
    )


def test_offsets_lengths_ok_clean_and_runs() -> None:
    # Issue #32 precision half: linear expression dims (`I64["N+1"]`)
    # spell the offsets/lengths duality, so `np.diff` of `N+1` is `N`
    # and `concatenate([[0], cumsum])` of `N` is `N+1` with no ignore.
    # Distinct TypeVars per function dodge the #43 `_seen_vars` leak
    # (same-name bare simplifications otherwise bind to the first scope).
    result = run_mypy(FIXTURES / "offsets_lengths_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    i64 = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.signedinteger[numpy._typing._nbit_base._64Bit]]]"
    assert reveals_by_line(result) == {
        35: f'"{i64.format("K+1")}"',
        36: f'"{i64.format("K")}"',
    }, result.stdout
    assert _run_fixture("offsets_lengths_ok.py") == ["(6,)", "(5,)", "(5,)", "(6,)"]


def test_offsets_lengths_mismatch() -> None:
    # Issue #32: N+1 offsets do not fit an N annotation.
    assert_single_error(
        run_mypy(FIXTURES / "bad_offsets_lengths.py"),
        "tuple[N+1]",
        "tuple[N]",
    )


def test_provable_linear_mismatch() -> None:
    # Issue #32: same-var different-const dims (N vs N-1 from np.diff,
    # N vs N+2 from np.pad) are provably unequal, so broadcast/matmul/
    # stack fail instead of bailing to a silent ?U/Any.
    result = run_mypy(FIXTURES / "bad_provable_linear.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 4, result.stdout
    assert "broadcast shapes disagree at trailing axis 0" in errors[0], result.stdout
    assert "N vs N-1" in errors[0], result.stdout
    assert "Unsupported operand types for -" in errors[1], result.stdout
    assert "matmul inner dimensions disagree: N vs N+2" in errors[2], result.stdout
    assert "stack shapes differ: (N+2,), (N,)" in errors[3], result.stdout


def test_comprehension_unknown_bails_not_false_proof() -> None:
    # Issue #63: every element of [x[m] for m in masks] shares one ?U,
    # so matmul must bail (Any) and never claim a false (3, 3).
    result = run_mypy(FIXTURES / "unknown_comprehension_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    assert reveals_by_line(result) == {
        13: '"Any"',
        18: '"Any"',
    }, result.stdout


def test_einsum_symbolic_conflict() -> None:
    # Issue #63: einsum label N vs N+2 from off-by-pad is provable.
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_einsum_symbolic.py"),
        "einsum label 'n' has conflicting sizes: N vs N+2",
        '"einsum"',
    )


def test_reshape_symbolic_total_mismatch() -> None:
    # Issue #63: reshape of N+1 to N and N+2 to (N, 1) always raises.
    result = run_mypy(FIXTURES / "bad_reshape_symbolic_total.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 2, result.stdout
    assert "reshape changes total size: N+1 -> N" in errors[0], result.stdout
    assert "reshape changes total size: N+2 -> N" in errors[1], result.stdout


def test_no_crash_ok_clean() -> None:
    # Issue #41: sort(axis=None) and 0-d dot/inner must never crash.
    result = run_mypy(FIXTURES / "no_crash_ok.py")
    assert result.returncode == 0, result.stdout
    assert "INTERNAL ERROR" not in result.stdout, result.stdout
    assert "internal error" not in result.stdout.lower(), result.stdout
    f64 = "numpy.dtype[numpy.float64]"
    i32 = "numpy.dtype[numpy.signedinteger[numpy._typing._nbit_base._32Bit | numpy._typing._nbit_base._64Bit]]"
    assert reveals_by_line(result) == {
        12: f'"numpy.ndarray[tuple[3*N], {f64}]"',
        13: f'"numpy.ndarray[tuple[3*N], {i32}]"',
        17: f'"numpy.ndarray[tuple[Literal[4], Literal[3]], {f64}]"',
        18: f'"numpy.ndarray[tuple[Literal[2], Literal[4], Literal[3]], {f64}]"',
        19: f'"numpy.ndarray[tuple[Literal[4], Literal[3]], {f64}]"',
    }, result.stdout


def test_solve_0d_mismatch() -> None:
    # Issue #41: 0-d b in linalg.solve is a normal error, not a crash.
    result = run_mypy(FIXTURES / "bad_solve_0d.py")
    assert result.returncode == 1, result.stdout
    assert "INTERNAL ERROR" not in result.stdout, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 1, result.stdout
    assert "linalg.solve: b has 0 dimensions" in errors[0], result.stdout


def test_offsets_calls_ok_clean_and_runs() -> None:
    # Issue #32 finish: expression returns infer through calls instead of
    # going Any. `make_offsets(a: I64[K]) -> I64["M+1"]` reveals K+1 in the
    # caller, and `take_lengths(K+1)` solves back to K. Distinct TypeVars
    # per function dodge the #43 `_seen_vars` leak.
    result = run_mypy(FIXTURES / "offsets_calls_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    i64 = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.signedinteger[numpy._typing._nbit_base._64Bit]]]"
    assert reveals_by_line(result) == {
        37: f'"{i64.format("K+1")}"',
        39: f'"{i64.format("K")}"',
    }, result.stdout
    assert _run_fixture("offsets_calls_ok.py") == ["(6,)", "(5,)", "(6,)", "(5,)"]


def test_offsets_calls_mismatch() -> None:
    # Issue #32 finish: K+1 from a call does not fit a K annotation.
    assert_single_error(
        run_mypy(FIXTURES / "bad_offsets_calls.py"),
        "tuple[K+1]",
        "tuple[K]",
    )


def test_same_tvar_roundtrip_ok() -> None:
    # Issue #43: two byte-identical bodies sharing one TypeVar check
    # identically. The reshape -1 roundtrip (repeat -> N*2 -> N) used to
    # return the first-seen N@first for the second function, failing with
    # `got tuple[2, N@first], expected tuple[2, N@second]`.
    result = run_mypy(FIXTURES / "same_tvar_roundtrip_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    f64 = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.float64]]"
    assert reveals_by_line(result) == {
        24: f'"{f64.format("Literal[2], N")}"',
        30: f'"{f64.format("Literal[2], N")}"',
    }, result.stdout
    assert _run_fixture("same_tvar_roundtrip_ok.py") == [
        "(2, 5)",
        "(2, 6)",
        "ok",
    ]


def test_dataclass_implicit_method_ok() -> None:
    # Issue #29: methods (including dataclass methods) with undeclared
    # string dims infer like functions instead of rigid-failing.
    # `c.m(concrete)` and `a.with_grid(dynamic)` used to report
    # `expected tuple[P, Q, ...]`; now they infer.
    result = run_mypy(FIXTURES / "dataclass_implicit_method_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    f32 = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.floating[numpy._typing._nbit_base._32Bit]]]"
    assert reveals_by_line(result) == {
        33: f'"{f32.format("U, V, Literal[2]")}"',
        40: f'"{f32.format("Literal[9], Literal[23], Literal[2]")}"',
        44: f'"{f32.format("T, X, Literal[2]")}"',
    }, result.stdout
    assert _run_fixture("dataclass_implicit_method_ok.py") == [
        "(9, 23, 2)",
        "(9, 23, 2)",
        "(4, 5, 2)",
    ]


def test_bare_shape_generic_is_loud() -> None:
    # Issue #38: list[Box] with Box holding F64[...] was green while every
    # shape downstream was Any. The bare use now fails loudly so it can be
    # parameterized as Box[T_ctx].
    assert_single_error(
        run_mypy(FIXTURES / "bad_bare_generic.py"),
        "Box used bare",
        "parameterize as Box[...]",
        line=20,
    )


def test_parameterized_shape_generic_stays_clean() -> None:
    result = run_mypy(FIXTURES / "bare_generic_ok.py")
    assert result.returncode == 0, result.stdout
    assert reveals_by_line(result) == {
        30: '"numpy.ndarray[tuple[T_ctx, Literal[3]], numpy.dtype[numpy.float64]]"',
        35: '"numpy.ndarray[tuple[Literal[4], Literal[3]], numpy.dtype[numpy.float64]]"',
    }, result.stdout
    assert _run_fixture("bare_generic_ok.py") == ["ok"]


def test_star_kwargs_ok() -> None:
    # Issue #52: *tuple stars expand, shape=/axes=/min=/max= fall back
    # from keywords, and out=/dtype= select the result. All match runtime.
    result = run_mypy(FIXTURES / "star_kwargs_ok.py")
    assert result.returncode == 0, result.stdout
    f64 = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.float64]]"
    f32 = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.floating[numpy._typing._nbit_base._32Bit]]]"
    assert reveals_by_line(result) == {
        28: f'"{f64.format("Literal[4], Literal[5], Literal[6]")}"',
        29: f'"{f64.format("Literal[2], Literal[4]")}"',
        30: f'"{f64.format("Literal[2], Literal[4], Literal[3]")}"',
        31: f'"{f64.format("Literal[4], Literal[3], Literal[2]")}"',
        32: f'"{f64.format("Literal[3], Literal[2], Literal[4]")}"',
        33: f'"{f64.format("Literal[2], Literal[3]")}"',
        34: f'"{f64.format("Literal[2], Literal[3]")}"',
        35: f'"{f32.format("Literal[2], Literal[3]")}"',
        36: f'"{f64.format("Literal[4], Literal[2], Literal[3]")}"',
    }, result.stdout
    assert _run_fixture("star_kwargs_ok.py") == ["ok"]


def test_star_kwargs_bad() -> None:
    # Issue #52: dropped bounds/shapes/out used to pass silently with the
    # wrong shape; now each fails loudly.
    result = run_mypy(FIXTURES / "bad_star_kwargs.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 4, result.stdout
    assert "clip shapes disagree" in errors[0], result.stdout
    assert "reshape changes total size: 6 -> 8" in errors[1], result.stdout
    assert "output shape mismatch" in errors[2], result.stdout
    assert "output shape mismatch" in errors[3], result.stdout


def test_expr_dims_rejected() -> None:
    # Expression dims stay linear: name*name and division fail loudly
    # instead of going quietly Any.
    for bad, fragment in (
        ('F64["N*M"]', "is not linear"),
        ('F64["N//2"]', "is not linear"),
    ):
        fixture = FIXTURES / "_tmp_expr_bad.py"
        fixture.write_text(
            "from numty import F64\ndef f(x: " + bad + ") -> None: ...\n"
        )
        try:
            assert_single_error(run_mypy(fixture), fragment)
        finally:
            fixture.unlink(missing_ok=True)


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


def test_issue58_builders_dot_ok() -> None:
    # Issue #58 slice 2: linspace/identity/tri were rank-only, dot twin Any.
    result = run_mypy(FIXTURES / "issue58_builders_dot_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    assert notes[6] == (
        '"numpy.ndarray[tuple[Literal[7]], numpy.dtype[numpy.float64]]"'
    ), result.stdout
    assert notes[7] == (
        '"numpy.ndarray[tuple[Literal[4], Literal[4]], numpy.dtype[numpy.float64]]"'
    ), result.stdout
    assert notes[10] == (
        '"numpy.ndarray[tuple[Literal[2], Literal[4]], numpy.dtype[numpy.float64]]"'
    ), result.stdout


def test_dot_method_inner_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_dot_method_inner.py"),
        "matmul inner dimensions disagree: 3 vs 4",
    )


def test_linspace_neg_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_linspace_neg.py"),
        "linspace num must be non-negative, got -1",
    )


def test_issue58_copy_repeat_isclose_ok() -> None:
    result = run_mypy(FIXTURES / "issue58_copy_repeat_isclose_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    assert notes[7] == (
        '"numpy.ndarray[tuple[Literal[2], Literal[3]], numpy.dtype[numpy.float64]]"'
    ), result.stdout
    assert notes[9] == (
        '"numpy.ndarray[tuple[Literal[4], Literal[3]], numpy.dtype[numpy.float64]]"'
    ), result.stdout
    assert notes[10] == (
        '"numpy.ndarray[tuple[Literal[2], Literal[3]], numpy.dtype[numpy.bool[bool]]]"'
    ), result.stdout


def test_repeat_method_axis_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_repeat_method_axis.py"),
        "repeat axis 5 is out of range for 2 dimensions",
    )


def test_isclose_broadcast_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_isclose_broadcast.py"),
        "broadcast shapes disagree at trailing axis 0",
        "3 vs 5",
        "(2, 3)",
        "(5,)",
    )


def test_issue58_fft_keepshape_ok() -> None:
    result = run_mypy(FIXTURES / "issue58_fft_keepshape_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    assert notes[7] == (
        '"numpy.ndarray[tuple[Literal[3], Literal[4]], numpy.dtype[numpy.complex128]]"'
    ), result.stdout
    assert notes[10] == (
        '"numpy.ndarray[tuple[Literal[3], Literal[4]], numpy.dtype[numpy.float64]]"'
    ), result.stdout


def test_fft2_broadcast_mismatch() -> None:
    assert_operator_mismatch(
        run_mypy(FIXTURES / "bad_fft2_broadcast.py"),
        "broadcast shapes disagree at trailing axis 0",
        "4 vs 5",
        "(3, 4)",
        "(3, 5)",
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
    # Issue #71: annotation pins accept Any, so each shape is also pinned
    # by a golden reveal compared per line, plus a runtime oracle.
    result = run_mypy(FIXTURES / "atleast_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    f64 = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.float64]]"
    assert reveals_by_line(result) == {
        20: f'"{f64.format("Literal[1], N")}"',
        21: f'"{f64.format("N, M")}"',
        22: f'"{f64.format("Literal[1], N, Literal[1]")}"',
        # numpy appends the new axis for 2-D input, not prepends it.
        24: f'"{f64.format("N, M, Literal[1]")}"',
        25: f'"{f64.format("Literal[2], Literal[3], Literal[4]")}"',
        26: f'"{f64.format("N")}"',
    }, result.stdout
    assert _run_fixture("atleast_ok.py") == [
        "(1, 5)",
        "(5, 7)",
        "(1, 5, 1)",
        "(5, 7, 1)",
        "(2, 3, 4)",
        "(5,)",
    ]


def test_stack_ok_clean() -> None:
    result = run_mypy(FIXTURES / "stack_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout


def test_construct_shapes_ok_clean_and_runs() -> None:
    # Issue #46: hooked ops used to claim precise-but-wrong literal
    # shapes (a wrong concrete shape is worse than Any). Each reveal is
    # pinned against the runtime oracle below.
    result = run_mypy(FIXTURES / "construct_shapes_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    f64 = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.float64]]"
    i64 = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.signedinteger[numpy._typing._nbit_base._64Bit]]]"
    notes = {
        line: re.sub(r"\?U@\d+[a-z]?", "?U", note)
        for line, note in reveals_by_line(result).items()
    }
    assert notes == {
        41: f'"{f64.format("?U")}"',
        42: f'"{f64.format("?U")}"',
        43: f'"{f64.format("?U")}"',
        44: f'"{f64.format("N+2")}"',
        49: f'"{f64.format("Literal[2], Literal[3], Literal[1]")}"',
        50: f'"{f64.format("Literal[2], Literal[3], Literal[2]")}"',
        51: f'"{f64.format("Literal[12]")}"',
        52: f'"{f64.format("Literal[5]")}"',
        55: f'"{f64.format("Literal[8], Literal[4]")}"',
        56: f'"{f64.format("Literal[2], Literal[3], Literal[4]")}"',
        57: f'"{f64.format("Literal[10], Literal[5]")}"',
        58: f'"{f64.format("Literal[2], Literal[3]")}"',
        59: f'"{f64.format("Literal[2], Literal[1]")}"',
        60: f'"{i64.format("Literal[5]")}"',
        61: f'"{i64.format("Literal[4]")}"',
        62: f'"{f64.format("Literal[1], Literal[1], Literal[5]")}"',
    }, result.stdout
    assert _run_fixture("construct_shapes_ok.py") == [
        "(2, 3, 1)",
        "(2, 3, 2)",
        "(12,)",
        "(5,)",
        "(8, 4)",
        "(2, 3, 4)",
        "(10, 5)",
        "(2, 3)",
        "(2, 1)",
        "(5,)",
        "(4,)",
        "(1, 1, 5)",
        "(6,)",
        "(4,)",
        "(4,)",
        "(8,)",
    ]


def test_construct_shapes_mismatch() -> None:
    # Issue #46: every wrong concrete shape must fail, loudly.
    result = run_mypy(FIXTURES / "bad_construct_shapes.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 9, result.stdout
    assert errors[0].startswith("tests/fixtures/bad_construct_shapes.py:11: error:"), result.stdout
    assert "tuple[Literal[2], Literal[3], Literal[1]]" in errors[0], result.stdout
    assert "dstack dimension 1 differs: 3 vs 4" in errors[1], result.stdout
    assert "tuple[Literal[12]]" in errors[2] and "tuple[Literal[36]]" in errors[2], result.stdout
    assert "broadcast_to of shape (3, 4) is incompatible with target shape (3, 5)" in errors[3], result.stdout
    assert "cannot broadcast to scalar shape ()" in errors[4], result.stdout
    assert "tuple[Literal[2], Literal[1]]" in errors[5], result.stdout
    assert "tuple[Literal[8], Literal[4]]" in errors[6], result.stdout
    assert "tuple[Literal[5]]" in errors[7] and "tuple[Literal[10]]" in errors[7], result.stdout
    assert "tuple[Literal[1], Literal[1], Literal[5]]" in errors[8], result.stdout


def test_hstack_rank_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_hstack_rank.py"),
        "hstack ranks differ: (3,), (2, 2)",
    )


def test_shapemath_ok_clean() -> None:
    # Issue #71: bare-CN annotation meant CN[Any]; the exact rfft answer
    # is pinned by golden reveal plus a runtime oracle instead.
    result = run_mypy(FIXTURES / "shapemath_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    assert reveals_by_line(result) == {
        58: '"numpy.ndarray[tuple[N//2+1], numpy.dtype[numpy.complex128]]"',
    }, result.stdout
    assert _run_fixture("shapemath_ok.py") == ["(6,)"]


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
    # Issue #71: the "stay exact, not ?U" comment asserted nothing; each
    # formula is now pinned by golden reveal plus a runtime oracle.
    result = run_mypy(FIXTURES / "symexpr_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    f64 = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.float64]]"
    assert reveals_by_line(result) == {
        28: f'"{f64.format("N+2")}"',
        29: f'"{f64.format("2*N-1")}"',
        30: f'"list[{f64.format("N//2")}]"',
        31: '"numpy.ndarray[tuple[N//2+1], numpy.dtype[numpy.complex128]]"',
    }, result.stdout
    assert _run_fixture("symexpr_ok.py") == [
        "(14,)",
        "(23,)",
        "[(6,), (6,)]",
        "(7,)",
    ]


def test_symexpr_product_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_symexpr_product.py"),
        "reshape changes total size: 2*N -> 3*N",
        "(2*N,)",
        "(3, N)",
    )


def test_nonlinear_ok_clean() -> None:
    # Issue #40 stepping stone: monomial products stay exact, not ?U.
    result = run_mypy(FIXTURES / "nonlinear_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    f64 = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.float64]]"
    assert reveals_by_line(result) == {
        29: f'"{f64.format("B, C*H*W")}"',
        31: f'"{f64.format("N, K")}"',
        33: f'"{f64.format("K, N")}"',
        39: f'"{f64.format("B, S, D*H")}"',
    }, result.stdout
    assert _run_fixture("nonlinear_ok.py") == [
        "(2, 70)",
        "(5, 7)",
        "(7, 5)",
        "(2, 3, 35)",
    ]


def test_nonlinear_size_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_nonlinear_size.py"),
        "reshape changes total size: M*N -> 12*N",
        "(N, M)",
        "(N, 3, 4)",
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
