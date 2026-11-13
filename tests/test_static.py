"""Static checks: each bad fixture must fail mypy with exactly one error."""
import os
import re
import runpy
import shutil
import subprocess
import sys
from collections.abc import Iterator
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
    # mypy exits 0 or 1. Anything else is a crash or a killed process
    # (out of memory on a small runner), and its output stops partway:
    # the fixtures after that point would read as clean. Fail here, with
    # the reason; the tests after this one run mypy per file.
    assert result.returncode in (0, 1), (
        f"batched mypy run exited with {result.returncode}\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
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
def _fresh_mypy_cache() -> Iterator[None]:
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


def test_matmul_operator_mismatch() -> None:
    # Issue #140: @ checks exactly like np.matmul: inner, batch,
    # chained, vector, 0-d, scalar and @= fail with the same
    # [numty-shape] message and no mechanical [operator] duplicate.
    result = run_mypy(FIXTURES / "bad_matmul_operator.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 7, result.stdout
    assert "matmul inner dimensions disagree: 3 vs 2" in errors[0], result.stdout
    assert "matmul batch dimensions disagree" in errors[1], result.stdout
    assert "(5,)" in errors[1] and "(4,)" in errors[1], result.stdout
    assert "matmul inner dimensions disagree: 4 vs 2" in errors[2], result.stdout
    assert "matmul inner dimensions disagree: 3 vs 4" in errors[3], result.stdout
    assert "matmul operand 0 has no dimensions (a 0-d array)" in errors[4], result.stdout
    assert "matmul operand 1 has no dimensions (a scalar float)" in errors[5], result.stdout
    assert "matmul inner dimensions disagree: 3 vs 2" in errors[6], result.stdout
    assert all("[numty-shape]" in line for line in errors), result.stdout
    assert not [line for line in errors if "[operator]" in line], result.stdout


def test_matmul_operator_ok_clean_and_runs() -> None:
    # Issue #140: correct @ solves precise shapes (no more Any) and the
    # runtime oracle agrees, including batched, chained, vector, @=
    # and *Ts forms.
    result = run_mypy(FIXTURES / "matmul_operator_ok.py")
    assert result.returncode == 0, result.stdout
    f64 = "numpy.dtype[numpy.float64]"
    assert reveals_by_line(result) == {
        14: f'"numpy.ndarray[tuple[Literal[2], Literal[4]], {f64}]"',
        15: f'"numpy.ndarray[tuple[Literal[2], Literal[4]], {f64}]"',
        20: f'"numpy.ndarray[tuple[Literal[5], Literal[2], Literal[4]], {f64}]"',
        25: f'"numpy.ndarray[tuple[Literal[2], Literal[5]], {f64}]"',
        29: '"numpy.float64"',
        30: f'"numpy.ndarray[tuple[Literal[2]], {f64}]"',
        31: f'"numpy.ndarray[tuple[Literal[4]], {f64}]"',
        37: f'"numpy.ndarray[tuple[Literal[2], Literal[2]], {f64}]"',
        48: f'"numpy.ndarray[tuple[*Ts, Literal[4], Literal[6]], {f64}]"',
    }, result.stdout
    assert _run_fixture("matmul_operator_ok.py") == [
        "(2, 4)",
        "(2, 4)",
        "(5, 2, 4)",
        "(2, 5)",
        "0.0",
        "(2,)",
        "(4,)",
        "(2, 2)",
    ]


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
    # Issue #148: single targeted error, no mechanical [operator] duplicate.
    assert_single_error(
        run_mypy(FIXTURES / "bad_broadcast.py"),
        "broadcast shapes disagree at trailing axis 0",
        "3 vs 4",
        "(2, 3)",
        "(2, 4)",
    )


def test_broadcast_abstract_dtype_mismatch() -> None:
    # Issue #217: the broadcast check does not depend on the dtype.
    # Non-concrete dtypes (floating, Any, intp) used to skip the check
    # and keep the left shape or dissolve to Any.
    result = run_mypy(FIXTURES / "bad_broadcast_abstract_dtype.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 4, result.stdout
    for line in errors:
        assert "broadcast shapes disagree at trailing axis 0" in line, result.stdout
        assert "[numty-broadcast]" in line, result.stdout
    assert "3 vs 4" in errors[0] and "3 vs 4" in errors[1], result.stdout
    assert "N-1 vs N" in errors[2] and "N-1 vs N" in errors[3], result.stdout


def test_unknown_broadcast_keeps_rank_and_dtype() -> None:
    # Issue #56: one unprovable axis used to dissolve the whole result
    # into Any; now the axis is a fresh unknown and rank, known dims
    # and dtype survive. Issue #138: a literal against a symbolic dim
    # is likewise unprovable, so it mints a fresh unknown instead of
    # claiming the literal.
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
        f'numpy.ndarray[tuple[?U, Literal[3]], {f64}]"',
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


def test_broadcast_int_symbolic_mints_unknown() -> None:
    # Issue #138: literal-vs-symbolic claimed the literal; now it mints
    # a fresh ?U (1s and same-name symbolics stay precise).
    result = run_mypy(FIXTURES / "broadcast_int_symbolic_ok.py")
    assert result.returncode == 0, result.stdout
    notes = [
        re.sub(r"\?U@\d+[a-z]?", "?U", line.split('Revealed type is "', 1)[1])
        for line in result.stdout.splitlines()
        if "Revealed type" in line
    ]
    f64 = "numpy.dtype[numpy.float64]"
    assert notes == [
        f'numpy.ndarray[tuple[?U], {f64}]"',
        f'numpy.ndarray[tuple[?U], {f64}]"',
        f'numpy.ndarray[tuple[N], {f64}]"',
        f'numpy.ndarray[tuple[N], {f64}]"',
        f'numpy.ndarray[tuple[N], {f64}]"',
    ], result.stdout
    assert _run_fixture("broadcast_int_symbolic_ok.py") == [
        "(5,)",
        "(3,)",
    ]


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
        # A bound over `int` variables is a formula over them: still an
        # unknown, compared only under the opt-in code (symbolic_slice.py).
        f'numpy.ndarray[tuple[?bs, Literal[3]], {f64}]"',
        f'numpy.ndarray[tuple[-?i+5, Literal[7]], {f64}]"',
        f'numpy.ndarray[tuple[Literal[5], ?k], {f64}]"',
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
        "stack dimension 0 differs: 2 vs 3",
        "(2, 3)",
        "(3, 2))  [numty-shape]",
    )


def test_stack_empty_sequence() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_stack_empty.py"),
        "stack needs at least one array (got an empty sequence)",
        "[numty-shape]",
    )


def test_concatenate_empty_sequence() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_concat_empty.py"),
        "concatenate needs at least one array (got an empty sequence)",
        "[numty-shape]",
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
    # Issue #148: OOB names shape (take/split/squeeze form).
    assert_single_error(
        run_mypy(FIXTURES / "bad_concat_axis.py"),
        "concatenate axis 5 is out of range for 2 dimensions",
        "(2, 3)",
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


def test_index_pos_ok() -> None:
    # Issues #285/#286/#287: recorded index positions check clean by
    # default; producers (argmin/argmax, arange, flatnonzero, nonzero,
    # searchsorted) record it themselves. Index[P, N] spells which axis
    # the values index into.
    # Under the opt-in names code (issue #288, shared with #280) the
    # order_ok pair broadcasts two different counts (J,) vs (K,), so
    # line 19 reports the broadcast names error there while everything
    # else stays clean. The default run keeps the full reveal pins.
    off = run_mypy(FIXTURES / "index_pos_ok.py")
    assert off.returncode == 0, off.stdout
    on = run_mypy(FIXTURES / "index_pos_ok.py", flags=("--enable-error-code", "numty-names"))
    on_errors = _errors_by_line(on)
    assert sorted(on_errors) == [19], on.stdout
    assert len(on_errors[19]) == 1, on.stdout
    assert (
        "broadcast at trailing axis 0 (shapes (J,) vs (K,)): J and K are different names"
        in on_errors[19][0]
    ), on.stdout
    assert on_errors[19][0].endswith("[numty-names]"), on.stdout
    f64 = "numpy.dtype[numpy.float64]"
    i64 = "numpy.dtype[numpy.signedinteger[numpy._typing._nbit_base._64Bit]]"
    pi64 = (
        "numpy.dtype[numpy.signedinteger[numpy._typing._nbit_base._32Bit"
        " | numpy._typing._nbit_base._64Bit]]"
    )
    assert reveals_by_line(run_mypy(FIXTURES / "index_pos_ok.py")) == {
        17: f'"numpy.ndarray[tuple[J], {i64}]"',
        18: f'"numpy.ndarray[tuple[K], {i64}]"',
        20: f'"numpy.ndarray[tuple[?U@19], {f64}]"',
        22: f'"numpy.ndarray[tuple[P, K], {f64}]"',
        24: f'"numpy.ndarray[tuple[J, Q], {f64}]"',
        28: f'"numpy.ndarray[tuple[N_count], {i64}]"',
        30: f'"numpy.ndarray[tuple[N, N_count], {f64}]"',
        32: f'"numpy.ndarray[tuple[N_count], {f64}]"',
        36: f'"numpy.ndarray[tuple[K], {i64}]"',
        38: f'"numpy.ndarray[tuple[P, K], {f64}]"',
        43: f'"numpy.ndarray[tuple[N], {pi64}]"',
        45: f'"numpy.ndarray[tuple[K, N], {f64}]"',
        50: f'"numpy.ndarray[tuple[N], {pi64}]"',
        52: f'"numpy.ndarray[tuple[N, M], {f64}]"',
        58: f'"numpy.ndarray[tuple[?U@56], {f64}]"',
        64: f'"numpy.ndarray[tuple[?U@63], {f64}]"',
        69: f'"numpy.ndarray[tuple[K], {i64}]"',
        71: f'"numpy.ndarray[tuple[K], {f64}]"',
        76: f'"numpy.ndarray[tuple[K], {f64}]"',
        82: f'"numpy.ndarray[tuple[J, Q], {f64}]"',
        90: f'"numpy.ndarray[tuple[K], {f64}]"',
        97: f'"numpy.ndarray[tuple[K], {f64}]"',
    }, run_mypy(FIXTURES / "index_pos_ok.py").stdout


def test_bad_index_callsite() -> None:
    # Issue #286: an index recorded for one size passed where an Index
    # formal declares another is an opt-in [numty-names] error at the
    # call. Same-module plain positional calls only; producers, stored
    # results and matching positions stay clean, and the default run
    # stays silent everywhere (unknown is not an error).
    fixture = FIXTURES / "bad_index_callsite.py"
    assert run_mypy(fixture).returncode == 0, run_mypy(fixture).stdout
    on = run_mypy(fixture, flags=("--enable-error-code", "numty-names"))
    errors = _errors_by_line(on)
    assert sorted(errors) == [31], on.stdout
    assert len(errors[31]) == 1, on.stdout
    assert (
        "index positions on argument 'rows': N and P are different names"
        in errors[31][0]
    ), on.stdout
    assert errors[31][0].endswith("[numty-names]"), on.stdout


def test_bad_index_pos() -> None:
    # Issues #285/#286/#287: recorded positions on one size used on
    # another. A certain overrun (`arange(5)` on a size-3 axis) errors by
    # default. A possible one (`searchsorted` positions run to N on an
    # axis of size N) and different names need the opt-in code. Mixed slice+fancy
    # lines also carry mypy's own [index] error, as every mixed-tuple
    # solver error does (see assert_rank_mismatch).
    # Merged with issue #288: two fancy indices broadcast their counts,
    # so (J,) vs (K,) is a broadcast names error too — line 17 gains one
    # alongside its two position errors and the correct-positions line 18
    # reports only the broadcast.
    fixture = FIXTURES / "bad_index_pos.py"
    off = run_mypy(fixture)
    assert sorted(_errors_by_line(off)) == [44], off.stdout
    on = run_mypy(fixture, flags=("--enable-error-code", "numty-names"))
    errors = _errors_by_line(on)
    expected = {
        17: [
            "index positions on axis 0 (shape (P, Q)): Q and P are different names",
            "index positions on axis 1 (shape (P, Q)): P and Q are different names",
            "broadcast at trailing axis 0 (shapes (K,) vs (J,)): K and J are different names",
        ],
        18: [
            "broadcast at trailing axis 0 (shapes (J,) vs (K,)): J and K are different names",
        ],
        22: ["index positions on axis 1 (shape (P, Q)): M and Q are different names"],
        23: ["index positions on axis 0 (shape (P, Q)): M and P are different names"],
        28: ["index positions on axis 1 (shape (M, K)): M and K are different names"],
        33: [
            "index positions on axis 0 (shape (N, M)): M and N are different names",
            "index positions on axis 1 (shape (N, M)): N and M are different names",
        ],
        38: ["index positions on N+1 used on axis 0 of size N (shape (N,)): they may run past the axis"],
        44: ["index positions on 5 used on axis 0 of size 3 (shape (3,))"],
        48: ["index positions on axis 1 (shape (P, Q)): M and Q are different names"],
    }
    assert sorted(errors) == [17, 18, 22, 23, 28, 33, 38, 44, 48], on.stdout
    for line, messages in expected.items():
        assert len(errors[line]) == len(messages) + (1 if line in (22, 23, 28) else 0), on.stdout
        for message in messages:
            assert any(
                message in text
                and text.endswith(
                    "[numty-names]" if "different names" in message or "may run past" in message else "[numty-axis]"
                )
                for text in errors[line]
            ), on.stdout
    for line in (22, 23, 28):
        assert any("Invalid index type" in text and text.endswith("[index]") for text in errors[line]), on.stdout


def test_einsum_label_conflict() -> None:
    # Issue #148: single targeted error with both shapes, no bool follow-ons.
    assert_single_error(
        run_mypy(FIXTURES / "bad_einsum_labels.py"),
        "einsum label 'j' has conflicting sizes: 3 vs 5",
        "(2, 3)",
        "(5, 4)",
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
        86: '"numpy.ndarray[tuple[Any], numpy.dtype[numpy.float64]]"',
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
    # Issue #148: single targeted error, no operator duplicate.
    assert_single_error(
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


def test_det_square_mismatch() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_det_square.py"),
        "linalg.det requires a square (..., N, N) array, got shape (2, 3)",
        '"det"',
    )


def test_slogdet_square_mismatch() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_slogdet_square.py"),
        "linalg.slogdet requires a square (..., N, N) array, got shape (2, 3)",
        '"slogdet"',
    )


def test_qr_rank_mismatch() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_qr_rank.py"),
        "linalg.qr requires at least 2 dimensions, got shape (3,)",
        '"qr"',
    )


def test_matrix_power_square_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_matrix_power_square.py"),
        "linalg.matrix_power requires a square (..., N, N) array, got shape (2, 3)",
    )


def test_cond_rank_mismatch() -> None:
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_cond_rank.py"),
        "linalg.cond requires at least 2 dimensions, got shape (3,)",
        '"cond"',
    )


def test_tensorsolve_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_tensorsolve_mismatch.py"),
        "linalg.tensorsolve requires prod(a.shape[b.ndim:]) == "
        "prod(a.shape[:b.ndim]) (shapes (2, 3) vs (4,))",
    )


def test_cross_vector_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_cross_vector.py"),
        "cross requires 3-dimensional vectors, got 3 and 4",
        "(3,)",
        "(4,)",
    )


def test_issue141_linalg_guards_ok() -> None:
    # Issue #141: the seven repro calls now error above and solve here.
    result = run_mypy(FIXTURES / "issue141_linalg_guards_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    assert notes[6] == '"numpy.float64"', result.stdout
    assert notes[7] == '"tuple[numpy.float64, numpy.float64]"', result.stdout
    assert notes[8] == (
        '"tuple[numpy.ndarray[tuple[Literal[3], Literal[3]], '
        'numpy.dtype[numpy.float64]], numpy.ndarray[tuple[Literal[3], '
        'Literal[4]], numpy.dtype[numpy.float64]]]"'
    ), result.stdout
    assert notes[9] == (
        '"numpy.ndarray[tuple[Literal[2], Literal[2]], '
        'numpy.dtype[numpy.float64]]"'
    ), result.stdout
    assert notes[10] == '"numpy.float64"', result.stdout
    assert notes[11] == (
        '"numpy.ndarray[tuple[Literal[2], Literal[2]], '
        'numpy.dtype[numpy.float64]]"'
    ), result.stdout
    assert notes[12] == (
        '"numpy.ndarray[tuple[Literal[3]], numpy.dtype[numpy.float64]]"'
    ), result.stdout


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
    # Issue #148: single targeted error per line, no operator duplicates.
    result = run_mypy(FIXTURES / "bad_arith_broadcast.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 4, result.stdout
    for line in errors:
        assert "broadcast shapes disagree at trailing axis 0" in line, result.stdout
        assert "3 vs 5" in line and "(2, 3)" in line and "(5,)" in line, result.stdout


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


def test_dtype_144_ok() -> None:
    # Issue #144: datetime family, reciprocal/power scalar paths, det/qr.
    result = run_mypy(FIXTURES / "dtype_144_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    assert len(notes) == 11, result.stdout
    assert "datetime64" in notes[16] and "Literal[2]" in notes[16], result.stdout
    assert "datetime64" in notes[17] and "Literal[2]" in notes[17], result.stdout
    assert "timedelta64" in notes[18] and "Literal[2]" in notes[18], result.stdout
    assert "timedelta64" in notes[19] and "Literal[1]" in notes[19], result.stdout
    assert "datetime64" in notes[20] and "Literal[2]" in notes[20], result.stdout
    assert "_32Bit" in notes[24] and "floating" in notes[24], result.stdout
    assert "complexfloating" in notes[25] and "_32Bit" in notes[25], result.stdout
    assert "floating" in notes[26] and "_32Bit" in notes[26], result.stdout
    assert "signedinteger" in notes[27] and "_32Bit" in notes[27], result.stdout
    assert '"numpy.floating' in notes[31] and "_32Bit" in notes[31], result.stdout
    assert "Literal[2], Literal[2]" in notes[32] and "_32Bit" in notes[32], result.stdout
    assert _run_fixture("dtype_144_ok.py") == ["ok"]


def test_mean_datetime_bails() -> None:
    # Issue #144: mean(datetime) always raises: error, no float64 claim.
    result = run_mypy(FIXTURES / "bad_mean_datetime.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 1, result.stdout
    assert "arg-type" in errors[0], result.stdout
    notes = reveals_by_line(result)
    assert notes[11] == '"Any"', result.stdout
    assert "float64" not in notes[11], result.stdout


def test_any_axis_out_of_range() -> None:
    # Issue #148: OOB names op and shape.
    assert_single_error(
        run_mypy(FIXTURES / "bad_any_axis.py"),
        "any axis 5 is out of range for 2 dimensions",
        "(2, 3)",
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
    # Issue #148: OOB names op and shape.
    assert_single_error(
        run_mypy(FIXTURES / "bad_min_axis.py"),
        "min axis 5 is out of range for 2 dimensions",
        "(2, 3)",
    )


def test_callsite_formula_no_size() -> None:
    # A fixed size no value of the name can produce: `n+2` is never 1,
    # `2*n` is never 5. Sizes that do fit are solved, not flagged.
    result = run_mypy(FIXTURES / "bad_callsite_no_size.py")
    assert result.returncode == 1, result.stdout
    errors = [line.split(": error: ")[1] for line in result.stdout.splitlines() if ": error:" in line]
    assert errors == [
        "call-site shape conflict: no size 'n' makes n+2 equal 1"
        " (argument 'x', axis 1)  [numty-callsite]",
        "call-site shape conflict: no size 'n' makes 2*n equal 5"
        " (argument 'x', axis 1)  [numty-callsite]",
    ], result.stdout


def test_callsite_formula_returns() -> None:
    # Found by the oracle's generic-call family. A formal `N+1` given
    # `2*M+2` solves N = 2*M+1; before, N stayed unbound and the
    # callee's `N` leaked into the result as the caller's own `N`
    # (`N+M+1` came back as `M+2*N+1`), or as `Never`. What cannot be
    # solved, or cannot be told from the callee's own dim, is unknown.
    result = run_mypy(FIXTURES / "callsite_formula_ok.py")
    assert result.returncode == 0, result.stdout
    f64 = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.float64]]"
    assert reveals_by_line(result) == {
        36: f'"{f64.format("2*M+1, 3*M+N+2")}"',
        37: f'"{f64.format("N+2, N+3")}"',
        38: f'"{f64.format("N+1, N+2")}"',
        39: f'"{f64.format("?U@39, Literal[2]")}"',
        40: f'"{f64.format("?U@40, ?U@40b")}"',
        41: f'"{f64.format("Literal[3], Literal[7]")}"',
    }, result.stdout
    assert _run_fixture("callsite_formula_ok.py") == [
        "(7, 13)", "(4, 5)", "(3, 4)", "(3, 2)", "(3, 4)", "(3, 7)",
    ]


def test_hooks_the_oracle_reached_first() -> None:
    # permute_dims, np.round, linalg.cross, np.all/np.any and builtin
    # divmod had no test until the oracle generated them. It found:
    # `permute_dims(a, axes=...)` by keyword reversed the axes instead of
    # applying them; `np.round` of a non-bool array lost the shape and
    # `np.around` was not hooked at all; `linalg.cross` with a symbolic
    # vector axis fell back to the stubs, which answer int8 for rank 3
    # and reject the call.
    result = run_mypy(FIXTURES / "oracle_hooks_ok.py")
    nd = "numpy.ndarray[tuple[{}], numpy.dtype[{}]]"
    f64, boolean = "numpy.float64", "numpy.bool[bool]"
    bits = "numpy._typing._nbit_base._{}Bit"
    f16 = f"numpy.floating[{bits.format(16)}]"
    i64 = f"numpy.signedinteger[{bits.format(64)}]"
    lit = lambda *dims: ", ".join(f"Literal[{d}]" for d in dims)  # noqa: E731
    assert reveals_by_line(result) == {
        26: f'"{nd.format(lit(4, 2, 3), f64)}"',
        27: f'"{nd.format(lit(3, 2, 4), f64)}"',
        28: f'"{nd.format("M, N", f64)}"',
        29: f'"{nd.format("N, M", f64)}"',
        30: f'"{nd.format(lit(2, 3, 4), f64)}"',
        31: f'"{nd.format(lit(2, 3), f16)}"',
        32: f'"{nd.format(lit(2, 3), f16)}"',
        33: f'"{nd.format("K, Literal[3], Literal[3]", i64)}"',
        34: f'"{nd.format("M", boolean)}"',
        35: f'"{nd.format(lit(1, 3, 1), boolean)}"',
        36: f'"{nd.format("N, M", f64)}"',
    }, result.stdout
    # Line 33 is rank 3: the stubs type linalg.cross for 1-D and 2-D
    # only, and used to add two [arg-type] errors to a call numpy accepts.
    assert result.returncode == 0, result.stdout
    assert _run_fixture("oracle_hooks_ok.py") == [
        "(4, 2, 3) float64", "(3, 2, 4) float64", "(6, 5) float64", "(5, 6) float64",
        "(2, 3, 4) float64", "(2, 3) float16", "(2, 3) float16", "(3, 3, 3) int64",
        "(6,) bool", "(1, 3, 1) bool", "(5, 6) float64",
    ]


def test_round_bool_with_decimals_claims_nothing() -> None:
    # numpy cannot cast the scaled values back to bool and raises; numty
    # used to reveal the input's shape for a call that never returns.
    result = run_mypy(FIXTURES / "round_bool_decimals.py")
    assert reveals_by_line(result) == {8: '"Any"', 9: '"Any"', 10: '"Any"'}, result.stdout


def test_callsite_formula_returns_through_methods() -> None:
    # Found by the oracle once it declared its generic helpers on
    # classes: methods had the call-site conflict check but not the
    # formula-return handling functions have, so `-> F64[N, "N+1"]` on
    # an instance, static or class method or on `__call__` came back
    # with `Never` for the formula dim.
    result = run_mypy(FIXTURES / "callsite_formula_methods_ok.py")
    assert result.returncode == 0, result.stdout
    f64 = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.float64]]"
    assert reveals_by_line(result) == {
        36: f'"{f64.format("2*M+1, 2*M+2")}"',
        37: f'"{f64.format("2*M+2*N, M+N")}"',
        38: f'"{f64.format("N+2")}"',
        39: f'"{f64.format("M, 2*M")}"',
        40: f'"{f64.format("Literal[4], Literal[5]")}"',
        41: f'"{f64.format("Literal[3]")}"',
    }, result.stdout
    assert _run_fixture("callsite_formula_methods_ok.py") == [
        "(7, 8)", "(10, 5)", "(4,)", "(3, 6)", "(4, 5)", "(3,)",
    ]


def test_callsite_unknown_return_compatible() -> None:
    # Issue 218: unknown-shape actuals give Any dims, not rigid ?U, so a
    # caller returning the result against declared TypeVars stays clean.
    result = run_mypy(FIXTURES / "callsite_unknown_return_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout


def test_callsite_unknown_tuple_and_box() -> None:
    # Issue 218: tuple/Box returns with unknown args do not need
    # annotations; they match the direct (Any-dim) result.
    result = run_mypy(FIXTURES / "callsite_unknown_tuple_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    assert "tuple[Any]" in notes[29], result.stdout
    assert "tuple[Any]" in notes[31], result.stdout
    assert "float64" not in notes[33] or "Any" in notes[33] or "Never" in notes[33], result.stdout


def test_callsite_tuple_formula_solved() -> None:
    # Issue 222: formula dims inside tuple returns solve like top-level.
    result = run_mypy(FIXTURES / "callsite_tuple_formula_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    assert "Literal[5]" in notes[21], result.stdout
    assert "Literal[5]" in notes[22] and "Literal[4]" in notes[22], result.stdout
    assert "Never" not in result.stdout, result.stdout
    assert "var-annotated" not in result.stdout, result.stdout


def test_generic_namedtuple_context_kept() -> None:
    # Issue 224: Frames(rows) in Frames[int] context takes int from the
    # context, not N from the argument.
    result = run_mypy(FIXTURES / "generic_namedtuple_context_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout


def test_recursive_alias_return_ok() -> None:
    # Issue 243: a call returning a recursive alias stays clean (no
    # RecursionError -> [numty-internal]).
    result = run_mypy(FIXTURES / "recursive_alias_return_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    assert "internal" not in result.stdout, result.stdout


def test_generic_namedtuple_method_keeps_receiver() -> None:
    # Issue 244 (+220 for NamedTuple): methods of a generic NamedTuple
    # keep receiver TypeVars; field dims are not minted as ?U.
    result = run_mypy(FIXTURES / "generic_namedtuple_method_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    assert "Block[int, F]" in notes[27], result.stdout
    assert "?U" not in notes[27], result.stdout
    assert "Block[Q, K]" in notes[31], result.stdout
    assert "Never" not in result.stdout, result.stdout


def test_generic_namedtuple_box_unknown_arg() -> None:
    # Issue 245: Box built from an unknown-shape argument is Box[Any];
    # item access and attribute access agree (no Never fallback).
    result = run_mypy(FIXTURES / "generic_namedtuple_box_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    assert "Box[Any]" in notes[25], result.stdout
    assert "tuple[Any]" in notes[26], result.stdout
    assert "tuple[Any]" in notes[27], result.stdout
    assert "Never" not in result.stdout, result.stdout
    assert "var-annotated" not in result.stdout, result.stdout


def test_ret_int_ok() -> None:
    # Issues #269/#271: a plain-`int` size in a returned
    # value fits a declared named size, inline and through a name, bound
    # or unbound. Each call of an unbound-name function still returns a
    # fresh unknown; literals and unknowns keep checking
    # (see bad_ret_literal, bad_mask_ts).
    result = run_mypy(FIXTURES / "ret_int_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    notes = {
        line: re.sub(r"\?U@\d+[a-z]?", "?U", shape)
        for line, shape in reveals_by_line(result).items()
    }
    f64 = "numpy.dtype[numpy.float64]"
    assert notes == {
        93: '"numpy.ndarray[tuple[?N@93], numpy.dtype[numpy.bool[Any]]]"',
        94: '"numpy.ndarray[tuple[?N@94], numpy.dtype[numpy.bool[Any]]]"',
        95: f'"numpy.ndarray[tuple[Literal[4]], {f64}]"',
    }, result.stdout
    assert _run_fixture("ret_int_ok.py") == [
        "(5,)",
        "(6,)",
        "(4,)",
    ]


def test_bad_ret_literal() -> None:
    # Issues #269/#271: a literal size in a returned value still checks
    # against a declared named size that a parameter binds. (A name of
    # the return alone takes it, see own_size_return.py.)
    assert_single_error(
        run_mypy(FIXTURES / "bad_ret_literal.py"),
        "Incompatible return value",
        "tuple[Literal[0]]",
        "tuple[N]",
    )


def test_builder_int_tuple_return_checked() -> None:
    # Issue 233: a tuple with one plain-int size reports the R-vs-A swap
    # both inline and through a name (same [return-value] twice). A
    # dynamic dim is `Any` at a return (issues #269/#271), but not here:
    # `R` is declared on the axis that holds the `int` and sits on the
    # other axis of the value, so the `int` is kept (issue 297).
    result = run_mypy(FIXTURES / "bad_builder_int_tuple.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 2, result.stdout
    for line in errors:
        assert "[return-value]" in line, result.stdout
        assert "tuple[int, R]" in line, result.stdout
        assert "tuple[R, A]" in line, result.stdout


def test_abstract_dtype_promotion() -> None:
    # Issue 248: with an abstract dtype (np.integer, np.intp, np.floating)
    # the result dtype follows numpy promotion on the kinds. It is not the
    # dtype of the first operand.
    result = run_mypy(FIXTURES / "abstract_dtype_promotion_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    f64 = "numpy.dtype[numpy.float64]"
    floating = "numpy.dtype[numpy.floating[Any]]"
    for line in (20, 21, 22, 23, 24, 25, 26, 27):
        assert notes[line] == f'"numpy.ndarray[tuple[N], {f64}]"', result.stdout
    for line in (28, 29, 30, 31, 32, 39):
        assert notes[line] == f'"numpy.ndarray[tuple[N], {floating}]"', result.stdout
    assert notes[33] == f'"numpy.ndarray[_S, {floating}]"', result.stdout
    assert notes[34] == f'"numpy.ndarray[_S, {f64}]"', result.stdout
    assert notes[35] == f'"numpy.ndarray[_S, {f64}]"', result.stdout
    assert "signedinteger[numpy._typing._nbit_base._64Bit]" in notes[36], result.stdout
    assert "_32Bit | numpy._typing._nbit_base._64Bit" in notes[37], result.stdout
    # Issue 267: two np.integer operands can be int64 with uint64 (float64).
    assert notes[38] == '"numpy.ndarray[tuple[N], numpy.dtype[Any]]"', result.stdout
    # A TypeVar dtype: kept with Python scalars and with itself.
    assert notes[66] == '"numpy.ndarray[tuple[N], numpy.dtype[ST]]"', result.stdout
    assert notes[67] == '"numpy.ndarray[tuple[N], numpy.dtype[ST]]"', result.stdout
    # timedelta64 and object_ decide the result with a numeric operand, in
    # each operand order.
    td = '"numpy.ndarray[tuple[N], numpy.dtype[numpy.timedelta64[Any]]]"'
    for line in (72, 73, 74, 75):
        assert notes[line] == td, result.stdout
    # Issue 267: timedelta64 / timedelta64 is float64.
    assert notes[76] == '"numpy.ndarray[tuple[N], numpy.dtype[numpy.float64]]"', result.stdout
    for line in (77, 78, 79):
        assert notes[line] == '"numpy.ndarray[tuple[N], numpy.dtype[numpy.object_]]"', result.stdout
    assert notes[80] == '"numpy.ndarray[tuple[N], numpy.dtype[Any]]"', result.stdout
    assert notes[81] == '"numpy.ndarray[tuple[N], numpy.dtype[numpy.complex128]]"', result.stdout


def test_abstract_dtype_promotion_errors() -> None:
    # Issue 248: `intp / 2` is float64 and `floating / float` keeps the
    # floating dtype, so both wrong return dtypes are reported.
    result = run_mypy(FIXTURES / "bad_abstract_dtype_promotion.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 2, result.stdout
    assert ":13:" in errors[0] and "dtype[float64]" in errors[0], result.stdout
    assert ":17:" in errors[1] and "dtype[floating[Any]]" in errors[1], result.stdout


def test_column_stack_bare_names_items() -> None:
    # Issue 234: the bail names the item type, not the sequence length.
    result = run_mypy(FIXTURES / "bad_column_stack_bare.py")
    assert result.returncode == 1, result.stdout
    assert "the items have no known shape" in result.stdout, result.stdout
    assert "annotate the items or ignore" in result.stdout, result.stdout
    assert "known length" not in result.stdout, result.stdout
    assert "preallocate" not in result.stdout, result.stdout


def test_permutation_of_a_0d_array() -> None:
    # numpy raises; numty used to reveal shape () for it.
    result = run_mypy(FIXTURES / "bad_permutation_0d.py")
    assert result.returncode == 1, result.stdout
    errors = [line.split(": error: ")[1] for line in result.stdout.splitlines() if ": error:" in line]
    assert errors == [
        "permutation requires an int or at least 1 dimension, got shape ()  [numty-shape]"
    ] * 2, result.stdout
    assert len(reveals_by_line(result)) == 2, result.stdout


def test_reshape_targets_written_as_arithmetic() -> None:
    # Found by the oracle's symbolic reshape family.
    # - `x.reshape(x.shape[0] * x.shape[1])` was read as a plain `int`
    #   and the size lost; arithmetic over sizes is now folded.
    # - `.T`, swapaxes and moveaxis on a shape with an axis of unknown
    #   size took the stubs' Self: the shape came back unmoved.
    # - `sum` along an axis of such an array took the stubs' dtype,
    #   signed for a uint8 input where numpy gives uint64.
    # - (N+1, 2) to (2*N+2,) was "reshape changes total size: 2 -> 1":
    #   only the literal factors were compared, and 2*N+2 hides one.
    result = run_mypy(FIXTURES / "reshape_arithmetic_ok.py")
    assert result.returncode == 0, result.stdout
    nd = "numpy.ndarray[tuple[{}], numpy.dtype[{}]]"
    bits = "numpy._typing._nbit_base._{}Bit"
    f64 = "numpy.float64"
    u8, u64 = (f"numpy.unsignedinteger[{bits.format(n)}]" for n in (8, 64))
    assert reveals_by_line(result) == {
        19: f'"{nd.format("M*N", u8)}"',
        20: f'"{nd.format("Literal[1], 3*M", f64)}"',
        21: f'"{nd.format("2*M, ?U@21", u8)}"',
        22: f'"{nd.format("Literal[1], M*N", u8)}"',
        24: f'"{nd.format("Literal[3], Any, Literal[1]", f64)}"',
        25: f'"{nd.format("Literal[3], Any, Literal[1]", f64)}"',
        26: f'"{nd.format("Any, Literal[3], Literal[1]", f64)}"',
        27: f'"{nd.format("Any", u64)}"',
        29: f'"{nd.format("int", f64)}"',
        30: f'"{nd.format("Literal[2]", f64)}"',
    }, result.stdout
    assert _run_fixture("reshape_arithmetic_ok.py") == [
        "(24,) uint8", "(1, 12) float64", "(8, 3) uint8", "(1, 24) uint8",
        "(3, 4, 1) float64", "(3, 4, 1) float64", "(4, 3, 1) float64", "(24,) uint64",
        "(14,) float64", "(2,) float64",
    ]


def test_builders_keep_arithmetic_sizes() -> None:
    # Issue #186: a size written as arithmetic was typed `int` by mypy,
    # so builders lost it where reshape (#184) already folded it.
    result = run_mypy(FIXTURES / "builders_arithmetic_ok.py")
    assert result.returncode == 0, result.stdout
    nd = "numpy.ndarray[tuple[{}], numpy.dtype[{}]]"
    bits = "numpy._typing._nbit_base._{}Bit"
    f64 = "numpy.float64"
    i64 = f"numpy.signedinteger[{bits.format(64)}]"
    assert reveals_by_line(result) == {
        20: f'"{nd.format("M*N, M+1", f64)}"',
        21: f'"{nd.format("2*M", f64)}"',
        22: f'"{nd.format("M+1, 2*N", f64)}"',
        23: f'"{nd.format("M*N", f64)}"',
        24: f'"{nd.format("M*N", f64)}"',
        25: f'"{nd.format("M-1, N//2", f64)}"',
        27: f'"{nd.format("?U@27", f64)}"',
        29: f'"{nd.format("M*N, M+1", f64)}"',
        30: f'"{nd.format("2*M", f64)}"',
        31: f'"{nd.format("M+1, 2*N", i64)}"',
        35: f'"{nd.format("M*N, M+1", f64)}"',
        36: f'"{nd.format("2*M", f64)}"',
    }, result.stdout
    assert _run_fixture("builders_arithmetic_ok.py") == [
        "(24, 5) float64", "(8,) float64", "(5, 12) float64", "(24,) float64",
        "(24,) float64", "(3, 3) float64", "(35,) float64",
        "(24, 5) float64", "(8,) float64", "(5, 12) int64",
        "(24, 5) float64", "(8,) float64",
    ]


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
    # Issue #148: OOB names op and shape (take/split/squeeze form).
    assert_single_error(
        run_mypy(FIXTURES / "bad_sum_axis.py"),
        "sum axis 5 is out of range for 2 dimensions",
        "(2, 3)",
    )


def test_expand_axis_mismatch() -> None:
    # Issue #148: OOB names shape alongside the new-axes count.
    assert_rank_mismatch(
        run_mypy(FIXTURES / "bad_expand_axis.py"),
        "expand_dims axis 5 is out of range",
        '"expand_dims"',
        "(3,)",
    )


def test_folded_size_mismatch() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_fold_size.py"),
        "tuple[Literal[4], Literal[3]]",
        "tuple[Literal[2], Literal[4]]",
    )


def test_folded_reshape_mismatch() -> None:
    # Issue #148: single targeted error, no Sequence[Never] follow-on.
    assert_single_error(
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
    # Issue #148: single targeted error, no Sequence[Never] follow-on.
    assert_single_error(
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
    # The padded block keeps the time axis (`*frames.shape`, not `[1:]`).
    # The two dynamic dims are `Any` at the return (issues #269/#271), so
    # the error names the kept `total` axis against the declared shape.
    assert_single_error(
        run_mypy(FIXTURES / "bad_pad_time_axis.py"),
        "Incompatible return value",
        "tuple[Any, Any, total, *Ts]",
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
    # Unknowns keep checking at a return (only plain `int` is rewritten
    # to Any there, issues #269/#271), so the count still mismatches N.
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


def test_mask_int_dims_keep_dtype() -> None:
    # Issue #226: Bool[int] mask on F64[int] keeps dtype, not Any.
    result = run_mypy(FIXTURES / "index_mask_int_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    assert "float64" in notes[11] and "?U@" in notes[11], result.stdout
    assert "float64" in notes[12] and "?U@" in notes[12], result.stdout
    assert notes[13] == '"numpy.float64"', result.stdout


def test_mask_symbolic_length_mismatch() -> None:
    # Issue #228: N-1 vs N mask reports like literal 4 vs 5, for getitem
    # and setitem.
    result = run_mypy(FIXTURES / "bad_mask_symbolic_len.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 3, result.stdout
    for line in errors:
        assert "boolean index did not match" in line, result.stdout
        assert "[numty-axis]" in line, result.stdout


def test_int_scalar_and_list_indices() -> None:
    # Issue #231: np.intp/np.int64 scalars index like int; x[[-1]] is (1, N).
    result = run_mypy(FIXTURES / "index_int_scalar_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    assert "tuple[N]" in notes[15], result.stdout
    assert "tuple[N]" in notes[16], result.stdout
    assert "tuple[N]" in notes[17], result.stdout
    assert "tuple[Literal[1], N]" in notes[20], result.stdout


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
        # keepdims sum over the literal axis keeps (int, 1): the reduction
        # axis is known to be 1, not widened (issue #227 movable dims).
        '"numpy.ndarray[tuple[int, Literal[1]], numpy.dtype[numpy.float64]]"',  # keepdims sum
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


def test_f64_star_needs_typevartuple() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_f64_star.py"),
        "F64 star dims take a TypeVarTuple, e.g. *Ts",
        "[numty-annotation]",
    )


def test_f64_star_form() -> None:
    # Issue 282: `F32[*Ts]` was refused with "write Unpack[Ts] instead",
    # and ruff's UP044 asks for the star. Both spellings are one type.
    result = run_mypy(FIXTURES / "f64_star_ok.py")
    assert result.returncode == 0, result.stdout
    f32 = "numpy.dtype[numpy.floating[numpy._typing._nbit_base._32Bit]]"
    assert reveals_by_line(result) == {
        24: f'"numpy.ndarray[tuple[Literal[3], Literal[4]], {f32}]"',
        25: '"numpy.ndarray[tuple[Literal[5], Literal[7]], numpy.dtype[numpy.float64]]"',
        26: f'"numpy.ndarray[tuple[Literal[3], Literal[4]], {f32}]"',
    }, result.stdout
    assert _run_fixture("f64_star_ok.py") == ["(3, 4)", "(5, 7)"]


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


def test_recursive_alias_formal_keeps_callsite_check() -> None:
    # Issue #164: walking a recursive alias hit RecursionError, which the
    # call-site hook swallowed, silently skipping the whole call.
    assert_single_error(
        run_mypy(FIXTURES / "bad_recursive_alias_callsite.py"),
        "call-site shape conflict for 'batch'",
        "4 (argument 'x', axis 0) vs 5 (argument 'y', axis 0)",
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
        "(2, 3)",
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
    # A loop-built list of (n_samples,) arrays stacks to an unknown length.
    # The new axis mints ?U (issue #227): the length is unknowable, but the
    # element rank is kept instead of bailing the whole result to Any.
    # Downstream the ?U then mismatches the annotated shapes.
    pytest.importorskip("sklearn")
    result = run_mypy(FIXTURES / "sklearn_gnb.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 2, result.stdout
    assert "tuple[n_samples, ?U" in errors[0] and "[assignment]" in errors[0], result.stdout
    # Without allow-redefinition the rebind fails, so the return still
    # sees the list type rather than the stacked ?U array.
    assert "list[ndarray" in errors[1] and "[return-value]" in errors[1], result.stdout
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
    iu = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.signedinteger[numpy._typing._nbit_base._32Bit | numpy._typing._nbit_base._64Bit]]]"  # noqa: E501
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
        45: '"numpy.ndarray[tuple[Literal[1], Literal[1]], numpy.dtype[numpy.bool[bool]]]"',
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


def test_ufunc_methods_ok_clean_and_runs() -> None:
    # Issue #142: ufunc reduce drops the axis, accumulate keeps the
    # shape, outer concatenates operand shapes, reduceat sizes the axis
    # by the indices length; add.at still returns None. Golden reveals
    # plus a runtime oracle.
    result = run_mypy(FIXTURES / "ufunc_methods_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    any_dt = "numpy.ndarray[tuple[{}], Any]"
    # Issue 304: add and multiply keep a float operand's dtype in reduce,
    # accumulate and reduceat. outer is still shape only.
    f64 = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.float64]]"
    assert reveals_by_line(result) == {
        20: f'"{f64.format("M")}"',
        21: f'"{f64.format("N")}"',
        22: f'"{f64.format("N, M")}"',
        23: f'"{any_dt.format("N, M, Literal[2]")}"',
        28: f'"{f64.format("Literal[3]")}"',
        29: f'"{f64.format("Literal[2]")}"',
        30: f'"{f64.format("Literal[2], Literal[1]")}"',
        31: '"numpy.float64"',
        32: f'"{f64.format("Literal[2], Literal[3]")}"',
        33: f'"{any_dt.format("Literal[3], Literal[4]")}"',
        34: f'"{any_dt.format("Literal[3]")}"',
        # Issue 283: minimum picks elements, so the dtype is the operand's.
        35: '"numpy.ndarray[tuple[Literal[2]], numpy.dtype[numpy.float64]]"',
        36: f'"{f64.format("Literal[2]")}"',
        37: f'"{f64.format("Literal[4], Literal[2]")}"',
        38: '"numpy.ndarray[tuple[Literal[2]], '
        'numpy.dtype[numpy.signedinteger[numpy._typing._nbit_base._64Bit]]]"',
        39: '"None"',
    }, result.stdout
    assert _run_fixture("ufunc_methods_ok.py") == [
        "(7,)",
        "(5,)",
        "(5, 1)",
        "()",
        "(5, 7)",
        "(3, 4)",
        "(3,)",
        "(5,)",
        "(2,)",
        "(4, 2)",
        "(5,)",
        "None",
        "(3,)",
        "(2,)",
        "(2, 3)",
        "(2, 3, 2)",
    ]


def test_ufunc_method_axis_mismatch() -> None:
    # Issue #142: out-of-range ufunc-method axes fail with [numty-axis].
    result = run_mypy(FIXTURES / "bad_ufunc_method_axis.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 4, result.stdout
    assert errors[0].startswith(
        "tests/fixtures/bad_ufunc_method_axis.py:6: error:"
    ), result.stdout
    assert "reduce axis 5 is out of range for 2 dimensions" in errors[0], result.stdout
    assert errors[1].startswith(
        "tests/fixtures/bad_ufunc_method_axis.py:7: error:"
    ), result.stdout
    assert "accumulate axis -3 is out of range for 2 dimensions" in errors[1], (
        result.stdout
    )
    assert errors[2].startswith(
        "tests/fixtures/bad_ufunc_method_axis.py:8: error:"
    ), result.stdout
    assert "reduce axis 2 is out of range for 2 dimensions" in errors[2], result.stdout
    assert errors[3].startswith(
        "tests/fixtures/bad_ufunc_method_axis.py:9: error:"
    ), result.stdout
    assert "reduceat axis 1 is out of range for 1 dimensions" in errors[3], (
        result.stdout
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
    # stack fail instead of bailing to a silent ?U/Any. Each is a single
    # targeted error (issue #148: no mechanical operator/stack follow-on).
    result = run_mypy(FIXTURES / "bad_provable_linear.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 3, result.stdout
    assert "broadcast shapes disagree at trailing axis 0" in errors[0], result.stdout
    assert "N vs N-1" in errors[0], result.stdout
    assert "matmul inner dimensions disagree: N vs N+2" in errors[1], result.stdout
    assert "stack dimension 0 differs: N+2 vs N" in errors[2], result.stdout
    assert "(N+2,)" in errors[2] and "(N,)" in errors[2], result.stdout
    # Issue 266: the stack axis is not in the message.
    assert "axis 0" not in errors[2], result.stdout


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


def test_reshape_total_sums_monos_divs() -> None:
    # Issue #137: (N+1)*M vs N*M and 2*N*M vs N*M always raise;
    # N//2+1 vs N (rfft, N=1,2 legal) bails to Any instead of
    # tracking a precise impossible (N,).
    result = run_mypy(FIXTURES / "bad_reshape_total_137.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 2, result.stdout
    assert "reshape changes total size: N -> N+1" in errors[0], result.stdout
    assert "(N, M) vs (N+1, M)" in errors[0], result.stdout
    assert "reshape changes total size: M*N -> 2*M*N" in errors[1], result.stdout
    assert "(N, M) vs (2*N, M)" in errors[1], result.stdout
    assert reveals_by_line(result) == {
        29: '"numpy.ndarray[Any, Any]"',
    }, result.stdout


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


def test_diff_0d_mismatch() -> None:
    # Issue #130: np.diff on 0-d is a normal error, not a silent Any bail.
    assert_single_error(
        run_mypy(FIXTURES / "bad_diff_0d.py"),
        "np.diff requires at least 1 dimension, got shape ()",
    )


def test_take_convolve_0d_ok_clean() -> None:
    # Issue #131: take/convolve accept 0-d via the flattening path.
    result = run_mypy(FIXTURES / "take_convolve_0d_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    f64 = "numpy.dtype[numpy.float64]"
    assert reveals_by_line(result) == {
        6: f'"numpy.ndarray[tuple[Literal[1]], {f64}]"',
        7: f'"numpy.ndarray[tuple[Literal[1]], {f64}]"',
        8: '"numpy.float64"',
    }, result.stdout


def test_matrix_transpose_rank_mismatch() -> None:
    # Issue #129: matrix_transpose/.mT require rank >= 2.
    result = run_mypy(FIXTURES / "bad_matrix_transpose_rank.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 3, result.stdout
    assert "matrix_transpose requires rank >= 2, got shape (3,)" in errors[0], result.stdout
    assert "matrix_transpose requires rank >= 2, got shape (3,)" in errors[1], result.stdout
    assert "matrix_transpose requires rank >= 2, got shape ()" in errors[2], result.stdout


def test_class_attr_substitution() -> None:
    # Issue #128: Padded[Literal[4]].p substitutes N := 4 into N+2.
    result = run_mypy(FIXTURES / "bad_class_attr.py")
    assert result.returncode == 1, result.stdout
    assert reveals_by_line(result) == {
        17: '"numpy.ndarray[tuple[Literal[6]], numpy.dtype[numpy.float64]]"',
    }, result.stdout
    # Issue #148: single targeted error, no operator duplicate.
    assert_single_error(
        run_mypy(FIXTURES / "bad_class_attr.py"),
        "broadcast shapes disagree at trailing axis 0",
        "6 vs 9",
        "(6,)",
        "(9,)",
    )


def test_inline_generic_call_ok_clean_and_runs() -> None:
    # Issue #55: nested generic calls keep shapes inline (1 level from
    # the issue plus 2-4 level ML nesting), concrete and symbolic.
    result = run_mypy(FIXTURES / "inline_generic_call_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    f64 = "numpy.dtype[numpy.float64]"
    assert reveals_by_line(result) == {
        36: f'"numpy.ndarray[tuple[Literal[4], Literal[3]], {f64}]"',
        37: f'"numpy.ndarray[tuple[Literal[4], Literal[5]], {f64}]"',
        38: f'"numpy.ndarray[tuple[Literal[4], Literal[5]], {f64}]"',
        39: f'"numpy.ndarray[tuple[Literal[3]], {f64}]"',
        40: f'"numpy.ndarray[tuple[Literal[4], Literal[5]], {f64}]"',
        41: '"numpy.float64"',
        42: f'"numpy.ndarray[tuple[Literal[4], Literal[5]], {f64}]"',
        43: f'"numpy.ndarray[tuple[Literal[4], Literal[7]], {f64}]"',
        47: f'"numpy.ndarray[tuple[n, h], {f64}]"',
        48: f'"numpy.ndarray[tuple[n, h], {f64}]"',
        49: f'"numpy.ndarray[tuple[h], {f64}]"',
    }, result.stdout
    assert _run_fixture("inline_generic_call_ok.py") == [
        "(4, 3)",
        "(4, 5)",
        "(4, 5)",
        "(4, 5)",
        "(4, 5)",
        "(4, 7)",
    ]


def test_inline_generic_call_mismatch() -> None:
    # Issue #55: the nested shape survives, so the (4, 5) vs (4, 6)
    # mismatch is flagged instead of passing silently through Any.
    # Issue #148: single targeted error, no operator duplicate.
    assert_single_error(
        run_mypy(FIXTURES / "bad_inline_generic_call.py"),
        "broadcast shapes disagree at trailing axis 0",
        "5 vs 6",
        "(4, 5)",
        "(4, 6)",
    )


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


def test_union_attr_substitutes_expr_dims() -> None:
    # A union-typed attribute with expression dims (`F64["N+1"] | None`)
    # reported [numty-internal] at every access: the substitution
    # called UnionType.copy_modified, which does not exist. Found by ty.
    result = run_mypy(FIXTURES / "union_attr_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    f64 = "numpy.ndarray[tuple[Literal[{}]], numpy.dtype[numpy.float64]]"
    assert reveals_by_line(result) == {
        30: f'"{f64.format(4)} | None"',
        31: f'"{f64.format(4)} | {f64.format(6)}"',
    }, result.stdout
    assert _run_fixture("union_attr_ok.py") == ["None", "(6,)"]


def test_class_tvar_in_method_call_ok() -> None:
    # `self.same(y)` inside a generic class has the class TypeVar in its
    # signature. `_is_class_tvar` read `.fullname` from a list of str and
    # raised in the method signature hook: an INTERNAL ERROR in debug
    # mode, a swallowed exception otherwise. Found by ty (#213).
    result = run_mypy(FIXTURES / "class_tvar_method_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    f64 = "numpy.ndarray[tuple[{}], numpy.dtype[numpy.float64]]"
    assert reveals_by_line(result) == {
        33: f'"{f64.format("N")}"',
        38: f'"{f64.format("Literal[3]")}"',
    }, result.stdout
    assert _run_fixture("class_tvar_method_ok.py") == ["(3,)", "(3,)"]


def test_symbolic_axes_bail_without_internal_error() -> None:
    # An axis tuple typed with TypeVars reached integer comparisons and
    # reported [numty-internal] (TypeError) for squeeze. The axis
    # callers now use `_int_shape_from_arg` and bail (#213).
    result = run_mypy(FIXTURES / "symbolic_axes_ok.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    assert reveals_by_line(result) == {
        line: '"numpy.ndarray[Any, Any]"' for line in (19, 20, 21)
    }, result.stdout


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


def test_generic_namedtuple_replace_and_unpack() -> None:
    # Issues #219/#221: a generic NamedTuple keeps mypy's TupleType handling,
    # so _replace checks and tuple unpacking stay per-field.
    for name in ("generic_replace_ok.py", "generic_unpack_ok.py"):
        result = run_mypy(FIXTURES / name)
        assert result.returncode == 0, result.stdout
    assert reveals_by_line(run_mypy(FIXTURES / "generic_unpack_ok.py")) == {
        22: '"numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]"',
    }


def test_generic_method_keeps_receiver_tvars() -> None:
    # Issue #220: G[Q, H].take(typed: K) is G[Q, K], not G[Never, K].
    result = run_mypy(FIXTURES / "generic_method_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    assert "G[Q, K]" in notes[23], result.stdout
    assert "G[P, K]" in notes[24], result.stdout
    assert "Never" not in result.stdout, result.stdout


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
    # Issue #148: single targeted error, no operator duplicate.
    assert_single_error(
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


def test_contract_pair_symbolic_mismatch() -> None:
    # Issue #139: dot/inner/tensordot/vdot missed N+1 vs N that
    # matmul/broadcast already flag; distinct ?U bails (Any) not scalar.
    result = run_mypy(FIXTURES / "bad_contract_pair_symbolic.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    solver = [
        line
        for line in errors
        if "contracted dimensions disagree" in line or "different sizes" in line
    ]
    assert len(solver) == 6, result.stdout
    assert "dot contracted dimensions disagree: N+1 vs N" in solver[0], result.stdout
    assert "dot contracted dimensions disagree: N+1 vs N" in solver[1], result.stdout
    assert "inner contracted dimensions disagree: N+1 vs N" in solver[2], result.stdout
    assert "tensordot contracted dimensions disagree: N+1 vs N" in solver[3], result.stdout
    assert "dot contracted dimensions disagree: N+1 vs N" in solver[4], result.stdout
    assert "vdot operands have different sizes: N+1 vs N" in solver[5], result.stdout


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


def test_builders_intp_keeps_typevar() -> None:
    # Issue #223: np.full(len(x), v, dtype=np.intp) is (T,) like the tuple
    # shape and like np.int64, not (int,).
    result = run_mypy(FIXTURES / "builders_intp_ok.py")
    assert result.returncode == 0, result.stdout


def test_builders_r_nan_concat_float() -> None:
    # Issue #230: float list items, np.r_ and nan_to_num are solved.
    result = run_mypy(FIXTURES / "builders_r_nan_concat_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    assert "tuple[N+1]" in notes[11], result.stdout
    assert "tuple[N+1]" in notes[12] and "float64" in notes[12], result.stdout
    assert "tuple[N+1]" in notes[13] and "float64" in notes[13], result.stdout
    assert "tuple[N+1]" in notes[14] and "float64" in notes[14], result.stdout
    assert "tuple[N]" in notes[15] and "bool" in notes[15], result.stdout
    assert "tuple[N]" in notes[16] and "float64" in notes[16], result.stdout
    assert "tuple[N]" in notes[17], result.stdout


def test_stack_any_static_keeps_shape() -> None:
    # Issue #232: static-length stack with dtype[Any] items is solved.
    result = run_mypy(FIXTURES / "stack_any_static_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    assert "tuple[R, A, Literal[2]]" in notes[28], result.stdout
    assert "tuple[R, A, Literal[3]]" in notes[29], result.stdout


def test_head_int_matches_spelled() -> None:
    # Issue #225: F64[int] has the same result types as the spelled ndarray.
    result = run_mypy(FIXTURES / "head_int_ok.py")
    assert result.returncode == 0, result.stdout
    notes = {line: re.sub(r"\?U@\d+[a-z]?", "?U", n) for line, n in reveals_by_line(result).items()}
    assert notes[10] == notes[17], result.stdout
    assert notes[11] == notes[18], result.stdout
    assert notes[12] == notes[19], result.stdout
    assert notes[13] == notes[20], result.stdout
    assert "float64" in notes[10], result.stdout


def test_int_dim_keeps_rank_and_checks() -> None:
    # Issue #227: an int dim is one unknown axis, not a whole-shape bail.
    # The rank and the literal axis survive, and 3 vs 4 is reported.
    result = run_mypy(FIXTURES / "bad_int_dim_broadcast.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 2, result.stdout
    for line in errors:
        assert "broadcast shapes disagree at trailing axis 0: 3 vs 4" in line, result.stdout
        assert "[numty-broadcast]" in line, result.stdout
    assert "tuple[N]" in result.stdout and "?U" in result.stdout, result.stdout


def test_whole_shape_typevar_kept() -> None:
    # Issue #229: elementwise ops on ndarray[_S, ...] keep _S.
    result = run_mypy(FIXTURES / "whole_shape_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    for line in (10, 11, 12):
        assert notes[line] == '"numpy.ndarray[_S, numpy.dtype[numpy.float64]]"', result.stdout


def test_bundle_b_broadcast_int_whole_r() -> None:
    # Bundle B (#217 #227 #229 #230): abstract dtypes keep shape,
    # plain-int dims keep rank (?U for unprovable axes), whole-shape _S
    # survives truediv/round, and r_/concat float lists solve.
    result = run_mypy(FIXTURES / "bundle_b_217_227_229_230_ok.py")
    assert result.returncode == 0, result.stdout
    raw = reveals_by_line(result)
    notes = {line: re.sub(r"\?U@\d+[a-z]?", "?U", n) for line, n in raw.items()}
    # #217: (N, 3) op (3,) keeps both dims with the floating dtype.
    assert notes[25] == '"numpy.ndarray[tuple[N, Literal[3]], numpy.dtype[numpy.floating[Any]]]"', result.stdout
    # Issue 248: with a dtype[Any] operand the result dtype is not known
    # (the operand can be complex), so it is dtype[Any]. The dims stay.
    assert notes[26] == '"numpy.ndarray[tuple[N, Literal[3]], numpy.dtype[Any]]"', result.stdout
    assert notes[27] == '"numpy.ndarray[tuple[N], numpy.dtype[Any]]"', result.stdout
    # #227: int dims keep rank; unprovable axes mint ?U.
    assert "tuple[int]" in notes[33] and "float64" in notes[33], result.stdout
    assert "tuple[?U]" in notes[34] and "bool" in notes[34], result.stdout
    assert "tuple[int, Literal[3]]" in notes[40], result.stdout
    assert "tuple[?U]" in notes[41], result.stdout
    assert "tuple[?U, Literal[2]]" in notes[44], result.stdout
    assert "tuple[int, int, Literal[3]]" in notes[46], result.stdout
    # #229: every elementwise op keeps _S, including truediv and round.
    for line in (50, 51, 52, 54, 55, 56):
        assert notes[line] == '"numpy.ndarray[_S, numpy.dtype[numpy.float64]]"', result.stdout
    assert "bool" in notes[53], result.stdout
    # #230: int-TypeVar scalars and nan float lists solve.
    assert "tuple[T+1]" in notes[61], result.stdout
    assert "tuple[T+2]" in notes[63], result.stdout
    assert "tuple[T+1]" in notes[64] and "float64" in notes[64], result.stdout
    assert "tuple[T+1]" in notes[65] and "float64" in notes[65], result.stdout


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
    # Inline "fro" with a single axis is a matrix-norm misuse (issue #148):
    # the signature hook widens ord/axis so the function hook's targeted
    # [numty-axis] error survives instead of overload spam.
    assert_single_error(
        run_mypy(FIXTURES / "bad_norm_fro_display.py"),
        "linalg.norm ord='fro' is a matrix norm and needs 2 axes, got 1",
        "(2, 3, 4)",
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
    # Issue #148: duplicate-axis names op, tuple and shape (take/split/squeeze form).
    assert_single_error(
        run_mypy(FIXTURES / "bad_sum_dup_axis.py"),
        "sum has duplicate axis -2 in (0, -2)",
        "(3, 4)",
    )


def test_negative_dims_are_refused_in_every_spelling() -> None:
    # `F64[-1]` was refused, but the same size spelled `"2-3"`, `"-1"`
    # or `Literal[-1]` went through as a dim of -1 (found by fuzzing
    # the string-dim parser against Python's own arithmetic).
    result = run_mypy(FIXTURES / "bad_f64_negative.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert [line.split(": error: ")[1] for line in errors] == [
        "F64 dim -1 is negative; a size cannot be below 0  [numty-annotation]",
        "F64 dim -1 is negative; a size cannot be below 0  [numty-annotation]",
        "F64 dim -1 is negative; a size cannot be below 0  [numty-annotation]",
        "F64 dim -time-1 is negative for every size; a size cannot be below 0"
        "  [numty-annotation]",
    ], result.stdout
    assert [int(line.split(":")[1]) for line in errors] == [8, 11, 14, 17]


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
    # Issue #148: single [numty-broadcast] error, no [operator] duplicate.
    assert_single_error(
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
    # Scan every .py under numty_plugin/ so Phase 2 leaf moves (_*.py)
    # never silently drop coverage. Floor pinned at 232 (233 on the
    # monolith, minus one for issue #217: the broadcast check moved
    # before dtype promotion, so the datetime path no longer needs its
    # own duplicate broadcast fail) so a lost site or missed file fails
    # loudly.
    import ast

    plugin_dir = ROOT / "numty_plugin"
    files = sorted(
        p for p in plugin_dir.rglob("*.py") if "__pycache__" not in p.parts
    )
    assert files, f"no .py files found under {plugin_dir} - glob broken?"
    assert plugin_dir / "__init__.py" in files, "scan must include __init__.py"
    sites: list[tuple[Path, ast.Call]] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        sites += [
            (path, node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fail"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "api"
        ]
    assert len(sites) > 100, len(sites)
    assert len(sites) >= 232, f"fail-site total dropped: {len(sites)} < 232"
    missing = [
        f"{path.relative_to(plugin_dir)}:{n.lineno}"
        for path, n in sites
        if not any(k.arg == "code" for k in n.keywords)
    ]
    assert not missing, f"ctx.api.fail without code= at {missing}"


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


def test_issue145_flatten_pin_ok() -> None:
    # Issue #145: method flatten paths (cumsum/cumprod axis=None, ravel,
    # flatten) pinned to the literal total like the function forms, and
    # the implicit string-spec local pin reuses the signature's dims.
    result = run_mypy(FIXTURES / "issue145_flatten_pin_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    f64_12 = '"numpy.ndarray[tuple[Literal[12]], numpy.dtype[numpy.float64]]"'
    for line in (28, 29, 30, 31, 32, 33, 34):
        assert notes[line] == f64_12, result.stdout
    assert notes[39] == (
        '"numpy.ndarray[tuple[batch, out], numpy.dtype[numpy.float64]]"'
    ), result.stdout
    assert notes[40] == (
        '"numpy.ndarray[tuple[batch, out], numpy.dtype[numpy.float64]]"'
    ), result.stdout
    assert _run_fixture("issue145_flatten_pin_ok.py") == ["ok"]


# README showcase: the opening examples are these fixtures verbatim, so
# the first thing a reader sees is pinned like everything else.


def test_showcase_plain_numpy_two_bugs() -> None:
    fixture = FIXTURES / "bad_showcase_plain.py"
    # The point of the example: no numty import, no annotation.
    source = fixture.read_text(encoding="utf-8")
    assert "import numty" not in source and "from numty" not in source
    assert "def " not in source and ": np." not in source
    result = run_mypy(fixture)
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 2, result.stdout
    assert errors[0].endswith(
        ":13: error: matmul inner dimensions disagree: 784 vs 10  [numty-shape]"
    ), result.stdout
    assert errors[1].endswith(
        ":17: error: true_divide shapes disagree: broadcast shapes disagree at "
        "trailing axis 0: 10 vs 64 (shapes (64, 10) vs (64,))  [numty-broadcast]"
    ), result.stdout


def test_showcase_attention_clean_and_runs() -> None:
    result = run_mypy(FIXTURES / "showcase_attention_ok.py")
    assert result.returncode == 0, result.stdout
    reveals = reveals_by_line(result)
    assert list(reveals) == [25], result.stdout
    assert "tuple[batch, heads, seq, seq]" in reveals[25], result.stdout
    assert _run_fixture("showcase_attention_ok.py") == ["(2, 4, 7, 16)"]


def test_showcase_attention_returning_weights() -> None:
    assert_single_error(
        run_mypy(FIXTURES / "bad_showcase_attention.py"),
        "Incompatible return value",
        "tuple[batch, heads, seq, seq]",
        "tuple[batch, heads, seq, dim]",
    )


def test_showcase_algebra_formulas() -> None:
    result = run_mypy(FIXTURES / "bad_showcase_algebra.py")
    reveals = reveals_by_line(result)
    golden = {
        25: "tuple[batch, time-1]",
        26: "tuple[batch, time//2+1]",
        27: "tuple[batch, time-4, Literal[5]]",
        28: "tuple[batch, time+4]",
        29: "tuple[batch, 2*time]",
        30: "tuple[batch, chan*height*width]",
        31: "tuple[batch]",
        32: "tuple[batch, time-1]",
    }
    assert sorted(reveals) == sorted(golden), result.stdout
    for line, shape in golden.items():
        assert f"ndarray[{shape}, " in reveals[line], (line, result.stdout)
    assert "numpy.complex128" in reveals[26], result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 1, result.stdout
    assert errors[0].endswith(
        ":33: error: broadcast shapes disagree at trailing axis 0: time-1 vs time "
        "(shapes (batch, time-1) vs (batch, time))  [numty-broadcast]"
    ), result.stdout


def test_rebind_allow_redefinition_ok() -> None:
    # Issue #162: ordinary rebinds need allow_redefinition=True; with it
    # the solver stays precise (reduction/transpose/reshape, if/else
    # union with per-member matmul). The loop accumulator is the known
    # weak spot: union plus the stubs fallback, so matmul on it is Any.
    result = run_mypy(
        FIXTURES / "rebind_allow_ok.py", flags=("--allow-redefinition",)
    )
    assert result.returncode == 0, result.stdout
    f64 = "numpy.dtype[numpy.float64]"
    iu = "numpy.dtype[numpy.signedinteger[numpy._typing._nbit_base._32Bit | numpy._typing._nbit_base._64Bit]]"
    assert reveals_by_line(result) == {
        22: f'"numpy.ndarray[tuple[batch], {f64}]"',
        28: f'"numpy.ndarray[tuple[Literal[3], Literal[4]], {f64}]"',
        30: f'"numpy.ndarray[tuple[Literal[4], Literal[3]], {f64}]"',
        32: f'"numpy.ndarray[tuple[Literal[12]], {iu}]"',
        34: f'"numpy.ndarray[tuple[Literal[3], Literal[4]], {iu}]"',
        38: f'"numpy.ndarray[tuple[Literal[0], Literal[3]], {f64}] | numpy.ndarray[tuple[Literal[1], Literal[3]], {f64}] | numpy.ndarray[tuple[Any, ...], numpy.dtype[Any]]"',  # noqa: E501
        46: f'"numpy.ndarray[tuple[Literal[3], Literal[4]], {f64}] | numpy.ndarray[tuple[Literal[5], Literal[4]], {f64}]"',  # noqa: E501
        47: f'"numpy.ndarray[tuple[Literal[3], Literal[2]], {f64}] | numpy.ndarray[tuple[Literal[5], Literal[2]], {f64}]"',  # noqa: E501
    }, result.stdout
    assert _run_fixture("rebind_allow_ok.py") == [
        "(5,)",
        "(4, 3)",
        "(3, 4)",
        "(3, 3)",
        "(3, 2)",
        "(5, 2)",
    ]


def test_rebind_strict_fails() -> None:
    # Same file under default flags: each rebind is an [assignment]
    # error (plus the return-value follow-on), documenting why the flag
    # is recommended.
    result = run_mypy(FIXTURES / "rebind_allow_ok.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 6, result.stdout
    assert sum("[assignment]" in line for line in errors) == 5, result.stdout
    assert sum("Incompatible return value" in line for line in errors) == 1, (
        result.stdout
    )


def test_ok_fixtures_clean_under_allow_redefinition() -> None:
    # Issue #162: every *_ok fixture stays clean with the flag (one
    # mypy run, not one subprocess per file).
    ok_files = sorted(FIXTURES.glob("*_ok.py"))
    assert ok_files, "no *_ok fixtures found"
    result = run_mypy(*ok_files, flags=("--allow-redefinition",))
    assert result.returncode == 0, result.stdout


def test_sklearn_clean_under_allow_redefinition() -> None:
    # Issue #162: sklearn dogfood stays precise with the flag; gnb keeps
    # its ?U length (the rebind split from #163 is flag-proof).
    pytest.importorskip("sklearn")
    for name in ("sklearn_ridge.py", "sklearn_nmf.py", "sklearn_normalize.py"):
        result = run_mypy(FIXTURES / name, flags=("--allow-redefinition",))
        assert result.returncode == 0, result.stdout
    result = run_mypy(FIXTURES / "sklearn_gnb.py", flags=("--allow-redefinition",))
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 1, result.stdout
    assert "tuple[n_samples, ?U" in errors[0] and "[return-value]" in errors[0], result.stdout


def test_any_dim_arithmetic_keeps_rank() -> None:
    # Issue 249: an `Any` dim (tuple[Any]) is a known rank. A binary
    # operation keeps the rank and the dtype; an axis that is not provable
    # is a fresh unknown, as with a plain `int` dim.
    result = run_mypy(FIXTURES / "any_dim_arith_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    f64 = "numpy.dtype[numpy.float64]"
    for line in (17, 18, 19, 29):
        assert notes[line] == f'"numpy.ndarray[tuple[?U@{line}], {f64}]"', result.stdout
    for line in (20, 21):
        assert notes[line] == f'"numpy.ndarray[tuple[Literal[3], ?U@{line}], {f64}]"', result.stdout
    assert notes[22] == '"numpy.ndarray[tuple[?U@22], numpy.dtype[numpy.bool[bool]]]"', result.stdout
    assert notes[23] == f'"numpy.ndarray[tuple[Any], {f64}]"', result.stdout
    assert notes[28] == f'"numpy.ndarray[tuple[Any], {f64}]"', result.stdout
    # The result of the call keeps its rank in the next operations.
    assert notes[30] == f'"numpy.ndarray[tuple[?U@30c], {f64}]"', result.stdout


def test_object_array_element_is_any() -> None:
    # Issue 250: an element of an object array is the Python object that
    # the array holds. Its type is Any, as in the numpy stubs.
    result = run_mypy(FIXTURES / "object_element_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    for line in (13, 16, 17, 19, 20):
        assert notes[line] == '"Any"', result.stdout
    assert notes[18] == '"numpy.ndarray[tuple[N], numpy.dtype[numpy.object_]]"', result.stdout
    assert notes[21] == '"numpy.float64"', result.stdout


def test_stack_error_names_the_provable_axis() -> None:
    # Issue 251: (?, 3) with (?, 4) differs at axis 1. Two dynamic dims
    # are not a provable difference, and alone they give no error.
    result = run_mypy(FIXTURES / "bad_stack_int_leading.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 1, result.stdout
    assert ":7:" in errors[0], result.stdout
    # Issue 266: without the stack axis, the form of the concatenate message.
    assert "stack dimension 1 differs: 3 vs 4 (shapes (?, 3) vs (?, 4))  [numty-shape]" in errors[0], (
        result.stdout
    )
    notes = reveals_by_line(result)
    assert notes[8].startswith('"numpy.ndarray[tuple[Literal[2], ?U@8, Literal[3]]'), result.stdout


def test_union_operator_broadcast_mismatch() -> None:
    # Issue 252: with an array union as the right operand, mypy calls the
    # reflected method on the union and no ndarray hook runs. The mismatch
    # is reported for each operator, once per line.
    result = run_mypy(FIXTURES / "bad_union_operator_broadcast.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    by_line: dict[int, list[str]] = {}
    for text in errors:
        by_line.setdefault(int(text.split(":")[1]), []).append(text)
    assert sorted(by_line) == [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 29], result.stdout
    for line in range(16, 26):
        assert len(by_line[line]) == 1, result.stdout
        assert "[numty-broadcast]" in by_line[line][0], result.stdout
        assert "N-1 vs N (shapes (N-1,) vs (N,))" in by_line[line][0], result.stdout
    assert "true_divide shapes disagree" in by_line[21][0], result.stdout
    # Issue 265: one error for the line, for the first member that fits none.
    assert len(by_line[29]) == 1, result.stdout
    assert "3 vs 5" in by_line[29][0], result.stdout


def test_union_operator_no_false_error() -> None:
    # Issue 252: a left operand that fits one member of the union is not
    # an error (the same union on both sides).
    result = run_mypy(FIXTURES / "union_operator_ok.py")
    assert result.returncode == 0, result.stdout


def test_union_eq_ne_broadcast_mismatch() -> None:
    # Issue 255: for `==` and `!=` mypy names the reflected call on a union
    # "__eq__ of _ArrayOrScalarCommon", the class that defines the method.
    result = run_mypy(FIXTURES / "bad_union_eq_broadcast.py")
    assert result.returncode == 1, result.stdout
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 5, result.stdout
    for line, text in zip((11, 12, 13, 14, 15), errors):
        assert f":{line}:" in text and "[numty-broadcast]" in text, result.stdout


def test_union_operator_result_is_solved() -> None:
    # Issue 256: a valid operator with an array union as the right operand
    # has the union of the solved results, not the stub answer.
    result = run_mypy(FIXTURES / "union_operator_result_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    nb = "numpy._typing._nbit_base."
    f64 = "numpy.ndarray[tuple[N], numpy.dtype[numpy.float64]]"
    f32 = f"numpy.ndarray[tuple[N], numpy.dtype[numpy.floating[{nb}_32Bit]]]"
    boolean = '"numpy.ndarray[tuple[N], numpy.dtype[numpy.bool[bool]]]"'
    for line in (15, 16, 17, 18, 19):
        assert notes[line] == f'"{f64} | {f32}"', result.stdout
    for line in (20, 21, 22):
        assert notes[line] == boolean, result.stdout
    for line in (23, 24, 25):
        assert notes[line] == f'"{f64}"', result.stdout
    assert notes[26] == (
        f'"numpy.ndarray[tuple[N], numpy.dtype[numpy.signedinteger[{nb}_64Bit]]]'
        f' | numpy.ndarray[tuple[N], numpy.dtype[numpy.signedinteger[{nb}_32Bit]]]"'
    ), result.stdout
    assert notes[27] == f'"{f64} | {f32}"'.replace("tuple[N]", "tuple[N-1]"), result.stdout
    # Only the members that fit: (3,) with (4,) is not in the result.
    assert notes[31] == (
        '"numpy.ndarray[tuple[Literal[3]], numpy.dtype[numpy.float64]]'
        ' | numpy.ndarray[tuple[Literal[4]], numpy.dtype[numpy.float64]]"'
    ), result.stdout


def test_any_dim_keeps_rank_in_ufuncs() -> None:
    # Issue 257: true division, np.where and the ufuncs read a shape with
    # an `Any` dim, as `+ - *` do since issue 249.
    result = run_mypy(FIXTURES / "any_dim_ufunc_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    f64 = "numpy.dtype[numpy.float64]"
    for line in (12, 14, 15, 16, 21):
        assert notes[line] == f'"numpy.ndarray[tuple[?U@{line}], {f64}]"', result.stdout
    for line in (13, 17, 18, 19, 20):
        assert notes[line] == f'"numpy.ndarray[tuple[Any], {f64}]"', result.stdout
    assert notes[22] == '"numpy.ndarray[tuple[?U@22], numpy.dtype[numpy.bool[bool]]]"', result.stdout


def test_any_dim_keeps_rank_outside_broadcast() -> None:
    # Issue 259: np.take, the items of np.stack and np.concatenate, and the
    # reductions read a shape with an `Any` dim, as with a plain `int` dim.
    result = run_mypy(FIXTURES / "any_dim_readers_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    f64 = "numpy.dtype[numpy.float64]"
    intp = "numpy.dtype[numpy.signedinteger[numpy._typing._nbit_base._32Bit | numpy._typing._nbit_base._64Bit]]"
    assert notes[14] == f'"numpy.ndarray[tuple[Literal[2]], {f64}]"', result.stdout
    assert notes[15] == f'"numpy.ndarray[tuple[Any], {f64}]"', result.stdout
    for line in (16, 17, 25):
        assert notes[line] == f'"numpy.ndarray[tuple[Literal[2], ?U@{line}], {f64}]"', result.stdout
    for line in (18, 19):
        assert notes[line] == f'"numpy.ndarray[tuple[?U@{line}], {f64}]"', result.stdout
    assert notes[20] == f'"numpy.ndarray[tuple[Any], {intp}]"', result.stdout
    assert notes[21] == f'"numpy.ndarray[tuple[Any, Any], {f64}]"', result.stdout
    assert notes[22] == f'"numpy.ndarray[tuple[Any], {intp}]"', result.stdout
    assert notes[23] == f'"numpy.ndarray[tuple[Any], {f64}]"', result.stdout
    assert notes[24] == f'"numpy.ndarray[tuple[Any], {f64}]"', result.stdout
    assert notes[26] == f'"numpy.ndarray[tuple[int], {intp}]"', result.stdout


def _errors_by_line(result: "subprocess.CompletedProcess[str]") -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for text in result.stdout.splitlines():
        if ": error:" in text:
            out.setdefault(int(text.split(":")[1]), []).append(text)
    return out


def test_union_matmul() -> None:
    # Issue 261: `@` with an array union as an operand is checked by the
    # matmul rule and solved, with one error and no [operator] error.
    result = run_mypy(FIXTURES / "bad_union_matmul.py")
    assert result.returncode == 1, result.stdout
    errors = _errors_by_line(result)
    assert sorted(errors) == [12, 13, 17], result.stdout
    for line, pair in ((12, "N-1 vs N"), (13, "N vs N-1"), (17, "N vs N-1")):
        assert len(errors[line]) == 1, result.stdout
        assert f"matmul inner dimensions disagree: {pair}  [numty-shape]" in errors[line][0], result.stdout
    notes = reveals_by_line(result)
    assert notes[14] == '"numpy.float64"', result.stdout
    assert notes[15] == '"numpy.ndarray[tuple[M], numpy.dtype[numpy.float64]]"', result.stdout
    assert notes[16] == '"numpy.ndarray[tuple[M], numpy.dtype[numpy.float64]]"', result.stdout


def test_union_chained_comparison() -> None:
    # Issue 262: each pair of a chained comparison is checked and solved.
    result = run_mypy(FIXTURES / "bad_union_chained_comparison.py")
    assert result.returncode == 1, result.stdout
    errors = _errors_by_line(result)
    assert sorted(errors) == [12, 13, 15], result.stdout
    for line in (12, 13, 15):
        assert len(errors[line]) == 1, result.stdout
        assert "N-1 vs N (shapes (N-1,) vs (N,))  [numty-broadcast]" in errors[line][0], result.stdout
    notes = reveals_by_line(result)
    assert notes[14] == '"numpy.ndarray[tuple[N], numpy.dtype[numpy.bool[bool]]]"', result.stdout


def test_union_inplace() -> None:
    # Issue 263: an in-place operator with an array union on the right is
    # an error when no member fits the target.
    result = run_mypy(FIXTURES / "bad_union_inplace.py")
    assert result.returncode == 1, result.stdout
    errors = _errors_by_line(result)
    assert sorted(errors) == [13, 14, 16, 18], result.stdout
    for line in (13, 14, 16, 18):
        assert len(errors[line]) == 1, result.stdout
        assert "[numty-broadcast]" in errors[line][0], result.stdout


def test_unpack_shape_mismatch_has_one_error() -> None:
    # Issue 264: a mismatch between Unpack[Ts] shapes has no [operator] error.
    result = run_mypy(FIXTURES / "bad_unpack_operator.py")
    assert result.returncode == 1, result.stdout
    errors = _errors_by_line(result)
    assert sorted(errors) == [18, 19, 20, 21], result.stdout
    for line, pair in ((18, "4 vs 3"), (19, "4 vs 3"), (20, "3 vs 4"), (21, "4 vs 3")):
        assert len(errors[line]) == 1, result.stdout
        assert pair in errors[line][0] and "[numty-broadcast]" in errors[line][0], result.stdout


def test_union_operator_one_error_per_line() -> None:
    # Issue 265: one error for the first member of the left operand that
    # fits no member of the right operand.
    result = run_mypy(FIXTURES / "bad_union_one_error.py")
    assert result.returncode == 1, result.stdout
    errors = _errors_by_line(result)
    assert sorted(errors) == [16, 17, 18], result.stdout
    for line, pair in ((16, "3 vs 5"), (17, "4 vs 3"), (18, "3 vs 5")):
        assert len(errors[line]) == 1, result.stdout
        assert f": {pair} (shapes" in errors[line][0], result.stdout


def test_stack_messages_have_no_axis_suffix() -> None:
    # Issue 266: the `axis` argument has no effect on a stack mismatch.
    result = run_mypy(FIXTURES / "bad_stack_axis_suffix.py")
    assert result.returncode == 1, result.stdout
    errors = _errors_by_line(result)
    assert sorted(errors) == [7, 8], result.stdout
    assert "stack dimension 1 differs: 3 vs 4 (shapes (2, 3) vs (2, 4))  [numty-shape]" in errors[7][0], (
        result.stdout
    )
    assert "stack ranks differ: (2, 3), (2, 3, 1)  [numty-shape]" in errors[8][0], result.stdout


def test_integer_timedelta_bitwise_dtypes() -> None:
    # Issue 267: two np.integer operands have no known dtype; timedelta64
    # divides to int64 (//) and float64 (/); a bitwise operator on integer
    # arrays with no common integer type is an error.
    result = run_mypy(FIXTURES / "bad_integer_timedelta_bitwise.py")
    assert result.returncode == 1, result.stdout
    errors = _errors_by_line(result)
    assert sorted(errors) == [19, 20, 21, 22, 23, 24], result.stdout
    names = ("bitwise_and", "bitwise_and", "bitwise_or", "bitwise_xor", "left_shift", "right_shift")
    for line, name in zip((19, 20, 21, 22, 23, 24), names):
        assert len(errors[line]) == 1, result.stdout
        assert f"ufunc '{name}' not supported for int64 with uint64" in errors[line][0], result.stdout
        assert "[numty-broadcast]" in errors[line][0], result.stdout
    notes = reveals_by_line(result)
    nb = "numpy._typing._nbit_base."
    assert notes[15] == '"numpy.ndarray[tuple[N], numpy.dtype[Any]]"', result.stdout
    assert notes[16] == '"numpy.ndarray[tuple[N], numpy.dtype[numpy.float64]]"', result.stdout
    assert notes[17] == f'"numpy.ndarray[tuple[N], numpy.dtype[numpy.signedinteger[{nb}_64Bit]]]"', result.stdout
    assert notes[18] == '"numpy.ndarray[tuple[N], numpy.dtype[numpy.float64]]"', result.stdout
    # A common integer type exists: no error.
    assert notes[25] == f'"numpy.ndarray[tuple[N], numpy.dtype[numpy.signedinteger[{nb}_64Bit]]]"', result.stdout
    assert notes[26] == f'"numpy.ndarray[tuple[N], numpy.dtype[numpy.unsignedinteger[{nb}_64Bit]]]"', result.stdout


def test_arr_head_ok() -> None:
    # Issue #278: the generic head takes the dtype first, then flat dims
    # like a fixed head: builders with dtype=m.dtype and unary ufuncs
    # keep the dtype TypeVar, call sites solve the concrete dtype.
    result = run_mypy(FIXTURES / "arr_head_ok.py")
    assert result.returncode == 0, result.stdout
    f32 = "numpy.dtype[numpy.floating[numpy._typing._nbit_base._32Bit]]"
    assert reveals_by_line(result) == {
        24: '"numpy.ndarray[tuple[E, M], numpy.dtype[T]]"',
        25: '"numpy.ndarray[tuple[E, Literal[2]], numpy.dtype[T]]"',
        31: '"numpy.ndarray[tuple[N, M], numpy.dtype[FT]]"',
        32: '"numpy.ndarray[tuple[N, M], numpy.dtype[FT]]"',
        33: '"numpy.ndarray[tuple[N, M], numpy.dtype[FT]]"',
        46: '"numpy.ndarray[tuple[N], numpy.dtype[Any]]"',
        51: f'"numpy.ndarray[tuple[E, M], {f32}]"',
        52: f'"numpy.ndarray[tuple[N, M], {f32}]"',
        53: f'"numpy.ndarray[tuple[E, M], {f32}]"',
        54: f'"numpy.ndarray[tuple[N, M], {f32}]"',
    }, result.stdout
    assert _run_fixture("arr_head_ok.py") == [
        "(2, 3) float32",
        "(2, 2) float32",
        "(2, 3) float32",
        "(2, 3) float32",
        "(2, 3) float32",
        "(2, 3) float32",
        "(2, 3) float32",
        "(2, 3) float32",
        "(2, 2) float32",
        "(2, 3) float32",
        "(2, 3) float32",
    ]


def test_arr_head_wrong_axis() -> None:
    # Issue #278: a result built on the wrong axis is reported, with the
    # dtype variable on both sides.
    assert_single_error(
        run_mypy(FIXTURES / "bad_arr_head.py"),
        'got "ndarray[tuple[E, N], dtype[T]]", expected "ndarray[tuple[E, M], dtype[T]]"',
        line=14,
    )


def test_arr_head_misuse() -> None:
    # Issue #278: a missing dtype, an int-bound dim first, and a literal
    # first all fail loudly with the dtype-first redirect.
    result = run_mypy(FIXTURES / "bad_arr_dtype.py")
    assert result.returncode == 1, result.stdout
    errors = _errors_by_line(result)
    assert sorted(errors) == [12, 15, 18], result.stdout
    assert len(errors[12]) == 1 and len(errors[15]) == 1 and len(errors[18]) == 1, result.stdout
    assert "Arr takes a dtype and at least one dim" in errors[12][0], result.stdout
    assert "[numty-annotation]" in errors[12][0], result.stdout
    assert 'Arr takes a dtype as its first argument, but "N" is bound to int' in errors[15][0], (
        result.stdout
    )
    assert "[numty-annotation]" in errors[15][0], result.stdout
    assert "Arr takes a dtype as its first argument" in errors[18][0], result.stdout
    assert "[numty-annotation]" in errors[18][0], result.stdout


def test_any_bound_size_is_stable() -> None:
    # Issue 284: with an argument of unknown rank and a union argument,
    # mypy solved `T` by a join whose order changed from run to run, and
    # about one cold run in four reported `cannot be "object"`. Eight
    # cold runs must all be clean. Two at a time: each cold run reads
    # numpy's stubs from scratch, and eight at once next to the batched
    # runs of the other workers ran a CI runner out of memory.
    from concurrent.futures import ThreadPoolExecutor

    fixture = FIXTURES / "callsite_any_bound_ok.py"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run_mypy(fixture, flags=("--cache-dir", os.devnull)), range(8)))
    for result in results:
        assert result.returncode == 0, result.stdout
        nb = "numpy._typing._nbit_base."
        assert reveals_by_line(result)[17] == (
            f'"numpy.ndarray[tuple[Any, Literal[5]], numpy.dtype[numpy.floating[{nb}_32Bit]]]"'
        ), result.stdout


_NB = "numpy._typing._nbit_base."
_I64 = f"numpy.dtype[numpy.signedinteger[{_NB}_64Bit]]"
_F32 = f"numpy.dtype[numpy.floating[{_NB}_32Bit]]"
_F64 = "numpy.dtype[numpy.float64]"
_BOOL = "numpy.dtype[numpy.bool[bool]]"


def test_namedtuple_method_with_unknown_index() -> None:
    # Issue 274: `block.take(fi)` with an untyped index made the item of
    # the returned NamedTuple `(?U, Any)` and left its fallback `Block[P,
    # Any]`, so passing it on was `"Block[P, Any]"; expected "Block[P, Any]"`.
    result = run_mypy(FIXTURES / "generic_namedtuple_unknown_index_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    block = "tests.fixtures.generic_namedtuple_unknown_index_ok.Block"
    assert notes[27] == f'"tuple[numpy.ndarray[tuple[P, Any], {_F32}], fallback={block}[P, Any]]"', result.stdout
    assert notes[29] == f'"tuple[numpy.ndarray[tuple[P, R], {_F32}], fallback={block}[P, R]]"', result.stdout


def test_property_and_method_own_size_is_unknown() -> None:
    # Issue 276: a size that only the return of a property or a method
    # names is unknown at each read. A property gave the caller's size of
    # the same name, a method gave `Never`.
    result = run_mypy(FIXTURES / "property_own_size.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    # Read through a variable, the size is the object's: two properties
    # declared with `E` have one size (issue 306). A method call chooses
    # a size per call (issue 307).
    assert notes[45] == f'"numpy.ndarray[tuple[?c.E], {_I64}]"', result.stdout
    assert notes[46] == f'"numpy.ndarray[tuple[?c.E], {_I64}]"', result.stdout
    assert notes[47] == f'"numpy.ndarray[tuple[?E@47], {_I64}]"', result.stdout
    # `scores: F64[E]` plus the property is not `F64[E]`.
    assert notes[48] == f'"numpy.ndarray[tuple[?U@48], {_F64}]"', result.stdout
    # A size of the class is still the caller's.
    assert notes[49] == f'"numpy.ndarray[tuple[E], {_F64}]"', result.stdout
    assert notes[50] == (
        f'"tuple[numpy.ndarray[tuple[E], {_F64}], numpy.ndarray[tuple[?b.E], {_I64}]]"'
    ), result.stdout
    assert notes[51] == f'"numpy.ndarray[tuple[E], {_F64}]"', result.stdout
    assert notes[52] == f'"numpy.ndarray[tuple[E], {_F64}]"', result.stdout


def test_int_dims_keep_shape_in_bitwise_and_searchsorted() -> None:
    # Issues 270 and 275: `&` and `|` read a shape with an `int` size,
    # as a comparison does, and `searchsorted` has the shape of its
    # values. Both gave the stubs' rank-less answer, and for the results
    # of a generic call `ndarray[Any, Any]`, which hid a wrong dtype.
    result = run_mypy(FIXTURES / "bad_int_dims_bitwise.py")
    assert result.returncode == 1, result.stdout
    errors = _errors_by_line(result)
    assert sorted(errors) == [27, 34], result.stdout
    assert "N-1 vs N (shapes (?, N-1) vs (?, N))  [numty-broadcast]" in errors[27][0], result.stdout
    assert "[return-value]" in errors[34][0] and "numpy.bool" in errors[34][0], result.stdout
    notes = reveals_by_line(result)
    assert notes[17] == f'"numpy.ndarray[tuple[?U@17], {_BOOL}]"', result.stdout
    assert notes[18] == f'"numpy.ndarray[tuple[?U@18, N], {_BOOL}]"', result.stdout
    assert notes[19] == f'"numpy.ndarray[tuple[int, N], {_BOOL}]"', result.stdout
    assert notes[20] == f'"numpy.ndarray[tuple[int, N], {_BOOL}]"', result.stdout
    assert notes[21] == f'"numpy.ndarray[tuple[?U@21], {_I64}]"', result.stdout
    assert notes[22] == f'"numpy.ndarray[tuple[int], {_I64}]"', result.stdout
    assert notes[23] == f'"numpy.ndarray[tuple[int], {_I64}]"', result.stdout
    assert notes[24] == f'"numpy.ndarray[tuple[N], {_I64}]"', result.stdout
    assert notes[25] == f'"numpy.ndarray[tuple[int], {_I64}]"', result.stdout
    assert notes[26] == f'"numpy.ndarray[tuple[N], {_I64}]"', result.stdout
    assert notes[33] == f'"numpy.ndarray[tuple[?U@33, N], {_BOOL}]"', result.stdout


def test_setitem_symbolic_sizes() -> None:
    # Issue 290: an assignment compared sizes only when both were
    # literal, so `out[i] = v[:-1]` was silent and `out[i] + v[:-1]` was
    # an error. Two names, an unknown and an `int` stay silent.
    result = run_mypy(FIXTURES / "bad_setitem_symbolic.py")
    assert result.returncode == 1, result.stdout
    errors = _errors_by_line(result)
    assert sorted(errors) == [13, 14, 15, 16, 18, 19], result.stdout
    expected = {
        13: "cannot assign value of shape (C-1,) into target of shape (C,)",
        14: "broadcast shapes disagree at trailing axis 0: C vs C-1",
        15: "cannot assign value of shape (S-1,) into target of shape (S,)",
        16: "cannot assign value of shape (S, C) into target of shape (S-1, C)",
        18: "cannot assign value of shape (S, C-2) into target of shape (S, C-1)",
        19: "cannot assign value of shape (S-2, C) into target of shape (S-1, C)",
    }
    for line, message in expected.items():
        assert len(errors[line]) == 1, result.stdout
        assert message in errors[line][0] and "[numty-broadcast]" in errors[line][0], result.stdout


def test_container_int_sizes_and_bare_container() -> None:
    # Issue 279: at a call, `int` in a size position of a container of
    # arrays accepts a named or a literal size (NamedTuple, dataclass,
    # Optional, inside a list, a method); an array argument with an
    # `int` size does not stop the container from binding the name; a
    # bare generic NamedTuple of arrays is reported.
    result = run_mypy(FIXTURES / "bad_container_int_sizes.py")
    assert result.returncode == 1, result.stdout
    errors = _errors_by_line(result)
    assert sorted(errors) == [53, 64, 66, 67], result.stdout
    assert "Box used bare" in errors[53][0] and "[numty-annotation]" in errors[53][0], result.stdout
    assert '"Data[F, str]"; expected "Data[Any, int]"' in errors[64][0], result.stdout
    assert '"Box[Literal[1]]"; expected "Box[Literal[3]]"' in errors[66][0], result.stdout
    assert '"Box[F]"; expected "Box[Literal[3]]"' in errors[67][0], result.stdout
    notes = reveals_by_line(result)
    assert notes[60] == f'"numpy.ndarray[tuple[int], {_I64}]"', result.stdout
    assert notes[69] == f'"numpy.ndarray[tuple[F], {_F64}]"', result.stdout
    assert notes[70] == f'"numpy.ndarray[tuple[F], {_F64}]"', result.stdout


_ND = "numpy.ndarray"


def test_coverage_batch() -> None:
    # Issue 283: np.array of a list built at runtime or of arrays,
    # broadcast_to with runtime sizes, split at an index array, the
    # dtype of max/min ufunc methods, the dtype a builder is given when
    # every size is a runtime int, and np.histogram.
    result = run_mypy(FIXTURES / "coverage_batch_ok.py")
    assert result.returncode == 0, result.stdout
    assert reveals_by_line(result) == {
        13: f'"{_ND}[tuple[?U@13], {_BOOL}]"',
        14: f'"{_ND}[tuple[?U@14], {_BOOL}]"',
        15: f'"{_ND}[tuple[?U@15], numpy.dtype[numpy.signedinteger[{_NB}_32Bit]]]"',
        16: f'"{_ND}[tuple[Literal[2], N], {_F64}]"',
        17: f'"{_ND}[tuple[N, Literal[2]], {_F64}]"',
        18: f'"{_ND}[tuple[?U@18, N], {_F64}]"',
        19: f'"{_ND}[tuple[int, N], {_F64}]"',
        20: f'"{_ND}[tuple[int, int], {_F64}]"',
        21: f'"list[{_ND}[tuple[?U@21b], {_I64}]]"',
        22: f'"list[{_ND}[tuple[?U@22], {_I64}]]"',
        # one entry per True of `alive`: the count minted on line 21
        23: f'"{_ND}[tuple[?U@21], {_F32}]"',
        24: f'"{_ND}[tuple[N], {_F32}]"',
        25: f'"{_ND}[tuple[N], {_F64}]"',
        26: f'"{_ND}[tuple[N], {_F32}]"',
        27: f'"{_ND}[tuple[int, int], {_BOOL}]"',
        28: f'"{_ND}[tuple[int, int], {_I64}]"',
        29: f'"{_ND}[tuple[int, int], {_F32}]"',
        30: f'"{_ND}[tuple[int, int], {_F64}]"',
        31: f'"{_ND}[tuple[int, int], numpy.dtype[Any]]"',
        35: f'"tuple[{_ND}[tuple[Literal[15]], {_I64}], {_ND}[tuple[Literal[16]], {_F64}]]"',
        36: f'"tuple[{_ND}[tuple[Literal[10]], {_I64}], {_ND}[tuple[Literal[11]], {_F64}]]"',
        37: f'"tuple[{_ND}[tuple[Literal[4]], {_I64}], {_ND}[tuple[Literal[5]], {_F32}]]"',
        38: f'"tuple[{_ND}[tuple[B-1], {_I64}], {_ND}[tuple[B], {_F64}]]"',
        39: f'"tuple[{_ND}[tuple[?U@39], {_I64}], {_ND}[tuple[?U@39b], {_F64}]]"',
        40: f'"tuple[{_ND}[tuple[Literal[3]], numpy.dtype[Any]], {_ND}[tuple[Literal[4]], {_F64}]]"',
        42: f'"{_ND}[tuple[B-1], {_F64}]"',
    }, result.stdout
    assert _run_fixture("coverage_batch_ok.py") == [
        "(15,) int64 (16,) float64",
        "(4,) int64 (5,) float32",
        "bool (7, 2)",
        "(2, 7) float32",
        "bool float64",
        "[(2,), (3,), (2,)]",
    ]


def test_dtype_generic_helpers() -> None:
    # Issue 278:  and ufuncs keep a dtype TypeVar, so the
    # body of a helper generic in its dtype is checked: a result built
    # on the wrong axis is reported.
    result = run_mypy(FIXTURES / "bad_dtype_generic.py")
    assert result.returncode == 1, result.stdout
    errors = _errors_by_line(result)
    assert sorted(errors) == [30], result.stdout
    assert 'got "ndarray[tuple[E, N], dtype[T]]", expected "ndarray[tuple[E, M], dtype[T]]"' in errors[30][0], (
        result.stdout
    )
    assert reveals_by_line(result) == {
        16: f'"{_ND}[tuple[E, M], numpy.dtype[T]]"',
        17: f'"{_ND}[tuple[E, Literal[2]], numpy.dtype[T]]"',
        23: f'"{_ND}[tuple[N, M], numpy.dtype[FT]]"',
        24: f'"{_ND}[tuple[N, M], numpy.dtype[FT]]"',
        25: f'"{_ND}[tuple[N, M], numpy.dtype[FT]]"',
        34: f'"{_ND}[tuple[E, M], {_F32}]"',
        35: f'"{_ND}[tuple[N, M], {_F32}]"',
        36: f'"{_ND}[tuple[E, M], {_F32}]"',
    }, result.stdout


def test_family_dtype_heads() -> None:
    # Issue 277: Floating / Integer heads, elementwise ops that keep the
    # shape and the family, a numpy scalar of unknown width as an
    # operand, max() of such a scalar and a float, and the heads U16,
    # U32, U64 and Intp.
    result = run_mypy(FIXTURES / "bad_family_dtype.py")
    assert result.returncode == 1, result.stdout
    errors = _errors_by_line(result)
    assert sorted(errors) == [48, 51, 55], result.stdout
    assert "dtype[integer[Any]]" in errors[48][0] and "[arg-type]" in errors[48][0], result.stdout
    assert "dtype[floating[Any]]" in errors[51][0] and "[arg-type]" in errors[51][0], result.stdout
    assert "N-1 vs N (shapes (N-1,) vs (N,))  [numty-broadcast]" in errors[55][0], result.stdout
    assert reveals_by_line(result) == {
        28: f'"{_ND}[tuple[N], numpy.dtype[numpy.floating[Any]]]"',
        29: f'"{_ND}[tuple[N], numpy.dtype[numpy.integer[Any]]]"',
        30: f'"{_ND}[tuple[N], numpy.dtype[numpy.unsignedinteger[{_NB}_32Bit]]]"',
        31: f'"{_ND}[tuple[N, M], numpy.dtype[numpy.unsignedinteger[{_NB}_16Bit]]]"',
        32: f'"{_ND}[tuple[N], numpy.dtype[numpy.unsignedinteger[{_NB}_64Bit]]]"',
        33: f'"{_ND}[tuple[N], numpy.dtype[numpy.signedinteger[{_NB}_32Bit | {_NB}_64Bit]]]"',
        34: f'"{_ND}[tuple[N], numpy.dtype[numpy.floating[Any]]]"',
        35: f'"{_ND}[tuple[N], numpy.dtype[numpy.floating[Any]]]"',
        36: f'"{_ND}[tuple[N], numpy.dtype[numpy.floating[Any]]]"',
        37: f'"{_ND}[tuple[N], numpy.dtype[numpy.floating[Any]]]"',
        38: f'"{_ND}[tuple[int, N], numpy.dtype[numpy.floating[Any]]]"',
        39: f'"{_ND}[tuple[N], numpy.dtype[numpy.floating[Any]]]"',
        40: '"numpy.floating[Any] | float"',
        41: f'"{_ND}[tuple[N], numpy.dtype[Any]]"',
        42: f'"{_ND}[tuple[N], numpy.dtype[numpy.integer[Any]]]"',
        43: f'"{_ND}[tuple[N], numpy.dtype[Any]]"',
        44: f'"{_ND}[tuple[N], numpy.dtype[numpy.unsignedinteger[{_NB}_32Bit]]]"',
        45: f'"{_ND}[tuple[N], numpy.dtype[numpy.floating[Any]]]"',
    }, result.stdout


def test_literal_formula_spelling() -> None:
    # Issue 282: `U8[Literal["N+1"]]` is the string spec `U8["N+1"]` in a
    # spelling that ruff's UP037 does not rewrite.
    result = run_mypy(FIXTURES / "bad_literal_formula.py")
    assert result.returncode == 1, result.stdout
    errors = _errors_by_line(result)
    assert sorted(errors) == [25], result.stdout
    assert "tuple[N+1, M]" in errors[25][0] and "tuple[N+1, M+1]" in errors[25][0], result.stdout
    u8 = f"numpy.dtype[numpy.unsignedinteger[{_NB}_8Bit]]"
    assert reveals_by_line(result) == {
        29: f'"numpy.ndarray[tuple[N+1, M+1], {u8}]"',
        30: f'"numpy.ndarray[tuple[Literal[4], Literal[5]], {u8}]"',
        31: f'"numpy.ndarray[tuple[N+1, M+1], {u8}]"',
    }, result.stdout


def test_polars_to_numpy() -> None:
    # Issue 272: DataFrame.to_numpy() is 2-D; Series.to_numpy() is 1-D

    # after a visible cast to a numeric or boolean dtype, and a float
    # cast names the dtype. A plain Series.to_numpy(), a cast to Array
    # or String, and structured=True keep polars' own answer.
    pytest.importorskip("polars")
    result = run_mypy(FIXTURES / "polars_to_numpy.py")
    assert result.returncode == 1, result.stdout
    errors = _errors_by_line(result)
    assert sorted(errors) == [21], result.stdout
    assert any("too many indices for 1 dimensions" in line and "[numty-axis]" in line for line in errors[21]), (
        result.stdout
    )
    polars_answer = '"numpy.ndarray[Any, Any]"'
    assert reveals_by_line(result) == {
        7: polars_answer,
        8: f'"{_ND}[tuple[int], {_F64}]"',
        9: f'"{_ND}[tuple[int], {_F32}]"',
        10: f'"{_ND}[tuple[int], numpy.dtype[Any]]"',
        11: f'"{_ND}[tuple[int], {_F64}]"',
        12: polars_answer,
        13: polars_answer,
        14: f'"{_ND}[tuple[int, int], numpy.dtype[Any]]"',
        15: f'"{_ND}[tuple[int, int], numpy.dtype[Any]]"',
        16: polars_answer,
        17: f'"{_ND}[tuple[int, int], numpy.dtype[Any]]"',
        19: f'"{_ND}[tuple[?U@19c], {_F64}]"',
        20: f'"{_ND}[tuple[int, int], numpy.dtype[Any]]"',
    }, result.stdout
    assert _run_fixture("polars_to_numpy.py") == [
        "(3,) float64",
        "float32 float64",
        "(3, 2) (3,)",
        "(3, 2)",
    ]
def test_mask_count_reuses_one_unknown_per_variable() -> None:
    # Issue #288 (1): the same mask variable gives the same unknown, so
    # ids[keep] and d[0, keep] share ?U and their sum stays clean. Index
    # arrays keep reusing their named size (K). Different masks, a
    # rebinding, different scopes and inline masks all mint fresh.
    result = run_mypy(FIXTURES / "mask_count_ok.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    i64 = "numpy.dtype[numpy.signedinteger[numpy._typing._nbit_base._64Bit]]"
    f64 = "numpy.dtype[numpy.float64]"
    assert notes[11] == f'"numpy.ndarray[tuple[K], {i64}]"', result.stdout
    assert notes[12] == f'"numpy.ndarray[tuple[P, K], {f64}]"', result.stdout
    assert notes[13] == f'"numpy.ndarray[tuple[K], {i64}]"', result.stdout
    # Line 14 pairs (N,) against (K,): silent by default.
    assert notes[14] == f'"numpy.ndarray[tuple[?U@14], {f64}]"', result.stdout
    # Lines 15-16 share one count for `keep`; line 17 keeps it.
    assert notes[15] == f'"numpy.ndarray[tuple[?U@15], {i64}]"', result.stdout
    assert notes[16] == f'"numpy.ndarray[tuple[?U@15], {f64}]"', result.stdout
    assert notes[17] == f'"numpy.ndarray[tuple[?U@15], {f64}]"', result.stdout
    assert _run_fixture("mask_count_ok.py") == ["ok"]
    # Under the opt-in only line 14 reports (N vs K); line 17 stays clean.
    on = run_mypy(FIXTURES / "mask_count_ok.py", flags=("--enable-error-code", "numty-names"))
    errors = _errors_by_line(on)
    assert sorted(errors) == [14], on.stdout
    assert "N and K are different names" in errors[14][0], on.stdout
    assert errors[14][0].endswith("[numty-names]"), on.stdout


def test_mask_count_optin_reports_selection_counts() -> None:
    # Issue #288 (2): under the opt-in `numty-names` code a selection
    # count is a size of its own. A mask count against a named size, two
    # different masks' counts, and a call-site binding of a count against
    # a name are errors; the same mask variable stays clean. By default
    # all of it is silent (unknown is not an error).
    fixture = FIXTURES / "bad_mask_count_optin.py"
    off = run_mypy(fixture)
    assert off.returncode == 0, off.stdout
    assert "error:" not in off.stdout
    on = run_mypy(fixture, flags=("--enable-error-code", "numty-names"))
    errors = _errors_by_line(on)
    assert sorted(errors) == [28, 29, 31, 32], on.stdout
    assert "N and ?U@28 are different names" in errors[28][0], on.stdout
    assert "?U@28 and ?U@29 are different names" in errors[29][0], on.stdout
    assert "N and P are different names" in errors[31][0], on.stdout
    assert "?U@28 and N are different names" in errors[32][0], on.stdout
    for line in (28, 29, 31, 32):
        assert len(errors[line]) == 1, on.stdout
        assert errors[line][0].endswith("[numty-names]"), on.stdout


def test_own_size_return() -> None:
    # Issues 306/307: a name of the return alone is the function's to
    # choose. Its positions in a `return` hold one size (a literal, a
    # selection count, a local that takes the name), a local annotated
    # with it names a size born in the body, and a call gives one
    # unknown per name, shared by the positions.
    result = run_mypy(FIXTURES / "own_size_return.py")
    assert result.returncode == 0, result.stdout
    assert "error:" not in result.stdout
    notes = reveals_by_line(result)
    assert notes[27] == f'"numpy.ndarray[tuple[K], {_I64}]"', result.stdout
    assert notes[28] == f'"numpy.ndarray[tuple[K], {_I64}]"', result.stdout
    assert notes[52] == f'"numpy.ndarray[tuple[O, E], {_F64}]"', result.stdout
    assert notes[68] == f'"numpy.ndarray[tuple[?K@67], {_I64}]"', result.stdout
    assert notes[69] == f'"numpy.ndarray[tuple[?K@67], {_I64}]"', result.stdout
    assert notes[71] == f'"numpy.ndarray[tuple[?K@70], {_F64}]"', result.stdout
    assert notes[72] == f'"numpy.ndarray[tuple[?J@70], {_F64}]"', result.stdout
    assert notes[73] == f'"numpy.ndarray[tuple[?K@73], {_F64}]"', result.stdout
    assert _run_fixture("own_size_return.py") == ["(3,) (3,)", "(3,) (3,)", "(2, 3)", "(3, 2)"]
    # Nothing in it is two sizes that meet, so the opt-in adds no error.
    on = run_mypy(FIXTURES / "own_size_return.py", flags=("--enable-error-code", "numty-names"))
    assert "error:" not in on.stdout, on.stdout


def test_bad_own_size_return() -> None:
    # Issue 307: positions declared with one free name that hold two
    # sizes are refused by mypy, with the first position under the name
    # and the other under its own size. A name a parameter binds still
    # refuses a selection count. numpy: the ends are (4,) for 3 starts.
    result = run_mypy(FIXTURES / "bad_own_size_return.py")
    errors = _errors_by_line(result)
    assert sorted(errors) == [14, 16, 20, 25, 29], result.stdout
    for line in errors:
        assert len(errors[line]) == 1 and errors[line][0].endswith("[return-value]"), result.stdout
    assert "tuple[K]" in errors[14][0] and "tuple[Literal[1]]" in errors[14][0], result.stdout
    assert "tuple[K+1]" in errors[16][0], result.stdout
    assert "tuple[?U@20b]" in errors[20][0], result.stdout
    # A local that is returned takes the name where it is assigned, so
    # here the item of the `return` is the one that keeps its own size.
    assert 'got "tuple[ndarray[tuple[?U@25]' in errors[25][0], result.stdout
    assert "tuple[?U@29]" in errors[29][0] and '"ndarray[tuple[N]' in errors[29][0], result.stdout
    assert _run_fixture("bad_own_size_return.py") == ["(3,) (4,)"]


def test_bad_own_size_optin() -> None:
    # Issues 306/307 (opt-in): the size a call chose, and the size an
    # object's properties name, are sizes of their own. Silent by default.
    fixture = FIXTURES / "bad_own_size_optin.py"
    off = run_mypy(fixture)
    assert off.returncode == 0, off.stdout
    assert "error:" not in off.stdout
    assert reveals_by_line(off)[74] == f'"numpy.ndarray[tuple[?ctx.E, T], {_F64}]"', off.stdout
    on = run_mypy(fixture, flags=("--enable-error-code", "numty-names"))
    errors = _errors_by_line(on)
    assert sorted(errors) == [58, 65, 67, 75, 81], on.stdout
    # A local that took a free name is that count: only another count reports.
    assert "K and ?U@58 are different names" in errors[58][0], on.stdout
    assert "N and ?K@63 are different names" in errors[65][0], on.stdout
    assert "?K@66 and ?J@66 are different names" in errors[67][0], on.stdout
    assert "?ctx.E and T are different names" in errors[75][0], on.stdout
    assert "?a.E and ?b.E are different names" in errors[81][0], on.stdout
    for line in errors:
        assert len(errors[line]) == 1, on.stdout
        assert errors[line][0].endswith("[numty-names]"), on.stdout
    assert _run_fixture("bad_own_size_optin.py") == ["(3, 4)"]


def test_local_name_and_flatnonzero_count() -> None:
    # An `int` size fits a declared name at an annotated local, as at a
    # `return`, and `np.flatnonzero(mask)` has the count of the mask
    # variable. Silent by default; the opt-in reports the two mistakes.
    fixture = FIXTURES / "local_name_flatnonzero.py"
    off = run_mypy(fixture)
    assert off.returncode == 0, off.stdout
    notes = reveals_by_line(off)
    assert notes[20] == f'"numpy.ndarray[tuple[S, C], {_F64}]"', off.stdout
    assert notes[33] == f'"numpy.ndarray[tuple[?U@30], {_I64}]"', off.stdout
    assert notes[34] == f'"numpy.ndarray[tuple[?U@30], {_F64}]"', off.stdout
    assert notes[35] == f'"numpy.ndarray[tuple[?U@30], {_F64}]"', off.stdout
    on = run_mypy(fixture, flags=("--enable-error-code", "numty-names"))
    errors = _errors_by_line(on)
    assert sorted(errors) == [23, 38], on.stdout
    assert "?U@21 and C are different names" in errors[23][0], on.stdout
    assert "?U@30 and F are different names" in errors[38][0], on.stdout
    assert _run_fixture("local_name_flatnonzero.py") == ["(2, 3)", "(5,)"]


def test_bad_local_name() -> None:
    # The annotation of a local takes an `int` size only: a literal, a
    # selection count, a transposed value and an axis declared with a
    # literal are still refused.
    result = run_mypy(FIXTURES / "bad_local_name.py")
    errors = _errors_by_line(result)
    assert sorted(errors) == [14, 15, 16, 17], result.stdout
    for line in errors:
        assert len(errors[line]) == 1 and errors[line][0].endswith("[assignment]"), result.stdout


def test_array_any_items() -> None:
    # `np.array` of rows or of a display with `Any` items next to a
    # certain scalar: one value per item, or numpy refuses the ragged
    # display. An item indexing every axis with an integer is a scalar.
    fixture = FIXTURES / "array_any_items.py"
    off = run_mypy(fixture)
    assert off.returncode == 0, off.stdout
    notes = reveals_by_line(off)
    assert notes[25] == f'"numpy.ndarray[tuple[?U@22, Literal[6]], {_F64}]"', off.stdout
    assert notes[26] == f'"numpy.ndarray[tuple[?U@22-1], {_F64}]"', off.stdout
    assert notes[39] == f'"numpy.ndarray[tuple[Literal[2]], {_F64}]"', off.stdout
    assert notes[40] == '"numpy.ndarray[tuple[Literal[2]], numpy.dtype[Any]]"', off.stdout
    # A row of the grid is an array, and a dict is not a scalar: no claim.
    assert notes[41] == '"numpy.ndarray[tuple[Literal[2], T], numpy.dtype[Any]]"', off.stdout
    assert "tuple[Any, ...]" in notes[42], off.stdout
    on = run_mypy(fixture, flags=("--enable-error-code", "numty-names"))
    errors = _errors_by_line(on)
    assert sorted(errors) == [27], on.stdout
    assert "?U@22 and ?U@22-1 are 1 apart" in errors[27][0], on.stdout
    assert errors[27][0].endswith("[numty-names]"), on.stdout
    assert _run_fixture("array_any_items.py") == ["(3, 4)", "(3,)", "ragged"]


def test_repeat_counts() -> None:
    # Issue 289: `np.repeat(x, counts)` is `(int,)`, which a declared
    # name takes at an annotated local and at a `return`; the per-run
    # array assigned where the per-entry one is declared is refused.
    result = run_mypy(FIXTURES / "repeat_counts.py")
    notes = reveals_by_line(result)
    assert notes[23] == f'"numpy.ndarray[tuple[int], {_F64}]"', result.stdout
    assert notes[24] == f'"numpy.ndarray[tuple[int], {_F64}]"', result.stdout
    assert notes[25] == f'"numpy.ndarray[tuple[2*?U@18], {_F64}]"', result.stdout
    errors = _errors_by_line(result)
    assert sorted(errors) == [26], result.stdout
    assert "tuple[?U@18]" in errors[26][0] and errors[26][0].endswith("[assignment]"), result.stdout
    assert _run_fixture("repeat_counts.py") == ["[False, True, False, True, False] (5,)"]


def test_symbolic_slice() -> None:
    # A slice bound that is a formula over `int` variables gives a size
    # over them (`N-?k+1`). Nothing is proved from it by default; the
    # opt-in reports the window written one entry late.
    fixture = FIXTURES / "symbolic_slice.py"
    off = run_mypy(fixture)
    assert off.returncode == 0, off.stdout
    assert "error:" not in off.stdout
    notes = reveals_by_line(off)
    assert notes[24] == f'"numpy.ndarray[tuple[-?k+N+1], {_F64}]"', off.stdout
    assert notes[25] == f'"numpy.ndarray[tuple[-?k+N+1], {_F64}]"', off.stdout
    assert notes[26] == f'"numpy.ndarray[tuple[-?k+N], {_F64}]"', off.stdout
    assert notes[27] == f'"numpy.ndarray[tuple[?k-1], {_F64}]"', off.stdout
    # No `int` variable is left in it: the bounds may be clipped, so nothing is claimed.
    assert notes[28] == f'"numpy.ndarray[tuple[?U@28], {_F64}]"', off.stdout
    assert notes[40] == f'"numpy.ndarray[tuple[-?k+N+1], {_F64}]"', off.stdout
    on = run_mypy(fixture, flags=("--enable-error-code", "numty-names"))
    errors = _errors_by_line(on)
    assert sorted(errors) == [29, 41], on.stdout
    assert "-?k+N+1 and -?k+N are 1 apart" in errors[29][0], on.stdout
    assert "-?k+N+1 and -?k+N are 1 apart" in errors[41][0], on.stdout
    assert _run_fixture("symbolic_slice.py") == ["(4,)"]


def test_join_int_items() -> None:
    # Issue 316: items with an `int` size have a shape; they bailed
    # with "the items have no known shape".
    result = run_mypy(FIXTURES / "join_int_items.py")
    assert result.returncode == 0, result.stdout
    notes = reveals_by_line(result)
    assert notes[11] == f'"numpy.ndarray[tuple[N, ?U@11], {_F64}]"', result.stdout
    assert notes[12] == f'"numpy.ndarray[tuple[int, ?U@12], {_F64}]"', result.stdout
    assert notes[13] == f'"numpy.ndarray[tuple[int, ?U@13], {_F64}]"', result.stdout
    assert notes[14] == f'"numpy.ndarray[tuple[?U@14], {_F64}]"', result.stdout


def test_bad_bincount_weights() -> None:
    # Issue 317: numpy raises "The weights and list don't have the same length".
    result = run_mypy(FIXTURES / "bad_bincount_weights.py")
    errors = _errors_by_line(result)
    assert sorted(errors) == [13, 19], result.stdout
    assert "bincount weights must have the length of x: (N,) vs (N-1,)" in errors[13][0], result.stdout
    assert "(4,) vs (3,)" in errors[19][0] and errors[19][0].endswith("[numty-shape]"), result.stdout
    assert _run_fixture("bad_bincount_weights.py") == ["The weights and list don't have the same length."]


def test_array_unpack() -> None:
    # Issue 318: the items of an iterated or unpacked array keep their
    # shape, and a literal first axis is checked against the names.
    result = run_mypy(FIXTURES / "array_unpack.py")
    errors = _errors_by_line(result)
    assert sorted(errors) == [6, 16], result.stdout
    assert "cannot unpack an array of shape (3, 2) into 2 names: axis 0 has size 3" in errors[6][0], result.stdout
    notes = reveals_by_line(result)
    assert notes[9] == '"numpy.ndarray[tuple[?U@7], numpy.dtype[Any]]"', result.stdout
    assert notes[11] == f'"numpy.ndarray[tuple[Literal[2]], {_F64}]"', result.stdout
    assert _run_fixture("array_unpack.py") == ["too many values to unpack (expected 2)"]


def test_union_index_optin() -> None:
    # Issue 319: an index assigned in both branches of an `if` is a
    # union under allow_redefinition. It is solved member by member,
    # and the positions check runs for each.
    fixture = FIXTURES / "union_index_optin.py"
    off = run_mypy(fixture, flags=("--allow-redefinition",))
    assert off.returncode == 0, off.stdout
    notes = reveals_by_line(off)
    assert notes[20] == f'"numpy.ndarray[tuple[?U@20, A, Literal[3]], {_F64}]"', off.stdout
    assert notes[21] == f'"numpy.ndarray[tuple[?U@21, A], {_F64}]"', off.stdout
    on = run_mypy(fixture, flags=("--allow-redefinition", "--enable-error-code", "numty-names"))
    errors = _errors_by_line(on)
    assert sorted(errors) == [14, 20, 21], on.stdout
    for line in (14, 20, 21):
        assert "F and S are different names" in errors[line][0], on.stdout


def test_index_clip_ok() -> None:
    # Issue 315: `np.clip(np.searchsorted(a, v), 0, len(a) - 1)` and
    # `np.minimum(...)` changed the values, so the name assigned from
    # them holds no record; the plain call and the method still do.
    fixture = FIXTURES / "index_clip_ok.py"
    assert run_mypy(fixture).returncode == 0
    errors = _errors_by_line(run_mypy(fixture, flags=("--enable-error-code", "numty-names")))
    assert sorted(errors) == [23, 28], errors
    for line in (23, 28):
        assert "they may run past the axis" in errors[line][0], errors


def test_local_only_name() -> None:
    # A size TypeVar the signature does not mention can be named in the
    # body (mypy alone: `Type variable "Rows" is unbound`). One name is
    # one size per function; at module level it is still unbound.
    fixture = FIXTURES / "local_only_name.py"
    off = _errors_by_line(run_mypy(fixture))
    assert sorted(off) == [26, 37], off
    assert "Rows-1 vs Rows" in off[26][0] and off[26][0].endswith("[numty-broadcast]"), off
    assert 'Rows" is unbound' in off[37][0], off
    on_result = run_mypy(fixture, flags=("--enable-error-code", "numty-names"))
    on = _errors_by_line(on_result)
    assert sorted(on) == [25, 26, 33, 37], on
    assert "Rows and Other are different names" in on[25][0], on
    assert "N and Rows are different names" in on[33][0], on
    assert reveals_by_line(on_result)[24] == '"numpy.ndarray[tuple[Rows], numpy.dtype[Any]]"', on_result.stdout
    assert _run_fixture("local_only_name.py") == ["7.0"]


def test_callsite_symbolic_conflict() -> None:
    # A callee's size name bound to `R` and to `R-1` at one call was
    # silent: only literal sizes were compared (issue #53). Positional,
    # keyword and `X | None` formals are all covered.
    result = run_mypy(FIXTURES / "bad_callsite_symbolic.py")
    errors = _errors_by_line(result)
    assert sorted(errors) == [21, 22, 23, 24], result.stdout
    for line in errors:
        assert len(errors[line]) == 1, result.stdout
        want = "call-site shape conflict for 'N': R (argument 'values', axis 0) vs R-1"
        assert want in errors[line][0], result.stdout
        assert errors[line][0].endswith("[numty-callsite]"), result.stdout


def test_numba_dispatcher_call() -> None:
    # Issue 281: a call of a jitted function is checked as a call of the
    # function: sizes are solved (they were `Never`), a formula return is
    # put in the caller's terms, and the call-site checks apply.
    pytest.importorskip("numba")
    result = run_mypy(FIXTURES / "numba_dispatch.py")
    assert result.returncode == 1, result.stdout
    errors = _errors_by_line(result)
    assert sorted(errors) == [34, 35, 36], result.stdout
    assert "call-site shape conflict for 'N': 5 (argument 'a', axis 0) vs 6 (argument 'b', axis 0)" in errors[34][0], (
        result.stdout
    )
    assert 'Argument 1 to "total"' in errors[35][0] and "[arg-type]" in errors[35][0], result.stdout
    assert 'Argument 2 to "total" has incompatible type "str"; expected "float"' in errors[36][0], result.stdout
    assert "Never" not in "".join(line for lines in errors.values() for line in lines if "total" not in line)
    assert reveals_by_line(result) == {
        29: f'"{_ND}[tuple[M], {_F32}]"',
        30: f'"{_ND}[tuple[M-1], {_F32}]"',
        31: f'"tuple[{_ND}[tuple[M+1], {_F32}], {_ND}[tuple[N+1, M+1], {_F32}]]"',
        32: f'"tuple[{_ND}[tuple[M+1], {_F32}], {_ND}[tuple[N+1, M+1], {_F32}]]"',
        33: f'"{_ND}[tuple[Literal[5]], {_F32}]"',
    }, result.stdout
    assert _run_fixture("numba_dispatch.py") == ["(4,) [(5,), (4, 5)]"]


def test_names_opt_in() -> None:
    # Issue 280: two different names where one size is needed. Off by
    # default: `F64[N] + F64[M]` runs when N == M, and nothing is said
    # (not even mypy's own [operator] error). With the code on, each
    # such line of a generic body has one [numty-names] error.
    fixture = FIXTURES / "names_opt_in.py"
    off = run_mypy(fixture)
    assert sorted(_errors_by_line(off)) == [30], off.stdout
    on = run_mypy(fixture, flags=("--enable-error-code", "numty-names"))
    errors = _errors_by_line(on)
    expected = {
        19: "boolean index on axis 0 (shape (A, B)): B and A are different names",
        21: "broadcast at trailing axis 0 (shapes (B, A) vs (B,)): A and B are different names",
        22: "broadcast at trailing axis 0 (shapes (B,) vs (A,)): B and A are different names",
        23: "assignment of shape (A,) into target of shape (B,): A and B are different names",
        24: "boolean index on axis 0 (shape (A, B)): B and A are different names",
        25: "concatenate dimension 0 (shapes (A, B) vs (B, A)): A and B are different names",
        26: "stack dimension 0 (shapes (B,) vs (A,)): B and A are different names",
        28: "call-site binding of 'K' (argument 'ids', axis 0, and argument 'd', axis 1): "
        "B and A are different names",
        29: "broadcast at trailing axis 0 (shapes (A,) vs (B,)): A and B are different names",
        31: "broadcast at trailing axis 0 (shapes (?U@18,) vs (A,)): ?U@18 and A are different names",
        34: "broadcast at trailing axis 0 (shapes (B-1,) vs (A,)): B-1 and A are different names",
    }
    assert sorted(errors) == sorted([*expected, 30]), on.stdout
    for line, message in expected.items():
        assert len(errors[line]) == 1, on.stdout
        assert message in errors[line][0] and errors[line][0].endswith("[numty-names]"), on.stdout
    assert errors[30][0].endswith("[numty-broadcast]"), on.stdout


def test_index_overrun_certain_or_possible() -> None:
    # A recorded size above the axis is an error by default only when the
    # index holds every position below it (`arange`, `argsort`). For
    # `searchsorted`, `argmax` and `flatnonzero` the size is an upper
    # bound on the values: the usual ways to use such positions safely
    # (fix in place, `out=`, a guard, values known to be in range) are
    # clean, and the possible overrun is an opt-in report.
    fixture = FIXTURES / "bad_index_overrun.py"
    off = _errors_by_line(run_mypy(fixture))
    assert sorted(off) == [13, 14, 16], off
    for line in (13, 14, 16):
        assert len(off[line]) == 1 and off[line][0].endswith("[numty-axis]"), off
    assert "index positions on 5 used on axis 0 of size 3 (shape (3,))" in off[13][0], off
    assert "index positions on N+1 used on axis 0 of size N (shape (N,))" in off[14][0], off
    on = _errors_by_line(run_mypy(fixture, flags=("--enable-error-code", "numty-names")))
    possible = [22, 23, 25, 31, 37, 43, 48, 56]
    assert sorted(on) == sorted([13, 14, 16, *possible]), on
    for line in possible:
        assert len(on[line]) == 1, on
        assert "they may run past the axis" in on[line][0] and on[line][0].endswith("[numty-names]"), on


def test_return_of_helper_result() -> None:
    # Issues 269/271: the return rule covered the result of a numpy call
    # only, so `return np.zeros(n)` passed and `return helper(n)`, with
    # `helper(...) -> F64[int]`, did not. A call of a project function or
    # method is covered through its signature. A literal that disagrees,
    # an unknown, two names and a bare parameter (which is no call) are
    # still errors.
    result = run_mypy(FIXTURES / "bad_ret_helper.py")
    assert result.returncode == 1, result.stdout
    errors = _errors_by_line(result)
    assert sorted(errors) == [39, 48, 52, 56], result.stdout
    for line in (39, 48, 52, 56):
        assert len(errors[line]) == 1 and errors[line][0].endswith("[return-value]"), result.stdout
    assert "tuple[Any, Literal[3]]" in errors[39][0] and "tuple[N, Literal[4]]" in errors[39][0], result.stdout
    assert "tuple[int]" in errors[52][0], result.stdout
    notes = reveals_by_line(result)
    # A local that is returned has `Any` there; one that is not keeps `int`.
    assert notes[30] == '"numpy.ndarray[tuple[Any], numpy.dtype[numpy.float64]]"', result.stdout
    assert notes[44] == '"numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]"', result.stdout


_FAM = "numpy.dtype[numpy.floating[Any]]"
_INTP = f"numpy.dtype[numpy.signedinteger[{_NB}_32Bit | {_NB}_64Bit]]"


def test_coverage_batch_2() -> None:
    # Issue 298: np.arange with an arithmetic bound or a literal start.
    # Issue 300: np.array of numpy scalars and of rows written as tuples.
    # Issue 301: axis reductions and cumsum on the float family.
    # Issue 302: a comparison with a numpy scalar of any width.
    # Issue 303: a builder whose dtype= argument is Any keeps its shape.
    # Issue 304: reduceat with an int-sized index array, and the dtype of
    # add / multiply / logical_* in reduce, accumulate and reduceat.
    result = run_mypy(FIXTURES / "coverage_batch_2.py")
    assert result.returncode == 1, result.stdout
    errors = _errors_by_line(result)
    assert sorted(errors) == [23, 43, 48, 65], result.stdout
    assert "cannot assign value of shape (M+1,) into target of shape (M,)" in errors[23][0], result.stdout
    assert "tuple[Literal[2]]" in errors[43][0] and "tuple[Literal[3]]" in errors[43][0], result.stdout
    assert "Literal[3]]" in errors[48][0] and "tuple[int, Literal[4]]" in errors[48][0], result.stdout
    assert "tuple[M]" in errors[65][0] and "tuple[N]" in errors[65][0], result.stdout
    assert reveals_by_line(result) == {
        14: f'"{_ND}[tuple[M+1], {_INTP}]"',
        15: f'"{_ND}[tuple[M], {_INTP}]"',
        16: f'"{_ND}[tuple[M+1], {_INTP}]"',
        17: f'"{_ND}[tuple[N+1], {_INTP}]"',
        18: f'"{_ND}[tuple[Literal[5]], {_INTP}]"',
        19: f'"{_ND}[tuple[int], {_INTP}]"',
        20: f'"{_ND}[tuple[int], {_INTP}]"',
        27: f'"{_ND}[tuple[Literal[3]], {_F64}]"',
        28: f'"{_ND}[tuple[Literal[2]], {_F32}]"',
        29: f'"{_ND}[tuple[Literal[2]], numpy.dtype[Any]]"',
        30: f'"{_ND}[tuple[Literal[3]], {_F64}]"',
        31: f'"{_ND}[tuple[Literal[2]], numpy.dtype[Any]]"',
        32: f'"{_ND}[tuple[Literal[2]], {_F64}]"',
        33: f'"{_ND}[tuple[Any, ...], {_F64}]"',
        # an `Any` next to a certain scalar: one value, or numpy refuses the ragged display
        34: f'"{_ND}[tuple[Literal[2]], numpy.dtype[Any]]"',
        36: f'"{_ND}[tuple[?U@36, Literal[3]], numpy.dtype[Any]]"',
        37: f'"{_ND}[tuple[?U@37, Literal[3]], {_F64}]"',
        38: f'"{_ND}[tuple[Literal[3], Literal[2]], {_F32}]"',
        39: f'"{_ND}[tuple[Any, ...], numpy.dtype[Any]]"',
        52: f'"{_ND}[tuple[N], {_FAM}]"',
        53: f'"{_ND}[tuple[N], {_FAM}]"',
        54: f'"{_ND}[tuple[M], {_FAM}]"',
        55: f'"{_ND}[tuple[Literal[1], M], {_FAM}]"',
        56: f'"{_ND}[tuple[N], {_FAM}]"',
        57: f'"{_ND}[tuple[N], {_FAM}]"',
        58: f'"{_ND}[tuple[N], {_FAM}]"',
        59: f'"{_ND}[tuple[N, M], {_FAM}]"',
        60: f'"{_ND}[tuple[M*N], {_FAM}]"',
        61: '"numpy.floating[Any]"',
        69: f'"{_ND}[tuple[N], {_BOOL}]"',
        70: f'"{_ND}[tuple[N], {_BOOL}]"',
        71: f'"{_ND}[tuple[N], {_BOOL}]"',
        72: f'"{_ND}[tuple[N], {_BOOL}]"',
        73: f'"{_ND}[tuple[N], {_BOOL}]"',
        74: f'"{_ND}[tuple[N], {_BOOL}]"',
        75: f'"{_ND}[tuple[N], {_BOOL}]"',
        76: f'"{_ND}[tuple[N], {_BOOL}]"',
        80: f'"{_ND}[tuple[int, int], numpy.dtype[Any]]"',
        81: f'"{_ND}[tuple[int, int], numpy.dtype[Any]]"',
        82: f'"{_ND}[tuple[int, int], numpy.dtype[Any]]"',
        83: f'"{_ND}[tuple[N, Literal[3]], numpy.dtype[Any]]"',
        84: f'"{_ND}[tuple[N], numpy.dtype[Any]]"',
        85: f'"{_ND}[tuple[int, int], {_F32}]"',
        86: f'"{_ND}[tuple[Literal[2], Literal[3]], numpy.dtype[Any]]"',
        87: f'"{_ND}[tuple[int, Literal[2]], numpy.dtype[Any]]"',
        91: f'"{_ND}[tuple[int], {_F64}]"',
        92: f'"{_ND}[tuple[K], {_F64}]"',
        93: f'"{_ND}[tuple[int], {_F32}]"',
        94: f'"{_ND}[tuple[K], {_BOOL}]"',
        95: f'"{_ND}[tuple[int], {_BOOL}]"',
        96: f'"{_ND}[tuple[K], {_I64}]"',
        97: f'"{_ND}[tuple[N], {_F32}]"',
        98: '"numpy.bool[bool]"',
    }, result.stdout
    assert _run_fixture("coverage_batch_2.py") == [
        "(4,) (5,) (5,) (5,)",
        "float64 float32 float64",
        "float64 float64 float32",
        "(2, 3) (2,)",
        "(3,) (4,) (1, 4) (3, 4)",
        "float32 float32 float32 float32 (12,)",
        "(3,) bool (3,)",
        "(2, 3) float32",
        "(2,) float32 bool",
        "int64 float32 bool",
    ]


def test_return_transposed_is_not_let_through() -> None:
    # Issue 297: a plain `int` size fits a declared name at a return
    # (issues 269/271), and a name fits a declared `int`, so `(int, N)`
    # passed where `(N, int)` is declared. The `int` is kept when the
    # name declared on its axis is a size of the value on another axis.
    result = run_mypy(FIXTURES / "bad_ret_transposed.py")
    assert result.returncode == 1, result.stdout
    errors = _errors_by_line(result)
    assert sorted(errors) == [13, 18, 24, 28], result.stdout
    for line in (13, 18, 24, 28):
        assert len(errors[line]) == 1 and errors[line][0].endswith("[return-value]"), result.stdout
    assert 'got "ndarray[tuple[int, N]' in errors[13][0] and "tuple[N, int]" in errors[13][0], result.stdout
    assert 'got "ndarray[tuple[N, int]' in errors[18][0] and "tuple[int, N]" in errors[18][0], result.stdout
    assert "| None" in errors[24][0], result.stdout
    assert "tuple[N, int, Any]" in errors[28][0] and "tuple[int, N, int]" in errors[28][0], result.stdout
    assert _run_fixture("bad_ret_transposed.py") == ["(2, 3) (3, 2) (3, 2) (3, 2, 4)"]


def test_index_record_survives_a_selection() -> None:
    # Issue 305: `rows[known]`, `rows[1:]` and `rows[order]` still hold
    # positions on the size `rows` does. Silent by default, like every
    # report of one named size on an axis of another.
    fixture = FIXTURES / "index_selection.py"
    off = run_mypy(fixture)
    assert off.returncode == 0, off.stdout
    on = _errors_by_line(run_mypy(fixture, flags=("--enable-error-code", "numty-names")))
    assert sorted(on) == [15, 17, 19, 20], on
    for line in (15, 17, 19, 20):
        assert len(on[line]) == 2, on
        assert "index positions on axis 0 (shape (P, E)): E and P are different names" in on[line][0], on
        assert "index positions on axis 1 (shape (P, E)): P and E are different names" in on[line][1], on
        assert all(text.endswith("[numty-names]") for text in on[line]), on
    assert _run_fixture("index_selection.py") == ["(3,)", "IndexError"]


def test_polars_to_numpy_cast_chain() -> None:
    # Issue 299: the cast is seen behind the Series methods that keep the
    # dtype (fill_null, fill_nan, drop_nulls, sort, head, ...), `astype`
    # names the dtype of an array of unknown rank for `float` as for
    # `np.float64`, and `astype(d)` with a dtype variable keeps an
    # `int`-sized array. A method that is not on the list (`cum_sum`)
    # and a plain `fill_null` with no cast keep polars' own answer.
    pytest.importorskip("polars")
    result = run_mypy(FIXTURES / "polars_cast_chain.py")
    assert result.returncode == 1, result.stdout
    errors = _errors_by_line(result)
    assert sorted(errors) == [36], result.stdout
    assert "tuple[int, Literal[1]]" in errors[36][0] and errors[36][0].endswith("[return-value]"), result.stdout
    polars_answer = '"numpy.ndarray[Any, Any]"'
    assert reveals_by_line(result) == {
        16: f'"{_ND}[tuple[int], {_F64}]"',
        17: f'"{_ND}[tuple[int], {_F32}]"',
        18: f'"{_ND}[tuple[int], {_F64}]"',
        19: f'"{_ND}[tuple[int], {_F64}]"',
        20: f'"{_ND}[tuple[int], numpy.dtype[Any]]"',
        21: polars_answer,
        22: f'"{_ND}[tuple[int, int], numpy.dtype[Any]]"',
        23: polars_answer,
        24: f'"{_ND}[Any, {_F64}]"',
        25: f'"{_ND}[Any, {_F64}]"',
        26: f'"{_ND}[Any, {_BOOL}]"',
        30: f'"{_ND}[tuple[N], numpy.dtype[FloatT]]"',
        31: f'"{_ND}[tuple[int], numpy.dtype[FloatT]]"',
        32: f'"{_ND}[tuple[int], {_F32}]"',
    }, result.stdout
    assert _run_fixture("polars_cast_chain.py") == [
        "(3,) float32",
        "(2,) float64",
        "float64 (3,)",
    ]


def test_size_int_is_a_scalar_in_arithmetic(tmp_path: Path) -> None:
    # Issue 325: an `int` typed by a size TypeVar (`X.shape[0]`, `len(X)`)
    # is a Python int in array arithmetic, like `float(X.shape[0])` is a
    # Python float: `X / len(X)` kept no rank, and under NUMTY_STRICT=1
    # `n_samples * p` was Any. The transposed covariance and the mean over
    # the wrong axis are now reported at the return.
    bad = run_mypy(FIXTURES / "bad_size_scalar_cov.py")
    assert bad.returncode == 1, bad.stdout
    errors = _errors_by_line(bad)
    assert sorted(errors) == [12, 16], bad.stdout
    assert 'got "ndarray[tuple[N, N]' in errors[12][0] and errors[12][0].endswith("[return-value]"), bad.stdout
    assert 'got "ndarray[tuple[D]' in errors[16][0] and errors[16][0].endswith("[return-value]"), bad.stdout
    assert _run_fixture("bad_size_scalar_cov.py") == ["(5, 5) (3,)"]
    fixture = FIXTURES / "size_scalar_arith_ok.py"
    ok = run_mypy(fixture)
    assert ok.returncode == 0, ok.stdout
    assert reveals_by_line(ok) == {
        14: f'"{_ND}[tuple[N, D], {_F64}]"',
        15: f'"{_ND}[tuple[N, D], {_F64}]"',
        16: f'"{_ND}[tuple[C], {_F64}]"',
        17: f'"{_ND}[tuple[N, D], {_F64}]"',
        18: f'"{_ND}[tuple[N], {_F64}]"',
        19: f'"{_ND}[tuple[N], {_I64}]"',
        20: f'"{_ND}[tuple[D], {_F32}]"',
        21: f'"{_ND}[tuple[N, D], {_F64}]"',
    }, ok.stdout
    assert _run_fixture("size_scalar_arith_ok.py") == ["(5, 3) (5, 3) (3, 3)", "float64 int64 float32"]
    # Under NUMTY_STRICT=1 the function body proves every step.
    strict = subprocess.run(
        [sys.executable, "-m", "mypy", "--cache-dir", str(tmp_path / "cache"), str(fixture)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT), "NUMTY_STRICT": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    in_body = [
        line for line in strict.stdout.splitlines() if ": error:" in line and int(line.split(":")[1]) < 23
    ]
    assert in_body == [], strict.stdout
    assert reveals_by_line(strict) == reveals_by_line(ok), strict.stdout


def test_own_size_a_constant_apart() -> None:
    # Issue 328: a size a call chose (`?K@17`) and one mask's count are
    # one size, like a name, so `?K-1` against `?K-2` is a default error
    # worded like `N-2` against `N-1`, not two names, and the `np.where`
    # around one no longer gets a `tuple[Never, ...]` [arg-type].
    fixture = FIXTURES / "bad_own_size_offset.py"
    for result in (run_mypy(fixture), run_mypy(fixture, flags=("--enable-error-code", "numty-names"))):
        assert result.returncode == 1, result.stdout
        errors = _errors_by_line(result)
        assert sorted(errors) == [16, 18, 19, 21, 24], result.stdout
        assert all(len(errors[line]) == 1 for line in errors), result.stdout
        assert all(errors[line][0].endswith("[numty-broadcast]") for line in errors), result.stdout
        assert "?K@17-1 vs ?K@17-2 (shapes (?K@17-1,) vs (?K@17-2,))" in errors[18][0], result.stdout
        assert "?K@17-1 vs ?K@17 " in errors[19][0], result.stdout
        assert "?K@17-2 vs ?K@17-1 " in errors[21][0], result.stdout
        assert "?U@23-1 vs ?U@23 " in errors[24][0], result.stdout
        assert "Never" not in result.stdout and "different names" not in result.stdout, result.stdout
    assert _run_fixture("bad_own_size_offset.py") == [
        "operands could not be broadcast together with shapes (3,) (2,) ",
        "operands could not be broadcast together with shapes (3,) (4,)",
    ]
    # A call or a mask variable in a loop or a comprehension gives a size
    # per run: two runs may differ by one, so only the opt-in reports it,
    # as `1 apart` and with no [operator] follow-on.
    fixture = FIXTURES / "own_size_offset_loop_ok.py"
    assert run_mypy(fixture).returncode == 0
    on = run_mypy(fixture, flags=("--enable-error-code", "numty-names"))
    errors = _errors_by_line(on)
    assert sorted(errors) == [17, 22, 24], on.stdout
    for line in (17, 22, 24):
        assert len(errors[line]) == 1 and errors[line][0].endswith("are 1 apart  [numty-names]"), on.stdout
    assert "?K@16-1 and ?K@16 are 1 apart" in errors[17][0], on.stdout
    assert _run_fixture("own_size_offset_loop_ok.py") == ["(3,) (2,) (2,)"]
