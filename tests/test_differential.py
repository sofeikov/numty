"""numty's inferred shapes and dtypes against what numpy returns (issue #165).

Generated numpy expressions are type-checked with the plugin and
evaluated with numpy; see tests/differential.py for the rules. A
failure prints each disagreement with numty's type and numpy's result.

The default run is small and fixed-seed so the suite stays fast and
repeatable. For a longer hunt:

    NUMTY_FUZZ_SEEDS=1-30 NUMTY_FUZZ_COUNT=700 uv run pytest tests/test_differential.py
    PYTHONPATH=. python -m tests.differential --seeds 1-30 --count 700
"""

import os
from pathlib import Path

import numpy as np
import pytest

from tests import differential as d


def _seeds() -> list[int]:
    spec = os.environ.get("NUMTY_FUZZ_SEEDS", "1-3")
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


COUNT = int(os.environ.get("NUMTY_FUZZ_COUNT", "400"))


@pytest.mark.parametrize("seed", _seeds())
@pytest.mark.parametrize("mode", ["concrete", "symbolic"])
def test_numty_agrees_with_numpy(mode: str, seed: int, tmp_path: Path) -> None:
    cases = d.generate(seed, COUNT, symbolic=mode == "symbolic")
    findings, stats = d.check(cases, tmp_path)
    assert not findings, (
        f"{len(findings)} disagreement(s) with numpy ({mode}, seed {seed}):\n"
        + "\n".join(str(f) for f in findings)
    )
    # The oracle must be looking at real answers: if reveals stopped
    # parsing, or the plugin stopped solving, everything would "agree".
    exact = stats.get("agree: exact shape", 0)
    assert exact >= COUNT // 4, stats
    flagged = stats.get("numpy raises, flagged", 0)
    assert flagged >= COUNT // 20, stats


def test_generation_is_deterministic() -> None:
    first = [c.expr for c in d.generate(7, 50, symbolic=False)]
    again = [c.expr for c in d.generate(7, 50, symbolic=False)]
    assert first == again
    assert first != [c.expr for c in d.generate(8, 50, symbolic=False)]


ARRAY = "numpy.ndarray[tuple[{}], numpy.dtype[{}]]"
F32 = "numpy.floating[numpy._typing._nbit_base._32Bit]"
C64 = (
    "numpy.complexfloating[numpy._typing._nbit_base._32Bit, "
    "numpy._typing._nbit_base._32Bit]"
)


@pytest.mark.parametrize(
    "revealed, value, expected",
    [
        (ARRAY.format("Literal[2], Literal[3]", "numpy.float64"), np.zeros((2, 3)), None),
        (ARRAY.format("Literal[2], Literal[3]", "numpy.float64"), np.zeros((3, 2)), "WRONG_SHAPE"),
        (ARRAY.format("Literal[2]", "numpy.float64"), np.zeros((2, 1)), "WRONG_SHAPE"),
        (ARRAY.format("Literal[2]", "numpy.float64"), np.zeros(2, np.float32), "WRONG_DTYPE"),
        (ARRAY.format("Literal[2]", F32), np.zeros(2, np.float32), None),
        (ARRAY.format("Literal[2]", C64), np.zeros(2, np.complex64), None),
        (ARRAY.format("Literal[2]", C64), np.zeros(2, np.complex128), "WRONG_DTYPE"),
        # Unknown dims, unknown rank and unknown dtypes claim nothing.
        (ARRAY.format("Any, Literal[3]", "numpy.float64"), np.zeros((9, 3)), None),
        (ARRAY.format("Any, ...", "Any"), np.zeros((9, 3), np.int8), None),
        (ARRAY.format("?U@7, Literal[3]", "numpy.float64"), np.zeros((9, 3)), None),
        (ARRAY.format("()", "numpy.float64"), np.float64(1.0), None),
        # Symbolic dims are evaluated under SYMBOL_SIZES (N=3, M=5).
        (ARRAY.format("N, M", "numpy.float64"), np.zeros((3, 5)), None),
        (ARRAY.format("N+1, N*M", "numpy.float64"), np.zeros((4, 15)), None),
        (ARRAY.format("N//2+1", "numpy.float64"), np.zeros(2), None),
        (ARRAY.format("N-1", "numpy.float64"), np.zeros(3), "WRONG_SHAPE"),
        # Tuples and lists of arrays compare item by item.
        (
            "tuple[{}, {}]".format(
                ARRAY.format("Literal[2]", "numpy.float64"),
                ARRAY.format("Literal[3]", "numpy.float64"),
            ),
            (np.zeros(2), np.zeros(4)),
            "WRONG_SHAPE",
        ),
        (
            "builtins.list[{}]".format(ARRAY.format("Literal[2]", "numpy.float64")),
            [np.zeros(2), np.zeros(2)],
            None,
        ),
    ],
)
def test_comparison_rules(revealed: str, value: object, expected: str | None) -> None:
    assert d._compare(d.parse_type(revealed), value) == expected


def test_symbolic_claims_are_checked_at_every_size(tmp_path: Path) -> None:
    # `N+1` is right for every N; `N-1` for `a[1:]` is right until the
    # axis is empty, where numpy returns 0 and the formula says -1.
    sizes = d.SIZE_ASSIGNMENTS
    assert any(0 in s.values() for s in sizes)
    assert {s["N"] % 2 for s in sizes if s["N"]} == {0, 1}
    revealed = d.parse_type(ARRAY.format("N//2+1", "numpy.float64"))
    assert d._compare(revealed, np.zeros(2), {"N": 3}) is None
    assert d._compare(revealed, np.zeros(2), {"N": 4}) == "WRONG_SHAPE"
    assert d._compare(revealed, np.zeros(3), {"N": 4}) is None

    grow = d.Case("np.pad(a0, 1)", [("a0", "F64", ("N",))])
    shrink = d.Case("a0[1:]", [("a0", "F64", ("N",))])
    # Off by one for every N >= 2, but (0,) + (0,) at N=0: the error
    # names `N-1`, which is -1 there, so it is the same known gap.
    off = d.Case("a0[1:] + a0", [("a0", "F64", ("N",))])
    findings, stats = d.check([grow, shrink, off], tmp_path)
    assert not findings, "\n".join(str(f) for f in findings)
    assert stats["agree: exact shape"] == 2 * len(sizes) - 1, stats
    assert stats[d.BELOW_DOMAIN] == 2, stats
    assert stats["numpy raises, flagged"] == len(sizes) - 1, stats
    assert d._negative_dim("shapes (N-1,) vs (N,)", {"N": 0})
    assert not d._negative_dim("shapes (N-1,) vs (N,)", {"N": 3})
    assert not d._negative_dim("different sizes: 2*M vs M", {"M": 0})
    assert not d._negative_dim("axis -1 is out of range (shape (N,))", {"N": 0})

    # A claim that is wrong without going negative is a finding, and
    # says which sizes exposed it.
    wrong = d.Finding("WRONG_SHAPE", shrink, "N", "shape (0,)", {"N": 0})
    assert "numpy with N=0" in str(wrong)


def test_known_gaps_are_narrow() -> None:
    # A known gap excuses one finding kind on one call shape, nothing else.
    import re

    for kind, pattern, why in d.KNOWN_GAPS:
        assert kind in {"WRONG_SHAPE", "WRONG_DTYPE", "CLAIM_ON_ERROR", "FALSE_POSITIVE"}
        assert why
        assert not re.search(pattern, "np.zeros((2, 3)) + np.zeros((2, 3))")


# Expressions the oracle once caught numty getting wrong, pinned so they
# stay fixed whatever the generator happens to produce. `raises` marks
# the ones numpy rejects: numty must flag those rather than type them.
REGRESSIONS: list[tuple[str, bool]] = [
    # #166: an inline complex operand was typed against the float overload.
    ("np.zeros((3, 2, 2, 3)) * np.ones((5, 5, 5), dtype=np.complex64)", True),
    ("np.zeros((3, 2)) + np.ones((3, 2), dtype=np.complex64)", False),
    ("np.ones((2, 3), dtype=np.int8) @ np.ones((3,), dtype=np.complex128)", False),
    ("np.ones((3,), dtype=np.float32) ** np.ones((3,), dtype=np.complex128)", False),
    ("np.ones((6,), dtype=np.int8) // np.ones((6,), dtype=np.complex64)", True),
    ("np.ones((4,)) % np.ones((4,), dtype=np.complex128)", True),
    # #167: searchsorted only searches a 1-D array.
    ("np.searchsorted(np.ones((3, 3)), np.ones((6,)))", True),
    ("np.searchsorted(np.ones((3,)), np.ones((6, 2)))", False),
    # #168: ascontiguousarray/asfortranarray lift 0-d to (1,).
    ("np.ascontiguousarray(np.ones(()))", False),
    ("np.asfortranarray(np.ones((), dtype=np.float32))", False),
    # #170: argsort treats a 0-d array as one element.
    ("np.argsort(np.zeros((), dtype=np.uint8), axis=0)", False),
    ("np.zeros(()).argsort()", False),
    # Found by the oracle itself.
    ("np.percentile(np.zeros((5, 3), dtype=np.float16), np.zeros((2,)))", False),
    ("np.quantile(np.ones((4, 4), dtype=np.float32), 0.5, axis=0)", False),
    ("np.gradient(np.ones((1, 6, 3), dtype=np.int64), axis=1)", False),
    ("np.gradient(np.zeros((4, 1, 5)), axis=-2)", True),
    ("np.fft.fft2(np.zeros((4,)))", True),
    ("np.fft.rfft(np.zeros((5, 6), dtype=np.complex64))", True),
    ("np.fft.fft(np.zeros((5, 6), dtype=np.complex64))", False),
    ("np.fft.irfft(np.ones((2,), dtype=np.complex64), n=5)", False),
    ("np.fft.irfft(np.ones((4, 6)))", False),
    ("np.linspace((2, 2), 7)", False),
    ("np.average(np.zeros((4, 2, 6), dtype=np.uint8), axis=2, weights=np.ones((6,)))", False),
    ("np.average(np.ones((5, 4), dtype=np.complex64), axis=1, weights=np.ones((4,)))", False),
    ("np.clip(np.zeros((5, 6), dtype=bool), 0, np.ones((5, 6), dtype=np.uint8))", False),
    ("np.ones((4,), dtype=bool) ** 2", False),
    ("2 ** np.ones((4,), dtype=bool)", False),
    ("np.zeros((1, 6), dtype=bool).round()", False),
    ("np.angle(np.zeros((3, 2), dtype=np.complex64))", False),
    ("np.cbrt(np.ones((4, 3), dtype=np.complex64))", True),
    ("np.arctan2(np.ones((6,), dtype=np.int8), np.zeros((6,), dtype=np.uint8))", False),
    ("np.correlate(np.zeros(()), np.ones((2,)))", True),
    ("np.zeros(()).mean(axis=0, keepdims=True)", True),
    ("np.zeros(()).sum(axis=0, keepdims=True)", False),
]


def test_empty_axis_is_not_a_size_mismatch(tmp_path: Path) -> None:
    # Found by the zero-size assignments: `2*M` and `M` are the same
    # size when M is 0, so numpy accepts these and numty must not call
    # them a mismatch. `M+1` vs `M` differ for every M and still error.
    params = [("a0", "F64", (2, "M")), ("a1", "F64", ("M",))]
    mono = [("a0", "F64", ("N", "M", 2)), ("a1", "I64", ("M", "N"))]
    cases = [
        d.Case("np.vdot(a0, a1)", params),
        d.Case("np.vdot(a0, a1)", mono),
        d.Case("a0.ravel() + a1", params),
        d.Case("np.vdot(np.pad(a1, (0, 1)), a1)", params),
    ]
    empty = {"N": 3, "M": 0, "K": 7, "P": 4}
    for case in cases[:3]:
        assert not isinstance(d.evaluate(case, empty), Exception), case.expr
    assert isinstance(d.evaluate(cases[3], empty), Exception)
    findings, stats = d.check(cases, tmp_path)
    assert not findings, "\n".join(str(f) for f in findings)
    # Only the pad case is flagged, under all four assignments.
    assert stats["numpy raises, flagged"] == len(d.SIZE_ASSIGNMENTS), stats


def test_regressions_stay_fixed(tmp_path: Path) -> None:
    cases = [d.Case(expr) for expr, _ in REGRESSIONS]
    for case, (expr, raises) in zip(cases, REGRESSIONS):
        # Guard the table itself: `raises` must describe numpy today.
        assert isinstance(d.evaluate(case), Exception) == raises, expr
    findings, stats = d.check(cases, tmp_path)
    assert not findings, "\n".join(str(f) for f in findings)
    expected = sum(1 for _, raises in REGRESSIONS if raises)
    assert stats.get("numpy raises, flagged", 0) == expected, stats


def test_regressions_from_unseen_seeds(tmp_path: Path) -> None:
    # Second batch, from seeds the first round of fixes never saw.
    table = [
        ("np.nanmean(np.ones((), dtype=np.float32), axis=0)", False),
        ("np.mean(np.ones(()), axis=0)", True),
        ("np.broadcast_to(np.ones((1,)), ())", True),
        ("np.broadcast_to(np.ones(()), (3,))", False),
        ("np.quantile(np.ones((4, 4), dtype=np.complex128), 0.5)", True),
        ("np.fft.rfft2(np.zeros(()))", True),
        ("np.fft.rfft2(np.zeros((2, 4, 6)))", False),
    ]
    cases = [d.Case(expr) for expr, _ in table]
    for case, (expr, raises) in zip(cases, table):
        assert isinstance(d.evaluate(case), Exception) == raises, expr
    findings, stats = d.check(cases, tmp_path)
    assert not findings, "\n".join(str(f) for f in findings)
    assert stats.get("numpy raises, flagged", 0) == sum(1 for _, r in table if r), stats
