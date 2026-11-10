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


def test_regressions_stay_fixed(tmp_path: Path) -> None:
    cases = [d.Case(expr) for expr, _ in REGRESSIONS]
    for case, (expr, raises) in zip(cases, REGRESSIONS):
        # Guard the table itself: `raises` must describe numpy today.
        assert isinstance(d.evaluate(case), Exception) == raises, expr
    findings, stats = d.check(cases, tmp_path)
    assert not findings, "\n".join(str(f) for f in findings)
    expected = sum(1 for _, raises in REGRESSIONS if raises)
    assert stats.get("numpy raises, flagged", 0) == expected, stats
