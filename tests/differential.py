"""Differential oracle: numty's static answer against what numpy returns.

Issue #165. The fixtures pin behaviours someone thought of; this
generates numpy expressions nobody wrote down, asks mypy+numty for
their type (`reveal_type`) and evaluates the same text with numpy. The
two must agree:

- a shape or dtype numty claims must be the one numpy produces;
- an expression numpy rejects must not get a fully concrete type with
  no diagnostic (concrete mode, `CLAIM_ON_ERROR`);
- a symbolic expression numpy rejects under *every* small size
  assignment (1-3 for each name, including equal names) must not pass
  silently either (issue #183, `MISSED_ERROR`);
- an expression numpy accepts must not draw a numty shape error.

Losing precision (an unknown dim, `Any`, a bail) is allowed only when
the rejection is unprovable: some size assignment would make the
expression work (`F64[N] + F64[M]` with `N == M`). An assignment-proof
rejection is a miss, not prudence. `SIZE_ASSIGNMENTS` keeps names
distinct by design, so it cannot make this distinction on its own.

Two modes share the machinery. Concrete mode builds every operand with
a literal shape. Symbolic mode wraps the expression in a function whose
parameters carry TypeVar dims (`F64[N, M]`), then calls it with fixed
sizes and substitutes them into the revealed dims (`N+1`, `N*M`, ...).
A symbolic answer is a claim about every size, so each case is run
under every assignment in SIZE_ASSIGNMENTS (one mypy run, several numpy
evaluations): odd and even sizes, and an empty axis.

Issue #185: parameters may also carry a variadic prefix
(`F64[Unpack[Ts], N, M]`), instantiated at runtime with the prefixes in
TS_PREFIXES (ranks 0, 1 and 2). Every such case is evaluated under each
(size, prefix) combination, and a revealed `*Ts` is substituted with the
prefix before the shape is compared. Generic helpers may carry `*Ts`
too, with the reference in `Helper.__call__` binding the shared prefix.

Used by tests/test_differential.py; run it by hand for a longer hunt:

    PYTHONPATH=. python -m tests.differential --seeds 1-20 --count 800
"""

from __future__ import annotations

import ast
import itertools
import os
import random
import re
import shutil
import subprocess
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Exceptions that mean "these shapes/axes/dtypes do not fit": the cases
# numty exists to catch. Data-dependent failures (singular matrices,
# non-convergence) say nothing about shapes and are ignored.
_SHAPE_ERRORS = (ValueError, IndexError, TypeError)
_DATA_ERRORS = (np.linalg.LinAlgError, ZeroDivisionError, OverflowError, FloatingPointError)

# numty codes that assert the code is wrong. `numty-bail` and
# `numty-unsolved` say "cannot check", so they are not counted as
# false positives on working code.
_ASSERTING_CODES = ("numty-shape", "numty-broadcast", "numty-axis", "numty-callsite")

# Wrong answers that are known and not numty's to give: the call is not
# hooked, so the type comes from numpy's own stubs. Each entry is
# (finding kind, pattern on the expression, why). Matching findings are
# counted, not failed, so a new case of the same kind still stands out.
KNOWN_GAPS: tuple[tuple[str, str, str], ...] = (
    (
        "WRONG_DTYPE",
        r"^np\.(cov|corrcoef)\(np\.\w+\(\(.*\), dtype=np\.complex\d+\)\)$",
        "numpy's stubs type cov/corrcoef of a complex array as float64",
    ),
    (
        "CLAIM_ON_ERROR",
        r"^np\.digitize\(np\.\w+\(\(.*?\), dtype=np\.complex\d+\), ",
        "numpy's stubs accept a complex x for digitize, numpy raises",
    ),
)

# A symbolic dim that comes out negative under a size assignment (`N-1`
# at N=0, from `a[1:]`) is counted under this key, not failed, whether
# it sits in the revealed shape or in the error numty reports.
# numty's subtraction is exact where numpy's slicing and `diff`
# saturate at zero, so such a formula only holds for sizes at least as
# large as what it subtracts. A wrong answer whose dims all stay
# non-negative still fails.
BELOW_DOMAIN = "known gap: size below the formula's domain"
# A shape error on a symbolic case says the code is wrong for sizes in
# general, not for every size: `N+1` against `N` does not broadcast,
# except that at N=0 it is 1 against 0, which does. So an error is a
# false positive only when numpy accepts the expression under every
# assignment; where numpy raises under some and accepts under others,
# the accepting ones are counted under this key.
SPECIAL_SIZES = "error stands: numpy accepts only at special sizes"

DIMS = (1, 2, 3, 4, 5, 6)
DTYPES = (
    "",
    "",
    "",
    "np.float32",
    "np.float16",
    "np.int64",
    "np.int32",
    "np.int8",
    "np.uint8",
    "bool",
    "np.complex64",
    "np.complex128",
)
# Symbolic mode: sizes are distinct and never 1, so two different names
# can neither broadcast nor coincide by accident.
SYMBOL_SIZES = {"N": 3, "M": 5, "K": 7, "P": 4}
# What a symbolic case is evaluated under. The first is the default.
# The second flips every parity (`N//2+1`, `(N+1)//2`); the last two
# make one axis empty, where `2*N == N` and `N*M == N` hold after all.
# Zero keeps the rule above: unlike 1, an empty axis does not broadcast
# against other sizes.
SIZE_ASSIGNMENTS: tuple[dict[str, int], ...] = (
    SYMBOL_SIZES,
    {"N": 4, "M": 6, "K": 8, "P": 5},
    {"N": 0, "M": 5, "K": 7, "P": 4},
    {"N": 3, "M": 0, "K": 7, "P": 4},
)
# Dims that are formulas over the names, so rules are also tested on
# inputs that are already expressions (`F64[N, "N+1"]`). None can be
# negative, whatever the sizes: the operand has to exist at runtime.
SYMBOL_FORMULAS = ("N+1", "N+2", "2*N", "2*M+1", "N+M", "M+K")
SYMBOL_HEADS = {
    "F64": "float64",
    "F32": "float32",
    "I64": "int64",
    "U8": "uint8",
    "Bool": "bool",
}
# A variadic prefix, e.g. `F64[*Ts, N, M]` (issue #185). Operands carrying
# it are instantiated at runtime with prefixes of rank 0, 1 and 2, and
# every such case is evaluated under each prefix, the way sizes are
# varied. The marker lives in shape tuples as the string "*Ts"; it is
# spelled `Unpack[Ts]` in annotations and `*Ts` in reveals.
TS_DIM: object = "*Ts"
TS_PREFIXES: tuple[tuple[int, ...], ...] = ((), (2,), (2, 6))


def _distinct_names(case: "Case") -> list[str]:
    """Sorted TypeVar names appearing in a symbolic case's shapes.

    Issue #183: includes names inside formulas (`N+1` contains `N`)
    and inside helpers, so exhaustive probing varies every size that
    could make the expression work.
    """
    names: set[str] = set()
    for _, _, shape in case.params:
        for dim in shape:
            if dim == TS_DIM:
                continue
            if isinstance(dim, int):
                continue
            for cand in ("N", "M", "K", "P"):
                if cand in str(dim):
                    # Single-letter names cannot hide inside other
                    # identifiers here (dims are `N`, `N+1`, ...).
                    names.add(cand)
    for helper in case.helpers:
        for _, dims in helper.formals:
            for dim in dims:
                if dim == TS_DIM or isinstance(dim, int):
                    continue
                for cand in ("N", "M", "K", "P"):
                    if cand in str(dim):
                        names.add(cand)
        _, ret_dims = helper.ret
        for dim in ret_dims:
            if dim == TS_DIM or isinstance(dim, int):
                continue
            for cand in ("N", "M", "K", "P"):
                if cand in str(dim):
                    names.add(cand)
    return sorted(names)


def _always_raises(case: "Case") -> bool:
    """True when numpy rejects `case` under every small size assignment.

    Issue #183: symbolic mode counted every silent rejection as
    allowed, mixing correct silence (`F64[N] + F64[M]`, where `N` could
    equal `M`) with genuine misses (a literal axis out of range for a
    known rank). Trying every combination of sizes 1-3 for each
    distinct name separates them: if any assignment succeeds (or raises
    a data-dependent error, which the oracle skips), the miss is
    unprovable by design and stays allowed. Only an assignment-proof
    rejection becomes a `MISSED_ERROR` failure. `SIZE_ASSIGNMENTS`
    keeps names distinct (and never 1) by design, so it cannot make
    this distinction on its own.

    TypeError (dtype loops like float `bitwise_and`) is excluded by the
    caller: this only probes, it does not classify. Cases with `*Ts`
    need prefix reasoning as follow-up and stay unprovable for now.

    `cross` is excluded for now: its provability can hinge on the same
    TypeVar appearing both as the vector dim (which must be 3) and in
    the broadcast leading dims (e.g. `[P, N, N]` vs `[2, N]`, where N
    must be both 3 and 2). `repeat`, `vdot` and stacking (`stack`,
    including `dstack`) with a single repeated name are excluded for
    the same cross-position / number-theoretic reason.

    Helpers and formulas (`N+1`, `2*N`, ...) are excluded for now:
    they come from newer generator families (#182 methods with formula
    returns, #184 reshape targets, #185 helpers/*Ts) whose
    assignment-proof misses need formula reasoning as follow-up. This
    keeps #183's fix to the original families (literal axes/ranks and
    simple `N`/`M`/`K`/`P`/`2`/`3` shapes) that sampling found.
    """
    if case.has_ts:
        return False
    if case.helpers:
        return False
    for _, _, shape in case.params:
        for dim in shape:
            if dim == TS_DIM or isinstance(dim, int):
                continue
            if dim not in ("N", "M", "K", "P"):
                # A formula (e.g. "N+1", "2*N"): needs follow-up.
                return False
    if "cross" in case.expr:
        return False
    if any(op in case.expr for op in ("repeat", "vdot", "stack")):
        return False
    names = _distinct_names(case)
    if not names:
        return True
    for combo in itertools.product((1, 2, 3), repeat=len(names)):
        mapping = dict(zip(names, combo))
        try:
            ns = case.runtime_namespace(mapping, ())
        except Exception:
            # A size that cannot form an operand (e.g. negative from a
            # formula) is not a shape rejection: treat as unprovable.
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                try:
                    eval(case.expr, ns)
                except Exception as exc:  # noqa: BLE001 - the exception is the answer
                    if isinstance(exc, _SHAPE_ERRORS):
                        continue
                    return False
                else:
                    return False
    return True


# --------------------------------------------------------------------------
# Expression generation


@dataclass
class Operand:
    """One array operand: the text to write and the shape it has."""

    text: str
    shape: tuple[int, ...]


class Gen:
    """Random numpy expressions over operands from `self.operand`."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.current_family: str = ""

    # -- operands ---------------------------------------------------------

    def shape(self, rank: int | None = None) -> tuple[int, ...]:
        r = self.rng
        if rank is None:
            rank = r.choice((0, 1, 1, 2, 2, 2, 3, 3, 4))
        return tuple(r.choice(DIMS) for _ in range(rank))

    def operand(
        self, shape: tuple[int, ...] | None = None, dtype: str | None = None
    ) -> Operand:
        r = self.rng
        if shape is None:
            shape = self.shape()
        if dtype is None:
            dtype = r.choice(DTYPES)
        builder = r.choice(("zeros", "ones"))
        suffix = f", dtype={dtype}" if dtype else ""
        return Operand(f"np.{builder}({shape!r}{suffix})", shape)

    def near(self, shape: tuple[int, ...]) -> Operand:
        """An operand that often, but not always, fits next to `shape`."""
        r = self.rng
        roll = r.random()
        if roll < 0.45:
            return self.operand(shape)
        if roll < 0.6 and shape:
            return self.operand(shape[r.randrange(len(shape)) :])
        if roll < 0.75 and shape:
            mixed = tuple(1 if r.random() < 0.4 else d for d in shape)
            return self.operand(mixed)
        return self.operand()

    def index(self, shape: tuple[int, ...]) -> Operand:
        """An integer index array (all zeros, so always in range)."""
        return Operand(f"np.zeros({shape!r}, dtype=np.int64)", shape)

    def axis(self, rank: int) -> int | None:
        r = self.rng
        if rank == 0:
            return r.choice((0, -1, None))
        return r.randint(-rank - 1, rank)

    # -- expression families ---------------------------------------------

    def binary_operator(self) -> str:
        a = self.operand()
        op = self.rng.choice(("+", "-", "*", "/", "//", "%", "**", "<", "<=", "==", "!=", ">=", ">"))
        return f"{a.text} {op} {self.near(a.shape).text}"

    def bool_operator(self) -> str:
        a = self.operand(dtype=self.rng.choice(("bool", "np.int64", "np.uint8")))
        b = self.near(a.shape)
        dt = self.rng.choice(("bool", "np.int64"))
        return f"{a.text} {self.rng.choice('&|^')} np.zeros({b.shape!r}, dtype={dt})"

    def scalar_operator(self) -> str:
        a = self.operand()
        s = self.rng.choice(("2", "2.5", "True", "1j", "np.float32(2)", "np.int8(3)"))
        op = self.rng.choice(("+", "-", "*", "/", "**", "//"))
        return self.rng.choice((f"{a.text} {op} {s}", f"{s} {op} {a.text}"))

    def unary(self) -> str:
        a = self.operand()
        return self.rng.choice((f"-{a.text}", f"+{a.text}", f"abs({a.text})", f"~{a.text}"))

    def matmul(self) -> str:
        a = self.operand(self.shape(self.rng.choice((1, 2, 2, 3))))
        r = self.rng
        if r.random() < 0.6 and a.shape:
            b = self.operand((a.shape[-1],) + self.shape(r.choice((0, 1))))
        else:
            b = self.operand()
        form = r.choice(("np.matmul({}, {})", "{} @ {}", "np.dot({}, {})", "{}.dot({})"))
        return form.format(a.text, b.text)

    def products(self) -> str:
        a, b = self.operand(), self.operand()
        fn = self.rng.choice(("tensordot", "inner", "outer", "vdot", "kron", "cross", "convolve", "correlate"))
        return f"np.{fn}({a.text}, {b.text})"

    def tensordot_axes(self) -> str:
        a, b = self.operand(), self.operand()
        axes = self.rng.choice(("0", "1", "2", "([0], [0])", "([-1], [0])", "([0, 1], [1, 0])"))
        return f"np.tensordot({a.text}, {b.text}, axes={axes})"

    def method_reduction(self) -> str:
        a = self.operand()
        name = self.rng.choice(
            ("sum", "mean", "max", "min", "prod", "std", "var", "argmax", "argmin", "any", "all", "cumsum", "cumprod")
        )
        keep = ", keepdims=True" if self.rng.random() < 0.3 and not name.startswith("cum") else ""
        return f"{a.text}.{name}(axis={self.axis(len(a.shape))}{keep})"

    def function_reduction(self) -> str:
        a = self.operand()
        name = self.rng.choice(
            (
                "sum", "mean", "median", "nansum", "nanmean", "nanmax", "argmax", "argsort",
                "sort", "diff", "flip", "cumsum", "squeeze", "expand_dims", "gradient",
                "count_nonzero", "linalg.norm", "unique", "ptp", "amax", "nanargmin",
            )
        )
        return f"np.{name}({a.text}, axis={self.axis(len(a.shape))})"

    def no_axis(self) -> str:
        a = self.operand()
        name = self.rng.choice(
            (
                "sum()", "mean()", "max()", "argmax()", "ravel()", "flatten()", "copy()",
                "astype(np.float32)", "astype(int)", "squeeze()", "conj()", "round()",
                "clip(0, 1)", "nonzero()", "diagonal()", "trace()", "swapaxes(0, -1)",
                "cumsum()", "argsort()", "T", "real", "imag", "tolist()",
            )
        )
        return f"{a.text}.{name}"

    def reshape(self) -> str:
        a = self.operand()
        total = int(np.prod(a.shape, dtype=int))
        target = self.rng.choice(
            ((-1,), (total,), (1, -1), (-1, 1), (2, -1), (-1, 3), (total, 1, 1), a.shape[::-1], (2, 2, -1), (-1, 2, 3))
        )
        form = self.rng.choice(("{}.reshape({!r})", "{}.reshape{!r}", "np.reshape({}, {!r})"))
        return form.format(a.text, target)

    def transpose(self) -> str:
        a = self.operand()
        rank = len(a.shape)
        perm = tuple(self.rng.sample(range(rank), rank))
        return self.rng.choice(
            (
                f"{a.text}.T",
                f"{a.text}.transpose()",
                f"np.transpose({a.text})",
                f"{a.text}.transpose({perm!r})",
                f"np.transpose({a.text}, {perm!r})",
                f"np.transpose({a.text}, axes={perm!r})",
                f"np.permute_dims({a.text})",
                f"np.permute_dims({a.text}, {perm!r})",
                f"np.permute_dims({a.text}, axes={perm!r})",
                f"np.swapaxes({a.text}, {self.axis(rank)}, {self.axis(rank)})",
                f"np.moveaxis({a.text}, {self.axis(rank)}, {self.axis(rank)})",
                f"{a.text}.mT",
                f"np.rollaxis({a.text}, {self.axis(rank)})",
            )
        )

    def getitem(self) -> str:
        r = self.rng
        a = self.operand(self.shape(r.choice((1, 2, 2, 3, 3))))
        items = []
        for d in a.shape[: r.randint(0, len(a.shape))]:
            if r.random() < 0.12:
                items.append("None")  # a new axis: consumes no dim
            items.append(
                r.choice(
                    (
                        str(r.randint(-d - 1, d)),
                        ":",
                        f"{r.randint(-d, d)}:",
                        f":{r.randint(-d - 1, d + 1)}",
                        f"::{r.choice((1, 2, -1, 3))}",
                        f"{r.randint(0, d)}:{r.randint(0, d + 2)}",
                        "...",
                        f"[0, {d - 1}]",
                        f"np.arange({r.randint(1, d)})",
                        f"np.ones({d}, dtype=bool)",
                        f"np.zeros(({r.choice(DIMS)}, 2), dtype=np.int64)",
                    )
                )
            )
        return f"{a.text}[{', '.join(items) or '...'}]"

    def join(self) -> str:
        a = self.operand()
        b = self.near(a.shape)
        name = self.rng.choice(("concatenate", "stack", "vstack", "hstack", "dstack", "column_stack"))
        axis = ""
        if name in ("concatenate", "stack") and self.rng.random() < 0.6:
            axis = f", axis={self.axis(len(a.shape))}"
        seq = self.rng.choice(("[{}, {}]", "({}, {})")).format(a.text, b.text)
        return f"np.{name}({seq}{axis})"

    def unary_ufunc(self) -> str:
        a = self.operand()
        name = self.rng.choice(
            (
                "sqrt", "exp", "abs", "sin", "negative", "floor", "isnan", "sign", "square",
                "log1p", "conj", "real", "imag", "angle", "reciprocal", "logical_not",
                "rint", "isfinite", "absolute", "tanh", "ceil", "cbrt",
            )
        )
        return f"np.{name}({a.text})"

    def binary_ufunc(self) -> str:
        a = self.operand()
        name = self.rng.choice(
            (
                "add", "maximum", "minimum", "hypot", "arctan2", "power", "logical_and",
                "equal", "subtract", "multiply", "true_divide", "floor_divide", "fmod",
                "less", "bitwise_and", "copysign",
            )
        )
        return f"np.{name}({a.text}, {self.near(a.shape).text})"

    def where(self) -> str:
        a = self.operand()
        cond = self.operand(self.near(a.shape).shape, "bool")
        return f"np.where({cond.text}, {a.text}, {self.near(a.shape).text})"

    def clip(self) -> str:
        a = self.operand()
        lo = self.rng.choice(("0", self.near(a.shape).text))
        hi = self.rng.choice(("1", self.near(a.shape).text))
        return f"np.clip({a.text}, {lo}, {hi})"

    def shape_helper(self) -> str:
        a = self.operand()
        name = self.rng.choice(
            (
                "atleast_1d", "atleast_2d", "atleast_3d", "ravel", "flatnonzero", "zeros_like",
                "ones_like", "copy", "ascontiguousarray", "asfortranarray", "asarray", "fliplr",
                "flipud", "trace", "diag", "tril", "triu", "diagonal", "argwhere", "nonzero",
                "squeeze", "transpose", "isnan", "array", "empty_like", "cov", "corrcoef",
            )
        )
        return f"np.{name}({a.text})"

    def tile_repeat(self) -> str:
        a = self.operand()
        if self.rng.random() < 0.5:
            reps = self.rng.choice(("2", "3", "(2,)", "(2, 3)", "(1, 2, 2)"))
            return f"np.{self.rng.choice(('tile', 'repeat'))}({a.text}, {reps})"
        return f"np.repeat({a.text}, {self.rng.randint(1, 3)}, axis={self.axis(len(a.shape))})"

    def pad(self) -> str:
        a = self.operand()
        rank = max(len(a.shape), 1)
        width = self.rng.choice(("1", "(1, 2)", repr(((1, 2),) * rank), "((0, 1), (2, 0))"))
        return f"np.pad({a.text}, {width})"

    def linalg(self) -> str:
        r = self.rng
        n = r.choice(DIMS)
        a = self.operand(r.choice(((n, n), (n, n), self.shape(2), self.shape(3), self.shape(1))), r.choice(("", "", "np.float32", "np.int64", "np.complex128")))
        name = r.choice(
            ("inv", "det", "cholesky", "eigvals", "pinv", "svd", "qr", "eigh", "slogdet", "matrix_rank", "eig", "eigvalsh", "norm", "cond", "matrix_transpose")
        )
        return f"np.linalg.{name}({a.text})"

    def linalg_solve(self) -> str:
        r = self.rng
        n = r.choice(DIMS)
        a = self.operand(r.choice(((n, n), (n, n), self.shape(2))), "")
        b = self.operand(r.choice(((n,), (n, r.choice(DIMS)), self.shape())), "")
        return f"np.linalg.{r.choice(('solve', 'lstsq'))}({a.text}, {b.text})"

    def einsum(self) -> str:
        sub = self.rng.choice(
            (
                "ij,jk->ik", "ij->ji", "ii->i", "ij,ij->", "i,j->ij", "...ij,...jk->...ik",
                "ijk->kji", "ij,j", "bij,bjk->bik", "ii", "ij->", "i,i", "ij,kj->ik", "...i->...",
            )
        )
        count = sub.split("->")[0].count(",") + 1
        ops = ", ".join(self.operand(dtype="").text for _ in range(count))
        return f"np.einsum({sub!r}, {ops})"

    def take(self) -> str:
        a = self.operand()
        idx = self.rng.choice(("0", "[0, 0]", "np.zeros(2, dtype=np.int64)", "[[0], [0]]"))
        axis = "" if self.rng.random() < 0.4 else f", axis={self.axis(len(a.shape))}"
        return f"np.take({a.text}, {idx}{axis})"

    def take_along_axis(self) -> str:
        a = self.operand()
        idx = self.index(self.near(a.shape).shape)
        return f"np.take_along_axis({a.text}, {idx.text}, axis={self.axis(len(a.shape))})"

    def expand_squeeze(self) -> str:
        a = self.operand()
        rank = len(a.shape)
        if self.rng.random() < 0.5:
            where = self.rng.choice((str(self.axis(rank)), "(0, 1)", "(0, -1)"))
            return f"np.expand_dims({a.text}, {where})"
        axis = "" if self.rng.random() < 0.4 else f", axis={self.axis(rank)}"
        return f"np.squeeze({a.text}{axis})"

    def broadcast_to(self) -> str:
        a = self.operand()
        target = self.rng.choice((self.shape(), self.shape(1) + a.shape, a.shape))
        return f"np.broadcast_to({a.text}, {target!r})"

    def fft(self) -> str:
        a = self.operand()
        name = self.rng.choice(("fft", "rfft", "ifft", "fft2", "rfft2", "fftn", "irfft", "ifft2", "rfftn"))
        extra = self.rng.choice(("", f", n={self.rng.randint(1, 7)}", f", axis={self.axis(len(a.shape))}"))
        if name.endswith(("2", "n")):
            extra = ""
        return f"np.fft.{name}({a.text}{extra})"

    def split(self) -> str:
        a = self.operand()
        axis = "" if self.rng.random() < 0.4 else f", axis={self.axis(len(a.shape))}"
        return f"np.{self.rng.choice(('split', 'array_split'))}({a.text}, {self.rng.choice((1, 2, 3))}{axis})"

    def roll_rot(self) -> str:
        a = self.operand()
        if self.rng.random() < 0.5:
            axis = self.rng.choice(("", f", axis={self.axis(len(a.shape))}"))
            return f"np.roll({a.text}, {self.rng.choice((1, 2, -1))}{axis})"
        axes = self.rng.choice(("", ", axes=(0, 1)", ", axes=(1, 0)", ", axes=(0, 0)"))
        return f"np.rot90({a.text}, {self.rng.choice((1, 2, 3))}{axes})"

    def builder(self) -> str:
        name = self.rng.choice(("eye", "identity", "arange", "linspace", "zeros", "ones", "full", "empty"))
        args = self.rng.choice(("3", "(2, 3)", "4, 5", "(3,), 1.5", "0", "5, dtype=np.float32", "(2, 2), 7", "3, 4, dtype=int"))
        return f"np.{name}({args})"

    def average(self) -> str:
        a = self.operand()
        w = self.operand(self.shape(1), "")
        return f"np.average({a.text}, axis={self.axis(len(a.shape))}, weights={w.text})"

    def edit(self) -> str:
        a = self.operand()
        name = self.rng.choice(("insert", "append", "delete"))
        where = self.rng.choice(("0", "[0, 1]"))
        value = "" if name == "delete" else ", 1.0"
        if name == "append":
            where, value = "1.0", ""
        axis = "" if self.rng.random() < 0.5 or name == "append" else f", axis={self.axis(len(a.shape))}"
        return f"np.{name}({a.text}, {where}{value}{axis})"

    def window(self) -> str:
        a = self.operand()
        win = self.rng.choice(("1", "2", "3", "(2, 2)"))
        axis = "" if self.rng.random() < 0.5 else f", axis={self.axis(len(a.shape))}"
        return f"np.lib.stride_tricks.sliding_window_view({a.text}, {win}{axis})"

    def lookup(self) -> str:
        a, b = self.operand(), self.operand()
        name = self.rng.choice(("searchsorted", "isin", "digitize", "percentile", "quantile", "trapezoid", "meshgrid"))
        return f"np.{name}({a.text}, {b.text})"

    def ufunc_method(self) -> str:
        a = self.operand()
        name = self.rng.choice(("add", "multiply", "maximum", "logical_and"))
        rank = len(a.shape)
        return self.rng.choice(
            (
                f"np.{name}.reduce({a.text}, axis={self.axis(rank)})",
                f"np.{name}.accumulate({a.text}, axis={self.axis(rank)})",
                f"np.{name}.outer({a.text}, {self.operand().text})",
            )
        )

    def all_any(self) -> str:
        a = self.operand()
        rank = len(a.shape)
        name = self.rng.choice(("all", "any"))
        axis = self.rng.choice(
            ("", f", axis={self.axis(rank)}", f", {self.axis(rank)}", ", axis=None", ", axis=(0, -1)")
        )
        keep = ", keepdims=True" if self.rng.random() < 0.3 else ""
        return f"np.{name}({a.text}{axis}{keep})"

    def rounding(self) -> str:
        a = self.operand()
        return self.rng.choice(
            (
                "np.round({})", "np.round({}, 2)", "np.round({}, decimals=1)", "np.around({})",
                "np.around({}, 1)", "{}.round()", "{}.round(2)", "{}.round(decimals=1)",
            )
        ).format(a.text)

    def divmod_builtin(self) -> str:
        a = self.operand()
        other = self.rng.choice((self.near(a.shape).text, "2", "2.5", "np.int8(3)"))
        return self.rng.choice((f"divmod({a.text}, {other})", f"divmod({other}, {a.text})"))

    def linalg_cross(self) -> str:
        r = self.rng

        def vectors(shape: tuple[int, ...]) -> tuple[int, ...]:
            # Mostly a 3 somewhere, since that is what cross needs.
            if shape and r.random() < 0.8:
                at = r.randrange(len(shape))
                return shape[:at] + (3,) + shape[at + 1 :]
            return shape

        a = self.operand(vectors(self.shape(r.choice((1, 1, 2, 2, 3)))))
        roll = r.random()
        if roll < 0.5:
            b = self.operand(a.shape)
        elif roll < 0.8:
            b = self.operand(vectors(self.shape(r.choice((1, 2)))))
        else:
            b = self.operand()
        axis = f", axis={self.axis(len(a.shape))}" if r.random() < 0.5 else ""
        return f"np.linalg.cross({a.text}, {b.text}{axis})"

    def array_method(self) -> str:
        """ndarray methods, which have hooks of their own apart from `np.*`."""
        r = self.rng
        a = self.operand()
        rank = len(a.shape)
        ax = self.axis(rank)
        perm = tuple(r.sample(range(rank), rank))
        keep = ", keepdims=True" if r.random() < 0.3 else ""
        reduce = r.choice(("sum", "prod", "mean", "min", "max", "std", "var", "any", "all"))
        arg = r.choice(("argmax", "argmin", "argsort"))
        return r.choice(
            (
                f"{a.text}.take(0, axis={ax})",
                f"{a.text}.take([0, 0, 0], axis={ax})",
                f"{a.text}.take([0, 0])",
                f"{a.text}.take({self.index(self.shape(r.choice((1, 2)))).text}, {ax})",
                f"{a.text}.repeat(2)",
                f"{a.text}.repeat({r.randint(1, 3)}, axis={ax})",
                f"{a.text}.repeat(3, {ax})",
                f"{a.text}.squeeze(axis={ax})",
                f"{a.text}.squeeze({ax})",
                f"{a.text}.transpose(*{perm!r})",
                f"{a.text}.transpose({perm!r})",
                f"{a.text}.swapaxes({self.axis(rank)}, {self.axis(rank)})",
                f"{a.text}.ravel()",
                f"{a.text}.flatten()",
                f"{a.text}.cumsum({ax})",
                f"{a.text}.cumprod(axis={ax})",
                f"{a.text}.{reduce}({ax}{keep})",
                f"{a.text}.{reduce}(axis=(0, -1){keep})",
                f"{a.text}.{reduce}(axis=None{keep})",
                f"{a.text}.{arg}({ax})",
                f"{a.text}.{arg}(axis=None)",
                f"{a.text}.argpartition(0, axis={ax})",
                f"{a.text}.clip({self.near(a.shape).text}, 1)",
                f"{a.text}.clip(min=0)",
                f"{a.text}.dot({self.near(a.shape).text})",
                f"{a.text}.__divmod__({self.near(a.shape).text})",
                f"{a.text}.astype(np.complex64)",
                f"{a.text}.astype(bool)",
            )
        )

    def rng_method(self) -> str:
        r = self.rng
        a = self.operand()
        size = f"{a.text}.shape"
        source = r.choice(("np.random.default_rng(0)", "np.random.RandomState(0)"))
        if source.endswith("RandomState(0)") and r.random() < 0.3:
            return f"{source}.{r.choice(('rand', 'randn'))}(*{size})"
        return source + r.choice(
            (
                f".normal(size={size})",
                f".normal(0.0, 1.0, {size})",
                f".random({size})",
                f".random(size={size})",
                f".standard_normal({size})",
                f".uniform(size={size})",
                f".uniform(0, 1, {size})",
                f".integers(0, 5, size={size})" if "default_rng" in source else f".randint(0, 5, size={size})",
                f".choice(5, size={size})",
                f".choice({a.text}.ravel(), {size})",
                f".permutation({a.text})",
                ".permutation(4)",
                ".normal(size=3)",
                ".normal()",
            )
        )

    def chain(self) -> str:
        a = self.operand()
        b = self.operand(a.shape)
        tail = self.rng.choice(("", "[::2]", "[None]", ".mean()", ".T", ".sum(axis=0)", ".reshape(-1, 1)"))
        op = self.rng.choice("+-*")
        return f"({a.text} {op} {b.text}).reshape(-1){tail}"

    FAMILIES: tuple[Callable[..., str], ...] = (
        binary_operator, binary_operator, bool_operator, scalar_operator, unary,
        matmul, matmul, products, tensordot_axes, method_reduction, method_reduction,
        function_reduction, function_reduction, no_axis, reshape, reshape, transpose,
        transpose, getitem, getitem, getitem, join, join, unary_ufunc, binary_ufunc,
        where, clip, shape_helper, shape_helper, tile_repeat, pad, linalg, linalg_solve,
        einsum, take, take_along_axis, expand_squeeze, broadcast_to, fft, split,
        roll_rot, builder, average, edit, window, lookup, ufunc_method, chain,
        all_any, rounding, divmod_builtin, linalg_cross, array_method, array_method,
        rng_method,
    )

    def expression(self) -> str:
        family_fn = self.rng.choice(self.FAMILIES)
        # Single choice (same rng draw as before): the expr stream for a
        # seed is unchanged. No extra draw for the family record.
        self.current_family = family_fn.__name__
        return family_fn(self)


class SymGen(Gen):
    """Expressions over the parameters of a generic function.

    Operands are parameter names with symbolic shapes; `self.params`
    collects the ones the current expression used.
    """

    def __init__(self, rng: random.Random) -> None:
        super().__init__(rng)
        self.params: list[tuple[str, str, tuple[object, ...]]] = []
        self.helpers: list[Helper] = []
        self.helper_count = 0

    def shape(self, rank: int | None = None) -> tuple[int, ...]:  # type: ignore[override]
        r = self.rng
        if rank is None:
            rank = r.choice((1, 1, 2, 2, 2, 3))
        pool: tuple[object, ...] = ("N", "N", "M", "M", "K", "P", 2, 3)
        if r.random() < 0.3:
            pool += SYMBOL_FORMULAS
        return cast("tuple[int, ...]", tuple(r.choice(pool) for _ in range(rank)))

    def operand(self, shape=None, dtype=None) -> Operand:  # type: ignore[override]
        if shape is None:
            shape = self.shape()
        head = {
            None: self.rng.choice(("F64", "F64", "F64", "F32", "I64", "U8")),
            "": "F64",
            "bool": "Bool",
            "np.int64": "I64",
        }.get(dtype, "F64")
        name = f"a{len(self.params)}"
        self.params.append((name, head, tuple(shape)))
        return Operand(name, tuple(shape))  # type: ignore[arg-type]

    def near(self, shape) -> Operand:  # type: ignore[override]
        r = self.rng
        roll = r.random()
        if roll < 0.5:
            return self.operand(shape)
        if roll < 0.7 and shape:
            return self.operand(shape[r.randrange(len(shape)) :])
        return self.operand()

    # Families whose text depends on concrete sizes are rebuilt here.

    def reshape(self) -> str:
        a = self.operand()
        target = self.rng.choice(("-1", "(-1,)", "(1, -1)", "(-1, 1)", f"{a.text}.shape[::-1]", "(2, -1)"))
        return f"{a.text}.reshape({target})"

    def reshape_symbolic(self) -> str:
        """Reshape to a target spelled from the operands' own sizes.

        The total has to be conserved as a product of symbolic dims,
        which is the one place numty compares factors (`2*N` against
        `N`). Some targets conserve it, some cannot, and some only do
        for another operand's sizes.
        """
        r = self.rng
        a = self.operand(self.shape(r.choice((1, 2, 2, 3, 3))))
        rank = len(a.shape)
        x = a.text

        def dim(i: int) -> str:
            return f"{x}.shape[{i}]"

        every = [dim(i) for i in range(rank)]
        product = " * ".join(every)
        targets = [
            f"({product},)",
            f"({product}, 1)",
            f"(1, {product})",
            f"({', '.join(reversed(every))},)",
            f"({', '.join(r.sample(every, rank))},)",
            f"({dim(0)}, -1)",
            f"(-1, {dim(-1)})",
            f"(-1, {dim(0)})",
            f"(*{x}.shape, 1)",
            f"(1, *{x}.shape)",
            f"{x}.shape + (1,)",
            f"(-1, *{x}.shape[1:])",
            f"(*{x}.shape[:-1], -1)",
            f"(2, {product})",  # twice the total
            f"(2 * {dim(0)}, -1)",
            f"({dim(0)} + 1, -1)",
            f"({dim(0)}, {dim(0)})",
            f"({dim(0)} // 2, -1)",
            f"(2, -1, {dim(-1)})",
            "(-1, 2, 3)",
            f"({dim(0)}, 2, -1)",
        ]
        if rank >= 2:
            targets += [
                f"({dim(0)} * {dim(1)}, -1)",
                f"({dim(0)}, {' * '.join(every[1:])})",
                f"(-1, {dim(0)} * {dim(1)})",
                f"({dim(1)}, -1, {dim(0)})",
                f"({dim(0)} * {dim(1)},)",
            ]
        if r.random() < 0.3:
            b = self.near(a.shape)
            targets = [
                f"{b.text}.shape",
                f"({b.text}.shape[0], -1)",
                f"(-1, {b.text}.shape[-1])",
                f"(*{b.text}.shape, -1)",
                f"({b.text}.size,)",
                f"({x}.size // {b.text}.shape[0], {b.text}.shape[0])",
            ]
        target = r.choice(targets)
        tail = r.choice(("", "", "", ".T", ".sum(axis=0)", ".reshape(-1)", f".reshape({x}.shape)"))
        if target.startswith("(") and r.random() < 0.25:
            return f"{x}.reshape(*{target}){tail}"
        return r.choice(("{}.reshape({}){}", "np.reshape({}, {}){}")).format(x, target, tail)

    def getitem(self) -> str:
        a = self.operand(self.shape(self.rng.choice((1, 2, 2, 3))))
        items = [
            self.rng.choice(("0", ":", "1:", ":-1", "::2", "None", "...", "-1", "::-1", "1:-1"))
            for _ in a.shape[: self.rng.randint(0, len(a.shape))]
        ]
        return f"{a.text}[{', '.join(items) or '...'}]"

    def matmul(self) -> str:
        a = self.operand(self.shape(self.rng.choice((1, 2, 2, 3))))
        if self.rng.random() < 0.6:
            b = self.operand((a.shape[-1],) + self.shape(self.rng.choice((0, 1))))
        else:
            b = self.operand()
        form = self.rng.choice(("np.matmul({}, {})", "{} @ {}", "np.dot({}, {})"))
        return form.format(a.text, b.text)

    def builder_symbolic(self) -> str:
        """Builders with sizes spelled as arithmetic (issue #186).

        `np.zeros((a.shape[0] * a.shape[1], a.shape[0] + 1))` names
        exact sizes; mypy types them `int`, so the plugin folds them
        with the dim algebra. A product the algebra cannot spell (a sum
        times a sum) is an unknown size, which still matches.
        """
        r = self.rng
        a = self.operand(self.shape(r.choice((1, 2, 2, 3, 3))))
        rank = len(a.shape)
        x = a.text

        def dim(i: int) -> str:
            return f"{x}.shape[{i}]"

        every = [dim(i) for i in range(rank)]
        product = " * ".join(every)
        shapes = [
            f"({product},)",
            f"({dim(0)},)",
            f"(2 * {dim(0)},)",
            f"({dim(0)} + 1,)",
            f"({dim(0)} * {dim(0)},)",
            f"({dim(0)} // 2 + 1,)",
            f"({dim(0)}, {dim(-1)})",
            f"({dim(0)} * {dim(-1)}, {dim(0)} + 1)",
            f"({dim(0)} + 1, {dim(-1)} * 2)",
        ]
        if rank >= 2:
            shapes += [
                f"({dim(0)} * {dim(1)},)",
                f"({dim(0)} * {dim(1)}, {dim(0)} + 1)",
                f"({dim(0)}, {' * '.join(every[1:])})",
                f"(({dim(0)} + 1) * ({dim(1)} + 1),)",
            ]
        singles = [
            f"2 * {dim(0)}",
            f"{dim(0)} + 1",
            product,
        ]
        if rank >= 2:
            singles.append(f"{dim(0)} * {dim(1)}")
        kind = r.choice(("zeros", "ones", "empty", "full", "rng", "rng"))
        if kind in ("zeros", "ones", "empty"):
            shape = r.choice(singles if r.random() < 0.3 else shapes)
            return f"np.{kind}({shape})"
        if kind == "full":
            shape = r.choice(shapes + singles)
            return f"np.full({shape}, 1.5)"
        shape = r.choice(shapes + singles)
        src = "np.random.default_rng(0)"
        op = r.choice(("standard_normal", "random", "normal", "integers"))
        if op == "integers":
            return f"{src}.integers(0, 5, size={shape})"
        if op == "normal":
            return f"{src}.normal(size={shape})"
        return f"{src}.{op}(size={shape})"

    def chain(self) -> str:
        a = self.operand()
        b = self.operand(a.shape)
        tail = self.rng.choice(("", "[::2]", "[None]", ".T", ".reshape(-1, 1)"))
        return f"({a.text} + {b.text}).reshape(-1){tail}"

    def pad(self) -> str:
        a = self.operand()
        return f"np.pad({a.text}, {self.rng.choice(('1', '(1, 2)'))})"

    def bool_operator(self) -> str:
        r = self.rng
        a = self.operand(self.shape(r.choice((1, 2, 2, 3))), r.choice(("bool", "np.int64")))
        if r.random() < 0.8:
            b = self.operand(a.shape, r.choice(("bool", "np.int64")))
        else:
            b = self.operand(self.shape(), r.choice(("bool", "np.int64")))
        return f"{a.text} {r.choice('&|^')} {b.text}"

    def broadcast_to(self) -> str:
        r = self.rng
        a = self.operand()
        roll = r.random()
        if roll < 0.35:
            target = f"{a.text}.shape"
        elif roll < 0.55:
            b = self.operand()
            target = f"{b.text}.shape"
        elif roll < 0.75:
            target = f"(1, *{a.text}.shape)"
        else:
            target = r.choice(("1", "(1,)", "(2,)", "(1, 2)", "(2, 2)"))
        return f"np.broadcast_to({a.text}, {target})"

    def linalg(self) -> str:
        r = self.rng
        if r.random() < 0.6:
            n = r.choice(("N", "M", "K"))
            shape = (n, n)
        else:
            shape = self.shape(r.choice((2, 3)))
        a = self.operand(shape, "")
        name = r.choice(
            ("inv", "det", "cholesky", "eigvals", "svd", "qr", "eigh",
             "slogdet", "matrix_rank", "eig", "eigvalsh", "norm", "cond",
             "matrix_transpose")
        )
        return f"np.linalg.{name}({a.text})"

    def linalg_solve(self) -> str:
        r = self.rng
        s = r.choice(("N", "M", "K", "P"))
        fn = r.choice(("solve", "lstsq"))
        if r.random() < 0.2:
            m = r.choice(("N", "M"))
            a = self.operand((m, s, s), "")
            # numpy >=2 treats (m, s) as a stack of matrices, not a
            # vector stack: a batched RHS must be (m, s, 1) to solve.
            b = self.operand((m, s, 1), "")
            return f"np.linalg.{fn}({a.text}, {b.text})"
        a = self.operand((s, s), "")
        if r.random() < 0.5:
            b = self.operand((s,), "")
        else:
            b = self.operand((s, r.choice(("N", "M", "K", "P"))), "")
        return f"np.linalg.{fn}({a.text}, {b.text})"

    def lookup(self) -> str:
        r = self.rng
        name = r.choice(("searchsorted", "isin", "percentile", "quantile"))
        if name == "searchsorted":
            a = self.operand(self.shape(1))
            b = self.operand()
        elif name in ("percentile", "quantile"):
            a = self.operand()
            b = self.operand(self.shape(1))
        else:
            a = self.operand()
            b = self.operand()
        return f"np.{name}({a.text}, {b.text})"

    # -- *Ts operands (issue #185) ----------------------------------------
    # A variadic prefix, e.g. `F64[*Ts, N, M]`, instantiated at runtime
    # with prefixes of rank 0, 1 and 2 (see TS_PREFIXES). The marker lives
    # in shape tuples as "*Ts" and is spelled `Unpack[Ts]` in annotations.

    def ts_shape(self, rank_suffix: int | None = None) -> tuple[object, ...]:
        r = self.rng
        if rank_suffix is None:
            rank_suffix = r.choice((1, 1, 2, 2, 3))
        suffix = self.shape(rank_suffix)
        return (TS_DIM,) + tuple(suffix)  # type: ignore[arg-type]

    def ts_operand(self, shape: tuple[object, ...] | None = None, dtype: str | None = None) -> Operand:
        if shape is None:
            shape = self.ts_shape()
        head = {
            None: self.rng.choice(("F64", "F64", "F64", "F32", "I64", "U8")),
            "": "F64",
            "bool": "Bool",
            "np.int64": "I64",
        }.get(dtype, "F64")
        name = f"a{len(self.params)}"
        self.params.append((name, head, tuple(shape)))
        return Operand(name, tuple(shape))  # type: ignore[arg-type]

    def ts_near(self, shape: tuple[object, ...]) -> Operand:
        """An operand that often fits next to a *Ts shape, sometimes not."""
        r = self.rng
        roll = r.random()
        if roll < 0.45:
            return self.ts_operand(tuple(shape))
        if roll < 0.60 and len(shape) > 1:
            lst = list(shape)
            idx = r.randrange(1, len(lst))
            pool: tuple[object, ...] = ("N", "M", "K", "P", 2, 3) + SYMBOL_FORMULAS
            lst[idx] = r.choice(pool)
            return self.ts_operand(tuple(lst))
        if roll < 0.75:
            lst = list(shape)
            choice = r.choice(("drop_ts", "drop_last", "add"))
            if choice == "drop_ts" and TS_DIM in lst:
                lst = [d for d in lst if d != TS_DIM]
                if not lst:
                    lst = [2]
                # A fixed-rank shape: use the plain operand path.
                return self.operand(tuple(lst))
            if choice == "drop_last" and len(lst) > 1:
                return self.ts_operand(tuple(lst[:-1]))
            pool = ("N", "M", 2, 3)
            return self.ts_operand(tuple(lst) + (r.choice(pool),))
        return self.ts_operand()

    def ts_binary(self) -> str:
        """Same-shape ops threading a *Ts item through (issue #185)."""
        a = self.ts_operand()
        b = self.ts_near(tuple(a.shape))  # type: ignore[arg-type]
        if self.rng.random() < 0.3:
            name = self.rng.choice(("add", "subtract", "multiply", "maximum", "minimum"))
            return f"np.{name}({a.text}, {b.text})"
        op = self.rng.choice(("+", "-", "*", "/", "//", "%", "**", "<", "<=", "==", "!=", ">=", ">"))
        return f"{a.text} {op} {b.text}"

    def ts_matmul(self) -> str:
        """Matmul over (*Ts, N, M) pairs, plus inner and batch conflicts."""
        r = self.rng
        if r.random() < 0.6:
            s0: object = r.choice(("N", "M", "K", "P", 2, 3) + SYMBOL_FORMULAS)
            s1: object = r.choice(("N", "M", "K", "P", 2, 3) + SYMBOL_FORMULAS)
            s2: object = r.choice(("N", "M", "K", "P", 2, 3) + SYMBOL_FORMULAS)
            a_suffix = (s0, s1)
            b_suffix = (s1, s2)
        else:
            a_suffix = tuple(self.shape(2))
            b_suffix = tuple(self.shape(2))
        if r.random() < 0.15:
            lead_a: object = r.choice((2, 3, "N"))
            lead_b: object = r.choice((3, 2, "M"))
            a_shape: tuple[object, ...] = (lead_a, TS_DIM) + tuple(a_suffix)
            b_shape: tuple[object, ...] = (lead_b, TS_DIM) + tuple(b_suffix)
        else:
            a_shape = (TS_DIM,) + tuple(a_suffix)
            b_shape = (TS_DIM,) + tuple(b_suffix)
        if r.random() < 0.10:
            # Wrong rank: drop the unpack or a trailing dim.
            lst = list(b_shape)
            if TS_DIM in lst and r.random() < 0.5:
                lst = [d for d in lst if d != TS_DIM] or [2]
                b = self.operand(tuple(lst))
                a = self.ts_operand(a_shape)
                form = r.choice(("np.matmul({}, {})", "{} @ {}", "np.dot({}, {})"))
                return form.format(a.text, b.text)
            if len(lst) > 1:
                b_shape = tuple(lst[:-1])
        a = self.ts_operand(a_shape)
        b = self.ts_operand(b_shape)
        form = r.choice(("np.matmul({}, {})", "{} @ {}", "np.dot({}, {})"))
        return form.format(a.text, b.text)

    def ts_reshape(self) -> str:
        """Reshape over *Ts (issue #185): exact, bailing and error paths."""
        r = self.rng
        roll = r.random()
        if roll < 0.15:
            # Provable total-size clash on ints: 6 * prod(*Ts) vs 8.
            a = self.ts_operand((TS_DIM, 2, 3))
            target = r.choice(("((2, 4))", "(2, 4)"))
            form = r.choice(("{}.reshape({})", "np.reshape({}, {})"))
            return form.format(a.text, target)
        if roll < 0.30:
            # Exact for the empty prefix: (*Ts, 2, 3) -> (6,) holds at
            # rank 0 and raises otherwise (allowed), so the claim is
            # checked where it holds.
            a = self.ts_operand((TS_DIM, 2, 3))
            target = r.choice(("6", "(6,)", "(2, 3)"))
            form = r.choice(("{}.reshape({})", "np.reshape({}, {})"))
            return form.format(a.text, target)
        a = self.ts_operand()
        if r.random() < 0.5:
            target = r.choice(("-1", "(-1,)", "(1, -1)", "(-1, 1)", "(2, -1)"))
            return f"{a.text}.reshape({target})"
        target = f"{a.text}.shape[::-1]"
        return f"{a.text}.reshape({target})"

    def ts_reduction(self) -> str:
        """Reductions over *Ts: provable suffix axes, bails and errors."""
        r = self.rng
        a = self.ts_operand()
        if r.random() < 0.5:
            name = r.choice(("sum", "mean", "max", "min", "prod", "std", "var"))
            axis = r.choice((-1, -2, -1, 0, "(0, -1)", "(-1, -1)", "None"))
            keep = ", keepdims=True" if r.random() < 0.3 else ""
            return f"{a.text}.{name}(axis={axis}{keep})"
        name = r.choice(("sum", "mean", "max", "argmax", "sort", "cumsum"))
        axis = r.choice((-1, -2, 0))
        return f"np.{name}({a.text}, axis={axis})"

    def call(self) -> str:
        """A call to a generic function whose signature carries formulas.

        `def g(x0: F64[N, "N+1"]) -> F64["2*N"]` is a contract: bind the
        names from the argument shapes, then the result has the shape
        the return spells. numty has to work that out at the call site
        in the caller's own names; `Helper.__call__` does it on the
        actual arrays.
        """
        r = self.rng
        words = r.random() < 0.3
        names = r.sample(("N", "M", "K"), r.choice((1, 2, 2, 3)))

        def formula(pool: tuple[str, ...]) -> str:
            a = r.choice(names)
            return r.choice(pool).format(a=a, b=r.choice(names))

        def formal() -> object:
            roll = r.random()
            if roll < 0.65:
                return r.choice(names)
            if roll < 0.8:
                return r.choice((2, 3))
            return formula(("{a}+1", "{a}+2", "2*{a}"))

        formals = [
            (r.choice(tuple(SYMBOL_HEADS)), tuple(formal() for _ in range(r.choice((1, 2, 2, 3)))))
            for _ in range(r.choice((1, 1, 2)))
        ]
        bound = sorted({n for _, dims in formals for d in dims for n in _names_in(d)})
        if not bound:
            head, dims = formals[0]
            formals[0] = (head, (names[0],) + dims[1:])
            bound = [names[0]]
        names = bound

        def returned() -> object:
            roll = r.random()
            if roll < 0.4:
                return r.choice(names)
            if roll < 0.5:
                return r.choice((2, 3))
            return formula(("{a}+1", "{a}+2", "2*{a}", "{a}+{b}", "2*{a}+1", "{a}+{b}+1"))

        ret = (r.choice(tuple(SYMBOL_HEADS)), tuple(returned() for _ in range(r.choice((1, 2, 2, 3)))))
        # Undeclared words only where every spec can be one string: a
        # lone `F64["rows"]` is a forward reference to a declared name.
        specs = [dims for _, dims in formals] + [ret[1]]
        if words and not any(len(dims) == 1 and str(dims[0]).isidentifier() for dims in specs):
            word = {"N": "rows", "M": "cols", "K": "depth"}

            def reword(dims: tuple[object, ...]) -> tuple[object, ...]:
                return tuple(
                    re.sub(r"[A-Z]", lambda m: word[m[0]], d) if isinstance(d, str) else d
                    for d in dims
                )

            formals = [(head, reword(dims)) for head, dims in formals]
            ret = (ret[0], reword(ret[1]))
            names = [word[n] for n in names]
        else:
            words = False
        kind = r.choice(Helper.KINDS) if r.random() < 0.5 else "function"
        helper = Helper(f"g{self.helper_count}", formals, ret, words, kind)
        self.helper_count += 1
        self.helpers.append(helper)

        # What each of the helper's names stands for in the caller.
        pool: tuple[object, ...] = ("N", "N", "M", "M", "K", "P", 2, 3) + SYMBOL_FORMULAS
        binding = {n: r.choice(pool) for n in names}
        args = []
        for head, dims in formals:
            shape = [_substitute(d, binding) for d in dims]
            roll = r.random()
            if roll < 0.1:
                shape[r.randrange(len(shape))] = r.choice(pool)  # a conflict, or a wrong literal
            elif roll < 0.15:
                shape = shape[1:] or shape + shape  # wrong rank
            if r.random() < 0.08:
                head = r.choice(tuple(SYMBOL_HEADS))  # maybe the wrong dtype
            name = f"a{len(self.params)}"
            self.params.append((name, head, tuple(shape)))
            roll = r.random()
            if roll < 0.7:
                args.append(name)
            elif roll < 0.8:
                args.append(f"np.concatenate([{name}, {name}])")
            elif roll < 0.9:
                args.append(f"np.pad({name}, 1)")
            else:
                args.append(f"{name}.T")
        tail = r.choice(("", "", "", ".T", ".sum(axis=0)", "[::-1]"))
        return f"{helper.call(', '.join(args))}{tail}"

    def ts_call(self) -> str:
        """A call to a generic function whose signature carries *Ts.

        Like `call`, but the formals thread a shared `Unpack[Ts]` segment
        (at the front, sometimes after a leading dim) and the return keeps
        or drops it. `Helper.__call__` binds the variadic segment from
        the actual arrays.
        """
        r = self.rng
        names = r.sample(("N", "M", "K"), r.choice((1, 2, 2, 3)))

        def formula(pool: tuple[str, ...]) -> str:
            a = r.choice(names)
            return r.choice(pool).format(a=a, b=r.choice(names))

        def formal_suffix() -> object:
            roll = r.random()
            if roll < 0.65:
                return r.choice(names)
            if roll < 0.8:
                return r.choice((2, 3))
            return formula(("{a}+1", "{a}+2", "2*{a}"))

        formals: list[tuple[str, tuple[object, ...]]] = []
        for _ in range(r.choice((1, 1, 2))):
            suffix = tuple(formal_suffix() for _ in range(r.choice((1, 2, 2))))
            if r.random() < 0.2:
                lead: object = r.choice((2, 3, "N", "M"))
                dims: tuple[object, ...] = (lead, TS_DIM) + suffix
            else:
                dims = (TS_DIM,) + suffix
            formals.append((r.choice(tuple(SYMBOL_HEADS)), dims))
        bound = sorted({n for _, dims in formals for d in dims for n in _names_in(d) if n != "Ts"})
        if not bound:
            head, dims = formals[0]
            # dims starts with Ts (or lead+Ts); append a name to the suffix.
            if TS_DIM in dims:
                idx = list(dims).index(TS_DIM)
                dims = tuple(list(dims[: idx + 1]) + [names[0]] + list(dims[idx + 1 :]))
            else:
                dims = (names[0],) + dims[1:]
            formals[0] = (head, dims)
            bound = [names[0]]
        names = bound

        def returned() -> object:
            roll = r.random()
            if roll < 0.4:
                return r.choice(names)
            if roll < 0.5:
                return r.choice((2, 3))
            return formula(("{a}+1", "{a}+2", "2*{a}", "{a}+{b}"))

        ret_suffix = tuple(returned() for _ in range(r.choice((1, 2))))
        if r.random() < 0.8:
            ret_dims: tuple[object, ...] = (TS_DIM,) + ret_suffix
        else:
            ret_dims = ret_suffix
        ret = (r.choice(tuple(SYMBOL_HEADS)), ret_dims)
        kind = r.choice(Helper.KINDS) if r.random() < 0.5 else "function"
        helper = Helper(f"g{self.helper_count}", formals, ret, False, kind)
        self.helper_count += 1
        self.helpers.append(helper)

        pool: tuple[object, ...] = ("N", "N", "M", "M", "K", "P", 2, 3) + SYMBOL_FORMULAS
        binding = {n: r.choice(pool) for n in names}
        args = []
        for head, dims in formals:
            shape: list[object] = []
            for d in dims:
                if d == TS_DIM:
                    shape.append(TS_DIM)
                else:
                    shape.append(_substitute(d, binding))
            roll = r.random()
            if roll < 0.10:
                idxs = [i for i, d in enumerate(shape) if d != TS_DIM]
                if idxs:
                    shape[r.choice(idxs)] = r.choice(pool)
            elif roll < 0.15:
                if TS_DIM in shape and r.random() < 0.5:
                    shape = [d for d in shape if d != TS_DIM] or [2]
                elif len(shape) > 1:
                    shape = shape[:-1]
                else:
                    shape = shape + shape
            if r.random() < 0.08:
                head = r.choice(tuple(SYMBOL_HEADS))
            name = f"a{len(self.params)}"
            self.params.append((name, head, tuple(shape)))
            roll = r.random()
            if roll < 0.7:
                args.append(name)
            elif roll < 0.8:
                args.append(f"np.concatenate([{name}, {name}])")
            elif roll < 0.9:
                args.append(f"np.pad({name}, 1)")
            else:
                # `.T` on a *Ts operand bails to Any (transpose cannot
                # spell a reversed *Ts), and an Any actual would leave
                # formula returns as the caller's own names (wrong exact,
                # not unknown). Keep the arg exact instead (issue #185).
                args.append(name)
        # No `.T` tail: `.T` on a *Ts result keeps the input shape in the
        # stubs (wrong exact), which the oracle would flag. `.sum(axis=0)`
        # bails to Any (safe) and `[::-1]` preserves the shape.
        tail = r.choice(("", "", "", "", ".sum(axis=0)", "[::-1]"))
        return f"{helper.call(', '.join(args))}{tail}"

    FAMILIES = (
        Gen.binary_operator, Gen.binary_operator, Gen.scalar_operator, Gen.unary,
        matmul, matmul, Gen.method_reduction, Gen.method_reduction,
        Gen.function_reduction, Gen.no_axis, reshape, reshape, Gen.transpose,
        getitem, getitem, Gen.join, Gen.join, Gen.unary_ufunc, Gen.binary_ufunc,
        Gen.where, Gen.clip, Gen.shape_helper, Gen.tile_repeat, pad,
        Gen.expand_squeeze, Gen.fft, Gen.roll_rot, Gen.take, Gen.products,
        Gen.einsum, Gen.window, chain, Gen.split, Gen.edit,
        Gen.all_any, Gen.rounding, Gen.divmod_builtin, Gen.linalg_cross,
        Gen.array_method, Gen.array_method, Gen.rng_method,
        reshape_symbolic, reshape_symbolic, reshape_symbolic,
        builder_symbolic, builder_symbolic, builder_symbolic,
        call, call, call, call,
        ts_binary, ts_binary, ts_matmul, ts_matmul, ts_reshape, ts_reshape,
        ts_reduction, ts_reduction, ts_call, ts_call,
    )

    def case(self) -> "Case":
        self.params = []
        self.helpers = []
        expr = self.expression()
        return Case(expr, list(self.params), list(self.helpers), family=self.current_family)


def _names_in(dim: object) -> list[str]:
    return re.findall(r"[A-Za-z_]\w*", dim) if isinstance(dim, str) else []


def _substitute(dim: object, binding: dict[str, object]) -> object:
    """A helper's dim with the caller's dims put in for its names."""
    if dim == TS_DIM:
        return TS_DIM
    if not isinstance(dim, str):
        return dim
    if dim in binding:
        return binding[dim]
    text = re.sub(r"[A-Za-z_]\w*", lambda m: f"({binding[m[0]]})", dim)
    return text if _names_in(text) else cast(int, eval(text, {"__builtins__": {}}))


def _value(dim: object, env: dict[str, int]) -> int:
    if isinstance(dim, int):
        return dim
    return cast(int, eval(cast(str, dim), {"__builtins__": {}}, dict(env)))


@dataclass
class Helper:
    """A generic function known only by its signature, and what it means.

    `words` spells the dims jaxtyping-style in one string of undeclared
    names (`F64["rows cols+1"]`); otherwise they are declared TypeVars
    and quoted formulas (`F64[N, "N+1"]`).
    """

    name: str
    formals: list[tuple[str, tuple[object, ...]]]
    ret: tuple[str, tuple[object, ...]]
    words: bool = False
    # How it is declared and called: a plain function, or on a class as
    # an instance method, a static method, a class method or `__call__`.
    kind: str = "function"

    KINDS = ("function", "method", "static", "class", "callable")

    @property
    def owner(self) -> str:
        return f"C{self.name[1:]}"

    def definition(self) -> list[str]:
        """The source lines that declare it; the body never runs."""
        if self.kind == "function":
            return [self.signature(), "    raise NotImplementedError"]
        decorator = {"static": ["    @staticmethod"], "class": ["    @classmethod"]}
        receiver = {"static": "", "class": "cls, "}.get(self.kind, "self, ")
        name = "__call__" if self.kind == "callable" else self.name
        head = self.signature().replace(f"def {self.name}(", f"def {name}({receiver}", 1)
        return [
            f"class {self.owner}:",
            *decorator.get(self.kind, []),
            f"    {head}",
            "        raise NotImplementedError",
        ]

    def call(self, args: str) -> str:
        return {
            "function": f"{self.name}({args})",
            "method": f"{self.owner}().{self.name}({args})",
            "static": f"{self.owner}.{self.name}({args})",
            "class": f"{self.owner}.{self.name}({args})",
            "callable": f"{self.owner}()({args})",
        }[self.kind]

    def bind(self, ns: dict[str, object]) -> None:
        """Put what `call` names into a runtime namespace."""
        if self.kind == "function":
            ns[self.name] = self
            return
        holder = type(self.owner, (), {self.name: staticmethod(self), "__call__": lambda _s, *a: self(*a)})
        ns[self.owner] = holder

    def _spec(self, head: str, dims: tuple[object, ...]) -> str:
        if self.words:
            return f'{head}["{" ".join(str(d) for d in dims)}"]'
        return f"{head}[{_spell_dims(dims)}]"

    def signature(self) -> str:
        params = ", ".join(f"x{i}: {self._spec(h, d)}" for i, (h, d) in enumerate(self.formals))
        return f"def {self.name}({params}) -> {self._spec(*self.ret)}:"

    def __call__(self, *arrays: object) -> np.ndarray:
        if len(arrays) != len(self.formals):
            raise TypeError("wrong number of arguments")
        has_ts = any(TS_DIM in dims for _, dims in self.formals) or TS_DIM in self.ret[1]
        if not has_ts:
            env: dict[str, int] = {}
            formulas: list[tuple[str, int]] = []
            for (head, dims), array in zip(self.formals, arrays):
                if not isinstance(array, np.ndarray) or str(array.dtype) != SYMBOL_HEADS[head]:
                    raise TypeError(f"expected a {SYMBOL_HEADS[head]} array")
                if array.ndim != len(dims):
                    raise ValueError(f"expected rank {len(dims)}, got {array.ndim}")
                for dim, size in zip(dims, array.shape):
                    if isinstance(dim, int):
                        if dim != size:
                            raise ValueError(f"expected {dim}, got {size}")
                    elif dim.isidentifier():  # type: ignore[attr-defined]
                        if env.setdefault(cast(str, dim), size) != size:
                            raise ValueError(f"{dim} is both {env[cast(str, dim)]} and {size}")
                    else:
                        formulas.append((cast(str, dim), size))
            for dim, size in formulas:
                free = sorted(set(_names_in(dim)) - set(env))
                if free:
                    # One name, increasing in it: solve by search.
                    (name,) = free
                    fits = [n for n in range(size + 1) if _value(dim, {**env, name: n}) == size]
                    if not fits:
                        raise ValueError(f"no {name} makes {dim} equal {size}")
                    env[name] = fits[0]
                elif _value(dim, env) != size:
                    raise ValueError(f"{dim} is {_value(dim, env)}, got {size}")
            head, dims = self.ret
            return np.ones(tuple(_value(d, env) for d in dims), dtype=SYMBOL_HEADS[head])
        # *Ts path: bind the shared variadic segment plus the known dims.
        env_ts: dict[str, int] = {}
        formulas_ts: list[tuple[str, int]] = []
        ts_binding: tuple[int, ...] | None = None

        def check_dim(dim: object, size: int) -> None:
            if isinstance(dim, int):
                if dim != size:
                    raise ValueError(f"expected {dim}, got {size}")
            elif isinstance(dim, str) and dim.isidentifier():
                if env_ts.setdefault(dim, size) != size:
                    raise ValueError(f"{dim} is both {env_ts[dim]} and {size}")
            else:
                assert isinstance(dim, str)
                formulas_ts.append((dim, size))

        for (head, dims), array in zip(self.formals, arrays):
            if not isinstance(array, np.ndarray) or str(array.dtype) != SYMBOL_HEADS[head]:
                raise TypeError(f"expected a {SYMBOL_HEADS[head]} array")
            if TS_DIM not in dims:
                if array.ndim != len(dims):
                    raise ValueError(f"expected rank {len(dims)}, got {array.ndim}")
                for dim, size in zip(dims, array.shape):
                    check_dim(dim, size)
                continue
            idx = list(dims).index(TS_DIM)
            pre = dims[:idx]
            post = dims[idx + 1 :]
            if array.ndim < len(pre) + len(post):
                raise ValueError(
                    f"expected at least rank {len(pre) + len(post)}, got {array.ndim}"
                )
            ts_part = array.shape[len(pre) : array.ndim - len(post) if post else array.ndim]
            if ts_binding is None:
                ts_binding = tuple(ts_part)
            elif tuple(ts_part) != ts_binding:
                raise ValueError(f"Ts is both {ts_binding} and {tuple(ts_part)}")
            for dim, size in zip(pre, array.shape[: len(pre)]):
                check_dim(dim, size)
            if post:
                for dim, size in zip(post, array.shape[array.ndim - len(post) :]):
                    check_dim(dim, size)
        for dim, size in formulas_ts:
            free = sorted(set(_names_in(dim)) - set(env_ts))
            if free:
                (name,) = free
                fits = [n for n in range(size + 1) if _value(dim, {**env_ts, name: n}) == size]
                if not fits:
                    raise ValueError(f"no {name} makes {dim} equal {size}")
                env_ts[name] = fits[0]
            elif _value(dim, env_ts) != size:
                raise ValueError(f"{dim} is {_value(dim, env_ts)}, got {size}")
        if ts_binding is None:
            ts_binding = ()
        head, dims = self.ret
        out: list[int] = []
        for d in dims:
            if d == TS_DIM:
                out.extend(ts_binding)
            else:
                assert isinstance(d, (int, str))
                out.append(_value(d, env_ts))
        return np.ones(tuple(out), dtype=SYMBOL_HEADS[head])


@dataclass
class Case:
    expr: str
    params: list[tuple[str, str, tuple[object, ...]]] = field(default_factory=list)
    helpers: list["Helper"] = field(default_factory=list)
    family: str = ""

    @property
    def symbolic(self) -> bool:
        return bool(self.params)

    @property
    def has_ts(self) -> bool:
        for _, _, shape in self.params:
            if TS_DIM in shape:
                return True
        for h in self.helpers:
            for _, dims in h.formals:
                if TS_DIM in dims:
                    return True
            if TS_DIM in h.ret[1]:
                return True
        return False

    def signature(self) -> str:
        parts = []
        for name, head, shape in self.params:
            parts.append(f"{name}: {head}[{_spell_dims(shape)}]")
        return ", ".join(parts)

    def runtime_namespace(
        self, sizes: dict[str, int] = SYMBOL_SIZES, ts_prefix: tuple[int, ...] = ()
    ) -> dict[str, object]:
        ns: dict[str, object] = {"np": np}
        for name, head, shape in self.params:
            concrete: list[int] = []
            for d in shape:
                if d == TS_DIM:
                    concrete.extend(ts_prefix)
                else:
                    concrete.append(_dim_size(d, sizes))
            ns[name] = np.ones(tuple(concrete), dtype=SYMBOL_HEADS[head])
        for helper in self.helpers:
            helper.bind(ns)
        return ns


def _spell_dims(shape: tuple[object, ...]) -> str:
    """Dims as written inside `F64[...]`: a formula has to be quoted."""
    parts: list[str] = []
    for d in shape:
        if d == TS_DIM:
            parts.append("Unpack[Ts]")
        elif isinstance(d, str) and not d.isidentifier():
            parts.append(f'"{d}"')
        else:
            parts.append(str(d))
    return ", ".join(parts)


def _dim_size(dim: object, sizes: dict[str, int]) -> int:
    if isinstance(dim, int):
        return dim
    value = _eval_dim(cast(str, dim), sizes)
    assert value is not None and value >= 0, dim
    return value


def generate(seed: int, count: int, symbolic: bool) -> list[Case]:
    rng = random.Random(seed)
    if symbolic:
        gen = SymGen(rng)
        return [gen.case() for _ in range(count)]
    plain = Gen(rng)
    concrete: list[Case] = []
    for _ in range(count):
        expr = plain.expression()
        concrete.append(Case(expr, family=plain.current_family))
    return concrete


def render(cases: list[Case]) -> tuple[str, list[int]]:
    """mypy source for the cases, and the line each `reveal_type` is on.

    Every other line is scaffolding that must type-check: `check`
    refuses a run with an error anywhere else.
    """
    out = [
        "from typing import TypeVar, TypeVarTuple, Unpack",
        "import numpy as np",
        "from numty import F64, F32, I64, U8, Bool",
        *(f'{name} = TypeVar("{name}", bound=int)' for name in SYMBOL_SIZES),
    ]
    if any(c.has_ts for c in cases):
        out.append('Ts = TypeVarTuple("Ts")')
    lines: list[int] = []
    for i, case in enumerate(cases):
        for helper in case.helpers:
            out.extend(helper.definition())
        if case.symbolic:
            out.append(f"def f{i}({case.signature()}) -> None:")
            out.append(f"    reveal_type({case.expr})")
        else:
            out.append(f"reveal_type({case.expr})")
        lines.append(len(out))
    return "\n".join(out) + "\n", lines


# --------------------------------------------------------------------------
# Reading mypy's answer


@dataclass
class Node:
    """A parsed type: `name[args]`, or a `|` union (name == "|")."""

    name: str
    args: list["Node"] = field(default_factory=list)


def _split_top(text: str, sep: str) -> list[str]:
    parts, depth, start = [], 0, 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif depth == 0 and text.startswith(sep, i):
            parts.append(text[start:i])
            i += len(sep)
            start = i
            continue
        i += 1
    parts.append(text[start:])
    return [p.strip() for p in parts]


def parse_type(text: str) -> Node:
    text = text.strip()
    union = _split_top(text, " | ")
    if len(union) > 1:
        return Node("|", [parse_type(p) for p in union])
    if text.endswith("]") and "[" in text:
        name, inner = text.split("[", 1)
        inner = inner[:-1]
        if name == "Literal":
            return Node("Literal", [Node(inner)])
        return Node(name, [parse_type(p) for p in _split_top(inner, ", ")])
    return Node(text)


_BITS = {"_8Bit": 8, "_16Bit": 16, "_32Bit": 32, "_64Bit": 64}
_DIRECT = {
    "float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8",
    "uint16", "uint32", "uint64", "complex64", "complex128", "bool",
}


def dtype_names(node: Node) -> set[str] | None:
    """The numpy dtype names a `dtype[...]` argument allows, or None if open."""
    short = node.name.rsplit(".", 1)[-1]
    if node.name == "|":
        names: set[str] = set()
        for arg in node.args:
            got = dtype_names(arg)
            if got is None:
                return None
            names |= got
        return names
    if short in _DIRECT:
        return {short}
    if short == "bool_":
        return {"bool"}
    if short == "intp":
        return {"int64", "int32"}
    if short == "uintp":
        return {"uint64", "uint32"}
    kinds = {"floating": "float", "signedinteger": "int", "unsignedinteger": "uint"}
    if short in kinds and len(node.args) == 1:
        widths = node.args[0].args if node.args[0].name == "|" else [node.args[0]]
        out = set()
        for w in widths:
            bits = _BITS.get(w.name.rsplit(".", 1)[-1])
            if bits is None:
                return None
            out.add(f"{kinds[short]}{bits}")
        return out
    if short == "complexfloating" and node.args:
        bits = _BITS.get(node.args[0].name.rsplit(".", 1)[-1])
        return {f"complex{bits * 2}"} if bits else None
    return None


def _eval_dim(text: str, sizes: dict[str, int] = SYMBOL_SIZES) -> int | None:
    """A revealed dim (`N`, `N+1`, `N*M`, `N//2+1`) under `sizes`."""
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return None
    allowed = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant, ast.Load,
        ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.USub,
    )
    for sub in ast.walk(tree):
        if not isinstance(sub, allowed):
            return None
        if isinstance(sub, ast.Name) and sub.id not in sizes:
            return None
    value = eval(compile(tree, "<dim>", "eval"), {"__builtins__": {}}, dict(sizes))
    return value if isinstance(value, int) else None


def _negative_dim(text: str, sizes: dict[str, int]) -> bool:
    """Whether a dim formula spelled in `text` is negative under `sizes`."""
    for token in re.findall(r"[A-Z0-9+*/-]+", text):
        # Only formulas over a size name: a bare `-1` is an axis.
        if not any(name in token for name in sizes):
            continue
        value = _eval_dim(token.strip("+*/"), sizes)
        if value is not None and value < 0:
            return True
    return False


@dataclass
class ArrayClaim:
    """What a revealed ndarray type asserts. None means "not claimed"."""

    dims: list[int | None] | None  # None: rank unknown
    dtypes: set[str] | None

    @property
    def concrete(self) -> bool:
        return self.dims is not None and all(d is not None for d in self.dims)


def array_claim(
    node: Node, sizes: dict[str, int] = SYMBOL_SIZES, ts_prefix: tuple[int, ...] = ()
) -> ArrayClaim | None:
    if node.name != "numpy.ndarray" or len(node.args) != 2:
        return None
    shape, dtype = node.args
    dims: list[int | None] | None
    if shape.name in ("tuple", "builtins.tuple"):
        if len(shape.args) == 1 and shape.args[0].name == "()":
            dims = []
        else:
            dims = []
            for a in shape.args:
                if a.name == "*Ts" or (
                    a.name in ("Unpack", "typing.Unpack", "typing_extensions.Unpack")
                    and len(a.args) == 1
                    and a.args[0].name == "Ts"
                ):
                    dims.extend(list(ts_prefix))
                    continue
                if a.name == "..." or a.name.startswith(("Unpack", "*")):
                    dims = None
                    break
                if a.name == "Literal" and a.args and a.args[0].name.lstrip("-").isdigit():
                    dims.append(int(a.args[0].name))
                elif not a.args:
                    dims.append(_eval_dim(a.name, sizes))
                else:
                    dims.append(None)
    else:
        dims = None
    names = None
    if dtype.name == "numpy.dtype" and len(dtype.args) == 1:
        names = dtype_names(dtype.args[0])
    return ArrayClaim(dims, names)


# --------------------------------------------------------------------------
# Comparison


@dataclass
class Finding:
    kind: str  # WRONG_SHAPE, WRONG_DTYPE, CLAIM_ON_ERROR, MISSED_ERROR, FALSE_POSITIVE
    case: Case
    static: str
    runtime: str
    sizes: dict[str, int] | None = None  # symbolic cases: the assignment
    ts_prefix: tuple[int, ...] | None = None  # *Ts cases: the prefix rank

    def __str__(self) -> str:
        where = f"def f({self.case.signature()}): " if self.case.symbolic else ""
        where = "".join(f"{h.signature()} ...\n    " for h in self.case.helpers) + where
        at = ""
        if self.sizes is not None:
            at = " with " + ", ".join(f"{k}={v}" for k, v in self.sizes.items())
            if self.ts_prefix:
                at += f", Ts={self.ts_prefix}"
            elif self.case.has_ts:
                at += ", Ts=()"
        return (
            f"{self.kind}: {where}{self.case.expr}\n    numty: {self.static}"
            f"\n    numpy{at}: {self.runtime}"
        )


def _check_array(claim: ArrayClaim, value: np.ndarray | np.generic) -> str | None:
    shape = tuple(np.shape(value))
    if claim.dims is not None:
        if isinstance(value, np.generic) and not claim.dims:
            pass
        elif len(claim.dims) != len(shape) or any(
            d is not None and d != s for d, s in zip(claim.dims, shape)
        ):
            return "WRONG_SHAPE"
    if claim.dtypes is not None and str(value.dtype) not in claim.dtypes:
        return "WRONG_DTYPE"
    return None


def _compare(
    node: Node,
    value: object,
    sizes: dict[str, int] = SYMBOL_SIZES,
    ts_prefix: tuple[int, ...] = (),
) -> str | None:
    """Kind of mismatch between a revealed type and a runtime value."""
    claim = array_claim(node, sizes, ts_prefix)
    if claim is not None:
        if isinstance(value, (np.ndarray, np.generic)):
            return _check_array(claim, value)
        return None
    if isinstance(value, (tuple, list)) and value and all(isinstance(v, np.ndarray) for v in value):
        if node.name in ("tuple", "builtins.tuple") and len(node.args) == len(value):
            if not any(a.name == "..." for a in node.args):
                for sub, item in zip(node.args, value):
                    bad = _compare(sub, item, sizes, ts_prefix)
                    if bad:
                        return bad
        elif node.name in ("list", "builtins.list") and len(node.args) == 1:
            for item in value:
                bad = _compare(node.args[0], item, sizes, ts_prefix)
                if bad:
                    return bad
    return None


def _describe(value: object) -> str:
    if isinstance(value, BaseException):
        return f"raises {type(value).__name__}: {str(value)[:120]}"
    if isinstance(value, (np.ndarray, np.generic)):
        return f"shape {tuple(np.shape(value))}, dtype {value.dtype}"
    if isinstance(value, (tuple, list)):
        return f"{type(value).__name__} of " + ", ".join(_describe(v) for v in value[:4])
    return type(value).__name__


def evaluate(
    case: Case, sizes: dict[str, int] = SYMBOL_SIZES, ts_prefix: tuple[int, ...] = ()
) -> object:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            try:
                return eval(case.expr, case.runtime_namespace(sizes, ts_prefix))
            except Exception as exc:  # the exception is the answer
                return exc


INTERNAL_MARK = "[numty-internal]"


def _internal_failures(
    errors: dict[int, list[str]], line_to_case: dict[int, "Case"]
) -> list[tuple[int, str, str]]:
    """(line, case.expr, err) for each [numty-internal] on a reveal line."""
    hits: list[tuple[int, str, str]] = []
    for line in sorted(errors):
        case = line_to_case.get(line)
        if case is None:
            continue
        for err in errors[line]:
            if INTERNAL_MARK in err:
                hits.append((line, case.expr, err))
    return hits


def run_mypy(
    source: str, workdir: Path, *, reuse_cache: bool = False
) -> tuple[dict[int, str], dict[int, list[str]], str]:
    """Reveals and errors by line for `source`, plus the raw output.

    Cold by default: the cache is wiped first, so every call is
    independent of whatever a previous call in this workdir left
    behind. `reuse_cache=True` keeps it (only for an immediate rerun
    of the same `source`, e.g. the warm leg of a parity check).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    cache = workdir / "cache"
    if not reuse_cache:
        shutil.rmtree(cache, ignore_errors=True)
    path = workdir / "cases.py"
    path.write_text(source, encoding="utf-8")
    (workdir / "mypy.ini").write_text("[mypy]\nplugins = numty_plugin\n", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [
            sys.executable, "-m", "mypy", "--config-file", "mypy.ini",
            "--cache-dir", str(cache), "--no-error-summary",
            "--show-traceback", "cases.py",
        ],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    raw = result.stdout + result.stderr
    reveals: dict[int, str] = {}
    errors: dict[int, list[str]] = {}
    for line in result.stdout.splitlines():
        m = re.match(r"cases\.py:(\d+): (note|error): (.*)", line)
        if not m:
            continue
        number, level, text = int(m[1]), m[2], m[3]
        if level == "note" and text.startswith('Revealed type is "'):
            reveals[number] = text[len('Revealed type is "') : -1]
        elif level == "error":
            errors.setdefault(number, []).append(text)
    return reveals, errors, raw


def check(
    cases: list[Case],
    workdir: Path,
    assignments: tuple[dict[str, int], ...] = SIZE_ASSIGNMENTS,
    *,
    warm_parity: bool = False,
) -> tuple[list[Finding], dict[str, int]]:
    """Findings and outcome counts for `cases`.

    mypy runs once. A symbolic case is then evaluated under every size
    assignment and counted once per assignment; a concrete case has no
    sizes to vary and is counted once.

    With `warm_parity`, mypy runs twice on the same `source` and workdir
    (second run served from `workdir/cache`); the parsed answers and raw
    output must match exactly before any case is judged against numpy.
    """
    source, lines = render(cases)
    reveals, errors, raw = run_mypy(source, workdir)
    if "INTERNAL ERROR" in raw or "Traceback (most recent call last)" in raw:
        raise AssertionError("mypy crashed on generated cases:\n" + raw[-4000:])
    if warm_parity:
        warm_reveals, warm_errors, warm_raw = run_mypy(source, workdir, reuse_cache=True)
        if "INTERNAL ERROR" in warm_raw or "Traceback (most recent call last)" in warm_raw:
            raise AssertionError("mypy crashed on warm rerun:\n" + warm_raw[-4000:])
        if reveals != warm_reveals:
            for n in sorted(set(reveals) | set(warm_reveals)):
                if reveals.get(n) != warm_reveals.get(n):
                    raise AssertionError(
                        f"warm/cold reveal mismatch at line {n}:\n"
                        f"  cold: {reveals.get(n)}\n  warm: {warm_reveals.get(n)}"
                    )
        if errors != warm_errors:
            for n in sorted(set(errors) | set(warm_errors)):
                if errors.get(n) != warm_errors.get(n):
                    raise AssertionError(
                        f"warm/cold error mismatch at line {n}:\n"
                        f"  cold: {errors.get(n)}\n  warm: {warm_errors.get(n)}"
                    )
        if raw != warm_raw:
            cold_lines, warm_lines = raw.splitlines(), warm_raw.splitlines()
            for i, (x, y) in enumerate(zip(cold_lines, warm_lines)):
                if x != y:
                    raise AssertionError(
                        f"warm/cold raw mismatch at line {i + 1}:\n  cold: {x}\n  warm: {y}"
                    )
            raise AssertionError(
                f"warm/cold raw length mismatch: {len(cold_lines)} vs {len(warm_lines)}"
            )
    stray = sorted(set(errors) - set(lines))
    if stray:
        # A helper signature or a `def` line: the generator's mistake,
        # and the reveal below it would be compared against nonsense.
        shown = "\n".join(
            f"{source.splitlines()[n - 1]}\n    {errors[n][0]}" for n in stray[:5]
        )
        raise AssertionError("generated scaffolding does not type-check:\n" + shown)
    # Tripwire: fail closed on any [numty-internal] on a reveal line,
    # before judge(). judge() would bury a crash three ways:
    # _DATA_ERRORS early-return, "numpy raises, flagged" for any errs,
    # and exact/partial scoring when the reveal is unchecked (internal
    # is not in _ASSERTING_CODES). Deliberately no Finding kind and no
    # report()/KNOWN_GAPS path: a crash is never a "known gap".
    # Scaffolding internals stay stray (already fail above).
    line_to_case = {line: case for case, line in zip(cases, lines)}
    hits = _internal_failures(errors, line_to_case)
    if hits or INTERNAL_MARK in raw:
        examples = "\n".join(
            f"line {line}: {expr!r}\n    {err}" for line, expr, err in hits[:5]
        )
        total = len(hits) or raw.count(INTERNAL_MARK)
        raise AssertionError(
            f"{total} {INTERNAL_MARK} error(s) on generated cases "
            f"(plugin bug, see numty_plugin._swallowed); check() aborts "
            f"before scoring — fix the plugin, do not count as a gap:\n"
            + (examples or f"raw tail:\n{raw[-2000:]}")
        )
    findings: list[Finding] = []
    stats: dict[str, int] = {}
    current_family = ""

    def count(key: str) -> None:
        stats[key] = stats.get(key, 0) + 1
        # Additive per-family keys "family|<global-key>"; "|" cannot
        # occur in global keys or family names. Each judge() call does
        # exactly one count(), so family totals sum to the family's
        # outcome count. Hand-made Cases (family=="") skip family keys.
        if current_family:
            family_key = f"{current_family}|{key}"
            stats[family_key] = stats.get(family_key, 0) + 1

    def report(
        kind: str,
        case: Case,
        static: str,
        value: object,
        sizes: dict[str, int],
        ts_prefix: tuple[int, ...] = (),
    ) -> None:
        for known_kind, pattern, _why in KNOWN_GAPS:
            if kind == known_kind and re.search(pattern, case.expr):
                count(f"known gap: {kind}")
                return
        shown = sizes if case.symbolic else None
        ts_shown: tuple[int, ...] | None = ts_prefix if case.has_ts else None
        # Empty prefix is still a prefix rank: show it for *Ts cases so
        # two findings that differ only by rank do not look identical.
        if case.has_ts and ts_shown is None:
            ts_shown = ()
        findings.append(Finding(kind, case, static, _describe(value), shown, ts_shown))
        count(kind)

    def judge(
        case: Case,
        line: int,
        sizes: dict[str, int],
        value: object,
        raises: bool,
        ts_prefix: tuple[int, ...] = (),
    ) -> None:
        static = reveals.get(line, "<no reveal>")
        errs = errors.get(line, [])
        asserting = [e for e in errs if any(f"[{c}]" in e for c in _ASSERTING_CODES)]
        if isinstance(value, _DATA_ERRORS):
            count("skipped: data-dependent runtime error")
            return
        if isinstance(value, BaseException):
            if errs:
                count("numpy raises, flagged")
                return
            # Issue #183: symbolic mode counted every silent rejection
            # as allowed. A rejection that survives every small size
            # assignment (1-3 for each name, including equal names) is
            # provable: no choice of N/M/K/P would make it work, so
            # silence is a miss, not prudence. TypeError (dtype loops)
            # is excluded; ValueError/IndexError cover shapes, axes,
            # ranks. Concrete keeps its rule (only a concrete claim
            # fails); *Ts, cross and repeat/vdot/stack cross-position
            # need follow-up and stay unprovable for now.
            if isinstance(value, (ValueError, IndexError)):
                if static != "<no reveal>":
                    if not case.symbolic:
                        node = parse_type(static)
                        claim = array_claim(node, sizes, ts_prefix)
                        if claim is not None and claim.concrete:
                            report("CLAIM_ON_ERROR", case, static, value, sizes, ts_prefix)
                            return
                    elif not case.has_ts:
                        if _always_raises(case):
                            report("MISSED_ERROR", case, static, value, sizes, ts_prefix)
                            return
            count("numpy raises, not flagged (unprovable)")
            return
        # A call to a generated function has no stubs to blame: any
        # error on a call the signature allows is numty's.
        blamed = errs if case.helpers else asserting
        if blamed and raises:
            count(SPECIAL_SIZES)
            return
        if blamed:
            report("FALSE_POSITIVE", case, blamed[0], value, sizes, ts_prefix)
            return
        if static == "<no reveal>":
            count("no reveal")
            return
        bad = _compare(parse_type(static), value, sizes, ts_prefix)
        if bad == "WRONG_SHAPE" and case.symbolic and _negative_dim(static, sizes):
            count(BELOW_DOMAIN)
            return
        if bad:
            report(bad, case, static, value, sizes, ts_prefix)
            return
        claim = array_claim(parse_type(static), sizes, ts_prefix)
        if claim is not None and claim.concrete and isinstance(value, np.ndarray):
            count("agree: exact shape")
        else:
            count("agree: partial or no claim")

    for case, line in zip(cases, lines):
        current_family = case.family
        if not case.symbolic:
            used: list[tuple[dict[str, int], tuple[int, ...]]] = [(assignments[0], ())]
        elif case.has_ts:
            used = [(s, t) for s in assignments for t in TS_PREFIXES]
        else:
            used = [(s, ()) for s in assignments]
        values = [evaluate(case, sizes, ts) for sizes, ts in used]
        raises = any(
            isinstance(v, BaseException) and not isinstance(v, _DATA_ERRORS) for v in values
        )
        for (sizes, ts), value in zip(used, values):
            judge(case, line, sizes, value, raises, ts)
    return findings, stats


def main(argv: list[str]) -> int:
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", default="1-4", help="e.g. 7 or 1-20")
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--mode", choices=("concrete", "symbolic", "both"), default="both")
    args = parser.parse_args(argv)
    lo, _, hi = args.seeds.partition("-")
    seeds = range(int(lo), int(hi or lo) + 1)
    modes = ("concrete", "symbolic") if args.mode == "both" else (args.mode,)
    total = 0
    for mode in modes:
        for seed in seeds:
            cases = generate(seed, args.count, symbolic=mode == "symbolic")
            with tempfile.TemporaryDirectory() as tmp:
                findings, stats = check(cases, Path(tmp))
            print(f"== {mode} seed {seed}: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
            for finding in findings:
                print(finding)
            total += len(findings)
    print(f"{total} finding(s)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
