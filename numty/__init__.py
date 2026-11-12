"""numty: micro-MVP static shape checking for numpy.

Plain standard typing plus one small mypy plugin. Rank via tuple length,
fixed sizes via Literal, cross-argument solving in the plugin.
Rank aliases and sized arrays check under mypy, pyright, or ty; the
F64-style heads and the shape solver need the mypy plugin (#5).
"""
from __future__ import annotations

from typing import Any, Generic, TypeVar, TypeVarTuple, Unpack, cast

import numpy as np

# Rank-only aliases. Catch transposed / flattened mixups.
Vector = np.ndarray[tuple[int], np.dtype[np.float64]]
Matrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]

# Precise shapes without pre-baked sizes: define your own aliases in one
# line from the generic base. E.g. A2x3 = FloatArray[tuple[Literal[2],
# Literal[3]]] or B2x3 = Array2D[Literal[2], Literal[3]].
S = TypeVar("S", bound=tuple[int, ...])
FloatArray = np.ndarray[S, np.dtype[np.float64]]

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)
Array1D = np.ndarray[tuple[M], np.dtype[np.float64]]
Array2D = np.ndarray[tuple[M, N], np.dtype[np.float64]]

# Rank-polymorphic base: shape and dtype, both spelled at use sites, e.g.
# ``FlexArray[tuple[Unpack[Ts], Literal[2]], np.dtype[NP_FLOAT]]``. S2 is
# bound to ``tuple[Any, ...]``, not numpy's ``tuple[int, ...]``, so a shape
# containing ``*Ts`` fits the alias; mypy's solver unifies ``*Ts`` from the
# actual argument, so call sites stay precise. A PEP 695 alias rather than
# an assignment: ty resolved ``FlexArray = np.ndarray[S2, D2]`` to an
# Unknown shape (no rank checks at all, #36), and keeps both parameters of
# this one. mypy still checks the expanded ``ndarray`` against the stubs'
# bound, so the two ``type: ignore[type-var]`` notes each rank-generic
# signature needs are the only cost; bodies and call sites check fully.
# Runtime note: as a ``typing.TypeAliasType``, ``get_origin`` of a
# ``FlexArray[...]`` hint is ``FlexArray`` (``get_origin(FlexArray.__value__)``
# is ``np.ndarray``), and calling ``FlexArray(...)`` raises TypeError.
type FlexArray[S2: tuple[Any, ...], D2: np.dtype[Any]] = np.ndarray[S2, D2]


# Familiar-style shape heads: jaxtyping's feel, numty's solver. Spell dims
# flat — TypeVars (or the ready-made ones below), integers, Unpack[Ts] —
# and the plugin reads them statically, no converter needed:
# F64[batch, features], F64[BATCH, FEATURES], F64[3, 4].
# Same-name dims unify; anything else is an error, not a silent Any.
TsFam = TypeVarTuple("TsFam")


class _HeadMeta(type):
    """Metaclass that makes the heads honest at runtime.

    Statically the plugin rewrites `F64[...]` in annotations to an
    ndarray, so `isinstance(x, F64)` narrows `x` to an array. Back that up
    here: a head matches exactly the ndarrays of its dtype. Rank and dims
    are not checked, and a subscripted head (`F64[3]`) cannot be used with
    isinstance, as for any generic. Heads are annotations, never values,
    so calling one raises instead of building an object that is not an
    array.
    """

    _numty_dtype: np.dtype[Any] | None = None

    def __instancecheck__(cls, obj: object) -> bool:
        dtype = cls._numty_dtype
        if dtype is None:
            return super().__instancecheck__(obj)
        return isinstance(obj, np.ndarray) and obj.dtype == dtype

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(
            f"{cls.__name__} is a shape annotation, not a constructor; "
            "build the array with numpy (e.g. np.zeros(..., dtype=...))"
        )


class _StringDims(metaclass=_HeadMeta):
    """Runtime companion for the familiar heads below.

    Static checking happens in mypy (see numty_plugin); at runtime a
    multi-word spec like "batch features" is not a valid forward
    reference, so `Generic.__class_getitem__` would raise SyntaxError at
    import time. Fall back to the bare head (plain erasure): runtime
    never sees shapes anyway. Specs that subscribe fine today (bare
    names, integers, Unpack) pass through untouched.
    """

    def __class_getitem__(cls, params):
        try:
            # Generic supplies __class_getitem__ through the MRO of each
            # head; ty and pyright cannot see that from _StringDims alone.
            return super().__class_getitem__(params)  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        except (SyntaxError, TypeError):
            return cls


class F64(_StringDims, Generic[Unpack[TsFam]]):
    """float64 array with flat dims, e.g. F64[batch, features]."""

    _numty_dtype = np.dtype(np.float64)


class F32(_StringDims, Generic[Unpack[TsFam]]):
    """float32 array with flat dims, e.g. F32[batch, features]."""

    _numty_dtype = np.dtype(np.float32)


class F16(_StringDims, Generic[Unpack[TsFam]]):
    """float16 array with flat dims, e.g. F16[batch, features]."""

    _numty_dtype = np.dtype(np.float16)


class I64(_StringDims, Generic[Unpack[TsFam]]):
    """int64 array with flat dims, e.g. I64[batch, count]."""

    _numty_dtype = np.dtype(np.int64)


class I32(_StringDims, Generic[Unpack[TsFam]]):
    """int32 array with flat dims, e.g. I32[batch, count]."""

    _numty_dtype = np.dtype(np.int32)


class I16(_StringDims, Generic[Unpack[TsFam]]):
    """int16 array with flat dims, e.g. I16[batch, count]."""

    _numty_dtype = np.dtype(np.int16)


class I8(_StringDims, Generic[Unpack[TsFam]]):
    """int8 array with flat dims, e.g. I8[batch, features]."""

    _numty_dtype = np.dtype(np.int8)


class U8(_StringDims, Generic[Unpack[TsFam]]):
    """uint8 array with flat dims, e.g. U8[height, width, 3] for an image."""

    _numty_dtype = np.dtype(np.uint8)


class Bool(_StringDims, Generic[Unpack[TsFam]]):
    """boolean array with flat dims, e.g. Bool[batch, mask]."""

    _numty_dtype = np.dtype(np.bool_)


class Index(_StringDims, Generic[Unpack[TsFam]]):
    """Positions array: Index[P, N] holds N int64 values indexing size P.

    The shape says how many; the first arg says which axis the values are
    positions on (issues #285, #286). Index[P] leaves the count dynamic,
    Index["P N"] splits a string spec the same way, and a multi-dim shape
    (Index[P, N, M]) records P but is not checked. A scalar -1-for-none
    position is not tracked.
    """

    _numty_dtype = np.dtype(np.int64)


# Ready-made dim TypeVars for the heads above: `F64[batch, features]`
# works with a single import, no declarations. Same name unifies across
# annotations; declare your own TypeVar for anything not listed here.
batch = TypeVar("batch", bound=int)
channel = TypeVar("channel", bound=int)
height = TypeVar("height", bound=int)
width = TypeVar("width", bound=int)
depth = TypeVar("depth", bound=int)
features = TypeVar("features", bound=int)
classes = TypeVar("classes", bound=int)
seq = TypeVar("seq", bound=int)
hidden = TypeVar("hidden", bound=int)
hid = TypeVar("hid", bound=int)
inp = TypeVar("inp", bound=int)
out = TypeVar("out", bound=int)
head = TypeVar("head", bound=int)


# Shape-preserving pass-through. A generic over the shape keeps the exact
# shape instead of widening it to a rank-only alias.
def ident(m: FloatArray[S]) -> FloatArray[S]:
    return m


class NarrowError(ValueError, AssertionError):
    """`narrow` got an array that does not match its witness.

    A ValueError, and also an AssertionError so code written against the
    earlier assert-based check keeps catching it.
    """


# Narrowing for values whose shape is only known at runtime. The witness
# carries the desired shape as its type, so no per-shape helper is needed:
# narrow(m, np.zeros((2, 3))) -> precisely 2x3, checked at runtime. The
# checks raise explicitly rather than assert: `python -O` strips asserts,
# and the cast would then hand back a mismatched array typed as the witness.
def narrow(m: LooseArray, witness: FloatArray[S]) -> FloatArray[S]:
    if m.shape != witness.shape:
        raise NarrowError(
            f"narrow: shape {m.shape} does not match witness {witness.shape}"
        )
    if m.dtype != witness.dtype:
        raise NarrowError(
            f"narrow: dtype {m.dtype} does not match witness {witness.dtype}"
        )
    return cast(FloatArray[S], m)


# General matmul solving lives in the mypy plugin (numty_plugin/), which
# computes output shapes for any rank and promotes mixed dtypes. An earlier
# attempt with axis TypeVars failed silently: ndarray's shape parameter is
# effectively covariant, so a shared TypeVar joins conflicting literals
# (e.g. N becomes int for 3 vs 2) instead of erroring. The declared
# parameters stay loose so any rank and dtype reaches the hook; without the
# plugin this is unchecked (but usable).
LooseArray = np.ndarray[tuple[Any, ...], np.dtype[Any]]


def matmul(a: LooseArray, b: LooseArray) -> LooseArray:
    return np.matmul(a, b)
