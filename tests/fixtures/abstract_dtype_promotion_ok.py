"""Issue #248: an abstract-dtype operand does not give its dtype to the result."""
from typing import TypeVar

import numpy as np
from numty import F32, F64, Bool, I64

N = TypeVar("N", bound=int)
_S = TypeVar("_S", bound=tuple[int, ...])
Int = np.ndarray[tuple[N], np.dtype[np.integer]]
IntP = np.ndarray[tuple[N], np.dtype[np.intp]]
Fl = np.ndarray[tuple[N], np.dtype[np.floating]]
FlS = np.ndarray[_S, np.dtype[np.floating]]
IntS = np.ndarray[_S, np.dtype[np.integer]]


def show(
    i: Int[N], p: IntP[N], g: Fl[N], x: F64[N], y: F32[N], j: I64[N], m: Bool[N],
    a: FlS[_S], b: IntS[_S], s: float,
) -> None:
    reveal_type(i / 2)  # line 20: float64
    reveal_type(p / 2)  # float64
    reveal_type(i + x)  # float64
    reveal_type(p + x)  # float64
    reveal_type(p * 0.5)  # line 24: float64
    reveal_type(i * 0.5)  # float64
    reveal_type(np.where(m, p, x))  # float64
    reveal_type(np.maximum(i, x))  # float64
    reveal_type(m * g)  # line 28: floating
    reveal_type(g * m)  # floating
    reveal_type(i - y)  # floating: the width depends on the integer
    reveal_type(i * g)  # floating
    reveal_type(g / 2.0)  # line 32: floating
    reveal_type(a / s)  # _S, floating: float32 stays float32
    reveal_type(b / 2)  # _S, float64
    reveal_type(b * 0.5)  # _S, float64
    reveal_type(p - j)  # line 36: int64
    reveal_type(p + 1)  # intp, the operand's own spelling
    reveal_type(i + i)  # integer
    reveal_type(g + x)  # floating


def int_div(p: IntP[N]) -> F64[N]:
    return p / 2


def int_plus_float(p: IntP[N], x: F64[N]) -> F64[N]:
    return p + x


def int_times_float(i: Int[N]) -> F64[N]:
    return i * 0.5


DT = TypeVar("DT", bound=np.dtype)
ST = TypeVar("ST", bound=np.floating)
Td = np.ndarray[tuple[N], np.dtype[np.timedelta64]]
Ob = np.ndarray[tuple[N], np.dtype[np.object_]]
C128 = np.ndarray[tuple[N], np.dtype[np.complex128]]


def typevar_dtype(x: np.ndarray[tuple[N], DT], k: float) -> np.ndarray[tuple[N], DT]:
    return x * k  # a TypeVar dtype stays with a Python scalar


def bound_typevar(x: np.ndarray[tuple[N], np.dtype[ST]]) -> np.ndarray[tuple[N], np.dtype[ST]]:
    reveal_type(x * 0.5)  # line 66: dtype[ST]
    reveal_type(np.where(x > 0, x, 0.0))  # dtype[ST]
    return x / x


def other_kinds(td: Td[N], ob: Ob[N], x: F64[N], i: Int[N], g: Fl[N], c: C128[N], m: Bool[N]) -> None:
    reveal_type(td * 2)  # line 72: timedelta64
    reveal_type(x * td)  # timedelta64
    reveal_type(i * td)  # timedelta64
    reveal_type(td / 2.0)  # timedelta64
    reveal_type(td / td)  # line 76: dtype[Any] (numpy: float64)
    reveal_type(ob + 1)  # line 77: object_
    reveal_type(x + ob)  # object_
    reveal_type(np.where(m, x, ob))  # object_
    reveal_type(c + g)  # line 80: dtype[Any] (a longdouble gives complex256)
    reveal_type(c + x)  # complex128
