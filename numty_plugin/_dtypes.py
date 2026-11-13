"""Dtype promotion core (Phase 2, #196). PURE: no mypy import, no package imports."""
from __future__ import annotations

_DTYPE_INFO: dict[str, tuple[str, int]] = {
    "numpy.bool": ("b", 1),
    "numpy.int8": ("i", 1),
    "numpy.int16": ("i", 2),
    "numpy.int32": ("i", 4),
    "numpy.int64": ("i", 8),
    "numpy.uint8": ("u", 1),
    "numpy.uint16": ("u", 2),
    "numpy.uint32": ("u", 4),
    "numpy.uint64": ("u", 8),
    "numpy.float16": ("f", 2),
    "numpy.float32": ("f", 4),
    "numpy.float64": ("f", 8),
    "numpy.complex64": ("c", 8),
    "numpy.complex128": ("c", 16),
}


_NAME_OF_KIND_SIZE = {v: k for k, v in _DTYPE_INFO.items()}


# Generic scalar classes whose width is a bit-width tag argument
# (floating[_32Bit] is float32), by the kind they stand for.
_GENERIC_KIND = {
    "numpy.floating": "f",
    "numpy.complexfloating": "c",
    "numpy.signedinteger": "i",
    "numpy.unsignedinteger": "u",
}


def _nbit_tag_bits(fullname: str) -> int | None:
    """Bits of a bit-width tag class (numpy._typing._nbit_base._32Bit -> 32)."""
    tail = fullname.rsplit(".", 1)[-1]
    digits = tail[1:-3]
    if not (tail.startswith("_") and tail.endswith("Bit") and digits.isdecimal()):
        return None
    return int(digits)


def _kind_bits_name(kind: str, bits: int) -> str | None:
    """Canonical dtype fullname for a kind and a width in bits, else None.

    The width counts the whole item: ("c", 64) is complex64.
    """
    if bits % 8:
        return None
    return _NAME_OF_KIND_SIZE.get((kind, bits // 8))


_DATETIME_FULLNAMES = frozenset({"numpy.datetime64", "numpy.timedelta64"})


def _datetime_string_kind(value: str) -> str | None:
    """'M'/'m' for datetime64/timedelta64 strings with any unit, else None."""
    text = value.strip()
    if text in ("datetime64", "timedelta64"):
        return "M" if text == "datetime64" else "m"
    for base, kind in (("datetime64", "M"), ("timedelta64", "m")):
        if text.startswith(base + "[") and text.endswith("]"):
            return kind
    return None


_WEAK_SCALARS = frozenset(
    {"builtins.int", "builtins.float", "builtins.complex", "builtins.bool"}
)


def _promote_strong(a: str, b: str) -> str | None:
    """numpy.result_type for two array dtypes, else None to bail."""
    if a == b:
        return a
    info_a = _DTYPE_INFO.get(a)
    info_b = _DTYPE_INFO.get(b)
    if info_a is None or info_b is None:
        return None
    (ka, sa), (kb, sb) = info_a, info_b
    if ka == "b":
        return b
    if kb == "b":
        return a
    if ka == kb:
        return _NAME_OF_KIND_SIZE[(ka, max(sa, sb))]
    if {ka, kb} == {"i", "u"}:
        si, su = (sa, sb) if ka == "i" else (sb, sa)
        if si > su:
            return _NAME_OF_KIND_SIZE[("i", si)]
        big = 2 * su
        return _NAME_OF_KIND_SIZE[("i", big)] if big <= 8 else "numpy.float64"
    if "c" in (ka, kb):
        c = sb if ka != "c" else sa
        if "f" in (ka, kb):
            f = sa if ka == "f" else sb
            result = c if f <= c // 2 else min(2 * c, 16)
        else:
            s = sa if ka != "c" else sb
            result = c if c // 2 > s else min(2 * c, 16)
        return _NAME_OF_KIND_SIZE[("c", result)]
    f = sa if ka == "f" else sb
    s = sb if ka == "f" else sa
    full = max(f, min(2 * s, 8))
    return _NAME_OF_KIND_SIZE[("f", full)]


def _promote_weak(scalar: str, strong: str) -> str | None:
    """numpy.result_type for a Python scalar and an array dtype."""
    info = _DTYPE_INFO.get(strong)
    if info is None:
        return None
    kind, size = info
    if scalar == "builtins.bool":
        return strong
    if scalar == "builtins.int":
        return "numpy.int64" if kind == "b" else strong
    if scalar == "builtins.float":
        return strong if kind in ("f", "c") else "numpy.float64"
    if scalar == "builtins.complex":
        if kind == "c":
            return strong
        if kind == "f" and size <= 4:
            return "numpy.complex64"
        return "numpy.complex128"
    return None


_DTYPE_BASE = {
    "numpy.float16": "numpy.floating",
    "numpy.float32": "numpy.floating",
    "numpy.complex64": "numpy.complexfloating",
    "numpy.int8": "numpy.signedinteger",
    "numpy.int16": "numpy.signedinteger",
    "numpy.int32": "numpy.signedinteger",
    "numpy.int64": "numpy.signedinteger",
    "numpy.uint8": "numpy.unsignedinteger",
    "numpy.uint16": "numpy.unsignedinteger",
    "numpy.uint32": "numpy.unsignedinteger",
    "numpy.uint64": "numpy.unsignedinteger",
}


_NBIT_TAG_BY_BYTES = {
    1: "_8Bit",
    2: "_16Bit",
    4: "_32Bit",
    8: "_64Bit",
    16: "_128Bit",
}


_NBIT_MODULES = ("numpy._typing._nbit_base", "numpy._typing._nbit")


_UFUNC_KEEP = frozenset(
    {
        "square",
        "reciprocal",
        "floor",
        "ceil",
        "trunc",
        "sign",
        "conjugate",
        "conj",
        "positive",
        "negative",
        "invert",
        "bitwise_not",
    }
)


_UFUNC_ABS = frozenset({"absolute", "abs"})
# Ufuncs with no complex loop (numpy raises TypeError for complex input).


_UFUNC_NO_COMPLEX = frozenset(
    {
        "cbrt", "floor", "ceil", "trunc", "fabs", "arctan2", "hypot", "fmod",
        "copysign", "floor_divide", "remainder", "mod", "logaddexp",
        "logaddexp2", "signbit", "deg2rad", "rad2deg", "degrees", "radians",
        "nextafter", "spacing", "heaviside",
    }
)


_UFUNC_FLOAT = frozenset(
    {
        "sqrt",
        "cbrt",
        "exp",
        "exp2",
        "expm1",
        "log",
        "log2",
        "log10",
        "log1p",
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "rint",
        "sinc",
    }
)


_UFUNC_REAL = frozenset({"real", "imag"})


_UFUNC_BOOL = frozenset(
    {
        "isfinite",
        "isinf",
        "isnan",
        "isneginf",
        "isposinf",
        "signbit",
        "logical_not",
    }
)


_WEAK_DEFAULT = {
    "builtins.bool": "numpy.bool",
    "builtins.int": "numpy.int64",
    "builtins.float": "numpy.float64",
    "builtins.complex": "numpy.complex128",
}


_UFUNC_BINARY_PROMOTE = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "power",
        "floor_divide",
        "remainder",
        "fmod",
        "minimum",
        "maximum",
        "fmin",
        "fmax",
    }
)


_UFUNC_BINARY_FLOAT = frozenset({"hypot", "arctan2", "logaddexp", "logaddexp2"})


_UFUNC_BINARY_BOOL = frozenset(
    {
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "logical_and",
        "logical_or",
        "logical_xor",
    }
)
