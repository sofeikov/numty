"""The dtype of an indexed assignment: opt-in `[numty-cast]` (issue 308).

numpy's `a[idx] = value` casts with `casting="unsafe"`, so floats written
into an integer array are truncated with no error at run time. With
`enable_error_code = numty-cast` a value whose kind the target cannot
hold is reported. Top-level imports go to earlier layers only
(tests/test_layout.py, _LAYERS).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.maptype import map_instance_to_supertype
from mypy.types import Instance, LiteralType, Type, get_proper_type

from ._dtypes import _DTYPE_INFO
from ._errors import NUMTY_CAST
from ._builders import _scalar_ret_name
from ._linalg import _dtype_family, _scalar_family
from ._reshape import _dtype_fullname, _weak_name

if TYPE_CHECKING:
    from mypy.plugin import MethodContext

# numpy's kind order: `np.can_cast(a, b, casting="same_kind")` holds for
# two kinds exactly when the first is not later than the second here
# (bool into anything, unsigned into signed, any integer into float,
# float into complex). Widths do not matter to it: float64 into float32
# and int64 into int8 are same_kind and stay silent.
_KIND_ORDER = {"b": 0, "u": 1, "i": 2, "f": 3, "c": 4}

_FAMILY_NAME = {
    "b": "bool",
    "u": "unsignedinteger",
    "i": "signedinteger",
    "f": "floating",
    "c": "complexfloating",
}

_WEAK_FAMILY = {
    "builtins.bool": "b",
    "builtins.int": "i",
    "builtins.float": "f",
    "builtins.complex": "c",
}

_SEQUENCES = frozenset({"builtins.list", "builtins.tuple"})


def _cast_enabled(ctx: MethodContext) -> bool:
    """Whether `enable_error_code = numty-cast` is set for the module being checked."""
    options = getattr(ctx.api, "options", None)
    return options is not None and NUMTY_CAST in options.enabled_error_codes


def _array_dtype(tp: Type) -> Type | None:
    """The `dtype[...]` of an ndarray (or a subclass of one), else None."""
    proper = get_proper_type(tp)
    if not isinstance(proper, Instance):
        return None
    for base in proper.type.mro:
        if base.fullname == "numpy.ndarray":
            mapped = map_instance_to_supertype(proper, base)
            return mapped.args[1] if len(mapped.args) == 2 else None
    return None


def _int_literal(tp: Type) -> int | None:
    """The value of an `int` literal (`Literal[1]` or the inferred `1?`), else None."""
    proper = get_proper_type(tp)
    if isinstance(proper, Instance) and proper.last_known_value is not None:
        proper = proper.last_known_value
    if isinstance(proper, LiteralType) and type(proper.value) is int:
        return proper.value
    return None


def _is_sequence(tp: Type) -> bool:
    proper = get_proper_type(tp)
    return isinstance(proper, Instance) and proper.type.fullname in _SEQUENCES


def _value_kind(tp: Type, depth: int = 0) -> tuple[str, str, bool] | None:
    """(kind, display name, is a Python scalar) for an assigned value, else None.

    An ndarray by its dtype, a numpy scalar by its class, a Python
    scalar by its builtin, and a list or tuple of Python scalars (as
    `[0.5, 1.7]`) by its element. Anything else (Any, a TypeVar dtype
    with no single kind, object or string arrays) is None: unknown.
    """
    dtype = _array_dtype(tp)
    if dtype is not None:
        kind = _dtype_family(dtype)
        if kind is None:
            return None
        name = _dtype_fullname(dtype)
        return kind, name.removeprefix("numpy.") if name else _FAMILY_NAME[kind], False
    weak = _weak_name(tp)
    if weak is not None:
        return _WEAK_FAMILY[weak], weak.removeprefix("builtins."), True
    proper = get_proper_type(tp)
    if not isinstance(proper, Instance):
        return None
    if proper.type.fullname in _SEQUENCES and depth < 8:
        if len(proper.args) != 1:
            return None
        inner = _value_kind(proper.args[0], depth + 1)
        # Only Python scalars inside: a list of arrays is read like an
        # array by numpy, but its dtype is the promotion of the members.
        return inner if inner is not None and inner[2] else None
    kind = _scalar_family(proper)
    if kind is None:
        return None
    name = proper.type.fullname if proper.type.fullname in _DTYPE_INFO else _scalar_ret_name(proper)
    return kind, name.removeprefix("numpy.") if name else _FAMILY_NAME[kind], False


def _loss(value_kind: str, target_kind: str) -> str:
    """What the cast does to the values, for the message."""
    if target_kind == "b":
        return "values become True or False"
    if value_kind == "c":
        return "the imaginary part is dropped" + ("" if target_kind == "f" else " and values are truncated")
    if value_kind == "f":
        return "values are truncated"
    return "negative values wrap around"


def _report_cast(ctx: MethodContext, target: Type, value: Type) -> None:
    """The opt-in `[numty-cast]` report for `target[idx] = value` (issue 308).

    Reported when numpy's `can_cast(value, target, "same_kind")` is
    False: float or complex into integer or bool, complex into float,
    integer into bool, signed into unsigned. A Python scalar is the one
    exception: an `int` into a bool array is reported only when it is not
    the literal 0 or 1, and a Python `int` into an unsigned array is
    silent (numpy raises for a negative one). Nothing is emitted with the
    code off (the default), as for `[numty-names]`.
    """
    if not _cast_enabled(ctx):
        return
    target_dtype = _array_dtype(target)
    if target_dtype is None:
        return
    target_kind = _dtype_family(target_dtype)
    if target_kind is None:
        return
    found = _value_kind(value)
    if found is None:
        return
    value_kind, value_name, weak = found
    if _KIND_ORDER[value_kind] <= _KIND_ORDER[target_kind]:
        return
    if weak and value_kind == "i":
        if target_kind == "u":
            return
        # `mask[i] = 1`, `mask[:] = [0, 1, 1]`: the values fit. A list's
        # element literals are not in its type, so a list of ints is
        # taken as fitting too.
        if target_kind == "b" and (_int_literal(value) in (0, 1) or _is_sequence(value)):
            return
    target_name = _dtype_fullname(target_dtype)
    target_shown = target_name.removeprefix("numpy.") if target_name else _FAMILY_NAME[target_kind]
    ctx.api.fail(
        f"assignment casts {value_name} into {target_shown} ({_loss(value_kind, target_kind)}); "
        "use .astype(...) if this is meant",
        ctx.context,
        code=NUMTY_CAST,
    )
