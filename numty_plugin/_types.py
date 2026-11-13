"""_types.py — pure algebra/fold/shape (Phase 5, #199). Owns NDARRAY_FULLNAME
etc. (Phase 5, #199). Top-level imports go to earlier layers only (order in
tests/test_layout.py, _LAYERS). A function-local facade import is used only
for a monkeypatch seam (_LAZY_SEAMS).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

from mypy.errorcodes import ARG_TYPE
from mypy.nodes import (
    ARG_NAMED, ARG_NAMED_OPT, ARG_OPT, ARG_POS, CallExpr, Context, Expression, IntExpr,
    ListExpr, MemberExpr, NameExpr, OpExpr, StarExpr, TupleExpr, TypeInfo, UnaryExpr, Var,
)
from mypy.messages import callable_name
from mypy.operators import op_methods_to_symbols
from mypy.plugin import AttributeContext, FunctionContext, FunctionSigContext, MethodContext, MethodSigContext
from mypy.types import (
    AnyType, CallableType, Instance, LiteralType, NoneType, TupleType, Type, TypeOfAny, TypeVarId,
    TypeVarType, UnionType, UnpackType, get_proper_type,
)

if TYPE_CHECKING:
    from mypy.checker import TypeChecker

from ._errors import NUMTY_NAMES, NUMTY_SHAPE, _swallowed
from ._forms import (
    _DivForm, _LinForm, _MonoForm, _coeff_vars_to_form, _form_has_unknown, _form_mul,
    _forms_provably_ne, _is_unknown_fullname, _lin_add, _lin_scale, _mono_normalize,
)
from ._exprs import (
    _expr_constituents, _expr_str, _form_of, _seen_vars,
)

NDARRAY_FULLNAME = "numpy.ndarray"
INT_FULLNAME = "builtins.int"
TUPLE_FULLNAME = "builtins.tuple"
FLOAT64_FULLNAME = "numpy.float64"
DTYPE_FULLNAME = "numpy.dtype"
STR_FULLNAME = "builtins.str"

# Familiar-style heads (defined in numty/__init__.py) and the numpy scalar each one
# stands for. Bool maps to numpy.bool, matching _DTYPE_STRINGS below.
def _int_value(tp: Type) -> int | None:
    """A known axis size, or None when symbolic or unknown."""
    proper = get_proper_type(tp)
    if isinstance(proper, LiteralType):
        value = proper.value
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    if isinstance(proper, Instance):
        if proper.type.fullname == INT_FULLNAME and proper.last_known_value is not None:
            return _int_value(proper.last_known_value)
    return None


def _dim_value(tp: Type) -> int | Type | None:
    """A known axis size: int literal, symbolic TypeVar, or None."""
    proper = get_proper_type(tp)
    if isinstance(proper, LiteralType):
        value = proper.value
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    if isinstance(proper, Instance):
        if proper.type.fullname == INT_FULLNAME and proper.last_known_value is not None:
            return _dim_value(proper.last_known_value)
    if isinstance(proper, TypeVarType):
        return proper
    return None


def _dim_eq(a: int | Type, b: int | Type) -> bool:
    """Provably-equal dims: same ints, same TypeVar, or same canonical form.

    Expression variables resolve through the registry, so N*2 built at two
    sites compares equal; anything unresolvable falls back to False.
    ?U unknowns never prove equal, not even to themselves (issue #63):
    one mask site serves every comprehension element, so sharing ?U0
    would claim false (3, 3) matmuls. The result bails instead.
    """
    from numty_plugin import _resolve_dim
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, TypeVarType) and isinstance(b, TypeVarType):
        if a.fullname == b.fullname:
            if _is_unknown_fullname(a.fullname):
                return False
            return True
    fa = _resolve_dim(a)
    fb = _resolve_dim(b)
    if fa is not None and fb is not None:
        if fa == fb:
            if _form_has_unknown(fa):
                return False
            return True
    return False


def _dims_provably_ne(a: int | Type, b: int | Type, nonzero: bool = False) -> bool:
    """Provably-unequal dims: no sizes at all make them equal.

    `N` vs `N-1`, `N+1` vs `N`, `M+2` vs `M` can never be equal (the
    equality would require `0 == nonzero`), and neither can
    `N//2+1` vs `N//2`. Anything else (different vars `N` vs `M`,
    unknowns, div-vs-linear) might be equal at runtime, so it stays
    unprovable (issue #56: no false errors). Used by broadcast/matmul/
    stack to flag real off-by-one bugs from `np.diff` / `N+1` offsets
    (issue #32) instead of bailing to a silent `?U`.

    `nonzero=True` also counts pairs that are equal only on an empty
    axis (`2*N` vs `N`, see `_forms_provably_ne`). Only reshape asks
    for that.
    """
    from numty_plugin import _resolve_dim
    if _dim_eq(a, b):
        return False
    if isinstance(a, int) and isinstance(b, int):
        return a != b
    fa = _resolve_dim(a)
    fb = _resolve_dim(b)
    return _forms_provably_ne(fa, fb, nonzero)


def _named_sizes_differ(a: int | Type, b: int | Type) -> bool:
    """Two sizes spelled over names, neither with an unknown, that are not the same size.

    `N` against `M`, `N+1` against `M`. A literal, an `int`, an `Any`
    and an unknown (`?U`) never count: nothing is known about them. A
    pair that is provably unequal (`N` against `N-1`) is an error
    already, under the code of its check.
    """
    from numty_plugin import _resolve_dim
    if not isinstance(a, TypeVarType) or not isinstance(b, TypeVarType):
        return False
    if _dim_eq(a, b) or _dims_provably_ne(a, b):
        return False
    fa = _resolve_dim(a)
    fb = _resolve_dim(b)
    if fa is None or fb is None or _form_has_unknown(fa) or _form_has_unknown(fb):
        return False
    return True


def _names_enabled(ctx: FunctionContext | MethodContext | FunctionSigContext | MethodSigContext) -> bool:
    """Whether `enable_error_code = numty-names` is set for the module being checked."""
    options = getattr(ctx.api, "options", None)
    return options is not None and NUMTY_NAMES in options.enabled_error_codes


def _named_pair_in(a: Sequence[int | Type], b: Sequence[int | Type]) -> bool:
    """Whether the broadcast of two shapes pairs two different named sizes."""
    return any(_named_sizes_differ(x, y) for x, y in zip(reversed(a), reversed(b)))


def _report_names(
    ctx: FunctionContext | MethodContext | FunctionSigContext | MethodSigContext,
    a: int | Type,
    b: int | Type,
    where: str,
) -> None:
    """The opt-in `[numty-names]` report for a pair of different named sizes (issue 280).

    Called where a check finds two sizes that must be equal and cannot
    prove them equal or unequal. Nothing is emitted with the code off
    (the default): mypy counts a message of a disabled code as a failure
    of the overload it is checking, so `v + w` would get mypy's own
    `[operator]` error. The result of the check is the same either way.
    """
    if _names_enabled(ctx) and _named_sizes_differ(a, b):
        ctx.api.fail(
            f"{where}: {_dim_str(a)} and {_dim_str(b)} are different names",
            ctx.context,
            code=NUMTY_NAMES,
        )


def _is_mask_count_dim(d: int | Type) -> str | None:
    """The mask-count identity of a selection count, else None (issue 288).

    Mask counts mint as unknowns (`__numty_unknown__.maskcount_...`), so
    the default checks treat them like any unknown: silent and unifying.
    The size a call chose for a name of its return alone (issue 307,
    `.ownsize_`) is the same kind of size.
    For the opt-in report they are sizes of their own: `?U_keep` against
    a named size, or against another mask's count, is a different-names
    pair, while the same mask variable is the same size.
    """
    if isinstance(d, TypeVarType) and (".maskcount_" in d.fullname or ".ownsize_" in d.fullname):
        return d.fullname
    return None


# A free return name -> the unknowns that took it in the body (see
# `_returns._name_born_sizes`): `sel = x[keep]`, returned as `F64[K]`,
# is `(K,)`, and `K` is the count of `keep`.
_OWN_NAME_OF: dict[TypeVarId, set[str]] = {}


def _same_own_size(a: int | Type, b: int | Type) -> bool:
    """One selection count, or one size a call chose, on both sides (issues 288, 307).

    Also a free name against the unknown that took it: `sel * y[keep]`
    with `sel = x[keep]` named `K` is one count on both sides.
    """
    if not isinstance(a, TypeVarType) or not isinstance(b, TypeVarType):
        return False
    ia = _is_mask_count_dim(a)
    ib = _is_mask_count_dim(b)
    if ia is not None and ia == ib:
        return True
    if b.fullname in _OWN_NAME_OF.get(a.id, ()) or a.fullname in _OWN_NAME_OF.get(b.id, ()):
        return True
    return False


def _mask_counts_differ(a: int | Type, b: int | Type) -> bool:
    """Whether two dims are different selection counts under the opt-in (issue 288).

    A mask count against a named size, or two different masks' counts,
    must be equal where they meet (broadcast, assignment, call) and are
    reported with the code on. The same mask variable is the same count.
    Generic unknowns, literals, `int` and `Any` never count.
    """
    from numty_plugin import _resolve_dim
    if _same_own_size(a, b):
        return False
    ma = _is_mask_count_dim(a)
    mb = _is_mask_count_dim(b)
    if ma is not None and mb is not None:
        return ma != mb
    other: int | Type | None = None
    if ma is not None:
        other = b
    elif mb is not None:
        other = a
    else:
        return False
    if not isinstance(other, TypeVarType):
        return False
    if _is_unknown_fullname(other.fullname) and _is_mask_count_dim(other) is None:
        return False
    if _dim_eq(a, b) or _dims_provably_ne(a, b):
        return False
    fa = _resolve_dim(a)
    fb = _resolve_dim(b)
    if fa is None or fb is None:
        return False
    named_form = fb if ma is not None else fa
    if _form_has_unknown(named_form):
        return False
    return True


def _mask_pair_in(a: Sequence[int | Type], b: Sequence[int | Type]) -> bool:
    """Whether the broadcast of two shapes pairs a reportable mask count (issue 288)."""
    return any(_mask_counts_differ(x, y) for x, y in zip(reversed(a), reversed(b)))


def _report_mask_names(
    ctx: FunctionContext | MethodContext | FunctionSigContext | MethodSigContext,
    a: int | Type,
    b: int | Type,
    where: str,
) -> None:
    """The opt-in `[numty-names]` report for a selection count (issue 288).

    Same emission rule as `_report_names`: nothing with the code off, so
    the default stays silent and no `[operator]` duplicate appears.
    """
    if not _names_enabled(ctx):
        return
    if _mask_counts_differ(a, b):
        ctx.api.fail(
            f"{where}: {_dim_str(a)} and {_dim_str(b)} are different names",
            ctx.context,
            code=NUMTY_NAMES,
        )
        return
    apart = _unknown_offset(a, b)
    if apart:
        ctx.api.fail(
            f"{where}: {_dim_str(a)} and {_dim_str(b)} are {abs(apart)} apart",
            ctx.context,
            code=NUMTY_NAMES,
        )


def _unknown_offset(a: int | Type, b: int | Type) -> int:
    """`a - b` when both are one formula over unknowns up to a constant, else 0.

    `?U` against `?U-1` (`gap = np.diff(c)`, then `np.r_[np.inf, gap]`
    against `gap`). An unknown is named after its source site, and one
    site can stand for several sizes (the items of a comprehension), so
    this is not a proof and is only reported under the opt-in code.
    """
    from numty_plugin import _resolve_dim
    if not isinstance(a, TypeVarType) or not isinstance(b, TypeVarType):
        return 0
    fa = _resolve_dim(a)
    fb = _resolve_dim(b)
    if not isinstance(fa, _LinForm) or not isinstance(fb, _LinForm):
        return 0
    if not fa.coeffs or fa.coeffs != fb.coeffs or not _form_has_unknown(fa):
        return 0
    return fa.const - fb.const


def _dim_str(d: int | Type) -> str:
    if isinstance(d, int):
        return str(d)
    if isinstance(d, UnpackType):
        inner = d.type
        name = getattr(inner, "name", None)
        return f"*{name}" if isinstance(name, str) else "*Ts"
    if isinstance(d, TypeVarType):
        form = _form_of(d)
        if form is not None:
            return _expr_str(form)
    name = getattr(d, "name", None)
    return str(name) if isinstance(name, str) else "?"


def _dim_add_const(
    ctx: FunctionContext | MethodContext, op: str, index: int, a: int | Type, b: int
) -> int | Type:
    """a + b for a constant b: exact linear form, else a fresh unknown."""
    from numty_plugin import _resolve_dim, _materialize_dim, _fresh_unknown
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    fa = _resolve_dim(a)
    if fa is None or isinstance(fa, (_DivForm, _MonoForm)):
        return _fresh_unknown(ctx, op, index)
    return _materialize_dim(ctx, op, index, _lin_add(fa, _LinForm((), b)), prefer=a)


def _dim_add2(
    ctx: FunctionContext | MethodContext,
    op: str,
    index: int,
    a: int | Type,
    b: int | Type,
) -> int | Type:
    """a + b: exact linear form, else a fresh unknown."""
    from numty_plugin import _resolve_dim, _materialize_dim, _fresh_unknown
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    fa = _resolve_dim(a)
    fb = _resolve_dim(b)
    if (
        fa is None
        or fb is None
        or isinstance(fa, (_DivForm, _MonoForm))
        or isinstance(fb, (_DivForm, _MonoForm))
    ):
        return _fresh_unknown(ctx, op, index)
    return _materialize_dim(ctx, op, index, _lin_add(fa, fb))


def _dim_mul_const(
    ctx: FunctionContext | MethodContext, op: str, index: int, a: int | Type, c: int
) -> int | Type:
    """a * c for a constant c: exact linear/monomial form, else unknown."""
    from numty_plugin import _resolve_dim, _materialize_dim, _fresh_unknown
    if isinstance(a, int):
        return a * c
    if c == 1:
        return a
    if c == 0:
        return 0
    fa = _resolve_dim(a)
    if fa is None or isinstance(fa, _DivForm):
        return _fresh_unknown(ctx, op, index)
    if isinstance(fa, _MonoForm):
        return _materialize_dim(
            ctx, op, index, _mono_normalize(fa.coeff * c, fa.vars), prefer=a
        )
    return _materialize_dim(ctx, op, index, _lin_scale(fa, c))


def _dim_div(
    ctx: FunctionContext | MethodContext,
    op: str,
    index: int,
    a: int | Type,
    div: int,
    plus: int,
) -> int | Type:
    """a // div + plus: simplified to linear when exact, else a Div node."""
    from numty_plugin import _resolve_dim, _materialize_dim, _fresh_unknown
    if div == 0:
        return _fresh_unknown(ctx, op, index)
    fa = _resolve_dim(a)
    if fa is None or isinstance(fa, _DivForm):
        return _fresh_unknown(ctx, op, index)
    if isinstance(fa, _MonoForm):
        # Exact int division of a product keeps it exact (2*N*M//2);
        # anything else (plus, inexact) bails — no Div-of-monomial.
        if plus == 0 and fa.coeff % div == 0:
            q = _coeff_vars_to_form(fa.coeff // div, fa.vars)
            return _materialize_dim(ctx, op, index, q, prefer=a)
        return _fresh_unknown(ctx, op, index)
    if isinstance(fa, _LinForm) and not fa.coeffs:
        return fa.const // div + plus
    if div < 0:
        return _fresh_unknown(ctx, op, index)
    # Whole multiples of div leave the floor: (2*N+1)//2 is N and
    # (N+3)//2 is (N+1)//2+1. Taking them out gives one spelling per
    # function, so equal dims built in different orders compare equal.
    if all(c % div == 0 for _, c in fa.coeffs):
        base = _LinForm(tuple((n, c // div) for n, c in fa.coeffs), fa.const // div)
        if plus:
            base = _lin_add(base, _LinForm((), plus))
        return _materialize_dim(ctx, op, index, base, prefer=a)
    inner = _LinForm(fa.coeffs, fa.const % div)
    return _materialize_dim(ctx, op, index, _DivForm(inner, div, plus + fa.const // div))


def _mul_dims(
    ctx: FunctionContext | MethodContext, op: str, index: int, dims: list[int | Type]
) -> int | Type:
    """Product of dims: exact linear/monomial form, else a fresh unknown."""
    from numty_plugin import _resolve_dim, _materialize_dim, _fresh_unknown
    acc: _LinForm | _MonoForm | None = _LinForm((), 1)
    for d in dims:
        f = _resolve_dim(d)
        if acc is None or f is None or isinstance(f, _DivForm):
            return _fresh_unknown(ctx, op, index)
        assert isinstance(f, (_LinForm, _MonoForm))
        acc = _form_mul(acc, f)
        if acc is None:
            return _fresh_unknown(ctx, op, index)
    assert acc is not None
    return _materialize_dim(ctx, op, index, acc)


def _is_bare_name_dim(d: int | Type) -> bool:
    """An int, or one declared size name: no formula and no unknown."""
    from numty_plugin import _resolve_dim
    if isinstance(d, int):
        return True
    form = _resolve_dim(d)
    return (
        isinstance(form, _LinForm)
        and form.const == 0
        and len(form.coeffs) == 1
        and form.coeffs[0][1] == 1
        and not _form_has_unknown(form)
    )


def _linear_product(dims: Sequence[int | Type]) -> _LinForm | _MonoForm | None:
    """Canonical total product: linear or single monomial, else None."""
    from numty_plugin import _resolve_dim
    acc: _LinForm | _MonoForm | None = _LinForm((), 1)
    for d in dims:
        f = _resolve_dim(d)
        if acc is None or f is None or isinstance(f, _DivForm):
            return None
        assert isinstance(f, (_LinForm, _MonoForm))
        acc = _form_mul(acc, f)
        if acc is None:
            return None
    return acc


def _reshape_total_form(dims: Sequence[int | Type]) -> _LinForm | _DivForm | _MonoForm | None:
    """Total elements for reshape legality: single dim resolves directly
    (so N+1, N+2 sums are comparable), multi-dim uses the pure-product
    canonical form; None when unrepresentable (issue #63)."""
    from numty_plugin import _resolve_dim
    if len(dims) == 1:
        return _resolve_dim(dims[0])
    return _linear_product(dims)


def _check_dim_eq(
    ctx: FunctionContext | MethodContext, x: int | Type, y: int | Type, message: str
) -> bool:
    """Provable dim equality for a solver check: True to proceed.

    Fails with message when both dims are ints that differ; bails out
    silently when symbolic dims cannot be proven equal.
    """
    if _dim_eq(x, y):
        return True
    if isinstance(x, int) and isinstance(y, int):
        ctx.api.fail(message, ctx.context, code=NUMTY_SHAPE)
    return False


def _shape_dims(tp: Type) -> list[int | Type] | None:
    """Dims of a shape tuple (literals and TypeVars), or None if unknown."""
    proper = get_proper_type(tp)
    if not isinstance(proper, TupleType):
        return None
    values: list[int | Type | None] = [_dim_value(item) for item in proper.items]
    if any(v is None for v in values):
        return None
    return [v for v in values if v is not None]


def _shape_dims_unpack(tp: Type) -> tuple[list[int | Type], UnpackType, list[int | Type]] | None:
    """Split a *Ts shape into (known prefix, unpack, known suffix).

    Only a single unpack with otherwise known dims (int literals and
    TypeVars) qualifies; anything else bails with None.
    """
    proper = get_proper_type(tp)
    if not isinstance(proper, TupleType):
        return None
    prefix: list[int | Type] = []
    suffix: list[int | Type] = []
    unpack: UnpackType | None = None
    seen = False
    for item in proper.items:
        if isinstance(item, UnpackType):
            if seen:
                return None
            seen = True
            unpack = item
            continue
        value = _dim_value(item)
        if value is None:
            return None
        (suffix if seen else prefix).append(value)
    if unpack is None:
        return None
    return prefix, unpack, suffix


def _fold_int(expr: Expression) -> int | None:
    """Value of a static int expression, or None when not provable.

    Handles literals, unary -/+, + - * // % ** over foldable operands, and
    names whose type is a known literal (Final, Literal-annotated). Plain
    variables and parameters are opaque: their value is gone from the type
    before the hook ever sees it.
    """
    if isinstance(expr, IntExpr):
        if isinstance(expr.value, bool):
            return None
        return expr.value if isinstance(expr.value, int) else None
    if isinstance(expr, UnaryExpr) and expr.op in ("-", "+"):
        value = _fold_int(expr.expr)
        if value is None:
            return None
        return -value if expr.op == "-" else value
    if isinstance(expr, OpExpr) and expr.op in ("+", "-", "*", "//", "%", "**"):
        left = _fold_int(expr.left)
        right = _fold_int(expr.right)
        if left is None or right is None:
            return None
        if expr.op in ("//", "%") and right == 0:
            return None
        if expr.op == "**":
            # Huge powers would hang or OOM the type checker, but sizes up
            # to 2**64 are cheap (at most 64 multiplications of small
            # numbers) and real: np.zeros((2**40, 0)) must keep its literal
            # so a Literal-demanding formal still matches (issue #233).
            # Larger results fail loudly downstream via _dims_fit_intp.
            if right < 0 or right > 64:
                return None
            if abs(left) > 1 << 64:
                return None
        try:
            if expr.op == "+":
                return left + right
            if expr.op == "-":
                return left - right
            if expr.op == "*":
                return left * right
            if expr.op == "//":
                return left // right
            if expr.op == "%":
                return left % right
            return left**right
        except (ArithmeticError, ValueError):
            return None
    if isinstance(expr, NameExpr):
        node = expr.node
        if isinstance(node, Var) and node.type is not None:
            return _int_value(node.type)
    if isinstance(expr, MemberExpr):
        # Enum members (PlayerFeat.x) carry their int value in the
        # member's type; anything else bails out. The member node
        # itself is often unset, so resolve through the enum base.
        node = expr.node
        if isinstance(node, Var) and node.type is not None:
            return _int_value(node.type)
        base = expr.expr
        if isinstance(base, NameExpr) and isinstance(base.node, TypeInfo):
            info = base.node
            if info.is_enum and expr.name in info.names:
                member = info.names[expr.name].node
                if isinstance(member, Var) and member.type is not None:
                    return _int_value(member.type)
        return None
    if (
        isinstance(expr, CallExpr)
        and isinstance(expr.callee, NameExpr)
        and expr.callee.name == "len"
        and len(expr.args) == 1
        and expr.arg_kinds[0] == ARG_POS
        and not isinstance(expr.args[0], StarExpr)
    ):
        fn = expr.callee.node
        if fn is None or getattr(fn, "fullname", "") != "builtins.len":
            return None
        return _fold_len_arg(expr.args[0])
    return None


def _fold_len_arg(arg: Expression) -> int | None:
    """len() of a Final/param array's first literal dim, or an int enum."""
    if not isinstance(arg, NameExpr):
        return None
    node = arg.node
    if isinstance(node, Var) and node.type is not None:
        parts = _ndarray_parts(node.type)
        if parts is not None and parts[0] and isinstance(parts[0][0], int):
            return parts[0][0]
        return None
    if isinstance(node, TypeInfo) and node.is_enum:
        return _enum_member_count(node)
    return None


def _fold_shape_expr(expr: Expression) -> list[int] | None:
    """Dims from a shape expression: tuple/list of foldable ints, else None."""
    if isinstance(expr, (TupleExpr, ListExpr)):
        values: list[int] = []
        for item in expr.items:
            value = _fold_int(item)
            if value is None:
                return None
            values.append(value)
        return values
    if isinstance(expr, StarExpr):
        return None
    value = _fold_int(expr)
    return [value] if value is not None else None


def _ndarray_parts(tp: Type) -> tuple[list[int | Type], Type] | None:
    """(dims, dtype) of a known-rank ndarray type, else None.

    Dims mix int literals and symbolic TypeVars; anything else bails.
    """
    proper = get_proper_type(tp)
    if not isinstance(proper, Instance):
        return None
    if proper.type.fullname != NDARRAY_FULLNAME or len(proper.args) != 2:
        return None
    dims = _shape_dims(proper.args[0])
    return (dims, proper.args[1]) if dims is not None else None


def _parts_or_unpack(tp: Type) -> tuple[list[int | Type], Type] | None:
    """(dims, dtype), preserving a *Ts shape as an Unpack item.

    Same-shape consumers thread the dims through untouched; anything
    rank-sensitive must keep using _ndarray_parts and bail out.
    """
    parts = _ndarray_parts(tp)
    if parts is not None:
        return parts
    split = _unpack_shape(tp)
    if split is None:
        return None
    (prefix, unpack, suffix), dtype = split
    return prefix + [unpack] + suffix, dtype


def _has_unpack(dims: list[int | Type]) -> bool:
    """Whether a dims list carries a *Ts unpack item."""
    return any(isinstance(d, UnpackType) for d in dims)


def _union_members(tp: Type) -> list[Type] | None:
    """Array members of a union type, else None.

    Only pure unions of ndarray types distribute: anything else
    (non-unions, unions with None/scalars) stays out, and the caller
    keeps today's behavior for it. Members may themselves carry *Ts;
    the unpack-aware paths handle those.
    """
    proper = get_proper_type(tp)
    if not isinstance(proper, UnionType):
        return None
    members = []
    for item in proper.items:
        item_proper = get_proper_type(item)
        if not isinstance(item_proper, Instance):
            return None
        if item_proper.type.fullname != NDARRAY_FULLNAME or len(item_proper.args) != 2:
            return None
        members.append(item)
    if len(members) < 2:
        return None
    return members


_PY_SCALARS = frozenset({"builtins.int", "builtins.float", "builtins.complex", "builtins.bool"})


def _is_scalar_member(tp: Type) -> bool:
    """A Python number (`int`, `float`, a literal of one) or a NumPy scalar."""
    proper = get_proper_type(tp)
    if isinstance(proper, LiteralType):
        proper = proper.fallback
    if not isinstance(proper, Instance):
        return False
    return proper.type.fullname in _PY_SCALARS or proper.type.has_base("numpy.generic")


def _scalar_union_members(tp: Type) -> list[Type] | None:
    """Members of a union of numbers, or of arrays and numbers, else None.

    Issue 331. `F64[M] | float` and `float | int`: a union with at least
    one Python or NumPy scalar member and nothing but arrays and scalars
    otherwise. A pure union of arrays is `_union_members`.
    """
    proper = get_proper_type(tp)
    if not isinstance(proper, UnionType):
        return None
    members: list[Type] = []
    scalar = False
    for item in proper.items:
        item_proper = get_proper_type(item)
        if _is_scalar_member(item_proper):
            scalar = True
        elif not (
            isinstance(item_proper, Instance)
            and item_proper.type.fullname == NDARRAY_FULLNAME
            and len(item_proper.args) == 2
        ):
            return None
        members.append(item)
    return members if scalar and len(members) >= 2 else None


def _optional_members(tp: Type) -> list[Type] | None:
    """Members of a union of arrays and `None` (`F64[N] | None`), else None.

    Issue 331: an optional array argument (`weights`, sklearn's
    `sample_weight=None`) distributes per member like a union of arrays.
    Only the call sites that know what `None` means there use it; the
    `None` member stays in the list for them.
    """
    proper = get_proper_type(tp)
    if not isinstance(proper, UnionType):
        return None
    members: list[Type] = []
    seen_none = False
    for item in proper.items:
        item_proper = get_proper_type(item)
        if isinstance(item_proper, NoneType):
            if seen_none:
                continue
            seen_none = True
        elif not (
            isinstance(item_proper, Instance)
            and item_proper.type.fullname == NDARRAY_FULLNAME
            and len(item_proper.args) == 2
        ):
            return None
        members.append(item)
    return members if len(members) >= 2 else None


def _distribute_pairings(types: list[Type]) -> list[list[Type]] | None:
    """Per-operand member lists when union distribution applies.

    Returns one member list per operand position: the union's members
    where that operand is a pure array union, else the single type.
    Requires at least one union and equal member counts everywhere
    (members pair index-wise, which is exact for co-derived unions —
    the overwhelmingly common source of multi-union operands).
    Anything else yields None and the caller keeps today's behavior.
    """
    member_lists: list[list[Type]] = []
    count = 0
    mixed = [m for m in (_scalar_union_members(tp) for tp in types) if m is not None]
    if mixed:
        # Issue 331: a union of arrays and scalars (`F64[M] | float`, a
        # `float | int` threshold) distributes per member too, when it is
        # the only union among the operands: nothing says how its members
        # pair with another union's.
        if len(mixed) != 1 or any(_union_members(tp) is not None for tp in types):
            return None
        return [_scalar_union_members(tp) or [tp] for tp in types]
    for tp in types:
        members = _union_members(tp)
        if members is None:
            member_lists.append([tp])
        else:
            if count == 0:
                count = len(members)
            elif len(members) != count:
                return None
            member_lists.append(members)
    if count == 0:
        return None
    return member_lists


def _in_loop_body(ctx: FunctionContext | MethodContext) -> bool:
    """Whether the call is checked inside the body of a `for` or `while` loop.

    mypy checks a loop body again while the types it binds change, and
    stops after a few passes. A union that a call grows by a member on
    every pass (`out = np.concatenate([out, row])`) is then cut off at
    the last pass, not widened: the joins that grow a size do not
    distribute there (issue 331).
    """
    binder = getattr(ctx.api, "binder", None)
    return bool(getattr(binder, "break_frames", None))


def _union_results(results: list[Type]) -> Type:
    """Union of per-member results (collapses when they coincide)."""
    if len(results) == 1:
        return results[0]
    return UnionType.make_union(results)


def _distribute_solve(types: list[Type], solve: Callable[..., Type]) -> Type | None:
    """Solve per union pairing, unioning results; None when no union.

    `solve` takes one type per operand position. Union members pair
    index-wise (see _distribute_pairings) while single-type positions
    broadcast across every combination; per-pair bails join the union
    as the stub's default, exactly as honest as solving alone.
    """
    pairings = _distribute_pairings(types)
    if pairings is None:
        return None
    width = max(len(members) for members in pairings)
    full = [members if len(members) == width else members * width for members in pairings]
    return _union_results([solve(*combo) for combo in zip(*full)])


def _split_flat(
    dims: list[int | Type],
) -> tuple[list[int | Type], UnpackType, list[int | Type]] | None:
    """Split a flat dims list at its single unpack, else None."""
    found = [i for i, d in enumerate(dims) if isinstance(d, UnpackType)]
    if len(found) != 1:
        return None
    item = dims[found[0]]
    assert isinstance(item, UnpackType)
    return dims[: found[0]], item, dims[found[0] + 1 :]


def _fixed_shape(tp: Type) -> TupleType | None:
    """The shape tuple of an ndarray type of fixed rank (no *Ts), else None."""
    proper = get_proper_type(tp)
    if (
        not isinstance(proper, Instance)
        or proper.type.fullname != NDARRAY_FULLNAME
        or len(proper.args) != 2
    ):
        return None
    shape = get_proper_type(proper.args[0])
    if not isinstance(shape, TupleType):
        return None
    if any(isinstance(get_proper_type(i), UnpackType) for i in shape.items):
        return None
    return shape


def _shape_note_str(shape: TupleType) -> str:
    """Shape for notes: declared names, literals, and plain `int` dims."""
    parts: list[str] = []
    for item in shape.items:
        value = _dim_value(item)
        if value is not None:
            parts.append(_dim_str(value))
        else:
            proper = get_proper_type(item)
            parts.append(proper.type.name if isinstance(proper, Instance) else "?")
    inner = ", ".join(parts)
    return f"({inner},)" if len(parts) == 1 else f"({inner})"


def _rank_mismatch_notes(
    ctx: FunctionSigContext | MethodSigContext, sig: CallableType
) -> list[tuple[str, Context]]:
    """Notes naming the declared shape for each wrong-rank array argument.

    A shape tuple of another length gives mypy's inference no
    constraints, so a dim TypeVar seen only there solves to Never and
    the arg-type error reads `expected "ndarray[tuple[Never], ...]"`
    (#35). mypy's error stands; the note says what the parameter
    declares (`(A,)`) next to what was passed. Only formals that are
    fixed-rank arrays with a dim TypeVar qualify (union members too,
    when every non-None member misses the actual's rank), and only
    actuals of known rank, so the note only lands on a call that the
    rank alone already rejects. Overload items are left out: there
    the error is "No overload variant", not per argument.
    """
    # Overload items reach this as @overload Decorators (or FuncDefs).
    if getattr(sig.definition, "is_overload", False):
        return []
    if len(ctx.args) != len(sig.arg_types):
        return []
    # Only explicit calls: an operator, index or `in` reaches its dunder
    # through this hook too, but mypy reports those under other codes
    # ([operator], [index], [assignment]), which the note's arg-type would
    # not match. Explicit `x.__add__(y)` calls get the operator wording too.
    call = ctx.context
    if not isinstance(call, CallExpr):
        return []
    callee = callable_name(sig)
    if callee is None or _reported_as_operator(callee):
        return []
    notes: list[tuple[str, Context]] = []
    for i, (formal, kind) in enumerate(zip(sig.arg_types, sig.arg_kinds)):
        if kind not in (ARG_POS, ARG_OPT, ARG_NAMED, ARG_NAMED_OPT):
            continue
        if len(ctx.args[i]) != 1:
            continue
        # Only plain positional or keyword actuals: a *args/**kwargs
        # actual spreads its elements, it is not the array itself.
        arg_index = next(
            (j for j, e in enumerate(call.args) if e is ctx.args[i][0]), None
        )
        if arg_index is None or call.arg_kinds[arg_index] not in (ARG_POS, ARG_NAMED):
            continue
        proper = get_proper_type(formal)
        members = proper.relevant_items() if isinstance(proper, UnionType) else [proper]
        members = [m for m in members if not isinstance(get_proper_type(m), NoneType)]
        shapes = [_fixed_shape(m) for m in members]
        if not shapes or any(s is None for s in shapes):
            continue
        declared = [s for s in shapes if s is not None]
        if not any(
            isinstance(get_proper_type(d), TypeVarType) for s in declared for d in s.items
        ):
            continue
        actual = _fixed_shape(ctx.api.get_expression_type(ctx.args[i][0]))
        if actual is None or any(len(s.items) == len(actual.items) for s in declared):
            continue
        name = sig.arg_names[i]
        param = f'parameter "{name}"' if name else f"parameter {i + 1}"
        wanted = " or ".join(_shape_note_str(s) for s in declared)
        ranks = f"rank {len(actual.items)}"
        if len(declared) == 1:
            ranks += f", not {len(declared[0].items)}"
        notes.append(
            (
                f"{callee} declares {param} with shape {wanted}; "
                f"the argument has shape {_shape_note_str(actual)} ({ranks})",
                ctx.args[i][0],
            )
        )
    return notes


# Methods whose bad arguments mypy words as operator/index errors, even
# when called explicitly (MessageBuilder.incompatible_argument): those
# carry [index], [assignment] or [operator], never arg-type.
_OPERATOR_DUNDERS = frozenset(
    ["__getitem__", "__setitem__"]
    + [m for op in op_methods_to_symbols for m in (op, "__r" + op[2:])]
)


def _reported_as_operator(callee: str) -> bool:
    """Whether `callee` (callable_name text) is a method in _OPERATOR_DUNDERS."""
    return any(callee.startswith(f'"{m}" of') for m in _OPERATOR_DUNDERS)


def _mentions_ndarray(tp: Type, depth: int = 0) -> bool:
    """Whether `tp` mentions numpy.ndarray (True when nested too deep to tell).

    Hand-rolled: mypy's TypeQuery visitors are compiled classes, which a
    plugin cannot subclass.
    """
    if depth > 8:
        return True
    proper = get_proper_type(tp)
    if isinstance(proper, Instance):
        if proper.type.fullname == NDARRAY_FULLNAME:
            return True
        parts: list[Type] = list(proper.args)
    elif isinstance(proper, (UnionType, TupleType)):
        parts = list(proper.items)
    elif isinstance(proper, CallableType):
        parts = [*proper.arg_types, proper.ret_type]
    elif isinstance(proper, UnpackType):
        parts = [proper.type]
    else:
        return False
    return any(_mentions_ndarray(part, depth + 1) for part in parts)


def _info_mentions_array(info: TypeInfo) -> bool:
    """Whether a generic class carries array shapes in its fields/bases."""
    try:
        for base in info.bases:
            if _mentions_ndarray(base):
                return True
        for name in info.names:
            try:
                sym = info.names[name]
            except Exception:
                _swallowed(None, "_info_mentions_array")
                continue
            node = sym.node
            tp = getattr(node, "type", None)
            if tp is not None and _mentions_ndarray(tp):
                return True
    except Exception:
        _swallowed(None, "_info_mentions_array")
        return False
    return False


def _note_rank_mismatches(ctx: FunctionSigContext | MethodSigContext, sig: Type) -> None:
    """Emit _rank_mismatch_notes, each printed after mypy's arg-type error.

    Signature hooks run before mypy checks the arguments, and messages
    at one position keep emission order, so each note goes on the
    argument's first line but past its first column (at its end when
    it fits on that line): mypy sorts it after the error, which sits at
    the argument's start (another note at that argument, such as a
    reveal_type, can still print between them). The note shares that
    error's code, so `ignore[arg-type]` and `--disable-error-code
    arg-type` silence both.
    """
    try:
        if not isinstance(sig, CallableType):
            return
        for message, arg in _rank_mismatch_notes(ctx, sig):
            where = Context()
            if arg.end_line == arg.line and arg.end_column is not None:
                where.set_line(arg.line, arg.end_column)
            else:
                where.set_line(arg.line, arg.column + 1)
            # `note` is on the concrete checker, not on the plugin interface.
            cast("TypeChecker", ctx.api).note(message, where, code=ARG_TYPE)
    except Exception:
        _swallowed(ctx, "_note_rank_mismatches")
        return


def _ndarray_type(
    ctx: FunctionContext | MethodContext | AttributeContext,
    dims: Sequence[int | Type],
    dtype: Type,
) -> Instance:
    checker = ctx.api
    int_instance = checker.named_generic_type(INT_FULLNAME, [])
    items: list[Type] = [
        LiteralType(v, fallback=int_instance) if isinstance(v, int) else v for v in dims
    ]
    # A plain `int` dim (a builder's dynamic axis, see _builder_shape)
    # takes mypy's own tuple[int] fallback: with tuple[Any] the stubs'
    # overloads see Any and ops the plugin leaves to them (np.flip, .T,
    # astype, ...) collapse to ndarray[Any, Any].
    plain_int = any(
        isinstance(i, Instance) and i.type.fullname == INT_FULLNAME for i in items
    )
    element: Type = int_instance if plain_int else AnyType(TypeOfAny.explicit)
    tuple_fallback = checker.named_generic_type(TUPLE_FULLNAME, [element])
    return checker.named_generic_type(NDARRAY_FULLNAME, [TupleType(items, tuple_fallback), dtype])


def _single_shape_arg(ctx: FunctionContext | MethodContext) -> Type | None:
    """The shape argument (first positional) of a call, else None.

    Extra named arguments (dtype=, order=, ...) are ignored here and
    handled by the caller's own formal lookups.
    """
    for kinds, types in zip(ctx.arg_kinds, ctx.arg_types):
        for kind, tp in zip(kinds, types):
            if kind == ARG_POS:
                return tp
    return None


def _single_shape_expr(ctx: FunctionContext | MethodContext) -> Expression | None:
    """The shape argument expression (first positional), else None."""
    for kinds, exprs in zip(ctx.arg_kinds, ctx.args):
        for kind, expr in zip(kinds, exprs):
            if kind == ARG_POS:
                return expr
    return None


def _shape_from_arg(tp: Type, expr: Expression | None) -> list[int | Type] | None:
    """Dims from an argument: fold the expression first, types as fallback."""
    if expr is not None:
        dims = _fold_shape_expr(expr)
        if dims is not None:
            return [*dims]
    shape = get_proper_type(tp)
    if isinstance(shape, TupleType):
        return _shape_dims(shape)
    single = _int_value(tp)
    return [single] if single is not None else None


def _int_shape_from_arg(tp: Type, expr: Expression | None) -> list[int] | None:
    """Like _shape_from_arg, for axis, index and count lists.

    These callers compute with the entries as integers. A symbolic
    entry (a TypeVar from the tuple type) is not usable there, so the
    result is None and the caller bails.
    """
    dims = _shape_from_arg(tp, expr)
    if dims is None:
        return None
    out: list[int] = []
    for d in dims:
        if not isinstance(d, int):
            return None
        out.append(d)
    return out


def _builder_fold_dim_expr(
    ctx: FunctionContext | MethodContext, expr: Expression, index: int
) -> int | Type | None:
    """_fold_dim_expr without per-scope side effects (issue #186).

    Form ids are shared on purpose (an inferred `N+1` must be the same
    variable as an annotated one), but the constituent objects (`N` in
    `N+1`) are per-function: overwriting them with the builder site's
    objects makes a later `x[1:]` in another function return the wrong
    `N` (`N@edges` vs `N@trimmed`). Snapshot and restore those two
    registries around the fold.
    """
    seen_snapshot = dict(_seen_vars)
    const_snapshot = {k: dict(v) for k, v in _expr_constituents.items()}
    try:
        return _fold_dim_expr(ctx, expr, index)
    finally:
        _seen_vars.clear()
        _seen_vars.update(seen_snapshot)
        _expr_constituents.clear()
        _expr_constituents.update(const_snapshot)


def _builder_dim_add_const(
    ctx: FunctionContext | MethodContext, dim: int | Type, const: int
) -> int | Type:
    """dim + const, side-effect free like _builder_fold_dim_expr (issue #298)."""
    seen_snapshot = dict(_seen_vars)
    const_snapshot = {k: dict(v) for k, v in _expr_constituents.items()}
    try:
        return _dim_add_const(ctx, "arange", 0, dim, const)
    finally:
        _seen_vars.clear()
        _seen_vars.update(seen_snapshot)
        _expr_constituents.clear()
        _expr_constituents.update(const_snapshot)


def _enum_member_count(info) -> int | None:
    """Member count of an int enum, deduped by value (aliases share one).

    Anything non-conforming bails: only all-int-literal members give a
    provable count, which is exactly the runtime len() of the class.
    """
    names = getattr(info, "enum_members", None)
    if not isinstance(names, list):
        return None
    values: set[int] = set()
    for name in names:
        node = info.names[name].node if name in info.names else None
        if not isinstance(node, Var):
            return None
        value = _int_value(node.type) if node.type is not None else None
        if value is None:
            return None
        values.add(value)
    return len(values)


def _fold_dim_expr(
    ctx: FunctionContext | MethodContext, expr: Expression, index: int
) -> int | Type | None:
    """A size written as arithmetic over known sizes, else None.

    `a.shape[0] * a.shape[1]`, `n + 1`, `2 * a.shape[0]`, `n // 2`:
    mypy types all of these `int`, which loses the size. The operands
    are read from their own types (a literal, or a shape variable) and
    combined with the dim algebra; what the algebra cannot spell (a
    sum times a sum) is a fresh unknown, which is still a size.
    """
    from numty_plugin import _resolve_dim, _materialize_dim, _fresh_unknown
    value = _fold_int(expr)
    if value is not None:
        return value
    if isinstance(expr, OpExpr) and expr.op in ("+", "-", "*", "//"):
        left = _fold_dim_expr(ctx, expr.left, index)
        right = _fold_dim_expr(ctx, expr.right, index)
        if left is None or right is None:
            return None
        op = "shape-arith"
        if expr.op == "+":
            return _dim_add2(ctx, op, index, left, right)
        if expr.op == "-":
            if isinstance(right, int):
                return _dim_add_const(ctx, op, index, left, -right)
            return _dim_add2(ctx, op, index, left, _dim_mul_const(ctx, op, index, right, -1))
        if expr.op == "*":
            if isinstance(right, int):
                return _dim_mul_const(ctx, op, index, left, right)
            if isinstance(left, int):
                return _dim_mul_const(ctx, op, index, right, left)
            fa, fb = _resolve_dim(left), _resolve_dim(right)
            if isinstance(fa, (_LinForm, _MonoForm)) and isinstance(fb, (_LinForm, _MonoForm)):
                product = _form_mul(fa, fb)
                if product is not None:
                    return _materialize_dim(ctx, op, index, product)
            return _fresh_unknown(ctx, op, index)
        if not isinstance(right, int) or right <= 0:
            return None
        if isinstance(left, int):
            return left // right
        return _dim_div(ctx, op, index, left, right, 0)
    lookup = getattr(ctx.api, "lookup_type_or_none", None)
    tp = lookup(expr) if lookup is not None else None
    if tp is None:
        return None
    dim = _dim_value(tp)
    return dim if isinstance(dim, (int, TypeVarType)) else None


def _has_unknown_shape(tp: Type) -> bool:
    """An ndarray whose shape is tuple[Any, ...] or Any (NDArray, bare ndarray)."""
    proper = get_proper_type(tp)
    if not (
        isinstance(proper, Instance)
        and proper.type.fullname == NDARRAY_FULLNAME
        and len(proper.args) == 2
    ):
        return False
    shape = get_proper_type(proper.args[0])
    if isinstance(shape, AnyType):
        return True
    return (
        isinstance(shape, Instance)
        and shape.type.fullname == "builtins.tuple"
        and isinstance(get_proper_type(shape.args[0]), AnyType)
    )


def _unpack_shape(tp: Type) -> tuple[tuple[list[int | Type], UnpackType, list[int | Type]], Type] | None:
    """A (*prefix, *Ts, *suffix) ndarray shape with its dtype, else None."""
    proper = get_proper_type(tp)
    if not isinstance(proper, Instance):
        return None
    if proper.type.fullname != NDARRAY_FULLNAME or len(proper.args) != 2:
        return None
    split = _shape_dims_unpack(proper.args[0])
    if split is None:
        return None
    return split, proper.args[1]
