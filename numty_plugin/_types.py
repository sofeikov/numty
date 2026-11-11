"""_types.py — pure algebra/fold/shape (Phase 5, #199). Owns NDARRAY_FULLNAME
etc. (Phase 5, #199). Top-level imports go to earlier layers only (order in
tests/test_layout.py, _LAYERS). A function-local facade import is used only
for a monkeypatch seam or for a helper in a later layer (_LAZY_UPWARD).
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, cast

from mypy.errorcodes import ARG_TYPE
from mypy.nodes import (
    ARG_NAMED, ARG_NAMED_OPT, ARG_OPT, ARG_POS, ARG_STAR, CallExpr, Context, Expression, IntExpr,
    ListExpr, MemberExpr, NameExpr, OpExpr, StarExpr, TupleExpr, TypeInfo, UnaryExpr, Var,
)
from mypy.messages import callable_name
from mypy.operators import op_methods_to_symbols
from mypy.plugin import AttributeContext, FunctionContext, FunctionSigContext, MethodContext, MethodSigContext
from mypy.types import (
    AnyType, CallableType, Instance, LiteralType, NoneType, TupleType, Type, TypeAliasType,
    TypeOfAny, TypeVarId, TypeVarLikeType, TypeVarTupleType, TypeVarType, UnionType, UnpackType,
    get_proper_type,
)

if TYPE_CHECKING:
    from mypy.checker import TypeChecker

from ._errors import NUMTY_CALLSITE, NUMTY_SHAPE, _swallowed
from ._forms import (
    _DivForm, _LinForm, _MonoForm, _coeff_vars_to_form, _form_has_unknown, _form_mul,
    _forms_provably_ne, _is_unknown_fullname, _lin_add, _lin_scale, _mono_normalize,
)
from ._exprs import (
    _EXPR_PREFIX, _expr_constituents, _expr_forms, _expr_ids, _expr_str, _form_of, _lookup_ns,
    _ns_of_ctx, _seen_vars, _short_var,
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
        if expr.op == "**" and (right < 0 or right > 32):
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
    from numty_plugin import _enum_member_count
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
    from numty_plugin import _unpack_shape
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


def _iter_tvarlikes(
    tp: Type, _seen: frozenset[TypeAliasType] = frozenset()
) -> Iterator[TypeVarLikeType]:
    """Every TypeVar-like nested in an analyzed type (any container).

    A recursive alias (`Json = dict[str, "Json"] | list["Json"]`, or the
    nested-sequence aliases behind numpy's ArrayLike) expands to a type
    that mentions itself, so each alias is expanded once per path:
    without that this recursed until RecursionError, which the
    call-site hooks used to swallow, dropping their checks for the
    whole call (issue #164).
    """
    if isinstance(tp, TypeAliasType):
        if tp in _seen:
            return
        _seen = _seen | {tp}
        for a in tp.args:
            yield from _iter_tvarlikes(a, _seen)
    tp = get_proper_type(tp)
    if isinstance(tp, TypeVarLikeType):
        yield tp
    elif isinstance(tp, Instance):
        for a in tp.args:
            yield from _iter_tvarlikes(a, _seen)
    elif isinstance(tp, TupleType):
        for item in tp.items:
            yield from _iter_tvarlikes(item, _seen)
    elif isinstance(tp, UnionType):
        for item in tp.relevant_items():
            yield from _iter_tvarlikes(item, _seen)
    elif isinstance(tp, UnpackType):
        yield from _iter_tvarlikes(tp.type, _seen)


def _shape_tvar_formals(sig: CallableType) -> list[tuple[int, list[tuple[int, TypeVarType]]]]:
    """Formals with shape TypeVars: (formal index, [(dim index, tvar)]).

    Only plain positional-or-defaulted ndarray formals; anything
    fancier (star args, non-array shapes) stays out of the check.
    """
    out: list[tuple[int, list[tuple[int, TypeVarType]]]] = []
    for i, (tp, kind) in enumerate(zip(sig.arg_types, sig.arg_kinds)):
        if kind not in (ARG_POS, ARG_OPT):
            continue
        parts = _parts_or_unpack(tp)
        if parts is None:
            continue
        positions = [(d, t) for d, t in enumerate(parts[0]) if isinstance(t, TypeVarType)]
        if positions:
            out.append((i, positions))
    return out


def _strip_optional(tp: Type) -> Type:
    """The single non-None union member, else the type itself."""
    proper = get_proper_type(tp)
    if not isinstance(proper, UnionType):
        return tp
    members = [m for m in proper.relevant_items() if not isinstance(get_proper_type(m), NoneType)]
    if len(members) != 1:
        return tp
    return members[0]


_SEQUENCE_FULLNAMES = frozenset(
    {"builtins.list", "collections.abc.Sequence", "typing.Sequence"}
)


def _sequence_element(tp: Type) -> Type | None:
    """Element type of a homogeneous list/Sequence formal, else None."""
    proper = get_proper_type(tp)
    if not isinstance(proper, Instance):
        return None
    if proper.type.fullname not in _SEQUENCE_FULLNAMES:
        return None
    if len(proper.args) != 1:
        return None
    return proper.args[0]


def _align_pair(
    fname: str, fdims: list[int | Type], adims: list[int | Type]
) -> list[tuple[str, int, TypeVarLikeType, int, int | None]]:
    """Align formal dims (possibly with one *Ts segment) to actual dims.

    Returns (formal name, axis, tvar, actual size, segment index) for
    each provable actual int under a shape tvar; unprovable dims are
    skipped. The segment index scopes *Ts comparisons element-wise:
    elements of one unpack segment never conflict with each other,
    only with the same position of another segment.
    """
    raw: list[tuple[int | Type, int | Type, int | None]] = []
    uidx = -1
    unpack_dim: Type | None = None
    for k, fd in enumerate(fdims):
        if isinstance(fd, int):
            continue
        if isinstance(get_proper_type(fd), UnpackType):
            uidx = k
            unpack_dim = fd
            break
    if uidx == -1:
        raw = [(fd, ad, None) for fd, ad in zip(fdims, adims)]
    else:
        pre, suf = fdims[:uidx], fdims[uidx + 1 :]
        if len(adims) < len(pre) + len(suf):
            return []
        raw = [(fd, ad, None) for fd, ad in zip(pre, adims[: len(pre)])]
        if suf:
            raw.extend(
                (fd, ad, None)
                for fd, ad in zip(suf, adims[len(adims) - len(suf) :])
            )
        seg = adims[len(pre) : len(adims) - len(suf) if suf else len(adims)]
        assert unpack_dim is not None
        upack = get_proper_type(unpack_dim)
        assert isinstance(upack, UnpackType)
        tvt = get_proper_type(upack.type)
        if isinstance(tvt, TypeVarTupleType):
            raw.extend((tvt, a, j) for j, a in enumerate(seg))
    out: list[tuple[str, int, TypeVarLikeType, int, int | None]] = []
    for axis, (fd, ad, seg) in enumerate(raw):
        if isinstance(fd, (TypeVarType, TypeVarTupleType)) and isinstance(ad, int):
            out.append((fname, axis, fd, ad, seg))
    return out


def _pairs_for_actual(
    fname: str, formal_tp: Type, actual_expr: Expression, ctx: FunctionSigContext | MethodSigContext
) -> list[tuple[str, int, TypeVarLikeType, int, int | None]]:
    """Pair one formal type with one actual expression (issue #53).

    Bare ndarrays (and single-non-None unions like Optional) align
    directly; homogeneous list/Sequence element types align element-wise
    for list displays (and once for stored lists via the joined arg).
    """
    fparts = _parts_or_unpack(_strip_optional(formal_tp))
    if fparts is not None:
        try:
            actual_tp = ctx.api.get_expression_type(actual_expr)
        except Exception:
            _swallowed(ctx, "_pairs_for_actual")
            return []
        aparts = _parts_or_unpack(actual_tp)
        if aparts is None:
            return []
        return _align_pair(fname, fparts[0], aparts[0])
    elem = _sequence_element(_strip_optional(formal_tp))
    if elem is None:
        return []
    eparts = _parts_or_unpack(elem)
    if eparts is None:
        return []
    edims = eparts[0]
    if isinstance(actual_expr, (ListExpr, TupleExpr)):
        out: list[tuple[str, int, TypeVarLikeType, int, int | None]] = []
        for el in actual_expr.items:
            if isinstance(el, StarExpr):
                continue
            try:
                el_tp = ctx.api.get_expression_type(el)
            except Exception:
                _swallowed(ctx, "_pairs_for_actual")
                continue
            el_parts = _parts_or_unpack(el_tp)
            if el_parts is None:
                continue
            out.extend(_align_pair(fname, edims, el_parts[0]))
        return out
    try:
        actual_tp = ctx.api.get_expression_type(actual_expr)
    except Exception:
        _swallowed(ctx, "_pairs_for_actual")
        return []
    aproper = get_proper_type(actual_tp)
    if isinstance(aproper, TupleType):
        out: list[tuple[str, int, TypeVarLikeType, int, int | None]] = []
        for item in aproper.items:
            iparts = _parts_or_unpack(item)
            if iparts is None:
                continue
            out.extend(_align_pair(fname, edims, iparts[0]))
        return out
    if isinstance(aproper, Instance) and len(aproper.args) == 1:
        iparts = _parts_or_unpack(aproper.args[0])
        if iparts is not None:
            return _align_pair(fname, edims, iparts[0])
    return []


def _conflict_pairs(
    sig: CallableType, ctx: FunctionSigContext | MethodSigContext
) -> list[tuple[str, int, TypeVarLikeType, int, int | None]]:
    """(formal name, axis, tvar, actual size) for the call-site check.

    Covers bare ndarray formals (positional, defaulted, keyword-only
    and named), single-non-None unions (Optional/PEP-604), homogeneous
    list/Sequence element types, *args element types (every actual), and
    *Ts unpack segments element-wise (issue #53).
    """
    out: list[tuple[str, int, TypeVarLikeType, int, int | None]] = []
    if len(ctx.args) != len(sig.arg_types):
        return out
    for i, (tp, kind, name) in enumerate(zip(sig.arg_types, sig.arg_kinds, sig.arg_names)):
        fname = name or "?"
        actuals = ctx.args[i]
        if kind in (ARG_POS, ARG_OPT, ARG_NAMED, ARG_NAMED_OPT):
            if len(actuals) != 1:
                continue
            out.extend(_pairs_for_actual(fname, tp, actuals[0], ctx))
        elif kind == ARG_STAR:
            # *args: the formal type is the element type; every actual
            # binds to it, so each one is checked.
            for expr in actuals:
                out.extend(_pairs_for_actual(fname, tp, expr, ctx))
    return out


def _emit_call_site_conflicts(
    ctx: FunctionSigContext | MethodSigContext,
    pairs: list[tuple[str, int, TypeVarLikeType, int, int | None]],
) -> None:
    """Report shared-TypeVar size conflicts collected from call actuals.

    Occurrences key on (tvar, segment index) so one *Ts segment's own
    elements never conflict with each other, only with the same
    position of another segment.
    """
    by_var: dict[tuple[str, int | None], tuple[str, list[tuple[str, int, int]]]] = {}
    for fname, axis, tvar, size, seg in pairs:
        tname = getattr(tvar, "name", None) or tvar.fullname.rpartition(".")[2]
        key = (tvar.fullname, seg)
        name, occs = by_var.get(key, (tname, []))
        occs.append((fname, axis, size))
        by_var[key] = (name, occs)
    for _, (name, occs) in by_var.items():
        first = occs[0]
        for other in occs[1:]:
            if other[2] != first[2]:
                message = (
                    f"call-site shape conflict for '{name}': "
                    f"{first[2]} (argument '{first[0]}', axis {first[1]}) vs "
                    f"{other[2]} (argument '{other[0]}', axis {other[1]})"
                )
                # No dedup here (issue #44): mypy already collapses
                # identical messages on one line, and a process-lifetime
                # set hides real errors — under dmypy the daemon keeps
                # one plugin instance across rechecks, and in plain mypy
                # the throw-away inference pass inside comprehensions /
                # generator expressions / lambdas consumes the single
                # allowed emission, so the pass that counts stays quiet.
                ctx.api.fail(message, ctx.context, code=NUMTY_CALLSITE)
                break


def _callsite_formula_return(
    sig: CallableType, ctx: FunctionSigContext | MethodSigContext
) -> CallableType:
    """`sig` with formula dims in its return put in the caller's terms.

    mypy cannot infer M+1 from M, so callers of `-> I64["M+1"]` saw Any
    (issue #32). The actual bindings are substituted into the return
    here: plain vars reuse the caller's object, new exprs reuse the
    form-keyed global id. Functions and methods share this.
    """
    try:
        formals = _shape_tvar_formals(sig)
        bindings = _callsite_expr_bindings(sig, formals, ctx)
        new_ret = _substitute_return_exprs(sig, bindings, ctx)
        if new_ret is None:
            return sig
        # Formula variables only the old return mentioned are no longer
        # the callee's to bind (see the note in `_substitute_return_exprs`).
        formal_ids = {tv.id for formal in sig.arg_types for tv in _iter_tvarlikes(formal)}
        return sig.copy_modified(
            ret_type=new_ret,
            variables=[
                v
                for v in sig.variables
                if v.id in formal_ids or not v.fullname.startswith(_EXPR_PREFIX)
            ],
        )
    except Exception:
        _swallowed(ctx, "_callsite_formula_return")
        return sig


def _callsite_expr_bindings(
    sig: CallableType,
    formals: list[tuple[int, list[tuple[int, TypeVarType]]]],
    ctx: FunctionSigContext | MethodSigContext,
) -> dict[str, int | Type | _LinForm]:
    """Callee var fullname -> actual dim for expression returns (issue #32).

    Plain formal dims bind directly to the actual dim object (preserving the
    caller's TypeVar id, which also dodges the #43 stale-object leak).
    Expr formal dims in one name, c*M + k, solve M from the actual's linear
    form: an int, one of the caller's variables, or a linear form in them
    (M+1 vs N+3 gives M = N+2). A name no formal dim can solve stays out
    of the result, and `_substitute_return_exprs` makes the return dims
    that need it unknown.
    """
    from numty_plugin import _has_unknown_shape
    from numty_plugin import _resolve_dim, _fresh_unknown
    bindings: dict[str, int | Type | _LinForm] = {}
    for i, positions in formals:
        if i >= len(ctx.args):
            continue
        actuals = ctx.args[i]
        if len(actuals) != 1:
            continue
        try:
            actual_tp = ctx.api.get_expression_type(actuals[0])
        except Exception:
            _swallowed(ctx, "_callsite_expr_bindings")
            continue
        parts = _parts_or_unpack(actual_tp)
        if parts is None:
            continue
        adims, _ = parts
        for dim, tvar in positions:
            if dim >= len(adims):
                continue
            # For *Ts formals the unpack itself sits in the dims list;
            # a formal Ts never binds like a sized dim (issue #185).
            adim = adims[dim]
            if not isinstance(adim, int) and isinstance(
                get_proper_type(adim), UnpackType
            ):
                continue
            try:
                form_f = _form_of(tvar)
                if form_f is None:
                    # Plain formal: direct bind (first wins).
                    if tvar.fullname not in bindings:
                        bindings[tvar.fullname] = adim
                    continue
                # Expr formal in one name, c*M + k: the actual A gives
                # M = (A - k) / c whenever that is a whole linear form
                # (M+1 vs K+1, M+1 vs 6, M+1 vs N+3, 2*M vs 2*N+4).
                # Anything else leaves M unbound, and the return dims
                # that need it become unknown.
                if not isinstance(form_f, _LinForm) or len(form_f.coeffs) != 1:
                    continue
                m_full, scale = form_f.coeffs[0]
                if scale <= 0 or m_full in bindings:
                    continue
                fa = _resolve_dim(adim)
                if fa is None or not isinstance(fa, _LinForm) or _form_has_unknown(fa):
                    continue
                rest = fa.const - form_f.const
                whole = rest % scale == 0 and all(c % scale == 0 for _, c in fa.coeffs)
                if not fa.coeffs:
                    if not whole or rest < 0:
                        # A fixed size no value of M can produce.
                        ctx.api.fail(
                            f"call-site shape conflict: no size '{_short_var(m_full)}' "
                            f"makes {_expr_str(form_f)} equal {fa.const} "
                            f"(argument '{sig.arg_names[i]}', axis {dim})",
                            ctx.context,
                            code=NUMTY_CALLSITE,
                        )
                        continue
                    bindings[m_full] = rest // scale
                    continue
                if not whole:
                    continue
                solved = _LinForm(tuple((n, c // scale) for n, c in fa.coeffs), rest // scale)
                if solved != _LinForm(((solved.coeffs[0][0], 1),), 0):
                    bindings[m_full] = solved
                    continue
                # M is one of the caller's own names: bind its object,
                # so the result is that variable and not a lookalike.
                a_full = solved.coeffs[0][0]
                found: Type | None = None
                if isinstance(adim, TypeVarType):
                    if adim.fullname == a_full:
                        found = adim
                    else:
                        found = _lookup_ns(
                            _expr_constituents.get(adim.fullname, {}), _ns_of_ctx(ctx), a_full
                        )
                if found is None:
                    found = _lookup_ns(_seen_vars, _ns_of_ctx(ctx), a_full)
                bindings[m_full] = found if found is not None else solved
            except Exception:
                _swallowed(ctx, "_callsite_expr_bindings")
                continue
    # Unknown-rank actuals (plain NDArray) never infer: bind the
    # formal's still-unbound tvars to a fresh unknown so the result
    # keeps its rank instead of collapsing to Never (issue #62).
    for i, positions in formals:
        if i >= len(ctx.args):
            continue
        actuals = ctx.args[i]
        if len(actuals) != 1:
            continue
        try:
            actual_tp = ctx.api.get_expression_type(actuals[0])
        except Exception:
            _swallowed(ctx, "_callsite_expr_bindings")
            continue
        proper_actual = get_proper_type(actual_tp)
        if not (
            _has_unknown_shape(actual_tp)
            or isinstance(proper_actual, AnyType)
            # A *Ts transpose bails to bare Any (issue #185): like an
            # unknown rank, it must not leave the return as the caller's
            # own names (which would read as exact and wrong).
            or (
                isinstance(proper_actual, Instance)
                and proper_actual.type.fullname == NDARRAY_FULLNAME
                and len(proper_actual.args) == 2
                and isinstance(get_proper_type(proper_actual.args[0]), AnyType)
            )
        ):
            continue
        for dim, tvar in positions:
            if tvar.fullname not in bindings:
                try:
                    bindings[tvar.fullname] = _fresh_unknown(ctx, "callsite", dim)
                except Exception:
                    _swallowed(ctx, "_callsite_expr_bindings")
                    continue
    return bindings


def _substitute_return_exprs(
    sig: CallableType,
    bindings: dict[str, int | Type | _LinForm],
    ctx: FunctionSigContext | MethodSigContext,
) -> Type | None:
    """Return type with callee expr vars substituted by call bindings.

    Returns the new return Type, or None when nothing changed / unsupported.
    Plain vars use the actual object directly (caller's id); new expr forms
    reuse the form-keyed global id so an inferred N+1 is the same variable
    as an annotated I64["N+1"].

    A return dim that needs a callee name the call did not bind is
    unknown. Left alone, a plain one is `Never` to mypy, and one inside
    a formula would read as the caller's variable of the same name:
    `-> F64["N+M"]` with only M bound is not the caller's `N` plus
    anything. The exception is a dim that is itself a formal dim
    (`x: F64["N+1"]) -> F64["N+1"]`), which mypy infers on its own.
    """
    from numty_plugin import _resolve_dim, _fresh_unknown
    # Dims mypy infers by itself: every variable a formal mentions,
    # wherever it sits (`x: F64["N+1"]`, `xs: dict[str, F64[N, M]]`).
    inferred: set[str] = set()
    for formal in sig.arg_types:
        try:
            inferred.update(
                tv.fullname for tv in _iter_tvarlikes(formal) if isinstance(tv, TypeVarType)
            )
        except Exception:
            _swallowed(ctx, "_substitute_return_exprs")
            return None
    try:
        ret_parts = _parts_or_unpack(sig.ret_type)
    except Exception:
        _swallowed(ctx, "_substitute_return_exprs")
        return None
    if ret_parts is None:
        return None
    rdims, _dtype = ret_parts
    new_dims: list[int | Type] = []
    changed = False
    # For building new expr vars we need constituent objects by fullname.
    # Caller-ns first (Phase 4 Phase 1 namespaced B keys), then any-ns.
    obj_by_fullname: dict[str, TypeVarType] = {}
    for v in bindings.values():
        if isinstance(v, TypeVarType):
            obj_by_fullname.setdefault(v.fullname, v)
    _ns = _ns_of_ctx(ctx)
    for consts in _expr_constituents.values():
        for (_n, _f), _v in consts.items():
            if _n == _ns:
                obj_by_fullname.setdefault(_f, _v)
    for (_n, _f), _v in _seen_vars.items():
        if _n == _ns:
            obj_by_fullname.setdefault(_f, _v)
    for consts in _expr_constituents.values():
        for (_n, _f), _v in consts.items():
            obj_by_fullname.setdefault(_f, _v)
    for (_n, _f), _v in _seen_vars.items():
        obj_by_fullname.setdefault(_f, _v)
    for index, d in enumerate(rdims):
        try:
            if isinstance(d, int):
                new_dims.append(d)
                continue
            if isinstance(get_proper_type(d), UnpackType):
                # A *Ts segment passes through untouched (issue #185).
                new_dims.append(d)
                continue
            if not isinstance(d, TypeVarType):
                return None
            form_early = _form_of(d)
            if form_early is None:
                bound = bindings.get(d.fullname)
                if bound is None:
                    if d.fullname in inferred:
                        new_dims.append(d)
                    else:
                        new_dims.append(_fresh_unknown(ctx, "callsite-return", index))
                        changed = True
                    continue
                if not isinstance(bound, _LinForm):
                    # Plain return var: direct substitution preserves caller id.
                    new_dims.append(bound)
                    changed = True
                    continue
                form_r = _LinForm(((d.fullname, 1),), 0)
            elif d.fullname in inferred:
                # The dim is a formal dim as it stands (`x: F64["N+1"])
                # -> F64["N+1"]`): mypy binds it from the argument.
                new_dims.append(d)
                continue
            elif isinstance(form_early, _LinForm):
                form_r = form_early
            else:
                return None
            if any(cfull not in bindings for cfull, _ in form_r.coeffs):
                # A name the call did not bind: mypy's own answer if
                # the whole dim is a formal dim, else nothing is known.
                if d.fullname in inferred:
                    new_dims.append(d)
                else:
                    new_dims.append(_fresh_unknown(ctx, "callsite-return", index))
                    changed = True
                continue
            merged: dict[str, int] = {}
            const = form_r.const
            abort = False
            for cfull, coeff in form_r.coeffs:
                b = bindings[cfull]
                if isinstance(b, int):
                    const += coeff * b
                    continue
                fb = b if isinstance(b, _LinForm) else _resolve_dim(b)
                if not isinstance(fb, _LinForm):
                    abort = True
                    break
                for bfull, bcoeff in fb.coeffs:
                    merged[bfull] = merged.get(bfull, 0) + coeff * bcoeff
                const += coeff * fb.const
            if abort:
                new_dims.append(_fresh_unknown(ctx, "callsite-return", index))
                changed = True
                continue
            merged = {k: v for k, v in merged.items() if v != 0}
            if not merged:
                new_dims.append(const)
                changed = True
                continue
            if len(merged) == 1:
                (only_full, only_coeff) = next(iter(merged.items()))
                if only_coeff == 1 and const == 0:
                    # Single var: the caller's object, or nothing.
                    obj = obj_by_fullname.get(only_full)
                    if obj is None:
                        obj = _fresh_unknown(ctx, "callsite-return", index)
                    new_dims.append(obj)
                    changed = True
                    continue
            new_form = _LinForm(tuple(sorted(merged.items())), const)
            new_fullname = f"__numty_expr__.{new_form!r}"
            vid = _expr_ids.get(new_fullname)
            if vid is None:
                vid = TypeVarId.new(0)
                _expr_ids[new_fullname] = vid
            _expr_forms[new_fullname] = new_form
            upper = getattr(d, "upper_bound", None)
            if upper is None:
                return None
            new_obj = TypeVarType(
                name=_expr_str(new_form),
                fullname=new_fullname,
                id=vid,
                values=[],
                upper_bound=upper,
                default=AnyType(TypeOfAny.from_omitted_generics),
                line=getattr(d, "line", 0),
                column=getattr(d, "column", 0),
            )
            # Constituents for later diff() of this inferred expr.
            try:
                const_map: dict[tuple[str, str], TypeVarType] = {}
                for cfull in merged:
                    o = obj_by_fullname.get(cfull)
                    if o is not None:
                        const_map[(_ns_of_ctx(ctx), cfull)] = o
                if const_map:
                    _expr_constituents.setdefault(new_fullname, {}).update(
                        const_map
                    )
            except Exception:
                _swallowed(ctx, "_substitute_return_exprs")
                pass
            new_dims.append(new_obj)
            changed = True
        except Exception:
            _swallowed(ctx, "_substitute_return_exprs")
            return None
    if not changed:
        return None
    # A formula's variable is keyed by its form, so the caller's `N+1`
    # and a `"N+1"` in this signature are one TypeVar when both are
    # written over the same module-level `N`. mypy binds the callee's
    # variables after this hook, and would read a caller's dim put into
    # the return as the callee's own: rebound to whatever that formal
    # received, or `Never`. `_callsite_check` drops the ones no formal
    # mentions; one a formal does mention cannot be told apart, so the
    # dim is unknown.
    formal_ids = {tv.id for formal in sig.arg_types for tv in _iter_tvarlikes(formal)}
    for index, (old_d, new_d) in enumerate(zip(rdims, new_dims)):
        if (
            new_d is not old_d
            and isinstance(new_d, TypeVarType)
            and new_d.fullname.startswith(_EXPR_PREFIX)
            and new_d.id in formal_ids
        ):
            new_dims[index] = _fresh_unknown(ctx, "callsite-return", index)
    # Rebuild the ndarray return with the new dims.
    try:
        proper = get_proper_type(sig.ret_type)
        if not isinstance(proper, Instance) or len(proper.args) != 2:
            return None
        shape = get_proper_type(proper.args[0])
        if not isinstance(shape, TupleType):
            return None
        if len(shape.items) != len(new_dims):
            return None
        new_items: list[Type] = []
        for old_item, new_d in zip(shape.items, new_dims):
            if isinstance(new_d, int):
                old_proper = get_proper_type(old_item)
                if isinstance(old_proper, LiteralType):
                    if old_proper.value == new_d:
                        new_items.append(old_item)
                    else:
                        new_items.append(
                            LiteralType(
                                new_d, old_proper.fallback, line=old_proper.line, column=old_proper.column
                            )
                        )
                elif isinstance(old_proper, TypeVarType) and isinstance(
                    get_proper_type(old_proper.upper_bound), Instance
                ):
                    fb = get_proper_type(old_proper.upper_bound)
                    assert isinstance(fb, Instance)
                    new_items.append(
                        LiteralType(new_d, fb, line=old_proper.line, column=old_proper.column)
                    )
                else:
                    # Int from a plain bound var: reuse its Literal fallback
                    # when the bound object carries one, else bail.
                    made: Type | None = None
                    for v in bindings.values():
                        if isinstance(v, (int, _LinForm)):
                            continue
                        vp = get_proper_type(v)
                        if isinstance(vp, LiteralType):
                            made = LiteralType(
                                new_d, vp.fallback, line=vp.line, column=vp.column
                            )
                            break
                    if made is None:
                        return None
                    new_items.append(made)
            else:
                if new_d is old_item:
                    new_items.append(old_item)
                else:
                    new_items.append(new_d)
        try:
            new_shape = shape.copy_modified(items=new_items)
        except Exception:
            _swallowed(ctx, "_substitute_return_exprs")
            new_shape = TupleType(new_items, shape.partial_fallback, line=shape.line, column=shape.column)
        try:
            return proper.copy_modified(args=[new_shape, proper.args[1]])
        except Exception:
            _swallowed(ctx, "_substitute_return_exprs")
            return Instance(proper.type, [new_shape, proper.args[1]], line=proper.line, column=proper.column)
    except Exception:
        _swallowed(ctx, "_substitute_return_exprs")
        return None


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
    from numty_plugin import _fold_dim_expr
    seen_snapshot = dict(_seen_vars)
    const_snapshot = {k: dict(v) for k, v in _expr_constituents.items()}
    try:
        return _fold_dim_expr(ctx, expr, index)
    finally:
        _seen_vars.clear()
        _seen_vars.update(seen_snapshot)
        _expr_constituents.clear()
        _expr_constituents.update(const_snapshot)


