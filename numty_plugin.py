"""mypy plugin: precise numpy shapes for constructors, matmul, and friends.

mypy-only, registered in mypy.ini. Without the plugin, numty.py still
checks with plain annotations; only the general solving below is lost
(matmul degrades to its loose fallback, stored constructors widen).

What it does:
- zeros/ones/empty with a fully literal shape return a precise
  ndarray[tuple[Literal[...], ...], ...], even through assignment.
  Anything else (dynamic dims, explicit dtype) bails out to mypy's
  default instead of erroring.
- numty.matmul solves inner dims for any rank, broadcasts the batch
  dims with NumPy rules, and propagates the outer dims. Dtype is taken
  from the left operand. Over a shared *Ts, (*Ts, N, M) pairs solve the
  same way (inner dims align, batches broadcast through the unpack; a
  lone concrete 1-D/2-D operand contributes an empty batch). The @
  operator is accepted but unchecked (see the method-hook note below).
- __add__/__mul__ unify operand shapes with NumPy broadcast rules, so
  precision survives whole function bodies. Scalars pass through;
  unknown dims bail out.
- reshape (method and np.reshape) checks total-size conservation and
  infers a single -1, returning the precise new shape. Over *Ts only
  block-preserving regroups resolve (the unknown middle stays put; -1
  would absorb it, so it bails), with provable int-only total clashes
  failing loudly. Anything dynamic bails out.
- rank-changing ops (transpose, .T, squeeze, expand_dims, sum/prod/mean
  and friends) track the new shape; transpose reversal over *Ts keeps
  the unknown block ordered in place (explicit axes bail). No-axis
  reductions and non-float dtypes bail out to the stubs.
- shape builders (*_like from the prototype, full, eye, arange,
  concatenate/stack over tuple or list displays) compute the result shape.
  An explicit, resolvable dtype= is used as-is; unknown dims bail out.
- np.array/asarray build from displays, .astype converts, len() folds
  known first dims, .shape reports literal (and symbolic) dims, and
  unary -/~ keep the shape.
- comparisons broadcast shapes and return bool arrays; argmax/argsort
  drop or keep the axis; Generator/RandomState methods solve from size=.
- getitem solves integer-array (fancy) indices and take_along_axis;
  boolean masks bail (data-dependent count), float indices error. Over
  *Ts, leading items consume the known prefix without an ellipsis
  (rank-generic bodies still spell the ellipsis for the middle).
- dims may be symbolic TypeVars: same name unifies, ranks and counts
  still check, and anything needing values bails instead of erroring.
- arithmetic operators (-, /, **) and bare ufuncs (np.sqrt, np.add, ...)
  broadcast and promote; the ufuncs are stub variables, reached through
  the Literal name in their generic type (raw np.matmul included).
  np.where/clip/broadcast_to/asarray solve; unary math ufuncs track
  int->float64 and complex->real rules.
- einsum parses the subscript string (explicit/implicit, ellipsis,
  diagonals) and solves label sizes; linalg solvers (solve, inv,
  cholesky, eig/eigh, svd, lstsq, norm) check squareness/inner dims,
  solve output shapes (tuples for eig/svd, solution-only for lstsq),
  and track result dtypes. Scalar results bail to the stubs.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import partial

from mypy.nodes import (
    ARG_NAMED,
    ARG_OPT,
    ARG_POS,
    ARG_STAR,
    ARG_STAR2,
    AssignmentStmt,
    CallExpr,
    EllipsisExpr,
    Expression,
    IntExpr,
    ListComprehension,
    ListExpr,
    MemberExpr,
    NameExpr,
    OpExpr,
    OverloadedFuncDef,
    SliceExpr,
    StarExpr,
    StrExpr,
    TupleExpr,
    TypeInfo,
    UnaryExpr,
    Var,
)
from mypy.plugin import (
    AttributeContext,
    FunctionContext,
    FunctionSigContext,
    MethodContext,
    Plugin,
)
from mypy.types import (
    AnyType,
    CallableType,
    Instance,
    LiteralType,
    NoneType,
    Overloaded,
    TupleType,
    Type,
    TypeOfAny,
    TypeVarTupleType,
    TypeVarType,
    UnionType,
    UnpackType,
    get_proper_type,
)

NDARRAY_FULLNAME = "numpy.ndarray"
INT_FULLNAME = "builtins.int"
TUPLE_FULLNAME = "builtins.tuple"
FLOAT64_FULLNAME = "numpy.float64"
DTYPE_FULLNAME = "numpy.dtype"


def plugin(version: str) -> type[Plugin]:
    return NumtyPlugin


class NumtyPlugin(Plugin):
    def __init__(self, options) -> None:
        super().__init__(options)
        self._seen_conflicts: set[tuple[str, int, str]] = set()

    def get_function_signature_hook(
        self, fullname: str
    ) -> Callable[[FunctionSigContext], Type] | None:
        # Dedicated hooks own numpy/numty calls; everything else with
        # shape TypeVars in its formals gets the call-site conflict
        # check. Overloads are skipped: the hook runs per candidate and
        # a non-matching item would false-positive.
        if fullname.startswith("numpy.") or fullname.startswith("numty."):
            return None
        return self.callsite_signature_hook

    def callsite_signature_hook(self, ctx: FunctionSigContext) -> Type:
        try:
            return self._callsite_check(ctx)
        except Exception:
            return ctx.default_signature

    def _callsite_check(self, ctx: FunctionSigContext) -> Type:
        sig = ctx.default_signature
        if not isinstance(sig, CallableType):
            return sig
        if isinstance(ctx.context, CallExpr) and isinstance(
            ctx.context.callee.node if hasattr(ctx.context.callee, "node") else None,
            OverloadedFuncDef,
        ):
            # The hook runs per overload candidate; a non-matching
            # item would false-positive, so overloaded callees stay out.
            return sig
        formals = _shape_tvar_formals(sig)
        if not formals or len(ctx.args) != len(sig.arg_types):
            return sig
        by_var: dict[str, tuple[str, list[tuple[str, int]]]] = {}
        for i, positions in formals:
            actuals = ctx.args[i]
            if len(actuals) != 1:
                continue
            try:
                actual = ctx.api.get_expression_type(actuals[0])
            except Exception:
                continue
            parts = _ndarray_parts(actual)
            if parts is None:
                continue
            adims, _ = parts
            formal_name = sig.arg_names[i] or "?"
            for dim, tvar in positions:
                if dim >= len(adims) or not isinstance(adims[dim], int):
                    continue
                name, occs = by_var.get(tvar.fullname, (tvar.name, []))
                occs.append((formal_name, adims[dim]))
                by_var[tvar.fullname] = (name, occs)
        for _, (name, occs) in by_var.items():
            first = occs[0]
            for other in occs[1:]:
                if other[1] != first[1]:
                    message = (
                        f"call-site shape conflict for '{name}': "
                        f"{first[1]} (argument '{first[0]}') vs "
                        f"{other[1]} (argument '{other[0]}')"
                    )
                    key = (ctx.api.path, ctx.context.line, message)
                    if key not in self._seen_conflicts:
                        self._seen_conflicts.add(key)
                        ctx.api.fail(message, ctx.context)
                    break
        return sig

    def get_function_hook(self, fullname: str) -> Callable[[FunctionContext], Type] | None:
        # numpy re-exports these from private modules (e.g.
        # numpy._core.multiarray), so match the public tail.
        short = fullname.rsplit(".", 1)[-1]
        if fullname == "builtins.len":
            return len_function_hook
        if fullname.startswith("numpy.") and short in {"zeros", "ones", "empty"}:
            return zeros_like_hook
        # Raw np.matmul solves through the ufunc __call__ hook; the
        # numty.matmul wrapper below stays as a convenience alias.
        if fullname == "numty.matmul":
            return matmul_hook
        # np.reshape lives in numpy._core.fromnumeric; match the tail like
        # the constructors above.
        if fullname.startswith("numpy.") and short == "reshape":
            return reshape_function_hook
        if fullname.startswith("numpy.") and short == "transpose":
            return transpose_function_hook
        if fullname.startswith("numpy.") and short == "squeeze":
            return squeeze_function_hook
        if fullname.startswith("numpy.") and short == "expand_dims":
            return expand_dims_function_hook
        if fullname.startswith("numpy.") and short in {
            "sum",
            "prod",
            "mean",
            "min",
            "max",
            "std",
            "var",
        }:
            return reduction_function_hook
        if fullname.startswith("numpy.") and short in {"any", "all"}:
            return allany_function_hook
        if fullname.startswith("numpy.") and short in {
            "zeros_like",
            "ones_like",
            "empty_like",
            "full_like",
        }:
            return like_function_hook
        if fullname.startswith("numpy.") and short == "full":
            return full_function_hook
        if fullname.startswith("numpy.") and short == "eye":
            return eye_function_hook
        if fullname.startswith("numpy.") and short == "arange":
            return arange_function_hook
        if fullname.startswith("numpy.") and short == "concatenate":
            return concatenate_function_hook
        if fullname.startswith("numpy.") and short == "stack":
            return stack_function_hook
        if fullname.startswith("numpy.") and short == "einsum":
            return einsum_function_hook
        if fullname.startswith("numpy.") and short in {
            "add",
            "multiply",
            "subtract",
            "power",
        }:
            return broadcast_function_hook
        if fullname.startswith("numpy.") and short in {"divide", "true_divide"}:
            return truediv_function_hook
        if fullname.startswith("numpy.") and short == "where":
            return where_function_hook
        if fullname.startswith("numpy.") and short == "clip":
            return clip_function_hook
        if fullname.startswith("numpy.") and short == "broadcast_to":
            return broadcast_to_function_hook
        if fullname.startswith("numpy.") and short in {"asarray", "asanyarray", "array"}:
            return asarray_function_hook
        if fullname.startswith("numpy.") and short == "take_along_axis":
            return take_along_axis_function_hook
        if fullname.startswith("numpy.") and short in _UFUNC_KEEP:
            return partial(ufunc_function_hook, group="keep")
        if fullname.startswith("numpy.") and short in _UFUNC_ABS:
            return partial(ufunc_function_hook, group="abs")
        if fullname.startswith("numpy.") and short in _UFUNC_FLOAT:
            return partial(ufunc_function_hook, group="float")
        if fullname.startswith("numpy.") and short in _UFUNC_REAL:
            return partial(ufunc_function_hook, group="real")
        if fullname.startswith("numpy.") and short in _UFUNC_BOOL:
            return partial(ufunc_function_hook, group="bool")
        if fullname.startswith("numpy.linalg.") and short in {
            "solve",
            "inv",
            "cholesky",
            "eig",
            "eigh",
            "svd",
            "lstsq",
            "norm",
        }:
            return partial(linalg_function_hook, op=short)
        if fullname.startswith("numpy.") and short in {
            "argmax",
            "argmin",
            "argsort",
            "sort",
            "partition",
            "argpartition",
        }:
            return partial(arg_function_hook, op=short)
        if fullname.startswith("numpy.random._generator.Generator.") and short in {
            "normal",
            "integers",
            "random",
            "standard_normal",
            "uniform",
            "choice",
            "permutation",
        }:
            return partial(rng_method_hook, op=short)
        return None

    def get_attribute_hook(
        self, fullname: str
    ) -> Callable[[AttributeContext], Type] | None:
        if fullname == "numpy.ndarray.shape":
            return shape_attribute_hook
        # T lives on the shared _ArrayOrScalarCommon base in the stubs.
        if fullname == "numpy._ArrayOrScalarCommon.T":
            return t_attribute_hook
        return None

    def get_method_hook(self, fullname: str) -> Callable[[MethodContext], Type] | None:
        # NOTE: no __matmul__ hook. In testing, operator dispatch consulted
        # the method hook inconsistently across statements for @ (silent
        # when it should fire, once misattributed to another line), while
        # the numty.matmul function hook stayed deterministic. @ therefore
        # uses numpy's default checking; use the wrapper for solving.
        if fullname in {
            "numpy.ndarray.__add__",
            "numpy.ndarray.__radd__",
            "numpy.ndarray.__mul__",
            "numpy.ndarray.__rmul__",
            "numpy.ndarray.__sub__",
            "numpy.ndarray.__rsub__",
            "numpy.ndarray.__pow__",
            "numpy.ndarray.__rpow__",
        }:
            return broadcast_method_hook
        if fullname == "numpy.ndarray.clip":
            return clip_method_hook
        # Bare ufunc objects (np.sqrt, np.add, ...) are stub variables, not
        # functions, so get_function_hook never sees them. Their generic
        # type carries the ufunc name as a Literal, which this hook reads.
        if fullname in {
            "numpy.ufunc.__call__",
            "numpy._typing._ufunc._UFunc_Nin1_Nout1.__call__",
            "numpy._typing._ufunc._UFunc_Nin2_Nout1.__call__",
            "numpy._typing._ufunc._UFunc_Nin1_Nout2.__call__",
            "numpy._typing._ufunc._UFunc_Nin2_Nout2.__call__",
            "numpy._typing._ufunc._GUFunc_Nin2_Nout1.__call__",
        }:
            return ufunc_call_hook
        if fullname in {
            "numpy.ndarray.__truediv__",
            "numpy.ndarray.__rtruediv__",
        }:
            return truediv_method_hook
        if fullname in {
            "numpy.ndarray.__invert__",
            "numpy.ndarray.__neg__",
            "numpy.ndarray.__pos__",
        }:
            return identity_method_hook
        if fullname == "numpy.ndarray.astype":
            return astype_method_hook
        if fullname == "numpy.ndarray.reshape":
            return reshape_method_hook
        if fullname == "numpy.ndarray.transpose":
            return transpose_method_hook
        if fullname == "numpy.ndarray.squeeze":
            return squeeze_method_hook
        if fullname == "numpy.ndarray.__getitem__":
            return getitem_method_hook
        short = fullname.rsplit(".", 1)[-1]
        if fullname.startswith("numpy.ndarray.") and short in {
            "__eq__",
            "__ne__",
            "__lt__",
            "__le__",
            "__gt__",
            "__ge__",
        }:
            return compare_method_hook
        if fullname.startswith("numpy.ndarray.") and short in {
            "__and__",
            "__rand__",
            "__or__",
            "__ror__",
        }:
            return logical_method_hook
        if fullname.startswith("numpy.ndarray.") and short in {"any", "all"}:
            return allany_method_hook
        if fullname.startswith("numpy.ndarray.") and short in {
            "argmax",
            "argmin",
            "argsort",
            "sort",
            "partition",
            "argpartition",
        }:
            return partial(arg_method_hook, op=short)
        if fullname.startswith("numpy.random._generator.Generator.") and short in {
            "normal",
            "integers",
            "random",
            "standard_normal",
            "uniform",
            "choice",
            "permutation",
        }:
            return partial(rng_method_hook, op=short)
        if fullname.startswith("numpy.random.mtrand.RandomState.") and short in {
            "normal",
            "integers",
            "random",
            "standard_normal",
            "uniform",
            "choice",
            "permutation",
        }:
            return partial(rng_method_hook, op=short)
        short = fullname.rsplit(".", 1)[-1]
        if fullname.startswith("numpy.ndarray.") and short in {
            "sum",
            "prod",
            "mean",
            "min",
            "max",
            "std",
            "var",
        }:
            return reduction_method_hook
        return None


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
    """Provably-equal dims: same ints, or the same TypeVar by fullname."""
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, TypeVarType) and isinstance(b, TypeVarType):
        return a.fullname == b.fullname
    return False


def _dim_str(d: int | Type) -> str:
    if isinstance(d, int):
        return str(d)
    if isinstance(d, UnpackType):
        inner = d.type
        name = getattr(inner, "name", None)
        return f"*{name}" if isinstance(name, str) else "*Ts"
    name = getattr(d, "name", None)
    return str(name) if isinstance(name, str) else "?"


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
        ctx.api.fail(message, ctx.context)
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


def _shape_tvar_formals(sig: CallableType) -> list[tuple[int, list[tuple[int, TypeVarType]]]]:
    """Formals with shape TypeVars: (formal index, [(dim index, tvar)]).

    Only plain positional-or-defaulted ndarray formals; anything
    fancier (star args, non-array shapes) stays out of the check.
    """
    out: list[tuple[int, list[tuple[int, TypeVarType]]]] = []
    for i, (tp, kind) in enumerate(zip(sig.arg_types, sig.arg_kinds)):
        if kind not in (ARG_POS, ARG_OPT):
            continue
        parts = _ndarray_parts(tp)
        if parts is None:
            continue
        positions = [(d, t) for d, t in enumerate(parts[0]) if isinstance(t, TypeVarType)]
        if positions:
            out.append((i, positions))
    return out


def _ndarray_type(
    ctx: FunctionContext | MethodContext | AttributeContext,
    dims: list[int | Type],
    dtype: Type,
) -> Instance:
    checker = ctx.api
    int_instance = checker.named_generic_type(INT_FULLNAME, [])
    items: list[Type] = [
        LiteralType(v, fallback=int_instance) if isinstance(v, int) else v for v in dims
    ]
    tuple_fallback = checker.named_generic_type(TUPLE_FULLNAME, [AnyType(TypeOfAny.explicit)])
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


def _shape_from_arg(tp: Type, expr: Expression | None) -> list[int] | None:
    """Dims from an argument: fold the expression first, types as fallback."""
    if expr is not None:
        dims = _fold_shape_expr(expr)
        if dims is not None:
            return dims
    shape = get_proper_type(tp)
    if isinstance(shape, TupleType):
        return _shape_dims(shape)
    single = _int_value(tp)
    return [single] if single is not None else None


def _default_float_dtype(ctx: FunctionContext | MethodContext) -> Type:
    return ctx.api.named_generic_type(
        DTYPE_FULLNAME, [ctx.api.named_generic_type(FLOAT64_FULLNAME, [])]
    )


def zeros_like_hook(ctx: FunctionContext) -> Type:
    shape_arg = _single_shape_arg(ctx)
    if shape_arg is None:
        return ctx.default_return_type
    dims = _shape_from_arg(shape_arg, _single_shape_expr(ctx))
    if dims is None:
        return ctx.default_return_type
    if _explicit_dtype(ctx):
        dtype = _named_dtype(ctx)
        if dtype is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, dtype)
    return _ndarray_type(ctx, dims, _default_float_dtype(ctx))


def _matmul_inner(x: int | Type, y: int | Type) -> tuple[bool, str | None]:
    """(match, error) for inner dims: match on equality, error when both
    ints differ, and (False, None) to bail on anything unprovable."""
    if _dim_eq(x, y):
        return True, None
    if isinstance(x, int) and isinstance(y, int):
        return False, f"matmul inner dimensions disagree: {x} vs {y}"
    return False, None


def _solve_matmul_inner(
    dims: list[int | Type] | None, message: str | None, matched: bool
) -> tuple[list[int | Type] | None, str | None]:
    if message is not None:
        return None, message
    if not matched:
        return None, None
    assert dims is not None
    return dims, None


def _matmul_result(
    a: list[int | Type], b: list[int | Type]
) -> tuple[list[int | Type] | None, str | None]:
    """Output dims on success; (None, None) to bail; (None, msg) on error."""
    if not a or not b:
        return None, None
    if len(a) == 1 and len(b) == 1:
        matched, message = _matmul_inner(a[0], b[0])
        return _solve_matmul_inner([], message, matched)
    if len(a) == 1:
        matched, message = _matmul_inner(a[0], b[-2])
        return _solve_matmul_inner([*b[:-2], b[-1]], message, matched)
    if len(b) == 1:
        matched, message = _matmul_inner(a[-1], b[0])
        return _solve_matmul_inner(a[:-1], message, matched)
    matched, message = _matmul_inner(a[-1], b[-2])
    if message is not None:
        return None, message
    if not matched:
        return None, None
    batch_a, batch_b = a[:-2], b[:-2]
    batch, cause = _broadcast_result(batch_a, batch_b)
    if cause is not None:
        return None, (
            f"matmul batch dimensions disagree: {_dims_str(batch_a)} vs {_dims_str(batch_b)}"
            f" ({cause})"
        )
    assert batch is not None
    return [*batch, a[-2], b[-1]], None


def _matmul_unpack_dims(
    left: Type, right: Type
) -> tuple[list[int | Type] | None, str | None, Type | None, Type | None]:
    """Batch-broadcast matmul dims when at least one side carries *Ts.

    Returns (dims, message, dtype_a, dtype_b); (None, None, None, None)
    to bail, (None, msg, None, None) on a provable inner/batch clash.
    Both unpack sides must share one TypeVarTuple and expose two known
    trailing dims; a lone concrete 1-D/2-D operand contributes an empty
    batch. Vectors over *Ts, mixed unpacks, and concrete batches against
    *Ts stay out of scope and bail.
    """
    sa = _unpack_shape(left)
    sb = _unpack_shape(right)
    if sa is None and sb is None:
        return None, None, None, None
    if sa is not None and sb is not None:
        (p1, u1, s1), dtype_a = sa
        (p2, u2, s2), dtype_b = sb
        if not _same_unpack(u1, u2) or len(s1) < 2 or len(s2) < 2:
            return None, None, None, None
        matched, message = _matmul_inner(s1[-1], s2[-2])
        if message is not None:
            return None, message, None, None
        if not matched:
            return None, None, None, None
        batch, cause = _merge_unpack_dims(p1, u1, s1[:-2], p2, u2, s2[:-2])
        if cause is not None:
            return None, (
                f"matmul batch dimensions disagree: "
                f"{_dims_str(p1 + [u1] + s1[:-2])} vs {_dims_str(p2 + [u2] + s2[:-2])} "
                f"({cause})"
            ), None, None
        if batch is None:
            return None, None, None, None
        return [*batch, s1[-2], s2[-1]], None, dtype_a, dtype_b
    sel, other = (sa, right) if sa is not None else (sb, left)
    assert sel is not None
    (prefix, unpack, suffix), dtype_u = sel
    if len(suffix) < 2:
        return None, None, None, None
    ob = _ndarray_parts(other)
    if ob is None or len(ob[0]) > 2:
        return None, None, None, None
    dims_o, dtype_o = ob
    if len(dims_o) == 2:
        pair = (suffix[-1], dims_o[-2]) if sa is not None else (dims_o[-1], suffix[-2])
        matched, message = _matmul_inner(*pair)
        if message is not None:
            return None, message, None, None
        if not matched:
            return None, None, None, None
        if sa is not None:
            dims = [*prefix, unpack, *suffix[:-2], suffix[-2], dims_o[-1]]
            return dims, None, dtype_u, dtype_o
        dims = [*prefix, unpack, *suffix[:-2], dims_o[-2], suffix[-1]]
        return dims, None, dtype_o, dtype_u
    if len(dims_o) == 1:
        pair = (suffix[-1], dims_o[0]) if sa is not None else (dims_o[0], suffix[-2])
        matched, message = _matmul_inner(*pair)
        if message is not None:
            return None, message, None, None
        if not matched:
            return None, None, None, None
        tail = [suffix[-2]] if sa is not None else [suffix[-1]]
        dims = [*prefix, unpack, *suffix[:-2], *tail]
        if sa is not None:
            return dims, None, dtype_u, dtype_o
        return dims, None, dtype_o, dtype_u
    return None, None, None, None


def _matmul_promoted(
    ctx: FunctionContext | MethodContext,
    dims: list[int | Type],
    dtype_a: Type,
    dtype_b: Type,
) -> Type:
    """Result array for matmul output dims with promoted dtypes."""
    name = _promote_dtypes(dtype_a, dtype_b)
    if name is None:
        return ctx.default_return_type
    if name == _dtype_fullname(dtype_a):
        return _ndarray_type(ctx, dims, dtype_a)
    if name == _dtype_fullname(dtype_b):
        return _ndarray_type(ctx, dims, dtype_b)
    dtype = _construct_dtype(ctx, name)
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def _solve_matmul(ctx: FunctionContext | MethodContext, left: Type, right: Type) -> Type:
    parts_a = _ndarray_parts(left)
    parts_b = _ndarray_parts(right)
    if parts_a is not None and parts_b is not None:
        (dims_a, dtype_a), (dims_b, dtype_b) = parts_a, parts_b
        dims, message = _matmul_result(dims_a, dims_b)
        if message is not None:
            ctx.api.fail(message, ctx.context)
            return ctx.default_return_type
        if dims is None:
            return ctx.default_return_type
        return _matmul_promoted(ctx, dims, dtype_a, dtype_b)
    dims, message, dtype_a, dtype_b = _matmul_unpack_dims(left, right)
    if message is not None:
        ctx.api.fail(message, ctx.context)
        return ctx.default_return_type
    if dims is None or dtype_a is None or dtype_b is None:
        return ctx.default_return_type
    return _matmul_promoted(ctx, dims, dtype_a, dtype_b)


def _positional_actuals(ctx: FunctionContext | MethodContext, n: int) -> list[Type] | None:
    """The last n positional actuals (method self excluded), else None."""
    actual = [kind for group in ctx.arg_kinds for kind in group]
    if actual != [ARG_POS] * n or len(ctx.arg_types) < n:
        return None
    groups = ctx.arg_types[-n:]
    if any(not group for group in groups):
        return None
    return [group[0] for group in groups]


def matmul_hook(ctx: FunctionContext) -> Type:
    args = _positional_actuals(ctx, 2)
    if args is None:
        return ctx.default_return_type
    return _solve_matmul(ctx, args[0], args[1])


def _dims_str(dims: list[int | Type]) -> str:
    """Shape for messages: tuple-style, symbolic dims by name."""
    inner = ", ".join(_dim_str(d) for d in dims)
    if len(dims) == 1:
        inner += ","
    return f"({inner})"


def _broadcast_result(
    a: list[int | Type], b: list[int | Type]
) -> tuple[list[int | Type] | None, str | None]:
    """Broadcast two known shapes; (None, None) to bail; (None, msg) on error.

    Same-name TypeVars are equal; anything else unprovable bails out.
    """
    out: list[int | Type] = []
    for k, (x, y) in enumerate(zip(reversed(a), reversed(b))):
        if _dim_eq(x, y):
            out.append(x)
        elif isinstance(x, int) and x == 1:
            out.append(y)
        elif isinstance(y, int) and y == 1:
            out.append(x)
        elif isinstance(x, int) and isinstance(y, int):
            return None, (
                f"broadcast shapes disagree at trailing axis {k}: "
                f"{x} vs {y} (shapes {_dims_str(a)} vs {_dims_str(b)})"
            )
        else:
            return None, None
    shared = min(len(a), len(b))
    prefix = a[: len(a) - shared] + b[: len(b) - shared]
    return [*prefix, *reversed(out)], None


def _same_unpack(a: UnpackType, b: UnpackType) -> bool:
    """Both unpacks draw from the same TypeVarTuple declaration."""
    t1 = get_proper_type(a.type)
    t2 = get_proper_type(b.type)
    return (
        isinstance(t1, TypeVarTupleType)
        and isinstance(t2, TypeVarTupleType)
        and t1.fullname == t2.fullname
    )


def _merge_known_pair(
    x: int | Type, y: int | Type, axis: str, full: str
) -> tuple[int | Type | None, str | None]:
    """One pairwise broadcast step: (merged dim, None), or (None, msg).

    (None, None) bails out silently; (None, msg) records a provable
    int clash for the caller to fail with.
    """
    if _dim_eq(x, y):
        return x, None
    if isinstance(x, int) and x == 1:
        return y, None
    if isinstance(y, int) and y == 1:
        return x, None
    if isinstance(x, int) and isinstance(y, int):
        return None, (
            f"broadcast shapes disagree at {axis}: {x} vs {y} (shapes {full})"
        )
    return None, None


def _merge_unpack_dims(
    p1: list[int | Type],
    u1: UnpackType,
    s1: list[int | Type],
    p2: list[int | Type],
    u2: UnpackType,
    s2: list[int | Type],
) -> tuple[list[int | Type] | None, str | None]:
    """Broadcast two (*prefix, *Ts, *suffix) shapes, or None to bail.

    The unpacks must draw from the same TypeVarTuple (equal unknown
    length, so they consume each other); suffixes must match in count
    (extras would align against *Ts); prefixes right-align with leading
    extras passing through, exactly like standard broadcasting.
    Returns (dims, None), (None, msg) on a provable clash, or
    (None, None) to bail out silently.
    """
    if not _same_unpack(u1, u2):
        return None, None
    if len(s1) != len(s2):
        return None, None
    full = f"{_dims_str(p1 + [u1] + s1)} vs {_dims_str(p2 + [u2] + s2)}"
    suf: list[int | Type] = []
    for k, (x, y) in enumerate(zip(reversed(s1), reversed(s2))):
        merged, message = _merge_known_pair(x, y, f"trailing axis {k}", full)
        if message is not None or merged is None:
            return None, message
        suf.append(merged)
    suf.reverse()
    shared = min(len(p1), len(p2))
    pre: list[int | Type] = []
    for k, (x, y) in enumerate(zip(reversed(p1), reversed(p2))):
        axis = f"axis {len(p1) - 1 - k}" if len(p1) == len(p2) else "leading dims"
        merged, message = _merge_known_pair(x, y, axis, full)
        if message is not None or merged is None:
            return None, message
        pre.append(merged)
    pre.reverse()
    dims = p1[: len(p1) - shared] + p2[: len(p2) - shared] + pre + [u1] + suf
    return dims, None


def _solve_broadcast_scalar(
    ctx: FunctionContext | MethodContext, dims: list[int | Type], dtype: Type, other: Type
) -> Type:
    """Array/scalar promotion keeping the array dims, or default."""
    key = _scalar_key(other)
    if key is None:
        return ctx.default_return_type
    strong = _dtype_fullname(dtype)
    if strong is None:
        return ctx.default_return_type
    kind, scalar_name = key
    name = (
        _promote_weak(scalar_name, strong)
        if kind == "weak"
        else _promote_strong(scalar_name, strong)
    )
    if name is None:
        return ctx.default_return_type
    if name == strong:
        return _ndarray_type(ctx, dims, dtype)
    promoted = _construct_dtype(ctx, name)
    if promoted is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, promoted)


def _solve_broadcast_unpack(
    ctx: FunctionContext | MethodContext,
    left: Type,
    right: Type,
    a: tuple[list[int | Type], Type] | None,
    b: tuple[list[int | Type], Type] | None,
) -> Type:
    """Broadcast where _ndarray_parts bailed on at least one side."""
    sa = _unpack_shape(left) if a is None else None
    sb = _unpack_shape(right) if b is None else None
    if sa is None and sb is None:
        if a is None:
            return ctx.default_return_type
        return _solve_broadcast_scalar(ctx, a[0], a[1], right)
    if sa is not None and sb is not None:
        (p1, u1, s1), dtype_a = sa
        (p2, u2, s2), dtype_b = sb
        dims, message = _merge_unpack_dims(p1, u1, s1, p2, u2, s2)
        if message is not None:
            ctx.api.fail(message, ctx.context)
            return ctx.default_return_type
        if dims is None:
            return ctx.default_return_type
        name = _promote_dtypes(dtype_a, dtype_b)
        if name is None:
            return ctx.default_return_type
        if name == _dtype_fullname(dtype_a):
            return _ndarray_type(ctx, dims, dtype_a)
        if name == _dtype_fullname(dtype_b):
            return _ndarray_type(ctx, dims, dtype_b)
        dtype = _construct_dtype(ctx, name)
        if dtype is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, dtype)
    sel, other = (sa, right) if sa is not None else (sb, left)
    assert sel is not None
    (prefix, unpack, suffix), dtype_u = sel
    if _scalar_key(other) is not None:
        return _solve_broadcast_scalar(ctx, prefix + [unpack] + suffix, dtype_u, other)
    ob = _ndarray_parts(other)
    if ob is not None and ob[0] == []:
        name = _promote_dtypes(dtype_u, ob[1])
        if name is None:
            return ctx.default_return_type
        if name == _dtype_fullname(dtype_u):
            return _ndarray_type(ctx, prefix + [unpack] + suffix, dtype_u)
        if name == _dtype_fullname(ob[1]):
            return _ndarray_type(ctx, prefix + [unpack] + suffix, ob[1])
        dtype = _construct_dtype(ctx, name)
        if dtype is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, prefix + [unpack] + suffix, dtype)
    return ctx.default_return_type


def _solve_broadcast(ctx: FunctionContext | MethodContext, left: Type, right: Type) -> Type:
    dist = _distribute_solve([left, right], lambda l, r: _broadcast_types(ctx, l, r))
    if dist is not None:
        return dist
    return _broadcast_types(ctx, left, right)


def _broadcast_types(ctx: FunctionContext | MethodContext, left: Type, right: Type) -> Type:
    a = _ndarray_parts(left)
    b = _ndarray_parts(right)
    if a is None or b is None:
        return _solve_broadcast_unpack(ctx, left, right, a, b)
    dims_a, dtype_a = a
    dims_b, dtype_b = b
    dims_b, dtype_b = b
    name = _promote_dtypes(dtype_a, dtype_b)
    if name is None:
        return ctx.default_return_type
    dims, message = _broadcast_result(dims_a, dims_b)
    if message is not None:
        ctx.api.fail(message, ctx.context)
        return ctx.default_return_type
    if dims is None:
        return ctx.default_return_type
    if name == _dtype_fullname(dtype_a):
        return _ndarray_type(ctx, dims, dtype_a)
    if name == _dtype_fullname(dtype_b):
        return _ndarray_type(ctx, dims, dtype_b)
    dtype = _construct_dtype(ctx, name)
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def broadcast_method_hook(ctx: MethodContext) -> Type:
    args = _positional_actuals(ctx, 1)
    if args is None:
        return ctx.default_return_type
    return _solve_broadcast(ctx, ctx.type, args[0])


def broadcast_function_hook(ctx: FunctionContext) -> Type:
    args = _positional_actuals(ctx, 2)
    if args is None:
        return ctx.default_return_type
    return _solve_broadcast(ctx, args[0], args[1])


STR_FULLNAME = "builtins.str"


def _reshape_new_dims(
    pairs: list[tuple[Type, Expression | None]],
) -> list[int | TypeVarType] | None:
    """Target dims from leading positional reshape args, else None to bail.

    Accepts one tuple (reshape((2, 4))), one int (reshape(6)), or several
    ints (reshape(2, 4)); symbolic TypeVars pass through. Expressions fold
    first (Final names, arithmetic), types are the fallback. A positional
    order= string (np.reshape(a, shape, "F")) ends the scan; anything else
    unknown bails out.
    """
    if not pairs:
        return None
    first_tp, first_expr = pairs[0]
    if isinstance(first_expr, (TupleExpr, ListExpr)) or (
        first_expr is None and isinstance(get_proper_type(first_tp), TupleType)
    ):
        got = _shape_from_arg(first_tp, first_expr)
        if got is None or any(
            not isinstance(d, int) and not isinstance(d, TypeVarType) for d in got
        ):
            return None
        return [d for d in got if isinstance(d, int) or isinstance(d, TypeVarType)]
    dims: list[int | TypeVarType] = []
    for tp, expr in pairs:
        proper = get_proper_type(tp)
        if isinstance(proper, Instance) and proper.type.fullname == STR_FULLNAME:
            break
        value = _fold_int(expr) if expr is not None else None
        if value is None:
            dim = _dim_value(tp)
            value = dim if isinstance(dim, (int, TypeVarType)) else None
        if value is None or isinstance(value, bool):
            return None
        dims.append(value)
    return dims or None


def _regroup_run(
    run_old: list[int | Type], run_new: list[int | TypeVarType]
) -> list[int | TypeVarType] | None:
    """Regroup one known run of dims, else None to bail.

    Mirrors the fixed-rank conservation rule on a slice of the shape:
    int products must match and the TypeVar multisets must agree. The
    caller keeps -1 out: an inferred dim would absorb the unknown *Ts
    product, which no annotation can spell.
    """
    old_ints = [d for d in run_old if isinstance(d, int)]
    old_vars = sorted(d.fullname for d in run_old if isinstance(d, TypeVarType))
    if len(old_ints) + len(old_vars) != len(run_old):
        return None
    new_vars = sorted(d.fullname for d in run_new if isinstance(d, TypeVarType))
    size_old = 1
    for d in old_ints:
        size_old *= d
    size_new = 1
    for d in run_new:
        if isinstance(d, int):
            size_new *= d
    if -1 in run_new or size_new != size_old or old_vars != new_vars:
        return None
    return list(run_new)


def _solve_reshape_unpack(
    ctx: FunctionContext | MethodContext,
    prefix: list[int | Type],
    unpack: UnpackType,
    suffix: list[int | Type],
    dtype: Type,
    raw: list[int | TypeVarType],
) -> Type:
    """Reshape a (*prefix, *Ts, *suffix) shape, preserving the middle.

    Only block-preserving regroups are expressible: the target keeps
    the fixed prefix (suffix regroups) or the fixed suffix (prefix
    regroups), with the unknown block untouched in place. Anything else
    is unprovable through *Ts and bails — except a total-size clash on
    purely int dims, which no *Ts can absorb and which fails loudly.
    """
    old = prefix + [unpack] + suffix
    suffix_candidate = len(raw) >= len(prefix) and all(
        _dim_eq(raw[i], prefix[i]) for i in range(len(prefix))
    )
    prefix_candidate = len(raw) >= len(suffix) and (
        not suffix
        or all(
            _dim_eq(raw[len(raw) - len(suffix) + i], suffix[i])
            for i in range(len(suffix))
        )
    )
    if suffix_candidate:
        regrouped = _regroup_run(suffix, raw[len(prefix):])
        if regrouped is not None:
            return _ndarray_type(ctx, [*prefix, unpack, *regrouped], dtype)
    if prefix_candidate and not suffix_candidate:
        regrouped = _regroup_run(prefix, raw[: len(raw) - len(suffix)] if suffix else list(raw))
        if regrouped is not None:
            return _ndarray_type(ctx, [*regrouped, unpack, *suffix], dtype)
    if any(not isinstance(d, int) for d in prefix + suffix):
        return ctx.default_return_type
    if any(not isinstance(d, int) for d in raw):
        return ctx.default_return_type
    if -1 in raw:
        return ctx.default_return_type
    known = 1
    for d in prefix + suffix:
        assert isinstance(d, int)
        known *= d
    total = 1
    for d in raw:
        assert isinstance(d, int)
        total *= d
    if known != 0 and total != 0 and total % known != 0:
        ctx.api.fail(
            f"reshape changes total size: {total} is not a multiple of "
            f"known size {known} (shapes {_dims_str(old)} vs {tuple(raw)})",
            ctx.context,
        )
        return ctx.default_return_type
    if known == 0 and total != 0:
        ctx.api.fail(
            f"reshape changes total size: 0 -> {total} "
            f"(shapes {_dims_str(old)} vs {tuple(raw)})",
            ctx.context,
        )
        return ctx.default_return_type
    return ctx.default_return_type


def _solve_reshape(
    ctx: FunctionContext | MethodContext, old: Type, pairs: list[tuple[Type, Expression | None]]
) -> Type:
    parts = _ndarray_parts(old)
    if parts is None:
        split = _unpack_shape(old)
        if split is None:
            return ctx.default_return_type
        raw = _reshape_new_dims(pairs)
        if raw is None:
            return ctx.default_return_type
        raw_ints = [d for d in raw if isinstance(d, int)]
        if any(d < -1 for d in raw_ints):
            ctx.api.fail(
                f"reshape has invalid dimension {min(raw_ints)}; only -1 may be negative "
                f"(shape {_dims_str(raw)})",
                ctx.context,
            )
            return ctx.default_return_type
        if raw_ints.count(-1) > 1:
            ctx.api.fail(
                f"reshape accepts only one -1 dimension (shape {_dims_str(raw)})",
                ctx.context,
            )
            return ctx.default_return_type
        (prefix, unpack, suffix), dtype = split
        return _solve_reshape_unpack(ctx, prefix, unpack, suffix, dtype, raw)
    dims_old, dtype = parts
    raw = _reshape_new_dims(pairs)
    if raw is None:
        return ctx.default_return_type
    raw_ints = [d for d in raw if isinstance(d, int)]
    if any(d < -1 for d in raw_ints):
        ctx.api.fail(
            f"reshape has invalid dimension {min(raw_ints)}; only -1 may be negative "
            f"(shape {_dims_str(raw)})",
            ctx.context,
        )
        return ctx.default_return_type
    if raw_ints.count(-1) > 1:
        ctx.api.fail(
            f"reshape accepts only one -1 dimension (shape {_dims_str(raw)})",
            ctx.context,
        )
        return ctx.default_return_type
    # Conservation with symbolic sizes: int products must match and the
    # TypeVar multisets must agree; anything else bails out.
    old_ints = [d for d in dims_old if isinstance(d, int)]
    old_vars = sorted(d.fullname for d in dims_old if isinstance(d, TypeVarType))
    if len(old_ints) + len(old_vars) != len(dims_old):
        return ctx.default_return_type
    new_vars = sorted(d.fullname for d in raw if isinstance(d, TypeVarType))
    size_old = 1
    for d in old_ints:
        size_old *= d
    dims_new: list[int | TypeVarType] | None
    raw_str = _dims_str(raw) if new_vars else f"{tuple(raw)}"
    if -1 in raw_ints:
        known = 1
        for d in raw_ints:
            if d != -1:
                known *= d
        if old_vars != new_vars:
            # The inferred dim would absorb a symbolic size (N * 50 is not
            # expressible), so this stays out of scope instead of erroring.
            return ctx.default_return_type
        if known == 0 or size_old % known != 0:
            detail = (
                f"(shape {_dims_str(dims_old)} vs {raw_str})"
                if old_vars
                else f"(shape {raw_str})"
            )
            ctx.api.fail(
                f"reshape cannot infer -1: total size {size_old} is not divisible "
                f"by {known} {detail}",
                ctx.context,
            )
            return ctx.default_return_type
        inferred = size_old // known
        dims_new = [inferred if d == -1 else d for d in raw]
    else:
        size_new = 1
        for d in raw_ints:
            size_new *= d
        if size_new != size_old:
            ctx.api.fail(
                f"reshape changes total size: {size_old} -> {size_new} "
                f"(shapes {_dims_str(dims_old)} vs {raw_str})",
                ctx.context,
            )
            return ctx.default_return_type
        if old_vars != new_vars:
            # Same int total but different symbolic factors (N vs M): the
            # equality is unprovable, so bail out instead of erroring.
            return ctx.default_return_type
        dims_new = raw
    return _ndarray_type(ctx, dims_new, dtype)


def _all_positionals(ctx: FunctionContext | MethodContext) -> list[tuple[Type, Expression | None]]:
    """Every positional actual in call order, each with its expression."""
    out: list[tuple[Type, Expression | None]] = []
    for kinds, types, exprs in zip(ctx.arg_kinds, ctx.arg_types, ctx.args):
        for kind, tp, expr in zip(kinds, types, exprs):
            if kind == ARG_POS:
                out.append((tp, expr))
    return out


def _formal_pair(
    ctx: FunctionContext | MethodContext, name: str
) -> tuple[Type, Expression | None] | None:
    """The actual (type, expression) bound to formal `name`, else None."""
    try:
        index = ctx.callee_arg_names.index(name)
    except ValueError:
        return None
    if index >= len(ctx.arg_types) or not ctx.arg_types[index]:
        return None
    exprs = ctx.args[index] if index < len(ctx.args) else []
    return ctx.arg_types[index][0], exprs[0] if exprs else None


def _fold_bool(expr: Expression) -> bool | None:
    if isinstance(expr, NameExpr) and expr.name in ("True", "False"):
        return expr.name == "True"
    return None


def _bool_value(tp: Type) -> bool | None:
    proper = get_proper_type(tp)
    if isinstance(proper, LiteralType) and isinstance(proper.value, bool):
        return proper.value
    if isinstance(proper, Instance) and proper.last_known_value is not None:
        return _bool_value(proper.last_known_value)
    return None


def _fold_flag(pair: tuple[Type, Expression | None] | None) -> bool | None:
    """Bool value of an optional flag argument, else None (absent/unknown)."""
    if pair is None:
        return None
    tp, expr = pair
    if expr is not None:
        value = _fold_bool(expr)
        if value is not None:
            return value
    return _bool_value(tp)


def _is_none_type(tp: Type) -> bool:
    return isinstance(get_proper_type(tp), NoneType)


def _parse_axis_arg(pair: tuple[Type, Expression | None] | None) -> list[int] | None | str:
    """'absent' when missing, 'none' for axis=None, dims on success, None to bail."""
    if pair is None:
        return "absent"
    tp, expr = pair
    if _is_none_type(tp):
        return "none"
    return _shape_from_arg(tp, expr)


def _normalize_axis(ax: int, ndim: int) -> int | None:
    norm = ax + ndim if ax < 0 else ax
    return norm if 0 <= norm < ndim else None


def _nbit_bytes(tp: Type) -> int | None:
    """Byte size of a bit-width tag (_32Bit class, literal, ...)."""
    proper = get_proper_type(tp)
    if isinstance(proper, LiteralType) and isinstance(proper.value, int):
        bits = proper.value
    elif isinstance(proper, Instance):
        tail = proper.type.fullname.rsplit(".", 1)[-1]
        tagged = {
            "_8Bit": 8,
            "_16Bit": 16,
            "_32Bit": 32,
            "_64Bit": 64,
            "_128Bit": 128,
        }.get(tail)
        if tagged is None:
            return None
        bits = tagged
    else:
        return None
    return bits // 8 if bits in (8, 16, 32, 64, 128) else None


def _dtype_kind_size(dtype: Type) -> tuple[str, int] | None:
    """(kind, bytes) for a concrete dtype, else None to bail.

    Bare scalars normalize (np.float32 -> floating[_32Bit]), so resolve
    through the bit-width tag as well as the direct fullnames.
    """
    proper = get_proper_type(dtype)
    if not isinstance(proper, Instance) or not proper.args:
        return None
    element = get_proper_type(proper.args[0])
    if not isinstance(element, Instance):
        return None
    direct = _DTYPE_INFO.get(element.type.fullname)
    if direct is not None:
        return direct
    base = {
        "numpy.floating": "f",
        "numpy.complexfloating": "c",
        "numpy.signedinteger": "i",
        "numpy.unsignedinteger": "u",
    }.get(element.type.fullname)
    if base is None or len(element.args) != 1:
        return None
    size = _nbit_bytes(element.args[0])
    if size is None:
        return None
    if (base == "f" and size not in (2, 4, 8)) or (base == "c" and size not in (8, 16)):
        return None
    if base in ("i", "u") and size not in (1, 2, 4, 8):
        return None
    return base, size


def _dtype_fullname(dtype: Type) -> str | None:
    """Canonical numpy dtype name (float32, int64, ...), else None."""
    ks = _dtype_kind_size(dtype)
    if ks is None:
        return None
    kind, size = ks
    if kind == "b":
        return "numpy.bool"
    return _NAME_OF_KIND_SIZE.get((kind, size))


REDUCTION_DTYPES = frozenset(
    {
        "numpy.float16",
        "numpy.float32",
        "numpy.float64",
        "numpy.longdouble",
        "numpy.complex64",
        "numpy.complex128",
        "numpy.clongdouble",
    }
)


def _solve_transpose(
    ctx: FunctionContext | MethodContext, dims: list[int | Type], dtype: Type, rest: list[tuple[Type, Expression | None]]
) -> Type:
    ndim = len(dims)
    axes: list[int] | None = None
    if not rest:
        axes = list(reversed(range(ndim)))
    elif len(rest) == 1:
        single = _shape_from_arg(rest[0][0], rest[0][1])
        if single is None:
            first_tp = rest[0][0]
            if _is_none_type(first_tp):
                axes = list(reversed(range(ndim)))
            else:
                return ctx.default_return_type
        else:
            axes = single
    else:
        folded: list[int] = []
        for tp, expr in rest:
            value = _fold_int(expr) if expr is not None else None
            if value is None:
                value = _int_value(tp)
            if value is None:
                return ctx.default_return_type
            folded.append(value)
        axes = folded
    if axes is None:
        return ctx.default_return_type
    return _apply_permutation(ctx, dims, dtype, axes)


def _apply_permutation(
    ctx: FunctionContext | MethodContext, dims: list[int | Type], dtype: Type, axes: list[int]
) -> Type:
    ndim = len(dims)
    normalized: list[int] = []
    for ax in axes:
        norm = _normalize_axis(ax, ndim)
        if norm is None:
            ctx.api.fail(
                f"transpose axis {ax} is out of range for {ndim} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
            )
            return ctx.default_return_type
        normalized.append(norm)
    if sorted(normalized) != list(range(ndim)):
        ctx.api.fail(
            f"transpose axes {tuple(axes)} are not a permutation of {ndim} "
            f"dimensions (shape {_dims_str(dims)})",
            ctx.context,
        )
        return ctx.default_return_type
    return _ndarray_type(ctx, [dims[ax] for ax in normalized], dtype)


def _reverse_unpack_dims(
    prefix: list[int | Type], unpack: UnpackType, suffix: list[int | Type]
) -> list[int | Type]:
    """Full reversal with the unknown block kept in place, in order.

    A true reversal would also reverse *Ts itself, which no annotation
    can spell, so the block stays ordered: (*Ts, 2) reverses to
    (2, *Ts). Rank and size multiset stay exact; the order inside *Ts
    is not tracked (see README limitations).
    """
    return [*reversed(suffix), unpack, *reversed(prefix)]


def _solve_transpose_unpack(
    ctx: FunctionContext | MethodContext,
    prefix: list[int | Type],
    unpack: UnpackType,
    suffix: list[int | Type],
    dtype: Type,
    rest: list[tuple[Type, Expression | None]],
) -> Type:
    """Transpose a (*prefix, *Ts, *suffix) shape.

    Only the default full reversal is expressible (the unknown block
    stays in place per _reverse_unpack_dims). Explicit axes would pin
    the rank through the axis count but cannot order the unknown block,
    so they bail out instead of guessing.
    """
    if not rest:
        return _ndarray_type(ctx, _reverse_unpack_dims(prefix, unpack, suffix), dtype)
    if len(rest) == 1:
        single = _shape_from_arg(rest[0][0], rest[0][1])
        if single is None and _is_none_type(rest[0][0]):
            return _ndarray_type(ctx, _reverse_unpack_dims(prefix, unpack, suffix), dtype)
    return ctx.default_return_type


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


def _weak_name(tp: Type) -> str | None:
    """builtins scalar name for a Python scalar type, else None."""
    proper = get_proper_type(tp)
    if isinstance(proper, LiteralType):
        if isinstance(proper.value, bool):
            return "builtins.bool"
        if isinstance(proper.value, int):
            return "builtins.int"
        if isinstance(proper.value, float):
            return "builtins.float"
        if isinstance(proper.value, complex):
            return "builtins.complex"
        return None
    if isinstance(proper, Instance) and proper.type.fullname in _WEAK_SCALARS:
        return proper.type.fullname
    return None


def _scalar_key(tp: Type) -> tuple[str, str] | None:
    """("weak", name) for Python scalars, ("strong", name) for numpy scalars."""
    weak = _weak_name(tp)
    if weak is not None:
        return ("weak", weak)
    proper = get_proper_type(tp)
    if isinstance(proper, Instance) and proper.type.fullname in _DTYPE_INFO:
        return ("strong", proper.type.fullname)
    return None


def _promote_dtypes(dta: Type, dtb: Type) -> str | None:
    """Promoted numpy dtype fullname for two dtype[...]s, else None."""
    na = _dtype_fullname(dta)
    nb = _dtype_fullname(dtb)
    if na is None or nb is None:
        return None
    if na == nb:
        return na
    return _promote_strong(na, nb)


def _promote_all(ctx: FunctionContext | MethodContext, dtypes: list[Type]) -> Type | None:
    """Pairwise promotion across many dtypes, else None to bail."""
    names: list[str] = []
    for dtype in dtypes:
        name = _dtype_fullname(dtype)
        if name is None:
            return None
        names.append(name)
    result = names[0]
    for name in names[1:]:
        if name == result:
            continue
        promoted = _promote_strong(result, name)
        if promoted is None:
            return None
        result = promoted
    for dtype in dtypes:
        if _dtype_fullname(dtype) == result:
            return dtype
    return _construct_dtype(ctx, result)


def _solve_squeeze(
    ctx: FunctionContext | MethodContext,
    dims: list[int | Type],
    dtype: Type,
    axis: list[int] | None | str,
) -> Type:
    ndim = len(dims)
    if not isinstance(axis, list):
        if axis in ("absent", "none"):
            if any(not isinstance(d, int) for d in dims):
                # A symbolic dim could be 1 at runtime: bail out.
                return ctx.default_return_type
            return _ndarray_type(ctx, [d for d in dims if d != 1], dtype)
        return ctx.default_return_type
    chosen: list[int] = []
    for ax in axis:
        norm = _normalize_axis(ax, ndim)
        if norm is None:
            ctx.api.fail(
                f"squeeze axis {ax} is out of range for {ndim} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
            )
            return ctx.default_return_type
        if not isinstance(dims[norm], int):
            return ctx.default_return_type
        if dims[norm] != 1:
            ctx.api.fail(
                f"squeeze axis {ax} has size {dims[norm]}, not 1 "
                f"(shape {_dims_str(dims)})",
                ctx.context,
            )
            return ctx.default_return_type
        chosen.append(norm)
    return _ndarray_type(ctx, [d for i, d in enumerate(dims) if i not in chosen], dtype)


def _solve_expand_dims(
    ctx: FunctionContext | MethodContext, dims: list[int | Type], dtype: Type, axis: list[int] | None | str
) -> Type:
    if not isinstance(axis, list):
        return ctx.default_return_type
    ndim = len(dims)
    total = ndim + len(axis)
    normalized: list[int] = []
    for ax in axis:
        norm = ax + total if ax < 0 else ax
        if not 0 <= norm < total:
            ctx.api.fail(
                f"expand_dims axis {ax} is out of range for {ndim} dimensions "
                f"with {len(axis)} new axes",
                ctx.context,
            )
            return ctx.default_return_type
        if norm in normalized:
            ctx.api.fail(f"expand_dims has duplicate axis {ax}", ctx.context)
            return ctx.default_return_type
        normalized.append(norm)
    out = list(dims)
    for ax in sorted(normalized):
        out.insert(ax, 1)
    return _ndarray_type(ctx, out, dtype)


def _solve_reduction(
    ctx: FunctionContext | MethodContext,
    dims: list[int | Type],
    dtype: Type,
    axis: list[int] | None | str,
    keepdims: bool | None,
    has_dtype: bool,
    force_dtype: Type | None = None,
) -> Type:
    if not isinstance(axis, list) or keepdims is None or has_dtype:
        return ctx.default_return_type
    if force_dtype is None and _dtype_fullname(dtype) not in REDUCTION_DTYPES:
        return ctx.default_return_type
    out_dtype = force_dtype if force_dtype is not None else dtype
    ndim = len(dims)
    normalized: list[int] = []
    for ax in axis:
        norm = _normalize_axis(ax, ndim)
        if norm is None:
            ctx.api.fail(
                f"axis {ax} is out of range for {ndim} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
            )
            return ctx.default_return_type
        if norm in normalized:
            ctx.api.fail(f"duplicate axis {ax}", ctx.context)
            return ctx.default_return_type
        normalized.append(norm)
    chosen = set(normalized)
    if keepdims:
        return _ndarray_type(ctx, [1 if i in chosen else d for i, d in enumerate(dims)], out_dtype)
    return _ndarray_type(ctx, [d for i, d in enumerate(dims) if i not in chosen], out_dtype)


def _reduction_inputs_bool(ctx: FunctionContext | MethodContext, array: Type) -> Type:
    """any/all: same axis rules as reductions, but the output is bool."""
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    keep_pair = _formal_pair(ctx, "keepdims")
    keepdims = False if keep_pair is None else _fold_flag(keep_pair)
    dtype_pair = _formal_pair(ctx, "dtype")
    has_dtype = dtype_pair is not None and not _is_none_type(dtype_pair[0])
    dtype = _construct_dtype(ctx, "numpy.bool")
    if dtype is None:
        return ctx.default_return_type
    dist = _distribute_solve(
        [array], lambda t: _reduction_bool_types(ctx, t, axis, keepdims, has_dtype, dtype)
    )
    if dist is not None:
        return dist
    return _reduction_bool_types(ctx, array, axis, keepdims, has_dtype, dtype)


def _reduction_bool_types(
    ctx: FunctionContext | MethodContext,
    array: Type,
    axis: list[int] | None | str,
    keepdims: bool | None,
    has_dtype: bool,
    dtype: Type,
) -> Type:
    parts = _parts_or_unpack(array)
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    if _has_unpack(dims):
        split = _split_flat(dims)
        if split is None:
            return ctx.default_return_type
        return _solve_reduction_unpack(ctx, *split, dtype, axis, keepdims, has_dtype, dtype)
    return _solve_reduction(ctx, dims, dtype, axis, keepdims, has_dtype, dtype)


def allany_method_hook(ctx: MethodContext) -> Type:
    return _reduction_inputs_bool(ctx, ctx.type)


def allany_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _reduction_inputs_bool(ctx, pairs[0][0])


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


def _exact_element(
    ctx: FunctionContext | MethodContext, base: str, size: int
) -> Type | None:
    """base[_NBit] for a known byte size, else None when the tag won't load.

    Complex takes the component tag twice (complex64 is
    complexfloating[_32Bit, _32Bit]).
    """
    twice = base == "numpy.complexfloating"
    tag = _NBIT_TAG_BY_BYTES.get(size // 2 if twice else size)
    if tag is None:
        return None
    modules = getattr(ctx.api, "modules", None) or {}
    for module in _NBIT_MODULES:
        try:
            mod = modules.get(module)
            node = mod.names.get(tag) if mod is not None else None
            info = node.node if node is not None else None
            if info is None or not hasattr(info, "fullname"):
                continue
            bit = Instance(info, [])
            return ctx.api.named_generic_type(base, [bit, bit] if twice else [bit])
        except Exception:
            continue
    return None


def _construct_dtype(ctx: FunctionContext | MethodContext, name: str) -> Type | None:
    """A dtype[...] Instance for a promotion result, else None to bail.

    Most numpy scalar names are type aliases, which named_generic_type
    rejects (it asserts on TypeAlias). Reusable inputs keep their exact
    dtype; anything else builds the base with the exact bit-width tag
    (falling back to Any bits when the stub-only tag class won't load).
    Never crashes: failure means bail.
    """
    try:
        if name in ("numpy.float64", "numpy.complex128"):
            element: Type = ctx.api.named_generic_type(name, [])
        elif name == "numpy.bool":
            element = ctx.api.named_generic_type(
                name, [ctx.api.named_generic_type("builtins.bool", [])]
            )
        else:
            base = _DTYPE_BASE.get(name)
            if base is None:
                return None
            element = _exact_element(ctx, base, _DTYPE_INFO[name][1])
            if element is None:
                any_arg = AnyType(TypeOfAny.explicit)
                try:
                    element = ctx.api.named_generic_type(base, [any_arg])
                except AssertionError:
                    element = ctx.api.named_generic_type(base, [any_arg, any_arg])
        return ctx.api.named_generic_type(DTYPE_FULLNAME, [element])
    except AssertionError:
        return None


def _default_dtype(ctx: FunctionContext | MethodContext) -> Type:
    proper = get_proper_type(ctx.default_return_type)
    if isinstance(proper, Instance) and len(proper.args) == 2:
        return proper.args[1]
    return AnyType(TypeOfAny.explicit)


def _fill_dtype_name(tp: Type) -> str | None:
    proper = get_proper_type(tp)
    if isinstance(proper, Instance):
        if proper.type.fullname == "builtins.float":
            return "numpy.float64"
        if proper.type.fullname == "builtins.complex":
            return "numpy.complex128"
    return None


def _fold_single(pair: tuple[Type, Expression | None]) -> int | None:
    tp, expr = pair
    value = _fold_int(expr) if expr is not None else None
    return value if value is not None else _int_value(tp)


def _explicit_dtype(ctx: FunctionContext | MethodContext) -> bool:
    pair = _formal_pair(ctx, "dtype")
    return pair is not None and not _is_none_type(pair[0])


_BUILTIN_DTYPE = {
    "builtins.bool": "numpy.bool",
    "builtins.int": "numpy.int64",
    "builtins.float": "numpy.float64",
    "builtins.complex": "numpy.complex128",
}


def _instance_fullname(tp: Type) -> str | None:
    proper = get_proper_type(tp)
    if isinstance(proper, Instance):
        return proper.type.fullname
    return None


_DTYPE_STRINGS = {
    "bool": "numpy.bool",
    "int8": "numpy.int8",
    "int16": "numpy.int16",
    "int32": "numpy.int32",
    "int64": "numpy.int64",
    "uint8": "numpy.uint8",
    "uint16": "numpy.uint16",
    "uint32": "numpy.uint32",
    "uint64": "numpy.uint64",
    "float16": "numpy.float16",
    "float32": "numpy.float32",
    "float64": "numpy.float64",
    "complex64": "numpy.complex64",
    "complex128": "numpy.complex128",
}


def _scalar_ret_name(ret: Type) -> str | None:
    """Concrete dtype name for a scalar-constructor return type (np.float32
    is a stub function returning floating[_32Bit]), else None."""
    proper = get_proper_type(ret)
    if not isinstance(proper, Instance):
        return None
    base = proper.type.fullname
    if base == "numpy.bool":
        return "numpy.bool"
    tables = {
        "numpy.floating": {16: "numpy.float16", 32: "numpy.float32", 64: "numpy.float64"},
        "numpy.complexfloating": {64: "numpy.complex64", 128: "numpy.complex128"},
        "numpy.signedinteger": {
            8: "numpy.int8",
            16: "numpy.int16",
            32: "numpy.int32",
            64: "numpy.int64",
        },
        "numpy.unsignedinteger": {
            8: "numpy.uint8",
            16: "numpy.uint16",
            32: "numpy.uint32",
            64: "numpy.uint64",
        },
    }
    table = tables.get(base)
    if table is None or len(proper.args) != 1:
        return None
    bit = get_proper_type(proper.args[0])
    if not isinstance(bit, Instance):
        return None
    try:
        bits = int(bit.type.fullname.rsplit("_", 1)[-1].removesuffix("Bit"))
    except ValueError:
        return None
    return table.get(bits)


def _named_dtype(ctx: FunctionContext | MethodContext) -> Type | None:
    """dtype[...] for an explicit, resolvable dtype= argument.

    Uses the dtype the caller named (type objects, scalar constructors
    like np.float32, np.dtype(...) results, canonical strings); anything
    else bails out instead of guessing.
    """
    pair = _formal_pair(ctx, "dtype")
    if pair is None or _is_none_type(pair[0]):
        return None
    tp, expr = pair
    proper = get_proper_type(tp)
    if isinstance(proper, LiteralType) and isinstance(proper.value, str):
        name = _DTYPE_STRINGS.get(proper.value)
        return _construct_dtype(ctx, name) if name is not None else None
    if (
        isinstance(proper, Instance)
        and proper.type.fullname == "builtins.type"
        and len(proper.args) == 1
    ):
        element = get_proper_type(proper.args[0])
        if isinstance(element, Instance):
            fullname = element.type.fullname
            if fullname == "numpy.bool_":
                fullname = "numpy.bool"
            fullname = _BUILTIN_DTYPE.get(fullname, fullname)
            if fullname in _DTYPE_INFO:
                return _construct_dtype(ctx, fullname)
    if isinstance(proper, Instance) and proper.type.fullname == DTYPE_FULLNAME:
        if len(proper.args) == 1:
            element = get_proper_type(proper.args[0])
            if isinstance(element, Instance) and element.type.fullname in _DTYPE_INFO:
                return _construct_dtype(ctx, element.type.fullname)
        return None
    if isinstance(proper, CallableType):
        name = _scalar_ret_name(proper.ret_type)
        if name is None:
            name = _BUILTIN_DTYPE.get(_instance_fullname(proper.ret_type))
        return _construct_dtype(ctx, name) if name is not None else None
    if isinstance(proper, Overloaded):
        # Bare builtins (int, float, ...) arrive as constructor
        # overloads; use the dtype only when every item agrees.
        names = set()
        for item in proper.items:
            name = _scalar_ret_name(item.ret_type)
            if name is None:
                name = _BUILTIN_DTYPE.get(_instance_fullname(item.ret_type))
            if name is None:
                return None
            names.add(name)
        if len(names) != 1:
            return None
        return _construct_dtype(ctx, names.pop())
    if isinstance(expr, StrExpr):
        name = _DTYPE_STRINGS.get(expr.value)
        return _construct_dtype(ctx, name) if name is not None else None
    return None


def like_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    dist = _distribute_solve([pairs[0][0]], lambda t: _like_types(ctx, t))
    if dist is not None:
        return dist
    return _like_types(ctx, pairs[0][0])


def _like_types(ctx: FunctionContext, tp: Type) -> Type:
    parts = _parts_or_unpack(tp)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    shape_pair = _formal_pair(ctx, "shape")
    if shape_pair is not None and not _is_none_type(shape_pair[0]):
        override = _shape_from_arg(shape_pair[0], shape_pair[1])
        if override is None:
            return ctx.default_return_type
        dims = override
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
        if named is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, named)
    return _ndarray_type(ctx, dims, dtype)


def full_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    shape_pair = _formal_pair(ctx, "shape") or (pairs[0] if pairs else None)
    fill_pair = _formal_pair(ctx, "fill_value")
    if shape_pair is None or fill_pair is None:
        return ctx.default_return_type
    dims = _shape_from_arg(shape_pair[0], shape_pair[1])
    if dims is None:
        return ctx.default_return_type
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
        if named is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, named)
    name = _fill_dtype_name(fill_pair[0])
    if name is not None:
        dtype = _construct_dtype(ctx, name)
        if dtype is None:
            dtype = _default_dtype(ctx)
    else:
        dtype = _default_dtype(ctx)
    return _ndarray_type(ctx, dims, dtype)


def eye_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    n = _fold_single(pairs[0])
    if n is None:
        return ctx.default_return_type
    m_pair = _formal_pair(ctx, "M")
    if m_pair is None or _is_none_type(m_pair[0]):
        m = n
    else:
        folded_m = _fold_single(m_pair)
        if folded_m is None:
            return ctx.default_return_type
        m = folded_m
    if n < 0 or m < 0:
        ctx.api.fail(
            f"eye dimensions must be non-negative, got ({n}, {m})",
            ctx.context,
        )
        return ctx.default_return_type
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
        if named is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, [n, m], named)
    dtype = _construct_dtype(ctx, "numpy.float64")
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, [n, m], dtype)


def _arange_symbolic(
    pairs: list[tuple[Type, Expression | None]],
) -> list[int | TypeVarType] | None:
    """arange(N) / arange(0, N) / arange(0, N, 1): N elements, else None."""
    if len(pairs) == 1:
        start: int | None = 0
        stop_tp = pairs[0][0]
        step: int | None = 1
    elif len(pairs) in (2, 3):
        start = _fold_single(pairs[0])
        stop_tp = pairs[1][0]
        step = _fold_single(pairs[2]) if len(pairs) == 3 else 1
    else:
        return None
    if start != 0 or step != 1:
        return None
    stop = _dim_value(stop_tp)
    if not isinstance(stop, TypeVarType):
        return None
    return [stop]


def arange_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) not in (1, 2, 3):
        return ctx.default_return_type
    values: list[int] = []
    for pair in pairs:
        value = _fold_single(pair)
        if value is None:
            break
        values.append(value)
    dims: list[int | TypeVarType] | None
    if len(values) == len(pairs):
        if len(values) == 1:
            start, stop, step = 0, values[0], 1
        else:
            start, stop = values[0], values[1]
            step = values[2] if len(values) == 3 else 1
        try:
            dims = [len(range(start, stop, step))]
        except ValueError:
            ctx.api.fail("arange step is zero", ctx.context)
            return ctx.default_return_type
    else:
        dims = _arange_symbolic(pairs)
        if dims is None:
            return ctx.default_return_type
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
        if named is None:
            return ctx.default_return_type
        explicit_dtype: Type | None = named
    else:
        explicit_dtype = None
    if explicit_dtype is not None:
        return _ndarray_type(ctx, dims, explicit_dtype)
    return _ndarray_type(ctx, dims, _default_dtype(ctx))


def _comprehension_parts(
    ctx: FunctionContext | MethodContext, expr: Expression
) -> list[tuple[list[int | Type], Type]] | None:
    """[f(c) for c in EnumClass]: item type from the element expression,
    count from the enum members. Anything else bails out."""
    if not isinstance(expr, ListComprehension):
        return None
    gen = expr.generator
    if len(gen.indices) != 1 or len(gen.sequences) != 1 or any(gen.condlists):
        return None
    seq = gen.sequences[0]
    if not isinstance(seq, NameExpr):
        return None
    node = seq.node
    if not isinstance(node, TypeInfo) or not node.is_enum:
        return None
    count = _enum_member_count(node)
    if count is None:
        return None
    try:
        item_tp = ctx.api.get_expression_type(gen.left_expr)
    except Exception:
        return None
    if item_tp is None:
        return None
    parts = _ndarray_parts(item_tp)
    if parts is None:
        return None
    dims, dtype = parts
    return [(list(dims), dtype) for _ in range(count)] or None


def _sequence_parts(
    ctx: FunctionContext | MethodContext, tp: Type, expr: Expression | None
) -> list[tuple[list[int | Type], Type]] | None:
    """Per-array (dims, dtype) of a tuple display, else None to bail.

    List displays arrive already joined against the ArrayLike formal, so
    each item's type is re-read from its own expression instead.
    """
    comp = _comprehension_parts(ctx, expr) if expr is not None else None
    if comp is not None:
        return comp
    proper = get_proper_type(tp)
    if isinstance(proper, TupleType):
        items: list[tuple[list[int | Type], Type]] = []
        for item in proper.items:
            parts = _ndarray_parts(item)
            if parts is None:
                return None
            dims, dtype = parts
            items.append((list(dims), dtype))
        return items or None
    if isinstance(expr, ListExpr):
        items = []
        for item_expr in expr.items:
            item_tp = ctx.api.get_expression_type(item_expr)
            if item_tp is None:
                return None
            parts = _ndarray_parts(item_tp)
            if parts is None:
                return None
            dims, dtype = parts
            items.append((list(dims), dtype))
        return items or None
    return None


def _concat_axis(ctx: FunctionContext, pairs: list[tuple[Type, Expression | None]]) -> int | None | str:
    pair = _formal_pair(ctx, "axis")
    if pair is None:
        if len(pairs) > 1:
            vals = _shape_from_arg(pairs[1][0], pairs[1][1])
            if vals is None or len(vals) != 1:
                return None
            return vals[0]
        return 0
    return _single_axis_value(pair)


def _single_axis_value(pair: tuple[Type, Expression | None]) -> int | None | str:
    tp, expr = pair
    if _is_none_type(tp):
        return "none"
    vals = _shape_from_arg(tp, expr)
    if vals is None or len(vals) != 1:
        return None
    return vals[0]


def concatenate_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    items = _sequence_parts(ctx, pairs[0][0], pairs[0][1])
    if items is None:
        return ctx.default_return_type
    if _explicit_dtype(ctx):
        dtype = _named_dtype(ctx)
        if dtype is None:
            return ctx.default_return_type
    else:
        dtype = _promote_all(ctx, [dtype for _, dtype in items])
        if dtype is None:
            return ctx.default_return_type
    axis = _concat_axis(ctx, pairs)
    if axis is None:
        return ctx.default_return_type
    if axis == "none":
        if any(not isinstance(d, int) for dims, _ in items for d in dims):
            return ctx.default_return_type
        total = 1
        for dims, _ in items:
            for d in dims:
                assert isinstance(d, int)
                total *= d
        return _ndarray_type(ctx, [total], dtype)
    if not isinstance(axis, int):
        return ctx.default_return_type
    ranks = {len(dims) for dims, _ in items}
    if len(ranks) != 1:
        ctx.api.fail(
            "concatenate ranks differ: "
            + ", ".join(_dims_str(dims) for dims, _ in items),
            ctx.context,
        )
        return ctx.default_return_type
    ndim = len(items[0][0])
    norm = _normalize_axis(axis, ndim)
    if norm is None:
        ctx.api.fail(
            f"concatenate axis {axis} is out of range for {ndim} dimensions",
            ctx.context,
        )
        return ctx.default_return_type
    out = list(items[0][0])
    for dims, _ in items[1:]:
        for i, d in enumerate(dims):
            if i != norm and not _dim_eq(out[i], d):
                if isinstance(out[i], int) and isinstance(d, int):
                    ctx.api.fail(
                        f"concatenate dimension {i} differs: {out[i]} vs {d} "
                        f"(shapes {_dims_str(out)} vs {_dims_str(dims)})",
                        ctx.context,
                    )
                    return ctx.default_return_type
                return ctx.default_return_type
        if isinstance(out[norm], int) and isinstance(dims[norm], int):
            out[norm] = out[norm] + dims[norm]
        else:
            # N + N is not representable: bail out.
            return ctx.default_return_type
    return _ndarray_type(ctx, out, dtype)


def stack_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    items = _sequence_parts(ctx, pairs[0][0], pairs[0][1])
    if items is None:
        return ctx.default_return_type
    if _explicit_dtype(ctx):
        dtype = _named_dtype(ctx)
        if dtype is None:
            return ctx.default_return_type
    else:
        dtype = _promote_all(ctx, [dtype for _, dtype in items])
        if dtype is None:
            return ctx.default_return_type
    axis = _concat_axis(ctx, pairs)
    if axis is None or axis == "none":
        return ctx.default_return_type
    if not isinstance(axis, int):
        return ctx.default_return_type
    first = items[0][0]
    for dims, _ in items[1:]:
        if len(dims) != len(first) or not all(_dim_eq(x, y) for x, y in zip(first, dims)):
            if all(isinstance(d, int) for d in dims) and all(isinstance(d, int) for d in first):
                ctx.api.fail(
                    "stack shapes differ: "
                    + ", ".join(_dims_str(dims) for dims, _ in items),
                    ctx.context,
                )
                return ctx.default_return_type
            return ctx.default_return_type
    ndim = len(first)
    norm = axis + ndim + 1 if axis < 0 else axis
    if not 0 <= norm <= ndim:
        ctx.api.fail(
            f"stack axis {axis} is out of range for {ndim} dimensions",
            ctx.context,
        )
        return ctx.default_return_type
    out = list(first)
    out.insert(norm, len(items))
    return _ndarray_type(ctx, out, dtype)


def reshape_method_hook(ctx: MethodContext) -> Type:
    return _solve_reshape(ctx, ctx.type, _all_positionals(ctx))


def reshape_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    return _solve_reshape(ctx, pairs[0][0], pairs[1:])


def transpose_method_hook(ctx: MethodContext) -> Type:
    parts = _ndarray_parts(ctx.type)
    if parts is not None:
        dims, dtype = parts
        return _solve_transpose(ctx, dims, dtype, _all_positionals(ctx))
    split = _unpack_shape(ctx.type)
    if split is None:
        return ctx.default_return_type
    (prefix, unpack, suffix), dtype = split
    return _solve_transpose_unpack(ctx, prefix, unpack, suffix, dtype, _all_positionals(ctx))


def transpose_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        split = _unpack_shape(pairs[0][0])
        if split is None:
            return ctx.default_return_type
        (prefix, unpack, suffix), dtype = split
        return _solve_transpose_unpack(ctx, prefix, unpack, suffix, dtype, pairs[1:])
    dims, dtype = parts
    rest = pairs[1:]
    if not rest:
        return _apply_permutation(ctx, dims, dtype, list(reversed(range(len(dims)))))
    if len(rest) > 1:
        return ctx.default_return_type
    tp, expr = rest[0]
    if _is_none_type(tp):
        return _apply_permutation(ctx, dims, dtype, list(reversed(range(len(dims)))))
    if isinstance(expr, (TupleExpr, ListExpr)) or (
        expr is None and isinstance(get_proper_type(tp), TupleType)
    ):
        axes = _shape_from_arg(tp, expr)
        if axes is None:
            return ctx.default_return_type
        return _apply_permutation(ctx, dims, dtype, axes)
    return ctx.default_return_type


def squeeze_method_hook(ctx: MethodContext) -> Type:
    parts = _ndarray_parts(ctx.type)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    return _solve_squeeze(ctx, dims, dtype, _parse_axis_arg(_formal_pair(ctx, "axis")))


def squeeze_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    return _solve_squeeze(ctx, dims, dtype, _parse_axis_arg(_formal_pair(ctx, "axis")))


def expand_dims_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    return _solve_expand_dims(ctx, dims, dtype, _parse_axis_arg(_formal_pair(ctx, "axis")))


def _solve_reduction_unpack(
    ctx: FunctionContext | MethodContext,
    prefix: list[int | Type],
    unpack: UnpackType,
    suffix: list[int | Type],
    dtype: Type,
    axis: list[int] | None | str,
    keepdims: bool | None,
    has_dtype: bool,
    force_dtype: Type | None = None,
) -> Type:
    """Reduce a (*prefix, *Ts, *suffix) shape along provable axes.

    Only axes that resolve to a known dim reduce: non-negative axes
    into the prefix, negative axes into the suffix. Anything reaching
    into *Ts bails out, as do absent/None axes, exactly like the
    fixed-rank path.
    """
    if not isinstance(axis, list) or keepdims is None or has_dtype:
        return ctx.default_return_type
    if force_dtype is None and _dtype_fullname(dtype) not in REDUCTION_DTYPES:
        return ctx.default_return_type
    out_dtype = force_dtype if force_dtype is not None else dtype
    flat = prefix + [unpack] + suffix
    chosen: set[int] = set()
    for ax in axis:
        if isinstance(ax, bool) or not isinstance(ax, int):
            return ctx.default_return_type
        if ax < 0:
            if -ax > len(suffix):
                return ctx.default_return_type
            pos = len(flat) + ax
        else:
            if ax >= len(prefix):
                return ctx.default_return_type
            pos = ax
        if pos in chosen:
            ctx.api.fail(f"duplicate axis {ax}", ctx.context)
            return ctx.default_return_type
        chosen.add(pos)
    if keepdims:
        out = [1 if i in chosen else d for i, d in enumerate(flat)]
    else:
        out = [d for i, d in enumerate(flat) if i not in chosen]
    return _ndarray_type(ctx, out, out_dtype)


def _reduction_inputs(
    ctx: FunctionContext | MethodContext, array: Type
) -> Type:
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    keep_pair = _formal_pair(ctx, "keepdims")
    keepdims = False if keep_pair is None else _fold_flag(keep_pair)
    dtype_pair = _formal_pair(ctx, "dtype")
    has_dtype = dtype_pair is not None and not _is_none_type(dtype_pair[0])
    dist = _distribute_solve(
        [array], lambda t: _reduction_types(ctx, t, axis, keepdims, has_dtype)
    )
    if dist is not None:
        return dist
    return _reduction_types(ctx, array, axis, keepdims, has_dtype)


def _reduction_types(
    ctx: FunctionContext | MethodContext,
    array: Type,
    axis: list[int] | None | str,
    keepdims: bool | None,
    has_dtype: bool,
) -> Type:
    parts = _parts_or_unpack(array)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    if _has_unpack(dims):
        split = _split_flat(dims)
        if split is None:
            return ctx.default_return_type
        return _solve_reduction_unpack(ctx, *split, dtype, axis, keepdims, has_dtype)
    return _solve_reduction(ctx, dims, dtype, axis, keepdims, has_dtype)


def reduction_method_hook(ctx: MethodContext) -> Type:
    return _reduction_inputs(ctx, ctx.type)


def _enum_member_value(modules: dict, tp: Type) -> int | None:
    """Int value of an enum member literal (F.x -> 0), else None.

    The literal carries the member name; the value comes from the
    member's own type in the enum. Anything non-conforming bails out.
    """
    proper = get_proper_type(tp)
    name: str | None = None
    info = None
    if isinstance(proper, LiteralType) and isinstance(proper.value, str):
        fallback = get_proper_type(proper.fallback)
        if isinstance(fallback, Instance) and fallback.type.is_enum:
            name, info = proper.value, fallback.type
    elif (
        isinstance(proper, Instance)
        and proper.type.is_enum
        and proper.last_known_value is not None
    ):
        inner = get_proper_type(proper.last_known_value)
        if isinstance(inner, LiteralType) and isinstance(inner.value, str):
            name, info = inner.value, proper.type
    if name is None or info is None or name not in info.names:
        return None
    node = info.names[name].node
    if not isinstance(node, Var) or node.type is None:
        return None
    return _int_value(node.type)


def _fold_index_bound(
    ctx: FunctionContext | MethodContext, expr: Expression | None
) -> int | None | str:
    """Slice bound: 'missing' for omitted/None, value, or None to bail."""
    if expr is None:
        return "missing"
    if isinstance(expr, NameExpr) and expr.name == "None":
        return "missing"
    if isinstance(expr, MemberExpr) and expr.name == "newaxis":
        return "missing"
    value = _fold_int(expr)
    if value is None and isinstance(expr, MemberExpr):
        try:
            bound_tp = ctx.api.get_expression_type(expr)
        except Exception:
            bound_tp = None
        if bound_tp is not None:
            modules = getattr(ctx.api, "modules", None) or {}
            value = _enum_member_value(modules, bound_tp)
    return value


def _final_slice_call(
    ctx: FunctionContext | MethodContext, expr: Expression
) -> CallExpr | None:
    """slice(...) behind a Final slice-valued name, else None.

    The value lives in the defining module's assignment, reached
    through the Var's fullname; anything else (locals, non-slice
    initializers, shadowed builtins) bails out.
    """
    if not isinstance(expr, NameExpr):
        return None
    node = expr.node
    if not isinstance(node, Var) or not node.is_final:
        return None
    fullname = node.fullname
    if not fullname or "." not in fullname:
        return None
    modname, _, attr = fullname.rpartition(".")
    modules = getattr(ctx.api, "modules", None) or {}
    mod = modules.get(modname)
    # api.modules maps names to MypyFile (older mypy used State).
    tree = getattr(mod, "tree", None) or mod
    names = getattr(tree, "names", None)
    if names is None or attr not in names or names[attr].node is not node:
        return None
    for stmt in getattr(tree, "defs", []):
        if not isinstance(stmt, AssignmentStmt):
            continue
        for target in stmt.lvalues:
            if isinstance(target, NameExpr) and target.node is node:
                init = stmt.rvalue
                if not isinstance(init, CallExpr):
                    return None
                callee = init.callee
                if (
                    isinstance(callee, NameExpr)
                    and getattr(callee.node, "fullname", "") == "builtins.slice"
                ):
                    return init
                return None
    return None


def _parse_index_item(
    ctx: FunctionContext | MethodContext, tp: Type, expr: Expression | None
) -> tuple | None:
    """One index: int drops the axis, slice keeps it, None adds one.

    Integer arrays are fancy indices (their shape replaces the axis);
    0-d integer arrays act like scalars. Bool arrays are masks (bail:
    the selected count is data-dependent); other dtypes always raise.
    """
    if _is_none_type(tp):
        return ("newaxis",)
    if isinstance(expr, EllipsisExpr):
        return ("ellipsis",)
    part = _ndarray_parts(tp)
    if part is not None:
        kind = _dtype_base(part[1])
        if kind is None:
            return None
        if kind in ("i", "u"):
            return ("fancy0d",) if not part[0] else ("fancy", part[0])
        if kind == "b":
            return ("mask",)
        return ("badindex",)
    if isinstance(expr, SliceExpr):
        begin = _fold_index_bound(ctx, expr.begin_index)
        end = _fold_index_bound(ctx, expr.end_index)
        stride = _fold_index_bound(ctx, expr.stride)
        if begin is None or end is None or stride is None:
            return None
        return ("slice", begin, end, stride)
    if expr is not None:
        init = _final_slice_call(ctx, expr)
        if init is not None and all(k == ARG_POS for k in init.arg_kinds):
            if len(init.args) == 1:
                raw = (None, init.args[0], None)
            elif len(init.args) == 2:
                raw = (init.args[0], init.args[1], None)
            elif len(init.args) == 3:
                raw = (init.args[0], init.args[1], init.args[2])
            else:
                return None
            bounds = tuple(
                "missing" if b is None else _fold_int(b) for b in raw
            )
            if any(b is None for b in bounds):
                return None
            begin, end, stride = bounds
            return ("slice", begin, end, stride)
    if isinstance(expr, StarExpr):
        return None
    value = _fold_int(expr) if expr is not None else None
    if value is None:
        value = _int_value(tp)
    if value is None:
        modules = getattr(ctx.api, "modules", None) or {}
        value = _enum_member_value(modules, tp)
    if isinstance(value, bool):
        return None
    if value is not None:
        return ("int", value)
    # Unknown-value integer index: consumes the axis, no bounds check.
    proper_tp = get_proper_type(tp)
    if isinstance(proper_tp, Instance):
        if proper_tp.type.fullname == INT_FULLNAME:
            return ("int_any",)
        if proper_tp.type.fullname != "builtins.bool":
            key = _scalar_key(tp)
            if key is not None and key[0] == "strong" and _kind_of(key[1]) in ("i", "u"):
                return ("int_any",)
    return None


def _index_pairs(
    ctx: MethodContext, pair: tuple[Type, Expression | None]
) -> list[tuple[Type, Expression | None]] | None:
    """One (type, expr) pair per index position, else None to bail."""
    tp, expr = pair
    proper = get_proper_type(tp)
    if isinstance(expr, TupleExpr) and isinstance(proper, TupleType) and len(expr.items) == len(proper.items):
        return list(zip(proper.items, expr.items))
    if isinstance(proper, TupleType) and not isinstance(expr, TupleExpr):
        # Stored tuple variable: ints and None from types, rest bails.
        return [(item, None) for item in proper.items]
    if isinstance(expr, TupleExpr) or isinstance(proper, TupleType):
        return None
    return [(tp, expr)]


def _parse_index_items(
    ctx: MethodContext, pairs: list[tuple[Type, Expression | None]]
) -> list[tuple] | None:
    """Parse each index position, else None to bail."""
    items: list[tuple] = []
    for itp, iexpr in pairs:
        item = _parse_index_item(ctx, itp, iexpr)
        if item is None:
            return None
        items.append(item)
    return items


def _consume_one(
    ctx: MethodContext, item: tuple, size: int | Type, axis: str, shape_repr: str
) -> list[int | Type] | None:
    """Apply one parsed index item to a known dim: kept dims, or None.

    None means bail out silently, except provable errors (out-of-range
    int, zero step, bad index type), which fail inside and also yield
    None. Dropped dims contribute nothing.
    """
    kind = item[0]
    if kind in ("int", "int_any"):
        if kind == "int" and isinstance(size, int):
            norm = item[1] + size if item[1] < 0 else item[1]
            if not 0 <= norm < size:
                ctx.api.fail(
                    f"index {item[1]} is out of range for axis {axis} "
                    f"with size {size} (shape {shape_repr})",
                    ctx.context,
                )
        return []
    if kind == "fancy0d":
        return []
    if kind in ("fancy", "mask"):
        # Needs concrete ranks to place; with *Ts that is unknowable.
        return None
    if kind == "badindex":
        ctx.api.fail(
            "arrays used as indices must be of integer (or boolean) type",
            ctx.context,
        )
        return None
    _, begin, end, stride = item
    if not isinstance(size, int):
        if begin == "missing" and end == "missing" and stride == "missing":
            return [size]
        return None
    bounds = tuple(None if b == "missing" else b for b in (begin, end, stride))
    try:
        return [len(range(*slice(*bounds).indices(size)))]
    except ValueError:
        ctx.api.fail("slice step is zero", ctx.context)
        return None


def _consume_prefix(
    ctx: MethodContext,
    items: list[tuple],
    prefix: list[int | Type],
    shape_repr: str,
) -> tuple[list[int | Type], list[int | Type]] | None:
    """Items before ... consume the known prefix; (kept, unmatched)."""
    out: list[int | Type] = []
    i = 0
    for item in items:
        if item[0] == "newaxis":
            out.append(1)
            continue
        if i >= len(prefix):
            # Reaches into *Ts: the dim is unknowable, so bail out.
            return None
        frag = _consume_one(ctx, item, prefix[i], str(i), shape_repr)
        if frag is None:
            return None
        out.extend(frag)
        i += 1
    return out, prefix[i:]


def _consume_suffix(
    ctx: MethodContext,
    items: list[tuple],
    suffix: list[int | Type],
    shape_repr: str,
) -> tuple[list[int | Type], list[int | Type]] | None:
    """Items after ... consume the known suffix; (unmatched, kept)."""
    slot: list[list[int | Type] | None] = [None] * len(items)
    p = len(suffix) - 1
    for idx in range(len(items) - 1, -1, -1):
        item = items[idx]
        if item[0] == "newaxis":
            continue
        if p < 0:
            return None
        frag = _consume_one(ctx, item, suffix[p], f"-{len(suffix) - p}", shape_repr)
        if frag is None:
            return None
        slot[idx] = frag
        p -= 1
    out: list[int | Type] = []
    for idx, item in enumerate(items):
        if item[0] == "newaxis":
            out.append(1)
        else:
            assert slot[idx] is not None
            out.extend(slot[idx])
    return suffix[: p + 1], out


def _solve_getitem_unpack(
    ctx: MethodContext,
    prefix: list[int | Type],
    unpack: UnpackType,
    suffix: list[int | Type],
    dtype: Type,
    pair: tuple[Type, Expression | None],
    shape_repr: str,
) -> Type:
    """Index into a (*prefix, *Ts, *suffix) shape.

    An ellipsis aligns the index with the unknown middle; without one,
    leading items consume the known prefix (a `None` inserts 1 in
    place, exactly like the fixed-rank path) and the unknown block plus
    the suffix pass through. Anything reaching into *Ts bails out, as
    do fancy indices, which need concrete ranks to place.
    """
    pairs = _index_pairs(ctx, pair)
    if pairs is None:
        return ctx.default_return_type
    items = _parse_index_items(ctx, pairs)
    if items is None:
        return ctx.default_return_type
    if sum(1 for item in items if item[0] == "ellipsis") > 1:
        ctx.api.fail("only one Ellipsis (...) is allowed in an index", ctx.context)
        return ctx.default_return_type
    positions = [n for n, item in enumerate(items) if item[0] == "ellipsis"]
    if not positions:
        leading = _consume_prefix(ctx, items, prefix, shape_repr)
        if leading is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, leading[0] + leading[1] + [unpack] + suffix, dtype)
    pos = positions[0]
    left = _consume_prefix(ctx, items[:pos], prefix, shape_repr)
    if left is None:
        return ctx.default_return_type
    right = _consume_suffix(ctx, items[pos + 1 :], suffix, shape_repr)
    if right is None:
        return ctx.default_return_type
    out = left[0] + left[1] + [unpack] + right[0] + right[1]
    return _ndarray_type(ctx, out, dtype)


def _solve_getitem(
    ctx: MethodContext,
    dims: list[int | Type],
    dtype: Type,
    pair: tuple[Type, Expression | None],
) -> Type:
    pairs = _index_pairs(ctx, pair)
    if pairs is None:
        return ctx.default_return_type
    items = _parse_index_items(ctx, pairs)
    if items is None:
        return ctx.default_return_type
    if sum(1 for item in items if item[0] == "ellipsis") > 1:
        ctx.api.fail("only one Ellipsis (...) is allowed in an index", ctx.context)
        return ctx.default_return_type
    ndim = len(dims)
    consuming = sum(
        1 for item in items if item[0] in ("int", "int_any", "slice", "fancy", "fancy0d")
    )
    if consuming > ndim:
        ctx.api.fail(
            f"too many indices for {ndim} dimensions "
            f"(got {consuming}, shape {_dims_str(dims)})",
            ctx.context,
        )
        return ctx.default_return_type
    fancy_positions = [n for n, item in enumerate(items) if item[0] == "fancy"]
    if len(fancy_positions) > 1:
        between = items[fancy_positions[0] + 1 : fancy_positions[-1]]
        if any(item[0] not in ("int", "fancy", "fancy0d", "newaxis") for item in between):
            # Advanced indices separated by a slice/ellipsis broadcast to
            # the front; that reshuffle is out of scope, so bail out.
            return ctx.default_return_type
    fancy_dims: list[int] = []
    for n in fancy_positions:
        assert items[n][0] == "fancy"
        shape = items[n][1]
        if not fancy_dims:
            fancy_dims = shape
            continue
        merged, cause = _broadcast_result(fancy_dims, shape)
        if cause is not None:
            ctx.api.fail(
                f"indexing arrays could not be broadcast together: {cause}",
                ctx.context,
            )
            return ctx.default_return_type
        assert merged is not None
        fancy_dims = merged
    out: list[int | Type] = []
    dim = 0
    fancy_emitted = False
    for item in items:
        kind = item[0]
        if kind == "newaxis":
            out.append(1)
        elif kind == "ellipsis":
            span = ndim - consuming
            out.extend(dims[dim : dim + span])
            dim += span
        elif kind in ("int", "int_any"):
            size = dims[dim]
            if kind == "int" and isinstance(size, int):
                norm = item[1] + size if item[1] < 0 else item[1]
                if not 0 <= norm < size:
                    ctx.api.fail(
                        f"index {item[1]} is out of range for axis {dim} "
                        f"with size {size} (shape {_dims_str(dims)})",
                        ctx.context,
                    )
                    return ctx.default_return_type
            dim += 1
        elif kind == "fancy0d":
            dim += 1
        elif kind == "fancy":
            if not fancy_emitted:
                out.extend(fancy_dims)
                fancy_emitted = True
            dim += 1
        elif kind == "mask":
            # Boolean masks select a data-dependent count: bail out.
            return ctx.default_return_type
        elif kind == "badindex":
            ctx.api.fail(
                "arrays used as indices must be of integer (or boolean) type",
                ctx.context,
            )
            return ctx.default_return_type
        else:
            _, begin, end, stride = item
            size = dims[dim]
            if not isinstance(size, int):
                if begin == "missing" and end == "missing" and stride == "missing":
                    out.append(size)
                else:
                    return ctx.default_return_type
                dim += 1
                continue
            bounds = tuple(None if b == "missing" else b for b in (begin, end, stride))
            try:
                out.append(len(range(*slice(*bounds).indices(size))))
            except ValueError:
                ctx.api.fail("slice step is zero", ctx.context)
                return ctx.default_return_type
            dim += 1
    out.extend(dims[dim:])
    if not out:
        # Every axis consumed by integers: a numpy scalar, not an array.
        return ctx.default_return_type
    return _ndarray_type(ctx, out, dtype)


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


def getitem_method_hook(ctx: MethodContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) != 1:
        return ctx.default_return_type
    dist = _distribute_solve([ctx.type], lambda t: _getitem_types(ctx, t, pairs[0]))
    if dist is not None:
        return dist
    return _getitem_types(ctx, ctx.type, pairs[0])


def _getitem_types(
    ctx: MethodContext, tp: Type, pair: tuple[Type, Expression | None]
) -> Type:
    parts = _ndarray_parts(tp)
    if parts is not None:
        dims, dtype = parts
        return _solve_getitem(ctx, dims, dtype, pair)
    split = _unpack_shape(tp)
    if split is None:
        return ctx.default_return_type
    (prefix, unpack, suffix), dtype = split
    shape_repr = f"({_dims_str(prefix + [unpack] + suffix)})"
    return _solve_getitem_unpack(ctx, prefix, unpack, suffix, dtype, pair, shape_repr)


def _broadcast_dims_unpack(
    ctx: FunctionContext | MethodContext,
    left: Type,
    right: Type,
    a: tuple[list[int | Type], Type] | None,
    b: tuple[list[int | Type], Type] | None,
) -> list[int | Type] | None:
    """Dims-only broadcast where _ndarray_parts bailed on a side."""
    sa = _unpack_shape(left) if a is None else None
    sb = _unpack_shape(right) if b is None else None
    if sa is None and sb is None:
        if a is None:
            return None
        if _scalar_key(right) is None:
            return None
        return a[0]
    if sa is not None and sb is not None:
        (p1, u1, s1), _ = sa
        (p2, u2, s2), _ = sb
        dims, message = _merge_unpack_dims(p1, u1, s1, p2, u2, s2)
        if message is not None:
            ctx.api.fail(message, ctx.context)
            return None
        return dims
    sel, other = (sa, right) if sa is not None else (sb, left)
    assert sel is not None
    (prefix, unpack, suffix), _ = sel
    if _scalar_key(other) is not None:
        return prefix + [unpack] + suffix
    parts = _ndarray_parts(other)
    if parts is not None and parts[0] == []:
        return prefix + [unpack] + suffix
    return None


def _broadcast_dims(
    ctx: FunctionContext | MethodContext, left: Type, right: Type
) -> list[int] | None:
    """Broadcast output dims for two operands, else None to bail or fail."""
    a = _ndarray_parts(left)
    b = _ndarray_parts(right)
    if a is None or b is None:
        return _broadcast_dims_unpack(ctx, left, right, a, b)
    dims_a, _ = a
    dims_b, _ = b
    dims, message = _broadcast_result(dims_a, dims_b)
    if message is not None:
        ctx.api.fail(message, ctx.context)
        return None
    return dims


def compare_method_hook(ctx: MethodContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) != 1:
        return ctx.default_return_type
    dist = _distribute_solve([ctx.type, pairs[0][0]], lambda l, r: _compare_types(ctx, l, r))
    if dist is not None:
        return dist
    return _compare_types(ctx, ctx.type, pairs[0][0])


def _compare_types(ctx: FunctionContext | MethodContext, left: Type, right: Type) -> Type:
    dims = _broadcast_dims(ctx, left, right)
    if dims is None:
        return ctx.default_return_type
    dtype = _construct_dtype(ctx, "numpy.bool")
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def logical_method_hook(ctx: MethodContext) -> Type:
    """& and | on bool arrays: broadcast shapes, bool out.

    Anything else (int arrays, exotic scalars) bails to the stubs,
    exactly like before this hook existed.
    """
    pairs = _all_positionals(ctx)
    if len(pairs) != 1:
        return ctx.default_return_type
    dist = _distribute_solve([ctx.type, pairs[0][0]], lambda l, r: _logical_types(ctx, l, r))
    if dist is not None:
        return dist
    return _logical_types(ctx, ctx.type, pairs[0][0])


def _logical_types(ctx: FunctionContext | MethodContext, left: Type, right: Type) -> Type:
    parts = _parts_or_unpack(left)
    if parts is None or _dtype_base(parts[1]) != "b":
        return ctx.default_return_type
    other = _parts_or_unpack(right)
    if other is not None:
        if _dtype_base(other[1]) != "b":
            return ctx.default_return_type
    elif _bool_value(right) is None:
        return ctx.default_return_type
    dims = _broadcast_dims(ctx, left, right)
    if dims is None:
        return ctx.default_return_type
    dtype = _construct_dtype(ctx, "numpy.bool")
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def _solve_argdrop(
    ctx: FunctionContext | MethodContext, dims: list[int], axis: list[int] | None | str
) -> Type:
    """argmax/argmin/argpartition: drop the axis, integer positions out."""
    if not isinstance(axis, list):
        return ctx.default_return_type
    ndim = len(dims)
    normalized: list[int] = []
    for ax in axis:
        norm = _normalize_axis(ax, ndim)
        if norm is None:
            ctx.api.fail(
                f"axis {ax} is out of range for {ndim} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
            )
            return ctx.default_return_type
        if norm in normalized:
            ctx.api.fail(f"duplicate axis {ax}", ctx.context)
            return ctx.default_return_type
        normalized.append(norm)
    chosen = set(normalized)
    return _ndarray_type(ctx, [d for i, d in enumerate(dims) if i not in chosen], _default_dtype(ctx))


def _solve_same_or_flat(
    ctx: FunctionContext | MethodContext,
    dims: list[int | Type],
    dtype: Type,
    axis: list[int] | None | str,
    int_out: bool,
) -> Type:
    """sort/partition (same shape) and argsort/argpartition (integer out)."""
    if axis == "absent":
        # The default axis is -1, which never changes the shape.
        return _ndarray_type(ctx, dims, _default_dtype(ctx) if int_out else dtype)
    if not isinstance(axis, list):
        if axis == "none":
            total = 1
            for d in dims:
                total *= d
            return _ndarray_type(ctx, [total], _default_dtype(ctx) if int_out else dtype)
        return ctx.default_return_type
    ndim = len(dims)
    for ax in axis:
        if _normalize_axis(ax, ndim) is None:
            ctx.api.fail(
                f"axis {ax} is out of range for {ndim} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
            )
            return ctx.default_return_type
    return _ndarray_type(ctx, dims, _default_dtype(ctx) if int_out else dtype)


ARG_DROP = frozenset({"argmax", "argmin", "argpartition"})
ARG_SAME_INT = frozenset({"argsort"})
ARG_SAME = frozenset({"sort", "partition"})


def _arg_inputs(ctx: FunctionContext | MethodContext, array: Type, op: str) -> Type:
    parts = _ndarray_parts(array)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if op in ARG_DROP:
        return _solve_argdrop(ctx, dims, axis)
    return _solve_same_or_flat(ctx, dims, dtype, axis, int_out=op in ARG_SAME_INT)


def arg_method_hook(ctx: MethodContext, op: str) -> Type:
    return _arg_inputs(ctx, ctx.type, op)


def arg_function_hook(ctx: FunctionContext, op: str) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _arg_inputs(ctx, pairs[0][0], op)


RNG_FLOAT_OPS = frozenset({"normal", "random", "standard_normal", "uniform"})


def rng_method_hook(ctx: MethodContext, op: str) -> Type:
    if op in RNG_FLOAT_OPS:
        size = _parse_axis_arg(_formal_pair(ctx, "size"))
        if not isinstance(size, list):
            return ctx.default_return_type
        dtype = _construct_dtype(ctx, "numpy.float64")
        if dtype is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, size, dtype)
    if op == "integers":
        size = _parse_axis_arg(_formal_pair(ctx, "size"))
        if not isinstance(size, list):
            return ctx.default_return_type
        if _explicit_dtype(ctx):
            return ctx.default_return_type
        dtype = _construct_dtype(ctx, "numpy.int64")
        if dtype is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, size, dtype)
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    first_tp, first_expr = pairs[0]
    if op == "choice":
        size = _parse_axis_arg(_formal_pair(ctx, "size"))
        if not isinstance(size, list):
            return ctx.default_return_type
        value = _fold_int(first_expr) if first_expr is not None else None
        if value is None:
            value = _int_value(first_tp)
        if value is not None:
            dtype = _construct_dtype(ctx, "numpy.int64")
            if dtype is None:
                return ctx.default_return_type
            return _ndarray_type(ctx, size, dtype)
        parts = _ndarray_parts(first_tp)
        if parts is None:
            return ctx.default_return_type
        _, dtype = parts
        return _ndarray_type(ctx, size, dtype)
    if op == "permutation":
        value = _fold_int(first_expr) if first_expr is not None else None
        if value is None:
            value = _int_value(first_tp)
        if value is not None:
            dtype = _construct_dtype(ctx, "numpy.int64")
            if dtype is None:
                return ctx.default_return_type
            return _ndarray_type(ctx, [value], dtype)
        parts = _ndarray_parts(first_tp)
        if parts is None:
            return ctx.default_return_type
        dims, dtype = parts
        return _ndarray_type(ctx, dims, dtype)
    return ctx.default_return_type



def reduction_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _reduction_inputs(ctx, pairs[0][0])


def _promoted_result_type(
    ctx: FunctionContext | MethodContext, dta: Type, dtb: Type
) -> Type | None:
    """dtype[...] for the promotion of two dtypes, else None to bail.

    Reuses an input dtype when the promotion matches it, so precision
    (exact bit widths) survives; constructs only genuinely fresh dtypes.
    """
    name = _promote_dtypes(dta, dtb)
    if name is None:
        return None
    if name == _dtype_fullname(dta):
        return dta
    if name == _dtype_fullname(dtb):
        return dtb
    return _construct_dtype(ctx, name)


def _parse_einsum_term(term: str) -> tuple[list[str], list[str], bool] | None:
    """(leading labels, trailing labels, has_ellipsis) for one operand.

    The ellipsis covers the middle dims; without one, leading holds every
    label and trailing is empty. None when malformed: bail, never error.
    """
    if term.count("...") > 1:
        return None
    if "..." in term:
        pre, post = term.split("...")
        if "." in pre or "." in post:
            return None
    else:
        if "." in term:
            return None
        pre, post = term, ""
    labels = list(pre + post)
    if any(c in ",->" or c.isspace() for c in labels):
        return None
    return list(pre), list(post), "..." in term


def einsum_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    _, eq_expr = pairs[0]
    if not isinstance(eq_expr, StrExpr):
        return ctx.default_return_type
    # An explicit dtype or out means the caller controls the result shape.
    if _formal_pair(ctx, "dtype") is not None:
        return ctx.default_return_type
    if _formal_pair(ctx, "out") is not None:
        return ctx.default_return_type
    operands = pairs[1:]
    parts: list[tuple[list[int], Type]] = []
    for tp, _ in operands:
        part = _ndarray_parts(tp)
        if part is None:
            return ctx.default_return_type
        parts.append(part)
    equation = "".join(eq_expr.value.split())
    if equation.count("->") > 1:
        return ctx.default_return_type
    explicit = "->" in equation
    lhs, rhs = equation.split("->") if explicit else (equation, None)
    terms = lhs.split(",")
    if len(terms) != len(parts):
        ctx.api.fail(
            f"einsum subscripts describe {len(terms)} operand(s) but "
            f"{len(parts)} array(s) given",
            ctx.context,
        )
        return ctx.default_return_type
    label_size: dict[str, int | Type] = {}
    counts: dict[str, int] = {}
    ell_inputs: list[list[int | Type]] = []
    for (dims, _), term in zip(parts, terms):
        parsed = _parse_einsum_term(term)
        if parsed is None:
            return ctx.default_return_type
        pre, post, has_ell = parsed
        if has_ell:
            if len(dims) < len(pre) + len(post):
                ctx.api.fail(
                    f"einsum subscript '{term}' needs at least "
                    f"{len(pre) + len(post)} dimensions but operand has "
                    f"shape {_dims_str(dims)}",
                    ctx.context,
                )
                return ctx.default_return_type
            head = dims[: len(pre)]
            tail = dims[len(dims) - len(post) :] if post else []
            ell_inputs.append(dims[len(pre) : len(dims) - len(post) if post else len(dims)])
            bound = list(zip(pre + post, head + tail))
        else:
            if len(pre) != len(dims):
                ctx.api.fail(
                    f"einsum subscript '{term}' has {len(pre)} labels but "
                    f"operand has rank {len(dims)} (shape {_dims_str(dims)})",
                    ctx.context,
                )
                return ctx.default_return_type
            bound = list(zip(pre, dims))
        for label, size in bound:
            counts[label] = counts.get(label, 0) + 1
            known = label_size.get(label)
            if known is None:
                label_size[label] = size
            elif not _dim_eq(known, size):
                if isinstance(known, int) and isinstance(size, int):
                    ctx.api.fail(
                        f"einsum label '{label}' has conflicting sizes: "
                        f"{known} vs {size} (shape {_dims_str(dims)})",
                        ctx.context,
                    )
                    return ctx.default_return_type
                # Symbolic vs anything unprovable: bail out.
                return ctx.default_return_type
    broadcast: list[int] | None = None
    for edims in ell_inputs:
        if broadcast is None:
            broadcast = edims
            continue
        out, cause = _broadcast_result(broadcast, edims)
        if cause is not None:
            ctx.api.fail(
                f"einsum ellipsis dimensions cannot broadcast: {cause}",
                ctx.context,
            )
            return ctx.default_return_type
        assert out is not None
        broadcast = out
    ell_out = broadcast or []
    if explicit:
        assert rhs is not None
        parsed = _parse_einsum_term(rhs)
        if parsed is None:
            return ctx.default_return_type
        pre, post, has_ell = parsed
        if not has_ell and ell_inputs:
            ctx.api.fail(
                "einsum output omits the ellipsis dimensions present "
                "in the inputs",
                ctx.context,
            )
            return ctx.default_return_type
        seen: set[str] = set()
        for label in pre + post:
            if label in seen:
                ctx.api.fail(
                    f"einsum output subscript '{label}' appears more than once",
                    ctx.context,
                )
                return ctx.default_return_type
            seen.add(label)
            if label not in label_size:
                ctx.api.fail(
                    f"einsum output subscript '{label}' never appeared "
                    "in an input",
                    ctx.context,
                )
                return ctx.default_return_type
        out_dims = (
            [label_size[label] for label in pre]
            + ell_out
            + [label_size[label] for label in post]
        )
    else:
        once = sorted(label for label, count in counts.items() if count == 1)
        out_dims = ell_out + [label_size[label] for label in once]
    if not out_dims:
        # Scalar result (trace, full reduction): the stubs own it.
        return ctx.default_return_type
    dtype = _promote_all(ctx, [dtype for _, dtype in parts])
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, out_dims, dtype)


def _linalg_tuple(ctx: FunctionContext | MethodContext, items: list[Type]) -> Type:
    fallback = ctx.api.named_generic_type(
        TUPLE_FULLNAME, [AnyType(TypeOfAny.explicit)]
    )
    return TupleType(items, fallback)


def _linalg_real_name(kind: str, size: int) -> str | None:
    """Float dtype for real-valued results: int inputs, norm, svd values."""
    if kind == "f":
        return _NAME_OF_KIND_SIZE.get(("f", size))
    if kind == "c":
        return _NAME_OF_KIND_SIZE.get(("f", size // 2))
    if kind in ("i", "u", "b"):
        return "numpy.float64"
    return None


def _linalg_dtype(ctx: FunctionContext | MethodContext, dtype: Type, name: str) -> Type | None:
    if name == _dtype_fullname(dtype):
        return dtype
    return _construct_dtype(ctx, name)


def _linalg_eig_names(kind: str, size: int, hermitian: bool) -> tuple[str, str] | None:
    """(values, vectors) dtype names for eig/eigh, else None to bail."""
    if kind == "f" and size in (4, 8):
        if hermitian:
            name = _NAME_OF_KIND_SIZE.get(("f", size))
            assert name is not None
            return name, name
        name = _NAME_OF_KIND_SIZE.get(("c", size * 2))
        assert name is not None
        return name, name
    if kind == "c" and size in (8, 16):
        if hermitian:
            float_name = _NAME_OF_KIND_SIZE.get(("f", size // 2))
            complex_name = _NAME_OF_KIND_SIZE.get(("c", size))
            assert float_name is not None and complex_name is not None
            return float_name, complex_name
        name = _NAME_OF_KIND_SIZE.get(("c", size))
        assert name is not None
        return name, name
    if kind in ("i", "u", "b"):
        if hermitian:
            return "numpy.float64", "numpy.float64"
        return "numpy.complex128", "numpy.complex128"
    return None


def _linalg_solve(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    part_a = _ndarray_parts(pairs[0][0])
    part_b = _ndarray_parts(pairs[1][0])
    if part_a is None or part_b is None:
        return ctx.default_return_type
    dims_a, dtype_a = part_a
    dims_b, dtype_b = part_b
    if len(dims_a) < 2:
        ctx.api.fail(
            "linalg.solve requires a square (..., N, N) array, got shape "
            f"{_dims_str(dims_a)}",
            ctx.context,
        )
        return ctx.default_return_type
    if not _check_dim_eq(
        ctx,
        dims_a[-1],
        dims_a[-2],
        "linalg.solve requires a square (..., N, N) array, got shape "
        f"{_dims_str(dims_a)}",
    ):
        return ctx.default_return_type
    size = dims_a[-1]
    if len(dims_b) == 1:
        if not _check_dim_eq(
            ctx,
            dims_b[0],
            size,
            f"linalg.solve inner dimensions disagree: {_dim_str(size)} vs "
            f"{_dim_str(dims_b[0])} (shapes {_dims_str(dims_a)} vs {_dims_str(dims_b)})",
        ):
            return ctx.default_return_type
        batch_b: list[int | Type] = []
        core = [size]
    else:
        if not _check_dim_eq(
            ctx,
            dims_b[-2],
            size,
            f"linalg.solve inner dimensions disagree: {_dim_str(size)} vs "
            f"{_dim_str(dims_b[-2])} (shapes {_dims_str(dims_a)} vs {_dims_str(dims_b)})",
        ):
            return ctx.default_return_type
        batch_b = dims_b[:-2]
        core = dims_b[-2:]
    batch, cause = _broadcast_result(dims_a[:-2], batch_b)
    if cause is not None:
        ctx.api.fail(
            f"linalg.solve batch dimensions disagree: {_dims_str(dims_a[:-2])} vs "
            f"{_dims_str(batch_b)} ({cause})",
            ctx.context,
        )
        return ctx.default_return_type
    assert batch is not None
    dtype = _promoted_result_type(ctx, dtype_a, dtype_b)
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, [*batch, *core], dtype)


def _linalg_same_shape(ctx: FunctionContext, op: str) -> Type:
    """inv/cholesky: square in, same shape out, ints become float64."""
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    part = _ndarray_parts(pairs[0][0])
    if part is None:
        return ctx.default_return_type
    dims, dtype = part
    if len(dims) < 2:
        ctx.api.fail(
            f"linalg.{op} requires a square (..., N, N) array, got shape "
            f"{_dims_str(dims)}",
            ctx.context,
        )
        return ctx.default_return_type
    if not _check_dim_eq(
        ctx,
        dims[-1],
        dims[-2],
        f"linalg.{op} requires a square (..., N, N) array, got shape "
        f"{_dims_str(dims)}",
    ):
        return ctx.default_return_type
    ks = _dtype_kind_size(dtype)
    if ks is None:
        return ctx.default_return_type
    kind, size = ks
    if kind in ("i", "u", "b"):
        out_dtype = _construct_dtype(ctx, "numpy.float64")
    elif kind in ("f", "c"):
        out_dtype = dtype
    else:
        return ctx.default_return_type
    if out_dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, out_dtype)


def _linalg_eig(ctx: FunctionContext, op: str) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    part = _ndarray_parts(pairs[0][0])
    if part is None:
        return ctx.default_return_type
    dims, dtype = part
    if len(dims) < 2:
        ctx.api.fail(
            f"linalg.{op} requires a square (..., M, M) array, got shape "
            f"{_dims_str(dims)}",
            ctx.context,
        )
        return ctx.default_return_type
    if not _check_dim_eq(
        ctx,
        dims[-1],
        dims[-2],
        f"linalg.{op} requires a square (..., M, M) array, got shape "
        f"{_dims_str(dims)}",
    ):
        return ctx.default_return_type
    ks = _dtype_kind_size(dtype)
    if ks is None:
        return ctx.default_return_type
    names = _linalg_eig_names(*ks, hermitian=op == "eigh")
    if names is None:
        return ctx.default_return_type
    values_dtype = _linalg_dtype(ctx, dtype, names[0])
    vectors_dtype = _linalg_dtype(ctx, dtype, names[1])
    if values_dtype is None or vectors_dtype is None:
        return ctx.default_return_type
    batch = dims[:-2]
    size = dims[-1]
    return _linalg_tuple(
        ctx,
        [
            _ndarray_type(ctx, [*batch, size], values_dtype),
            _ndarray_type(ctx, [*batch, size, size], vectors_dtype),
        ],
    )


def _linalg_flag(ctx: FunctionContext, name: str, default: bool) -> bool | None:
    """Value of an optional bool kwarg (default when absent), else None."""
    pair = _formal_pair(ctx, name)
    if pair is None:
        return default
    return _fold_flag(pair)


def _linalg_svd(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    part = _ndarray_parts(pairs[0][0])
    if part is None:
        return ctx.default_return_type
    dims, dtype = part
    if len(dims) < 2:
        ctx.api.fail(
            f"linalg.svd requires at least 2 dimensions, got shape {_dims_str(dims)}",
            ctx.context,
        )
        return ctx.default_return_type
    if any(not isinstance(d, int) for d in dims[-2:]):
        # K = min(M, N) is unknowable with symbolic dims.
        return ctx.default_return_type
    full = _linalg_flag(ctx, "full_matrices", True)
    compute_uv = _linalg_flag(ctx, "compute_uv", True)
    if full is None or compute_uv is None:
        return ctx.default_return_type
    ks = _dtype_kind_size(dtype)
    if ks is None:
        return ctx.default_return_type
    kind, size = ks
    uv_name: str | None
    if kind in ("f", "c"):
        uv_name = _dtype_fullname(dtype)
    elif kind in ("i", "u", "b"):
        uv_name = "numpy.float64"
    else:
        return ctx.default_return_type
    s_name = _linalg_real_name(kind, size)
    if s_name is None:
        return ctx.default_return_type
    uv_dtype = _linalg_dtype(ctx, dtype, uv_name)
    s_dtype = _linalg_dtype(ctx, dtype, s_name)
    if uv_dtype is None or s_dtype is None:
        return ctx.default_return_type
    batch = dims[:-2]
    rows, cols = dims[-2], dims[-1]
    inner = min(rows, cols)
    s_type = _ndarray_type(ctx, [*batch, inner], s_dtype)
    if not compute_uv:
        return s_type
    if full:
        u_dims, vh_dims = [*batch, rows, rows], [*batch, cols, cols]
    else:
        u_dims, vh_dims = [*batch, rows, inner], [*batch, inner, cols]
    return _linalg_tuple(
        ctx,
        [
            _ndarray_type(ctx, u_dims, uv_dtype),
            s_type,
            _ndarray_type(ctx, vh_dims, uv_dtype),
        ],
    )


def _linalg_lstsq(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    part_a = _ndarray_parts(pairs[0][0])
    part_b = _ndarray_parts(pairs[1][0])
    if part_a is None or part_b is None:
        return ctx.default_return_type
    dims_a, dtype_a = part_a
    dims_b, dtype_b = part_b
    if len(dims_a) < 2:
        ctx.api.fail(
            f"linalg.lstsq requires a with at least 2 dimensions, got shape "
            f"{_dims_str(dims_a)}",
            ctx.context,
        )
        return ctx.default_return_type
    rows, cols = dims_a[-2], dims_a[-1]
    if len(dims_b) == 1:
        if not _check_dim_eq(
            ctx,
            dims_b[0],
            rows,
            f"linalg.lstsq inner dimensions disagree: {_dim_str(rows)} vs "
            f"{_dim_str(dims_b[0])} (shapes {_dims_str(dims_a)} vs {_dims_str(dims_b)})",
        ):
            return ctx.default_return_type
        batch_b: list[int | Type] = []
        core = [cols]
    else:
        if not _check_dim_eq(
            ctx,
            dims_b[-2],
            rows,
            f"linalg.lstsq inner dimensions disagree: {_dim_str(rows)} vs "
            f"{_dim_str(dims_b[-2])} (shapes {_dims_str(dims_a)} vs {_dims_str(dims_b)})",
        ):
            return ctx.default_return_type
        batch_b = dims_b[:-2]
        core = [cols, dims_b[-1]]
    batch, cause = _broadcast_result(dims_a[:-2], batch_b)
    if cause is not None:
        ctx.api.fail(
            f"linalg.lstsq batch dimensions disagree: {_dims_str(dims_a[:-2])} vs "
            f"{_dims_str(batch_b)} ({cause})",
            ctx.context,
        )
        return ctx.default_return_type
    assert batch is not None
    dtype = _promoted_result_type(ctx, dtype_a, dtype_b)
    if dtype is None:
        return ctx.default_return_type
    # Only the solution is provable (residuals depend on runtime rank),
    # so the remaining items stay Any.
    rest = [AnyType(TypeOfAny.explicit)] * 3
    return _linalg_tuple(ctx, [_ndarray_type(ctx, [*batch, *core], dtype), *rest])


def _linalg_norm(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    part = _ndarray_parts(pairs[0][0])
    if part is None:
        return ctx.default_return_type
    dims, dtype = part
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if axis is None:
        return ctx.default_return_type
    keepdims = _linalg_flag(ctx, "keepdims", False)
    if keepdims is None:
        return ctx.default_return_type
    if axis in ("absent", "none") and not keepdims:
        # Scalar result: the stubs own it.
        return ctx.default_return_type
    if axis in ("absent", "none"):
        axes = list(range(len(dims)))
    else:
        assert isinstance(axis, list)
        axes = []
        for ax in axis:
            norm = _normalize_axis(ax, len(dims))
            if norm is None:
                ctx.api.fail(
                    f"linalg.norm axis {ax} is out of range for {len(dims)} "
                    f"dimensions (shape {_dims_str(dims)})",
                    ctx.context,
                )
                return ctx.default_return_type
            axes.append(norm)
        if len(set(axes)) != len(axes):
            return ctx.default_return_type
    if keepdims:
        out_dims = [1 if i in axes else d for i, d in enumerate(dims)]
    else:
        out_dims = [d for i, d in enumerate(dims) if i not in axes]
    if not out_dims:
        return ctx.default_return_type
    ks = _dtype_kind_size(dtype)
    if ks is None:
        return ctx.default_return_type
    name = _linalg_real_name(*ks)
    if name is None:
        return ctx.default_return_type
    out_dtype = _linalg_dtype(ctx, dtype, name)
    if out_dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, out_dims, out_dtype)


def linalg_function_hook(ctx: FunctionContext, op: str) -> Type:
    if op == "solve":
        return _linalg_solve(ctx)
    if op in ("inv", "cholesky"):
        return _linalg_same_shape(ctx, op)
    if op in ("eig", "eigh"):
        return _linalg_eig(ctx, op)
    if op == "svd":
        return _linalg_svd(ctx)
    if op == "lstsq":
        return _linalg_lstsq(ctx)
    if op == "norm":
        return _linalg_norm(ctx)
    return ctx.default_return_type


_UFUNC_KEEP = frozenset(
    {
        "square",
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


def _operand_spec(tp: Type) -> tuple[list[int | Type] | None, str, str] | None:
    """(dims, kind, name) for an array or scalar operand, else None to bail.

    dims is None for scalars (broadcast-neutral); kind is "strong" for
    array dtypes and numpy scalars, "weak" for Python scalars.
    """
    part = _parts_or_unpack(tp)
    if part is not None:
        name = _dtype_fullname(part[1])
        if name is None:
            return None
        return (part[0], "strong", name)
    key = _scalar_key(tp)
    if key is None:
        return None
    return (None, key[0], key[1])


_WEAK_DEFAULT = {
    "builtins.bool": "numpy.bool",
    "builtins.int": "numpy.int64",
    "builtins.float": "numpy.float64",
    "builtins.complex": "numpy.complex128",
}


def _combine_specs(
    a: tuple[str, str], b: tuple[str, str]
) -> str | None:
    """Promoted dtype name for two (kind, name) operand specs."""
    (ka, na), (kb, nb) = a, b
    if ka == "strong" and kb == "strong":
        return na if na == nb else _promote_strong(na, nb)
    if ka == "weak" and kb == "weak":
        order = ("builtins.bool", "builtins.int", "builtins.float", "builtins.complex")
        if na not in order or nb not in order:
            return None
        winner = na if order.index(na) >= order.index(nb) else nb
        return _WEAK_DEFAULT[winner]
    weak, strong = (na, nb) if ka == "weak" else (nb, na)
    return _promote_weak(weak, strong)


def _merge_shape_lists(
    a: list[int | Type], b: list[int | Type]
) -> tuple[list[int | Type] | None, str | None]:
    """Pairwise merge where at least one side carries *Ts.

    Both unpacks merge through _merge_unpack_dims; a concrete empty
    shape (0-d, neutral in broadcasting) yields the other side;
    anything else against *Ts is unknowable and bails out silently.
    """
    sa = _split_flat(a) if _has_unpack(a) else None
    sb = _split_flat(b) if _has_unpack(b) else None
    if sa is None or sb is None:
        if sa is not None and b == []:
            return sa[0] + [sa[1]] + sa[2], None
        if sb is not None and a == []:
            return sb[0] + [sb[1]] + sb[2], None
        return None, None
    (p1, u1, s1), (p2, u2, s2) = sa, sb
    return _merge_unpack_dims(p1, u1, s1, p2, u2, s2)


def _broadcast_shapes(
    ctx: FunctionContext | MethodContext, shapes: list[list[int | Type]], op: str
) -> list[int | Type] | None:
    """Broadcast known shapes pairwise; None (after failing) on conflict."""
    dims: list[int | Type] = []
    first = True
    for other in shapes:
        if first:
            dims = list(other)
            first = False
            continue
        if not _has_unpack(dims) and not _has_unpack(other):
            out, cause = _broadcast_result(dims, other)
            if cause is not None:
                ctx.api.fail(f"{op} shapes disagree: {cause}", ctx.context)
                return None
            if out is None:
                return None
            dims = out
            continue
        merged, message = _merge_shape_lists(dims, other)
        if message is not None:
            ctx.api.fail(f"{op} shapes disagree: {message}", ctx.context)
            return None
        if merged is None:
            return None
        dims = merged
    return dims


def _kind_of(name: str) -> str | None:
    info = _DTYPE_INFO.get(name)
    if info is not None:
        return info[0]
    if name in ("builtins.bool", "builtins.int"):
        return "i"
    if name == "builtins.float":
        return "f"
    if name == "builtins.complex":
        return "c"
    return None


def _dtype_base(dtype: Type) -> str | None:
    """i/u/f/c/b kind from a dtype's base class, ignoring bit width.

    Constructors like arange produce platform-dependent unions
    (signedinteger[_32Bit | _64Bit]) that _dtype_kind_size cannot resolve;
    the kind alone is enough for index/ufunc classification.
    """
    proper = get_proper_type(dtype)
    if not isinstance(proper, Instance) or not proper.args:
        return None
    element = get_proper_type(proper.args[0])
    if not isinstance(element, Instance):
        return None
    direct = _DTYPE_INFO.get(element.type.fullname)
    if direct is not None:
        return direct[0]
    return {
        "numpy.floating": "f",
        "numpy.complexfloating": "c",
        "numpy.signedinteger": "i",
        "numpy.unsignedinteger": "u",
    }.get(element.type.fullname)


def _specs_dtype(
    ctx: FunctionContext | MethodContext,
    specs: list[tuple[str, str]],
    array_dtypes: list[Type],
) -> Type | None:
    """dtype[...] for combined (kind, name) specs, reusing an input dtype."""
    current: tuple[str, str] | None = None
    for kind, spec_name in specs:
        if current is None:
            current = (kind, spec_name)
            continue
        combined = _combine_specs(current, (kind, spec_name))
        if combined is None:
            return None
        # Once combined, the running result behaves as a strong dtype.
        current = ("strong", combined)
    if current is None:
        return None
    kind, name = current
    final = _WEAK_DEFAULT.get(name, name) if kind == "weak" else name
    for dtype in array_dtypes:
        if _dtype_fullname(dtype) == final:
            return dtype
    return _construct_dtype(ctx, final)


def _solve_where(ctx: FunctionContext) -> Type:
    args = _positional_actuals(ctx, 3)
    if args is None:
        # Single-argument where(cond) returns indices: the stubs own it.
        return ctx.default_return_type
    dist = _distribute_solve(args, lambda c, x, y: _where_types(ctx, c, x, y))
    if dist is not None:
        return dist
    return _where_types(ctx, args[0], args[1], args[2])


def _where_types(ctx: FunctionContext | MethodContext, cond: Type, x: Type, y: Type) -> Type:
    specs = [_operand_spec(tp) for tp in (cond, x, y)]
    if any(spec is None for spec in specs):
        return ctx.default_return_type
    assert all(spec is not None for spec in specs)
    shapes = [dims for dims, _, _ in specs if dims is not None]
    if not shapes:
        return ctx.default_return_type
    dims = _broadcast_shapes(ctx, shapes, "where")
    if dims is None:
        return ctx.default_return_type
    branches = [(specs[1][1], specs[1][2]), (specs[2][1], specs[2][2])]
    array_dtypes = []
    for tp in (x, y):
        part = _parts_or_unpack(tp)
        if part is not None:
            array_dtypes.append(part[1])
    dtype = _specs_dtype(ctx, branches, array_dtypes)
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def _solve_clip(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    # Keyword bounds not already covered positionally still broadcast.
    for name, position in (("a_min", 2), ("a_max", 3)):
        if len(pairs) < position:
            named = _formal_pair(ctx, name)
            if named is not None and not _is_none_type(named[0]):
                pairs.append(named)
    # None bounds mean "no clipping on that side": skip them.
    pairs = [(tp, expr) for tp, expr in pairs if not _is_none_type(tp)]
    if not pairs:
        return ctx.default_return_type
    types = [tp for tp, _ in pairs]
    dist = _distribute_solve(types, lambda *ts: _clip_types(ctx, list(ts)))
    if dist is not None:
        return dist
    return _clip_types(ctx, types)


def _clip_types(ctx: FunctionContext | MethodContext, types: list[Type]) -> Type:
    # clip preserves the first array operand's dtype (bounds cast to it).
    specs = [_operand_spec(tp) for tp in types]
    if any(spec is None for spec in specs):
        return ctx.default_return_type
    assert all(spec is not None for spec in specs)
    shapes = [dims for dims, _, _ in specs if dims is not None]
    if not shapes:
        return ctx.default_return_type
    dims = _broadcast_shapes(ctx, shapes, "clip")
    if dims is None:
        return ctx.default_return_type
    for tp, (arr_dims, _, _) in zip(types, specs):
        if arr_dims is not None:
            part = _parts_or_unpack(tp)
            assert part is not None
            return _ndarray_type(ctx, dims, part[1])
    return ctx.default_return_type


def _solve_broadcast_to(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    shape_pair = pairs[1] if len(pairs) >= 2 else _formal_pair(ctx, "shape")
    if shape_pair is None:
        return ctx.default_return_type
    dims = _shape_from_arg(shape_pair[0], shape_pair[1])
    if dims is None:
        return ctx.default_return_type
    if any(isinstance(d, int) and d < 0 for d in dims):
        ctx.api.fail(
            f"broadcast_to dimensions must be non-negative, got {_dims_str(dims)}",
            ctx.context,
        )
        return ctx.default_return_type
    dist = _distribute_solve([pairs[0][0]], lambda t: _broadcast_to_types(ctx, t, dims))
    if dist is not None:
        return dist
    return _broadcast_to_types(ctx, pairs[0][0], dims)


def _broadcast_to_types(ctx: FunctionContext, tp: Type, dims: list[int]) -> Type:
    spec = _operand_spec(tp)
    if spec is None:
        return ctx.default_return_type
    arr_dims, kind, name = spec
    if arr_dims is not None:
        if _has_unpack(arr_dims):
            # Compatibility with the target shape is unprovable.
            return ctx.default_return_type
        part = _ndarray_parts(tp)
        if part is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, part[1])
    dtype = _construct_dtype(ctx, _WEAK_DEFAULT.get(name, name))
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def _display_element_kind(tp: Type, expr: Expression | None) -> str | None:
    """weak scalar kind of a flat display element, else None to bail."""
    value = _fold_int(expr) if expr is not None else None
    if isinstance(value, bool):
        return "builtins.bool"
    if isinstance(value, int):
        return "builtins.int"
    proper = get_proper_type(tp)
    if isinstance(proper, LiteralType):
        if isinstance(proper.value, bool):
            return "builtins.bool"
        if isinstance(proper.value, int):
            return "builtins.int"
        if isinstance(proper.value, float):
            return "builtins.float"
        if isinstance(proper.value, complex):
            return "builtins.complex"
        return None
    if isinstance(proper, Instance) and proper.type.fullname in (
        "builtins.float",
        "builtins.complex",
    ):
        return proper.type.fullname
    return None


def _asarray_array_type(
    ctx: FunctionContext | MethodContext, tp: Type, named: Type | None
) -> Type:
    """asarray over one array operand: same shape, optional new dtype."""
    part = _parts_or_unpack(tp)
    if part is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, part[0], named if named is not None else part[1])


def _solve_asarray(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    named = _named_dtype(ctx) if _explicit_dtype(ctx) else None
    if _explicit_dtype(ctx) and named is None:
        return ctx.default_return_type
    dist = _distribute_solve(
        [pairs[0][0]], lambda t: _asarray_array_type(ctx, t, named)
    )
    if dist is not None:
        return dist
    part = _parts_or_unpack(pairs[0][0])
    if part is not None:
        return _ndarray_type(ctx, part[0], named if named is not None else part[1])
    tp, expr = pairs[0]
    elems: list[tuple[Type, Expression | None]] = []
    proper = get_proper_type(tp)
    if isinstance(proper, TupleType):
        elems = list(zip(proper.items, [None] * len(proper.items)))
    elif isinstance(expr, (ListExpr, TupleExpr)):
        checked = [ctx.api.get_expression_type(e) for e in expr.items]
        if any(t is None for t in checked):
            return ctx.default_return_type
        elems = list(zip([t for t in checked if t is not None], expr.items))
    if not elems:
        return ctx.default_return_type
    kinds = [_display_element_kind(t, e) for t, e in elems]
    if any(k is None for k in kinds):
        return ctx.default_return_type
    if named is None:
        if "builtins.complex" in kinds:
            default_name = "numpy.complex128"
        elif "builtins.float" in kinds:
            default_name = "numpy.float64"
        elif "builtins.int" in kinds:
            default_name = "numpy.int64"
        else:
            default_name = "numpy.bool"
        dtype = _construct_dtype(ctx, default_name)
        if dtype is None:
            return ctx.default_return_type
    else:
        dtype = named
    return _ndarray_type(ctx, [len(elems)], dtype)


def _solve_ufunc(ctx: FunctionContext | MethodContext, group: str) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) != 1:
        return ctx.default_return_type
    for name in ("out", "where", "dtype"):
        if _formal_pair(ctx, name) is not None:
            return ctx.default_return_type
    dist = _distribute_solve([pairs[0][0]], lambda t: _ufunc_types(ctx, group, t))
    if dist is not None:
        return dist
    return _ufunc_types(ctx, group, pairs[0][0])


def _ufunc_types(ctx: FunctionContext | MethodContext, group: str, tp: Type) -> Type:
    part = _parts_or_unpack(tp)
    if part is None:
        return ctx.default_return_type
    dims, dtype = part
    if group == "keep":
        return _ndarray_type(ctx, dims, dtype)
    if group == "bool":
        bool_dtype = _construct_dtype(ctx, "numpy.bool")
        if bool_dtype is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, bool_dtype)
    ks = _dtype_kind_size(dtype)
    if ks is None:
        return ctx.default_return_type
    kind, size = ks
    if group == "abs":
        # abs of complex is real; everything else keeps its dtype.
        if kind != "c":
            return _ndarray_type(ctx, dims, dtype)
        name = _linalg_real_name(kind, size)
    elif group == "real":
        # real/imag of real dtypes is the identity.
        if kind != "c":
            return _ndarray_type(ctx, dims, dtype)
        name = _linalg_real_name(kind, size)
    else:
        if kind in ("f", "c"):
            return _ndarray_type(ctx, dims, dtype)
        name = "numpy.float64"
    if name is None:
        return ctx.default_return_type
    out_dtype = _linalg_dtype(ctx, dtype, name)
    if out_dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, out_dtype)


def ufunc_function_hook(ctx: FunctionContext, group: str) -> Type:
    return _solve_ufunc(ctx, group)


def where_function_hook(ctx: FunctionContext) -> Type:
    return _solve_where(ctx)


def clip_function_hook(ctx: FunctionContext) -> Type:
    return _solve_clip(ctx)


def broadcast_to_function_hook(ctx: FunctionContext) -> Type:
    return _solve_broadcast_to(ctx)


def asarray_function_hook(ctx: FunctionContext) -> Type:
    return _solve_asarray(ctx)


def _solve_truediv_pair(
    ctx: FunctionContext | MethodContext, left: Type, right: Type
) -> Type:
    dist = _distribute_solve([left, right], lambda l, r: _truediv_types(ctx, l, r))
    if dist is not None:
        return dist
    return _truediv_types(ctx, left, right)


def _truediv_types(
    ctx: FunctionContext | MethodContext, left: Type, right: Type
) -> Type:
    specs = [_operand_spec(left), _operand_spec(right)]
    if any(spec is None for spec in specs):
        return ctx.default_return_type
    assert all(spec is not None for spec in specs)
    shapes = [dims for dims, _, _ in specs if dims is not None]
    dims = _broadcast_shapes(ctx, shapes, "true_divide") if shapes else []
    if dims is None:
        return ctx.default_return_type
    (_, kind_l, name_l), (_, kind_r, name_r) = specs
    names = {name_l, name_r}
    array_dtypes: list[Type] = []
    for tp in (left, right):
        part = _ndarray_parts(tp)
        if part is not None:
            array_dtypes.append(part[1])
    # Division by (or of) an integer array yields the default float.
    strong_int = any(
        kind == "strong" and _kind_of(name) in ("i", "u", "b")
        for kind, name in ((kind_l, name_l), (kind_r, name_r))
    )
    if strong_int and not any(_kind_of(name) == "c" for name in names):
        dtype = _construct_dtype(ctx, "numpy.float64")
        if dtype is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, dtype)
    dtype = _specs_dtype(ctx, [(kind_l, name_l), (kind_r, name_r)], array_dtypes)
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def truediv_function_hook(ctx: FunctionContext) -> Type:
    args = _positional_actuals(ctx, 2)
    if args is None:
        return ctx.default_return_type
    return _solve_truediv_pair(ctx, args[0], args[1])


def truediv_method_hook(ctx: MethodContext) -> Type:
    args = _positional_actuals(ctx, 1)
    if args is None:
        return ctx.default_return_type
    return _solve_truediv_pair(ctx, ctx.type, args[0])


def identity_method_hook(ctx: MethodContext) -> Type:
    dist = _distribute_solve([ctx.type], lambda t: _identity_type(ctx, t))
    if dist is not None:
        return dist
    return _identity_type(ctx, ctx.type)


def _identity_type(ctx: FunctionContext | MethodContext, tp: Type) -> Type:
    parts = _parts_or_unpack(tp)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    return _ndarray_type(ctx, dims, dtype)


def astype_method_hook(ctx: MethodContext) -> Type:
    # dtype is the first positional or the named formal; _formal_pair
    # covers both through the callee's argument names. Shared by all
    # members, so it resolves once outside distribution.
    named = _named_dtype(ctx)
    if named is None:
        return ctx.default_return_type
    dist = _distribute_solve([ctx.type], lambda t: _astype_type(ctx, t, named))
    if dist is not None:
        return dist
    return _astype_type(ctx, ctx.type, named)


def _astype_type(ctx: FunctionContext | MethodContext, tp: Type, named: Type) -> Type:
    parts = _parts_or_unpack(tp)
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    return _ndarray_type(ctx, dims, named)


def shape_attribute_hook(ctx: AttributeContext) -> Type:
    dist = _distribute_solve([ctx.type], lambda t: _shape_type(ctx, t))
    if dist is not None:
        return dist
    return _shape_type(ctx, ctx.type)


def _shape_type(ctx: AttributeContext, tp: Type) -> Type:
    parts = _parts_or_unpack(tp)
    if parts is None:
        return ctx.default_attr_type
    dims, _ = parts
    int_fallback = ctx.api.named_generic_type("builtins.int", [])
    items: list[Type] = [
        LiteralType(v, fallback=int_fallback) if isinstance(v, int) else v for v in dims
    ]
    fallback = ctx.api.named_generic_type(
        TUPLE_FULLNAME, [AnyType(TypeOfAny.explicit)]
    )
    return TupleType(items, fallback)


def t_attribute_hook(ctx: AttributeContext) -> Type:
    dist = _distribute_solve([ctx.type], lambda t: _t_type(ctx, t))
    if dist is not None:
        return dist
    return _t_type(ctx, ctx.type)


def _t_type(ctx: AttributeContext, tp: Type) -> Type:
    """The .T view: full reversal, concrete or over *Ts.

    Concrete shapes reverse exactly; unpack shapes reverse with the
    unknown block kept in place per _reverse_unpack_dims. Anything else
    stays on the stubs.
    """
    parts = _ndarray_parts(tp)
    if parts is not None:
        dims, dtype = parts
        return _ndarray_type(ctx, list(reversed(dims)), dtype)
    split = _unpack_shape(tp)
    if split is None:
        return ctx.default_attr_type
    (prefix, unpack, suffix), dtype = split
    return _ndarray_type(ctx, _reverse_unpack_dims(prefix, unpack, suffix), dtype)


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


def len_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) != 1:
        return ctx.default_return_type
    int_fallback = ctx.api.named_generic_type("builtins.int", [])
    proper = get_proper_type(pairs[0][0])
    if isinstance(proper, CallableType):
        # An enum class arrives as its constructor; len() counts members.
        ret = get_proper_type(proper.ret_type)
        if isinstance(ret, Instance) and ret.type.is_enum:
            count = _enum_member_count(ret.type)
            if count is not None:
                return LiteralType(count, fallback=int_fallback)
            return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    if not dims:
        ctx.api.fail("len() of unsized object", ctx.context)
        return ctx.default_return_type
    first = dims[0]
    if isinstance(first, TypeVarType):
        return first
    if not isinstance(first, int):
        return ctx.default_return_type
    return LiteralType(first, fallback=int_fallback)


def take_along_axis_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) < 2 or len(pairs) > 3:
        return ctx.default_return_type
    part_a = _ndarray_parts(pairs[0][0])
    part_i = _ndarray_parts(pairs[1][0])
    if part_a is None or part_i is None:
        return ctx.default_return_type
    dims_a, dtype_a = part_a
    dims_i, dtype_i = part_i
    if _dtype_base(dtype_i) not in ("i", "u"):
        ctx.api.fail(
            "take_along_axis `indices` must be an integer array",
            ctx.context,
        )
        return ctx.default_return_type
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if axis is None:
        return ctx.default_return_type
    if axis in ("absent", "none"):
        # axis=None flattens: indices must be 1-d, result is the indices.
        if len(dims_i) != 1:
            ctx.api.fail(
                "take_along_axis with axis=None needs 1-d `indices`, got "
                f"shape {_dims_str(dims_i)}",
                ctx.context,
            )
            return ctx.default_return_type
        return _ndarray_type(ctx, dims_i, dtype_a)
    if not isinstance(axis, list) or len(axis) != 1:
        return ctx.default_return_type
    norm = _normalize_axis(axis[0], len(dims_a))
    if norm is None:
        ctx.api.fail(
            f"take_along_axis axis {axis[0]} is out of range for "
            f"{len(dims_a)} dimensions (shape {_dims_str(dims_a)})",
            ctx.context,
        )
        return ctx.default_return_type
    if len(dims_a) != len(dims_i):
        ctx.api.fail(
            f"take_along_axis `indices` and `arr` must have the same number "
            f"of dimensions (shapes {_dims_str(dims_a)} vs {_dims_str(dims_i)})",
            ctx.context,
        )
        return ctx.default_return_type
    out: list[int | Type] = []
    for d, (x, y) in enumerate(zip(dims_a, dims_i)):
        if d == norm:
            # Along the axis the index wins, except a 1 falls back to
            # the array dim; anything symbolic is unprovable, so bail.
            if isinstance(y, int):
                out.append(x if y == 1 else y)
            else:
                return ctx.default_return_type
        elif _dim_eq(x, y):
            out.append(x)
        elif isinstance(x, int) and x == 1:
            out.append(y)
        elif isinstance(y, int) and y == 1:
            out.append(x)
        elif isinstance(x, int) and isinstance(y, int):
            ctx.api.fail(
                f"take_along_axis shapes mismatch at axis {d}: {x} vs {y} "
                f"(shapes {_dims_str(dims_a)} vs {_dims_str(dims_i)})",
                ctx.context,
            )
            return ctx.default_return_type
        else:
            return ctx.default_return_type
    return _ndarray_type(ctx, out, dtype_a)


def clip_method_hook(ctx: MethodContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    types = [ctx.type] + [tp for tp, _ in pairs]
    dist = _distribute_solve(types, lambda *ts: _clip_types(ctx, list(ts)))
    if dist is not None:
        return dist
    return _clip_types(ctx, types)


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
        "logaddexp",
        "logaddexp2",
    }
)
_UFUNC_BINARY_FLOAT = frozenset({"hypot", "arctan2"})
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


def _ufunc_name(ctx: FunctionContext | MethodContext) -> str | None:
    """The ufunc name from its Literal type argument, else None."""
    proper = get_proper_type(ctx.type)
    if not isinstance(proper, Instance) or not proper.args:
        return None
    first = get_proper_type(proper.args[0])
    if isinstance(first, LiteralType) and isinstance(first.value, str):
        return first.value
    return None


def ufunc_call_hook(ctx: MethodContext) -> Type:
    proper = get_proper_type(ctx.type)
    if isinstance(proper, Instance) and "Nout2" in proper.type.fullname:
        # Two-output ufuncs (frexp, modf): the stubs own the tuple.
        return ctx.default_return_type
    name = _ufunc_name(ctx)
    if name is None:
        return ctx.default_return_type
    if name in _UFUNC_KEEP:
        return _solve_ufunc(ctx, "keep")
    if name in _UFUNC_ABS:
        return _solve_ufunc(ctx, "abs")
    if name in _UFUNC_FLOAT:
        return _solve_ufunc(ctx, "float")
    if name in _UFUNC_REAL:
        return _solve_ufunc(ctx, "real")
    if name in _UFUNC_BOOL:
        return _solve_ufunc(ctx, "bool")
    if name == "matmul":
        pairs = _all_positionals(ctx)
        if len(pairs) != 2:
            return ctx.default_return_type
        return _solve_matmul(ctx, pairs[0][0], pairs[1][0])
    if name in {"divide", "true_divide"}:
        pairs = _all_positionals(ctx)
        if len(pairs) != 2:
            return ctx.default_return_type
        return _solve_truediv_pair(ctx, pairs[0][0], pairs[1][0])
    if name in _UFUNC_BINARY_PROMOTE:
        pairs = _all_positionals(ctx)
        if len(pairs) != 2:
            return ctx.default_return_type
        return _solve_broadcast(ctx, pairs[0][0], pairs[1][0])
    if name in _UFUNC_BINARY_FLOAT:
        pairs = _all_positionals(ctx)
        if len(pairs) != 2:
            return ctx.default_return_type
        return _solve_ufunc_binary_float(ctx, pairs[0][0], pairs[1][0])
    if name in _UFUNC_BINARY_BOOL:
        pairs = _all_positionals(ctx)
        if len(pairs) != 2:
            return ctx.default_return_type
        return _solve_ufunc_binary_bool(ctx, pairs[0][0], pairs[1][0])
    return ctx.default_return_type


def _solve_ufunc_binary_float(
    ctx: FunctionContext | MethodContext, left: Type, right: Type
) -> Type:
    dist = _distribute_solve([left, right], lambda l, r: _ufunc_float_types(ctx, l, r))
    if dist is not None:
        return dist
    return _ufunc_float_types(ctx, left, right)


def _ufunc_float_types(
    ctx: FunctionContext | MethodContext, left: Type, right: Type
) -> Type:
    specs = [_operand_spec(left), _operand_spec(right)]
    if any(spec is None for spec in specs):
        return ctx.default_return_type
    assert all(spec is not None for spec in specs)
    shapes = [dims for dims, _, _ in specs if dims is not None]
    dims = _broadcast_shapes(ctx, shapes, "ufunc") if shapes else []
    if dims is None:
        return ctx.default_return_type
    (_, kind_l, name_l), (_, kind_r, name_r) = specs
    combined = _combine_specs((kind_l, name_l), (kind_r, name_r))
    if combined is None:
        return ctx.default_return_type
    kind = _kind_of(combined)
    if kind in ("f", "c"):
        out_name = combined
    elif kind in ("i", "u", "b"):
        out_name = "numpy.float64"
    else:
        return ctx.default_return_type
    array_dtypes = []
    for tp in (left, right):
        part = _ndarray_parts(tp)
        if part is not None:
            array_dtypes.append(part[1])
    dtype = _specs_dtype(ctx, [("strong", out_name)], array_dtypes)
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def _solve_ufunc_binary_bool(
    ctx: FunctionContext | MethodContext, left: Type, right: Type
) -> Type:
    dist = _distribute_solve([left, right], lambda l, r: _ufunc_bool_types(ctx, l, r))
    if dist is not None:
        return dist
    return _ufunc_bool_types(ctx, left, right)


def _ufunc_bool_types(
    ctx: FunctionContext | MethodContext, left: Type, right: Type
) -> Type:
    specs = [_operand_spec(left), _operand_spec(right)]
    if any(spec is None for spec in specs):
        return ctx.default_return_type
    assert all(spec is not None for spec in specs)
    shapes = [dims for dims, _, _ in specs if dims is not None]
    if not shapes:
        return ctx.default_return_type
    dims = _broadcast_shapes(ctx, shapes, "ufunc")
    if dims is None:
        return ctx.default_return_type
    bool_dtype = _construct_dtype(ctx, "numpy.bool")
    if bool_dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, bool_dtype)

