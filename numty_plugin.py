"""mypy plugin: precise numpy shapes for constructors, matmul, and friends.

mypy-only, registered in mypy.ini. Without the plugin, numty/__init__.py still
checks with plain annotations; only the general solving below is lost
(matmul degrades to its loose fallback, stored constructors widen).

What it does:
- zeros/ones/empty with a fully literal shape return a precise
  ndarray[tuple[Literal[...], ...], ...], even through assignment. A
  tuple display mixing literals with plain int dims keeps the literals
  ((n, 3) -> tuple[int, Literal[3]]). reshape, dot/matmul and
  transposes carry them through (_reshape_dynamic_dims, _mixed_parts);
  ops the solver leaves to the stubs widen them back unless they keep
  the shape (_guard_mixed). Anything else bails out to mypy's default
  instead of erroring.
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
  infers a single -1, returning the precise new shape. A -1 divides the
  canonical total (linear and monomial products alike: N*K/N=K,
  B*C*H*W/B=C*H*W); only inexact/inexpressible inferences mint an
  unknown, and same-total renames track as-is (legality stays unchecked,
  never an error). Over *Ts only block-preserving regroups resolve (the
  unknown middle stays put; -1 would absorb it, so it bails), with
  provable int-only total clashes failing loudly. Anything dynamic bails
  out.
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
  boolean masks track the axis with a fresh unknown (?U), float indices
  error. Over *Ts, leading items consume the known prefix without an
  ellipsis (rank-generic bodies still spell the ellipsis for the middle).
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

import ast
import json
import keyword
import os
from collections.abc import Callable, Iterator
from functools import partial
from typing import NamedTuple, cast

from mypy.errorcodes import ARG_TYPE, MISC, ErrorCode
from mypy.nodes import (
    ARG_NAMED,
    ARG_NAMED_OPT,
    ARG_OPT,
    ARG_POS,
    ARG_STAR,
    ARG_STAR2,
    AssignmentStmt,
    CallExpr,
    Context,
    Decorator,
    EllipsisExpr,
    Expression,
    FuncDef,
    GeneratorExpr,
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
    TypeAlias,
    TypeInfo,
    TypeVarExpr,
    TypeVarLikeExpr,
    UnaryExpr,
    Var,
)
from mypy.messages import callable_name
from mypy.operators import op_methods_to_symbols
from mypy.plugin import (
    AnalyzeTypeContext,
    AttributeContext,
    FunctionContext,
    FunctionSigContext,
    MethodContext,
    MethodSigContext,
    Plugin,
)
from mypy.typeanal import TypeAnalyser
from mypy.tvar_scope import TypeVarLikeScope
from mypy.types import (
    AnyType,
    CallableType,
    EllipsisType,
    Instance,
    LiteralType,
    NoneType,
    Overloaded,
    RawExpressionType,
    TupleType,
    Type,
    TypeAliasType,
    TypeOfAny,
    TypeVarId,
    TypeVarLikeType,
    TypeVarTupleType,
    TypeVarType,
    UnboundType,
    UnionType,
    UnpackType,
    get_proper_type,
)


# Error codes, one per kind of diagnostic a user may want to silence or
# select on its own (`# type: ignore[numty-bail]`,
# `disable_error_code = numty-bail`). All are sub-codes of misc, so an
# existing `ignore[misc]` or `disable_error_code = misc` still covers them.
NUMTY_SHAPE = ErrorCode(
    "numty-shape",
    "Array dimensions, ranks or size arguments do not fit the operation",
    "numty",
    sub_code_of=MISC,
)
NUMTY_BROADCAST = ErrorCode(
    "numty-broadcast", "Array shapes do not broadcast together", "numty", sub_code_of=MISC
)
NUMTY_CALLSITE = ErrorCode(
    "numty-callsite",
    "Shape variable bound to conflicting sizes at a call",
    "numty",
    sub_code_of=MISC,
)
NUMTY_AXIS = ErrorCode(
    "numty-axis", "Axis or index argument out of range or invalid", "numty", sub_code_of=MISC
)
NUMTY_BAIL = ErrorCode(
    "numty-bail", "numty cannot check this expression", "numty", sub_code_of=MISC
)
NUMTY_ANNOTATION = ErrorCode(
    "numty-annotation", "Invalid numty shape annotation", "numty", sub_code_of=MISC
)
NUMTY_UNSOLVED = ErrorCode(
    "numty-unsolved",
    "numty could not prove this shape; the result is unchecked",
    "numty",
    sub_code_of=MISC,
)


def _numty_trace_path() -> str | None:
    """JSONL trace sink from NUMTY_TRACE, or None when tracing is off."""
    path = os.environ.get("NUMTY_TRACE")
    return path if path else None


def _numty_strict() -> bool:
    """Whether NUMTY_STRICT=1 asks for a loud error on every lost shape."""
    return os.environ.get("NUMTY_STRICT") == "1"


def _trace_event(
    ctx: FunctionContext | MethodContext | AttributeContext,
    op: str,
    outcome: str,
    reason: str,
) -> None:
    """Append one {file, line, col, op, outcome, reason} record to NUMTY_TRACE.

    Only modules that are actually re-checked emit events, so run mypy
    with a fresh --cache-dir when collecting a trace. Failures to write
    never break checking.
    """
    path = _numty_trace_path()
    if path is None:
        return
    try:
        api: object = getattr(ctx, "api", None)
        file = str(getattr(api, "path", "") or "")
        context = getattr(ctx, "context", None)
        line = int(getattr(context, "line", 0) or 0)
        col = int(getattr(context, "column", 0) or 0)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "file": file,
                        "line": line,
                        "col": col,
                        "op": op,
                        "outcome": outcome,
                        "reason": reason,
                    }
                )
                + "\n"
            )
    except Exception:
        return


def _operands_have_known_shape(operands: list[Type]) -> bool:
    """Whether any operand is a known-rank ndarray (unhooked trace gate).

    Unhooked numpy.* calls are only traced when precision was actually
    lost: a known-rank input flowing into a stub Any. *Ts shapes count
    as known via _parts_or_unpack.
    """
    try:
        for tp in operands:
            for arr in _ndarrays_in(tp):
                if _parts_or_unpack(arr) is not None:
                    return True
    except Exception:
        return False
    return False


def _bail(
    ctx: FunctionContext | MethodContext,
    op: str,
    reason: str,
) -> Type:
    """Give up on one call: trace it, optionally fail loudly, keep stubs.

    Central helper for the 506 `return ctx.default_return_type` sites:
    it records {op, bail, reason} to NUMTY_TRACE and, under
    NUMTY_STRICT=1, reports [numty-unsolved]. Strict fails for every
    bail, even scalar-only ones like np.sqrt(8): the Any born there
    poisons the known-shape truediv whose hook mypy then skips, so a
    known-shape gate would leave that loss silent. New hooks should
    call this instead of returning the default directly so the reason
    stays specific; _guard_mixed already applies the same policy
    generically to every hook that returns the default.
    """
    _trace_event(ctx, op, "bail", reason)
    if _numty_strict():
        ctx.api.fail(
            f"numty could not prove {op}: {reason} (result is unchecked)",
            ctx.context,
            code=NUMTY_UNSOLVED,
        )
    return ctx.default_return_type


NDARRAY_FULLNAME = "numpy.ndarray"
INT_FULLNAME = "builtins.int"
TUPLE_FULLNAME = "builtins.tuple"
FLOAT64_FULLNAME = "numpy.float64"
DTYPE_FULLNAME = "numpy.dtype"

# Familiar-style heads (defined in numty/__init__.py) and the numpy scalar each one
# stands for. Bool maps to numpy.bool, matching _DTYPE_STRINGS below.
_FAM_HEAD_DTYPES = {
    "numty.F64": "numpy.float64",
    "numty.F32": "numpy.float32",
    "numty.F16": "numpy.float16",
    "numty.I64": "numpy.int64",
    "numty.I32": "numpy.int32",
    "numty.I16": "numpy.int16",
    "numty.I8": "numpy.int8",
    "numty.U8": "numpy.uint8",
    "numty.Bool": "numpy.bool",
}


class _FamFail(Exception):
    """Malformed familiar-style annotation: emit api.fail, return Any."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self.msg = msg


class _FamBail(Exception):
    """Default analysis already errored: return Any without piling on."""


def plugin(version: str) -> type[Plugin]:
    return NumtyPlugin


class NumtyPlugin(Plugin):
    def __init__(self, options) -> None:
        super().__init__(options)
        # Fullnames of TypeVars first bound from inside a string dim spec
        # (see _fam_bind_words). The pre-pass never saw them, so they are
        # missing from the analyzed function's variables; the call-site
        # hook adds them back so concrete calls still check and infer.
        self._fam_late_bound: set[str] = set()

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
            sig = self._callsite_check(ctx)
        except Exception:
            return ctx.default_signature
        _note_rank_mismatches(ctx, sig)
        return sig

    def get_method_signature_hook(
        self, fullname: str
    ) -> Callable[[MethodSigContext], Type] | None:
        # Methods only get the rank-mismatch notes; the call-site conflict
        # check stays function-only.
        if fullname.startswith("numpy.") or fullname.startswith("numty."):
            return None
        # A hook here would make every method call in the program (list,
        # dict and str ones included) walk the plugin chain and scan its
        # arguments. A method none of whose parameters is declared with an
        # ndarray can never get a note, so leave it unhooked: mypy then
        # asks the later plugins itself.
        if not self._method_may_take_array(fullname):
            return None
        return partial(self.method_signature_hook, fullname)

    def _method_may_take_array(self, fullname: str) -> bool:
        """False when `fullname`'s declared parameters mention no ndarray.

        A cheap, conservative look at the declaration: anything it cannot
        resolve counts as possibly taking an array.
        """
        class_name, _, method = fullname.rpartition(".")
        if method in _OPERATOR_DUNDERS:
            return False  # never noted, see _rank_mismatch_notes
        try:
            sym = self.lookup_fully_qualified(class_name)
        except Exception:
            return True
        if sym is None or not isinstance(sym.node, TypeInfo):
            return True
        member = sym.node.get(method)
        if member is None:
            return True
        node = member.node
        if isinstance(node, Decorator):
            node = node.var
        tp = get_proper_type(getattr(node, "type", None))
        if tp is None and isinstance(node, FuncDef):
            return False  # unannotated: every parameter is Any
        if isinstance(tp, CallableType):
            items = [tp]
        elif isinstance(tp, Overloaded):
            items = tp.items
        else:
            return True
        return any(_mentions_ndarray(arg) for item in items for arg in item.arg_types)

    def method_signature_hook(self, fullname: str, ctx: MethodSigContext) -> Type:
        # This hook covers every method, so first run whatever a later
        # plugin (mypy's default one included) would have hooked here.
        sig: Type = ctx.default_signature
        hook = self._later_plugin_hook(ctx, "get_method_signature_hook", fullname)
        if hook is not None:
            sig = hook(ctx)
        # Dataclass __init__ (issue #29): string-spec words bound late
        # (see _fam_bind_words) are missing from the method's variables,
        # so `A1(grid=int-dims)` rigid-fails instead of inferring the
        # implicit dims like a generic function does. Add exactly those
        # back, same as the function call-site hook does.
        try:
            if isinstance(sig, CallableType):
                have = {v.fullname for v in sig.variables}
                missing: list[TypeVarLikeType] = []
                for formal in list(sig.arg_types) + [sig.ret_type]:
                    for tvar in _iter_tvarlikes(formal):
                        if (
                            tvar.fullname in self._fam_late_bound
                            and tvar.fullname not in have
                        ):
                            have.add(tvar.fullname)
                            missing.append(tvar)
                if missing:
                    sig = sig.copy_modified(
                        variables=tuple(sig.variables) + tuple(missing)
                    )
        except Exception:
            pass
        _note_rank_mismatches(ctx, sig)
        return sig

    def _later_plugin_hook(
        self, ctx: MethodSigContext, getter: str, fullname: str
    ) -> Callable[[MethodSigContext], Type] | None:
        """The hook the plugins after this one return for `fullname`.

        mypy asks plugins in order and takes the first hook, so a
        catch-all hook here would hide theirs. Reaches through the
        checker's ChainedPlugin (`_plugins`; pinned with mypy).
        """
        chain = getattr(getattr(ctx.api, "plugin", None), "_plugins", None)
        if not isinstance(chain, list) or self not in chain:
            return None
        for plugin in chain[chain.index(self) + 1 :]:
            hook = getattr(plugin, getter)(fullname)
            if hook is not None:
                return hook
        return None

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
        by_var: dict[str, tuple[str, list[tuple[str, int, int]]]] = {}
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
                occs.append((formal_name, dim, adims[dim]))
                by_var[tvar.fullname] = (name, occs)
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
        # String-spec words bound late (see _fam_bind_words) are missing
        # from the function's variables, so core checking would reject
        # concrete actuals against the free typevars. Add exactly those
        # back: the call then checks and infers like a written generic.
        # Names already generic (mixed bare + string) are left alone, so
        # no fullname ever appears twice.
        have = {v.fullname for v in sig.variables}
        missing: list[TypeVarLikeType] = []
        for formal in list(sig.arg_types) + [sig.ret_type]:
            for tvar in _iter_tvarlikes(formal):
                if (
                    tvar.fullname in self._fam_late_bound
                    and tvar.fullname not in have
                ):
                    have.add(tvar.fullname)
                    missing.append(tvar)
        if missing:
            sig = sig.copy_modified(
                variables=tuple(sig.variables) + tuple(missing)
            )
        # Expression returns (issue #32 finish): mypy cannot infer M+1
        # from M, so callers of `-> I64["M+1"]` saw Any. Substitute the
        # actual bindings into the return ourselves (plain vars reuse the
        # caller's object, new exprs reuse the form-keyed global id).
        try:
            bindings = _callsite_expr_bindings(sig, formals, ctx)
            new_ret = _substitute_return_exprs(sig, bindings, ctx)
            if new_ret is not None:
                sig = sig.copy_modified(ret_type=new_ret)
        except Exception:
            pass
        return sig

    # Familiar-style heads (_FAM_HEAD_DTYPES, defined in numty/__init__.py): the
    # dtype rides the head name, dims ride flat args. Hooked here because
    # mypy resolves `Head[...]` through Annotated-free generic application,
    # so the raw args (quoted names included) still reach us.
    def get_type_analyze_hook(
        self, fullname: str
    ) -> Callable[[AnalyzeTypeContext], Type] | None:
        dtype_fullname = _FAM_HEAD_DTYPES.get(fullname)
        if dtype_fullname is None:
            return None

        def hook(ctx: AnalyzeTypeContext) -> Type:
            return self._fam_head(ctx, dtype_fullname)

        return hook

    def _fam_head(self, ctx: AnalyzeTypeContext, dtype_fullname: str) -> Type:
        """Build the ndarray type for `F64[...]` from flat raw args.

        Every dim must resolve: a TypeVar (reused, so mypy binds it to
        the function and inference keeps working), a TypeVarTuple via
        Unpack[Ts], a Final integer, a bare integer, Literal[...], or a
        jaxtyping-style string spec (`"batch features"`, one dim per
        word). Anything else fails loudly; anything already errored
        bails silently to Any.
        """
        head = ctx.type.name
        raw_args = list(ctx.type.args)
        try:
            if not raw_args:
                raise _FamFail(f"{head} takes at least one dim")
            dims: list[int | Type] = []
            for a in raw_args:
                parsed = self._fam_arg(ctx, head, a)
                if isinstance(parsed, list):
                    dims.extend(parsed)
                else:
                    dims.append(parsed)
        except _FamFail as bad:
            ctx.api.fail(bad.msg, ctx.context, code=NUMTY_ANNOTATION)
            return AnyType(TypeOfAny.from_error)
        except _FamBail:
            return AnyType(TypeOfAny.from_error)
        line, column = ctx.context.line, ctx.context.column
        int_inst = ctx.api.named_type("builtins.int", [], line, column)
        int_tuple = ctx.api.named_type("builtins.tuple", [int_inst], line, column)
        items: list[Type] = []
        for d in dims:
            if isinstance(d, int):
                items.append(LiteralType(d, int_inst, line, column))
            elif isinstance(d, UnpackType) and isinstance(
                get_proper_type(d.type), TypeVarTupleType
            ):
                # A TypeVarTuple is bounded at tuple[object, ...], so
                # numpy's tuple[int, ...] shape bound rejects any shape
                # holding it (and every builder fed a slice of it, e.g.
                # np.zeros((n, *x.shape[1:]))). In a shape every member
                # is an int, so say so on this occurrence: same id, so
                # binding, inference and identity are unchanged.
                tvt = cast(TypeVarTupleType, get_proper_type(d.type))
                items.append(UnpackType(tvt.copy_modified(upper_bound=int_tuple)))
            else:
                items.append(d)
        shape = TupleType(
            items,
            ctx.api.named_type(
                "builtins.tuple", [AnyType(TypeOfAny.explicit)], line, column
            ),
            line=line,
            column=column,
        )
        # The element may be a stub alias (numpy.int32 is), not a class:
        # use the alias target itself, so formals match what the numpy
        # stubs produce for the same scalar (e.g. signedinteger[_32Bit]).
        element: Type | None = None
        sym = ctx.api.lookup_fully_qualified(dtype_fullname)
        node = sym.node if sym is not None else None
        if isinstance(node, TypeInfo):
            element = ctx.api.named_type(dtype_fullname, [], line, column)
        elif isinstance(node, TypeAlias):
            target = get_proper_type(node.target)
            if isinstance(target, Instance):
                element = target
        if element is None:
            raise _FamFail(f"{head}: cannot resolve dtype {dtype_fullname}")
        dtype = ctx.api.named_type("numpy.dtype", [element], line, column)
        return ctx.api.named_type("numpy.ndarray", [shape, dtype], line, column)

    def _fam_arg(
        self, ctx: AnalyzeTypeContext, head: str, arg: Type
    ) -> int | Type | list[int | Type]:
        """One flat dim, or a string spec expanding to several dims."""
        if isinstance(arg, EllipsisType):
            raise _FamFail(
                f"{head} '...' cannot be named for shape solving;"
                " use Unpack[Ts] with a declared TypeVarTuple"
            )
        if isinstance(arg, RawExpressionType):
            value = arg.literal_value
            if isinstance(value, str):
                return self._fam_string(ctx, head, value, arg)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise _FamFail(
                    f"{head} dims must be TypeVars, non-negative integers,"
                    ' or a string spec like "batch features"'
                )
            return value
        if isinstance(arg, UnboundType):
            if arg.args:
                return self._fam_applied(ctx, head, arg)
            if arg.original_str_expr is not None:
                # A quoted single-word spec ("hid"): same path as a
                # multi-word string, so undeclared words mint implicit
                # dims exactly like "hid rest" would.
                return self._fam_string(ctx, head, arg.name, arg)
            return self._fam_bare(ctx, head, arg)
        # Defensive: hooks see raw args, but accept analyzed nodes too.
        if isinstance(arg, TypeVarType):
            return arg
        if isinstance(arg, TypeVarTupleType):
            return UnpackType(arg)
        if isinstance(arg, UnpackType):
            if arg.from_star_syntax:
                raise _FamFail(
                    f"{head} star syntax (*Ts) is not supported;"
                    " write Unpack[Ts] instead"
                )
            return arg
        if isinstance(arg, LiteralType):
            if (
                isinstance(arg.value, int)
                and not isinstance(arg.value, bool)
            ):
                return arg.value
            raise _FamFail(f"{head} dims must be integer literals")
        if isinstance(arg, AnyType):
            return arg
        raise _FamFail(
            f"{head} dims must be TypeVars, integers, Unpack[Ts],"
            ' Literal[...], or a string spec like "batch features"'
        )

    def _fam_string(
        self,
        ctx: AnalyzeTypeContext,
        head: str,
        text: str,
        node: RawExpressionType | UnboundType,
    ) -> list[int | Type]:
        """A jaxtyping-style spec: space-separated dim names, one per dim.

        Each word resolves exactly like a bare name (same declared-TypeVar
        lookup, same case rule), so `"dim2"` in two parameters shares the
        dim with a bare `dim2` elsewhere in the signature. Words that only
        appear inside strings are bound into scope first (see
        `_fam_bind_words`); undeclared words in a multi-word spec mint
        implicit dims instead of erroring, so those need no declarations.
        A lone single-word spec is a forward reference, which mypy resolves
        before any hook runs, so it still needs a declared name. The rest
        of the lookup path is `_fam_bare`.

        A word may also be a linear expression over names (`"N+1"`,
        `"N-1"`, `"2*N"`, `"N+M"`), spelled without spaces so it stays one
        dim (issue #32: offsets/lengths duality). Each name inside resolves
        like a bare name (declared or implicit); the whole word denotes the
        linear form, so `I64["N+1"]` unifies with the `N+1` that
        `concatenate([[0], cumsum(x)])` infers, and `np.diff` of it is `N`.
        Only linear forms are supported; anything else fails loudly.
        """
        words = text.split()
        if not words:
            raise _FamFail(
                f'{head} string dims must name at least one dim, e.g. "batch features"'
            )
        bare = [w for w in words if w.isidentifier() and not keyword.iskeyword(w)]
        minted = self._fam_bind_words(ctx, bare, node)
        dims: list[int | Type] = []
        for word in words:
            if word.isidentifier() and not keyword.iskeyword(word):
                if word in minted:
                    dims.append(minted[word])
                else:
                    dims.append(
                        self._fam_bare(
                            ctx,
                            head,
                            UnboundType(word, [], line=node.line, column=node.column),
                        )
                    )
            else:
                dims.append(self._fam_expr(ctx, head, word, node))
        return dims

    def _fam_expr(
        self,
        ctx: AnalyzeTypeContext,
        head: str,
        word: str,
        node: RawExpressionType | UnboundType,
    ) -> int | Type:
        """A linear expression dim (`N+1`, `N-1`, `2*N`, `N+M`) for issue #32.

        Parses the word with `ast` (names, int constants, `+`/`-`/`*`
        only), resolves each name like a bare dim (declared TypeVar or
        implicit mint), and returns the dim for the canonical linear form:
        a plain int for const-only, otherwise a scope-bound variable whose
        registry form is the linear combination. Non-linear or unparsable
        words fail loudly so unsupported shapes never go quietly Any.
        """
        try:
            tree = ast.parse(word, mode="eval")
        except SyntaxError:
            raise _FamFail(
                f'{head} string dim "{word}" is not a name or linear expression;'
                ' write space-separated names like "batch features"'
                ' or linear forms like "N+1"'
            ) from None
        names: list[str] = []
        seen: set[str] = set()

        def collect(e: ast.expr) -> None:
            if isinstance(e, ast.Name):
                if e.id not in seen:
                    seen.add(e.id)
                    names.append(e.id)
            elif isinstance(e, ast.Constant):
                return
            elif isinstance(e, ast.BinOp) and isinstance(
                e.op, (ast.Add, ast.Sub, ast.Mult)
            ):
                collect(e.left)
                collect(e.right)
            elif isinstance(e, ast.UnaryOp) and isinstance(e.op, (ast.UAdd, ast.USub)):
                collect(e.operand)
            else:
                raise _FamFail(
                    f'{head} string dim "{word}" is not linear;'
                    ' only names, integers, "+", "-", "*" (e.g. "N+1", "2*N")'
                )

        try:
            collect(tree.body)
        except _FamFail:
            raise
        for name in names:
            if not name.isidentifier() or keyword.iskeyword(name):
                raise _FamFail(
                    f'{head} string dim "{word}" is not linear;'
                    ' only names, integers, "+", "-", "*" (e.g. "N+1", "2*N")'
                )
        # Bind constituent names exactly like bare dims (declared lookup or
        # implicit mint), then resolve each to its TypeVar or int value.
        minted = self._fam_bind_words(ctx, names, node)
        resolved: dict[str, int | Type] = {}
        for name in names:
            if name in minted:
                resolved[name] = minted[name]
            else:
                resolved[name] = self._fam_bare(
                    ctx, head, UnboundType(name, [], line=node.line, column=node.column)
                )
        coeffs: dict[str, int] = {}
        const = 0

        def add_scaled(fullname: str, c: int) -> None:
            if c:
                coeffs[fullname] = coeffs.get(fullname, 0) + c

        def walk(e: ast.expr) -> tuple[dict[str, int], int]:
            if isinstance(e, ast.Name):
                v = resolved[e.id]
                if isinstance(v, int):
                    return {}, v
                if isinstance(v, TypeVarType):
                    return {v.fullname: 1}, 0
                raise _FamFail(
                    f'{head} dim "{e.id}" must be a TypeVar, an integer,'
                    " Unpack[Ts], or Literal[...]"
                )
            if isinstance(e, ast.Constant):
                if isinstance(e.value, bool) or not isinstance(e.value, int):
                    raise _FamFail(
                        f'{head} string dim "{word}" is not linear;'
                        ' only names, integers, "+", "-", "*" (e.g. "N+1", "2*N")'
                    )
                return {}, e.value
            if isinstance(e, ast.UnaryOp) and isinstance(e.op, (ast.UAdd, ast.USub)):
                sub_c, sub_k = walk(e.operand)
                if isinstance(e.op, ast.USub):
                    return ({k: -v for k, v in sub_c.items()}, -sub_k)
                return sub_c, sub_k
            if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Add, ast.Sub)):
                lc, lk = walk(e.left)
                rc, rk = walk(e.right)
                if isinstance(e.op, ast.Sub):
                    rc = {k: -v for k, v in rc.items()}
                    rk = -rk
                merged = dict(lc)
                for k, v in rc.items():
                    merged[k] = merged.get(k, 0) + v
                return ({k: v for k, v in merged.items() if v}, lk + rk)
            if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Mult):
                lc, lk = walk(e.left)
                rc, rk = walk(e.right)
                if lc and rc:
                    raise _FamFail(
                        f'{head} string dim "{word}" is not linear'
                        ' (name * name); use "+", "-", or a single "*"'
                        ' with a constant (e.g. "N+1", "2*N")'
                    )
                if lc:
                    if rk == 0:
                        return {}, 0
                    return ({k: v * rk for k, v in lc.items()}, lk * rk)
                if rc:
                    if lk == 0:
                        return {}, 0
                    return ({k: v * lk for k, v in rc.items()}, rk * lk)
                return {}, lk * rk
            raise _FamFail(
                f'{head} string dim "{word}" is not linear;'
                ' only names, integers, "+", "-", "*" (e.g. "N+1", "2*N")'
            )

        part_c, part_k = walk(tree.body)
        for fullname, c in part_c.items():
            add_scaled(fullname, c)
        const = part_k
        coeffs = {k: v for k, v in coeffs.items() if v}
        if not coeffs:
            return const
        form = _LinForm(tuple(sorted(coeffs.items())), const)
        # Form-keyed identity shared with inferred expressions (_expr_var):
        # the same canonical form is the same variable everywhere, so an
        # inferred `N+1` is identical to an annotated `I64["N+1"]` and
        # native subtype checking accepts it. The fullname embeds the
        # canonical form (with fullnames), so different expressions never
        # collide; `?U`-containing forms stay per-site via the unknown key.
        fullname = f"__numty_expr__.{form!r}"
        analyzer = cast(TypeAnalyser, ctx.api)
        scope = analyzer.tvar_scope
        existing = scope.get_binding(fullname)
        if existing is not None:
            return existing
        vid = _expr_ids.get(fullname)
        if vid is None:
            vid = TypeVarId.new(0)
            _expr_ids[fullname] = vid
        _expr_forms[fullname] = form
        display = _expr_str(form)
        upper = ctx.api.named_type("builtins.int", [], node.line, node.column)
        obj = TypeVarType(
            name=display,
            fullname=fullname,
            id=vid,
            values=[],
            upper_bound=upper,
            default=AnyType(TypeOfAny.from_omitted_generics),
            line=node.line,
            column=node.column,
        )
        scope.bind_existing(obj)
        self._fam_late_bound.add(fullname)
        # Remember constituent vars so later simplifications (e.g.
        # diff(N+1) -> N) return the scope's own object, not a stale one.
        for name in names:
            v = resolved[name]
            if isinstance(v, TypeVarType):
                _seen_vars.setdefault(v.fullname, v)
        # Per-scope constituents for #32 finish: diff(M+1)->M looks up M
        # in the defining scope via the expr's fullname, dodging the
        # global _seen_vars first-wins leak (#43).
        try:
            _expr_constituents[fullname] = {
                v.fullname: v
                for v in resolved.values()
                if isinstance(v, TypeVarType)
            }
        except Exception:
            pass
        return obj

    def _fam_bind_words(
        self,
        ctx: AnalyzeTypeContext,
        words: list[str],
        node: RawExpressionType | UnboundType,
    ) -> dict[str, TypeVarLikeType]:
        """Bind string-spec words the way mypy's pre-pass would have.

        mypy binds a function's type variables by scanning the *written*
        signature (FindTypeVarVisitor) before analyzing it, so a name that
        only appears inside a string spec is still unbound when this hook
        runs and `analyze_type` would reject it as unbound. Bind each such
        word here, replicating the pre-pass rule: a TypeVar-like name with
        no binding yet (and no scoping prohibition) gets `bind_new` in the
        live scope. A word with no declaration at all mints an implicit
        dim (see `_fam_ensure_implicit`) instead of erroring; everything
        else (Final ints, non-tvar names, names bound already or further
        out) is left for the normal path, which reports them exactly as it
        would a bare name.

        Returns the implicit dims minted here, keyed by word:
        `analyze_type` can never find them (no symbol exists to look up),
        so callers must use the returned objects directly. Declared words
        resolve through `_fam_bare` as before.

        This reaches through `ctx.api` to the running TypeAnalyser
        (that is what mypy passes as the hook api). Pinned to
        mypy==2.4.1; the `f64_str_*` tests pin the behavior.
        """
        analyzer = cast(TypeAnalyser, ctx.api)
        scope = analyzer.tvar_scope
        minted: dict[str, TypeVarLikeType] = {}
        for word in words:
            # Probe, don't resolve: a miss means "undeclared, maybe
            # mint" (or "Final etc., normal path"), never an error here.
            sym = analyzer.lookup_qualified(word, node, suppress_errors=True)
            if sym is not None and isinstance(sym.node, TypeVarLikeExpr):
                if scope.get_binding(sym) is not None:
                    continue
                if sym.fullname and not scope.allow_binding(sym.fullname):
                    continue
                scope.bind_new(word, sym.node, analyzer.fail_func, node)
                if sym.fullname:
                    self._fam_late_bound.add(sym.fullname)
                continue
            if sym is not None:
                continue
            minted[word] = self._fam_ensure_implicit(
                ctx, analyzer, scope, word, node
            )
        return minted

    def _fam_ensure_implicit(
        self,
        ctx: AnalyzeTypeContext,
        analyzer: TypeAnalyser,
        scope: TypeVarLikeScope,
        word: str,
        node: RawExpressionType | UnboundType,
    ) -> TypeVarLikeType:
        """Mint (or reuse) an implicit dim for an undeclared string word.

        Only reached for words inside multi-word specs: a lone
        single-word spec is a forward reference, which mypy resolves
        (or rejects) before any hook runs. The first occurrence mints a
        bound-int TypeVar via the real `bind_new` (so ids, namespaces, and
        the resulting TypeVarType match declared dims exactly); later
        occurrences in the same scope chain reuse it, so one word is one
        dim per signature. The synthetic fullname carries a marker that
        can never collide with a real symbol. Recorded in
        `_fam_late_bound` like any other late binding, so calls check and
        infer normally.
        """
        fullname = f"__numty_implicit__.{word}"
        existing = scope.get_binding(fullname)
        if existing is not None:
            return existing
        upper = ctx.api.named_type("builtins.int", [], node.line, node.column)
        expr = TypeVarExpr(
            word,
            fullname,
            [],
            upper,
            AnyType(TypeOfAny.from_omitted_generics),
            line=node.line,
        )
        bound = scope.bind_new(word, expr, analyzer.fail_func, node)
        self._fam_late_bound.add(fullname)
        return bound

    def _fam_bare(
        self, ctx: AnalyzeTypeContext, head: str, arg: UnboundType
    ) -> int | Type:
        """A bare name: it must resolve (a TypeVar, TypeVarTuple, Final).

        mypy's own function-typevar visitor resolves every name in a
        signature before the hook runs, so an unresolvable name is
        already an error by the time we see it: analyze, reuse what it
        denotes, fail clearly on anything else.
        """
        if arg.name in ("True", "False", "None", "_", "..."):
            raise _FamFail(
                f'{head} dim "{arg.name}" is not a TypeVar or integer;'
                " declare a TypeVar for it, or use Unpack[Ts]"
            )
        analyzed = ctx.api.analyze_type(arg)
        if isinstance(analyzed, TypeVarType):
            upper = get_proper_type(analyzed.upper_bound)
            if (
                not analyzed.values
                and isinstance(upper, Instance)
                and upper.type.fullname == "builtins.object"
            ):
                # numpy bounds the shape at tuple[int, ...]: an unbounded
                # TypeVar can never fit, and mypy's own error blames numpy.
                raise _FamFail(
                    f'{head} dim "{arg.name}" is a TypeVar without an int bound;'
                    f' declare it as TypeVar("{arg.name}", bound=int)'
                )
            return analyzed
        if isinstance(analyzed, TypeVarTupleType):
            return UnpackType(analyzed)
        if (
            isinstance(analyzed, Instance)
            and analyzed.type.fullname == INT_FULLNAME
            and not analyzed.args
        ):
            # A runtime size nothing can name (F32[int, int, Unpack[Ts]]
            # for a gather over runtime counts): the same plain int dim a
            # builder fed a runtime value produces, never solved or unified.
            return analyzed
        if isinstance(analyzed, AnyType):
            if analyzed.type_of_any == TypeOfAny.from_error:
                raise _FamBail
            return analyzed
        if isinstance(analyzed, UnboundType):
            analyzer = cast(TypeAnalyser, ctx.api)
            sym = analyzer.lookup_qualified(arg.name, arg, suppress_errors=True)
            if (
                sym is not None
                and isinstance(sym.node, TypeVarLikeExpr)
                and not analyzer.allow_unbound_tvars
            ):
                # A TypeVar used outside any scope that binds it: mypy has
                # already said 'Type variable "X" is unbound', with hints.
                raise _FamBail
        value = _dim_value(get_proper_type(analyzed))
        if isinstance(value, int):
            return value  # Final / literal-valued name
        raise _FamFail(
            f'{head} dim "{arg.name}" must be a TypeVar, an integer,'
            " Unpack[Ts], or Literal[...]"
        )

    def _fam_applied(
        self, ctx: AnalyzeTypeContext, head: str, arg: UnboundType
    ) -> int | Type:
        """A parameterized dim: `Literal[3]`, `Unpack[Ts]`, never `tuple`."""
        short = arg.name.rpartition(".")[2]
        if short == "tuple":
            raise _FamFail(
                f"{head} takes flat dims: write {head}[N, M],"
                f" not {head}[tuple[N, M]]"
            )
        if short == "Unpack":
            if len(arg.args) != 1:
                raise _FamFail(f"{head} Unpack takes exactly one argument")
            # Neither the bare name (illegal outside Unpack) nor the
            # whole Unpack[...] (illegal outside a variadic position)
            # analyzes alone, so analyze tuple[Unpack[Ts]] and unwrap.
            wrapped = UnboundType(
                "tuple", [arg], line=arg.line, column=arg.column
            )
            analyzed = ctx.api.analyze_type(wrapped)
            proper = get_proper_type(analyzed)
            if isinstance(proper, TupleType) and len(proper.items) == 1:
                item = get_proper_type(proper.items[0])
                if isinstance(item, UnpackType):
                    return item
            if isinstance(proper, AnyType):
                if proper.type_of_any == TypeOfAny.from_error:
                    raise _FamBail
                return proper
            raise _FamFail(
                f"{head} Unpack takes a TypeVarTuple, e.g. Unpack[Ts]"
            )
        analyzed = ctx.api.analyze_type(arg)
        proper = get_proper_type(analyzed)
        if (
            isinstance(proper, LiteralType)
            and isinstance(proper.value, int)
            and not isinstance(proper.value, bool)
        ):
            return proper.value
        if isinstance(proper, TypeVarTupleType):
            return UnpackType(proper)
        if isinstance(proper, AnyType):
            if proper.type_of_any == TypeOfAny.from_error:
                raise _FamBail
            return proper
        raise _FamFail(
            f"{head} dims must be TypeVars, integers,"
            " Unpack[Ts], or Literal[...]"
        )

    # Every numpy hook goes through _guard_mixed (see there); the solver
    # dispatch itself lives in the underscored getters below.
    def get_function_hook(self, fullname: str) -> Callable[[FunctionContext], Type] | None:
        return _guard_mixed(fullname, self._function_hook(fullname))

    def get_attribute_hook(
        self, fullname: str
    ) -> Callable[[AttributeContext], Type] | None:
        return _guard_mixed(fullname, self._attribute_hook(fullname))

    def get_method_hook(self, fullname: str) -> Callable[[MethodContext], Type] | None:
        return _guard_mixed(fullname, self._method_hook(fullname))

    def _function_hook(self, fullname: str) -> Callable[[FunctionContext], Type] | None:
        if fullname == "builtins.len":
            return len_function_hook
        # Raw np.matmul solves through the ufunc __call__ hook; the
        # numty.matmul wrapper below stays as a convenience alias.
        if fullname == "numty.matmul":
            return matmul_hook
        # Exact canonical names only (see _numpy_function_hooks): a name
        # tail would also catch np.ma / np.emath / numpy.matlib twins.
        hook = _NUMPY_FUNCTION_HOOKS.get(fullname)
        if hook is not None:
            return hook
        short = fullname.rsplit(".", 1)[-1]
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

    def _attribute_hook(
        self, fullname: str
    ) -> Callable[[AttributeContext], Type] | None:
        if fullname == "numpy.ndarray.shape":
            return shape_attribute_hook
        # T lives on the shared _ArrayOrScalarCommon base in the stubs.
        if fullname == "numpy._ArrayOrScalarCommon.T":
            return t_attribute_hook
        if fullname in {
            "numpy.ndarray.mT",
            "numpy._ArrayOrScalarCommon.mT",
        }:
            return mt_attribute_hook
        return None

    def _method_hook(self, fullname: str) -> Callable[[MethodContext], Type] | None:
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
        if fullname.startswith("numpy.random.mtrand.RandomState.") and short in {
            "rand",
            "randn",
        }:
            return legacy_rand_hook
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
        if fullname.startswith("numpy.ndarray.") and short in {"cumsum", "cumprod"}:
            return cumsum_method_hook
        if fullname.startswith("numpy.ndarray.") and short == "take":
            return take_method_hook
        if fullname.startswith("numpy.ndarray.") and short == "swapaxes":
            return swapaxes_method_hook
        if fullname.startswith("numpy.ndarray.") and short == "dot":
            return dot_method_hook
        if fullname.startswith("numpy.ndarray.") and short == "repeat":
            return repeat_method_hook
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
    """Provably-equal dims: same ints, same TypeVar, or same canonical form.

    Expression variables resolve through the registry, so N*2 built at two
    sites compares equal; anything unresolvable falls back to False.
    """
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, TypeVarType) and isinstance(b, TypeVarType):
        if a.fullname == b.fullname:
            return True
    fa = _resolve_dim(a)
    fb = _resolve_dim(b)
    if fa is not None and fb is not None:
        return fa == fb
    return False


def _dim_str(d: int | Type) -> str:
    if isinstance(d, int):
        return str(d)
    if isinstance(d, UnpackType):
        inner = d.type
        name = getattr(inner, "name", None)
        return f"*{name}" if isinstance(name, str) else "*Ts"
    if isinstance(d, TypeVarType):
        form = _expr_forms.get(d.fullname)
        if form is not None:
            return _expr_str(form)
    name = getattr(d, "name", None)
    return str(name) if isinstance(name, str) else "?"


class _LinForm(NamedTuple):
    """Canonical linear dim: sum(coeff * var) + const, coeffs sorted by var."""

    coeffs: tuple[tuple[str, int], ...]
    const: int


class _DivForm(NamedTuple):
    """Floored division of a linear dim: inner // div + plus."""

    inner: _LinForm
    div: int
    plus: int


class _MonoForm(NamedTuple):
    """Opaque product term: coeff * prod(vars), vars sorted (repeats kept).

    Stepping stone for issue #40 (no Z3): single monomials only, no sums
    of products. Always holds >= 2 vars; single-var/const cases stay
    _LinForm so equality and existing paths are untouched. Zero normalizes
    to coeff 0 with no vars.
    """

    coeff: int
    vars: tuple[str, ...]


_expr_forms: dict[str, _LinForm | _DivForm | _MonoForm] = {}
_expr_ids: dict[str, TypeVarId] = {}
_seen_vars: dict[str, TypeVarType] = {}
# Per-expr constituent objects by scope (issue #32 finish): expr fullname
# -> {constituent fullname -> TypeVarType object from the defining scope}.
# Lets diff(M+1) -> M return the *current* function's M instead of the
# first-seen global in _seen_vars (which causes N@first vs N@second).
_expr_constituents: dict[str, dict[str, TypeVarType]] = {}


def _current_scope_vars(
    ctx: FunctionContext | MethodContext,
) -> dict[str, TypeVarType]:
    """TypeVars visible in the current call's actual types, by fullname.

    Walks all positional actual types (and self for method hooks) so
    materialization prefers the caller's own objects over the process-wide
    _seen_vars fallback (issue #43 leak scoped to #32's diff path).
    """
    found: dict[str, TypeVarType] = {}
    try:
        nested: list[Type] = []
        for group in getattr(ctx, "arg_types", []):
            # mypy passes groups of overload-candidate types; flatten.
            if isinstance(group, list):
                nested.extend(group)
            else:
                nested.append(group)
        self_tp = getattr(ctx, "type", None)
        if self_tp is not None:
            nested.append(self_tp)
        for tp in nested:
            try:
                for tv in _iter_tvarlikes(tp):
                    if isinstance(tv, TypeVarType):
                        found.setdefault(tv.fullname, tv)
            except Exception:
                continue
    except Exception:
        pass
    return found


def _short_var(fullname: str) -> str:
    """Display name for a var: ?U name when unknown, else the short name."""
    return _unknown_names.get(fullname) or fullname.rsplit(".", 1)[-1]


def _mono_str(form: _MonoForm) -> str:
    """Readable product: H*D, 2*M*N, 6, 0."""
    if form.coeff == 0:
        return "0"
    if not form.vars:
        return str(form.coeff)
    body = "*".join(_short_var(v) for v in form.vars)
    if form.coeff == 1:
        return body
    if form.coeff == -1:
        return f"-{body}"
    return f"{form.coeff}*{body}"


def _resolve_dim(d: int | Type) -> _LinForm | _DivForm | _MonoForm | None:
    """Canonical form of a dim: ints and TypeVars (expressions included).

    Plain TypeVars read as single-term linears, so N and N*1 compare
    equal; anything else is unresolvable. Seen variables are remembered
    by fullname so materialization can return the real object.
    """
    if isinstance(d, int):
        return _LinForm((), d)
    if isinstance(d, TypeVarType):
        form = _expr_forms.get(d.fullname)
        if form is not None:
            return form
        _seen_vars.setdefault(d.fullname, d)
        return _LinForm(((d.fullname, 1),), 0)
    return None


def _expr_str(form: _LinForm | _DivForm | _MonoForm) -> str:
    """Readable dim formula: 2*N+3, N-1, (N+1)//2, N//2+1, H*D."""
    if isinstance(form, _MonoForm):
        return _mono_str(form)
    if isinstance(form, _DivForm):
        inner = _expr_str(form.inner)
        compound = len(form.inner.coeffs) + (form.inner.const != 0) > 1
        out = f"({inner})" if compound else inner
        out += f"//{form.div}"
        if form.plus:
            out += f"+{form.plus}"
        return out
    parts = []
    for fullname, coeff in form.coeffs:
        # Unknowns carry a display name (?U@line); their fullname is an
        # internal site key that must never reach a message.
        short = _unknown_names.get(fullname) or fullname.rsplit(".", 1)[-1]
        if coeff == 1:
            parts.append(short)
        elif coeff == -1:
            parts.append(f"-{short}")
        else:
            parts.append(f"{coeff}*{short}")
    if form.const or not parts:
        parts.append(str(form.const))
    return "+".join(parts).replace("+-", "-")


def _expr_var(
    ctx: FunctionContext | MethodContext, op: str, index: int, form: _LinForm | _DivForm | _MonoForm
) -> TypeVarType:
    """Mint (or reuse) the variable denoting a dim expression (issue #32).

    Form-keyed (not site-keyed): the same canonical form reuses one
    variable id everywhere, so an inferred `N+1` is the same variable as
    an annotated `I64["N+1"]` with the same form and native subtype
    checking accepts it. Different forms never collide. Re-checks and
    overload candidates with the same form still mint identically, and
    `?U`-containing forms stay per-site via the unknown's own site key.
    The registry maps the fullname back to the canonical form.
    """
    # `op`/`index` only disambiguate the call site for debugging; identity
    # comes from the form itself.
    _ = op, index
    tree = getattr(ctx.api, "tree", None)
    _ = getattr(tree, "fullname", "") or ""
    line = ctx.context.line
    column = ctx.context.column
    fullname = f"__numty_expr__.{form!r}"
    vid = _expr_ids.get(fullname)
    if vid is None:
        vid = TypeVarId.new(0)
        _expr_ids[fullname] = vid
    _expr_forms[fullname] = form
    upper = ctx.api.named_generic_type("builtins.int", [])
    obj = TypeVarType(
        name=_expr_str(form),
        fullname=fullname,
        id=vid,
        values=[],
        upper_bound=upper,
        default=AnyType(TypeOfAny.from_omitted_generics),
        line=line,
        column=column,
    )
    # Remember current-scope constituents so a later diff of this inferred
    # expr finds its own M (concat N -> N+1 -> diff -> N roundtrip).
    if isinstance(form, _LinForm) and form.coeffs:
        try:
            current = _current_scope_vars(ctx)
            consts = {
                fullname_c: current[fullname_c]
                for fullname_c, _ in form.coeffs
                if fullname_c in current
            }
            if consts:
                _expr_constituents.setdefault(fullname, {}).update(consts)
        except Exception:
            pass
    return obj


def _lin_add(a: _LinForm, b: _LinForm) -> _LinForm:
    merged: dict[str, int] = {}
    for name, coeff in a.coeffs + b.coeffs:
        merged[name] = merged.get(name, 0) + coeff
    return _LinForm(
        tuple(sorted((n, c) for n, c in merged.items() if c != 0)),
        a.const + b.const,
    )


def _lin_scale(form: _LinForm, c: int) -> _LinForm:
    if c == 0:
        return _LinForm((), 0)
    return _LinForm(tuple((n, k * c) for n, k in form.coeffs), form.const * c)


def _mono_normalize(coeff: int, vars: tuple[str, ...]) -> _MonoForm:
    """Canonical monomial: sorted vars, zero collapsed."""
    if coeff == 0:
        return _MonoForm(0, ())
    return _MonoForm(coeff, tuple(sorted(vars)))


def _form_to_coeff_vars(
    form: _LinForm | _MonoForm,
) -> tuple[int, tuple[str, ...]] | None:
    """Pure-product view: (coeff, vars) or None for sums.

    Linear single-terms (c, c*N) and monos map directly; multi-term
    linears (N+M, N+2 with a var) are polynomial sums — out of scope,
    bail with None.
    """
    if isinstance(form, _MonoForm):
        return form.coeff, form.vars
    if not form.coeffs:
        return form.const, ()
    if len(form.coeffs) == 1 and form.const == 0:
        name, c = form.coeffs[0]
        return c, (name,)
    return None


def _coeff_vars_to_form(coeff: int, vars: tuple[str, ...]) -> _LinForm | _MonoForm:
    """Form for a pure product: linear for <= 1 var, else monomial."""
    if coeff == 0:
        return _LinForm((), 0)
    if len(vars) <= 1:
        if not vars:
            return _LinForm((), coeff)
        return _LinForm(((vars[0], coeff),), 0)
    return _mono_normalize(coeff, vars)


def _form_mul(
    a: _LinForm | _MonoForm, b: _LinForm | _MonoForm
) -> _LinForm | _MonoForm | None:
    """Product of pure-product forms, else None (sums/Div out of scope)."""
    pa = _form_to_coeff_vars(a)
    pb = _form_to_coeff_vars(b)
    if pa is None or pb is None:
        return None
    ca, va = pa
    cb, vb = pb
    return _coeff_vars_to_form(ca * cb, tuple(sorted(va + vb)))


def _canon_div(
    numer: _LinForm | _MonoForm, denom: _LinForm | _MonoForm
) -> _LinForm | _MonoForm | None:
    """Exact quotient numer/denom for pure products, else None.

    Covers int division (N*2/2) and variable cancellation (N*K/N,
    B*C*H*W/B). Sums and Divs bail.
    """
    pn = _form_to_coeff_vars(numer)
    pd = _form_to_coeff_vars(denom)
    if pn is None or pd is None:
        return None
    cn, vn = pn
    cd, vd = pd
    if cd == 0:
        return None
    if cn % cd != 0:
        return None
    # Multiset subtraction of divisor vars from numerator vars.
    rest = list(vn)
    for v in vd:
        try:
            rest.remove(v)
        except ValueError:
            return None
    return _coeff_vars_to_form(cn // cd, tuple(sorted(rest)))


def _materialize_dim(
    ctx: FunctionContext | MethodContext,
    op: str,
    index: int,
    form: _LinForm | _DivForm | _MonoForm,
    prefer: int | Type | None = None,
) -> int | Type:
    """A dim for a canonical form: int, the real variable, or a fresh var.

    A form that is just one known variable returns that variable's own
    object (so N*2/2 checks as F64[N], not as a lookalike); anything else
    mints a formula-named variable.
    """
    if isinstance(form, _LinForm) and not form.coeffs:
        return form.const
    if prefer is not None and _resolve_dim(prefer) == form:
        return prefer
    if (
        isinstance(form, _LinForm)
        and len(form.coeffs) == 1
        and form.coeffs[0][1] == 1
        and form.const == 0
    ):
        want = form.coeffs[0][0]
        # 1) diff(M+1)->M: the input expr knows its own scope's M.
        if isinstance(prefer, TypeVarType):
            consts = _expr_constituents.get(prefer.fullname)
            if consts is not None and want in consts:
                return consts[want]
        # 2) Current call's own object (reshape/split second-function
        # leak): prefer it over the process-wide first-wins registry.
        try:
            current = _current_scope_vars(ctx)
            if want in current:
                return current[want]
        except Exception:
            pass
        seen = _seen_vars.get(want)
        if seen is not None:
            return seen
    return _expr_var(ctx, op, index, form)


def _dim_add_const(
    ctx: FunctionContext | MethodContext, op: str, index: int, a: int | Type, b: int
) -> int | Type:
    """a + b for a constant b: exact linear form, else a fresh unknown."""
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
    if all(c % div == 0 for _, c in fa.coeffs) and fa.const % div == 0:
        base = _LinForm(tuple((n, c // div) for n, c in fa.coeffs), fa.const // div)
        if plus:
            base = _lin_add(base, _LinForm((), plus))
        return _materialize_dim(ctx, op, index, base, prefer=a)
    return _materialize_dim(ctx, op, index, _DivForm(fa, div, plus))


def _mul_dims(
    ctx: FunctionContext | MethodContext, op: str, index: int, dims: list[int | Type]
) -> int | Type:
    """Product of dims: exact linear/monomial form, else a fresh unknown."""
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


def _linear_product(dims: list[int | Type]) -> _LinForm | _MonoForm | None:
    """Canonical total product: linear or single monomial, else None."""
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


def _iter_tvarlikes(tp: Type) -> Iterator[TypeVarLikeType]:
    """Every TypeVar-like nested in an analyzed type (any container)."""
    tp = get_proper_type(tp)
    if isinstance(tp, TypeVarLikeType):
        yield tp
    elif isinstance(tp, Instance):
        for a in tp.args:
            yield from _iter_tvarlikes(a)
    elif isinstance(tp, TupleType):
        for item in tp.items:
            yield from _iter_tvarlikes(item)
    elif isinstance(tp, UnionType):
        for item in tp.relevant_items():
            yield from _iter_tvarlikes(item)
    elif isinstance(tp, UnpackType):
        yield from _iter_tvarlikes(tp.type)


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


def _callsite_expr_bindings(
    sig: CallableType,
    formals: list[tuple[int, list[tuple[int, TypeVarType]]]],
    ctx: FunctionSigContext,
) -> dict[str, int | Type]:
    """Callee var fullname -> actual dim for expression returns (issue #32).

    Plain formal dims bind directly to the actual dim object (preserving the
    caller's TypeVar id, which also dodges the #43 stale-object leak).
    Expr formal dims of the form M+const solve M from the actual's linear
    form when it is const or a single var with matching const (M+1 vs K+1,
    M+1 vs 6); anything fancier bails for that dim.
    """
    bindings: dict[str, int | Type] = {}
    for i, positions in formals:
        if i >= len(ctx.args):
            continue
        actuals = ctx.args[i]
        if len(actuals) != 1:
            continue
        try:
            actual_tp = ctx.api.get_expression_type(actuals[0])
        except Exception:
            continue
        parts = _ndarray_parts(actual_tp)
        if parts is None:
            continue
        adims, _ = parts
        for dim, tvar in positions:
            if dim >= len(adims):
                continue
            adim = adims[dim]
            try:
                form_f = _expr_forms.get(tvar.fullname)
                if form_f is None:
                    # Plain formal: direct bind (first wins).
                    if tvar.fullname not in bindings:
                        bindings[tvar.fullname] = adim
                    continue
                # Expr formal: only single-var linear M+const for now.
                if not isinstance(form_f, _LinForm):
                    continue
                if len(form_f.coeffs) != 1 or form_f.coeffs[0][1] != 1:
                    continue
                m_full, const_f = form_f.coeffs[0][0], form_f.const
                fa = _resolve_dim(adim)
                if fa is None or not isinstance(fa, _LinForm):
                    continue
                if not fa.coeffs:
                    # Actual const: M = const_a - const_f.
                    if tvar.fullname not in bindings and m_full not in bindings:
                        bindings[m_full] = fa.const - const_f
                    continue
                if len(fa.coeffs) == 1 and fa.coeffs[0][1] == 1:
                    a_full, const_a = fa.coeffs[0][0], fa.const
                    if const_a != const_f:
                        continue  # M = A + diff needs new expr; skip.
                    if m_full in bindings:
                        continue
                    if isinstance(adim, TypeVarType) and not (
                        adim.fullname.startswith("__numty_expr__.")
                    ):
                        # Actual plain A: M = A (caller's object).
                        bindings[m_full] = adim
                    else:
                        # Actual expr K+1 with same const: M = K via
                        # the actual's own constituents (caller's scope).
                        consts = None
                        if isinstance(adim, TypeVarType):
                            consts = _expr_constituents.get(adim.fullname)
                        if consts is not None and a_full in consts:
                            bindings[m_full] = consts[a_full]
                        else:
                            found = _seen_vars.get(a_full)
                            if found is not None:
                                bindings[m_full] = found
            except Exception:
                continue
    return bindings


def _substitute_return_exprs(
    sig: CallableType,
    bindings: dict[str, int | Type],
    ctx: FunctionSigContext,
) -> Type | None:
    """Return type with callee expr vars substituted by call bindings.

    Returns the new return Type, or None when nothing changed / unsupported.
    Plain vars use the actual object directly (caller's id); new expr forms
    reuse the form-keyed global id so an inferred N+1 is the same variable
    as an annotated I64["N+1"].
    """
    if not bindings:
        return None
    try:
        ret_parts = _ndarray_parts(sig.ret_type)
    except Exception:
        return None
    if ret_parts is None:
        return None
    rdims, _dtype = ret_parts
    new_dims: list[int | Type] = []
    changed = False
    # For building new expr vars we need constituent objects by fullname.
    obj_by_fullname: dict[str, TypeVarType] = {}
    for v in bindings.values():
        if isinstance(v, TypeVarType):
            obj_by_fullname.setdefault(v.fullname, v)
    for consts in _expr_constituents.values():
        for k, v in consts.items():
            obj_by_fullname.setdefault(k, v)
    for k, v in _seen_vars.items():
        obj_by_fullname.setdefault(k, v)
    for d in rdims:
        try:
            if isinstance(d, int):
                new_dims.append(d)
                continue
            if not isinstance(d, TypeVarType):
                return None
            if d.fullname not in _expr_forms:
                # Plain return var: direct substitution preserves caller id.
                if d.fullname in bindings:
                    b = bindings[d.fullname]
                    new_dims.append(b)
                    changed = True
                else:
                    new_dims.append(d)
                continue
            # Expr return var.
            form_r = _expr_forms.get(d.fullname)
            if not isinstance(form_r, _LinForm):
                return None
            merged: dict[str, int] = {}
            const = form_r.const
            substituted = False
            abort = False
            for cfull, coeff in form_r.coeffs:
                if cfull in bindings:
                    b = bindings[cfull]
                    if isinstance(b, int):
                        const += coeff * b
                        substituted = True
                    else:
                        fb = _resolve_dim(b)
                        if not isinstance(fb, _LinForm):
                            abort = True
                            break
                        for bfull, bcoeff in fb.coeffs:
                            merged[bfull] = merged.get(bfull, 0) + coeff * bcoeff
                        const += coeff * fb.const
                        substituted = True
                else:
                    merged[cfull] = merged.get(cfull, 0) + coeff
            if abort:
                return None
            if not substituted:
                new_dims.append(d)
                continue
            merged = {k: v for k, v in merged.items() if v != 0}
            if not merged:
                new_dims.append(const)
                changed = True
                continue
            if len(merged) == 1:
                (only_full, only_coeff) = next(iter(merged.items()))
                if only_coeff == 1 and const == 0:
                    # Single var: reuse the caller's object when known.
                    obj = obj_by_fullname.get(only_full)
                    if obj is not None:
                        new_dims.append(obj)
                    else:
                        new_dims.append(d)
                        continue
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
                const_map: dict[str, TypeVarType] = {}
                for cfull in merged:
                    o = obj_by_fullname.get(cfull)
                    if o is not None:
                        const_map[cfull] = o
                if const_map:
                    _expr_constituents.setdefault(new_fullname, {}).update(
                        const_map
                    )
            except Exception:
                pass
            new_dims.append(new_obj)
            changed = True
        except Exception:
            return None
    if not changed:
        return None
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
                        if isinstance(v, int):
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
            new_shape = TupleType(new_items, shape.fallback, line=shape.line, column=shape.column)
        try:
            return proper.copy_modified(args=[new_shape, proper.args[1]])
        except Exception:
            return Instance(proper.type, [new_shape, proper.args[1]], line=proper.line, column=proper.column)
    except Exception:
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
            ctx.api.note(message, where, code=ARG_TYPE)
    except Exception:
        return


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


def _builder_shape(
    tp: Type, expr: Expression | None, default: Type, shaped_stub: bool = True
) -> list[int | Type] | None:
    """Dims of a builder's shape (zeros/ones/empty/full, *_like shape=, size=).

    Like _shape_from_arg, plus a tuple display that mixes provable dims
    with plain `int` ones: np.zeros((0, t, 23, 2)) with t: int is
    (0, int, 23, 2). Left to the stubs, mypy widens every literal of such
    a display to int once the result is assigned, so one dynamic dim
    erased the rest (#34). A display with no provable dim stays with the
    stubs (None), as does anything else unproven (Any, bool, numpy ints,
    *unpacks). A dynamic position keeps what the stubs' answer (default)
    proved there from the call's type context (2**40 passed where
    Literal[1099511627776] is expected): never less than before.
    shaped_stub says the stubs type this shape argument into the result
    (zeros/ones/empty/full); *_like shape= and size= stubs answer
    tuple[Any, ...], which there is nothing to take from.
    """
    dims = _shape_from_arg(tp, expr)
    if dims is not None:
        return list(dims)
    proper = get_proper_type(tp)
    if (
        not isinstance(expr, TupleExpr)
        or not isinstance(proper, TupleType)
        or len(proper.items) != len(expr.items)
    ):
        return None
    # With shaped_stub, only upgrade a stub answer of the display's own
    # rank. The hook also runs for overloads that end up unused (the 1-d
    # SupportsIndex ones answer tuple[int], the catch-all tuple[Any, ...]),
    # and overriding those can win over the stubs' context-solved answer.
    stub_items: list[Type] | None = None
    proper_default = get_proper_type(default)
    if isinstance(proper_default, Instance) and proper_default.args:
        stub_shape = get_proper_type(proper_default.args[0])
        if isinstance(stub_shape, TupleType) and len(stub_shape.items) == len(expr.items):
            stub_items = stub_shape.items
    if shaped_stub and stub_items is None:
        return None
    out: list[int | Type] = []
    for i, (item, item_tp) in enumerate(zip(expr.items, proper.items)):
        if isinstance(item, StarExpr):
            return None
        value = _fold_int(item)
        dim = value if value is not None else _dim_value(item_tp)
        if dim is None and stub_items is not None:
            dim = _dim_value(stub_items[i])
        if dim is not None:
            out.append(dim)
            continue
        dynamic = get_proper_type(item_tp)
        if not (isinstance(dynamic, Instance) and dynamic.type.fullname == INT_FULLNAME):
            return None
        out.append(dynamic.copy_modified(last_known_value=None))
    if not any(isinstance(d, (int, TypeVarType)) for d in out):
        return None
    return out


def _ndarrays_in(tp: Type, seen: set[TypeAliasType] | None = None) -> Iterator[Instance]:
    """Every ndarray type in tp, through unions, tuples and generics.

    A recursive alias (JSON = dict[str, "JSON"] | list["JSON"] | int) is
    walked once: seen holds the ones already being expanded.
    """
    if isinstance(tp, TypeAliasType) and tp.is_recursive:
        # One set for the whole walk, shared by sibling branches too: an
        # alias met again anywhere (not only below itself) is skipped.
        # That only drops repeats, since an alias always expands to the
        # same ndarrays, and the callers ask whether one exists.
        seen = set() if seen is None else seen
        if tp in seen:
            return
        seen.add(tp)
    proper = get_proper_type(tp)
    if isinstance(proper, (UnionType, TupleType)):
        for item in proper.items:
            yield from _ndarrays_in(item, seen)
    elif isinstance(proper, Instance):
        if proper.type.fullname == NDARRAY_FULLNAME:
            yield proper
        else:
            for arg in proper.args:
                yield from _ndarrays_in(arg, seen)


def _has_literal_dim(arr: Instance) -> bool:
    shape = get_proper_type(arr.args[0]) if arr.args else None
    return isinstance(shape, TupleType) and any(
        isinstance(get_proper_type(i), LiteralType) for i in shape.items
    )


def _has_int_dim(arr: Instance) -> bool:
    shape = get_proper_type(arr.args[0]) if arr.args else None
    return isinstance(shape, TupleType) and any(
        _is_plain_int(get_proper_type(i)) for i in shape.items
    )


def _is_mixed_shape(arr: Instance) -> bool:
    """A shape with both a Literal dim and a plain `int` one (a builder's
    mixed result, see _builder_shape)."""
    shape = get_proper_type(arr.args[0]) if arr.args else None
    if not isinstance(shape, TupleType):
        return False
    items = [get_proper_type(i) for i in shape.items]
    return any(isinstance(i, LiteralType) for i in items) and any(
        isinstance(i, Instance) and i.type.fullname == INT_FULLNAME for i in items
    )


def _is_plain_int(d: int | Type) -> bool:
    """A plain `int` dim (dynamic size) from _mixed_parts."""
    return isinstance(d, Instance) and d.type.fullname == INT_FULLNAME


def _has_mixed_operand(*types: Type) -> bool:
    return any(
        isinstance(p, Instance) and p.type.fullname == NDARRAY_FULLNAME and _is_mixed_shape(p)
        for p in map(get_proper_type, types)
    )


def _mixed_parts(tp: Type, partner: bool = False) -> tuple[list[int | Type], Type] | None:
    """(dims, dtype) of a mixed-shape ndarray, plain `int` dims as types.

    Only for ops that move dims without reading them (transposes) or
    that only pass dims through and treat a plain `int` as matching
    anything (matmul/dot, see _matmul_inner): the rest of the solver
    keeps bailing on int dims via _ndarray_parts. partner also accepts
    a shape with no Literal dim, for the other operand of a mixed one.
    """
    proper = get_proper_type(tp)
    if not (
        isinstance(proper, Instance)
        and proper.type.fullname == NDARRAY_FULLNAME
        and len(proper.args) == 2
        and (partner or _is_mixed_shape(proper))
    ):
        return None
    shape = get_proper_type(proper.args[0])
    if not isinstance(shape, TupleType):
        return None
    dims: list[int | Type] = []
    for item in shape.items:
        value = _dim_value(item)
        plain = get_proper_type(item)
        if value is not None:
            dims.append(value)
        elif isinstance(plain, Instance) and plain.type.fullname == INT_FULLNAME:
            dims.append(plain.copy_modified(last_known_value=None))
        else:
            return None
    return dims, proper.args[1]


def _literal_positions(arrays: list[Instance]) -> dict[int, set[int]]:
    """Per rank, the axes where some mixed array has a Literal dim."""
    out: dict[int, set[int]] = {}
    for arr in arrays:
        if not _is_mixed_shape(arr):
            continue
        shape = get_proper_type(arr.args[0])
        assert isinstance(shape, TupleType)
        axes = out.setdefault(len(shape.items), set())
        axes.update(
            k for k, i in enumerate(shape.items) if isinstance(get_proper_type(i), LiteralType)
        )
    return out


def _widen_literal_dims(
    tp: Type, int_axes: dict[int, set[int]] | None = None, literals: bool = True
) -> Type:
    """tp with every Literal dim of its ndarray shapes made unknown (Any).

    Any rather than int: a widened dim must still fit wherever the
    literal was expected (np.diff(x, axis=0) returned as x's
    tuple[int, Literal[3]] type), since unknown is not an error. A
    plain `int` dim goes unknown too where int_axes (rank -> axes, see
    _literal_positions) says a mixed operand of the same rank had a
    literal there: np.resize(x, x.shape) answers tuple[int, int] for x's
    tuple[int, Literal[3]], which would not fit x again. A recursive
    alias is left as is: it cannot be rebuilt widened, and expanding it
    would not terminate. literals=False widens only those int dims, for
    a result the plugin solved itself (_mixed_result).
    """
    if isinstance(tp, TypeAliasType) and tp.is_recursive:
        return tp
    proper = get_proper_type(tp)
    if isinstance(proper, UnionType):
        return UnionType.make_union(
            [_widen_literal_dims(i, int_axes, literals) for i in proper.items]
        )
    if isinstance(proper, TupleType):
        return proper.copy_modified(
            items=[_widen_literal_dims(i, int_axes, literals) for i in proper.items]
        )
    if not isinstance(proper, Instance):
        return tp
    if proper.type.fullname == NDARRAY_FULLNAME and len(proper.args) == 2:
        shape = get_proper_type(proper.args[0])
        if isinstance(shape, TupleType):
            axes = (int_axes or {}).get(len(shape.items), set())
            items = [
                AnyType(TypeOfAny.explicit)
                if (literals and isinstance(i, LiteralType)) or (k in axes and _is_plain_int(i))
                else i
                for k, i in enumerate(map(get_proper_type, shape.items))
            ]
            shape = shape.copy_modified(items=items)
        return proper.copy_modified(args=[shape, proper.args[1]])
    return proper.copy_modified(
        args=[_widen_literal_dims(a, int_axes, literals) for a in proper.args]
    )


# numpy ops whose stub answer keeps the shape of their one array operand:
# elementwise operators, copies, casts, same-shape rearrangements, sorts
# and scans along an axis (their axis=None overloads answer 1-d
# themselves). Ops that can resize (fft n=, *_like shape=, view(dtype))
# are deliberately absent, and a second array operand (a broadcasting
# summand, array clip bounds) voids the promise.
_SHAPE_KEEPERS = frozenset({
    "__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
    "__truediv__", "__rtruediv__", "__floordiv__", "__rfloordiv__",
    "__mod__", "__rmod__", "__pow__", "__rpow__", "__neg__", "__pos__",
    "__abs__", "__invert__", "__and__", "__rand__", "__or__", "__ror__",
    "__xor__", "__rxor__", "__eq__", "__ne__", "__lt__", "__le__",
    "__gt__", "__ge__",
    "astype", "copy", "clip", "round", "around", "conj", "conjugate",
    "byteswap", "real", "imag", "asarray", "asanyarray",
    "ascontiguousarray", "asfortranarray", "nan_to_num", "flip", "fliplr",
    "flipud", "roll", "fftshift", "ifftshift", "triu", "tril", "unwrap",
    "digitize", "interp", "sort", "argsort", "partition", "argpartition",
    "cumsum", "cumprod", "nancumsum", "nancumprod", "cumulative_sum",
    "cumulative_prod",
})


def _guard_mixed(fullname: str, hook: Callable[..., Type] | None) -> Callable[..., Type] | None:
    """Wrap a numpy hook so a mixed-shape operand cannot leak misplaced dims.

    Mixed shapes (tuple[int, Literal[3]]) come from builders over (n, 3)
    with n: int, or straight from annotations. The solver does not read
    int dims, so every op on such an array falls back to the stubs, and
    many stubs return the input's shape unchanged for ops that move or
    resize axes (.T, pad, rot90, rfft, split, swapaxes, diff, keepdims
    reductions, ...): the 3 would sit on the wrong axis. Unless the op
    keeps its one array operand's shape (_SHAPE_KEEPERS, *_like without
    shape=), the stubs' answer widens its Literal dims to unknown (Any;
    see _widen_literal_dims). Builder results got int dims there before
    they kept literals (#34); Any accepts everything int did. Its plain
    `int` dims on axes where a mixed operand had a literal go unknown as
    well: the stubs answer int there for ops that keep or rebuild the
    shape (np.resize), and int would not fit back where the literal is
    expected. A result the plugin built itself is never touched.
    """
    if hook is None and not fullname.startswith("numpy."):
        return None
    short = fullname.rsplit(".", 1)[-1]
    keeper = short in _SHAPE_KEEPERS
    # zeros_like & co keep the prototype's shape unless shape= replaces
    # it. view() counts as keeping only with no argument at all: a dtype
    # (or type) argument may change the last dim's width, and its width
    # is not compared here.
    like = short in ("zeros_like", "ones_like", "empty_like", "full_like")

    def keeps(ctx: FunctionContext | MethodContext | AttributeContext) -> bool:
        if keeper:
            return True
        if like and isinstance(ctx, FunctionContext):
            return _formal_pair(ctx, "shape") is None
        return short == "view" and isinstance(ctx, MethodContext) and not any(ctx.arg_types)

    def guarded(ctx: FunctionContext | MethodContext | AttributeContext) -> Type:
        if isinstance(ctx, AttributeContext):
            default, operands = ctx.default_attr_type, [ctx.type]
        else:
            default = ctx.default_return_type
            operands = [t for ts in ctx.arg_types for t in ts]
            if isinstance(ctx, MethodContext):
                operands.append(ctx.type)
        if hook is None:
            # Unhooked numpy.* op: observer hook (issue #60). Return the
            # stubs' type unchanged, but trace when a known shape flows
            # in unchecked. Strict does NOT fail here: unhooked numpy
            # functions are heavily overloaded (e.g. kron has 7
            # overloads) and any diagnostic inside the hook aborts
            # overload resolution, turning a correct call into arg-type
            # noise. Use the trace to find these; strict covers hooked
            # bails below.
            if _numty_trace_path() is not None and _operands_have_known_shape(
                operands
            ):
                _trace_event(ctx, fullname, "unhooked", "no hook for this op")
            result = default
        else:
            result = hook(ctx)
            if result is default:
                # Generic bail: any hook that kept the stubs' type. This
                # makes the 506 give-up sites visible without churning
                # them; specific reasons live in _bail() for new hooks.
                # Scalar-only bails (e.g. np.sqrt(8)) are traced too so
                # the trace shows where Any is born. Strict fails for
                # every bail: the sqrt birth on `... / np.sqrt(8)` has
                # no known-shape input itself, but its Any poisons the
                # known-shape truediv whose hook mypy then skips, so
                # gating on known inputs would leave the repro silent.
                _trace_event(
                    ctx, fullname, "bail", "unsolved (default stub type kept)"
                )
                if _numty_strict():
                    ctx.api.fail(
                        f"numty could not prove {short}: "
                        "unsolved (result is unchecked)",
                        ctx.context,
                        code=NUMTY_UNSOLVED,
                    )
            elif _numty_trace_path() is not None:
                _trace_event(ctx, fullname, "solved", "")
        # Only a stub answer holding a Literal or plain `int` dim can be
        # widened, so most calls (np.isscalar, .shape, scalar reductions)
        # skip the operands.
        if result is not default or not any(
            _has_literal_dim(a) or _has_int_dim(a) for a in _ndarrays_in(default)
        ):
            return result
        arrays = [a for t in operands for a in _ndarrays_in(t)]
        if not any(_is_mixed_shape(a) for a in arrays):
            return result
        if len(arrays) == 1 and keeps(ctx):
            return result
        return _widen_literal_dims(result, _literal_positions(arrays))

    return guarded


def _default_float_dtype(ctx: FunctionContext | MethodContext) -> Type:
    return ctx.api.named_generic_type(
        DTYPE_FULLNAME, [ctx.api.named_generic_type(FLOAT64_FULLNAME, [])]
    )


INTP_MAX = 2**63 - 1


def _dims_fit_intp(ctx: FunctionContext | MethodContext, dims: list[int | Type]) -> bool:
    """numpy stores dims as intp: a larger literal dim always raises."""
    for d in dims:
        if isinstance(d, int) and d > INTP_MAX:
            ctx.api.fail(
                f"dimension {d} exceeds the maximum allowed dimension "
                f"{INTP_MAX} (shape {_dims_str(list(dims))})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return False
    return True


def zeros_like_hook(ctx: FunctionContext) -> Type:
    shape_arg = _single_shape_arg(ctx)
    if shape_arg is None:
        return ctx.default_return_type
    dims = _builder_shape(shape_arg, _single_shape_expr(ctx), ctx.default_return_type)
    if dims is None:
        return ctx.default_return_type
    if not _dims_fit_intp(ctx, dims):
        return ctx.default_return_type
    if _explicit_dtype(ctx):
        dtype = _named_dtype(ctx)
        if dtype is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, dtype)
    return _ndarray_type(ctx, dims, _default_float_dtype(ctx))


def _matmul_inner(x: int | Type, y: int | Type) -> tuple[bool, str | None]:
    """(match, error) for inner dims: match on equality, error when both
    ints differ, and (False, None) to bail on anything unprovable. A
    plain `int` (dynamic) dim is assumed to match: it never reaches the
    output, and unknown is not an error."""
    if _dim_eq(x, y) or _is_plain_int(x) or _is_plain_int(y):
        return True, None
    if isinstance(x, int) and isinstance(y, int):
        return False, f"matmul inner dimensions disagree: {x} vs {y}"
    return False, None


def _matmul_result(
    a: list[int | Type],
    b: list[int | Type],
    ctx: FunctionContext | MethodContext | None = None,
) -> tuple[list[int | Type] | None, str | None]:
    """Output dims on success; (None, None) to bail; (None, msg) on error.

    An unprovable inner pair (N vs M, ?U vs ?U) is not a clash and the
    output dims do not depend on it (issue #56), so the outer dims are
    still returned instead of bailing the whole result to Any.
    """
    if not a or not b:
        return None, None
    if len(a) == 1 and len(b) == 1:
        _, message = _matmul_inner(a[0], b[0])
        if message is not None:
            return None, message
        return [], None
    if len(a) == 1:
        _, message = _matmul_inner(a[0], b[-2])
        if message is not None:
            return None, message
        return [*b[:-2], b[-1]], None
    if len(b) == 1:
        _, message = _matmul_inner(a[-1], b[0])
        if message is not None:
            return None, message
        return a[:-1], None
    _, message = _matmul_inner(a[-1], b[-2])
    if message is not None:
        return None, message
    batch_a, batch_b = a[:-2], b[:-2]
    batch, cause = _broadcast_result(batch_a, batch_b, ctx)
    if cause is not None:
        return None, (
            f"matmul batch dimensions disagree: {_dims_str(batch_a)} vs {_dims_str(batch_b)}"
            f" ({cause})"
        )
    if batch is None:
        return None, None
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


def _operand_parts(
    left: Type, right: Type
) -> tuple[tuple[list[int | Type], Type] | None, tuple[list[int | Type], Type] | None]:
    """Both operands' (dims, dtype) for a contraction (matmul, dot).

    With a mixed operand ((n, 3) from np.zeros((n, 3))), plain `int`
    dims ride along as types (_mixed_parts) so the literal dims that
    reach the output stay: (n, 3) @ (3, 3) is (int, 3), not the stubs'
    (int, int). Pass the result through _mixed_result.
    """
    parts_a = _ndarray_parts(left)
    parts_b = _ndarray_parts(right)
    if (parts_a is None or parts_b is None) and _has_mixed_operand(left, right):
        return _mixed_parts(left, partner=True), _mixed_parts(right, partner=True)
    return parts_a, parts_b


def _mixed_result(result: Type, *operands: Type) -> Type:
    """A contraction's result over mixed operands, see _operand_parts.

    Its literal dims are solved and stay. A plain `int` dim lands where a
    mixed operand of the same rank had a literal only when it is another
    operand's dynamic size ((n, 3) by (3, k) is (n, k)), and nothing
    proves k is not 3: that dim goes unknown, as the stubs' answers do
    in _guard_mixed, so x = np.dot(x, w) keeps fitting x.
    """
    arrays = [a for t in operands for a in _ndarrays_in(t)]
    if not any(_is_mixed_shape(a) for a in arrays):
        return result
    return _widen_literal_dims(result, _literal_positions(arrays), literals=False)


def _solve_matmul(ctx: FunctionContext | MethodContext, left: Type, right: Type) -> Type:
    parts_a, parts_b = _operand_parts(left, right)
    if parts_a is not None and parts_b is not None:
        (dims_a, dtype_a), (dims_b, dtype_b) = parts_a, parts_b
        dims, message = _matmul_result(dims_a, dims_b, ctx)
        if message is not None:
            ctx.api.fail(message, ctx.context, code=NUMTY_SHAPE)
            return ctx.default_return_type
        if dims is None:
            return ctx.default_return_type
        return _mixed_result(_matmul_promoted(ctx, dims, dtype_a, dtype_b), left, right)
    dims, message, dtype_a, dtype_b = _matmul_unpack_dims(left, right)
    if message is not None:
        ctx.api.fail(message, ctx.context, code=NUMTY_SHAPE)
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


def _matmul_has_dims(ctx: FunctionContext | MethodContext, operands: list[Type]) -> bool:
    """matmul rejects 0-d operands (scalars, shape-() arrays) outright.

    Only matmul itself: dot treats a 0-d operand as multiplication.
    """
    for i, tp in enumerate(operands):
        parts = _ndarray_parts(tp)
        key = _scalar_key(tp)
        if parts is not None and not parts[0]:
            what = "a 0-d array"
        elif key is not None:
            what = f"a scalar {key[1].rsplit('.', 1)[-1]}"
        else:
            continue
        ctx.api.fail(
            f"matmul operand {i} has no dimensions ({what}); "
            "matmul needs arrays with at least 1 dimension",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return False
    return True


def matmul_hook(ctx: FunctionContext) -> Type:
    args = _positional_actuals(ctx, 2)
    if args is None:
        return ctx.default_return_type
    if not _matmul_has_dims(ctx, args):
        return ctx.default_return_type
    return _solve_matmul(ctx, args[0], args[1])


def _two_arrays(ctx: FunctionContext) -> tuple[Type, Type] | None:
    """Two array actuals, formal-mapped when possible, raw otherwise.

    Under overload fallback mypy replays the hook against an erased
    callee whose formal mapping can be unusual; the actuals are still
    ours, so read them raw as a last resort. Errors recorded here must
    survive: the filtered candidate passes discard them.
    """
    args = _positional_actuals(ctx, 2)
    if args is not None:
        return args[0], args[1]
    raw: list[Type] = []
    for kinds, types in zip(ctx.arg_kinds, ctx.arg_types):
        for kind, tp in zip(kinds, types):
            if kind == ARG_POS:
                raw.append(tp)
    if len(raw) < 2:
        return None
    return raw[0], raw[1]


def _contract_pair(
    op: str, x: int | Type, y: int | Type
) -> tuple[bool, str | None]:
    """(match, error) for one contracted dim pair.

    Match on equality, error when both ints differ, and (False, None)
    to bail on anything unprovable. Mirrors _matmul_inner with an
    op-specific message.
    """
    if _dim_eq(x, y) or _is_plain_int(x) or _is_plain_int(y):
        return True, None
    if isinstance(x, int) and isinstance(y, int):
        return False, f"{op} contracted dimensions disagree: {x} vs {y}"
    return False, None


def _promoted_scalar(
    ctx: FunctionContext | MethodContext, dtype_a: Type, dtype_b: Type
) -> Type:
    """Scalar of the promoted dtype, else the default return."""
    promoted = _promoted_result_type(ctx, dtype_a, dtype_b)
    if promoted is None:
        return ctx.default_return_type
    scalar = _scalar_of_dtype(promoted)
    if scalar is None:
        return ctx.default_return_type
    return scalar


def dot_function_hook(ctx: FunctionContext) -> Type:
    pair = _two_arrays(ctx)
    if pair is None:
        return ctx.default_return_type
    parts_a, parts_b = _operand_parts(pair[0], pair[1])
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    (dims_a, dtype_a), (dims_b, dtype_b) = parts_a, parts_b
    if len(dims_a) == 1 and len(dims_b) == 1:
        # Vector inner product is a scalar (the stubs agree).
        matched, message = _contract_pair("dot", dims_a[0], dims_b[0])
        if message is not None:
            ctx.api.fail(message, ctx.context, code=NUMTY_SHAPE)
            return ctx.default_return_type
        if not matched:
            return ctx.default_return_type
        return _promoted_scalar(ctx, dtype_a, dtype_b)
    if max(len(dims_a), len(dims_b)) <= 2:
        # Rank <= 2 is otherwise exactly matmul: reuse its solver.
        return _solve_matmul(ctx, pair[0], pair[1])
    if len(dims_b) == 1:
        contracted, out = (dims_a[-1], dims_b[0]), dims_a[:-1]
    else:
        contracted = (dims_a[-1], dims_b[-2])
        out = dims_a[:-1] + dims_b[:-2] + dims_b[-1:]
    matched, message = _contract_pair("dot", *contracted)
    if message is not None:
        ctx.api.fail(message, ctx.context, code=NUMTY_SHAPE)
        return ctx.default_return_type
    if not matched:
        return ctx.default_return_type
    return _mixed_result(_matmul_promoted(ctx, out, dtype_a, dtype_b), pair[0], pair[1])


def dot_method_hook(ctx: MethodContext) -> Type:
    """ndarray.dot twin of np.dot: same solver on (self, arg) (issue #58)."""
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    left, right = ctx.type, pairs[0][0]
    parts_a, parts_b = _operand_parts(left, right)
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    (dims_a, dtype_a), (dims_b, dtype_b) = parts_a, parts_b
    if len(dims_a) == 1 and len(dims_b) == 1:
        matched, message = _contract_pair("dot", dims_a[0], dims_b[0])
        if message is not None:
            ctx.api.fail(message, ctx.context, code=NUMTY_SHAPE)
            return ctx.default_return_type
        if not matched:
            return ctx.default_return_type
        return _promoted_scalar(ctx, dtype_a, dtype_b)
    if max(len(dims_a), len(dims_b)) <= 2:
        return _solve_matmul(ctx, left, right)
    if len(dims_b) == 1:
        contracted, out = (dims_a[-1], dims_b[0]), dims_a[:-1]
    else:
        contracted = (dims_a[-1], dims_b[-2])
        out = dims_a[:-1] + dims_b[:-2] + dims_b[-1:]
    matched, message = _contract_pair("dot", *contracted)
    if message is not None:
        ctx.api.fail(message, ctx.context, code=NUMTY_SHAPE)
        return ctx.default_return_type
    if not matched:
        return ctx.default_return_type
    return _mixed_result(_matmul_promoted(ctx, out, dtype_a, dtype_b), left, right)


def inner_function_hook(ctx: FunctionContext) -> Type:
    pair = _two_arrays(ctx)
    if pair is None:
        return ctx.default_return_type
    parts_a = _ndarray_parts(pair[0])
    parts_b = _ndarray_parts(pair[1])
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    (dims_a, dtype_a), (dims_b, dtype_b) = parts_a, parts_b
    matched, message = _contract_pair("inner", dims_a[-1], dims_b[-1])
    if message is not None:
        ctx.api.fail(message, ctx.context, code=NUMTY_SHAPE)
        return ctx.default_return_type
    if not matched:
        return ctx.default_return_type
    out = dims_a[:-1] + dims_b[:-1]
    if not out:
        return _promoted_scalar(ctx, dtype_a, dtype_b)
    return _matmul_promoted(ctx, out, dtype_a, dtype_b)


def _total_size(dims: list[int | Type]) -> int | None:
    """Product of concrete dims, else None (unspellable, not an error)."""
    total = 1
    for d in dims:
        if not isinstance(d, int) or isinstance(d, bool):
            return None
        total *= d
    return total


def outer_function_hook(ctx: FunctionContext) -> Type:
    pair = _two_arrays(ctx)
    if pair is None:
        return ctx.default_return_type
    parts_a = _ndarray_parts(pair[0])
    parts_b = _ndarray_parts(pair[1])
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    (dims_a, dtype_a), (dims_b, dtype_b) = parts_a, parts_b
    total_a = _total_size(dims_a)
    total_b = _total_size(dims_b)
    if total_a is None or total_b is None:
        return ctx.default_return_type
    return _matmul_promoted(ctx, [total_a, total_b], dtype_a, dtype_b)


def kron_function_hook(ctx: FunctionContext) -> Type:
    """np.kron: right-aligned dim products, dtype result_type (issue #58).

    (2, 3) kron (4, 5) is (8, 15): shorter inputs pad with 1s on the
    left, each output dim is the product of the aligned pair. Only
    int dims solve (symbolic products are nonlinear); anything else
    bails to the stubs instead of guessing.
    """
    pair = _two_arrays(ctx)
    if pair is None:
        return ctx.default_return_type
    parts_a = _ndarray_parts(pair[0])
    parts_b = _ndarray_parts(pair[1])
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    (dims_a, dtype_a), (dims_b, dtype_b) = parts_a, parts_b
    if _has_unpack(dims_a) or _has_unpack(dims_b):
        return ctx.default_return_type
    width = max(len(dims_a), len(dims_b))
    left = [1] * (width - len(dims_a)) + list(dims_a)
    right = [1] * (width - len(dims_b)) + list(dims_b)
    out: list[int | Type] = []
    for x, y in zip(left, right):
        if isinstance(x, int) and isinstance(y, int):
            if isinstance(x, bool) or isinstance(y, bool):
                return ctx.default_return_type
            out.append(x * y)
        else:
            return ctx.default_return_type
    dtype = _promoted_result_type(ctx, dtype_a, dtype_b)
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, out, dtype)


def vdot_function_hook(ctx: FunctionContext) -> Type:
    pair = _two_arrays(ctx)
    if pair is None:
        return ctx.default_return_type
    parts_a = _ndarray_parts(pair[0])
    parts_b = _ndarray_parts(pair[1])
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    (dims_a, _), (dims_b, _) = parts_a, parts_b
    total_a = _total_size(dims_a)
    total_b = _total_size(dims_b)
    if total_a is not None and total_b is not None and total_a != total_b:
        ctx.api.fail(
            f"vdot operands have different sizes: {total_a} vs {total_b}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    return _promoted_scalar(ctx, parts_a[1], parts_b[1])


def _tensordot_axes(
    pair: tuple[Type, Expression | None] | None,
) -> int | tuple[list[int], list[int]] | None:
    """int k, or a pair of axis lists; None when unparseable.

    Mirrors numpy: an int contracts the last k of a with the first k
    of b; otherwise a pair of axis lists, each entry an int or a list
    of ints.
    """
    if pair is None:
        return 2  # numpy's default
    tp, expr = pair
    k = _int_value(tp)
    if k is None and expr is not None:
        k = _fold_int(expr)
    if k is not None:
        return k
    if not isinstance(expr, (TupleExpr, ListExpr)) or len(expr.items) != 2:
        return None
    sides: list[list[int]] = []
    for item in expr.items:
        if isinstance(item, (TupleExpr, ListExpr)):
            values = [_fold_int(e) for e in item.items]
            if any(v is None for v in values):
                return None
            sides.append([v for v in values if v is not None])
        else:
            v = _fold_int(item)
            if v is None:
                return None
            sides.append([v])
    return (sides[0], sides[1])


def tensordot_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        raw: list[tuple[Type, Expression | None]] = []
        for kinds, types, exprs in zip(ctx.arg_kinds, ctx.arg_types, ctx.args):
            for kind, tp, expr in zip(kinds, types, exprs):
                if kind == ARG_POS:
                    raw.append((tp, expr))
        pairs = raw
    if len(pairs) < 2:
        return ctx.default_return_type
    parts_a = _ndarray_parts(pairs[0][0])
    parts_b = _ndarray_parts(pairs[1][0])
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    (dims_a, dtype_a), (dims_b, dtype_b) = parts_a, parts_b
    form = _tensordot_axes(_formal_pair(ctx, "axes"))
    if form is None:
        return ctx.default_return_type
    if isinstance(form, int):
        if form < 0:
            return ctx.default_return_type
        if form > len(dims_a) or form > len(dims_b):
            ctx.api.fail(
                f"tensordot axes {form} contracts more axes than the operands "
                f"have (shapes {_dims_str(dims_a)} vs {_dims_str(dims_b)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        contract = list(zip(range(len(dims_a) - form, len(dims_a)), range(form)))
    else:
        axes_a, axes_b = form
        if len(axes_a) != len(axes_b):
            ctx.api.fail(
                f"tensordot axes have different lengths: {len(axes_a)} vs "
                f"{len(axes_b)}",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        contract = []
        seen_a: set[int] = set()
        seen_b: set[int] = set()
        for ia, ib in zip(axes_a, axes_b):
            na = ia + len(dims_a) if ia < 0 else ia
            nb = ib + len(dims_b) if ib < 0 else ib
            if not 0 <= na < len(dims_a):
                ctx.api.fail(
                    f"tensordot axis {ia} is out of range for {len(dims_a)} "
                    f"dimensions (shape {_dims_str(dims_a)})",
                    ctx.context,
                    code=NUMTY_AXIS,
                )
                return ctx.default_return_type
            if not 0 <= nb < len(dims_b):
                ctx.api.fail(
                    f"tensordot axis {ib} is out of range for {len(dims_b)} "
                    f"dimensions (shape {_dims_str(dims_b)})",
                    ctx.context,
                    code=NUMTY_AXIS,
                )
                return ctx.default_return_type
            if na in seen_a or nb in seen_b:
                bad = ia if na in seen_a else ib
                ctx.api.fail(
                    f"duplicate axis {bad} in tensordot axes",
                    ctx.context,
                    code=NUMTY_AXIS,
                )
                return ctx.default_return_type
            seen_a.add(na)
            seen_b.add(nb)
            contract.append((na, nb))
    drop_a = {i for i, _ in contract}
    drop_b = {j for _, j in contract}
    for i, j in contract:
        matched, message = _contract_pair("tensordot", dims_a[i], dims_b[j])
        if message is not None:
            ctx.api.fail(message, ctx.context, code=NUMTY_SHAPE)
            return ctx.default_return_type
        if not matched:
            return ctx.default_return_type
    out = [d for i, d in enumerate(dims_a) if i not in drop_a] + [
        d for j, d in enumerate(dims_b) if j not in drop_b
    ]
    return _matmul_promoted(ctx, out, dtype_a, dtype_b)


def _take_indices(
    pair: tuple[Type, Expression | None],
) -> tuple[list[int | Type] | None, bool]:
    """(index dims, is_scalar) for take indices; (None, False) to bail.

    Arrays keep their dims, flat int displays spell [len], and a bare
    int selects one entry (dropping the axis).
    """
    tp, expr = pair
    parts = _parts_or_unpack(tp)
    if parts is not None:
        dims, _ = parts
        if _has_unpack(dims):
            return None, False
        return list(dims), False
    if (
        isinstance(expr, (TupleExpr, ListExpr))
        and expr.items
        and all(isinstance(e, IntExpr) for e in expr.items)
    ):
        return [len(expr.items)], False
    if _int_value(tp) is not None or (
        expr is not None and _fold_int(expr) is not None
    ):
        return [], True
    return None, False


def _take_index_values(pair: tuple[Type, Expression | None]) -> list[int] | None:
    """Literal take indices (an int or a flat int display), else None."""
    tp, expr = pair
    if isinstance(expr, (TupleExpr, ListExpr)):
        values = [_fold_int(e) for e in expr.items]
        if any(v is None for v in values):
            return None
        return [v for v in values if v is not None]
    value = _int_value(tp)
    if value is None and expr is not None:
        value = _fold_int(expr)
    return [value] if value is not None else None


def _take_in_bounds(
    ctx: FunctionContext | MethodContext,
    pair: tuple[Type, Expression | None],
    size: int | Type,
    where: str,
    dims: list[int | Type],
) -> bool:
    """take with mode='raise' (the default): literal indices must exist.

    clip/wrap remap any index, so only a literal 'raise' mode, or one
    provably absent (the callee names a mode formal left unbound), is
    checked.
    """
    if not isinstance(size, int) or "mode" not in ctx.callee_arg_names:
        return True
    mode = _formal_pair(ctx, "mode")
    if mode is not None and _string_value(mode[0]) != "raise":
        return True
    values = _take_index_values(pair)
    for value in values or []:
        if not -size <= value < size:
            ctx.api.fail(
                f"take index {value} is out of bounds for {where} "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return False
    return True


def _take_solve(
    ctx: FunctionContext | MethodContext,
    array: Type,
    pair: tuple[Type, Expression | None] | None,
) -> Type:
    if pair is None:
        return ctx.default_return_type
    parts = _ndarray_parts(array)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    idx_dims, is_scalar = _take_indices(pair)
    if idx_dims is None:
        return ctx.default_return_type
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if axis == "absent" or axis == "none":
        # axis=None flattens: the result is exactly the index shape,
        # or a scalar for a scalar index.
        total: int | None = 1
        for d in dims:
            total = total * d if isinstance(d, int) and total is not None else None
        if total is not None and not _take_in_bounds(
            ctx, pair, total, f"the flattened size {total}", dims
        ):
            return ctx.default_return_type
        if is_scalar:
            scalar = _scalar_of_dtype(dtype)
            return scalar if scalar is not None else ctx.default_return_type
        return _ndarray_type(ctx, idx_dims, dtype)
    if not isinstance(axis, list) or len(axis) != 1 or not isinstance(axis[0], int):
        ctx.api.fail("np.take axis must be an integer or None", ctx.context, code=NUMTY_AXIS)
        return ctx.default_return_type
    norm = _normalize_axis(axis[0], len(dims))
    if norm is None:
        ctx.api.fail(
            f"axis {axis[0]} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    # numpy skips the index check when a dim before the axis is 0:
    # there is no row to take from.
    empty_outer = any(isinstance(d, int) and d == 0 for d in dims[:norm])
    if not empty_outer and not _take_in_bounds(
        ctx, pair, dims[norm], f"axis {axis[0]} of size {dims[norm]}", dims
    ):
        return ctx.default_return_type
    if is_scalar:
        out = dims[:norm] + dims[norm + 1 :]
        if not out:
            scalar = _scalar_of_dtype(dtype)
            return scalar if scalar is not None else ctx.default_return_type
        return _ndarray_type(ctx, out, dtype)
    return _ndarray_type(ctx, dims[:norm] + idx_dims + dims[norm + 1 :], dtype)


def take_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    return _take_solve(ctx, pairs[0][0], pairs[1])


def take_method_hook(ctx: MethodContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _take_solve(ctx, ctx.type, pairs[0])


def _positionals_only(ctx: FunctionContext | MethodContext) -> bool:
    """True when every actual is positional: nothing hides behind keywords."""
    return all(kind == ARG_POS for kinds in ctx.arg_kinds for kind in kinds)


def _picture_complete(ctx: FunctionContext | MethodContext, first: str) -> bool:
    """True when every actual is visible to the hook.

    Overload fallback erases formal names, so keyword actuals become
    unreadable; a mixed call under erasure bails instead of misreading.
    All-positional calls are always fully visible.
    """
    if _positionals_only(ctx):
        return True
    return _formal_pair(ctx, first) is not None


def _int_or_none(pair: tuple[Type, Expression | None] | None) -> int | None | str:
    """A folded int, None when absent/None, 'bail' when unprovable."""
    if pair is None:
        return None
    tp, expr = pair
    if _is_none_type(tp):
        return None
    value = _int_value(tp)
    if value is None and expr is not None:
        value = _fold_int(expr)
    return value if value is not None else "bail"


def _fft_pair(
    ctx: FunctionContext, names: tuple[str, str], indices: tuple[int, int]
) -> list[tuple[Type, Expression | None] | None] | None:
    """Two optional actuals: raw positionals when complete, formals when live."""
    if _positionals_only(ctx):
        pos = _all_positionals(ctx)
        return [pos[i] if i < len(pos) else None for i in indices]
    if _formal_pair(ctx, "a") is None:
        return None
    return [_formal_pair(ctx, name) for name in names]


def _check_each_axis(
    ctx: FunctionContext | MethodContext, dims: list[int | Type], axes: list[int], op: str
) -> bool:
    """True when every axis is in range, else a loud error."""
    for ax in axes:
        if _normalize_axis(ax, len(dims)) is None:
            ctx.api.fail(
                f"{op} axis {ax} is out of range for {len(dims)} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return False
    return True


def _check_unique_axes(
    ctx: FunctionContext | MethodContext, dims: list[int | Type], axes: list[int], op: str
) -> bool:
    """True when the in-range axes name distinct dims, else a loud error."""
    seen: list[int] = []
    for ax in axes:
        norm = _normalize_axis(ax, len(dims))
        if norm is None:
            continue
        if norm in seen:
            ctx.api.fail(
                f"{op} has duplicate axis {ax} in {tuple(axes)} "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return False
        seen.append(norm)
    return True


def _roll_shift_matches(
    ctx: FunctionContext, dims: list[int | Type], axis: list[int] | None | str
) -> bool:
    """roll: shift and axis sequences must broadcast, else a loud error.

    numpy broadcasts shift against axis, so equal lengths or a length of
    one on either side pass; anything else always raises.
    """
    if not isinstance(axis, list):
        return True
    found = _hook_pair(ctx, "a", "shift", 1)
    if found is None or isinstance(found, str):
        return True
    tp, expr = found
    shift = _shape_from_arg(tp, expr)
    if shift is None or len(shift) == len(axis) or 1 in (len(shift), len(axis)):
        return True
    ctx.api.fail(
        f"roll shift has {len(shift)} entries for {len(axis)} axes "
        f"(shift {tuple(shift)}, axis {tuple(axis)}, shape {_dims_str(dims)})",
        ctx.context,
        code=NUMTY_AXIS,
    )
    return False


def _same_shape_hook(
    ctx: FunctionContext,
    op: str,
    first: str,
    axis_index: int | None,
    rank_min: int = 0,
    unique_axes: bool = False,
) -> Type:
    """Shape-preserving ops: optional axis validated, input returned.

    The returned array object is the input type itself, so overload
    candidates always agree. The axis reads from its formal, falling back
    to the positional slot when the call is positional-only. unique_axes
    rejects an axis named twice (flip); roll legitimately allows it.
    """
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    if len(dims) < rank_min:
        ctx.api.fail(
            f"{op} requires at least {rank_min} dimensions, got {len(dims)} "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    if not _picture_complete(ctx, first):
        return ctx.default_return_type
    pair = _formal_pair(ctx, "axis")
    if pair is None and axis_index is not None and _positionals_only(ctx):
        pos = _all_positionals(ctx)
        pair = pos[axis_index] if axis_index < len(pos) else None
    axis = _parse_axis_arg(pair)
    if isinstance(axis, list) and not _check_each_axis(ctx, dims, axis, op):
        return ctx.default_return_type
    if (
        unique_axes
        and isinstance(axis, list)
        and not _check_unique_axes(ctx, dims, axis, op)
    ):
        return ctx.default_return_type
    if op == "roll" and not _roll_shift_matches(ctx, dims, axis):
        return ctx.default_return_type
    if axis is not None and not isinstance(axis, (list, str)):
        return ctx.default_return_type
    return pairs[0][0]


def flip_function_hook(ctx: FunctionContext) -> Type:
    return _same_shape_hook(ctx, "flip", first="m", axis_index=1, unique_axes=True)


def fliplr_function_hook(ctx: FunctionContext) -> Type:
    return _same_shape_hook(ctx, "fliplr", first="m", axis_index=None, rank_min=2)


def flipud_function_hook(ctx: FunctionContext) -> Type:
    return _same_shape_hook(ctx, "flipud", first="m", axis_index=None, rank_min=1)


def roll_function_hook(ctx: FunctionContext) -> Type:
    return _same_shape_hook(ctx, "roll", first="a", axis_index=2)


def gradient_function_hook(ctx: FunctionContext) -> Type:
    """gradient: one resolved axis keeps the shape, else a tuple of arrays.

    axis=None (or absent) differentiates along every axis, returning a
    tuple for ndim > 1; an int or single-element tuple resolves one axis
    and keeps the shape. Multi-axis tuples bail to the stub type once
    their axes are checked to be in range and distinct.
    """
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        # Gradient along one axis keeps the operand's type whatever its
        # dims are, so plain `int` dims (mixed or all-int) only need a
        # rank for the axis checks.
        parts = _mixed_parts(pairs[0][0], partner=True)
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    if not _picture_complete(ctx, "f"):
        return ctx.default_return_type
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if axis is None:
        return ctx.default_return_type
    if isinstance(axis, str):
        if len(dims) == 1:
            return pairs[0][0]
        return ctx.default_return_type
    if not _check_each_axis(ctx, dims, axis, "gradient") or not _check_unique_axes(
        ctx, dims, axis, "gradient"
    ):
        return ctx.default_return_type
    if len(axis) != 1:
        return ctx.default_return_type
    return pairs[0][0]


def _parse_int_pair(
    expr: Expression | None,
) -> tuple[int, int] | None:
    """A pair of ints from a 2-tuple/list display, else None."""
    if not isinstance(expr, (TupleExpr, ListExpr)) or len(expr.items) != 2:
        return None
    values = [_fold_int(e) for e in expr.items]
    if any(v is None for v in values):
        return None
    assert values[0] is not None and values[1] is not None
    return (values[0], values[1])


def rot90_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    if _positionals_only(ctx):
        k_pair = pairs[1] if len(pairs) > 1 else None
        axes_pair = pairs[2] if len(pairs) > 2 else None
    elif _formal_pair(ctx, "m") is None:
        return ctx.default_return_type
    else:
        k_pair = _formal_pair(ctx, "k")
        axes_pair = _formal_pair(ctx, "axes")
    k = 1
    if k_pair is not None:
        folded = _int_value(k_pair[0])
        if folded is None and k_pair[1] is not None:
            folded = _fold_int(k_pair[1])
        if folded is None:
            return ctx.default_return_type
        k = folded
    ia, ib = 0, 1
    if axes_pair is not None:
        parsed = _parse_int_pair(axes_pair[1])
        if parsed is None:
            return ctx.default_return_type
        ia, ib = parsed
    for ax in (ia, ib):
        if _normalize_axis(ax, len(dims)) is None:
            ctx.api.fail(
                f"rot90 axis {ax} is out of range for {len(dims)} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
    na = ia + len(dims) if ia < 0 else ia
    nb = ib + len(dims) if ib < 0 else ib
    if na == nb:
        # numpy checks this before looking at k, so every k raises.
        ctx.api.fail(
            f"rot90 axes must be different (got ({ia}, {ib}) on shape "
            f"{_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims)
    if k % 4 in (1, 3):
        out[na], out[nb] = out[nb], out[na]
    return _ndarray_type(ctx, out, dtype)


def _string_value(tp: Type) -> str | None:
    """String of a literal mode/kind argument, else None."""
    proper = get_proper_type(tp)
    if isinstance(proper, LiteralType) and isinstance(proper.value, str):
        return proper.value
    return None


def _convolve_solve(
    ctx: FunctionContext, op: str, a: Type, b: Type, mode: str | None
) -> Type:
    parts_a = _ndarray_parts(a)
    parts_b = _ndarray_parts(b)
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    (dims_a, dtype_a), (dims_b, dtype_b) = parts_a, parts_b
    if len(dims_a) != 1 or len(dims_b) != 1:
        ctx.api.fail(
            f"{op} only supports 1-dimensional arrays "
            f"(shapes {_dims_str(dims_a)} and {_dims_str(dims_b)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    for which, dim in (("first", dims_a[0]), ("second", dims_b[0])):
        if isinstance(dim, int) and dim == 0:
            ctx.api.fail(
                f"{op} {which} array argument cannot be empty "
                f"(shapes {_dims_str(dims_a)} and {_dims_str(dims_b)})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
    if mode not in ("full", "valid", "same"):
        return ctx.default_return_type
    na, nb = dims_a[0], dims_b[0]
    if not isinstance(na, int) or not isinstance(nb, int):
        size = _dim_add_const(ctx, op, 0, _dim_add2(ctx, op, 0, na, nb), -1)
        return _matmul_promoted(ctx, [size], dtype_a, dtype_b)
    if mode == "full":
        n = na + nb - 1
    elif mode == "same":
        n = max(na, nb)
    else:
        n = max(na, nb) - min(na, nb) + 1
    return _matmul_promoted(ctx, [n], dtype_a, dtype_b)


def convolve_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    mode_pair = _formal_pair(ctx, "mode")
    mode = "full" if mode_pair is None else _string_value(mode_pair[0])
    return _convolve_solve(ctx, "convolve", pairs[0][0], pairs[1][0], mode)


def correlate_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    mode_pair = _formal_pair(ctx, "mode")
    mode = "valid" if mode_pair is None else _string_value(mode_pair[0])
    return _convolve_solve(ctx, "correlate", pairs[0][0], pairs[1][0], mode)


def _complex_dtype(ctx: FunctionContext, dtype: Type) -> Type:
    """complex128 for an fft-family input, complex64 for float32."""
    name = _dtype_fullname(dtype)
    if name is None:
        return _default_dtype(ctx)
    out = "numpy.complex64" if name == "numpy.float32" else "numpy.complex128"
    built = _construct_dtype(ctx, out)
    return built if built is not None else _default_dtype(ctx)


def fft_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    parsed = _fft_axis_n(ctx)
    if parsed is None:
        return ctx.default_return_type
    n, axis = parsed
    target = _normalize_axis(axis, len(dims))
    if target is None:
        ctx.api.fail(
            f"fft axis {axis} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims)
    if isinstance(n, int):
        out[target] = n
    return _ndarray_type(ctx, out, _complex_dtype(ctx, dtype))


def _rfft_size(
    ctx: FunctionContext, op: str, index: int, dim: int | Type
) -> int | Type:
    return _dim_div(ctx, op, index, dim, 2, 1)


def _rfft_solve(
    ctx: FunctionContext, op: str, arg: Type, axis: int, n: int | None
) -> Type:
    parts = _ndarray_parts(arg)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    target = _normalize_axis(axis, len(dims))
    if target is None:
        ctx.api.fail(
            f"{op} axis {axis} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims)
    if isinstance(n, int):
        out[target] = n // 2 + 1
    else:
        out[target] = _rfft_size(ctx, op, target, out[target])
    return _ndarray_type(ctx, out, _complex_dtype(ctx, dtype))


def rfft_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parsed = _fft_axis_n(ctx)
    if parsed is None:
        return ctx.default_return_type
    n, axis = parsed
    return _rfft_solve(ctx, "rfft", pairs[0][0], axis, n)


def _fft_axis_n(ctx: FunctionContext) -> tuple[int | None, int] | None:
    """(n, axis) for fft/rfft: None to bail, axis defaulting to -1."""
    found = _fft_pair(ctx, ("n", "axis"), (1, 2))
    if found is None:
        return None
    n = _int_or_none(found[0])
    axis = _int_or_none(found[1])
    if n == "bail" or axis == "bail":
        return None
    assert not isinstance(n, str) and not isinstance(axis, str)
    if isinstance(n, int) and n < 1:
        ctx.api.fail(
            f"fft n must be a positive number of data points, got {n}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return None
    return (n, -1 if axis is None else axis)


def _fft_seq_entry(expr: Expression, tp: Type | None) -> int | None | str:
    """An s/axes entry: int, None for keep, 'bad' when unprovable."""
    value = _fold_int(expr)
    if value is not None:
        return value
    if tp is not None and _is_none_type(tp):
        return None
    if isinstance(expr, NameExpr) and expr.name == "None":
        return None
    return "bad"


def _fft_seq_arg(
    pair: tuple[Type, Expression | None] | None,
) -> list[int | None] | None | str:
    """An s/axes argument: None absent, parsed list, 'scalar', or 'bail'."""
    if pair is None:
        return None
    tp, expr = pair
    if _is_none_type(tp):
        return None
    if expr is not None and isinstance(expr, (TupleExpr, ListExpr)):
        proper = get_proper_type(tp)
        types: list[Type | None]
        if isinstance(proper, TupleType) and len(proper.items) == len(expr.items):
            types = list(proper.items)
        else:
            types = [None] * len(expr.items)
        out = [_fft_seq_entry(e, t) for e, t in zip(expr.items, types)]
        return out if all(v != "bad" for v in out) else "bail"
    proper = get_proper_type(tp)
    if isinstance(proper, TupleType):
        out2: list[int | None] = []
        for item in proper.items:
            if _is_none_type(item):
                out2.append(None)
                continue
            value = _dim_value(item)
            if not isinstance(value, int) or isinstance(value, bool):
                return "bail"
            out2.append(value)
        return out2
    if _int_value(tp) is not None:
        return "scalar"
    return "bail"


def _fft_s_axes(
    ctx: FunctionContext,
) -> tuple[list[int | None] | None | str, list[int | None] | None | str] | None:
    """(s, axes) for fftn/rfftn: None to bail on an unreadable call."""
    found = _fft_pair(ctx, ("s", "axes"), (1, 2))
    if found is None:
        return None
    s = _fft_seq_arg(found[0])
    axes = _fft_seq_arg(found[1])
    if s == "bail" or axes == "bail":
        return None
    return (s, axes)


def _fftn_solve(
    ctx: FunctionContext,
    op: str,
    arg: Type,
    axes: list[int | None] | None,
    s: list[int | None] | None,
    rfft: bool,
) -> Type:
    """fftn/rfftn output shape: s resizes, the last listed axis halves for rfft.

    axes=None with s aligns s to the trailing axes. Unlisted dims keep
    their size; None entries in s keep too. Duplicate axes bail: the
    runtime result follows application order, which has no static shape.
    """
    parts = _ndarray_parts(arg)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    ndim = len(dims)
    if s is None and axes is None:
        out = list(dims)
        if rfft and out:
            out[-1] = _rfft_size(ctx, op, len(out) - 1, out[-1])
        return _ndarray_type(ctx, out, _complex_dtype(ctx, dtype))
    sizes: list[int | None]
    if axes is None:
        assert s is not None
        if len(s) > ndim:
            ctx.api.fail(
                f"{op} s has {len(s)} entries for {ndim} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        eff = list(range(ndim - len(s), ndim))
        sizes = list(s)
    else:
        if any(ax is None for ax in axes):
            return ctx.default_return_type
        raw = [ax for ax in axes if ax is not None]
        assert len(raw) == len(axes)
        eff = []
        for ax in raw:
            target = _normalize_axis(ax, ndim)
            if target is None:
                ctx.api.fail(
                    f"{op} axis {ax} is out of range for {ndim} dimensions "
                    f"(shape {_dims_str(dims)})",
                    ctx.context,
                    code=NUMTY_AXIS,
                )
                return ctx.default_return_type
            eff.append(target)
        if len(set(eff)) != len(eff):
            return ctx.default_return_type
        if s is not None and len(s) != len(eff):
            ctx.api.fail(
                f"{op} Shape and axes have different lengths "
                f"({len(s)} vs {len(eff)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        sizes = list(s) if s is not None else [None] * len(eff)
    out = list(dims)
    for ax, size in zip(eff, sizes):
        if size is None:
            continue
        if size < 1:
            ctx.api.fail(
                f"{op} s entries must be positive, got {size}",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        out[ax] = size
    if rfft and eff:
        last = eff[-1]
        out[last] = _dim_div(ctx, op, last, out[last], 2, 1)
    return _ndarray_type(ctx, out, _complex_dtype(ctx, dtype))


def fftn_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parsed = _fft_s_axes(ctx)
    if parsed is None:
        return ctx.default_return_type
    s, axes = parsed
    if isinstance(s, str) or isinstance(axes, str):
        # A bare scalar matches no stub overload, so the native
        # call-overload error already fires; bail silently here.
        return ctx.default_return_type
    return _fftn_solve(ctx, "fftn", pairs[0][0], axes, s, False)


def fft2_function_hook(ctx: FunctionContext) -> Type:
    """np.fft.fft2/ifft2: 2-D fft, same shape when s is absent (issue #58).

    s given resizes (and its default axes differ from fftn), so bail
    there instead of guessing; the common no-s call keeps the shape
    with the fft complex dtype, and bad axes still fail.
    """
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parsed = _fft_s_axes(ctx)
    if parsed is None:
        return ctx.default_return_type
    s, axes = parsed
    if isinstance(s, str) or isinstance(axes, str):
        return ctx.default_return_type
    if s is not None:
        return ctx.default_return_type
    return _fftn_solve(ctx, "fft2", pairs[0][0], axes, s, False)


def fftshift_function_hook(ctx: FunctionContext) -> Type:
    """np.fft.fftshift/ifftshift: same shape and dtype (issue #58).

    Axes only select which axes shift; the shape never changes, so no
    axis parsing is needed for precision. Unranked shapes bail.
    """
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    return _ndarray_type(ctx, list(dims), dtype)


def rfftn_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parsed = _fft_s_axes(ctx)
    if parsed is None:
        return ctx.default_return_type
    s, axes = parsed
    if isinstance(s, str) or isinstance(axes, str):
        # A bare scalar matches no stub overload, so the native
        # call-overload error already fires; bail silently here.
        return ctx.default_return_type
    return _fftn_solve(ctx, "rfftn", pairs[0][0], axes, s, True)


def _hook_pair(
    ctx: FunctionContext, first: str, name: str, index: int
) -> tuple[Type, Expression | None] | None | str:
    """An optional actual: formal, else the positional slot when complete.

    Returns "incomplete" when keywords may hide behind erased formals.
    """
    if not _picture_complete(ctx, first):
        return "incomplete"
    pair = _formal_pair(ctx, name)
    if pair is None and _positionals_only(ctx):
        pos = _all_positionals(ctx)
        pair = pos[index] if index < len(pos) else None
    return pair


def _tile_reps(pair: tuple[Type, Expression | None]) -> list[int | None] | None:
    """reps entries (None = unknown); None whole when absent or garbage."""
    tp, expr = pair
    if _is_none_type(tp):
        return None
    if expr is not None and isinstance(expr, (TupleExpr, ListExpr)):
        proper = get_proper_type(tp)
        types: list[Type | None]
        if isinstance(proper, TupleType) and len(proper.items) == len(expr.items):
            types = list(proper.items)
        else:
            types = [None] * len(expr.items)
        out: list[int | None] = []
        for e, t in zip(expr.items, types):
            value = _fold_int(e)
            if value is None and t is not None and not _is_none_type(t):
                dim = _dim_value(t)
                value = dim if isinstance(dim, int) and not isinstance(dim, bool) else None
            out.append(value)
        return out
    proper = get_proper_type(tp)
    if isinstance(proper, TupleType):
        out2: list[int | None] = []
        for item in proper.items:
            if _is_none_type(item):
                return None
            dim = _dim_value(item)
            out2.append(dim if isinstance(dim, int) and not isinstance(dim, bool) else None)
        return out2
    value = _int_value(tp)
    if value is None and expr is not None:
        value = _fold_int(expr)
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    return [None]


def tile_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    found = _hook_pair(ctx, "A", "reps", 1)
    if found == "incomplete" or found is None:
        return ctx.default_return_type
    assert not isinstance(found, str)
    reps = _tile_reps(found)
    if reps is None:
        return ctx.default_return_type
    shown = "[" + ", ".join("?" if r is None else str(r) for r in reps) + "]"
    if any(isinstance(r, int) and r < 0 for r in reps):
        ctx.api.fail(f"tile reps must be non-negative, got {shown}", ctx.context, code=NUMTY_SHAPE)
        return ctx.default_return_type
    if len(reps) > len(dims):
        eff = [1] * (len(reps) - len(dims)) + list(dims)
    else:
        eff = list(dims)
        reps = [1] * (len(eff) - len(reps)) + reps
    out: list[int | Type] = []
    for i, (d, r) in enumerate(zip(eff, reps)):
        if isinstance(r, int):
            out.append(_dim_mul_const(ctx, "tile", i, d, r))
        else:
            out.append(_fresh_unknown(ctx, "tile", i))
    return _ndarray_type(ctx, out, dtype)


def _repeat_count(pair: tuple[Type, Expression | None] | None) -> int | str | None:
    """Scalar repeats, 'bail' for array-form/unprovable, None when absent."""
    if pair is None:
        return None
    tp, expr = pair
    if _is_none_type(tp):
        return None
    value = _int_value(tp)
    if value is None and expr is not None:
        value = _fold_int(expr)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return "bail"


def _repeat_display(
    pair: tuple[Type, Expression | None] | None,
) -> list[int | None] | None:
    """Entries of a list/tuple repeats display (None = unknown), else None."""
    if pair is None or not isinstance(pair[1], (TupleExpr, ListExpr)):
        return None
    if any(isinstance(e, StarExpr) for e in pair[1].items):
        return None
    return [_fold_int(e) for e in pair[1].items]


def _repeat_display_size(
    ctx: FunctionContext,
    dims: list[int | Type],
    axis: int | None,
    entries: list[int | None],
) -> int | str | None:
    """New size for a multi-entry repeats display; 'bail', or None on error.

    numpy broadcasts the counts against the repeated axis (or the
    flattened array for axis=None), so a display of more than one entry
    needs exactly one per element. Known non-negative counts add up to
    the new size.
    """
    target: int | Type
    if axis is None:
        total = 1
        for d in dims:
            if not isinstance(d, int):
                return "bail"
            total *= d
        target, where = total, f"the flattened size {total}"
    else:
        norm = _normalize_axis(axis, len(dims))
        if norm is None:
            # The hook's axis range check reports this.
            return "bail"
        target, where = dims[norm], f"axis {axis} of size {dims[norm]}"
    if not isinstance(target, int):
        return "bail"
    if len(entries) != target:
        ctx.api.fail(
            f"repeat repeats has {len(entries)} entries for {where} "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return None
    counts = [e for e in entries if e is not None and e >= 0]
    return sum(counts) if len(counts) == len(entries) else "bail"


def repeat_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    found_repeats = _hook_pair(ctx, "a", "repeats", 1)
    found_axis = _hook_pair(ctx, "a", "axis", 2)
    if found_repeats == "incomplete" or found_axis == "incomplete":
        return ctx.default_return_type
    assert not isinstance(found_repeats, str) and not isinstance(found_axis, str)
    axis: int | None = None
    if found_axis is not None:
        parsed = _parse_axis_arg(found_axis)
        if parsed is None:
            return ctx.default_return_type
        if isinstance(parsed, str):
            axis = None
        elif len(parsed) == 1:
            axis = parsed[0]
        else:
            return ctx.default_return_type
    count = _repeat_count(found_repeats)
    if count is None:
        return ctx.default_return_type
    if isinstance(count, int) and count < 0:
        ctx.api.fail(
            f"repeat repeats must be non-negative, got {count}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    # A multi-entry display fixes the new size outright (or proves the
    # call raises); a single entry broadcasts and stays unknown below.
    size: int | str = "bail"
    entries = _repeat_display(found_repeats)
    if entries is not None and len(entries) != 1:
        checked = _repeat_display_size(ctx, dims, axis, entries)
        if checked is None:
            return ctx.default_return_type
        size = checked
    if axis is None:
        if isinstance(size, int):
            return _ndarray_type(ctx, [size], dtype)
        if isinstance(count, int):
            return _ndarray_type(
                ctx, [_mul_dims(ctx, "repeat", 0, [*dims, count])], dtype
            )
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "repeat", 0)], dtype)
    target = _normalize_axis(axis, len(dims))
    if target is None:
        ctx.api.fail(
            f"repeat axis {axis} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims)
    if isinstance(size, int):
        out[target] = size
    elif isinstance(count, int):
        out[target] = _dim_mul_const(ctx, "repeat", target, out[target], count)
    else:
        out[target] = _fresh_unknown(ctx, "repeat", target)
    return _ndarray_type(ctx, out, dtype)


def repeat_method_hook(ctx: MethodContext) -> Type:
    """ndarray.repeat twin of np.repeat (issue #58).

    Self is the array; repeats/axis come from formals or positionals.
    Only the common scalar-repeats path solves precisely (like the
    function hook); displays and dynamic counts use the same helpers,
    anything else bails to the stubs.
    """
    parts = _ndarray_parts(ctx.type)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    repeats_pair = _formal_pair(ctx, "repeats")
    axis_pair = _formal_pair(ctx, "axis")
    pairs = _all_positionals(ctx)
    if repeats_pair is None:
        # repeats is positional-only (def repeat(self, repeats, /, ...)):
        # callee_arg_names holds None there, so the formal lookup misses
        # even though the positional is visible. Fall back to it.
        repeats_pair = pairs[0] if pairs else None
    if repeats_pair is None:
        return ctx.default_return_type
    if axis_pair is None and _positionals_only(ctx):
        axis_pair = pairs[1] if len(pairs) > 1 else None
    axis: int | None = None
    if axis_pair is not None:
        parsed = _parse_axis_arg(axis_pair)
        if parsed is None:
            return ctx.default_return_type
        if isinstance(parsed, str):
            axis = None
        elif len(parsed) == 1:
            axis = parsed[0]
        else:
            return ctx.default_return_type
    count = _repeat_count(repeats_pair)
    if count is None:
        return ctx.default_return_type
    if isinstance(count, int) and count < 0:
        ctx.api.fail(
            f"repeat repeats must be non-negative, got {count}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    size: int | str = "bail"
    entries = _repeat_display(repeats_pair)
    if entries is not None and len(entries) != 1:
        checked = _repeat_display_size(ctx, dims, axis, entries)
        if checked is None:
            return ctx.default_return_type
        size = checked
    if axis is None:
        if isinstance(size, int):
            return _ndarray_type(ctx, [size], dtype)
        if isinstance(count, int):
            return _ndarray_type(
                ctx, [_mul_dims(ctx, "repeat", 0, [*dims, count])], dtype
            )
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "repeat", 0)], dtype)
    target = _normalize_axis(axis, len(dims))
    if target is None:
        ctx.api.fail(
            f"repeat axis {axis} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims)
    if isinstance(size, int):
        out[target] = size
    elif isinstance(count, int):
        out[target] = _dim_mul_const(ctx, "repeat", target, out[target], count)
    else:
        out[target] = _fresh_unknown(ctx, "repeat", target)
    return _ndarray_type(ctx, out, dtype)


def _pad_side(expr: Expression, tp: Type | None) -> int | None | str:
    """A pad width side: int, None when unknown, 'bad' when garbage."""
    value = _fold_int(expr)
    if value is not None:
        return value
    if tp is not None:
        if _is_none_type(tp):
            return "bad"
        dim = _dim_value(tp)
        if dim is None:
            proper = get_proper_type(tp)
            if (
                isinstance(proper, Instance)
                and proper.type.fullname == "builtins.int"
            ) or isinstance(proper, TypeVarType):
                return None
    return "bad"


def _pad_widths(
    pair: tuple[Type, Expression | None] | None, ndim: int
) -> list[tuple[int | None, int | None]] | str | None:
    """Per-axis (before, after) pad widths (None = unknown side).

    Returns "mismatch" when the entry count fits neither one axis nor all
    of them, None to bail.
    """
    if pair is None:
        return None
    tp, expr = pair
    if _is_none_type(tp):
        return None

    def entry(e: Expression, t: Type | None) -> tuple[int | None, int | None] | str:
        if isinstance(e, (TupleExpr, ListExpr)) and len(e.items) == 2:
            proper = get_proper_type(t) if t is not None else None
            subs: list[Type | None]
            if isinstance(proper, TupleType) and len(proper.items) == 2:
                subs = list(proper.items)
            else:
                subs = [None, None]
            before = _pad_side(e.items[0], subs[0])
            after = _pad_side(e.items[1], subs[1])
            if before == "bad" or after == "bad":
                return "bad"
            assert before != "bad" and after != "bad"
            return (before, after)
        single = _pad_side(e, t)
        if single == "bad":
            return "bad"
        assert single != "bad"
        return (single, single)

    def collect(
        raw: list[tuple[int | None, int | None] | str],
    ) -> list[tuple[int | None, int | None]] | str | None:
        good: list[tuple[int | None, int | None]] = []
        for item in raw:
            if item == "bad":
                return None
            assert not isinstance(item, str)
            good.append(item)
        if len(good) == 1:
            return good * ndim
        if len(good) != ndim:
            return "mismatch"
        return good

    if expr is not None and isinstance(expr, (TupleExpr, ListExpr)):
        proper = get_proper_type(tp)
        types: list[Type | None]
        if isinstance(proper, TupleType) and len(proper.items) == len(expr.items):
            types = list(proper.items)
        else:
            types = [None] * len(expr.items)
        if len(expr.items) == 2 and not any(
            isinstance(e, (TupleExpr, ListExpr)) for e in expr.items
        ):
            return collect([entry(expr, None)])
        return collect([entry(e, t) for e, t in zip(expr.items, types)])
    proper = get_proper_type(tp)
    if isinstance(proper, TupleType):
        nested = [
            isinstance(get_proper_type(item), TupleType) for item in proper.items
        ]
        if len(proper.items) == 2 and not any(nested):
            return collect([_pad_side_fallback(proper)])
        return collect([_pad_side_fallback(item) for item in proper.items])
    value = _int_value(tp)
    if value is None and expr is not None:
        value = _fold_int(expr)
    if isinstance(value, int) and not isinstance(value, bool):
        return [(value, value)] * ndim
    return None


def _pad_side_fallback(tp: Type) -> tuple[int | None, int | None] | str:
    """A pad width entry from its type only."""
    proper = get_proper_type(tp)
    if isinstance(proper, TupleType) and len(proper.items) == 2:
        before = _pad_side_fallback_single(proper.items[0])
        after = _pad_side_fallback_single(proper.items[1])
        if before == "bad" or after == "bad":
            return "bad"
        assert before != "bad" and after != "bad"
        return (before, after)
    single = _pad_side_fallback_single(tp)
    if single == "bad":
        return "bad"
    assert single != "bad"
    return (single, single)


def _pad_side_fallback_single(tp: Type) -> int | None | str:
    if _is_none_type(tp):
        return "bad"
    dim = _dim_value(tp)
    if isinstance(dim, int) and not isinstance(dim, bool):
        return dim
    if dim is not None:
        return None
    proper = get_proper_type(tp)
    if (
        isinstance(proper, Instance) and proper.type.fullname == "builtins.int"
    ) or isinstance(proper, TypeVarType):
        return None
    return "bad"


def pad_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    found = _hook_pair(ctx, "array", "pad_width", 1)
    if found == "incomplete" or found is None:
        return ctx.default_return_type
    assert not isinstance(found, str)
    widths = _pad_widths(found, len(dims))
    if widths is None:
        return ctx.default_return_type
    if widths == "mismatch":
        ctx.api.fail(
            f"pad pad_width entries do not match {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    assert isinstance(widths, list)
    out: list[int | Type] = []
    for i, (d, (before, after)) in enumerate(zip(dims, widths)):
        if before is not None and before < 0 or after is not None and after < 0:
            ctx.api.fail(
                "pad pad_width entries must be non-negative, "
                f"got ({before}, {after})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        if isinstance(before, int) and isinstance(after, int):
            out.append(_dim_add_const(ctx, "pad", i, d, before + after))
        else:
            out.append(_fresh_unknown(ctx, "pad", i))
    return _ndarray_type(ctx, out, dtype)


def split_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    found_axis = _hook_pair(ctx, "ary", "axis", 2)
    found_sections = _hook_pair(ctx, "ary", "indices_or_sections", 1)
    if found_axis == "incomplete" or found_sections == "incomplete":
        return ctx.default_return_type
    assert not isinstance(found_axis, str) and not isinstance(found_sections, str)
    axis = 0
    if found_axis is not None:
        folded = _int_or_none(found_axis)
        if folded == "bail" or folded is None:
            return ctx.default_return_type
        assert isinstance(folded, int)
        axis = folded
    target = _normalize_axis(axis, len(dims))
    if target is None:
        ctx.api.fail(
            f"split axis {axis} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims)
    if found_sections is None:
        return ctx.default_return_type
    sections = _int_or_none(found_sections)
    if isinstance(sections, int):
        if sections < 1:
            ctx.api.fail(
                f"split sections must be positive, got {sections}",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        dim = out[target]
        if isinstance(dim, int):
            if dim % sections != 0:
                ctx.api.fail(
                    f"split cannot divide axis {target} of size {dim} into "
                    f"{sections} equal sections (shape {_dims_str(dims)})",
                    ctx.context,
                    code=NUMTY_SHAPE,
                )
                return ctx.default_return_type
            out[target] = dim // sections
        else:
            out[target] = _dim_div(ctx, "split", target, dim, sections, 0)
    elif sections == "bail":
        seq = _fft_seq_arg(found_sections)
        if seq == "bail" or isinstance(seq, str):
            proper = get_proper_type(found_sections[0])
            if (
                isinstance(proper, Instance)
                and proper.type.fullname == "builtins.int"
            ) or isinstance(proper, TypeVarType):
                out[target] = _fresh_unknown(ctx, "split", target)
            else:
                return ctx.default_return_type
        else:
            out[target] = _fresh_unknown(ctx, "split", target)
    else:
        return ctx.default_return_type
    element = _ndarray_type(ctx, out, dtype)
    return ctx.api.named_generic_type("builtins.list", [element])


def _select_index_elem(
    ctx: FunctionContext, op: str, index: int
) -> Type | None:
    """A 1-d int64 ?U index array, else None to bail."""
    dtype = _construct_dtype(ctx, "numpy.int64")
    if dtype is None:
        return None
    return _ndarray_type(ctx, [_fresh_unknown(ctx, op, index)], dtype)


def _nonzero_like(ctx: FunctionContext, op: str, arg: Type) -> Type:
    """Tuple of 1-d int ?U arrays, one per dim; 0-d fails loudly."""
    parts = _ndarray_parts(arg)
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    if not dims:
        ctx.api.fail(
            f"{op} on 0d arrays is not allowed (use atleast_1d first)",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    elems: list[Type] = []
    for i in range(len(dims)):
        elem = _select_index_elem(ctx, op, i)
        if elem is None:
            return ctx.default_return_type
        elems.append(elem)
    return _linalg_tuple(ctx, elems)


def nonzero_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _nonzero_like(ctx, "nonzero", pairs[0][0])


def flatnonzero_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    elem = _select_index_elem(ctx, "flatnonzero", 0)
    if elem is None:
        return ctx.default_return_type
    return elem


def argwhere_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    dtype = _construct_dtype(ctx, "numpy.int64")
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, [_fresh_unknown(ctx, "argwhere", 0), len(dims)], dtype)


def unique_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    if not _picture_complete(ctx, "ar"):
        return ctx.default_return_type
    for name, index in (
        ("return_index", 1),
        ("return_inverse", 2),
        ("return_counts", 3),
    ):
        found = _hook_pair(ctx, "ar", name, index)
        if found == "incomplete":
            return ctx.default_return_type
        assert not isinstance(found, str)
        if found is None:
            continue
        if _fold_flag(found) is not False:
            return ctx.default_return_type
    found_axis = _hook_pair(ctx, "ar", "axis", 4)
    if found_axis == "incomplete":
        return ctx.default_return_type
    assert not isinstance(found_axis, str)
    if found_axis is None:
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "unique", 0)], dtype)
    folded = _int_or_none(found_axis)
    if folded == "bail":
        return ctx.default_return_type
    if folded is None:
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "unique", 0)], dtype)
    assert isinstance(folded, int)
    target = _normalize_axis(folded, len(dims))
    if target is None:
        ctx.api.fail(
            f"unique axis {folded} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims)
    out[target] = _fresh_unknown(ctx, "unique", target)
    return _ndarray_type(ctx, out, dtype)


def isin_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    dtype = _construct_dtype(ctx, "numpy.bool")
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, list(dims), dtype)


def _search_values_dims(
    tp: Type, expr: Expression | None
) -> list[int | Type] | None:
    """Shape of searchsorted values: [] scalar, dims otherwise, None to bail."""
    if expr is not None and isinstance(expr, (TupleExpr, ListExpr)):
        if not expr.items:
            return [0]
        subs: list[list[int | Type]] = []
        for item in expr.items:
            if isinstance(item, StarExpr):
                return None
            if isinstance(item, (TupleExpr, ListExpr)):
                sub = _search_values_dims(None, item)
            else:
                sub = []
            if sub is None:
                return None
            subs.append(sub)
        if any(sub != subs[0] for sub in subs[1:]):
            return None
        return [len(expr.items), *subs[0]]
    parts = _ndarray_parts(tp)
    if parts is not None:
        return list(parts[0])
    proper = get_proper_type(tp)
    if not isinstance(proper, Instance):
        return None
    if proper.type.fullname in (
        "builtins.int",
        "builtins.float",
        "builtins.complex",
        "builtins.bool",
    ):
        return []
    if proper.type.fullname in _DTYPE_INFO:
        return []
    return None


def searchsorted_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    values_dims = _search_values_dims(pairs[1][0], pairs[1][1])
    if values_dims is None:
        return ctx.default_return_type
    dtype = _construct_dtype(ctx, "numpy.int64")
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, values_dims, dtype)


def sliding_window_view_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    ndim = len(dims)
    found_w = _hook_pair(ctx, "x", "window_shape", 1)
    found_ax = _hook_pair(ctx, "x", "axis", 2)
    if found_w == "incomplete" or found_ax == "incomplete":
        return ctx.default_return_type
    assert not isinstance(found_w, str) and not isinstance(found_ax, str)
    if found_w is None:
        return ctx.default_return_type
    window = _int_or_none(found_w)
    widths: list[int] | None = None
    if isinstance(window, int):
        widths = [window]
        scalar_window = True
    elif window == "bail":
        seq = _fft_seq_arg(found_w)
        if not isinstance(seq, list):
            return ctx.default_return_type
        items: list[int] = []
        for entry in seq:
            if not isinstance(entry, int):
                return ctx.default_return_type
            items.append(entry)
        widths = items
        scalar_window = False
    else:
        return ctx.default_return_type
    axes: list[int] | None = None
    if found_ax is not None and not _is_none_type(found_ax[0]):
        axis_val = _int_or_none(found_ax)
        if isinstance(axis_val, int):
            axes = [axis_val]
        elif axis_val == "bail":
            seq = _fft_seq_arg(found_ax)
            if not isinstance(seq, list):
                return ctx.default_return_type
            items = []
            for entry in seq:
                if not isinstance(entry, int):
                    return ctx.default_return_type
                items.append(entry)
            axes = items
        else:
            axes = None
    if axes is None:
        # A scalar window reads as one entry, which only fits 1-d input;
        # anything else must spell every dimension.
        if len(widths) != ndim:
            ctx.api.fail(
                "sliding_window_view with axis=None must provide window_shape "
                f"for all {ndim} dimensions (shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        eff = list(range(ndim))
    else:
        eff = []
        for ax in axes:
            target = _normalize_axis(ax, ndim)
            if target is None:
                ctx.api.fail(
                    f"sliding_window_view axis {ax} is out of range for "
                    f"{ndim} dimensions (shape {_dims_str(dims)})",
                    ctx.context,
                    code=NUMTY_AXIS,
                )
                return ctx.default_return_type
            eff.append(target)
        if len(set(eff)) != len(eff):
            return ctx.default_return_type
        if scalar_window:
            widths = widths * len(eff)
        elif len(widths) != len(eff):
            ctx.api.fail(
                "sliding_window_view must provide matching length window_shape "
                f"and axis (got {len(widths)} window entries for {len(eff)} axes)",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
    out = list(dims)
    for ax, wval in zip(eff, widths):
        if wval < 0:
            ctx.api.fail(
                "sliding_window_view `window_shape` cannot contain "
                f"negative values (got {wval})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        d = out[ax]
        if isinstance(d, int) and wval > d:
            ctx.api.fail(
                "sliding_window_view window shape cannot be larger than "
                f"input array shape (window {wval} over dim {d} of shape "
                f"{_dims_str(dims)})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        out[ax] = _dim_add_const(ctx, "sliding_window_view", ax, d, 1 - wval)
    out.extend(widths)
    return _ndarray_type(ctx, out, dtype)


def atleast_function_hook(ctx: FunctionContext, rank: int) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) != 1:
        # Multi-arg form returns a tuple the stubs own.
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    if len(dims) >= rank:
        return _ndarray_type(ctx, list(dims), dtype)
    out: list[int | Type]
    if rank == 1:
        out = [1]
    elif rank == 2:
        out = [1, 1] if not dims else [1, *dims]
    elif not dims:
        out = [1, 1, 1]
    elif len(dims) == 1:
        out = [1, dims[0], 1]
    else:
        # rank == 3 with 2-D input: numpy appends the new axis.
        out = [*dims, 1]
    return _ndarray_type(ctx, out, dtype)


def _dims_str(dims: list[int | Type]) -> str:
    """Shape for messages: tuple-style, symbolic dims by name."""
    inner = ", ".join(_dim_str(d) for d in dims)
    if len(dims) == 1:
        inner += ","
    return f"({inner})"


def _broadcast_result(
    a: list[int | Type],
    b: list[int | Type],
    ctx: FunctionContext | MethodContext | None = None,
) -> tuple[list[int | Type] | None, str | None]:
    """Broadcast two known shapes; (None, None) to bail; (None, msg) on error.

    Same-name TypeVars are equal. Anything else unprovable keeps the
    axis instead of bailing the whole shape (issue #56): a literal
    against a symbolic dim is that literal (if the broadcast succeeds
    the axis is that size), two different symbolics become a fresh
    unknown. Without ctx the old bail is kept (callers without shape
    context); every ctx-bearing caller passes it.
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
        elif ctx is None:
            return None, None
        elif isinstance(x, int) and isinstance(y, TypeVarType):
            out.append(x)
        elif isinstance(y, int) and isinstance(x, TypeVarType):
            out.append(y)
        else:
            out.append(_fresh_unknown(ctx, "bcast", k))
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
            ctx.api.fail(message, ctx.context, code=NUMTY_BROADCAST)
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
    dims, message = _broadcast_result(dims_a, dims_b, ctx)
    if message is not None:
        ctx.api.fail(message, ctx.context, code=NUMTY_BROADCAST)
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


_unknown_ids: dict[str, TypeVarId] = {}
# Display names of unknowns by fullname, and how many distinct unknowns
# each (module, line) has minted. Names only: identity stays the id.
_unknown_names: dict[str, str] = {}
_unknown_per_line: dict[tuple[str, int], int] = {}


def _unknown_display_name(module: str, line: int, fullname: str) -> str:
    """Readable, per-site name for an unknown: ?U@14, then ?U@14b, ...

    Named after the line that minted it, so unrelated unknowns print
    differently and the name points at its source; a second distinct
    site on the same line gets a letter suffix. Stable per site, like
    the id, so re-checks and overload candidates print identically.
    """
    name = _unknown_names.get(fullname)
    if name is None:
        n = _unknown_per_line.get((module, line), 0)
        _unknown_per_line[(module, line)] = n + 1
        if n == 0:
            suffix = ""
        elif n < 26:
            suffix = chr(ord("a") + n)
        else:
            suffix = f"_{n + 1}"
        name = f"?U@{line}{suffix}"
        _unknown_names[fullname] = name
    return name


def _fresh_unknown(
    ctx: FunctionContext | MethodContext, role: str, index: int
) -> TypeVarType:
    """A fresh unknown dim, stable per source site across passes.

    Data-dependent sizes (mask counts, dynamic lengths, uninferable
    reshapes) cannot be spelled, but their positions are known: mint a
    variable for the position instead of failing or bailing. Site-keyed
    (target + line + column + role + index) so overload candidates and
    re-checks mint the identical variable — differing ids would look
    like disagreement to overload resolution and collapse to Any.
    Unknowns unify with anything, so they dissolve precision but can
    never produce a false error; ranks and known dims keep checking
    around them.
    """
    tree = getattr(ctx.api, "tree", None)
    module = getattr(tree, "fullname", "") or ""
    line = ctx.context.line
    column = ctx.context.column
    key = f"{module}_{line}_{column}_{role}_{index}"
    vid = _unknown_ids.get(key)
    if vid is None:
        vid = TypeVarId.new(0)
        _unknown_ids[key] = vid
    fullname = f"__numty_unknown__.{key}"
    upper = ctx.api.named_generic_type("builtins.int", [])
    return TypeVarType(
        name=_unknown_display_name(module, line, fullname),
        fullname=fullname,
        id=vid,
        values=[],
        upper_bound=upper,
        default=AnyType(TypeOfAny.from_omitted_generics),
        line=line,
        column=column,
    )


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
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    if known == 0 and total != 0:
        ctx.api.fail(
            f"reshape changes total size: 0 -> {total} "
            f"(shapes {_dims_str(old)} vs {tuple(raw)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    return ctx.default_return_type


def _reshape_dynamic_dims(
    pairs: list[tuple[Type, Expression | None]],
) -> list[int | Type] | None:
    """Target dims of a reshape, plain `int` sizes as types, else None.

    reshape(n, 3), reshape((n, 3)) and reshape(x.shape) with n: int: the
    sizes the caller spells are the result's, whatever the input, so the
    literal ones are kept next to plain `int`s instead of the stubs'
    all-int answer (#34). Same argument forms as _reshape_new_dims, plus
    a tuple-typed shape value; None when anything is unproven.
    """
    if not pairs:
        return None
    first_tp, first_expr = pairs[0]
    first = get_proper_type(first_tp)
    entries: list[tuple[Type, Expression | None]]
    if isinstance(first_expr, TupleExpr):
        if not isinstance(first, TupleType) or len(first.items) != len(first_expr.items):
            return None
        entries = list(zip(first.items, first_expr.items))
    elif isinstance(first, TupleType):
        entries = [(item, None) for item in first.items]
    else:
        entries = []
        for tp, expr in pairs:
            proper = get_proper_type(tp)
            if isinstance(proper, Instance) and proper.type.fullname == STR_FULLNAME:
                break
            entries.append((tp, expr))
    dims: list[int | Type] = []
    for tp, expr in entries:
        if isinstance(expr, StarExpr) or isinstance(get_proper_type(tp), UnpackType):
            return None
        value = _fold_int(expr) if expr is not None else None
        dim = value if value is not None else _dim_value(tp)
        if dim is not None:
            dims.append(dim)
            continue
        plain = get_proper_type(tp)
        if not (isinstance(plain, Instance) and plain.type.fullname == INT_FULLNAME):
            return None
        dims.append(plain.copy_modified(last_known_value=None))
    return dims


def _solve_reshape_dynamic(
    ctx: FunctionContext | MethodContext, old: Type, dims: list[int | Type]
) -> Type:
    """The requested shape of a reshape that names a plain `int` size or
    whose input shape the exact paths cannot read (_solve_reshape).

    Holds by construction whenever the call is legal at runtime. Nothing
    is reported here, although some illegal calls are provable (a known
    total of 12 cannot reshape to (n, 5)); the exact paths miss those
    too. -1 next to only literal sizes is the input's size over theirs,
    a plain `int` like the input's dynamic dims ((n, 3) -> (-1, 3) is
    (int, 3)). Next to a plain `int` size it is a ratio the caller may
    know to be fixed ((n, 3) -> (n, -1) is (n, 3) again), so it is
    unknown (Any), which fits where a literal was expected. Plain `int`
    target sizes go unknown where they may stand for a literal the name
    was first bound with: on a mixed input's literal axes, and anywhere
    when the input has no literal axis to line them up with. With
    neither a literal nor a plain `int` size (reshape(-1), TypeVar
    sizes), the stubs' answer stays.
    """
    proper = get_proper_type(old)
    if not (
        isinstance(proper, Instance)
        and proper.type.fullname == NDARRAY_FULLNAME
        and len(proper.args) == 2
    ):
        return ctx.default_return_type
    ints = [d for d in dims if isinstance(d, int)]
    if any(d < -1 for d in ints) or ints.count(-1) > 1:
        return ctx.default_return_type
    inferred: Type = ctx.api.named_generic_type(INT_FULLNAME, [])
    plain = any(_is_plain_int(d) for d in dims)
    if plain:
        inferred = AnyType(TypeOfAny.explicit)
    if not plain and not any(d >= 0 for d in ints):
        return ctx.default_return_type
    out = [inferred if d == -1 else d for d in dims]
    if not _is_mixed_shape(proper):
        # No literal axis to line the sizes up with (x.ravel().reshape(n,
        # d), a 1-D source): any plain `int` size may be the literal the
        # name was first bound with, so it is unknown.
        out = [AnyType(TypeOfAny.explicit) if _is_plain_int(d) else d for d in out]
    # A plain `int` target size on an axis where a mixed input had a
    # literal ((n, 3) -> (-1, d)) may well be that literal: unknown, as
    # for contractions, so x = x.reshape(-1, d) keeps fitting x.
    return _mixed_result(_ndarray_type(ctx, out, proper.args[1]), old)


def _solve_reshape(
    ctx: FunctionContext | MethodContext, old: Type, pairs: list[tuple[Type, Expression | None]]
) -> Type:
    # A plain `int` target size, or an input whose shape the exact paths
    # below cannot read (mixed, all-int): keep the requested literals.
    dynamic = _reshape_dynamic_dims(pairs)
    if dynamic is not None and (
        any(_is_plain_int(d) for d in dynamic)
        or (_ndarray_parts(old) is None and _unpack_shape(old) is None)
    ):
        return _solve_reshape_dynamic(ctx, old, dynamic)
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
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        if raw_ints.count(-1) > 1:
            ctx.api.fail(
                f"reshape accepts only one -1 dimension (shape {_dims_str(raw)})",
                ctx.context,
                code=NUMTY_SHAPE,
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
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    if raw_ints.count(-1) > 1:
        ctx.api.fail(
            f"reshape accepts only one -1 dimension (shape {_dims_str(raw)})",
            ctx.context,
            code=NUMTY_SHAPE,
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
        # Exact division of the canonical total by the known divisor
        # (ints and symbolic dims alike): N*2/2, N*K/N=K, B*C*H*W/B=C*H*W.
        # Beats the ?U below whenever exactly divisible.
        old_form = _linear_product(dims_old)
        divisor_form = _linear_product([d for d in raw if d != -1])
        quot = (
            _canon_div(old_form, divisor_form)
            if old_form is not None and divisor_form is not None
            else None
        )
        if quot is not None:
            inferred = _materialize_dim(ctx, "reshape", 0, quot)
            return _ndarray_type(
                ctx,
                [inferred if d == -1 else d for d in raw],
                dtype,
            )
        if old_vars != new_vars:
            # The inferred dim would absorb a symbolic size (N * 50 is not
            # expressible): track it with a fresh unknown instead of
            # bailing. Requested ints stay (they hold by construction);
            # only the -1 is unknowable.
            return _ndarray_type(
                ctx,
                [_fresh_unknown(ctx, "reshape", j) if d == -1 else d for j, d in enumerate(raw)],
                dtype,
            )
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
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        inferred = size_old // known
        dims_new = [inferred if d == -1 else d for d in raw]
    else:
        size_new = 1
        for d in raw_ints:
            size_new *= d
        old_form = _linear_product(dims_old)
        new_form = _linear_product(raw)
        forms_equal = (
            old_form is not None
            and new_form is not None
            and old_form == new_form
        )
        # Canonical equality only ever silences: N*2 vs 2*N is proven, and
        # every loud below also fired before (or shows the same numbers).
        if size_new != size_old and not forms_equal:
            if old_form is not None and new_form is not None:
                ctx.api.fail(
                    f"reshape changes total size: {_expr_str(old_form)} -> "
                    f"{_expr_str(new_form)} "
                    f"(shapes {_dims_str(dims_old)} vs {raw_str})",
                    ctx.context,
                    code=NUMTY_SHAPE,
                )
            else:
                ctx.api.fail(
                    f"reshape changes total size: {size_old} -> {size_new} "
                    f"(shapes {_dims_str(dims_old)} vs {raw_str})",
                    ctx.context,
                    code=NUMTY_SHAPE,
                )
            return ctx.default_return_type
        if old_vars != new_vars:
            # Same int total but different symbolic factors (N vs M): the
            # equality is unprovable, but the requested shape holds by
            # construction when it is legal at all, so track it instead
            # of bailing. Legality itself stays unchecked, never an error.
            return _ndarray_type(ctx, list(raw), dtype)
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
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        normalized.append(norm)
    if sorted(normalized) != list(range(ndim)):
        ctx.api.fail(
            f"transpose axes {tuple(axes)} are not a permutation of {ndim} "
            f"dimensions (shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
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
    if not _check_unique_axes(ctx, dims, axis, "squeeze"):
        return ctx.default_return_type
    chosen: list[int] = []
    for ax in axis:
        norm = _normalize_axis(ax, ndim)
        if norm is None:
            ctx.api.fail(
                f"squeeze axis {ax} is out of range for {ndim} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        if not isinstance(dims[norm], int):
            return ctx.default_return_type
        if dims[norm] != 1:
            ctx.api.fail(
                f"squeeze axis {ax} has size {dims[norm]}, not 1 "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_SHAPE,
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
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        if norm in normalized:
            ctx.api.fail(f"expand_dims has duplicate axis {ax}", ctx.context, code=NUMTY_AXIS)
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
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        if norm in normalized:
            ctx.api.fail(f"duplicate axis {ax}", ctx.context, code=NUMTY_AXIS)
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
        override = _builder_shape(*shape_pair, ctx.default_return_type, shaped_stub=False)
        if override is None:
            return ctx.default_return_type
        dims = override
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
        if named is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, named)
    return _ndarray_type(ctx, dims, dtype)


def copy_function_hook(ctx: FunctionContext) -> Type:
    """np.copy: same shape and dtype (issue #58).

    Trivial keep-shape twin of the hooked method (m.copy() already
    solved); previously fell back to Any and accepted any annotation.
    """
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _parts_or_unpack(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    return _ndarray_type(ctx, list(dims), dtype)


def ascontiguousarray_function_hook(ctx: FunctionContext) -> Type:
    """np.ascontiguousarray: same shape, explicit dtype or kept (issue #58)."""
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _parts_or_unpack(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
        if named is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, list(dims), named)
    return _ndarray_type(ctx, list(dims), dtype)


def full_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    shape_pair = _formal_pair(ctx, "shape") or (pairs[0] if pairs else None)
    fill_pair = _formal_pair(ctx, "fill_value")
    if shape_pair is None or fill_pair is None:
        return ctx.default_return_type
    dims = _builder_shape(*shape_pair, ctx.default_return_type)
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
            code=NUMTY_SHAPE,
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


def _builder_single_dim(
    pair: tuple[Type, Expression | None] | None,
) -> int | TypeVarType | None:
    """One builder dim: int literal or symbolic TypeVar, else None to bail."""
    if pair is None:
        return None
    folded = _fold_single(pair)
    if folded is not None:
        return folded
    dim = _dim_value(pair[0])
    if isinstance(dim, TypeVarType):
        return dim
    return None


def linspace_function_hook(ctx: FunctionContext) -> Type:
    """np.linspace: (num,) with explicit dtype or float64 (issue #58).

    Rank-only stubs reject correct Literal annotations, so solve the
    count exactly. Symbolic counts propagate; anything else bails.
    """
    num_pair = _formal_pair(ctx, "num")
    if num_pair is None:
        pairs = _all_positionals(ctx)
        if len(pairs) >= 3:
            num_pair = pairs[2]
        else:
            # Default num=50.
            num_pair = None
    if num_pair is None:
        num: int | TypeVarType | None = 50
    else:
        num = _builder_single_dim(num_pair)
        if num is None:
            return ctx.default_return_type
    if isinstance(num, int) and num < 0:
        ctx.api.fail(
            f"linspace num must be non-negative, got {num}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
        if named is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, [num], named)
    dtype = _construct_dtype(ctx, "numpy.float64")
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, [num], dtype)


def identity_function_hook(ctx: FunctionContext) -> Type:
    """np.identity: (n, n) with explicit dtype or float64 (issue #58)."""
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    n = _builder_single_dim(pairs[0])
    if n is None:
        return ctx.default_return_type
    if isinstance(n, int) and n < 0:
        ctx.api.fail(
            f"identity dimensions must be non-negative, got ({n}, {n})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
        if named is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, [n, n], named)
    dtype = _construct_dtype(ctx, "numpy.float64")
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, [n, n], dtype)


def tri_function_hook(ctx: FunctionContext) -> Type:
    """np.tri: (N, M or N) with explicit dtype or float64 (issue #58)."""
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    n = _builder_single_dim(pairs[0])
    if n is None:
        return ctx.default_return_type
    m_pair = _formal_pair(ctx, "M")
    if m_pair is None or _is_none_type(m_pair[0]):
        if len(pairs) >= 2:
            # Positional M.
            m_pair = pairs[1]
        else:
            m: int | TypeVarType | None = n
            m_pair = None
    if m_pair is not None:
        m = _builder_single_dim(m_pair)
        if m is None:
            return ctx.default_return_type
    assert m is not None
    if isinstance(n, int) and isinstance(m, int) and (n < 0 or m < 0):
        ctx.api.fail(
            f"tri dimensions must be non-negative, got ({n}, {m})",
            ctx.context,
            code=NUMTY_SHAPE,
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
            ctx.api.fail("arange step is zero", ctx.context, code=NUMTY_SHAPE)
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
                # Plain Python list-of-ints (e.g. [0] in
                # concatenate([[0], cumsum(lengths)]), the N+1 offsets
                # idiom from issue #32): numpy reads it as a 1-D array
                # of that length. Only flat int lists; nested bails.
                if isinstance(item_expr, ListExpr) and all(
                    isinstance(e, IntExpr) for e in item_expr.items
                ):
                    dtype = _construct_dtype(ctx, "numpy.int64")
                    if dtype is None:
                        return None
                    items.append(([len(item_expr.items)], dtype))
                    continue
                return None
            dims, dtype = parts
            items.append((list(dims), dtype))
        return items or None
    return None


def _is_dynamic_sequence(tp: Type, expr: Expression | None) -> bool:
    """A sequence whose length is unknowable statically.

    A list-typed name or attribute (built dynamically, e.g. by appending
    in a loop), a generator, or a non-enum comprehension. Fixed-length
    forms (tuple displays and types, list literals, enum comprehensions)
    are solved elsewhere, so they never reach this check.
    """
    if isinstance(expr, (GeneratorExpr, ListComprehension)):
        return True
    if not isinstance(expr, (NameExpr, MemberExpr)):
        return False
    proper = get_proper_type(tp)
    return (
        isinstance(proper, Instance)
        and proper.type.fullname == "builtins.list"
    )


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


def _element_expr_type(ctx: FunctionContext, expr: Expression) -> Type | None:
    """Type of a comprehension's element expression, else None.

    Re-checked without type context first, so every overload candidate
    reads the same item (the stored type was inferred against that
    candidate's formal and can differ, e.g. astype's dtype). The
    stored type is the fallback: it keeps the if-clause narrowing
    (`a for a in xs if a is not None`) that a re-check loses.
    """
    lookup = getattr(ctx.api, "lookup_type_or_none", None)
    # Read before the re-check, which overwrites the stored type.
    stored = lookup(expr) if lookup is not None else None
    try:
        tp: Type | None = ctx.api.get_expression_type(expr)
    except Exception:
        tp = None
    if tp is not None and _ndarray_parts(tp) is not None:
        return tp
    return stored if stored is not None else tp


def _dynamic_sequence_element(
    ctx: FunctionContext, tp: Type, expr: Expression | None
) -> tuple[list[int | Type], Type] | None:
    """(dims, dtype) of a homogeneous dynamically-built sequence's items.

    A loop-built list[T] or a list comprehension has unknowable length
    but one item type. A comprehension arrives joined against the
    ArrayLike formal, so its item type is re-read from the element
    expression. Anything else (bare lists, generators, unknown or
    mixed elements) stays unprovable.
    """
    if isinstance(expr, ListComprehension):
        item = _element_expr_type(ctx, expr.generator.left_expr)
        return _ndarray_parts(item) if item is not None else None
    proper = get_proper_type(tp)
    if not (
        isinstance(proper, Instance)
        and proper.type.fullname == "builtins.list"
        and proper.args
    ):
        return None
    return _ndarray_parts(proper.args[0])


def concatenate_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    items = _sequence_parts(ctx, pairs[0][0], pairs[0][1])
    if items is None:
        if _is_dynamic_sequence(pairs[0][0], pairs[0][1]):
            elem = _dynamic_sequence_element(ctx, pairs[0][0], pairs[0][1])
            if elem is not None:
                return _concat_dynamic(ctx, pairs, elem)
            ctx.api.fail(
                "concatenate needs a sequence of known length "
                "(built dynamically; preallocate or ignore)",
                ctx.context,
                code=NUMTY_BAIL,
            )
            return AnyType(TypeOfAny.from_error)
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
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    ndim = len(items[0][0])
    norm = _normalize_axis(axis, ndim)
    if norm is None:
        ctx.api.fail(
            f"concatenate axis {axis} is out of range for {ndim} dimensions",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    return _concat_merge(ctx, "concatenate", items, dtype, norm)


def _stack_promote(dims: list[int | Type], kind: str) -> list[int | Type]:
    """atleast-style promotion each stacking helper applies per item."""
    if kind in ("vstack", "column_stack"):
        if not dims:
            return [1, 1]
        if len(dims) == 1:
            return [1, *dims] if kind == "vstack" else [*dims, 1]
        return list(dims)
    if kind == "hstack":
        return [1] if not dims else list(dims)
    if not dims:
        return [1, 1, 1]
    if len(dims) == 1:
        return [1, dims[0], 1]
    if len(dims) == 2:
        # dstack appends the new axis: (M, N) -> (M, N, 1), so a dynamic
        # stack gives (M, N, ?U) (numpy: (2, 3, 2), not (1, 2, 3)).
        return [*dims, 1]
    return list(dims)


def _concat_merge(
    ctx: FunctionContext,
    op: str,
    items: list[tuple[list[int | Type], Type]],
    dtype: Type,
    norm: int,
) -> Type:
    """Merge rank-checked items along norm; mismatches fail with op."""
    out = list(items[0][0])
    for dims, _ in items[1:]:
        for i, d in enumerate(dims):
            if i != norm and not _dim_eq(out[i], d):
                if isinstance(out[i], int) and isinstance(d, int):
                    ctx.api.fail(
                        f"{op} dimension {i} differs: {out[i]} vs {d} "
                        f"(shapes {_dims_str(out)} vs {_dims_str(dims)})",
                        ctx.context,
                        code=NUMTY_SHAPE,
                    )
                    return ctx.default_return_type
                return ctx.default_return_type
        if isinstance(out[norm], int) and isinstance(dims[norm], int):
            out[norm] = out[norm] + dims[norm]
        else:
            # Symbolic lengths add exactly (issue #32): 1+N -> N+1,
            # N+M -> N+M via linear forms. Unrepresentable (Div/Mono)
            # mints a fresh unknown, never a bail to Any.
            out[norm] = _dim_add2(ctx, op, norm, out[norm], dims[norm])
    return _ndarray_type(ctx, out, dtype)


def _stack_solve(
    ctx: FunctionContext,
    pairs: list[tuple[Type, Expression | None]],
    kind: str,
    axis: int,
) -> Type:
    """Shared vstack/hstack/dstack/column_stack solver after promotion."""
    items = _sequence_parts(ctx, pairs[0][0], pairs[0][1])
    if items is None:
        if _is_dynamic_sequence(pairs[0][0], pairs[0][1]):
            elem = _dynamic_sequence_element(ctx, pairs[0][0], pairs[0][1])
            if elem is None:
                ctx.api.fail(
                    f"{kind} needs a sequence of known length "
                    "(built dynamically; preallocate or ignore)",
                    ctx.context,
                    code=NUMTY_BAIL,
                )
                return AnyType(TypeOfAny.from_error)
            dims_e, dtype_e = elem
            dtype = _dynamic_explicit_dtype(ctx, dtype_e)
            if dtype is None:
                return ctx.default_return_type
            promoted = _stack_promote(dims_e, kind)
            norm = axis
            if kind == "hstack":
                norm = 0 if len(promoted) == 1 else 1
            out = list(promoted)
            if norm >= len(out):
                return ctx.default_return_type
            out[norm] = _fresh_unknown(ctx, kind, norm)
            return _ndarray_type(ctx, out, dtype)
        return ctx.default_return_type
    if _explicit_dtype(ctx):
        dtype = _named_dtype(ctx)
        if dtype is None:
            return ctx.default_return_type
    else:
        dtype = _promote_all(ctx, [dtype for _, dtype in items])
        if dtype is None:
            return ctx.default_return_type
    promoted = [_stack_promote(dims, kind) for dims, _ in items]
    ranks = {len(dims) for dims in promoted}
    if len(ranks) != 1:
        ctx.api.fail(
            f"{kind} ranks differ: "
            + ", ".join(_dims_str(dims) for dims in promoted),
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    norm = axis
    if kind == "hstack":
        norm = 0 if len(promoted[0]) == 1 else 1
    if norm >= len(promoted[0]):
        return ctx.default_return_type
    return _concat_merge(
        ctx, kind, list(zip(promoted, [dtype] * len(promoted))), dtype, norm
    )


def vstack_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _stack_solve(ctx, pairs, "vstack", 0)


def hstack_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _stack_solve(ctx, pairs, "hstack", 1)


def dstack_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _stack_solve(ctx, pairs, "dstack", 2)


def column_stack_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _stack_solve(ctx, pairs, "column_stack", 1)


def _dynamic_explicit_dtype(
    ctx: FunctionContext, fallback: Type
) -> Type | None:
    """User's explicit dtype when given and resolvable, else fallback."""
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
        if named is None:
            return None
        return named
    return fallback


def _concat_dynamic(
    ctx: FunctionContext,
    pairs: list[tuple[Type, Expression | None]],
    elem: tuple[list[int | Type], Type],
) -> Type:
    """Concatenate of a homogeneous dynamic sequence: rank is known
    from the element type, only the axis length is unknowable."""
    dims_e, dtype_e = elem
    dtype = _dynamic_explicit_dtype(ctx, dtype_e)
    if dtype is None:
        return ctx.default_return_type
    axis = _concat_axis(ctx, pairs)
    if axis is None or not isinstance(axis, int):
        if axis == "none":
            return _ndarray_type(ctx, [_fresh_unknown(ctx, "concat", 0)], dtype)
        return ctx.default_return_type
    norm = _normalize_axis(axis, len(dims_e))
    if norm is None:
        ctx.api.fail(
            f"concatenate axis {axis} is out of range for {len(dims_e)} dimensions",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims_e)
    out[norm] = _fresh_unknown(ctx, "concat", norm)
    return _ndarray_type(ctx, out, dtype)


def _stack_dynamic(
    ctx: FunctionContext,
    pairs: list[tuple[Type, Expression | None]],
    elem: tuple[list[int | Type], Type],
) -> Type:
    """Stack of a homogeneous dynamic sequence: the new axis holds the
    unknowable sequence length, every other dim rides the element."""
    dims_e, dtype_e = elem
    dtype = _dynamic_explicit_dtype(ctx, dtype_e)
    if dtype is None:
        return ctx.default_return_type
    axis = _concat_axis(ctx, pairs)
    if axis is None or axis == "none" or not isinstance(axis, int):
        return ctx.default_return_type
    ndim = len(dims_e)
    norm = axis + ndim + 1 if axis < 0 else axis
    if not 0 <= norm <= ndim:
        ctx.api.fail(
            f"stack axis {axis} is out of range for {ndim} dimensions",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims_e)
    out.insert(norm, _fresh_unknown(ctx, "stack", norm))
    return _ndarray_type(ctx, out, dtype)


def stack_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    items = _sequence_parts(ctx, pairs[0][0], pairs[0][1])
    if items is None:
        if _is_dynamic_sequence(pairs[0][0], pairs[0][1]):
            elem = _dynamic_sequence_element(ctx, pairs[0][0], pairs[0][1])
            if elem is not None:
                return _stack_dynamic(ctx, pairs, elem)
            ctx.api.fail(
                "stack needs a sequence of known length "
                "(built dynamically; preallocate or ignore)",
                ctx.context,
                code=NUMTY_BAIL,
            )
            return AnyType(TypeOfAny.from_error)
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
                    code=NUMTY_SHAPE,
                )
                return ctx.default_return_type
            return ctx.default_return_type
    ndim = len(first)
    norm = axis + ndim + 1 if axis < 0 else axis
    if not 0 <= norm <= ndim:
        ctx.api.fail(
            f"stack axis {axis} is out of range for {ndim} dimensions",
            ctx.context,
            code=NUMTY_AXIS,
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
    parts = _ndarray_parts(ctx.type) or _mixed_parts(ctx.type)
    if parts is not None:
        dims, dtype = parts
        return _solve_transpose(ctx, dims, dtype, _all_positionals(ctx))
    split = _unpack_shape(ctx.type)
    if split is None:
        return ctx.default_return_type
    (prefix, unpack, suffix), dtype = split
    return _solve_transpose_unpack(ctx, prefix, unpack, suffix, dtype, _all_positionals(ctx))


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


def transpose_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0]) or _mixed_parts(pairs[0][0])
    if parts is None:
        split = _unpack_shape(pairs[0][0])
        if split is None:
            if _has_unknown_shape(pairs[0][0]):
                # Transposing an unknown shape leaves it unknown. The
                # stubs' answer may instead be a shape mypy took from the
                # assignment's type context (t: A = np.transpose(
                # np.transpose(x))), which an outer transpose would then
                # reverse into a false error.
                return pairs[0][0]
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


def _swapaxes_solve(
    ctx: FunctionContext | MethodContext,
    dims: list[int | Type],
    dtype: Type,
    axis1: int,
    axis2: int,
) -> Type:
    ndim = len(dims)
    n1 = _normalize_axis(axis1, ndim)
    n2 = _normalize_axis(axis2, ndim)
    if n1 is None:
        ctx.api.fail(
            f"swapaxes axis1 {axis1} is out of range for {ndim} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    if n2 is None:
        ctx.api.fail(
            f"swapaxes axis2 {axis2} is out of range for {ndim} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims)
    out[n1], out[n2] = out[n2], out[n1]
    return _ndarray_type(ctx, out, dtype)


def swapaxes_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) < 3:
        # Try formals: a, axis1, axis2
        if not _picture_complete(ctx, "a"):
            return ctx.default_return_type
        arr_pair = _formal_pair(ctx, "a") or (pairs[0] if pairs else None)
        a1_pair = _formal_pair(ctx, "axis1")
        a2_pair = _formal_pair(ctx, "axis2")
        if a1_pair is None and len(pairs) > 1:
            a1_pair = pairs[1]
        if a2_pair is None and len(pairs) > 2:
            a2_pair = pairs[2]
        if arr_pair is None:
            return ctx.default_return_type
        parts = _ndarray_parts(arr_pair[0])
        if parts is None:
            return ctx.default_return_type
        dims, dtype = parts
        ax1 = _int_or_none(a1_pair)
        ax2 = _int_or_none(a2_pair)
        if ax1 in (None, "bail") or ax2 in (None, "bail"):
            return ctx.default_return_type
        assert isinstance(ax1, int) and isinstance(ax2, int)
        return _swapaxes_solve(ctx, dims, dtype, ax1, ax2)
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    ax1 = _int_or_none(pairs[1])
    ax2 = _int_or_none(pairs[2])
    if ax1 in (None, "bail") or ax2 in (None, "bail"):
        return ctx.default_return_type
    assert isinstance(ax1, int) and isinstance(ax2, int)
    return _swapaxes_solve(ctx, dims, dtype, ax1, ax2)


def swapaxes_method_hook(ctx: MethodContext) -> Type:
    parts = _ndarray_parts(ctx.type)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    ax1 = _int_or_none(pairs[0])
    ax2 = _int_or_none(pairs[1])
    if ax1 in (None, "bail") or ax2 in (None, "bail"):
        return ctx.default_return_type
    assert isinstance(ax1, int) and isinstance(ax2, int)
    return _swapaxes_solve(ctx, dims, dtype, ax1, ax2)


def _moveaxis_permutation(
    ndim: int, source: list[int], dest: list[int]
) -> list[int] | None:
    """Permutation for moveaxis: dest positions hold source axes."""
    if len(source) != len(dest):
        return None
    if len(set(source)) != len(source):
        return None
    if len(set(dest)) != len(dest):
        return None
    # Normalize already done by caller; just validate range here.
    for s in source:
        if not 0 <= s < ndim:
            return None
    for d in dest:
        if not 0 <= d < ndim:
            return None
    res: list[int | None] = [None] * ndim
    for s, d in zip(source, dest):
        if res[d] is not None:
            return None
        res[d] = s
    remaining = [i for i in range(ndim) if i not in source]
    it = iter(remaining)
    for i in range(ndim):
        if res[i] is None:
            try:
                res[i] = next(it)
            except StopIteration:
                return None
    # res now holds original axis indices in new order
    return [int(x) for x in res if x is not None]


def _parse_shape_like(pair: tuple[Type, Expression | None] | None) -> list[int] | None:
    if pair is None:
        return None
    return _shape_from_arg(pair[0], pair[1])


def moveaxis_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    ndim = len(dims)
    # source and destination: formal names source, destination
    src_pair = _formal_pair(ctx, "source")
    dst_pair = _formal_pair(ctx, "destination")
    if src_pair is None and len(pairs) > 1:
        src_pair = pairs[1]
    if dst_pair is None and len(pairs) > 2:
        dst_pair = pairs[2]
    src_list = _parse_shape_like(src_pair)
    dst_list = _parse_shape_like(dst_pair)
    if src_list is None or dst_list is None:
        return ctx.default_return_type
    # Normalize
    norm_src: list[int] = []
    for ax in src_list:
        n = _normalize_axis(ax, ndim)
        if n is None:
            ctx.api.fail(
                f"moveaxis source axis {ax} is out of range for {ndim} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        norm_src.append(n)
    norm_dst: list[int] = []
    for ax in dst_list:
        n = _normalize_axis(ax, ndim)
        if n is None:
            ctx.api.fail(
                f"moveaxis destination axis {ax} is out of range for {ndim} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        norm_dst.append(n)
    perm = _moveaxis_permutation(ndim, norm_src, norm_dst)
    if perm is None:
        ctx.api.fail(
            f"moveaxis source {tuple(src_list)} and destination {tuple(dst_list)} "
            f"must have same length and no duplicates",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = [dims[i] for i in perm]
    return _ndarray_type(ctx, out, dtype)


def rollaxis_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    ndim = len(dims)
    # axis formal
    axis_pair = _formal_pair(ctx, "axis")
    if axis_pair is None and len(pairs) > 1:
        axis_pair = pairs[1]
    start_pair = _formal_pair(ctx, "start")
    if start_pair is None and len(pairs) > 2:
        start_pair = pairs[2]
    ax_val = _int_or_none(axis_pair)
    if ax_val in (None, "bail"):
        return ctx.default_return_type
    assert isinstance(ax_val, int)
    norm_ax = _normalize_axis(ax_val, ndim)
    if norm_ax is None:
        ctx.api.fail(
            f"rollaxis axis {ax_val} is out of range for {ndim} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    if start_pair is None:
        start = 0
    else:
        s_val = _int_or_none(start_pair)
        if s_val in (None, "bail"):
            return ctx.default_return_type
        assert isinstance(s_val, int)
        start = s_val
    # start can be 0..ndim inclusive, normalized with ndim
    if start < 0:
        norm_start = start + ndim
    else:
        norm_start = start
    if not 0 <= norm_start <= ndim:
        ctx.api.fail(
            f"rollaxis start {start} is out of range for {ndim} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    # Rollaxis algorithm per numpy docs
    # If start <= axis, move axis to start, else to start-1 after removal
    # Equivalent to: remove axis, then insert at start position
    remaining = [d for i, d in enumerate(dims) if i != norm_ax]
    # Adjust start for removal
    if norm_start > norm_ax:
        ins = norm_start - 1
    else:
        ins = norm_start
    out = remaining[:ins] + [dims[norm_ax]] + remaining[ins:]
    return _ndarray_type(ctx, out, dtype)


def _matrix_transpose_solve(
    ctx: FunctionContext | MethodContext, dims: list[int | Type], dtype: Type
) -> Type:
    ndim = len(dims)
    if ndim < 2:
        return _ndarray_type(ctx, list(dims), dtype)
    out = list(dims)
    out[-2], out[-1] = out[-1], out[-2]
    return _ndarray_type(ctx, out, dtype)


def matrix_transpose_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    return _matrix_transpose_solve(ctx, dims, dtype)


def mt_attribute_hook(ctx: AttributeContext) -> Type:
    parts = _ndarray_parts(ctx.type) or _mixed_parts(ctx.type)
    if parts is None:
        return ctx.default_attr_type
    dims, dtype = parts
    return _matrix_transpose_solve(ctx, dims, dtype)


def permute_dims_function_hook(ctx: FunctionContext) -> Type:
    # permute_dims is alias to transpose: first arg array, second optional axes
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0]) or _mixed_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    if len(pairs) == 1:
        # No axes: reverse
        return _ndarray_type(ctx, list(reversed(dims)), dtype)
    # axes as second positional or formal "axes"
    axes_pair = _formal_pair(ctx, "axes")
    if axes_pair is None and len(pairs) > 1:
        axes_pair = pairs[1]
    if axes_pair is None:
        return ctx.default_return_type
    axes = _shape_from_arg(axes_pair[0], axes_pair[1])
    if axes is None:
        return ctx.default_return_type
    return _apply_permutation(ctx, dims, dtype, axes)


def append_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    parts_a = _ndarray_parts(pairs[0][0])
    parts_b = _ndarray_parts(pairs[1][0])
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    dims_a, dtype_a = parts_a
    dims_b, dtype_b = parts_b
    dtype = _promote_all(ctx, [dtype_a, dtype_b])
    if dtype is None:
        return ctx.default_return_type
    axis_pair = _formal_pair(ctx, "axis")
    if axis_pair is None and len(pairs) > 2:
        axis_pair = pairs[2]
    axis_val = _int_or_none(axis_pair)
    if axis_val is None:
        # axis=None: flatten and concat -> 1-D
        # total = product(a) + product(b) if concrete, else unknown
        if all(isinstance(d, int) for d in dims_a) and all(isinstance(d, int) for d in dims_b):
            prod_a = 1
            for d in dims_a:
                assert isinstance(d, int)
                prod_a *= d
            prod_b = 1
            for d in dims_b:
                assert isinstance(d, int)
                prod_b *= d
            return _ndarray_type(ctx, [prod_a + prod_b], dtype)
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "append", 0)], dtype)
    if axis_val == "bail":
        return ctx.default_return_type
    assert isinstance(axis_val, int)
    # axis specified: must be valid for both, ranks must match?
    # Reuse concat logic: out dims same except axis summed
    # For append, values can be 1-D? but we check ranks equal for simplicity, else bail to Any-like (unknown)
    if len(dims_a) != len(dims_b):
        # If ranks differ, numpy still allows? Actually append with axis requires same shape except axis.
        # We'll bail if ranks differ.
        return ctx.default_return_type
    norm = _normalize_axis(axis_val, len(dims_a))
    if norm is None:
        ctx.api.fail(
            f"append axis {axis_val} is out of range for {len(dims_a)} dimensions "
            f"(shape {_dims_str(dims_a)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims_a)
    for i, (da, db) in enumerate(zip(dims_a, dims_b)):
        if i == norm:
            continue
        if not _dim_eq(da, db):
            if isinstance(da, int) and isinstance(db, int):
                ctx.api.fail(
                    f"append dimension {i} differs: {da} vs {db} "
                    f"(shapes {_dims_str(dims_a)} vs {_dims_str(dims_b)})",
                    ctx.context,
                    code=NUMTY_SHAPE,
                )
                return ctx.default_return_type
            return ctx.default_return_type
    out[norm] = _dim_add2(ctx, "append", norm, dims_a[norm], dims_b[norm])
    return _ndarray_type(ctx, out, dtype)


def insert_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    axis_pair = _formal_pair(ctx, "axis")
    if axis_pair is None and len(pairs) > 3:
        # insert(arr, obj, values, axis) -> axis is 4th? Actually signature: arr, obj, values, axis
        # positionals: arr, obj, values, axis
        axis_pair = pairs[3] if len(pairs) > 3 else None
    if axis_pair is None:
        # axis=None flattens -> 1-D unknown (insertion increases size, but hard to know)
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "insert", 0)], dtype)
    axis_val = _int_or_none(axis_pair)
    if axis_val == "bail":
        return ctx.default_return_type
    if axis_val is None:
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "insert", 0)], dtype)
    assert isinstance(axis_val, int)
    norm = _normalize_axis(axis_val, len(dims))
    if norm is None:
        ctx.api.fail(
            f"insert axis {axis_val} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims)
    out[norm] = _fresh_unknown(ctx, "insert", norm)
    return _ndarray_type(ctx, out, dtype)


def delete_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    axis_pair = _formal_pair(ctx, "axis")
    if axis_pair is None and len(pairs) > 2:
        axis_pair = pairs[2]
    if axis_pair is None:
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "delete", 0)], dtype)
    axis_val = _int_or_none(axis_pair)
    if axis_val == "bail":
        return ctx.default_return_type
    if axis_val is None:
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "delete", 0)], dtype)
    assert isinstance(axis_val, int)
    norm = _normalize_axis(axis_val, len(dims))
    if norm is None:
        ctx.api.fail(
            f"delete axis {axis_val} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims)
    out[norm] = _fresh_unknown(ctx, "delete", norm)
    return _ndarray_type(ctx, out, dtype)


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
            ctx.api.fail(f"duplicate axis {ax}", ctx.context, code=NUMTY_AXIS)
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
    if axis == "none":
        # Explicit axis=None reduces every axis: keepdims keeps them all
        # as 1s, otherwise the result is a scalar of the output dtype
        # (the stubs leave explicit None as Any).
        if keepdims is None or has_dtype:
            return ctx.default_return_type
        if _dtype_fullname(dtype) not in REDUCTION_DTYPES:
            return ctx.default_return_type
        if keepdims:
            return _solve_reduction(
                ctx, dims, dtype, list(range(len(dims))), True, has_dtype
            )
        scalar = _scalar_of_dtype(dtype)
        return scalar if scalar is not None else ctx.default_return_type
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
            return ("mask", len(part[0]))
        return ("badindex",)
    if isinstance(expr, SliceExpr):
        begin = _fold_index_bound(ctx, expr.begin_index)
        end = _fold_index_bound(ctx, expr.end_index)
        stride = _fold_index_bound(ctx, expr.stride)
        # Unprovable bounds no longer bail the whole getitem (issue #57):
        # they become "unknown" and the consumer mints a fresh unknown
        # for that axis while keeping the sibling dims.
        return (
            "slice",
            "unknown" if begin is None else begin,
            "unknown" if end is None else end,
            "unknown" if stride is None else stride,
        )
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
            # An explicit None inside slice(...) means missing, exactly
            # like an omitted bound; an unprovable bound is "unknown"
            # (issue #57) rather than a whole-getitem bail.
            bounds: list[int | str] = []
            for b in raw:
                if b is None:
                    bounds.append("missing")
                elif isinstance(b, NameExpr) and b.name == "None":
                    bounds.append("missing")
                else:
                    value = _fold_int(b)
                    bounds.append(value if value is not None else "unknown")
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
    ctx: MethodContext,
    item: tuple,
    size: int | Type,
    axis: str,
    shape_repr: str,
    pos: int = 0,
    role: str = "slice",
) -> list[int | Type] | None:
    """Apply one parsed index item to a known dim: kept dims, or None.

    None means bail out silently, except provable errors (out-of-range
    int, zero step, bad index type), which fail inside and also yield
    None. Dropped dims contribute nothing. A slice bound the parser
    could not prove ("unknown", issue #57) mints a fresh unknown for a
    concrete axis instead of bailing the whole shape; over a symbolic
    axis only a full-range unit-stride slice keeps the dim.
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
                    code=NUMTY_AXIS,
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
            code=NUMTY_AXIS,
        )
        return None
    _, begin, end, stride = item
    if isinstance(size, int):
        if "unknown" in (begin, end, stride):
            return [_fresh_unknown(ctx, role, pos)]
        bounds = tuple(None if b == "missing" else b for b in (begin, end, stride))
        try:
            return [len(range(*slice(*bounds).indices(size)))]
        except ValueError:
            ctx.api.fail("slice step is zero", ctx.context, code=NUMTY_AXIS)
            return None
    if begin == "missing" and end == "missing":
        if stride == "missing" or stride in (1, -1):
            return [size]
        if stride == 0:
            ctx.api.fail("slice step is zero", ctx.context, code=NUMTY_AXIS)
            return None
    return [_fresh_unknown(ctx, role, pos)]


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
        if item[0] in ("fancy", "mask"):
            # The caller admits only a lone advanced index, which stays
            # in place: a gather over the known prefix keeps *Ts and the
            # rest untouched. A k-d mask consumes k prefix axes for one
            # data-dependent count; one reaching into *Ts bails like any
            # other index.
            if item[0] == "fancy":
                out.extend(item[1])
                i += 1
                continue
            rank = item[1]
            if rank == 0 or i + rank > len(prefix):
                return None
            out.append(_fresh_unknown(ctx, "mask", i))
            i += rank
            continue
        frag = _consume_one(ctx, item, prefix[i], str(i), shape_repr, i, "slice")
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
        frag = _consume_one(
            ctx, item, suffix[p], f"-{len(suffix) - p}", shape_repr, p, "slicesuf"
        )
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
    the suffix pass through. Anything reaching into *Ts bails out. A
    lone integer array or boolean mask on the known prefix gathers in
    place (`frames[idx]` on (total, *Ts) is (K, *Ts)); several advanced
    indices, an integer beside one, or one on the suffix bails, since
    numpy may move their block to the front.
    """
    pairs = _index_pairs(ctx, pair)
    if pairs is None:
        return ctx.default_return_type
    items = _parse_index_items(ctx, pairs)
    if items is None:
        return ctx.default_return_type
    if sum(1 for item in items if item[0] == "ellipsis") > 1:
        ctx.api.fail(
            "only one Ellipsis (...) is allowed in an index",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    if any(item[0] in ("fancy", "mask") for item in items) and (
        sum(1 for item in items if item[0] in ("fancy", "mask", "int", "int_any", "fancy0d"))
        > 1
    ):
        # Beside an array index numpy treats integers as advanced indices
        # too, and a block split by a slice or ellipsis moves to the
        # front: only a lone array index has a place we can name.
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
        ctx.api.fail(
            "only one Ellipsis (...) is allowed in an index",
            ctx.context,
            code=NUMTY_AXIS,
        )
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
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    fancy_positions = [n for n, item in enumerate(items) if item[0] == "fancy"]
    if len(fancy_positions) > 1:
        between = items[fancy_positions[0] + 1 : fancy_positions[-1]]
        if any(item[0] not in ("int", "fancy", "fancy0d", "newaxis") for item in between):
            # Advanced indices separated by a slice/ellipsis broadcast to
            # the front; that reshuffle is out of scope, so bail out.
            return ctx.default_return_type
    fancy_dims: list[int | Type] = []
    for n in fancy_positions:
        assert items[n][0] == "fancy"
        shape = items[n][1]
        if not fancy_dims:
            fancy_dims = shape
            continue
        merged, cause = _broadcast_result(fancy_dims, shape, ctx)
        if cause is not None:
            ctx.api.fail(
                f"indexing arrays could not be broadcast together: {cause}",
                ctx.context,
                code=NUMTY_BROADCAST,
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
                        code=NUMTY_AXIS,
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
            # Boolean masks select a data-dependent count: track the axis
            # with a fresh unknown instead of failing. A 0-d mask adds a
            # leading axis (newaxis); combined advanced indices collapse
            # to one block whose rank is unknowable, so bail there.
            if sum(1 for it in items if it[0] in ("fancy", "mask")) > 1:
                return ctx.default_return_type
            if item[1] == 0:
                out.append(1)
            else:
                if dim >= len(dims):
                    return ctx.default_return_type
                out.append(_fresh_unknown(ctx, "mask", dim))
                dim += 1
        elif kind == "badindex":
            ctx.api.fail(
                "arrays used as indices must be of integer (or boolean) type",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        else:
            # Slice: an unprovable bound ("unknown", issue #57) mints a
            # fresh unknown for a concrete axis instead of bailing the
            # whole shape; over a symbolic axis only a full-range
            # unit-stride slice keeps the dim.
            _, begin, end, stride = item
            size = dims[dim]
            if isinstance(size, int):
                if "unknown" in (begin, end, stride):
                    out.append(_fresh_unknown(ctx, "slice", dim))
                else:
                    bounds = tuple(
                        None if b == "missing" else b for b in (begin, end, stride)
                    )
                    try:
                        out.append(len(range(*slice(*bounds).indices(size))))
                    except ValueError:
                        ctx.api.fail(
                            "slice step is zero", ctx.context, code=NUMTY_AXIS
                        )
                        return ctx.default_return_type
                dim += 1
                continue
            if begin == "missing" and end == "missing":
                if stride == "missing" or stride in (1, -1):
                    out.append(size)
                elif stride == 0:
                    ctx.api.fail(
                        "slice step is zero", ctx.context, code=NUMTY_AXIS
                    )
                    return ctx.default_return_type
                else:
                    out.append(_fresh_unknown(ctx, "slice", dim))
            else:
                out.append(_fresh_unknown(ctx, "slice", dim))
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
            ctx.api.fail(message, ctx.context, code=NUMTY_BROADCAST)
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
) -> list[int | Type] | None:
    """Broadcast output dims for two operands, else None to bail or fail."""
    a = _ndarray_parts(left)
    b = _ndarray_parts(right)
    if a is None or b is None:
        return _broadcast_dims_unpack(ctx, left, right, a, b)
    dims_a, _ = a
    dims_b, _ = b
    dims, message = _broadcast_result(dims_a, dims_b, ctx)
    if message is not None:
        ctx.api.fail(message, ctx.context, code=NUMTY_BROADCAST)
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


def isclose_function_hook(ctx: FunctionContext) -> Type:
    """np.isclose: broadcast bool like ==, mismatch fails (issue #58).

    rtol/atol/equal_nan are scalars and never affect the shape; only
    the first two array operands solve. allclose returns a Python bool
    and stays with the stubs.
    """
    pair = _two_arrays(ctx)
    if pair is None:
        a_pair = _formal_pair(ctx, "a")
        b_pair = _formal_pair(ctx, "b")
        if a_pair is None or b_pair is None:
            return ctx.default_return_type
        pair = (a_pair[0], b_pair[0])
    dist = _distribute_solve([pair[0], pair[1]], lambda l, r: _compare_types(ctx, l, r))
    if dist is not None:
        return dist
    return _compare_types(ctx, pair[0], pair[1])


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
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        if norm in normalized:
            ctx.api.fail(f"duplicate axis {ax}", ctx.context, code=NUMTY_AXIS)
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
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
    return _ndarray_type(ctx, dims, _default_dtype(ctx) if int_out else dtype)


def _kth_in_bounds(
    ctx: FunctionContext | MethodContext,
    dims: list[int | Type],
    axis: list[int] | None | str,
    op: str,
) -> bool:
    """partition/argpartition: every literal kth must index the axis.

    kth may be negative (counted from the end) but must land in
    [-size, size) of the partitioned axis, or of the flattened array
    for axis=None. Anything unproven passes. The method spells kth
    positional-only, so it reads from the first positional there, as
    long as the callee's formals (axis) are visible at all.
    """
    pair = _formal_pair(ctx, "kth")
    if pair is None and isinstance(ctx, MethodContext) and "axis" in ctx.callee_arg_names:
        pos = _all_positionals(ctx)
        pair = pos[0] if pos else None
    if pair is None or not dims:
        return True
    if any(isinstance(d, int) and d == 0 for d in dims):
        # numpy accepts any kth when there is nothing to partition.
        return True
    kths = _shape_from_arg(*pair)
    if not kths:
        return True
    size: int | Type
    if axis == "none":
        total = 1
        for d in dims:
            if not isinstance(d, int):
                return True
            total *= d
        size = total
        where = f"the flattened size {size}"
    else:
        if axis == "absent":
            ax = -1
        elif isinstance(axis, list) and len(axis) == 1:
            ax = axis[0]
        else:
            return True
        norm = _normalize_axis(ax, len(dims))
        if norm is None:
            # The axis range check reports this.
            return True
        size = dims[norm]
        where = f"axis {ax} of size {size}"
    if not isinstance(size, int):
        return True
    for k in kths:
        if not -size <= k < size:
            ctx.api.fail(
                f"{op} kth {k} is out of bounds for {where} "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return False
    return True


ARG_DROP = frozenset({"argmax", "argmin", "nanargmax", "nanargmin", "argpartition"})
ARG_SAME_INT = frozenset({"argsort"})
ARG_SAME = frozenset({"sort", "partition"})


def _arg_inputs(ctx: FunctionContext | MethodContext, array: Type, op: str) -> Type:
    parts = _ndarray_parts(array)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if op in ("partition", "argpartition") and not _kth_in_bounds(ctx, dims, axis, op):
        return ctx.default_return_type
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


def _parse_size_arg(ctx: MethodContext) -> list[int | Type] | str | None:
    """A sampler's size=: 'absent'/'none' as _parse_axis_arg, else builder dims."""
    pair = _formal_pair(ctx, "size")
    if pair is None:
        return "absent"
    if _is_none_type(pair[0]):
        return "none"
    dims = _builder_shape(*pair, ctx.default_return_type, shaped_stub=False)
    if dims is not None and any(_is_plain_int(d) for d in dims):
        # A mixed size= stays with the stubs (tuple[Any, ...]). Unlike
        # np.zeros, whose overloads take the shape from the assignment's
        # type context, a sampler's answer is final, so x =
        # rng.standard_normal((n, d, 4)) could not rebind a name first
        # bound to (n, 3, 4).
        return None
    return dims


def rng_method_hook(ctx: MethodContext, op: str) -> Type:
    if op in RNG_FLOAT_OPS:
        size = _parse_size_arg(ctx)
        if not isinstance(size, list):
            return ctx.default_return_type
        dtype = _construct_dtype(ctx, "numpy.float64")
        if dtype is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, size, dtype)
    if op == "integers":
        size = _parse_size_arg(ctx)
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


def legacy_rand_hook(ctx: FunctionContext | MethodContext) -> Type:
    """np.random.rand/randn(*dims): shape is the int dims, float64 (issue #58).

    No args is a scalar (stubs own it). Each positional must fold to a
    non-negative int; anything dynamic bails instead of guessing.
    """
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    dims: list[int | Type] = []
    for tp, expr in pairs:
        value = _fold_int(expr) if expr is not None else None
        if value is None:
            value = _int_value(tp)
        if value is None or value < 0:
            return ctx.default_return_type
        dims.append(value)
    if not _dims_fit_intp(ctx, dims):
        return ctx.default_return_type
    dtype = _construct_dtype(ctx, "numpy.float64")
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def _percentile_q_dims(
    tp: Type, expr: Expression | None
) -> list[int | Type] | None:
    """Shape of a percentile/quantile q argument, else None to bail.

    1-D/2-D list displays give [n]/[n, m]; ndarrays give their dims;
    Python/numpy scalars give [] (0-d). Anything else (a bare list
    variable, a dynamic value) stays with the stubs.
    """
    if isinstance(expr, (ListExpr, TupleExpr)):
        if not expr.items:
            return [0]
        if all(not isinstance(e, (ListExpr, TupleExpr)) for e in expr.items):
            return [len(expr.items)]
        if all(isinstance(e, (ListExpr, TupleExpr)) for e in expr.items):
            inner = [len(e.items) for e in expr.items]  # type: ignore[union-attr]
            if len(set(inner)) == 1 and all(
                not isinstance(sub, (ListExpr, TupleExpr))
                for e in expr.items
                for sub in e.items  # type: ignore[union-attr]
            ):
                return [len(expr.items), inner[0]]
        return None
    parts = _ndarray_parts(tp)
    if parts is not None:
        dims, _ = parts
        if _has_unpack(dims):
            return None
        return list(dims)
    if _weak_name(tp) is not None or _scalar_key(tp) is not None:
        return []
    if _int_value(tp) is not None:
        return []
    return None


def percentile_function_hook(ctx: FunctionContext) -> Type:
    """np.percentile/quantile (+nan twins): q-shape + reduced shape (issue #58).

    result is shape(q) followed by shape(a) with axis removed (or 1s
    with keepdims). Float inputs keep dtype, ints become float64;
    anything unprovable bails to the stubs instead of guessing.
    """
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    part = _ndarray_parts(pairs[0][0])
    if part is None:
        return ctx.default_return_type
    dims_a, dtype_a = part
    if _has_unpack(dims_a):
        return ctx.default_return_type
    q_dims = _percentile_q_dims(pairs[1][0], pairs[1][1])
    if q_dims is None:
        return ctx.default_return_type
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if axis is None:
        return ctx.default_return_type
    keepdims = _fold_flag(_formal_pair(ctx, "keepdims"))
    if keepdims is None:
        keepdims = False
    ndim = len(dims_a)
    if axis in ("absent", "none"):
        remaining: list[int | Type] = [1] * ndim if keepdims else []
    else:
        assert isinstance(axis, list)
        normed: list[int] = []
        for ax in axis:
            norm = _normalize_axis(ax, ndim)
            if norm is None:
                ctx.api.fail(
                    f"percentile axis {ax} is out of range for {ndim} dimensions "
                    f"(shape {_dims_str(dims_a)})",
                    ctx.context,
                    code=NUMTY_AXIS,
                )
                return ctx.default_return_type
            normed.append(norm)
        if len(set(normed)) != len(normed):
            return ctx.default_return_type
        removed = set(normed)
        if keepdims:
            remaining = [1 if i in removed else d for i, d in enumerate(dims_a)]
        else:
            remaining = [d for i, d in enumerate(dims_a) if i not in removed]
    ks = _dtype_kind_size(dtype_a)
    if ks is None:
        return ctx.default_return_type
    kind, _size = ks
    if kind == "f":
        out_dtype = dtype_a
    elif kind in ("i", "u", "b"):
        out_dtype = _construct_dtype(ctx, "numpy.float64")
    else:
        return ctx.default_return_type
    if out_dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, [*q_dims, *remaining], out_dtype)



def reduction_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _reduction_inputs(ctx, pairs[0][0])


def _scalar_of_dtype(dtype: Type) -> Type | None:
    """The numpy scalar Instance carried by a dtype[...], else None."""
    proper = get_proper_type(dtype)
    if isinstance(proper, Instance) and proper.args:
        inner = get_proper_type(proper.args[0])
        if isinstance(inner, Instance):
            return inner
    return None


def _same_dims(a: list[int | Type], b: list[int | Type]) -> bool:
    """Provably-equal dim lists: same ints, same TypeVars by fullname."""
    return len(a) == len(b) and all(_dim_eq(x, y) for x, y in zip(a, b))


def _check_average_weights(
    ctx: FunctionContext, array: Type, weights: Type
) -> str | None:
    """Mismatch message when 1-D weights provably disagree with average.

    NumPy accepts 1-D weights only when their length matches the averaged
    axis, or weights shaped exactly like the input. Anything else is left
    alone: no false positives on shapes we cannot judge.
    """
    wparts = _ndarray_parts(weights)
    aparts = _ndarray_parts(array)
    if wparts is None or aparts is None:
        return None
    wdims, _ = wparts
    adims, _ = aparts
    if len(wdims) != 1 or _same_dims(wdims, adims):
        return None
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if not isinstance(axis, list) or len(axis) != 1:
        return None
    norm = _normalize_axis(axis[0], len(adims))
    if norm is None:
        return None
    size = adims[norm]
    if not isinstance(size, int) or not isinstance(wdims[0], int):
        # Symbolic sizes cannot be compared: unknown dims (#14) will make
        # this exact instead of silent.
        return None
    if size == wdims[0]:
        return None
    return (
        f"average weights length {wdims[0]} does not match "
        f"axis {axis[0]} size {size} (shape {_dims_str(adims)})"
    )


def average_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    weights = _formal_pair(ctx, "weights")
    if weights is not None and not _is_none_type(weights[0]):
        message = _check_average_weights(ctx, pairs[0][0], weights[0])
        if message is not None:
            ctx.api.fail(message, ctx.context, code=NUMTY_SHAPE)
            return ctx.default_return_type
    return _reduction_inputs(ctx, pairs[0][0])


def _cumsum_solve(ctx: FunctionContext | MethodContext, array: Type) -> Type:
    """cumsum/cumprod keep the input shape; only axis=None flattens.

    The flattened size is a products of dims we cannot spell, so
    axis=None (the default) stays with the stubs, except for 1-D
    inputs where axis=None keeps the shape (issue #32: np.cumsum
    without axis on I64[N] is (N,)). dtype is whatever the
    stubs claim: only the shape is solved here.
    """
    parts = _parts_or_unpack(array)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    if _has_unpack(dims):
        return ctx.default_return_type
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if not isinstance(axis, list) or len(axis) != 1 or not isinstance(axis[0], int):
        # Missing axis (default None): 1-D keeps shape, ND flattens (bail).
        if axis in ("absent", "none") and len(dims) == 1:
            axis = [0]
        else:
            return ctx.default_return_type
    norm = _normalize_axis(axis[0], len(dims))
    if norm is None:
        ctx.api.fail(
            f"axis {axis[0]} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    dtype_pair = _formal_pair(ctx, "dtype")
    if dtype_pair is not None and not _is_none_type(dtype_pair[0]):
        return ctx.default_return_type
    # Accumulate promotes small ints (verified against the runtime:
    # bool/signed -> int64, unsigned -> uint64, float/complex keep).
    # Solving from the input keeps every overload candidate identical,
    # which keeps mypy's overload resolution from discarding it.
    name = _dtype_fullname(dtype)
    if name in ("numpy.bool", "numpy.int8", "numpy.int16", "numpy.int32", "numpy.int64"):
        promoted = _construct_dtype(ctx, "numpy.int64")
        if promoted is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, list(dims), promoted)
    if name in ("numpy.uint8", "numpy.uint16", "numpy.uint32", "numpy.uint64"):
        promoted = _construct_dtype(ctx, "numpy.uint64")
        if promoted is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, list(dims), promoted)
    if name is None or name not in REDUCTION_DTYPES:
        # Non-numeric dtypes (strings, datetime64, ...): the stubs own it.
        return ctx.default_return_type
    return _ndarray_type(ctx, list(dims), dtype)


def cumsum_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _cumsum_solve(ctx, pairs[0][0])


def cumsum_method_hook(ctx: MethodContext) -> Type:
    return _cumsum_solve(ctx, ctx.type)


def diff_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _parts_or_unpack(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    if _has_unpack(dims):
        return ctx.default_return_type
    ndim = len(dims)
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if axis == "absent":
        ax = ndim - 1
    elif isinstance(axis, list) and len(axis) == 1 and isinstance(axis[0], int):
        norm = _normalize_axis(axis[0], ndim)
        if norm is None:
            ctx.api.fail(
                f"axis {axis[0]} is out of range for {ndim} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        ax = norm
    else:
        return ctx.default_return_type
    n_pair = _formal_pair(ctx, "n")
    n = 1 if n_pair is None else _int_value(n_pair[0])
    if n is None:
        return ctx.default_return_type
    if n < 0:
        ctx.api.fail(f"np.diff order n={n} must be non-negative", ctx.context, code=NUMTY_SHAPE)
        return ctx.default_return_type
    for name in ("prepend", "append"):
        pair = _formal_pair(ctx, name)
        if pair is not None and not _is_none_type(pair[0]):
            return ctx.default_return_type
    if _dtype_fullname(dtype) == "numpy.datetime64":
        # Datetime diffs change dtype (datetime64 -> timedelta64), which
        # only the stubs' dedicated overload tracks.
        return ctx.default_return_type
    size = dims[ax]
    out = list(dims)
    if isinstance(size, int):
        out[ax] = max(0, size - n)
    else:
        # Symbolic dims spell the subtraction (N-1, N-2, ?U-1): never
        # reuse the input dim, or diff would claim the length is unchanged
        # and real broadcast errors would pass (issue #32).
        out[ax] = _dim_add_const(ctx, "diff", ax, size, -n)
    # Diff preserves dtype exactly, so solve from the input: the result
    # is identical for every overload candidate, which keeps mypy's
    # overload resolution from discarding it as ambiguous.
    return _ndarray_type(ctx, out, dtype)


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
            code=NUMTY_SHAPE,
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
                    code=NUMTY_SHAPE,
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
                    code=NUMTY_SHAPE,
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
                        code=NUMTY_SHAPE,
                    )
                    return ctx.default_return_type
                # Symbolic vs anything unprovable: bail out.
                return ctx.default_return_type
    broadcast: list[int | Type] | None = None
    for edims in ell_inputs:
        if broadcast is None:
            broadcast = edims
            continue
        out, cause = _broadcast_result(broadcast, edims, ctx)
        if cause is not None:
            ctx.api.fail(
                f"einsum ellipsis dimensions cannot broadcast: {cause}",
                ctx.context,
                code=NUMTY_BROADCAST,
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
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        seen: set[str] = set()
        for label in pre + post:
            if label in seen:
                ctx.api.fail(
                    f"einsum output subscript '{label}' appears more than once",
                    ctx.context,
                    code=NUMTY_AXIS,
                )
                return ctx.default_return_type
            seen.add(label)
            if label not in label_size:
                ctx.api.fail(
                    f"einsum output subscript '{label}' never appeared "
                    "in an input",
                    ctx.context,
                    code=NUMTY_AXIS,
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
            code=NUMTY_SHAPE,
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
    batch, cause = _broadcast_result(dims_a[:-2], batch_b, ctx)
    if cause is not None:
        ctx.api.fail(
            f"linalg.solve batch dimensions disagree: {_dims_str(dims_a[:-2])} vs "
            f"{_dims_str(batch_b)} ({cause})",
            ctx.context,
            code=NUMTY_SHAPE,
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
            code=NUMTY_SHAPE,
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


def _linalg_pinv(ctx: FunctionContext) -> Type:
    """linalg.pinv: (..., M, N) -> (..., N, M), ints become float64 (issue #58)."""
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    part = _ndarray_parts(pairs[0][0])
    if part is None:
        return ctx.default_return_type
    dims, dtype = part
    if _has_unpack(dims):
        return ctx.default_return_type
    if len(dims) < 2:
        ctx.api.fail(
            f"linalg.pinv requires at least 2 dimensions, got shape {_dims_str(dims)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    ks = _dtype_kind_size(dtype)
    if ks is None:
        return ctx.default_return_type
    kind, _size = ks
    if kind in ("i", "u", "b"):
        out_dtype = _construct_dtype(ctx, "numpy.float64")
    elif kind in ("f", "c"):
        out_dtype = dtype
    else:
        return ctx.default_return_type
    if out_dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, [*dims[:-2], dims[-1], dims[-2]], out_dtype)


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
            code=NUMTY_SHAPE,
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
            code=NUMTY_SHAPE,
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
            code=NUMTY_SHAPE,
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
    batch, cause = _broadcast_result(dims_a[:-2], batch_b, ctx)
    if cause is not None:
        ctx.api.fail(
            f"linalg.lstsq batch dimensions disagree: {_dims_str(dims_a[:-2])} vs "
            f"{_dims_str(batch_b)} ({cause})",
            ctx.context,
            code=NUMTY_SHAPE,
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


def _norm_ord_fits(ctx: FunctionContext, dims: list[int | Type], reduced: int) -> bool:
    """linalg.norm: an explicit ord needs a vector or matrix norm.

    With ord=None numpy flattens any rank; any other ord picks a vector
    norm (one reduced axis) or a matrix norm (two), and string orders
    ('fro', 'nuc') exist only for matrices. An ord of unknown value
    passes.
    """
    pair = _formal_pair(ctx, "ord")
    if pair is None or _is_none_type(pair[0]):
        return True
    order = _string_value(pair[0])
    if order is None and _weak_name(pair[0]) not in ("builtins.int", "builtins.float"):
        return True
    number = _int_value(pair[0])
    if order is not None:
        shown = f"ord={order!r}"
    elif number is not None:
        shown = f"ord={number}"
    else:
        shown = "an explicit ord"
    if reduced > 2:
        ctx.api.fail(
            f"linalg.norm with {shown} reduces {reduced} dimensions; only "
            f"vector (1) or matrix (2) norms exist (shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return False
    if order is not None and reduced == 1:
        ctx.api.fail(
            f"linalg.norm ord={order!r} is a matrix norm and needs 2 axes, "
            f"got 1 (shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return False
    return True


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
                    code=NUMTY_AXIS,
                )
                return ctx.default_return_type
            axes.append(norm)
        if len(set(axes)) != len(axes):
            return ctx.default_return_type
    if not _norm_ord_fits(ctx, dims, len(axes)):
        return ctx.default_return_type
    if keepdims:
        out_dims = [1 if i in axes else d for i, d in enumerate(dims)]
    else:
        out_dims = [d for i, d in enumerate(dims) if i not in axes]
    ks = _dtype_kind_size(dtype)
    if ks is None:
        return ctx.default_return_type
    name = _linalg_real_name(*ks)
    if name is None:
        return ctx.default_return_type
    out_dtype = _linalg_dtype(ctx, dtype, name)
    if out_dtype is None:
        return ctx.default_return_type
    if not out_dims:
        # Fully reduced: a scalar of the real dtype, which the stubs
        # leave as Any.
        scalar = _scalar_of_dtype(out_dtype)
        return scalar if scalar is not None else ctx.default_return_type
    return _ndarray_type(ctx, out_dims, out_dtype)


def linalg_function_hook(ctx: FunctionContext, op: str) -> Type:
    if op == "solve":
        return _linalg_solve(ctx)
    if op in ("inv", "cholesky"):
        return _linalg_same_shape(ctx, op)
    if op == "pinv":
        return _linalg_pinv(ctx)
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
            out, cause = _broadcast_result(dims, other, ctx)
            if cause is not None:
                ctx.api.fail(f"{op} shapes disagree: {cause}", ctx.context, code=NUMTY_BROADCAST)
                return None
            if out is None:
                return None
            dims = out
            continue
        merged, message = _merge_shape_lists(dims, other)
        if message is not None:
            ctx.api.fail(f"{op} shapes disagree: {message}", ctx.context, code=NUMTY_BROADCAST)
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
        # Single-argument where(cond) returns indices like nonzero.
        pairs = _all_positionals(ctx)
        if len(pairs) == 1:
            return _nonzero_like(ctx, "where", pairs[0][0])
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
            code=NUMTY_SHAPE,
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

    Concrete and mixed shapes reverse exactly; unpack shapes reverse with the
    unknown block kept in place per _reverse_unpack_dims. Anything else
    stays on the stubs.
    """
    parts = _ndarray_parts(tp) or _mixed_parts(tp)
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
        ctx.api.fail("len() of unsized object", ctx.context, code=NUMTY_SHAPE)
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
            code=NUMTY_AXIS,
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
                code=NUMTY_SHAPE,
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
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    if len(dims_a) != len(dims_i):
        ctx.api.fail(
            f"take_along_axis `indices` and `arr` must have the same number "
            f"of dimensions (shapes {_dims_str(dims_a)} vs {_dims_str(dims_i)})",
            ctx.context,
            code=NUMTY_SHAPE,
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
                code=NUMTY_SHAPE,
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
        if not _matmul_has_dims(ctx, [pairs[0][0], pairs[1][0]]):
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


def _numpy_function_hooks() -> dict[str, Callable[[FunctionContext], Type]]:
    """Exact fullname -> hook for the numpy functions numty solves.

    mypy reports a re-exported function under its defining module
    (np.sum is numpy._core.fromnumeric.sum), so the keys are those
    canonical names. Matching on the name tail instead would also catch
    same-named twins with different semantics: numpy.ma (MaskedArray
    results), numpy.emath / numpy.lib.scimath (complex results),
    numpy.matlib (matrix results), numpy.char / numpy.strings,
    numpy.rec and numpy.linalg.outer (1-D only). Those keep their stub
    return types.
    """
    by_module: dict[str, dict[str, Callable[[FunctionContext], Type]]] = {
        "numpy._core.einsumfunc": {"einsum": einsum_function_hook},
        "numpy._core.fromnumeric": {
            "all": allany_function_hook,
            "any": allany_function_hook,
            "clip": clip_function_hook,
            "cumprod": cumsum_function_hook,
            "cumsum": cumsum_function_hook,
            "max": reduction_function_hook,
            "mean": reduction_function_hook,
            "min": reduction_function_hook,
            "nonzero": nonzero_function_hook,
            "prod": reduction_function_hook,
            "repeat": repeat_function_hook,
            "reshape": reshape_function_hook,
            "searchsorted": searchsorted_function_hook,
            "squeeze": squeeze_function_hook,
            "std": reduction_function_hook,
            "sum": reduction_function_hook,
            "take": take_function_hook,
            "transpose": transpose_function_hook,
            "swapaxes": swapaxes_function_hook,
            "matrix_transpose": matrix_transpose_function_hook,
            "var": reduction_function_hook,
            **{
                op: partial(arg_function_hook, op=op)
                for op in (
                    "argmax",
                    "argmin",
                    "argpartition",
                    "argsort",
                    "partition",
                    "sort",
                )
            },
        },
        "numpy._core.multiarray": {
            "arange": arange_function_hook,
            "array": asarray_function_hook,
            "asanyarray": asarray_function_hook,
            "asarray": asarray_function_hook,
            "ascontiguousarray": ascontiguousarray_function_hook,
            "concatenate": concatenate_function_hook,
            "dot": dot_function_hook,
            "empty": zeros_like_hook,
            "empty_like": like_function_hook,
            "inner": inner_function_hook,
            "vdot": vdot_function_hook,
            "where": where_function_hook,
            "zeros": zeros_like_hook,
        },
        "numpy._core.numeric": {
            "argwhere": argwhere_function_hook,
            "convolve": convolve_function_hook,
            "correlate": correlate_function_hook,
            "flatnonzero": flatnonzero_function_hook,
            "full": full_function_hook,
            "full_like": like_function_hook,
            "identity": identity_function_hook,
            "isclose": isclose_function_hook,
            "ones": zeros_like_hook,
            "ones_like": like_function_hook,
            "outer": outer_function_hook,
            "roll": roll_function_hook,
            "moveaxis": moveaxis_function_hook,
            "rollaxis": rollaxis_function_hook,
            "swapaxes": swapaxes_function_hook,
            "matrix_transpose": matrix_transpose_function_hook,
            "tensordot": tensordot_function_hook,
            "zeros_like": like_function_hook,
        },
        "numpy._core.function_base": {
            "linspace": linspace_function_hook,
        },
        "numpy._core.shape_base": {
            "atleast_1d": partial(atleast_function_hook, rank=1),
            "atleast_2d": partial(atleast_function_hook, rank=2),
            "atleast_3d": partial(atleast_function_hook, rank=3),
            "hstack": hstack_function_hook,
            "stack": stack_function_hook,
            "vstack": vstack_function_hook,
        },
        "numpy.fft._pocketfft": {
            "fft": fft_function_hook,
            "fft2": fft2_function_hook,
            "fftn": fftn_function_hook,
            "ifft": fft_function_hook,
            "ifft2": fft2_function_hook,
            "ifftn": fftn_function_hook,
            "rfft": rfft_function_hook,
            "rfftn": rfftn_function_hook,
        },
        "numpy.fft._helper": {
            "fftshift": fftshift_function_hook,
            "ifftshift": fftshift_function_hook,
        },
        "numpy.lib._arraypad_impl": {"pad": pad_function_hook},
        "numpy.lib._arraysetops_impl": {
            "isin": isin_function_hook,
            "unique": unique_function_hook,
        },
        "numpy.lib._function_base_impl": {
            "average": average_function_hook,
            "copy": copy_function_hook,
            "diff": diff_function_hook,
            "flip": flip_function_hook,
            "gradient": gradient_function_hook,
            "median": reduction_function_hook,
            "percentile": percentile_function_hook,
            "quantile": percentile_function_hook,
            "rot90": rot90_function_hook,
            "append": append_function_hook,
            "insert": insert_function_hook,
            "delete": delete_function_hook,
            "sinc": partial(ufunc_function_hook, group="float"),
        },
        "numpy.lib._nanfunctions_impl": {
            "nanargmax": partial(arg_function_hook, op="nanargmax"),
            "nanargmin": partial(arg_function_hook, op="nanargmin"),
            "nancumprod": cumsum_function_hook,
            "nancumsum": cumsum_function_hook,
            "nanmax": reduction_function_hook,
            "nanmean": reduction_function_hook,
            "nanmedian": reduction_function_hook,
            "nanmin": reduction_function_hook,
            "nanpercentile": percentile_function_hook,
            "nanprod": reduction_function_hook,
            "nanquantile": percentile_function_hook,
            "nanstd": reduction_function_hook,
            "nansum": reduction_function_hook,
            "nanvar": reduction_function_hook,
        },
        "numpy.lib._shape_base_impl": {
            "column_stack": column_stack_function_hook,
            "dstack": dstack_function_hook,
            "expand_dims": expand_dims_function_hook,
            "kron": kron_function_hook,
            "split": split_function_hook,
            "take_along_axis": take_along_axis_function_hook,
            "tile": tile_function_hook,
        },
        "numpy.lib._stride_tricks_impl": {
            "broadcast_to": broadcast_to_function_hook,
            "sliding_window_view": sliding_window_view_function_hook,
        },
        "numpy.lib._twodim_base_impl": {
            "eye": eye_function_hook,
            "fliplr": fliplr_function_hook,
            "flipud": flipud_function_hook,
            "tri": tri_function_hook,
        },
        "numpy.lib._type_check_impl": {
            "imag": partial(ufunc_function_hook, group="real"),
            "real": partial(ufunc_function_hook, group="real"),
        },
        "numpy.lib._ufunclike_impl": {
            "isneginf": partial(ufunc_function_hook, group="bool"),
            "isposinf": partial(ufunc_function_hook, group="bool"),
        },
        "numpy.linalg._linalg": {
            # numpy.linalg.tensordot is np.tensordot with keyword-only
            # axes; numpy.linalg.outer (1-D only) is deliberately absent.
            "tensordot": tensordot_function_hook,
            **{
                op: partial(linalg_function_hook, op=op)
                for op in (
                    "cholesky",
                    "eig",
                    "eigh",
                    "inv",
                    "lstsq",
                    "norm",
                    "pinv",
                    "solve",
                    "svd",
                )
            },
        },
        "numpy.random.mtrand": {
            "rand": legacy_rand_hook,
            "randn": legacy_rand_hook,
        },
        "numpy._core": {
            "moveaxis": moveaxis_function_hook,
            "swapaxes": swapaxes_function_hook,
            "rollaxis": rollaxis_function_hook,
            "matrix_transpose": matrix_transpose_function_hook,
            "permute_dims": permute_dims_function_hook,
        },
        "numpy": {
            "moveaxis": moveaxis_function_hook,
            "swapaxes": swapaxes_function_hook,
            "rollaxis": rollaxis_function_hook,
            "matrix_transpose": matrix_transpose_function_hook,
            "permute_dims": permute_dims_function_hook,
            "append": append_function_hook,
            "insert": insert_function_hook,
            "delete": delete_function_hook,
            "transpose": transpose_function_hook,
        },
    }
    return {
        f"{module}.{name}": hook
        for module, hooks in by_module.items()
        for name, hook in hooks.items()
    }


_NUMPY_FUNCTION_HOOKS = _numpy_function_hooks()
