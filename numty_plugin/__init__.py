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
  operator solves through the same matmul hooks (__matmul__/__rmatmul__,
  with __imatmul__ checked in place).
- __add__/__mul__ unify operand shapes with NumPy broadcast rules, so
  precision survives whole function bodies. `//`, `%`, `divmod`, `^`,
  `<<`, `>>` and int/bool `&`/`|` use the same broadcast (divmod
  returns a tuple of two). Scalars pass through; unknown dims bail out.
  In-place ops (__iadd__ and friends) reuse the broadcast check and
  additionally require the result to fit the target.
- reshape (method and np.reshape) checks total-size conservation and
  infers a single -1, returning the precise new shape. A -1 divides the
  canonical total (linear and monomial products alike: N*K/N=K,
  B*C*H*W/B=C*H*W); only inexact/inexpressible inferences mint an
  unknown, and same-total renames track as-is (legality stays unchecked,
  never an error). Over *Ts the spelled target is the whole new shape
  (correct whenever the call succeeds at runtime); a -1 would absorb
  the unknown product, so it bails, with provable int-only total
  clashes failing loudly. Anything dynamic bails out.
- rank-changing ops (transpose, .T, squeeze, expand_dims, sum/prod/mean
  and friends) track the new shape; transpose and .T bail out over
  *Ts (explicit axes bail too). No-axis reductions and non-float
  dtypes bail out to the stubs.
- shape builders (*_like from the prototype, full, eye, arange,
  concatenate/stack over tuple or list displays) compute the result shape.
  An explicit, resolvable dtype= is used as-is; unknown dims bail out.
- np.array/asarray build from displays, .astype converts, len() folds
  known first dims, .shape reports literal (and symbolic) dims, and
  unary -/~ keep the shape.
- comparisons broadcast shapes and return bool arrays; argmax/argsort
  drop or keep the axis; Generator/RandomState methods solve from size=.
- getitem solves integer-array (fancy) indices and take_along_axis;
  boolean masks consume n axes for one fresh unknown (?U) with lengths
  checked, and a split advanced block moves to the front; float indices
  error. setitem reuses the target solving and requires the value to
  broadcast into it. Over *Ts, leading items consume the known prefix without an
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
import sys
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
    UninhabitedType,
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
NUMTY_INTERNAL = ErrorCode(
    "numty-internal",
    "numty itself failed on this expression; the result is unchecked",
    "numty",
    sub_code_of=MISC,
)


def _numty_debug() -> bool:
    """Whether NUMTY_DEBUG=1 asks plugin exceptions to propagate."""
    return os.environ.get("NUMTY_DEBUG") == "1"


def _swallowed(ctx: object, where: str) -> None:
    """The one policy for an exception caught inside the plugin (issue #164).

    Call it first in every `except Exception:` block. A plugin bug must
    never abort the mypy run, but it must not pass for a clean result
    either:

    - NUMTY_DEBUG=1 re-raises the exception being handled, so the test
      suite (which sets it) fails on any swallowed exception and
      developers get the traceback;
    - otherwise the failure is recorded in NUMTY_TRACE with outcome
      `crash` and reported at the call site as [numty-internal]. mypy
      drops repeated identical messages on a line, so no dedup state is
      kept here (process-lifetime dedup hid errors under dmypy, #44).

    `ctx` is any hook context (or None where none is in reach: those
    sites are trace-only outside debug mode). Conditions that are
    expected in normal operation should be caught by a narrower
    `except` instead of coming through here.
    """
    exc = sys.exception()
    if _numty_debug() and exc is not None:
        raise exc
    detail = f"{type(exc).__name__}: {exc}" if exc is not None else "unknown error"
    api = getattr(ctx, "api", None)
    context = getattr(ctx, "context", None)
    if ctx is not None:
        _trace_event(ctx, where, "crash", detail)  # type: ignore[arg-type]
    fail = getattr(api, "fail", None)
    if fail is None or context is None:
        return
    try:
        fail(
            f"numty internal error in {where}: {detail} "
            "(result is unchecked; please report this)",
            context,
            code=NUMTY_INTERNAL,
        )
    except Exception:  # reporting must never be the thing that crashes
        if _numty_debug():
            raise


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
    except OSError:
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
        _swallowed(None, "_operands_have_known_shape")
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
        if fullname == "numpy.linalg._linalg.norm":
            return _norm_signature_hook
        if fullname == "numpy._core.einsumfunc.einsum":
            return _einsum_signature_hook
        if fullname == "numpy.linalg._linalg.cross":
            return _linalg_cross_signature_hook
        if fullname.startswith("numpy.") or fullname.startswith("numty."):
            return None
        return self.callsite_signature_hook

    def callsite_signature_hook(self, ctx: FunctionSigContext) -> Type:
        try:
            sig = self._callsite_check(ctx)
        except Exception:
            _swallowed(ctx, "callsite_signature_hook")
            return ctx.default_signature
        _note_rank_mismatches(ctx, sig)
        return sig

    def get_method_signature_hook(
        self, fullname: str
    ) -> Callable[[MethodSigContext], Type] | None:
        # Instance/class/static methods and __call__ get the call-site
        # conflict check plus rank notes; anything else the checker
        # resolves without shape generics stays unhooked.
        if fullname == "numpy.ndarray.reshape":
            return _reshape_signature_hook
        # Broadcast operators: suppress mechanical [operator] duplicate
        # when our hook already reports the mismatch (issue #148). Only
        # for dunders that broadcast; clean calls keep default.
        if fullname in {
            "numpy.ndarray.__add__",
            "numpy.ndarray.__radd__",
            "numpy.ndarray.__sub__",
            "numpy.ndarray.__rsub__",
            "numpy.ndarray.__mul__",
            "numpy.ndarray.__rmul__",
            "numpy.ndarray.__truediv__",
            "numpy.ndarray.__rtruediv__",
            "numpy.ndarray.__floordiv__",
            "numpy.ndarray.__rfloordiv__",
            "numpy.ndarray.__mod__",
            "numpy.ndarray.__rmod__",
            "numpy.ndarray.__pow__",
            "numpy.ndarray.__rpow__",
            "numpy.ndarray.__xor__",
            "numpy.ndarray.__rxor__",
            "numpy.ndarray.__lshift__",
            "numpy.ndarray.__rlshift__",
            "numpy.ndarray.__rshift__",
            "numpy.ndarray.__rrshift__",
            "numpy.ndarray.__divmod__",
            "numpy.ndarray.__rdivmod__",
            "numpy.ndarray.__eq__",
            "numpy.ndarray.__ne__",
            "numpy.ndarray.__lt__",
            "numpy.ndarray.__le__",
            "numpy.ndarray.__gt__",
            "numpy.ndarray.__ge__",
            "numpy.ndarray.__and__",
            "numpy.ndarray.__rand__",
            "numpy.ndarray.__or__",
            "numpy.ndarray.__ror__",
        }:
            return _broadcast_signature_hook
        # Matmul operators: suppress the mechanical [operator] duplicate
        # when the method hook already reports the mismatch (issue #140).
        # Same widening trick as broadcast; clean calls keep default.
        if fullname in {
            "numpy.ndarray.__matmul__",
            "numpy.ndarray.__rmatmul__",
            "numpy.ndarray.__imatmul__",
        }:
            return _matmul_signature_hook
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
            _swallowed(None, "_method_may_take_array")
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
        # back, same as the function call-site hook does. The registry
        # behind _fam_late_bound is empty on warm cache hits (issue #42),
        # so the marker/declared checks there keep this working
        # incrementally; a tvar owned by the method's own class is never
        # added (it stays class-bound).
        try:
            if isinstance(sig, CallableType):
                class_fullname, _, _ = fullname.rpartition(".")
                have = {v.fullname for v in sig.variables}
                missing: list[TypeVarLikeType] = []
                for formal in list(sig.arg_types) + [sig.ret_type]:
                    for tvar in _iter_tvarlikes(formal):
                        if tvar.fullname in have:
                            continue
                        if tvar.fullname in self._fam_late_bound or tvar.fullname.startswith(
                            _LATE_PREFIXES
                        ):
                            have.add(tvar.fullname)
                            missing.append(tvar)
                        elif self._resolves_to_tvar(
                            tvar.fullname
                        ) and not self._is_class_tvar(class_fullname, tvar.fullname):
                            have.add(tvar.fullname)
                            missing.append(tvar)
                if missing:
                    sig = sig.copy_modified(
                        variables=tuple(sig.variables) + tuple(missing)
                    )
        except Exception:
            _swallowed(ctx, "method_signature_hook")
            pass
        # Call-site shape conflicts (issue #53): the same shared-TypeVar
        # check free functions get. Skipped for overloads (per-candidate
        # hooks would false-positive) and guarded separately so a failure
        # here can never lose the repaired signature above.
        overloaded = isinstance(ctx.context, CallExpr) and isinstance(
            ctx.context.callee.node if hasattr(ctx.context.callee, "node") else None,
            OverloadedFuncDef,
        )
        try:
            if isinstance(sig, CallableType) and not overloaded:
                _emit_call_site_conflicts(ctx, _conflict_pairs(sig, ctx))
        except Exception:
            _swallowed(ctx, "method_signature_hook")
            pass
        if isinstance(sig, CallableType) and not overloaded:
            sig = _callsite_formula_return(sig, ctx)
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

    def _resolves_to_tvar(self, fullname: str) -> bool:
        """Whether `fullname` names a declared type variable."""
        try:
            sym = self.lookup_fully_qualified(fullname)
        except Exception:
            _swallowed(None, "_resolves_to_tvar")
            return False
        return sym is not None and isinstance(sym.node, TypeVarLikeExpr)

    def _is_class_tvar(self, class_fullname: str, fullname: str) -> bool:
        """Whether `fullname` is one of the class's own type variables."""
        try:
            sym = self.lookup_fully_qualified(class_fullname)
        except Exception:
            _swallowed(None, "_is_class_tvar")
            return False
        if sym is None or not isinstance(sym.node, TypeInfo):
            return False
        try:
            return any(v.fullname == fullname for v in sym.node.type_vars)
        except Exception:
            _swallowed(None, "_is_class_tvar")
            return False

    def _is_fam_int_name(self, fullname: str) -> bool:
        """Whether a resolved name denotes the bare-int dim (issue #62).

        `F64[int, ...]` spells a dynamic dim; the same word in a string
        spec (`F64["int ..."]`) must mean the same thing rather than
        minting a fresh implicit dim.
        """
        return fullname == "builtins.int"

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
        # String-spec words bound late (see _fam_bind_words) are missing
        # from the function's variables, so core checking would reject
        # concrete actuals against the free typevars. Add exactly those
        # back: the call then checks and infers like a written generic.
        # Names already generic (mixed bare + string) are left alone, so
        # no fullname ever appears twice. The registry behind
        # _fam_late_bound is empty on warm cache hits (issue #42), so the
        # marker/declared checks there keep this working incrementally.
        # Declared words are only added for real function callees: a
        # Callable-typed value keeps its free tvars rigid (issue #35).
        callee_node = None
        if isinstance(ctx.context, CallExpr):
            callee_node = getattr(ctx.context.callee, "node", None)
        is_func = isinstance(callee_node, (FuncDef, Decorator))
        have = {v.fullname for v in sig.variables}
        missing: list[TypeVarLikeType] = []
        for formal in list(sig.arg_types) + [sig.ret_type]:
            for tvar in _iter_tvarlikes(formal):
                if tvar.fullname in have:
                    continue
                if tvar.fullname in self._fam_late_bound or tvar.fullname.startswith(
                    _LATE_PREFIXES
                ):
                    have.add(tvar.fullname)
                    missing.append(tvar)
                elif is_func and self._resolves_to_tvar(tvar.fullname):
                    have.add(tvar.fullname)
                    missing.append(tvar)
        if missing:
            sig = sig.copy_modified(
                variables=tuple(sig.variables) + tuple(missing)
            )
        # A failure here must never lose the repaired signature above,
        # so it is guarded separately (the outer hook would otherwise
        # fall back to the default, unrepaired signature).
        try:
            _emit_call_site_conflicts(ctx, _conflict_pairs(sig, ctx))
        except Exception:
            _swallowed(ctx, "_callsite_check")
            pass
        return _callsite_formula_return(sig, ctx)

    # Familiar-style heads (_FAM_HEAD_DTYPES, defined in numty/__init__.py): the
    # dtype rides the head name, dims ride flat args. Hooked here because
    # mypy resolves `Head[...]` through Annotated-free generic application,
    # so the raw args (quoted names included) still reach us.
    def get_type_analyze_hook(
        self, fullname: str
    ) -> Callable[[AnalyzeTypeContext], Type] | None:
        dtype_fullname = _FAM_HEAD_DTYPES.get(fullname)
        if dtype_fullname is not None:

            def hook(ctx: AnalyzeTypeContext) -> Type:
                return self._fam_head(ctx, dtype_fullname)

            return hook
        # Issue #38: a bare shape-carrying generic (e.g. list[Box] where
        # Box has F64[...] fields) silently means Any shapes downstream.
        # Flag the bare use so it can be parameterized instead.
        if fullname.startswith(
            (
                "builtins.",
                "typing.",
                "collections.",
                "abc.",
                "numpy.",
                "numty.",
                "mypy.",
                "typing_extensions.",
            )
        ):
            return None
        try:
            sym = self.lookup_fully_qualified(fullname)
        except Exception:
            _swallowed(None, "get_type_analyze_hook")
            return None
        if sym is None or not isinstance(sym.node, TypeInfo):
            return None
        info = sym.node
        if not info.type_vars:
            return None
        if not _info_mentions_array(info):
            return None

        def bare_hook(ctx: AnalyzeTypeContext) -> Type:
            try:
                if list(ctx.type.args):
                    analyzed = [ctx.api.analyze_type(a) for a in ctx.type.args]
                    return ctx.api.named_type(fullname, analyzed)
            except Exception:
                _swallowed(ctx, "bare_hook")
                pass
            short = fullname.rsplit(".", 1)[-1]
            try:
                ctx.api.fail(
                    f"{short} used bare (implicit Any); parameterize as "
                    f"{short}[...] so shapes check instead of silently going Any",
                    ctx.context,
                    code=NUMTY_ANNOTATION,
                )
            except Exception:
                _swallowed(ctx, "bare_hook")
                pass
            return AnyType(TypeOfAny.from_error)

        return bare_hook

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
            # A size is never below 0. A bare `-1` is refused where it
            # is parsed; this covers every other spelling: `"2-3"`,
            # `"-1"`, `Literal[-1]`, `"-N-1"`.
            for d in dims:
                if isinstance(d, int) and d < 0:
                    raise _FamFail(f"{head} dim {d} is negative; a size cannot be below 0")
                form = _form_of(d)
                if form is not None and _form_never_size(form):
                    raise _FamFail(
                        f"{head} dim {_expr_str(form)} is negative for every size;"
                        " a size cannot be below 0"
                    )
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
        # Last-wins (issue #43): each scope's binding refreshes this.
        for name in names:
            v = resolved[name]
            if isinstance(v, TypeVarType):
                _seen_vars[v.fullname] = v
        # Per-scope constituents for #32 finish: diff(M+1)->M looks up M
        # in the defining scope via the expr's fullname, dodging the
        # global _seen_vars leak (#43).
        try:
            _expr_constituents[fullname] = {
                v.fullname: v
                for v in resolved.values()
                if isinstance(v, TypeVarType)
            }
        except Exception:
            _swallowed(ctx, "_fam_expr")
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
            if sym is not None and isinstance(sym.node, Var) and sym.node.is_final:
                # A Final int is a real dim; the normal path resolves it.
                continue
            if sym is not None and self._is_fam_int_name(
                getattr(sym.node, "fullname", "") or ""
            ):
                # The bare-int dim spelled by name: same as F64[int, ...].
                continue
            if sym is not None and not isinstance(sym.node, TypeVarLikeExpr):
                # A same-named module, variable, function or builtin is
                # not a dim. Dims live in their own namespace (jaxtyping
                # style), so mint an implicit dim instead of erroring
                # (issue #62): `F64["time freq"]` after `import time`,
                # `F64["n d"]` with a module-level `n = 10`.
                minted[word] = self._fam_ensure_implicit(
                    ctx, analyzer, scope, word, node
                )
                continue
            if sym is None:
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

        Issue #145: a local pin (`r: F64["batch out"]`) is analyzed in a
        scope that does not see the signature's bindings, so a fresh mint
        there has the same display name but a different id and native
        subtype checking fails on identical shapes. Reuse the signature's
        object for the same (function namespace, word) via bind_existing,
        so one word is one dim per function, like declared TypeVars.
        Different functions keep independent vars via the namespace key.

        Issue #146: the fullname itself must carry the namespace
        (`__numty_implicit__.<ns>.<word>`), not just the cache key. A
        shared `__numty_implicit__.myhid` fullname makes two functions'
        dims (and any `2*myhid` expr forms built on them) the same
        variable with the same per-scope id, so cold (both modules in
        one process) confuses caller and callee objects and bails to
        Any while warm (callee from cache) succeeds. Namespaced
        fullnames keep every scope's dims distinct; the `_LATE_PREFIXES`
        marker still matches.
        """
        try:
            ns = scope.namespace
        except Exception:
            _swallowed(ctx, "_fam_ensure_implicit")
            ns = ""
        key = (ns, word)
        fullname = f"__numty_implicit__.{ns}.{word}" if ns else f"__numty_implicit__.{word}"
        existing = scope.get_binding(fullname)
        if existing is not None:
            return existing
        # Same function, different scope tree (signature vs body): reuse
        # the signature's object instead of minting a lookalike.
        cached = _implicit_scope_vars.get(key)
        if cached is not None:
            # A stale entry from another process lifetime (or a pre-#146
            # cache with an unnamespaced fullname) must not shadow the
            # current scope's own dim: only reuse when the fullnames
            # agree, else fall through and mint fresh.
            try:
                if getattr(cached, "fullname", None) == fullname:
                    try:
                        scope.bind_existing(cached)
                    except Exception:
                        _swallowed(ctx, "_fam_ensure_implicit")
                        pass
                    return cached
            except Exception:
                _swallowed(ctx, "_fam_ensure_implicit")
                pass
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
        _implicit_scope_vars[key] = bound
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
        if fullname == "builtins.divmod":
            return divmod_function_hook
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
        return substituted_attribute_hook

    def _method_hook(self, fullname: str) -> Callable[[MethodContext], Type] | None:
        if fullname == "numpy.ndarray.__matmul__":
            return matmul_method_hook
        if fullname == "numpy.ndarray.__rmatmul__":
            return rmatmul_method_hook
        if fullname == "numpy.ndarray.__imatmul__":
            return imatmul_method_hook
        if fullname in {
            "numpy.ndarray.__add__",
            "numpy.ndarray.__radd__",
        }:
            return partial(broadcast_method_hook, op="add")
        if fullname in {
            "numpy.ndarray.__mul__",
            "numpy.ndarray.__rmul__",
            "numpy.ndarray.__xor__",
            "numpy.ndarray.__rxor__",
            "numpy.ndarray.__lshift__",
            "numpy.ndarray.__rlshift__",
            "numpy.ndarray.__rshift__",
            "numpy.ndarray.__rrshift__",
        }:
            return broadcast_method_hook
        if fullname in {
            "numpy.ndarray.__sub__",
            "numpy.ndarray.__rsub__",
        }:
            return subtract_method_hook
        if fullname in {
            "numpy.ndarray.__pow__",
            "numpy.ndarray.__rpow__",
            "numpy.ndarray.__floordiv__",
            "numpy.ndarray.__rfloordiv__",
            "numpy.ndarray.__mod__",
            "numpy.ndarray.__rmod__",
        }:
            return floordiv_mod_pow_method_hook
        if fullname in {
            "numpy.ndarray.__divmod__",
            "numpy.ndarray.__rdivmod__",
        }:
            return divmod_method_hook
        if fullname in {
            "numpy.ndarray.__iadd__",
            "numpy.ndarray.__isub__",
            "numpy.ndarray.__imul__",
            "numpy.ndarray.__itruediv__",
            "numpy.ndarray.__ifloordiv__",
            "numpy.ndarray.__imod__",
            "numpy.ndarray.__ipow__",
            "numpy.ndarray.__iand__",
            "numpy.ndarray.__ior__",
            "numpy.ndarray.__ixor__",
            "numpy.ndarray.__ilshift__",
            "numpy.ndarray.__irshift__",
        }:
            return inplace_method_hook
        if fullname == "numpy.ndarray.clip":
            return clip_method_hook
        if fullname == "numpy.ndarray.round":
            return round_method_hook
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
        if fullname == "numpy.ndarray.__setitem__":
            return setitem_method_hook
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
            return partial(allany_method_hook, op=short)
        if fullname.startswith("numpy.ndarray.") and short in {
            "argmax",
            "argmin",
            "argsort",
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
            return partial(reduction_method_hook, op=short)
        # ufunc methods (np.add.reduce, .accumulate, .outer, .reduceat):
        # axis-checked shape solves; .at keeps the stubs (issue #142).
        if short in {"reduce", "accumulate", "outer", "reduceat"} and (
            fullname.startswith("numpy.ufunc.")
            or fullname.startswith("numpy._typing._ufunc.")
        ):
            return partial(ufunc_method_hook, op=short)
        if fullname.startswith("numpy.ndarray.") and short in {"cumsum", "cumprod"}:
            return partial(cumsum_method_hook, op=short)
        if fullname.startswith("numpy.ndarray.") and short == "ravel":
            return ravel_method_hook
        if fullname.startswith("numpy.ndarray.") and short == "flatten":
            return flatten_method_hook
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
    ?U unknowns never prove equal, not even to themselves (issue #63):
    one mask site serves every comprehension element, so sharing ?U0
    would claim false (3, 3) matmuls. The result bails instead.
    """
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
    if _dim_eq(a, b):
        return False
    if isinstance(a, int) and isinstance(b, int):
        return a != b
    fa = _resolve_dim(a)
    fb = _resolve_dim(b)
    return _forms_provably_ne(fa, fb, nonzero)


def _forms_provably_ne(fa: object, fb: object, nonzero: bool = False) -> bool:
    """Provably-unequal forms: same linear vars, different consts.

    `N+1` vs `N` and `N+2` vs `N` can never be equal, and neither can
    same-division plus diffs (`N//2+1` vs `N//2`). Different vars,
    divs vs linears or unknowns stay unprovable (issue #56).

    Coefficient diffs are different: `2*N` vs `N` and `2*M*N` vs `M*N`
    are equal exactly when a variable is 0, and an empty axis is a
    legal size, so numpy accepts `np.vdot` of `(2, M)` and `(M,)` at
    `M == 0`. They count only under `nonzero=True`, which reshape
    passes because its literal-factor check already reads totals up
    to a zero factor (issues #63, #137). Every other caller keeps to
    what holds for all sizes.
    """
    if fa is None or fb is None:
        return False
    if fa == fb:
        return False
    try:
        if _form_has_unknown(fa) or _form_has_unknown(fb):
            return False
    except Exception:
        _swallowed(None, "_forms_provably_ne")
        pass
    if isinstance(fa, _LinForm) and isinstance(fb, _LinForm):
        if fa.coeffs == fb.coeffs and fa.const != fb.const:
            return True
        if (
            nonzero
            and len(fa.coeffs) == 1
            and len(fb.coeffs) == 1
            and fa.coeffs[0][0] == fb.coeffs[0][0]
            and fa.const == fb.const
            and fa.coeffs[0][1] != fb.coeffs[0][1]
        ):
            return True
        return False
    if isinstance(fa, _MonoForm) and isinstance(fb, _MonoForm):
        return nonzero and fa.vars == fb.vars and fa.coeff != fb.coeff
    if isinstance(fa, _DivForm) and isinstance(fb, _DivForm):
        return fa.inner == fb.inner and fa.div == fb.div and fa.plus != fb.plus
    return False


def _form_never_size(form: object) -> bool:
    """A form below 0 whatever its variables are: no array has that size.

    Variables are sizes, so never negative: a linear form with no
    positive coefficient is at most its constant. `-N-1` and `-3`
    qualify; `-N` (0 on an empty axis) and `N-1` do not.
    """
    if isinstance(form, _LinForm):
        return form.const < 0 and all(c < 0 for _, c in form.coeffs)
    return False


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
# Implicit dims by defining function (issue #145): (namespace, word) ->
# the TypeVarType minted for the signature. A local `r: F64["batch out"]`
# is analyzed in a scope that does not see the signature's bindings, so a
# fresh mint there has the same display name but a different id and native
# subtype checking fails on identical shapes. Reusing the signature's
# object (via bind_existing) keeps one word one dim per function, like
# declared TypeVars, while different functions keep independent vars.
_implicit_scope_vars: dict[tuple[str, str], TypeVarLikeType] = {}
# Per-expr constituent objects by scope (issue #32 finish): expr fullname
# -> {constituent fullname -> TypeVarType object from the defining scope}.
# Lets diff(M+1) -> M return the *current* function's M instead of the
# process-wide _seen_vars entry (which caused N@first vs N@second in
# issue #43).
_expr_constituents: dict[str, dict[str, TypeVarType]] = {}

# Synthetic tvar namespaces the plugin mints (issue #42): implicit dims
# for undeclared string-spec words, and formula-named expression dims.
# Anything under these markers in a signature is late-bound by
# construction, with or without the process-local `_fam_late_bound` set
# (which a module loaded from mypy's incremental cache never fills).
_LATE_PREFIXES = ("__numty_implicit__.", "__numty_expr__.")
_EXPR_PREFIX = "__numty_expr__."


def _eval_form(node: ast.AST) -> _LinForm | _DivForm | _MonoForm:
    """Rebuild a canonical form from its repr AST (issue #42).

    Expr fullnames embed `{form!r}`, a NamedTuple repr, so a type loaded
    from the incremental cache carries its own form. Only the three
    canonical shapes with int/str/tuple structure are accepted; anything
    else raises and the caller falls back to the registry.
    """

    def num(n: ast.AST) -> int:
        if type(n) is ast.Constant and type(n.value) is int:
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = num(n.operand)
            return v if isinstance(n.op, ast.UAdd) else -v
        raise ValueError("not an int")

    def strs(n: ast.AST) -> tuple[str, ...]:
        if not isinstance(n, ast.Tuple):
            raise ValueError("not a tuple")
        out = []
        for e in n.elts:
            if type(e) is not ast.Constant or type(e.value) is not str:
                raise ValueError("not a str")
            out.append(e.value)
        return tuple(out)

    def coeffs(n: ast.AST) -> tuple[tuple[str, int], ...]:
        if not isinstance(n, ast.Tuple):
            raise ValueError("not a tuple")
        out = []
        for e in n.elts:
            if not isinstance(e, ast.Tuple) or len(e.elts) != 2:
                raise ValueError("not a pair")
            name, coeff = e.elts
            if type(name) is not ast.Constant or type(name.value) is not str:
                raise ValueError("not a name")
            out.append((name.value, num(coeff)))
        return tuple(out)

    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError("not a form")
    if node.args:
        raise ValueError("positional args")
    kw = {k.arg: k.value for k in node.keywords if k.arg is not None}
    if set(kw) != set() and any(k.arg is None for k in node.keywords):
        raise ValueError("star kwargs")
    kind = node.func.id
    if kind == "_LinForm" and set(kw) == {"coeffs", "const"}:
        return _LinForm(coeffs(kw["coeffs"]), num(kw["const"]))
    if kind == "_DivForm" and set(kw) == {"inner", "div", "plus"}:
        return _DivForm(_eval_form(kw["inner"]), num(kw["div"]), num(kw["plus"]))
    if kind == "_MonoForm" and set(kw) == {"coeff", "vars"}:
        return _MonoForm(num(kw["coeff"]), strs(kw["vars"]))
    raise ValueError("unknown form")


def _form_from_fullname(fullname: str) -> _LinForm | _DivForm | _MonoForm | None:
    """Parse the canonical form embedded in an expr fullname (issue #42).

    A type deserialized from mypy's cache has no entry in the
    process-local `_expr_forms` registry; its fullname still spells the
    form, so read it back instead of degrading to Any.
    """
    if not fullname.startswith(_EXPR_PREFIX):
        return None
    src = fullname[len(_EXPR_PREFIX):]
    if len(src) > 4000:
        return None
    try:
        tree = ast.parse(src, mode="eval")
    except (SyntaxError, ValueError):
        return None
    try:
        form = _eval_form(tree.body)
    except Exception:
        _swallowed(None, "_form_from_fullname")
        return None
    _expr_forms[fullname] = form
    return form


def _form_of(d: int | Type) -> _LinForm | _DivForm | _MonoForm | None:
    """Canonical form of a dim var: registry first, fullname fallback."""
    if not isinstance(d, TypeVarType):
        return None
    form = _expr_forms.get(d.fullname)
    if form is not None:
        return form
    return _form_from_fullname(d.fullname)


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
                _swallowed(ctx, "_current_scope_vars")
                continue
    except Exception:
        _swallowed(ctx, "_current_scope_vars")
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
    by fullname so materialization can return the real object. Last-wins,
    not first-wins (issue #43): sequential checking refreshes this to the
    current function's object (its inner call resolves before any outer
    call in the same body materializes).
    """
    if isinstance(d, int):
        return _LinForm((), d)
    if isinstance(d, TypeVarType):
        form = _form_of(d)
        if form is not None:
            return form
        _seen_vars[d.fullname] = d
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
            out += f"{form.plus:+d}"
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
    ctx: FunctionContext | MethodContext | AttributeContext, op: str, index: int, form: _LinForm | _DivForm | _MonoForm
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
            _swallowed(ctx, "_expr_var")
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
        # Issue #146: the process-wide `_expr_constituents`/`_seen_vars`
        # registries hold whichever module was checked first, so a stale
        # entry (another scope's object for the same fullname) must never
        # shadow the current call's own object. Check the current call
        # first, then the input expr's constituents, then the registry.
        # 1) Current call's own object (reshape/split second-function
        # leak, #146 cold-vs-warm): prefer it over the process-wide
        # registry.
        try:
            current = _current_scope_vars(ctx)
            if want in current:
                return current[want]
            # 1b) Transitive (issue #43): the current call's args hold an
            # expr (e.g. repeat's N*2 feeding reshape's -1), not the bare
            # var. That expr's constituents were just recorded by the
            # inner call in this same body, so they are the current
            # function's objects; the global registry below still holds
            # whichever function was checked first.
            for tv in current.values():
                if isinstance(tv, TypeVarType):
                    consts = _expr_constituents.get(tv.fullname)
                    if consts is not None and want in consts:
                        return consts[want]
        except Exception:
            _swallowed(ctx, "_materialize_dim")
            pass
        # 2) diff(M+1)->M: the input expr knows its own scope's M.
        if isinstance(prefer, TypeVarType):
            consts = _expr_constituents.get(prefer.fullname)
            if consts is not None and want in consts:
                return consts[want]
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


def _reshape_total_form(dims: list[int | Type]) -> object:
    """Total elements for reshape legality: single dim resolves directly
    (so N+1, N+2 sums are comparable), multi-dim uses the pure-product
    canonical form; None when unrepresentable (issue #63)."""
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
    raw: list[tuple[Type, int | Type, int | None]] = []
    uidx = -1
    for k, fd in enumerate(fdims):
        if isinstance(fd, int):
            continue
        if isinstance(get_proper_type(fd), UnpackType):
            uidx = k
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
        upack = get_proper_type(fdims[uidx])
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
                        found = _expr_constituents.get(adim.fullname, {}).get(a_full)
                if found is None:
                    found = _seen_vars.get(a_full)
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
    obj_by_fullname: dict[str, TypeVarType] = {}
    for v in bindings.values():
        if isinstance(v, TypeVarType):
            obj_by_fullname.setdefault(v.fullname, v)
    for consts in _expr_constituents.values():
        for k, v in consts.items():
            obj_by_fullname.setdefault(k, v)
    for k, v in _seen_vars.items():
        obj_by_fullname.setdefault(k, v)
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
            _swallowed(ctx, "_substitute_return_exprs")
            new_shape = TupleType(new_items, shape.fallback, line=shape.line, column=shape.column)
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
            ctx.api.note(message, where, code=ARG_TYPE)
    except Exception:
        _swallowed(ctx, "_note_rank_mismatches")
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


def _builder_shape(
    ctx: FunctionContext | MethodContext,
    tp: Type,
    expr: Expression | None,
    default: Type,
    shaped_stub: bool = True,
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

    A size written as arithmetic over known sizes (issue #186) is folded
    with the dim algebra, like reshape targets (#184): `u.shape[0] *
    u.shape[1]`, `2 * u.shape[0]`, `u.shape[0] + 1`. What the algebra
    cannot spell (a sum times a sum) is a fresh unknown, which is still
    a size.

    Builder folding is side-effect free for the dim registries: it reuses
    the shared form ids (so an inferred `N+1` is the same variable as an
    annotated one) but never overwrites the per-scope constituents that
    later slices/diffs need (`x[1:]` on `F64[N+1]` must give the current
    function's `N`, not the builder site's).
    """
    dims = _shape_from_arg(tp, expr)
    if dims is not None:
        return list(dims)
    if expr is not None and not isinstance(expr, (TupleExpr, ListExpr, StarExpr)):
        if isinstance(expr, OpExpr) and _fold_int(expr) is None:
            try:
                folded = _builder_fold_dim_expr(ctx, expr, 0)
                if isinstance(folded, int):
                    return [folded]
                if isinstance(folded, TypeVarType):
                    return [folded]
            except Exception:
                _swallowed(None, "_builder_shape")
                pass
        dim = _dim_value(tp)
        if isinstance(dim, (int, TypeVarType)):
            return [dim]
        return None
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
        if isinstance(item, OpExpr) and _fold_int(item) is None:
            try:
                folded = _builder_fold_dim_expr(ctx, item, i)
                if isinstance(folded, int):
                    out.append(folded)
                    continue
                if isinstance(folded, TypeVarType):
                    out.append(folded)
                    continue
            except Exception:
                _swallowed(None, "_builder_shape")
                pass
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


def _movable_parts(tp: Type) -> tuple[list[int | Type], Type] | None:
    """(dims, dtype) for ops that only move axes around (transposes).

    Those never read a size, so an axis of unknown size (`Any`, as a
    dynamic reshape leaves) moves like any other. Without this the
    stubs answer: `.T` and `transpose` return Self there, the shape
    unreversed.
    """
    parts = _ndarray_parts(tp) or _mixed_parts(tp)
    if parts is not None:
        return parts
    proper = get_proper_type(tp)
    if not (
        isinstance(proper, Instance)
        and proper.type.fullname == NDARRAY_FULLNAME
        and len(proper.args) == 2
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
        elif isinstance(plain, AnyType):
            dims.append(plain)
        else:
            return None
    return dims, proper.args[1]


def _unknown_axis_parts(tp: Type) -> tuple[list[int | Type], Type] | None:
    """(dims, dtype) of a shape with an axis of unknown size (`Any`).

    For ops that drop or move axes without reading a size. The stubs'
    answer there is wrong: Self for swapaxes and moveaxis (the shape
    unmoved), a signed dtype for `sum` of uint8. Mixed shapes (a
    literal next to a plain `int`) keep their own rule, see #34.
    """
    if _mixed_parts(tp) is not None:
        return None
    parts = _movable_parts(tp)
    if parts is None or any(_is_plain_int(d) for d in parts[0]):
        # A widened mixed shape, (int, Any): moving its axes would stop
        # `x = np.moveaxis(x, 0, 1)` from fitting x again.
        return None
    return parts


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


_recheck_depth = 0


def _is_degraded_operand(tp: Type) -> bool:
    """An argument type that lost what its expression would show alone.

    Any, or an ndarray whose shape is not a fixed tuple or whose dtype
    is unknown: what an inline numpy call collapses to when mypy infers
    it against one overload's formal (see _undegraded_args).
    """
    proper = get_proper_type(tp)
    if isinstance(proper, AnyType):
        return True
    if not isinstance(proper, Instance) or proper.type.fullname != NDARRAY_FULLNAME:
        return False
    if len(proper.args) != 2:
        return True
    shape = get_proper_type(proper.args[0])
    if not isinstance(shape, TupleType):
        return True
    dtype = get_proper_type(proper.args[1])
    if isinstance(dtype, AnyType):
        return True
    return isinstance(dtype, Instance) and any(
        isinstance(get_proper_type(arg), AnyType) for arg in dtype.args
    )


def _undegraded_args(
    ctx: FunctionContext | MethodContext,
) -> FunctionContext | MethodContext:
    """Give a hook the operands as written, not as one overload saw them.

    mypy infers each argument against the formal of the overload it is
    trying. An inline builder with a dtype the first overload does not
    take (`f64 + np.ones((7, 7), dtype=np.complex64)` against the
    float64 overload of `__add__`) then falls through to the builder's
    catch-all overload and arrives here as `ndarray[Any, ...]`. The
    float64 overload accepts that, and the hook saw an operand with no
    shape and no dtype: mismatched shapes passed, and the result kept
    the float64 dtype of an overload that should never have matched
    (issue #166).

    So an inline call or operator whose type arrived degraded is
    re-inferred on its own, with no type context, and the hook gets
    that answer when it is a known array. Names and attributes are
    left alone (their type does not depend on context), which also
    keeps the cost off ordinary code: a precise operand is never
    re-inferred. Nested re-inference is capped, since each level
    re-checks the subexpression.
    """
    global _recheck_depth
    if _recheck_depth >= 2:
        return ctx
    new_types: list[list[Type]] | None = None
    for i, (exprs, types) in enumerate(zip(ctx.args, ctx.arg_types)):
        for j, (expr, tp) in enumerate(zip(exprs, types)):
            if not isinstance(expr, (CallExpr, OpExpr)) or not _is_degraded_operand(tp):
                continue
            _recheck_depth += 1
            try:
                fresh = ctx.api.get_expression_type(expr)
            except Exception:
                _swallowed(ctx, "_undegraded_args")
                continue
            finally:
                _recheck_depth -= 1
            if _is_degraded_operand(fresh) or _ndarray_parts(fresh) is None:
                continue
            if new_types is None:
                new_types = [list(ts) for ts in ctx.arg_types]
            new_types[i][j] = fresh
    if new_types is None:
        return ctx
    return ctx._replace(arg_types=new_types)


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
            try:
                if not isinstance(ctx, AttributeContext):
                    ctx = _undegraded_args(ctx)
                    operands = [t for ts in ctx.arg_types for t in ts]
                    if isinstance(ctx, MethodContext):
                        operands.append(ctx.type)
                result = hook(ctx)
            except Exception:
                _swallowed(ctx, fullname)
                # Crash barrier (issue #41): no shape hook may abort the
                # whole mypy run with INTERNAL ERROR, hiding every later
                # diagnostic. Reported above ([numty-internal], issue
                # #164; NUMTY_DEBUG=1 re-raises), then bail to the stubs.
                result = default
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
    shape_expr = _single_shape_expr(ctx)
    if shape_arg is None:
        # numpy >= 2 spellings like np.zeros(shape=(2, 3)).
        named = _formal_pair(ctx, "shape")
        if named is not None:
            shape_arg, shape_expr = named
    if shape_arg is None:
        return ctx.default_return_type
    dims = _builder_shape(ctx, shape_arg, shape_expr, ctx.default_return_type)
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
    ints differ or linear forms are provably unequal (N vs N-1 from
    np.diff, N vs N+2 from pad, issue #32), and (False, None) to bail
    on anything unprovable. A plain `int` (dynamic) dim is assumed to
    match: it never reaches the output, and unknown is not an error."""
    if _dim_eq(x, y) or _is_plain_int(x) or _is_plain_int(y):
        return True, None
    if isinstance(x, int) and isinstance(y, int):
        return False, f"matmul inner dimensions disagree: {x} vs {y}"
    if _dims_provably_ne(x, y):
        return False, f"matmul inner dimensions disagree: {_dim_str(x)} vs {_dim_str(y)}"
    return False, None


def _matmul_result(
    a: list[int | Type],
    b: list[int | Type],
    ctx: FunctionContext | MethodContext | None = None,
) -> tuple[list[int | Type] | None, str | None]:
    """Output dims on success; (None, None) to bail; (None, msg) on error.

    An unprovable inner pair (N vs M, ?U vs ?U') is not a clash and the
    output dims do not depend on it (issue #56), so the outer dims are
    still returned instead of bailing the whole result to Any. The one
    exception is the same ?U on both sides (issue #63): one mask site
    serves every comprehension element, so equality would falsely prove
    (3, 3) — bail instead.
    """
    if not a or not b:
        return None, None
    if len(a) == 1 and len(b) == 1:
        if _is_same_unknown(a[0], b[0]):
            return None, None
        _, message = _matmul_inner(a[0], b[0])
        if message is not None:
            return None, message
        return [], None
    if len(a) == 1:
        if _is_same_unknown(a[0], b[-2]):
            return None, None
        _, message = _matmul_inner(a[0], b[-2])
        if message is not None:
            return None, message
        return [*b[:-2], b[-1]], None
    if len(b) == 1:
        if _is_same_unknown(a[-1], b[0]):
            return None, None
        _, message = _matmul_inner(a[-1], b[0])
        if message is not None:
            return None, message
        return a[:-1], None
    if _is_same_unknown(a[-1], b[-2]):
        return None, None
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
        if not dims:
            # 1-D @ 1-D is a scalar, like dot/inner/vdot (issue #62).
            return _promoted_scalar(ctx, dtype_a, dtype_b)
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

    Match on equality, error when both ints differ or linear forms are
    provably unequal (N+1 vs N, issue #139), and (False, None) to bail
    on anything unprovable (N vs M, ?U vs ?U'). Mirrors _matmul_inner
    with an op-specific message.
    """
    if _dim_eq(x, y) or _is_plain_int(x) or _is_plain_int(y):
        return True, None
    if isinstance(x, int) and isinstance(y, int):
        return False, f"{op} contracted dimensions disagree: {x} vs {y}"
    if _dims_provably_ne(x, y):
        return False, f"{op} contracted dimensions disagree: {_dim_str(x)} vs {_dim_str(y)}"
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
    if not dims_a or not dims_b:
        # 0-d operand: dot is broadcast multiplication (issue #41).
        return _solve_broadcast(ctx, pair[0], pair[1])
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
    if not dims_a or not dims_b:
        # 0-d operand: dot is broadcast multiplication (issue #41).
        return _solve_broadcast(ctx, left, right)
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


def matmul_method_hook(ctx: MethodContext) -> Type:
    """ndarray.__matmul__ twin of np.matmul: same solver on (self, arg).

    Issue #140: `@` used to fall through to numpy's default checking,
    so wrong shapes passed silently and right shapes dissolved to Any.
    """
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    left, right = ctx.type, pairs[0][0]
    if not _matmul_has_dims(ctx, [left, right]):
        return ctx.default_return_type
    return _solve_matmul(ctx, left, right)


def rmatmul_method_hook(ctx: MethodContext) -> Type:
    """ndarray.__rmatmul__: reflected operands, same solver (issue #140)."""
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    left, right = pairs[0][0], ctx.type
    if not _matmul_has_dims(ctx, [left, right]):
        return ctx.default_return_type
    return _solve_matmul(ctx, left, right)


def imatmul_method_hook(ctx: MethodContext) -> Type:
    """ndarray.__imatmul__: check like matmul, keep the stubs' return.

    A legal `a @= b` computes into `a`, so the result shape equals the
    target's; like the other in-place hooks this keeps the default
    return and only adds the [numty-shape] error (issue #140).
    """
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    left, right = ctx.type, pairs[0][0]
    if not _matmul_has_dims(ctx, [left, right]):
        return ctx.default_return_type
    _solve_matmul(ctx, left, right)
    return ctx.default_return_type


def inner_function_hook(ctx: FunctionContext) -> Type:
    pair = _two_arrays(ctx)
    if pair is None:
        return ctx.default_return_type
    parts_a = _ndarray_parts(pair[0])
    parts_b = _ndarray_parts(pair[1])
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    (dims_a, dtype_a), (dims_b, dtype_b) = parts_a, parts_b
    if not dims_a or not dims_b:
        # 0-d operand: inner is broadcast multiplication (issue #41).
        return _solve_broadcast(ctx, pair[0], pair[1])
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
    form_a = _reshape_total_form(dims_a)
    form_b = _reshape_total_form(dims_b)
    if _forms_provably_ne(form_a, form_b):
        assert form_a is not None and form_b is not None
        ctx.api.fail(
            f"vdot operands have different sizes: {_expr_str(form_a)} vs {_expr_str(form_b)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    # Unknown (?U) totals never prove equal, not even to themselves
    # (issue #63): bail instead of claiming a scalar.
    for form in (form_a, form_b):
        if form is not None and _form_has_unknown(form):
            return ctx.default_return_type
    for d in (*dims_a, *dims_b):
        if isinstance(d, TypeVarType) and _is_unknown_fullname(d.fullname):
            return ctx.default_return_type
        f = _resolve_dim(d)
        if f is not None and _form_has_unknown(f):
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
                    f"tensordot has duplicate axis {bad} in {tuple(axes_a + axes_b)} "
                    f"(shapes {_dims_str(dims_a)} vs {_dims_str(dims_b)})",
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
    norm = _normalize_axis_0d(axis[0], len(dims))
    if norm is None:
        ctx.api.fail(
            f"take axis {axis[0]} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    # 0-d arrays flatten to one element (issue #131): bounds check
    # against size 1, result shape from the empty prefix/suffix.
    size = dims[norm] if dims else 1
    # numpy skips the index check when a dim before the axis is 0:
    # there is no row to take from.
    empty_outer = any(isinstance(d, int) and d == 0 for d in dims[:norm])
    if not empty_outer and not _take_in_bounds(
        ctx, pair, size, f"axis {axis[0]} of size {size}", dims
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
            return _gradient_result(ctx, pairs[0][0], dims, 0)
        return ctx.default_return_type
    if not _check_each_axis(ctx, dims, axis, "gradient") or not _check_unique_axes(
        ctx, dims, axis, "gradient"
    ):
        return ctx.default_return_type
    if len(axis) != 1:
        return ctx.default_return_type
    norm = _normalize_axis(axis[0], len(dims))
    if norm is None:
        return ctx.default_return_type
    return _gradient_result(ctx, pairs[0][0], dims, norm)


def _gradient_result(
    ctx: FunctionContext, operand: Type, dims: list[int | Type], axis: int
) -> Type:
    """gradient along one axis: the operand's shape, in a floating dtype.

    Central differences need two points, so a known size below 2 on
    the axis is numpy's "Shape of array too small" error. Integers
    come back float64, floats and complex keep their dtype, and bool
    is rejected by numpy (bail).
    """
    size = dims[axis]
    if isinstance(size, int) and size < 2:
        ctx.api.fail(
            f"gradient needs at least 2 elements along axis {axis}, got {size} "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    proper = get_proper_type(operand)
    if not isinstance(proper, Instance) or len(proper.args) != 2:
        return ctx.default_return_type
    ks = _dtype_kind_size(proper.args[1])
    if ks is None:
        return ctx.default_return_type
    if ks[0] in ("f", "c"):
        return operand
    if ks[0] in ("i", "u"):
        f64 = _construct_dtype(ctx, "numpy.float64")
        if f64 is None:
            return ctx.default_return_type
        return proper.copy_modified(args=[proper.args[0], f64])
    return ctx.default_return_type


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
    if isinstance(proper, Instance) and proper.last_known_value is not None:
        return _string_value(proper.last_known_value)
    return None


def _convolve_solve(
    ctx: FunctionContext, op: str, a: Type, b: Type, mode: str | None
) -> Type:
    parts_a = _ndarray_parts(a)
    parts_b = _ndarray_parts(b)
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    (dims_a, dtype_a), (dims_b, dtype_b) = parts_a, parts_b
    if op == "convolve":
        # convolve flattens 0-d arrays to one element (issue #131);
        # correlate rejects them ("object of too small depth").
        if len(dims_a) == 0:
            dims_a = [1]
        if len(dims_b) == 0:
            dims_b = [1]
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
        if mode == "full":
            size: int | Type = _dim_add_const(
                ctx, op, 0, _dim_add2(ctx, op, 0, na, nb), -1
            )
        elif mode == "same":
            size = _convolve_same_size(ctx, op, na, nb)
        else:
            size = _convolve_valid_size(ctx, op, na, nb)
        return _matmul_promoted(ctx, [size], dtype_a, dtype_b)
    if mode == "full":
        n = na + nb - 1
    elif mode == "same":
        n = max(na, nb)
    else:
        n = max(na, nb) - min(na, nb) + 1
    return _matmul_promoted(ctx, [n], dtype_a, dtype_b)


def _convolve_same_size(
    ctx: FunctionContext, op: str, na: int | Type, nb: int | Type
) -> int | Type:
    """Symbolic `same`-mode length: max(na, nb), else a fresh unknown.

    The order of two symbolic lengths is unprovable in general; only an
    exact match keeps the dim, everything else stays honest instead of
    claiming the `full` length.
    """
    if _dim_eq(na, nb):
        return na
    return _fresh_unknown(ctx, op, 0)


def _convolve_valid_size(
    ctx: FunctionContext, op: str, na: int | Type, nb: int | Type
) -> int | Type:
    """Symbolic `valid`-mode length: |na - nb| + 1, else a fresh unknown.

    An equal pair gives 1; a constant difference folds exactly, anything
    whose sign is unprovable stays honest instead of claiming `full`.
    """
    fa = _resolve_dim(na)
    fb = _resolve_dim(nb)
    if fa is None or fb is None:
        return _fresh_unknown(ctx, op, 0)
    if not isinstance(fa, _LinForm) or not isinstance(fb, _LinForm):
        return _fresh_unknown(ctx, op, 0)
    coeffs: dict[str, int] = {}
    for name, coeff in fa.coeffs:
        coeffs[name] = coeffs.get(name, 0) + coeff
    for name, coeff in fb.coeffs:
        coeffs[name] = coeffs.get(name, 0) - coeff
    const = fa.const - fb.const
    coeffs = {name: coeff for name, coeff in coeffs.items() if coeff}
    if not coeffs:
        return abs(const) + 1
    return _fresh_unknown(ctx, op, 0)


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
    """complex64 for float16/float32, else complex128 (issue #61)."""
    name = _dtype_fullname(dtype)
    if name is None:
        return _default_dtype(ctx)
    single = ("numpy.float16", "numpy.float32", "numpy.complex64")
    out = "numpy.complex64" if name in single else "numpy.complex128"
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


def irfft_function_hook(ctx: FunctionContext) -> Type:
    """np.fft.irfft: n points on the axis (default 2*(m-1)), real dtype.

    The stubs keep the input's shape and answer float64 whatever comes
    in. numpy returns float32 for float32/complex64 input and float16
    for float16; everything else is float64. A symbolic axis without n
    stays with the stubs.
    """
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
            f"irfft axis {axis} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims)
    if isinstance(n, int):
        out[target] = n
    else:
        m = out[target]
        if not isinstance(m, int):
            return ctx.default_return_type
        if m < 2:
            ctx.api.fail(
                f"irfft of {m} point{'s' if m != 1 else ''} along axis {target} has no "
                f"output points (2*(m-1) = 0); pass n= (shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        out[target] = 2 * (m - 1)
    name = _dtype_fullname(dtype)
    if name is None:
        return ctx.default_return_type
    if name == "numpy.float16":
        real = "numpy.float16"
    elif name in ("numpy.float32", "numpy.complex64"):
        real = "numpy.float32"
    else:
        real = "numpy.float64"
    built = _construct_dtype(ctx, real)
    if built is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, out, built)


def _rfft_size(
    ctx: FunctionContext, op: str, index: int, dim: int | Type
) -> int | Type:
    return _dim_div(ctx, op, index, dim, 2, 1)


def _rfft_complex_input(ctx: FunctionContext, op: str) -> Type:
    """The real transforms take real input only (numpy: TypeError)."""
    ctx.api.fail(
        f"{op} needs real input, got a complex array (use fft instead)",
        ctx.context,
        code=NUMTY_BROADCAST,
    )
    return ctx.default_return_type


def _rfft_solve(
    ctx: FunctionContext, op: str, arg: Type, axis: int, n: int | None
) -> Type:
    parts = _ndarray_parts(arg)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    if _dtype_base(dtype) == "c":
        return _rfft_complex_input(ctx, op)
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
    if rfft and _dtype_base(dtype) == "c":
        return _rfft_complex_input(ctx, op)
    if s is None and axes is None:
        # Default axes: the last two for fft2/ifft2, every axis otherwise.
        need = 2 if op in ("fft2", "rfft2") else 1 if rfft else 0
        if ndim < need:
            ctx.api.fail(
                f"{op} needs at least {need} dimension{'s' if need > 1 else ''}, "
                f"got {ndim} (shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        if ndim == 0:
            # fftn over no axes hands the 0-d input back untouched.
            return arg
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
    if not eff:
        # axes=(): no transform. fftn returns the input untouched;
        # rfftn raises IndexError at runtime.
        if rfft:
            ctx.api.fail(
                f"{op} with empty axes raises IndexError",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        part = _ndarray_parts(arg)
        assert part is not None
        return _ndarray_type(ctx, out, part[1])
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


def rfft2_function_hook(ctx: FunctionContext) -> Type:
    """np.fft.rfft2: real 2-D fft over the last two axes, no s.

    Same contract as fft2_function_hook; the last axis halves.
    """
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parsed = _fft_s_axes(ctx)
    if parsed is None:
        return ctx.default_return_type
    s, axes = parsed
    if isinstance(s, str) or isinstance(axes, str) or s is not None:
        return ctx.default_return_type
    return _fftn_solve(ctx, "rfft2", pairs[0][0], axes, s, True)


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
    target = _normalize_axis_0d(axis, len(dims))
    if target is None:
        ctx.api.fail(
            f"repeat axis {axis} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    if not dims:
        # A 0-d input with axis 0/-1 repeats the scalar (issue #62).
        if isinstance(count, int):
            return _ndarray_type(ctx, [count], dtype)
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "repeat", 0)], dtype)
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
    target = _normalize_axis_0d(axis, len(dims))
    if target is None:
        ctx.api.fail(
            f"repeat axis {axis} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    if not dims:
        # A 0-d input with axis 0/-1 repeats the scalar (issue #62).
        if isinstance(count, int):
            return _ndarray_type(ctx, [count], dtype)
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "repeat", 0)], dtype)
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
    # numpy only searches a 1-D array ("object too deep for desired
    # array" otherwise): a known other rank is an error, not a shape.
    sorted_parts = _ndarray_parts(pairs[0][0])
    if sorted_parts is not None and not _has_unpack(sorted_parts[0]):
        if len(sorted_parts[0]) != 1:
            ctx.api.fail(
                "searchsorted needs a 1-D sorted array, got "
                f"{len(sorted_parts[0])} dimensions (shape {_dims_str(sorted_parts[0])})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
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

    Same-name TypeVars are equal. Anything else unprovable mints a
    fresh unknown instead of bailing the whole shape (issue #56 for
    two different symbolics; issue #138 for a literal against a
    symbolic dim, which is unprovable either way and must not claim
    the literal). Provably-unequal linear forms (N vs N-1, N+1 vs N from
    np.diff / offsets, issue #32) are errors, not silent unknowns.
    Without ctx the old bail is kept (callers without shape
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
        elif _dims_provably_ne(x, y):
            return None, (
                f"broadcast shapes disagree at trailing axis {k}: "
                f"{_dim_str(x)} vs {_dim_str(y)} (shapes {_dims_str(a)} vs {_dims_str(b)})"
            )
        elif ctx is None:
            return None, None
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
    int clash — or a provably-unequal linear pair (N vs N-1, issue
    #32) — for the caller to fail with.
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
    if _dims_provably_ne(x, y):
        return None, (
            f"broadcast shapes disagree at {axis}: {_dim_str(x)} vs {_dim_str(y)} (shapes {full})"
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
        # Numpy scalars (np.int64(3), x.max()) are strong (issue #61).
        name = _scalar_ret_name(other)
        if name is None:
            return ctx.default_return_type
        key = ("strong", name)
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
        # Either side may be the array (issue #144): np.power(2, i32)
        # arrives scalar-left, while operators arrive array-left.
        if a is not None:
            return _solve_broadcast_scalar(ctx, a[0], a[1], right)
        if b is not None:
            return _solve_broadcast_scalar(ctx, b[0], b[1], left)
        return ctx.default_return_type
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


def _solve_broadcast(
    ctx: FunctionContext | MethodContext, left: Type, right: Type, op: str | None = None
) -> Type:
    dist = _distribute_solve([left, right], lambda l, r: _broadcast_types(ctx, l, r, op))
    if dist is not None:
        return dist
    return _broadcast_types(ctx, left, right, op)


def _broadcast_types(
    ctx: FunctionContext | MethodContext, left: Type, right: Type, op: str | None = None
) -> Type:
    a = _ndarray_parts(left)
    b = _ndarray_parts(right)
    if a is None or b is None:
        return _solve_broadcast_unpack(ctx, left, right, a, b)
    dims_a, dtype_a = a
    dims_b, dtype_b = b
    dims_b, dtype_b = b
    name = _promote_dtypes(dtype_a, dtype_b)
    if name is None:
        # Datetime family add/subtract (issue #144): promotion has no
        # entry, but the stubs already carry the right dtype
        # (datetime/timedelta). Keep the solved broadcast shape with the
        # stub dtype instead of dropping both to Any. Only add/subtract
        # are valid (M+m, m+M, m+m, M-m, M-M, m-m); anything else
        # (M+M, m-M, *, /) bails so invalid loops stay errors. The op
        # rides in from the method/ufunc caller; unknown callers bail.
        if op in ("add", "subtract") and (
            _datetime_kind(dtype_a) is not None or _datetime_kind(dtype_b) is not None
        ):
            # Valid combos only: M+m, m+M, m+m, M-m, M-M, m-m. Anything
            # else (M+M, m-M, datetime with a numeric dtype) bails so
            # invalid loops stay errors.
            ka, kb = _datetime_kind(dtype_a), _datetime_kind(dtype_b)
            if (ka, kb, op) not in {
                ("M", "m", "add"),
                ("m", "M", "add"),
                ("m", "m", "add"),
                ("M", "m", "subtract"),
                ("M", "M", "subtract"),
                ("m", "m", "subtract"),
            }:
                return ctx.default_return_type
            if isinstance(
                get_proper_type(ctx.default_return_type), UninhabitedType
            ):
                return ctx.default_return_type
            default_dtype = _default_dtype(ctx)
            if not isinstance(get_proper_type(default_dtype), AnyType):
                dims, message = _broadcast_result(dims_a, dims_b, ctx)
                if message is not None:
                    ctx.api.fail(message, ctx.context, code=NUMTY_BROADCAST)
                    return ctx.default_return_type
                if dims is None:
                    return ctx.default_return_type
                return _ndarray_type(ctx, dims, default_dtype)
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


def broadcast_method_hook(ctx: MethodContext, op: str | None = None) -> Type:
    # Never replace a stub Never (no loop) with a valid array (issue #61).
    if isinstance(get_proper_type(ctx.default_return_type), UninhabitedType):
        return ctx.default_return_type
    args = _positional_actuals(ctx, 1)
    if args is None:
        return ctx.default_return_type
    return _solve_broadcast(ctx, ctx.type, args[0], op)


def _is_bool_operand(tp: Type) -> bool:
    part = _parts_or_unpack(tp)
    if part is not None:
        return _dtype_base(part[1]) == "b"
    return _bool_value(tp) is not None or _is_plain_bool(tp)


def subtract_method_hook(ctx: MethodContext) -> Type:
    """`a - b`: bool subtract has no loop, fail instead of bool (issue #61)."""
    args = _positional_actuals(ctx, 1)
    if args is None:
        return ctx.default_return_type
    if _is_bool_operand(ctx.type) and _is_bool_operand(args[0]):
        ctx.api.fail(
            "numpy boolean subtract, the `-` operator, is not supported",
            ctx.context,
            code=NUMTY_BROADCAST,
        )
        return ctx.default_return_type
    return _solve_broadcast(ctx, ctx.type, args[0], "subtract")


def floordiv_mod_pow_method_hook(ctx: MethodContext) -> Type:
    """`//`, `%`, `**`: bool+bool is int8, not bool (issue #61)."""
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    arg = pairs[0][0]
    # `bool_array ** 2` is numpy's square fast path, which answers int8
    # like bool ** bool; any other Python int exponent lifts to int64.
    if isinstance(ctx.context, OpExpr) and ctx.context.op in ("//", "%"):
        # floor_divide and remainder have no complex loop (TypeError).
        for operand in (ctx.type, arg):
            part = _parts_or_unpack(operand)
            if part is not None and _dtype_base(part[1]) == "c":
                name = "floor_divide" if ctx.context.op == "//" else "remainder"
                ctx.api.fail(
                    f"ufunc '{name}' not supported for complex inputs",
                    ctx.context,
                    code=NUMTY_BROADCAST,
                )
                return ctx.default_return_type
    squared = (
        isinstance(ctx.context, OpExpr)
        and ctx.context.op == "**"
        and pairs[0][1] is ctx.context.right
        and _weak_name(arg) == "builtins.int"
        and _int_value(arg) == 2
    )
    if _is_bool_operand(ctx.type) and (squared or _is_bool_operand(arg)):
        dims = _broadcast_dims(ctx, ctx.type, arg)
        if dims is None:
            return ctx.default_return_type
        dtype = _construct_dtype(ctx, "numpy.int8")
        if dtype is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, dtype)
    return _solve_broadcast(ctx, ctx.type, arg)


def divmod_method_hook(ctx: MethodContext) -> Type:
    """`divmod(a, b)` / `a.__divmod__(b)`: tuple of two broadcasts (issue #54)."""
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    arg = pairs[0][0]
    if _is_bool_operand(ctx.type) and _is_bool_operand(arg):
        dims = _broadcast_dims(ctx, ctx.type, arg)
        if dims is None:
            return ctx.default_return_type
        dtype = _construct_dtype(ctx, "numpy.int8")
        if dtype is None:
            return ctx.default_return_type
        result = _ndarray_type(ctx, dims, dtype)
        return _linalg_tuple(ctx, [result, result])
    result = _solve_broadcast(ctx, ctx.type, arg)
    parts = _ndarray_parts(result)
    if parts is None:
        return ctx.default_return_type
    # Same result twice: quotient and remainder share shape and dtype.
    return _linalg_tuple(ctx, [result, result])


def divmod_function_hook(ctx: FunctionContext) -> Type:
    """Builtin `divmod(a, b)` on arrays: same tuple as the method (issue #54)."""
    args = _positional_actuals(ctx, 2)
    if args is None:
        return ctx.default_return_type
    if _is_bool_operand(args[0]) and _is_bool_operand(args[1]):
        dims = _broadcast_dims(ctx, args[0], args[1])
        if dims is None:
            return ctx.default_return_type
        dtype = _construct_dtype(ctx, "numpy.int8")
        if dtype is None:
            return ctx.default_return_type
        result = _ndarray_type(ctx, dims, dtype)
        return _linalg_tuple(ctx, [result, result])
    result = _solve_broadcast(ctx, args[0], args[1])
    parts = _ndarray_parts(result)
    if parts is None:
        return ctx.default_return_type
    return _linalg_tuple(ctx, [result, result])


def inplace_method_hook(ctx: MethodContext) -> Type:
    """Shape-check `a <op>= b`: broadcast must equal the LHS (issue #50).

    Broadcast mismatches fail with the binary wording; a broadcast that
    succeeds but would grow the target fails as non-broadcastable output.
    Scalars, unpack shapes and unprovable symbolic dims bail silently,
    keeping the stubs' dtype checking and Self return.
    """
    args = _positional_actuals(ctx, 1)
    if args is None:
        return ctx.default_return_type
    lhs = _ndarray_parts(ctx.type)
    if lhs is None:
        return ctx.default_return_type
    lhs_dims, _ = lhs
    if _scalar_key(args[0]) is not None:
        return ctx.default_return_type
    rhs = _ndarray_parts(args[0])
    if rhs is None:
        return ctx.default_return_type
    rhs_dims, _ = rhs
    dims, message = _broadcast_result(lhs_dims, rhs_dims, ctx)
    if message is not None:
        ctx.api.fail(message, ctx.context, code=NUMTY_BROADCAST)
        return ctx.default_return_type
    if dims is None:
        return ctx.default_return_type
    if len(dims) != len(lhs_dims) or any(
        not _dim_eq(d, l) for d, l in zip(dims, lhs_dims)
    ):
        # Only fail when provably different: rank growth always is;
        # same-rank needs an int mismatch. Symbolic mismatches stay
        # silent like the binary hook.
        if len(dims) != len(lhs_dims) or any(
            isinstance(d, int) and isinstance(l, int) and d != l
            for d, l in zip(dims, lhs_dims)
        ):
            ctx.api.fail(
                f"in-place result {_dims_str(dims)} does not fit "
                f"target {_dims_str(lhs_dims)}",
                ctx.context,
                code=NUMTY_BROADCAST,
            )
    return ctx.default_return_type


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


_unknown_ids: dict[str, TypeVarId] = {}
# Display names of unknowns by fullname, and how many distinct unknowns
# each (module, line) has minted. Names only: identity stays the id.
_unknown_names: dict[str, str] = {}
_unknown_per_line: dict[tuple[str, int], int] = {}
# Unknowns minted inside a comprehension (issue #63): one site is
# evaluated once per element at runtime, so every element shares the
# same ?U even when sizes differ. Never prove such an unknown equal,
# not even to itself — the result bails instead of claiming (3, 3).
# In practice every shared ?U is treated this way (not just
# comprehensions): same-array uses bail to a fresh ?U instead of a
# false proof, which is safe and never a false error.
_UNKNOWN_PREFIX = "__numty_unknown__."


def _is_unknown_fullname(name: str) -> bool:
    return name.startswith(_UNKNOWN_PREFIX)


def _form_has_unknown(form: object) -> bool:
    """Whether a canonical form mentions any ?U unknown."""
    if isinstance(form, _LinForm):
        return any(_is_unknown_fullname(name) for name, _ in form.coeffs)
    if isinstance(form, _MonoForm):
        return any(_is_unknown_fullname(v) for v in form.vars)
    return False


def _is_same_unknown(a: int | Type, b: int | Type) -> bool:
    """Same ?U on both sides (issue #63): one mask site serves every
    comprehension element, so equality would be a false proof."""
    return (
        isinstance(a, TypeVarType)
        and isinstance(b, TypeVarType)
        and a.fullname == b.fullname
        and _is_unknown_fullname(a.fullname)
    )


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
    ctx: FunctionContext | MethodContext | AttributeContext, role: str, index: int
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
    """Reshape a (*prefix, *Ts, *suffix) shape (issue #48).

    `raw` is the whole new shape: numpy reads the target as exactly
    that, so the result is `raw` itself (correct whenever the call
    succeeds at runtime) — `*Ts` is never put back in. Anything else
    unprovable bails, except a total-size clash on purely int sizes,
    which no `*Ts` can absorb and which fails loudly.
    """
    old = prefix + [unpack] + suffix
    if -1 in raw:
        # -1 would absorb the unknown *Ts product.
        return ctx.default_return_type
    if any(not isinstance(d, int) for d in prefix + suffix):
        # Symbolic known runs: the target holds by construction when the
        # call is legal, and legality itself stays unchecked.
        return _ndarray_type(ctx, list(raw), dtype)
    if any(not isinstance(d, int) for d in raw):
        return _ndarray_type(ctx, list(raw), dtype)
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
    return _ndarray_type(ctx, list(raw), dtype)


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


def _reshape_fold_arithmetic(
    ctx: FunctionContext | MethodContext, pairs: list[tuple[Type, Expression | None]]
) -> list[tuple[Type, Expression | None]]:
    """`pairs` with arithmetic target sizes replaced by the dim they spell.

    `x.reshape(x.shape[0] * x.shape[1], -1)` names its first size
    exactly; read as the plain `int` mypy gives it, the size is lost.
    """
    if not pairs:
        return pairs
    int_instance = ctx.api.named_generic_type(INT_FULLNAME, [])

    def folded(expr: Expression | None, index: int) -> Type | None:
        if not isinstance(expr, OpExpr) or _fold_int(expr) is not None:
            return None
        dim = _fold_dim_expr(ctx, expr, index)
        if dim is None:
            return None
        return LiteralType(dim, fallback=int_instance) if isinstance(dim, int) else dim

    first_tp, first_expr = pairs[0]
    first = get_proper_type(first_tp)
    if isinstance(first_expr, TupleExpr):
        if not isinstance(first, TupleType) or len(first.items) != len(first_expr.items):
            return pairs
        if any(isinstance(item, StarExpr) for item in first_expr.items):
            return pairs
        items: list[Type] = []
        changed = False
        for index, (item_tp, item_expr) in enumerate(zip(first.items, first_expr.items)):
            new = folded(item_expr, index)
            if new is None:
                # The expression is dropped below, so keep what it folds to.
                literal = _fold_int(item_expr)
                if literal is not None:
                    new = LiteralType(literal, fallback=int_instance)
            else:
                changed = True
            items.append(new if new is not None else item_tp)
        if not changed:
            return pairs
        return [(first.copy_modified(items=items), None), *pairs[1:]]
    out = list(pairs)
    for index, (_, expr) in enumerate(pairs):
        new = folded(expr, index)
        if new is not None:
            out[index] = (new, None)
    return out


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
    try:
        pairs = _reshape_fold_arithmetic(ctx, pairs)
    except Exception:
        _swallowed(ctx, "_reshape_fold_arithmetic")
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
        # Comparing the literal factors alone only means something when
        # every other dim is a bare name: `2*N+2` carries a factor of 2
        # that (N+1, 2) spells as a literal, and an unknown may hide any.
        both_formed = old_form is not None and new_form is not None
        literal_only = all(_is_bare_name_dim(d) for d in [*dims_old, *raw])
        if size_new != size_old and not forms_equal and (both_formed or literal_only):
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
        if _forms_provably_ne(old_form, new_form, nonzero=True):
            # Same int total but provably-unequal symbolic totals
            # (N+1 vs N, N+2 vs N*1 from off-by-pad, issue #63):
            # no assignment of N satisfies the equality, so the call
            # always raises — error instead of tracking the request.
            assert old_form is not None and new_form is not None
            ctx.api.fail(
                f"reshape changes total size: {_expr_str(old_form)} -> "
                f"{_expr_str(new_form)} "
                f"(shapes {_dims_str(dims_old)} vs {raw_str})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        old_total = _reshape_total_form(dims_old)
        new_total = _reshape_total_form(raw)
        if _forms_provably_ne(old_total, new_total, nonzero=True):
            # Single-dim sums the product form cannot spell (N+1 vs N,
            # N+2 vs N): same vars, different consts, always unequal.
            assert old_total is not None and new_total is not None
            ctx.api.fail(
                f"reshape changes total size: {_expr_str(old_total)} -> "
                f"{_expr_str(new_total)} "
                f"(shapes {_dims_str(dims_old)} vs {raw_str})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        if len(dims_old) == len(raw):
            # Sums the product form cannot spell ((N+1)*M vs N*M,
            # issue #137): same rank with a single provably-unequal dim
            # pair and the rest equal means the totals differ (up to a
            # zero factor the int path already ignores) — error instead
            # of tracking the request.
            ne_idx = -1
            single = True
            for i, (a, b) in enumerate(zip(dims_old, raw)):
                if _dim_eq(a, b):
                    continue
                if _dims_provably_ne(a, b, nonzero=True):
                    if ne_idx != -1:
                        single = False
                        break
                    ne_idx = i
                else:
                    single = False
                    break
            if ne_idx != -1 and single:
                a_ne = dims_old[ne_idx]
                b_ne = raw[ne_idx]
                ctx.api.fail(
                    f"reshape changes total size: {_dim_str(a_ne)} -> {_dim_str(b_ne)} "
                    f"(shapes {_dims_str(dims_old)} vs {raw_str})",
                    ctx.context,
                    code=NUMTY_SHAPE,
                )
                return ctx.default_return_type
        totals_equal = forms_equal or (
            old_total is not None and new_total is not None and old_total == new_total
        )
        if not totals_equal and old_vars != new_vars:
            # Unrepresentable or monomial/div totals that are not proven
            # equal (issue #137): the request might be impossible
            # (N//2+1 vs N, (N+1)*M vs N*M when the per-dim proof misses)
            # but we cannot prove it, so bail to Any instead of tracking
            # a precise impossible shape. Plain linear totals (N vs M)
            # keep tracking — they hold by construction when legal.
            # Verified equal totals (forms_equal) also keep tracking.
            nonlin = (
                old_form is None
                or new_form is None
                or isinstance(old_form, (_MonoForm, _DivForm))
                or isinstance(new_form, (_MonoForm, _DivForm))
                or old_total is None
                or new_total is None
                or isinstance(old_total, (_MonoForm, _DivForm))
                or isinstance(new_total, (_MonoForm, _DivForm))
            )
            if nonlin:
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


def _all_positionals_expanded(
    ctx: FunctionContext | MethodContext,
) -> list[tuple[Type, Expression | None]] | None:
    """Positionals with fixed `*tuple` stars expanded; None on dynamic unpack.

    An ARG_STAR whose type is a fixed-length TupleType contributes one
    entry per item (expression unknown, so None). Anything else unpacked
    (`*list`, `*variadic`, `**kwargs`) cannot be solved precisely: return
    None so the caller bails to the stubs instead of solving the wrong
    arity (issue #52).
    """
    out: list[tuple[Type, Expression | None]] = []
    seen_star: set[int] = set()
    for kinds, types, exprs in zip(ctx.arg_kinds, ctx.arg_types, ctx.args):
        for kind, tp, expr in zip(kinds, types, exprs):
            if kind == ARG_POS:
                out.append((tp, expr))
            elif kind == ARG_STAR:
                # One star actual can be bound to several formals when the
                # callee is overloaded (same expression object twice):
                # count it once.
                if id(expr) in seen_star:
                    continue
                seen_star.add(id(expr))
                proper = get_proper_type(tp)
                if isinstance(proper, TupleType):
                    for item in proper.items:
                        out.append((item, None))
                else:
                    return None
            elif kind == ARG_STAR2:
                return None
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


def _normalize_axis_0d(ax: int, ndim: int) -> int | None:
    """Like _normalize_axis, but numpy also accepts axis 0/-1 on 0-d
    input for reductions, cumsum, repeat and squeeze (issue #62)."""
    if ndim == 0 and ax in (0, -1):
        return 0
    return _normalize_axis(ax, ndim)


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
    if base is None:
        return None
    if base == "c" and len(element.args) == 2:
        # complexfloating[_32Bit, _32Bit]: two identical width tags.
        sizes = [_nbit_bytes(a) for a in element.args]
        if len(sizes) != 2 or sizes[0] is None or sizes[0] != sizes[1]:
            return None
        assert sizes[0] is not None
        # 2x4-byte -> 8 bytes (complex64), 2x8-byte -> 16 (complex128).
        total = sizes[0] * 2
        if total not in (8, 16):
            return None
        return base, total
    if len(element.args) != 1:
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


def _solve_transpose_unpack(
    ctx: FunctionContext | MethodContext,
    prefix: list[int | Type],
    unpack: UnpackType,
    suffix: list[int | Type],
    dtype: Type,
    rest: list[tuple[Type, Expression | None]],
) -> Type:
    """Transpose a (*prefix, *Ts, *suffix) shape (issue #48).

    Reversing would also reverse *Ts itself, which no annotation can
    spell, so every *Ts transpose bails out instead of guessing.
    """
    # The stubs preserve the input shape, which is a wrong exact claim
    # once reversed (issue #185): bail to Any instead of the default.
    return AnyType(TypeOfAny.from_omitted_generics)


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


def _datetime_kind(dtype: Type) -> str | None:
    """'M' for datetime64, 'm' for timedelta64, else None (issue #144).

    Units are erased in the stubs (datetime64[Any]), so only the base
    matters; anything else bails like before.
    """
    proper = get_proper_type(dtype)
    if not isinstance(proper, Instance) or not proper.args:
        return None
    element = get_proper_type(proper.args[0])
    if not isinstance(element, Instance):
        return None
    if element.type.fullname == "numpy.datetime64":
        return "M"
    if element.type.fullname == "numpy.timedelta64":
        return "m"
    return None


def _construct_datetime_dtype(
    ctx: FunctionContext | MethodContext, kind: str
) -> Type | None:
    """dtype[datetime64] / dtype[timedelta64] with an Any unit, else None."""
    try:
        name = "numpy.datetime64" if kind == "M" else "numpy.timedelta64"
        element: Type = ctx.api.named_generic_type(
            name, [AnyType(TypeOfAny.explicit)]
        )
        return ctx.api.named_generic_type(DTYPE_FULLNAME, [element])
    except Exception:
        _swallowed(ctx, "_construct_datetime_dtype")
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
        norm = _normalize_axis_0d(ax, ndim)
        if norm is None:
            ctx.api.fail(
                f"squeeze axis {ax} is out of range for {ndim} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        if ndim == 0:
            # 0-d input with axis 0/-1: nothing to squeeze.
            chosen.append(norm)
            continue
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
                f"with {len(axis)} new axes (shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        if norm in normalized:
            ctx.api.fail(
                f"expand_dims has duplicate axis {ax} in {tuple(axis)} "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        normalized.append(norm)
    out = list(dims)
    for ax in sorted(normalized):
        out.insert(ax, 1)
    return _ndarray_type(ctx, out, dtype)


def _reduction_out_dtype(
    ctx: FunctionContext | MethodContext, op: str, dtype: Type
) -> Type | None:
    """Output dtype for sum/prod/mean/std/var/max/min (issue #61).

    sum/prod of bool/int -> int64 (uint -> uint64); mean/std/var of
    bool/int/uint -> float64; std/var of complex -> real float; max/min
    keep the input dtype. Anything else bails.
    """
    name = _dtype_fullname(dtype)
    if name is None:
        ks = _dtype_kind_size(dtype)
        if ks is None:
            return None
        kind, _ = ks
        # Unresolvable generic (e.g. floating[Any]): keep dims with Any? Bail.
        return None
    kind = _kind_of(name)
    if kind is None:
        return None
    base = op.removeprefix("nan")
    if base in ("sum", "prod"):
        if kind == "u":
            return _construct_dtype(ctx, "numpy.uint64")
        if kind in ("b", "i"):
            return _construct_dtype(ctx, "numpy.int64")
        return dtype
    if base in ("mean", "std", "var"):
        if kind in ("b", "i", "u"):
            return _construct_dtype(ctx, "numpy.float64")
        if kind == "c" and base in ("std", "var"):
            ks = _dtype_kind_size(dtype)
            if ks is None:
                return None
            _, size = ks
            real = _linalg_real_name("c", size)
            if real is None:
                return None
            return _construct_dtype(ctx, real)
        return dtype
    if base in ("max", "min", "ptp", "amax", "amin"):
        return dtype
    return None


def _solve_reduction(
    ctx: FunctionContext | MethodContext,
    dims: list[int | Type],
    dtype: Type,
    axis: list[int] | None | str,
    keepdims: bool | None,
    has_dtype: bool,
    force_dtype: Type | None = None,
    op: str = "sum",
) -> Type:
    if not isinstance(axis, list) or keepdims is None or has_dtype:
        return ctx.default_return_type
    if force_dtype is None and _dtype_fullname(dtype) not in REDUCTION_DTYPES:
        return ctx.default_return_type
    out_dtype = force_dtype if force_dtype is not None else dtype
    ndim = len(dims)
    normalized: list[int] = []
    # sum/prod/max/min/any/all take axis 0 or -1 on a 0-d array; the
    # mean family counts elements along the axis first and rejects it.
    # (The nan* twins go through nansum-style code and accept it.)
    strict_0d = op in ("mean", "var", "std")
    for ax in axis:
        norm = _normalize_axis(ax, ndim) if strict_0d else _normalize_axis_0d(ax, ndim)
        if norm is None:
            ctx.api.fail(
                f"{op} axis {ax} is out of range for {ndim} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        if norm in normalized:
            ctx.api.fail(
                f"{op} has duplicate axis {ax} in {tuple(axis)} "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        normalized.append(norm)
    chosen = set(normalized)
    if keepdims:
        return _ndarray_type(ctx, [1 if i in chosen else d for i, d in enumerate(dims)], out_dtype)
    return _maybe_scalar(ctx, [d for i, d in enumerate(dims) if i not in chosen], out_dtype)


def _reduction_inputs_bool(
    ctx: FunctionContext | MethodContext, array: Type, op: str = "any"
) -> Type:
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
        [array], lambda t: _reduction_bool_types(ctx, t, axis, keepdims, has_dtype, dtype, op)
    )
    if dist is not None:
        return dist
    return _reduction_bool_types(ctx, array, axis, keepdims, has_dtype, dtype, op)


def _reduction_bool_types(
    ctx: FunctionContext | MethodContext,
    array: Type,
    axis: list[int] | None | str,
    keepdims: bool | None,
    has_dtype: bool,
    dtype: Type,
    op: str = "any",
) -> Type:
    parts = _parts_or_unpack(array) or _unknown_axis_parts(array)
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    if _has_unpack(dims):
        split = _split_flat(dims)
        if split is None:
            return ctx.default_return_type
        return _solve_reduction_unpack(ctx, *split, dtype, axis, keepdims, has_dtype, dtype, op)
    if axis == "absent":
        # An omitted axis reduces every axis, exactly like axis=None.
        axis = "none"
    if axis == "none":
        if keepdims is None or has_dtype:
            return ctx.default_return_type
        if keepdims:
            return _ndarray_type(ctx, [1] * len(dims), dtype)
        scalar = _scalar_of_dtype(dtype)
        return scalar if scalar is not None else ctx.default_return_type
    return _solve_reduction(ctx, dims, dtype, axis, keepdims, has_dtype, dtype, op)


def allany_method_hook(ctx: MethodContext, op: str) -> Type:
    return _reduction_inputs_bool(ctx, ctx.type, op)


def allany_function_hook(ctx: FunctionContext, op: str) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _reduction_inputs_bool(ctx, pairs[0][0], op)


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
            _swallowed(ctx, "_exact_element")
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
        if name is not None:
            return _construct_dtype(ctx, name)
        # Datetime units erase to Any in the stubs (issue #144):
        # "datetime64[D]", "datetime64[ns]", "timedelta64[D]", ...
        kind = _datetime_string_kind(proper.value)
        if kind is not None:
            return _construct_datetime_dtype(ctx, kind)
        return None
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
            if fullname in ("numpy.datetime64", "numpy.timedelta64"):
                kind = "M" if fullname == "numpy.datetime64" else "m"
                return _construct_datetime_dtype(ctx, kind)
    if isinstance(proper, Instance) and proper.type.fullname == DTYPE_FULLNAME:
        if len(proper.args) == 1:
            element = get_proper_type(proper.args[0])
            if isinstance(element, Instance) and element.type.fullname in _DTYPE_INFO:
                return _construct_dtype(ctx, element.type.fullname)
            if isinstance(element, Instance) and element.type.fullname in _DATETIME_FULLNAMES:
                kind = "M" if element.type.fullname == "numpy.datetime64" else "m"
                return _construct_datetime_dtype(ctx, kind)
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
        if name is not None:
            return _construct_dtype(ctx, name)
        kind = _datetime_string_kind(expr.value)
        if kind is not None:
            return _construct_datetime_dtype(ctx, kind)
        return None
    return None


def _apply_out_dtype(
    ctx: FunctionContext | MethodContext, result: Type
) -> Type:
    """Apply explicit out=/dtype= to a solved ufunc/reduction result.

    NumPy writes into out and returns it, and dtype= selects the loop:
    when either is present the stubs' shape is not the answer. A
    provable out shape clash fails loudly; anything unprovable keeps
    out's own type (it carries the requested dtype and shape). A dtype=
    the plugin cannot resolve bails out instead of guessing (issue #52).
    """
    if result is ctx.default_return_type:
        return result
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
        if named is None:
            return ctx.default_return_type
        parts = _parts_or_unpack(result)
        if parts is None:
            return ctx.default_return_type
        dims, _ = parts
        result = _ndarray_type(ctx, list(dims), named)
    out_pair = _formal_pair(ctx, "out")
    if out_pair is None or _is_none_type(out_pair[0]):
        return result
    out_parts = _parts_or_unpack(out_pair[0])
    if out_parts is None:
        return out_pair[0]
    out_dims, _ = out_parts
    res_parts = _parts_or_unpack(result)
    if res_parts is None:
        return out_pair[0]
    res_dims, _ = res_parts
    if all(isinstance(d, int) for d in res_dims) and all(
        isinstance(d, int) for d in out_dims
    ):
        assert all(isinstance(d, int) for d in res_dims)
        assert all(isinstance(d, int) for d in out_dims)
        rdims = cast(list[int], res_dims)
        odims = cast(list[int], out_dims)
        fits = len(odims) >= len(rdims) and all(
            r == o or r == 1
            for r, o in zip(rdims, odims[len(odims) - len(rdims):])
        ) if rdims else True
        if not fits:
            ctx.api.fail(
                f"output shape mismatch: computed shape {_dims_str(res_dims)} "
                f"does not fit out shape {_dims_str(out_dims)}",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
    return out_pair[0]


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
        override = _builder_shape(ctx, *shape_pair, ctx.default_return_type, shaped_stub=False)
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
    if not dims:
        # Both guarantee ndim >= 1: a 0-d input comes back as (1,).
        dims = [1]
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
    dims = _builder_shape(ctx, *shape_pair, ctx.default_return_type)
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
    # start/stop may be array-likes, and the result is then (num, *shape):
    # only scalar endpoints are solved here.
    for name, index in (("start", 0), ("stop", 1)):
        end_pair = _formal_pair(ctx, name)
        if end_pair is None:
            positional = _all_positionals(ctx)
            end_pair = positional[index] if len(positional) > index else None
        if end_pair is not None and _scalar_key(end_pair[0]) is None:
            return ctx.default_return_type
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


def _arange_symbolic_kw(
    ordered: list[tuple[Type, Expression | None] | None],
) -> list[int | TypeVarType] | None:
    """Symbolic arange through keyword-resolved start/stop/step.

    `arange(stop=N)`, `arange(0, stop=N)`, `arange(start=0, stop=N,
    step=1)`: N elements. Anything else (non-zero start, non-one step,
    unresolvable stop) stays with the stubs.
    """
    if len(ordered) != 3:
        return None
    start_p, stop_p, step_p = ordered
    if stop_p is None:
        return None
    if start_p is not None and _fold_single(start_p) != 0:
        return None
    if step_p is not None and _fold_single(step_p) != 1:
        return None
    stop = _dim_value(stop_p[0])
    if not isinstance(stop, TypeVarType):
        return None
    return [stop]


def arange_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) > 3:
        return ctx.default_return_type
    # start/stop/step may arrive as keywords (issue #46: `stop=`/`step=`
    # used to be silently ignored). The stubs name the first positional
    # `start_or_stop`, and it is positional-only (shows up as `None` in
    # `callee_arg_names`), so take it from call order: with an explicit
    # `stop` a leading positional is the start, otherwise a lone
    # positional is the stop (`arange(10)`, `arange(10, step=2)`).
    step_p = _formal_pair(ctx, "step")
    stop_p = _formal_pair(ctx, "stop")
    first_pos = pairs[0] if pairs else None
    if first_pos is None and stop_p is None and step_p is None:
        ordered: list[tuple[Type, Expression | None] | None] = list(pairs)
        by_name = False
    else:
        if stop_p is not None:
            start_p = first_pos
        else:
            start_p, stop_p = None, first_pos
        ordered = [start_p, stop_p, step_p]
        by_name = True
    if len(ordered) not in (1, 2, 3):
        return ctx.default_return_type
    if by_name:
        # A name resolved through the formals already accounts for how
        # mypy bound positionals; a missing stop is an invalid call.
        if ordered[1] is None:
            return ctx.default_return_type
        start_v = 0 if ordered[0] is None else _fold_single(ordered[0])
        stop_v = _fold_single(ordered[1])
        step_v = 1 if ordered[2] is None else _fold_single(ordered[2])
        if start_v is not None and stop_v is not None and step_v is not None:
            try:
                dims = [len(range(start_v, stop_v, step_v))]
            except ValueError:
                ctx.api.fail("arange step is zero", ctx.context, code=NUMTY_SHAPE)
                return ctx.default_return_type
        else:
            dims = _arange_symbolic_kw(ordered)
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
        _swallowed(ctx, "_comprehension_parts")
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


def _is_empty_sequence(tp: Type, expr: Expression | None) -> bool:
    """A statically-known empty sequence: `[]`, `()`, or an empty tuple type.

    `_sequence_parts` folds these to None (no items to read), which is
    indistinguishable from an unsolvable bail — but an empty sequence
    always raises at runtime, so it is a provable arity error (issue #143).
    """
    if isinstance(expr, (ListExpr, TupleExpr)) and not expr.items:
        return True
    proper = get_proper_type(tp)
    return isinstance(proper, TupleType) and not proper.items


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
        _swallowed(ctx, "_element_expr_type")
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
        if _is_empty_sequence(pairs[0][0], pairs[0][1]):
            ctx.api.fail(
                "concatenate needs at least one array (got an empty sequence)",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
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
        # Flattened size is the sum of each input's size (0-d counts as
        # 1, the empty product); symbolic sizes add/multiply exactly
        # (issue #46), never the product of every dim.
        total: int | Type = 0
        for dims, _ in items:
            total = _dim_add2(
                ctx,
                "concatenate",
                0,
                total,
                _mul_dims(ctx, "concatenate", 0, list(dims)),
            )
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
            f"concatenate axis {axis} is out of range for {ndim} dimensions "
            f"(shape {_dims_str(items[0][0])})",
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
                if (isinstance(out[i], int) and isinstance(d, int)) or _dims_provably_ne(
                    out[i], d
                ):
                    ctx.api.fail(
                        f"{op} dimension {i} differs: {_dim_str(out[i])} vs {_dim_str(d)} "
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
        if _is_empty_sequence(pairs[0][0], pairs[0][1]):
            ctx.api.fail(
                f"{kind} needs at least one array (got an empty sequence)",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
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
            f"concatenate axis {axis} is out of range for {len(dims_e)} dimensions "
            f"(shape {_dims_str(dims_e)})",
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
            f"stack axis {axis} is out of range for {ndim} dimensions "
            f"(shape {_dims_str(dims_e)})",
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
        if _is_empty_sequence(pairs[0][0], pairs[0][1]):
            ctx.api.fail(
                "stack needs at least one array (got an empty sequence)",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
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
            provable = len(dims) != len(first) or any(
                _dims_provably_ne(x, y) for x, y in zip(first, dims)
            )
            if len(dims) != len(first):
                ctx.api.fail(
                    f"stack ranks differ: "
                    + ", ".join(_dims_str(dims) for dims, _ in items)
                    + f" (axis {axis})",
                    ctx.context,
                    code=NUMTY_SHAPE,
                )
                return ctx.default_return_type
            # Same rank: name the first differing dimension and its values,
            # like concatenate does, plus the stack axis.
            diff_at: int | None = None
            for i, (x, y) in enumerate(zip(first, dims)):
                if not _dim_eq(x, y):
                    diff_at = i
                    break
            if diff_at is not None and (
                all(isinstance(d, int) for d in dims)
                and all(isinstance(d, int) for d in first)
            ):
                x, y = first[diff_at], dims[diff_at]
                ctx.api.fail(
                    f"stack dimension {diff_at} differs: {_dim_str(x)} vs {_dim_str(y)} "
                    f"(shapes {_dims_str(first)} vs {_dims_str(dims)}; axis {axis})",
                    ctx.context,
                    code=NUMTY_SHAPE,
                )
                return ctx.default_return_type
            if provable and diff_at is not None:
                x, y = first[diff_at], dims[diff_at]
                ctx.api.fail(
                    f"stack dimension {diff_at} differs: {_dim_str(x)} vs {_dim_str(y)} "
                    f"(shapes {_dims_str(first)} vs {_dims_str(dims)}; axis {axis})",
                    ctx.context,
                    code=NUMTY_SHAPE,
                )
                return ctx.default_return_type
            return ctx.default_return_type
    ndim = len(first)
    norm = axis + ndim + 1 if axis < 0 else axis
    if not 0 <= norm <= ndim:
        ctx.api.fail(
            f"stack axis {axis} is out of range for {ndim} dimensions "
            f"(shape {_dims_str(first)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(first)
    out.insert(norm, len(items))
    return _ndarray_type(ctx, out, dtype)


def reshape_method_hook(ctx: MethodContext) -> Type:
    pairs = _all_positionals_expanded(ctx)
    if pairs is None:
        return ctx.default_return_type
    return _solve_reshape(ctx, ctx.type, pairs)


def reshape_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals_expanded(ctx)
    if pairs is None:
        return ctx.default_return_type
    if len(pairs) < 2:
        shape_pair = _formal_pair(ctx, "shape")
        if shape_pair is not None and not _is_none_type(shape_pair[0]):
            pairs = pairs + [shape_pair]
    if len(pairs) < 2:
        # First positional may itself be spelled as a= keyword.
        first = _formal_pair(ctx, "a")
        if first is not None and pairs:
            pairs = [first] + pairs[1:]
        elif first is not None:
            pairs = [first]
        if len(pairs) < 2:
            return ctx.default_return_type
    return _solve_reshape(ctx, pairs[0][0], pairs[1:])


def transpose_method_hook(ctx: MethodContext) -> Type:
    pairs = _all_positionals_expanded(ctx)
    if pairs is None:
        return ctx.default_return_type
    parts = _movable_parts(ctx.type)
    if parts is not None:
        dims, dtype = parts
        return _solve_transpose(ctx, dims, dtype, pairs)
    split = _unpack_shape(ctx.type)
    if split is None:
        return ctx.default_return_type
    (prefix, unpack, suffix), dtype = split
    return _solve_transpose_unpack(ctx, prefix, unpack, suffix, dtype, pairs)


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
    pairs = _all_positionals_expanded(ctx)
    if pairs is None:
        return ctx.default_return_type
    if not pairs:
        first = _formal_pair(ctx, "a")
        if first is None:
            return ctx.default_return_type
        pairs = [first]
    parts = _movable_parts(pairs[0][0])
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
        axes_pair = _formal_pair(ctx, "axes")
        if axes_pair is not None and not _is_none_type(axes_pair[0]):
            axes = _shape_from_arg(axes_pair[0], axes_pair[1])
            if axes is None:
                return ctx.default_return_type
            return _apply_permutation(ctx, dims, dtype, axes)
        return _apply_permutation(ctx, dims, dtype, list(reversed(range(len(dims)))))
    if len(rest) > 1:
        folded: list[int] = []
        for tp, expr in rest:
            value = _fold_int(expr) if expr is not None else None
            if value is None:
                value = _int_value(tp)
            if value is None:
                return ctx.default_return_type
            folded.append(value)
        return _apply_permutation(ctx, dims, dtype, folded)
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
        parts = _ndarray_parts(arr_pair[0]) or _unknown_axis_parts(arr_pair[0])
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
    parts = _ndarray_parts(ctx.type) or _unknown_axis_parts(ctx.type)
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
    parts = _ndarray_parts(pairs[0][0]) or _unknown_axis_parts(pairs[0][0])
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
    parts = _ndarray_parts(pairs[0][0]) or _unknown_axis_parts(pairs[0][0])
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
        ctx.api.fail(
            f"matrix_transpose requires rank >= 2, got shape {_dims_str(dims)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        if isinstance(ctx, AttributeContext):
            return ctx.default_attr_type
        return ctx.default_return_type
    out = list(dims)
    out[-2], out[-1] = out[-1], out[-2]
    return _ndarray_type(ctx, out, dtype)


def matrix_transpose_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0]) or _unknown_axis_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    return _matrix_transpose_solve(ctx, dims, dtype)


def mt_attribute_hook(ctx: AttributeContext) -> Type:
    parts = _movable_parts(ctx.type)
    if parts is None:
        return ctx.default_attr_type
    dims, dtype = parts
    return _matrix_transpose_solve(ctx, dims, dtype)


def permute_dims_function_hook(ctx: FunctionContext) -> Type:
    # permute_dims is alias to transpose: first arg array, second optional axes
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _movable_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    # axes as second positional or as the keyword
    axes_pair = _formal_pair(ctx, "axes")
    if axes_pair is None and len(pairs) > 1:
        axes_pair = pairs[1]
    if axes_pair is None or _is_none_type(axes_pair[0]):
        # No axes: reverse
        return _ndarray_type(ctx, list(reversed(dims)), dtype)
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
    op: str = "sum",
) -> Type:
    """Reduce a (*prefix, *Ts, *suffix) shape along provable axes.

    Only axes that resolve to a known dim reduce: non-negative axes
    into the prefix, negative axes into the suffix. Anything reaching
    into *Ts bails out, as do absent/None axes, exactly like the
    fixed-rank path.
    """
    # Bailing here must not return the stubs' answer: for *Ts shapes the
    # stubs preserve the input shape (e.g. `var` keeps (*Ts, P)), which
    # is a wrong exact claim when an axis actually reduces. Return Any
    # so the shape stays unchecked instead of wrong (issue #185).
    if not isinstance(axis, list) or keepdims is None or has_dtype:
        return AnyType(TypeOfAny.from_omitted_generics)
    if force_dtype is None and _dtype_fullname(dtype) not in REDUCTION_DTYPES:
        return AnyType(TypeOfAny.from_omitted_generics)
    out_dtype = force_dtype if force_dtype is not None else dtype
    flat = prefix + [unpack] + suffix
    chosen: set[int] = set()
    for ax in axis:
        if isinstance(ax, bool) or not isinstance(ax, int):
            return AnyType(TypeOfAny.from_omitted_generics)
        if ax < 0:
            if -ax > len(suffix):
                return AnyType(TypeOfAny.from_omitted_generics)
            pos = len(flat) + ax
        else:
            if ax >= len(prefix):
                return AnyType(TypeOfAny.from_omitted_generics)
            pos = ax
        if pos in chosen:
            ctx.api.fail(
                f"{op} has duplicate axis {ax} in {tuple(axis)} "
                f"(shape {_dims_str(prefix + [unpack] + suffix)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        chosen.add(pos)
    if keepdims:
        out = [1 if i in chosen else d for i, d in enumerate(flat)]
    else:
        out = [d for i, d in enumerate(flat) if i not in chosen]
    return _ndarray_type(ctx, out, out_dtype)


def _reduction_inputs(
    ctx: FunctionContext | MethodContext, array: Type, op: str = "sum"
) -> Type:
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    keep_pair = _formal_pair(ctx, "keepdims")
    keepdims = False if keep_pair is None else _fold_flag(keep_pair)
    dtype_pair = _formal_pair(ctx, "dtype")
    has_dtype = dtype_pair is not None and not _is_none_type(dtype_pair[0])
    dist = _distribute_solve(
        [array], lambda t: _reduction_types(ctx, t, axis, keepdims, has_dtype, op)
    )
    if dist is not None:
        return _apply_out_dtype(ctx, dist)
    return _apply_out_dtype(ctx, _reduction_types(ctx, array, axis, keepdims, has_dtype, op))


def _reduction_dtype_only(
    ctx: FunctionContext | MethodContext, array: Type, has_dtype: bool, op: str
) -> Type:
    """The stubs' answer for a shape numty cannot read, with numty's dtype.

    The dtype of a reduction does not depend on the shape, and the stubs
    get it wrong for an array result: `sum` of uint8 along an axis comes
    back signed where numpy gives uint64.
    """
    default = get_proper_type(ctx.default_return_type)
    source = get_proper_type(array)
    if has_dtype or not all(
        isinstance(t, Instance) and t.type.fullname == NDARRAY_FULLNAME and len(t.args) == 2
        for t in (default, source)
    ):
        return ctx.default_return_type
    assert isinstance(default, Instance) and isinstance(source, Instance)
    if isinstance(get_proper_type(default.args[0]), TupleType):
        # A fixed-rank answer (keepdims on a mixed shape) is widened by
        # the caller's guard, which has to see the stubs' own object.
        return ctx.default_return_type
    out_dtype = _reduction_out_dtype(ctx, op, source.args[1])
    if out_dtype is None:
        return ctx.default_return_type
    return default.copy_modified(args=[default.args[0], out_dtype])


def _reduction_types(
    ctx: FunctionContext | MethodContext,
    array: Type,
    axis: list[int] | None | str,
    keepdims: bool | None,
    has_dtype: bool,
    op: str = "sum",
) -> Type:
    parts = _parts_or_unpack(array) or _unknown_axis_parts(array)
    if parts is None:
        return _reduction_dtype_only(ctx, array, has_dtype, op)
    dims, dtype = parts
    if _has_unpack(dims):
        split = _split_flat(dims)
        if split is None:
            return ctx.default_return_type
        out_dtype = _reduction_out_dtype(ctx, op, dtype)
        if out_dtype is None:
            if _datetime_kind(dtype) == "M" and op.removeprefix("nan") == "mean":
                return AnyType(TypeOfAny.explicit)
            return AnyType(TypeOfAny.from_omitted_generics)
        return _solve_reduction_unpack(
            ctx, *split, dtype, axis, keepdims, has_dtype, out_dtype, op=op
        )
    out_dtype = _reduction_out_dtype(ctx, op, dtype)
    if out_dtype is None:
        # mean(datetime) always raises (UFuncTypeError, issue #144):
        # bail to Any so no float64 (or other) dtype is claimed. The
        # stubs already emit the arg-type error.
        if _datetime_kind(dtype) == "M" and op.removeprefix("nan") == "mean":
            return AnyType(TypeOfAny.explicit)
        return ctx.default_return_type
    if axis == "absent":
        # An omitted axis reduces every axis, exactly like axis=None.
        axis = "none"
    if axis == "none":
        # Explicit axis=None reduces every axis: keepdims keeps them all
        # as 1s, otherwise the result is a scalar of the output dtype
        # (the stubs leave explicit None as Any).
        if keepdims is None or has_dtype:
            return ctx.default_return_type
        if keepdims:
            return _solve_reduction(
                ctx, dims, dtype, list(range(len(dims))), True, has_dtype, out_dtype, op
            )
        scalar = _scalar_of_dtype(out_dtype)
        return scalar if scalar is not None else ctx.default_return_type
    return _solve_reduction(ctx, dims, dtype, axis, keepdims, has_dtype, out_dtype, op)


def reduction_method_hook(ctx: MethodContext, op: str) -> Type:
    return _reduction_inputs(ctx, ctx.type, op)


def _ufunc_reduce_types(
    ctx: FunctionContext | MethodContext,
    array: Type,
    axis: list[int] | None | str,
    keepdims: bool | None,
    op: str = "reduce",
) -> Type:
    """ufunc reduce: like sum, but a missing axis reduces axis 0 (issue #142).

    Only the shape is solved here; the dtype stays unknown (see
    _ufunc_any_dtype).
    """
    parts = _parts_or_unpack(array)
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    dtype = _ufunc_any_dtype()
    if _has_unpack(dims):
        if axis == "none":
            return ctx.default_return_type
        split = _split_flat(dims)
        if split is None:
            return ctx.default_return_type
        return _solve_reduction_unpack(
            ctx, *split, dtype, axis, keepdims, False, dtype, op=op
        )
    if axis == "none":
        if keepdims is None:
            return ctx.default_return_type
        if keepdims:
            return _ndarray_type(ctx, [1] * len(dims), dtype)
        return _maybe_scalar(ctx, [], dtype)
    return _solve_reduction(ctx, dims, dtype, axis, keepdims, False, dtype, op=op)


def _ufunc_reduce_inputs(ctx: MethodContext, array: Type, op: str = "reduce") -> Type:
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if axis == "absent":
        # Unlike ndarray.sum (axis=None), ufunc reduce defaults to axis 0.
        axis = [0]
    keep_pair = _formal_pair(ctx, "keepdims")
    keepdims = False if keep_pair is None else _fold_flag(keep_pair)
    dist = _distribute_solve(
        [array], lambda t: _ufunc_reduce_types(ctx, t, axis, keepdims, op)
    )
    if dist is not None:
        return _apply_out_dtype(ctx, dist)
    return _apply_out_dtype(ctx, _ufunc_reduce_types(ctx, array, axis, keepdims, op))


def _ufunc_accumulate_types(
    ctx: FunctionContext | MethodContext, array: Type, op: str = "accumulate"
) -> Type:
    parts = _parts_or_unpack(array)
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    if _has_unpack(dims) or not dims:
        return ctx.default_return_type
    return _ndarray_type(ctx, list(dims), _ufunc_any_dtype())


def _ufunc_accumulate_inputs(
    ctx: MethodContext, array: Type, op: str = "accumulate"
) -> Type:
    """ufunc accumulate keeps the input shape (issue #142).

    axis=None is a runtime ValueError and a 0-d input has no loop
    ("cannot accumulate on a scalar"), so both stay with the stubs.
    """
    parts = _parts_or_unpack(array)
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    if _has_unpack(dims) or not dims:
        return ctx.default_return_type
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if axis == "absent":
        axis = [0]
    if not isinstance(axis, list) or len(axis) != 1 or not isinstance(axis[0], int):
        return ctx.default_return_type
    norm = _normalize_axis_0d(axis[0], len(dims))
    if norm is None:
        ctx.api.fail(
            f"{op} axis {axis[0]} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    dist = _distribute_solve([array], lambda t: _ufunc_accumulate_types(ctx, t, op))
    if dist is not None:
        return _apply_out_dtype(ctx, dist)
    return _apply_out_dtype(ctx, _ufunc_accumulate_types(ctx, array, op))


def _ufunc_outer_shape(tp: Type) -> list[int | Type] | None:
    """Operand dims for outer: arrays contribute their shape, scalars []."""
    part = _parts_or_unpack(tp)
    if part is not None:
        dims, _ = part
        if _has_unpack(dims):
            return None
        return list(dims)
    if _scalar_key(tp) is not None or _scalar_ret_name(tp) is not None:
        return []
    return None


def _ufunc_outer_types(
    ctx: FunctionContext | MethodContext, left: Type, right: Type
) -> Type:
    """ufunc outer concatenates the operand shapes (issue #142)."""
    first = _ufunc_outer_shape(left)
    second = _ufunc_outer_shape(right)
    if first is None or second is None:
        return ctx.default_return_type
    return _maybe_scalar(ctx, first + second, _ufunc_any_dtype())


def _ufunc_outer_inputs(ctx: MethodContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) != 2:
        return ctx.default_return_type
    dist = _distribute_solve(
        [pairs[0][0], pairs[1][0]], lambda l, r: _ufunc_outer_types(ctx, l, r)
    )
    if dist is not None:
        return _apply_out_dtype(ctx, dist)
    return _apply_out_dtype(ctx, _ufunc_outer_types(ctx, pairs[0][0], pairs[1][0]))


def _ufunc_any_dtype() -> Type:
    """Canonical unknown dtype for ufunc-method solves (issue #142).

    Bare Any, never a candidate's stub dtype: overload candidates carry
    different defaults (Incomplete vs NDArray[Incomplete]), and hook
    returns that differ across matching candidates trip mypy's
    overload-ambiguity erasure. Only the shape is solved here.
    """
    return AnyType(TypeOfAny.explicit)


def _reduceat_length(pair: tuple[Type, Expression | None] | None) -> int | Type | None:
    """Length of a reduceat indices argument, else None to bail.

    A display's length is its item count; a 1-D array's is its dim
    (concrete or symbolic). Anything else (scalars, which fail at
    runtime, 0-d/2-d arrays, dynamic sequences) stays with the stubs.
    """
    if pair is None:
        return None
    tp, expr = pair
    if isinstance(expr, (ListExpr, TupleExpr)):
        if any(isinstance(e, (ListExpr, TupleExpr, StarExpr)) for e in expr.items):
            return None
        return len(expr.items)
    parts = _ndarray_parts(tp)
    if parts is not None:
        dims, _ = parts
        if len(dims) == 1 and isinstance(dims[0], (int, TypeVarType)):
            return dims[0]
    return None


def _ufunc_reduceat_types(
    ctx: FunctionContext | MethodContext,
    array: Type,
    length: int | Type | None,
    axis: list[int] | None | str,
    op: str = "reduceat",
) -> Type:
    """ufunc reduceat: the axis dim becomes the indices length (issue #142)."""
    parts = _parts_or_unpack(array)
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    if _has_unpack(dims) or not dims or length is None:
        # *Ts shapes stay with the stubs, and numpy refuses 0-d input
        # ("cannot reduceat on a scalar") while the stubs accept it.
        return ctx.default_return_type
    if not isinstance(axis, list) or len(axis) != 1 or not isinstance(axis[0], int):
        return ctx.default_return_type
    norm = _normalize_axis_0d(axis[0], len(dims))
    if norm is None:
        ctx.api.fail(
            f"{op} axis {axis[0]} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims)
    out[norm] = length
    return _ndarray_type(ctx, out, _ufunc_any_dtype())


def _ufunc_reduceat_inputs(
    ctx: MethodContext, array: Type, op: str = "reduceat"
) -> Type:
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if axis == "absent":
        axis = [0]
    length = _reduceat_length(_formal_pair(ctx, "indices"))
    dist = _distribute_solve(
        [array], lambda t: _ufunc_reduceat_types(ctx, t, length, axis, op)
    )
    if dist is not None:
        return _apply_out_dtype(ctx, dist)
    return _apply_out_dtype(ctx, _ufunc_reduceat_types(ctx, array, length, axis, op))


def ufunc_method_hook(ctx: MethodContext, op: str) -> Type:
    """np.add.reduce/.accumulate/.outer/.reduceat: axis checks + shapes.

    reduce drops the axis (axis 0 by default), accumulate keeps the
    shape, outer concatenates the operand shapes, and reduceat sizes
    the axis by the indices length. Only the shape is solved here; the
    dtype stays unknown. add.at is unhooked (returns None).
    """
    if op == "outer":
        return _ufunc_outer_inputs(ctx)
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    if op == "reduce":
        return _ufunc_reduce_inputs(ctx, pairs[0][0], op)
    if op == "accumulate":
        return _ufunc_accumulate_inputs(ctx, pairs[0][0], op)
    if op == "reduceat":
        return _ufunc_reduceat_inputs(ctx, pairs[0][0], op)
    return ctx.default_return_type


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
            _swallowed(ctx, "_fold_index_bound")
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
    0-d integer arrays act like scalars. Bool arrays are masks (an n-D
    mask consumes n axes for one data-dependent count); other dtypes
    always raise.
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
            # Keep the full mask shape: an n-D mask consumes n axes for
            # one data-dependent count (issue #49).
            return ("mask", list(part[0]))
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
            mask_dims: list[int | Type] = list(item[1])
            rank = len(mask_dims)
            if rank == 0 or i + rank > len(prefix):
                return None
            for off, (md, pd) in enumerate(zip(mask_dims, prefix[i : i + rank])):
                if _dim_eq(md, pd):
                    continue
                if isinstance(md, int) and isinstance(pd, int):
                    ctx.api.fail(
                        f"boolean index did not match indexed array along axis {i + off}; "
                        f"size of axis is {pd} but size of corresponding boolean axis is {md} "
                        f"(shape {shape_repr})",
                        ctx.context,
                        code=NUMTY_AXIS,
                    )
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
    # A k-d mask consumes k axes for one count (issue #49); a 0-d mask
    # adds a leading axis like newaxis.
    consuming = 0
    for item in items:
        if item[0] in ("int", "int_any", "slice", "fancy", "fancy0d"):
            consuming += 1
        elif item[0] == "mask":
            consuming += len(item[1])
    if consuming > ndim:
        ctx.api.fail(
            f"too many indices for {ndim} dimensions "
            f"(got {consuming}, shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type

    def _is_advanced(item: tuple) -> bool:
        if item[0] in ("int", "int_any", "fancy", "fancy0d"):
            return True
        return item[0] == "mask" and len(item[1]) > 0

    has_trigger = any(
        item[0] == "fancy" or (item[0] == "mask" and len(item[1]) > 0)
        for item in items
    )
    if has_trigger and any(item[0] == "mask" and len(item[1]) == 0 for item in items):
        # 0-d mask plus another advanced index: placement is unclear.
        return ctx.default_return_type
    front = False
    first_adv = -1
    last_adv = -1
    if has_trigger:
        adv_positions = [n for n, item in enumerate(items) if _is_advanced(item)]
        first_adv, last_adv = adv_positions[0], adv_positions[-1]
        # NumPy moves the advanced block to the front whenever a slice,
        # ellipsis, newaxis (or any other non-advanced index) separates
        # two advanced indices.
        front = any(
            not _is_advanced(items[n]) for n in range(first_adv, last_adv + 1)
        )
    span = ndim - consuming
    # First pass: validate (int bounds, mask lengths, slice steps) and
    # collect kept dims plus the broadcast advanced block.
    kept_before: list[int | Type] = []
    kept_after: list[int | Type] = []
    adv_block: list[int | Type] = []
    adv_seen = False
    dim = 0
    for n, item in enumerate(items):
        kind = item[0]
        is_adv = _is_advanced(item) if has_trigger else False
        # Kept fragments before the first advanced index stay before the
        # block in place; everything else non-advanced goes after it.
        # Under front placement the split is irrelevant (block leads).
        target = kept_after if (has_trigger and not front and adv_seen) else kept_before
        if kind == "newaxis":
            # A newaxis inside a contiguous block cannot happen (it
            # would force front); under front it stays with the kept
            # dims in order.
            (kept_after if (has_trigger and not front and adv_seen) else kept_before).append(1)
        elif kind == "ellipsis":
            target.extend(dims[dim : dim + span])
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
            if has_trigger:
                adv_seen = adv_seen or is_adv
        elif kind == "fancy0d":
            dim += 1
            if has_trigger:
                adv_seen = adv_seen or is_adv
        elif kind == "fancy":
            shape: list[int | Type] = list(item[1])
            if not adv_block:
                adv_block = shape
            else:
                merged, cause = _broadcast_result(adv_block, shape, ctx)
                if cause is not None:
                    ctx.api.fail(
                        f"indexing arrays could not be broadcast together: {cause}",
                        ctx.context,
                        code=NUMTY_BROADCAST,
                    )
                    return ctx.default_return_type
                if merged is None:
                    return ctx.default_return_type
                adv_block = merged
            dim += 1
            if has_trigger:
                adv_seen = True
        elif kind == "mask":
            mask_dims: list[int | Type] = list(item[1])
            if not mask_dims:
                (kept_after if (has_trigger and not front and adv_seen) else kept_before).append(1)
            else:
                k = len(mask_dims)
                if dim + k > len(dims):
                    return ctx.default_return_type
                for off, (md, pd) in enumerate(zip(mask_dims, dims[dim : dim + k])):
                    if _dim_eq(md, pd):
                        continue
                    if isinstance(md, int) and isinstance(pd, int):
                        ctx.api.fail(
                            f"boolean index did not match indexed array along axis {dim + off}; "
                            f"size of axis is {pd} but size of corresponding boolean axis is {md} "
                            f"(shape {_dims_str(dims)})",
                            ctx.context,
                            code=NUMTY_AXIS,
                        )
                        return ctx.default_return_type
                unk = _fresh_unknown(ctx, "mask", dim)
                if not adv_block:
                    adv_block = [unk]
                else:
                    merged, cause = _broadcast_result(adv_block, [unk], ctx)
                    if cause is not None:
                        ctx.api.fail(
                            f"indexing arrays could not be broadcast together: {cause}",
                            ctx.context,
                            code=NUMTY_BROADCAST,
                        )
                        return ctx.default_return_type
                    if merged is None:
                        return ctx.default_return_type
                    adv_block = merged
                dim += k
                if has_trigger:
                    adv_seen = True
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
            frag: int | Type
            if isinstance(size, int):
                if "unknown" in (begin, end, stride):
                    frag = _fresh_unknown(ctx, "slice", dim)
                else:
                    bounds = tuple(
                        None if b == "missing" else b for b in (begin, end, stride)
                    )
                    try:
                        frag = len(range(*slice(*bounds).indices(size)))
                    except ValueError:
                        ctx.api.fail(
                            "slice step is zero", ctx.context, code=NUMTY_AXIS
                        )
                        return ctx.default_return_type
            elif begin == "missing" and end == "missing":
                if stride == "missing" or stride in (1, -1):
                    frag = size
                elif stride == 0:
                    ctx.api.fail(
                        "slice step is zero", ctx.context, code=NUMTY_AXIS
                    )
                    return ctx.default_return_type
                else:
                    frag = _fresh_unknown(ctx, "slice", dim)
            elif (stride == "missing" or stride == 1) and (
                begin == "missing" or (isinstance(begin, int) and begin >= 0)
            ) and (end == "missing" or (isinstance(end, int) and end < 0)):
                # Literal [s:e] on a linear dim N+c removes s from the
                # front and -e from the back: (N+2)[1:-1] is (N,)
                # (issue #62). Anything else (negative starts, positive
                # ends, steps) depends on the runtime length: bail out.
                fa = _resolve_dim(size)
                if isinstance(fa, _LinForm):
                    s = 0 if begin == "missing" else begin
                    e = 0 if end == "missing" else end
                    assert isinstance(s, int) and isinstance(e, int)
                    frag = _materialize_dim(
                        ctx,
                        "slice",
                        dim,
                        _LinForm(fa.coeffs, fa.const + e - s),
                        prefer=size if isinstance(size, TypeVarType) else None,
                    )
                else:
                    frag = _fresh_unknown(ctx, "slice", dim)
            else:
                frag = _fresh_unknown(ctx, "slice", dim)
            (kept_after if (has_trigger and not front and adv_seen) else kept_before).append(frag)
            dim += 1
    if not has_trigger:
        kept_before.extend(dims[dim:])
        out = kept_before
    elif front:
        out = [*adv_block, *kept_before, *kept_after, *dims[dim:]]
    else:
        out = [*kept_before, *adv_block, *kept_after, *dims[dim:]]
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


def setitem_method_hook(ctx: MethodContext) -> Type:
    """Shape-check `a[idx] = value` (issue #51).

    Solves the target with the read path so out-of-range, too-many and
    mask-length errors fire on writes too, then requires the value to
    broadcast one-directionally into the target. Scalars always fit;
    unpack/unknown/symbolic targets bail. Returns the stub default
    (None) in every case.
    """
    pairs = _all_positionals(ctx)
    if len(pairs) != 2:
        return ctx.default_return_type
    index_pair, value_pair = pairs[0], pairs[1]
    value_type = value_pair[0]
    # Union LHS: emit target errors per member, then bail on the value.
    if _union_members(ctx.type) is not None:
        _distribute_solve([ctx.type], lambda t: _getitem_types(ctx, t, index_pair))
        return ctx.default_return_type
    parts = _ndarray_parts(ctx.type)
    if parts is None:
        # Unpack or unknown LHS: still solve for target errors, no value check.
        _getitem_types(ctx, ctx.type, index_pair)
        return ctx.default_return_type
    lhs_dims, lhs_dtype = parts
    target_type = _solve_getitem(ctx, lhs_dims, lhs_dtype, index_pair)
    target_parts = _ndarray_parts(target_type)
    if target_parts is None:
        # Scalar target (all ints), an error already emitted, or a bail.
        # Only the scalar case needs a value check here.
        if _scalar_key(value_type) is not None:
            return ctx.default_return_type
        if _is_scalar_target(ctx, ctx.type, index_pair):
            value_parts = _ndarray_parts(value_type)
            if value_parts is None or not value_parts[0]:
                return ctx.default_return_type
            ctx.api.fail(
                f"cannot assign value of shape {_dims_str(value_parts[0])} "
                f"into scalar target (shape {_dims_str(lhs_dims)})",
                ctx.context,
                code=NUMTY_BROADCAST,
            )
        return ctx.default_return_type
    target_dims = target_parts[0]
    if _scalar_key(value_type) is not None:
        return ctx.default_return_type
    value_parts = _ndarray_parts(value_type)
    if value_parts is None:
        return ctx.default_return_type
    value_dims = value_parts[0]
    if len(value_dims) > len(target_dims):
        ctx.api.fail(
            f"cannot assign value of shape {_dims_str(value_dims)} "
            f"into target of shape {_dims_str(target_dims)}",
            ctx.context,
            code=NUMTY_BROADCAST,
        )
        return ctx.default_return_type
    offset = len(target_dims) - len(value_dims)
    for t, v in zip(target_dims[offset:], value_dims):
        if _dim_eq(t, v):
            continue
        if isinstance(v, int) and v == 1:
            continue
        if isinstance(t, int) and isinstance(v, int):
            ctx.api.fail(
                f"cannot assign value of shape {_dims_str(value_dims)} "
                f"into target of shape {_dims_str(target_dims)}",
                ctx.context,
                code=NUMTY_BROADCAST,
            )
            return ctx.default_return_type
        # Symbolic or unknown: unprovable, bail silently.
        return ctx.default_return_type
    return ctx.default_return_type


def _is_scalar_target(
    ctx: MethodContext, tp: Type, pair: tuple[Type, Expression | None]
) -> bool:
    """Whether the index consumes every axis (scalar target), else False.

    Only for the fixed-rank path; unpack or unknown bails as non-scalar.
    """
    parts = _ndarray_parts(tp)
    if parts is None:
        return False
    dims, _ = parts
    ndim = len(dims)
    split = _index_pairs(ctx, pair)
    if split is None:
        return False
    items = _parse_index_items(ctx, split)
    if items is None:
        return False
    if any(item[0] == "ellipsis" for item in items):
        # Ellipsis keeps span dims unless it spans zero and the rest are ints.
        consuming = 0
        for item in items:
            if item[0] in ("int", "int_any", "slice", "fancy", "fancy0d"):
                consuming += 1
            elif item[0] == "mask":
                consuming += len(item[1])
        if consuming != ndim:
            return False
    else:
        consuming = 0
        for item in items:
            if item[0] in ("int", "int_any", "fancy0d"):
                consuming += 1
            elif item[0] in ("slice", "fancy", "newaxis"):
                return False
            elif item[0] == "mask":
                return False
            elif item[0] == "badindex":
                return False
            else:
                return False
        if consuming != ndim:
            return False
    return True


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
    """`&` and `|` on bool/int arrays: broadcast shapes (issue #54).

    Bool pairs keep bool out; int/uint pairs promote like add. Plain
    `bool` variables (not just literals) count as bool scalars.
    Anything else bails to the stubs.
    """
    pairs = _all_positionals(ctx)
    if len(pairs) != 1:
        return ctx.default_return_type
    dist = _distribute_solve([ctx.type, pairs[0][0]], lambda l, r: _logical_types(ctx, l, r))
    if dist is not None:
        return dist
    return _logical_types(ctx, ctx.type, pairs[0][0])


def _is_plain_bool(tp: Type) -> bool:
    proper = get_proper_type(tp)
    return isinstance(proper, Instance) and proper.type.fullname == "builtins.bool"


def _is_builtin_int(tp: Type) -> bool:
    proper = get_proper_type(tp)
    return isinstance(proper, Instance) and proper.type.fullname == "builtins.int"


def _logical_types(ctx: FunctionContext | MethodContext, left: Type, right: Type) -> Type:
    parts = _parts_or_unpack(left)
    if parts is None:
        return ctx.default_return_type
    base = _dtype_base(parts[1])
    if base not in ("b", "i", "u"):
        return ctx.default_return_type
    other = _parts_or_unpack(right)
    if other is not None:
        if _dtype_base(other[1]) not in ("b", "i", "u"):
            return ctx.default_return_type
        # Bool pairs keep bool out; int pairs promote like add.
        if base == "b" and _dtype_base(other[1]) == "b":
            dims = _broadcast_dims(ctx, left, right)
            if dims is None:
                return ctx.default_return_type
            dtype = _construct_dtype(ctx, "numpy.bool")
            if dtype is None:
                return ctx.default_return_type
            return _ndarray_type(ctx, dims, dtype)
        return _solve_broadcast(ctx, left, right)
    if _bool_value(right) is not None or _is_plain_bool(right):
        if base != "b":
            # int & bool variable: a scalar always fits — keep left dims.
            left_parts = _ndarray_parts(left)
            if left_parts is None:
                return ctx.default_return_type
            return _ndarray_type(ctx, left_parts[0], left_parts[1])
        dims = _broadcast_dims(ctx, left, right)
        if dims is None:
            # Plain bool scalar: _broadcast_dims bails on non-literal,
            # but a scalar always fits — keep the left dims.
            left_parts = _ndarray_parts(left)
            if left_parts is None:
                return ctx.default_return_type
            dims = left_parts[0]
        dtype = _construct_dtype(ctx, "numpy.bool")
        if dtype is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, dtype)
    if _scalar_key(right) is not None:
        # Python int literal / numpy concrete scalar: broadcast keeps dims.
        return _solve_broadcast(ctx, left, right)
    if _is_builtin_int(right):
        # Plain int variable: a scalar always fits — keep left dims/dtype.
        left_parts = _ndarray_parts(left)
        if left_parts is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, left_parts[0], left_parts[1])
    return ctx.default_return_type


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
    ctx: FunctionContext | MethodContext,
    dims: list[int],
    axis: list[int] | None | str,
    op: str = "argmax",
) -> Type:
    """argmax/argmin: drop the axis, integer positions out (issue #47).

    With keepdims=True the dropped axes stay as size 1; an omitted (or
    None) axis flattens, so keepdims gives all ones and the plain form
    stays with the stubs (a scalar). An undecidable keepdims bails.
    """
    keep_pair = _formal_pair(ctx, "keepdims")
    keepdims = False if keep_pair is None else _fold_flag(keep_pair)
    if not isinstance(axis, list):
        if axis in ("absent", "none"):
            if keepdims is None:
                return ctx.default_return_type
            if keepdims:
                return _ndarray_type(ctx, [1] * len(dims), _default_dtype(ctx))
            return ctx.default_return_type
        return ctx.default_return_type
    if keepdims is None:
        return ctx.default_return_type
    ndim = len(dims)
    normalized: list[int] = []
    for ax in axis:
        norm = _normalize_axis_0d(ax, ndim)
        if norm is None:
            ctx.api.fail(
                f"{op} axis {ax} is out of range for {ndim} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        if norm in normalized:
            ctx.api.fail(
                f"{op} has duplicate axis {ax} in {tuple(axis)} "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        normalized.append(norm)
    chosen = set(normalized)
    if keepdims:
        return _ndarray_type(
            ctx, [d if i not in chosen else 1 for i, d in enumerate(dims)], _default_dtype(ctx)
        )
    return _ndarray_type(ctx, [d for i, d in enumerate(dims) if i not in chosen], _default_dtype(ctx))


def _solve_same_or_flat(
    ctx: FunctionContext | MethodContext,
    dims: list[int | Type],
    dtype: Type,
    axis: list[int] | None | str,
    int_out: bool,
    op: str = "sort",
) -> Type:
    """sort/partition (same shape) and argsort/argpartition (integer out)."""
    if int_out and not dims:
        # argsort/argpartition treat a 0-d array as one element and
        # return (1,); sort and partition reject it (axis out of range).
        dims = [1]
    if axis == "absent":
        # The default axis is -1, which never changes the shape.
        return _ndarray_type(ctx, dims, _default_dtype(ctx) if int_out else dtype)
    if not isinstance(axis, list):
        if axis == "none":
            total = _mul_dims(ctx, "sort", 0, dims)
            return _ndarray_type(ctx, [total], _default_dtype(ctx) if int_out else dtype)
        return ctx.default_return_type
    ndim = len(dims)
    for ax in axis:
        if _normalize_axis(ax, ndim) is None:
            ctx.api.fail(
                f"{op} axis {ax} is out of range for {ndim} dimensions "
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


ARG_DROP = frozenset({"argmax", "argmin", "nanargmax", "nanargmin"})
ARG_SAME_INT = frozenset({"argsort", "argpartition"})
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
        return _solve_argdrop(ctx, dims, axis, op)
    return _solve_same_or_flat(ctx, dims, dtype, axis, int_out=op in ARG_SAME_INT, op=op)


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
    dims = _builder_shape(ctx, *pair, ctx.default_return_type, shaped_stub=False)
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
        # random/standard_normal take dtype= (float32/float64); normal and
        # uniform have no dtype parameter and stay float64 (issue #61).
        if op in ("random", "standard_normal"):
            named = _named_dtype(ctx)
            if named is not None:
                name = _dtype_fullname(named)
                if name not in ("numpy.float32", "numpy.float64"):
                    return ctx.default_return_type
                return _ndarray_type(ctx, size, named)
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
        size = _parse_size_arg(ctx)
        if size is None:
            return ctx.default_return_type
        if isinstance(size, str):
            # size=None (or omitted) means a single sample.
            size = []
        if any(not isinstance(d, (int, TypeVarType)) for d in size):
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
        dims, dtype = parts
        axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
        if axis is None or axis == "none":
            return ctx.default_return_type
        if axis == "absent":
            k = 0
        else:
            if len(axis) != 1 or not isinstance(axis[0], int):
                return ctx.default_return_type
            norm = _normalize_axis(axis[0], len(dims))
            if norm is None:
                ctx.api.fail(
                    f"choice axis {axis[0]} is out of range for "
                    f"{len(dims)} dimensions (shape {_dims_str(dims)})",
                    ctx.context,
                    code=NUMTY_AXIS,
                )
                return ctx.default_return_type
            k = norm
        return _ndarray_type(ctx, dims[:k] + size + dims[k + 1 :], dtype)
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
        if not dims:
            ctx.api.fail(
                "permutation requires an int or at least 1 dimension, got shape ()",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
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
    if len(q_dims) > 1:
        # numpy takes some multi-dimensional q and rejects others
        # ("q must be a scalar or 1d") depending on the path taken.
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
    if kind == "c":
        ctx.api.fail(
            "percentile needs an array of real numbers, got a complex array",
            ctx.context,
            code=NUMTY_BROADCAST,
        )
        return ctx.default_return_type
    if kind not in ("f", "i", "u"):
        # bool input is rejected or kept as bool depending on q.
        return ctx.default_return_type
    # The result dtype is result_type(a, q), lifted to float64 when
    # that is an integer. A Python scalar q is weak and leaves a's
    # dtype alone; a list display is a float64 array; an array or
    # numpy scalar q brings its own dtype (f16 data, f32 q -> f32).
    q_tp, q_expr = pairs[1]
    q_dtype: Type | None
    if isinstance(q_expr, (ListExpr, TupleExpr)):
        q_dtype = _construct_dtype(ctx, "numpy.float64")
    elif _weak_name(q_tp) is not None:
        q_dtype = dtype_a
    else:
        q_parts = _ndarray_parts(q_tp)
        if q_parts is not None:
            q_dtype = q_parts[1]
        else:
            key = _scalar_key(q_tp)
            q_dtype = _construct_dtype(ctx, key[1]) if key is not None else None
    if q_dtype is None:
        return ctx.default_return_type
    q_ks = _dtype_kind_size(q_dtype)
    if q_ks is None or q_ks[0] != "f":
        # An integer q array takes numpy down a path that can keep the
        # input's integer dtype, and a complex q is rejected.
        return ctx.default_return_type
    out_dtype = _promote_all(ctx, [dtype_a, q_dtype])
    if out_dtype is None:
        return ctx.default_return_type
    out_ks = _dtype_kind_size(out_dtype)
    if out_ks is None:
        return ctx.default_return_type
    if out_ks[0] != "f":
        out_dtype = _construct_dtype(ctx, "numpy.float64")
        if out_dtype is None:
            return ctx.default_return_type
    return _ndarray_type(ctx, [*q_dims, *remaining], out_dtype)



def reduction_function_hook(ctx: FunctionContext, op: str = "sum") -> Type:
    pairs = _all_positionals_expanded(ctx)
    if pairs is None:
        return ctx.default_return_type
    if not pairs:
        return ctx.default_return_type
    return _reduction_inputs(ctx, pairs[0][0], op)


def _scalar_of_dtype(dtype: Type) -> Type | None:
    """The numpy scalar Instance carried by a dtype[...], else None."""
    proper = get_proper_type(dtype)
    if isinstance(proper, Instance) and proper.args:
        inner = get_proper_type(proper.args[0])
        if isinstance(inner, Instance):
            return inner
    return None


def _maybe_scalar(
    ctx: FunctionContext | MethodContext, dims: list[int | Type], dtype: Type
) -> Type:
    """An ndarray for the dims, or the numpy scalar when none are left.

    Fully reduced results (and scalar-only ufunc calls) are numpy
    scalars at runtime, not 0-d arrays, so `total: float = ...` checks
    (issue #62). Falls back to the stubs when the dtype has no scalar.
    """
    if dims:
        return _ndarray_type(ctx, dims, dtype)
    scalar = _scalar_of_dtype(dtype)
    return scalar if scalar is not None else ctx.default_return_type


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
    if axis in ("absent", "none"):
        # With no axis numpy requires weights shaped exactly like the
        # input ("Axis must be specified when shapes of a and weights
        # differ"), so 1-D weights only fit a 1-D input of the same size.
        if len(adims) != 1:
            return (
                f"average weights with shape {_dims_str(wdims)} need axis= "
                f"to match shape {_dims_str(adims)}"
            )
        if not isinstance(adims[0], int) or not isinstance(wdims[0], int):
            return None
        if adims[0] == wdims[0]:
            return None
        return (
            f"average weights length {wdims[0]} does not match "
            f"size {adims[0]} (shape {_dims_str(adims)})"
        )
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


def _average_weighted_dtype(
    ctx: FunctionContext, avg: Type, array: Type, weights: Type
) -> Type:
    """The weighted average's dtype: result_type(a, weights[, float64]).

    Unlike the unweighted mean, the weights join the promotion
    (complex64 data with float64 weights is complex128), and integer
    or bool data adds float64. Unknown weights bail to the stubs.
    """
    a_parts = _ndarray_parts(array)
    w_parts = _ndarray_parts(weights)
    if a_parts is None or w_parts is None:
        return ctx.default_return_type
    dtypes = [a_parts[1], w_parts[1]]
    ks = _dtype_kind_size(a_parts[1])
    if ks is None:
        return ctx.default_return_type
    if ks[0] in ("i", "u", "b"):
        f64 = _construct_dtype(ctx, "numpy.float64")
        if f64 is None:
            return ctx.default_return_type
        dtypes.append(f64)
    out = _promote_all(ctx, dtypes)
    if out is None:
        return ctx.default_return_type
    proper = get_proper_type(avg)
    if isinstance(proper, Instance) and proper.type.fullname == NDARRAY_FULLNAME:
        return proper.copy_modified(args=[proper.args[0], out])
    scalar = _scalar_of_dtype(out)
    return scalar if scalar is not None else ctx.default_return_type


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
    returned_pair = _formal_pair(ctx, "returned")
    returned = False if returned_pair is None else _fold_flag(returned_pair)
    if returned is None:
        return ctx.default_return_type
    avg = _reduction_inputs(ctx, pairs[0][0])
    if weights is not None and not _is_none_type(weights[0]):
        avg = _average_weighted_dtype(ctx, avg, pairs[0][0], weights[0])
    if not returned:
        return avg
    # returned=True: (average, sum_of_weights), same shape for both.
    return _linalg_tuple(ctx, [avg, avg])


def _flatten_total_dim(
    ctx: FunctionContext | MethodContext, dims: list[int | Type], op: str
) -> int | Type | None:
    """Total elements for a flatten (ravel/cumsum axis=None), else None.

    Issue #145: method flatten paths pinned to the literal total like
    reshape(-1). All-literal totals stay ints; pure-symbolic totals stay
    exact (monomials) via _mul_dims. Unknown-containing dims (?U, Div,
    unresolvable) stay with the stubs: the ?U-empty edge (empty-mask ?U
    vs ?U-1, which numpy clamps) is left untouched beyond this bail.
    """
    if _has_unpack(dims):
        return None
    for d in dims:
        if isinstance(d, TypeVarType) and _is_unknown_fullname(d.fullname):
            return None
        f = _resolve_dim(d)
        if f is None or isinstance(f, _DivForm) or _form_has_unknown(f):
            return None
    if not dims:
        return 1
    try:
        return _mul_dims(ctx, op, 0, dims)
    except Exception:
        _swallowed(ctx, "_flatten_total_dim")
        return None


def _cumsum_solve(
    ctx: FunctionContext | MethodContext, array: Type, op: str = "cumsum"
) -> Type:
    """cumsum/cumprod keep the input shape; only axis=None flattens.

    axis=None flattens to the literal total like reshape(-1) (issue
    #145): (4, 3) -> (12,). 1-D inputs keep the shape (issue #32:
    np.cumsum without axis on I64[N] is (N,)). Unknown-containing dims
    stay with the stubs. dtype is whatever the stubs claim: only the
    shape is solved here.
    """
    parts = _parts_or_unpack(array)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    if _has_unpack(dims):
        return ctx.default_return_type
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if not isinstance(axis, list) or len(axis) != 1 or not isinstance(axis[0], int):
        # Missing axis (default None): 1-D keeps shape, ND flattens.
        if axis in ("absent", "none") and len(dims) == 1:
            axis = [0]
        elif axis in ("absent", "none"):
            total = _flatten_total_dim(ctx, dims, "cumsum")
            if total is None:
                return ctx.default_return_type
            dtype_pair = _formal_pair(ctx, "dtype")
            if dtype_pair is not None and not _is_none_type(dtype_pair[0]):
                return ctx.default_return_type
            name = _dtype_fullname(dtype)
            if name in (
                "numpy.bool",
                "numpy.int8",
                "numpy.int16",
                "numpy.int32",
                "numpy.int64",
            ):
                promoted = _construct_dtype(ctx, "numpy.int64")
                if promoted is None:
                    return ctx.default_return_type
                return _ndarray_type(ctx, [total], promoted)
            if name in (
                "numpy.uint8",
                "numpy.uint16",
                "numpy.uint32",
                "numpy.uint64",
            ):
                promoted = _construct_dtype(ctx, "numpy.uint64")
                if promoted is None:
                    return ctx.default_return_type
                return _ndarray_type(ctx, [total], promoted)
            if name is None or name not in REDUCTION_DTYPES:
                return ctx.default_return_type
            return _ndarray_type(ctx, [total], dtype)
        else:
            return ctx.default_return_type
    norm = _normalize_axis_0d(axis[0], len(dims))
    if norm is None:
        ctx.api.fail(
            f"{op} axis {axis[0]} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    # A 0-d input with axis 0/-1 accumulates to one element (issue #62).
    out_dims = [1] if not dims else list(dims)
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
        return _ndarray_type(ctx, out_dims, promoted)
    if name in ("numpy.uint8", "numpy.uint16", "numpy.uint32", "numpy.uint64"):
        promoted = _construct_dtype(ctx, "numpy.uint64")
        if promoted is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, out_dims, promoted)
    if name is None or name not in REDUCTION_DTYPES:
        # Non-numeric dtypes (strings, datetime64, ...): the stubs own it.
        return ctx.default_return_type
    return _ndarray_type(ctx, out_dims, dtype)


def cumsum_function_hook(ctx: FunctionContext, op: str = "cumsum") -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _cumsum_solve(ctx, pairs[0][0], op)


def cumsum_method_hook(ctx: MethodContext, op: str = "cumsum") -> Type:
    return _cumsum_solve(ctx, ctx.type, op)


def _ravel_solve(ctx: FunctionContext | MethodContext, array: Type) -> Type:
    """ravel/flatten flatten to the literal total (issue #145).

    (4, 3) -> (12,), like reshape(-1); 0-d -> (1,). Unknown-containing
    dims stay with the stubs (the ?U-empty edge is untouched).
    """
    parts = _parts_or_unpack(array)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    if _has_unpack(dims):
        return ctx.default_return_type
    total = _flatten_total_dim(ctx, dims, "ravel")
    if total is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, [total], dtype)


def ravel_method_hook(ctx: MethodContext) -> Type:
    return _ravel_solve(ctx, ctx.type)


def flatten_method_hook(ctx: MethodContext) -> Type:
    return _ravel_solve(ctx, ctx.type)


def ravel_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _ravel_solve(ctx, pairs[0][0])


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
    if ndim < 1:
        ctx.api.fail(
            f"np.diff requires at least 1 dimension, got shape {_dims_str(dims)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
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
    if _datetime_kind(dtype) == "M":
        # Datetime diffs change dtype (datetime64 -> timedelta64, issue
        # #144): same shape logic, timedelta result.
        out_dtype = _construct_datetime_dtype(ctx, "m")
        if out_dtype is None:
            return ctx.default_return_type
        size = dims[ax]
        out = list(dims)
        if isinstance(size, int):
            out[ax] = max(0, size - n)
        else:
            out[ax] = _dim_add_const(ctx, "diff", ax, size, -n)
        return _ndarray_type(ctx, out, out_dtype)
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
    pairs = _all_positionals_expanded(ctx)
    if pairs is None:
        return ctx.default_return_type
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
    label_first_operand: dict[str, int] = {}
    label_first_dims: dict[str, list[int | Type]] = {}
    counts: dict[str, int] = {}
    ell_inputs: list[list[int | Type]] = []
    for oi, ((dims, _), term) in enumerate(zip(parts, terms)):
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
        bound_labels_seen: set[str] = set()
        for label, size in bound:
            counts[label] = counts.get(label, 0) + 1
            known = label_size.get(label)
            if known is None:
                label_size[label] = size
                label_first_operand[label] = oi
                label_first_dims[label] = list(dims)
            elif not _dim_eq(known, size):
                if (
                    isinstance(known, int)
                    and isinstance(size, int)
                    and (known == 1 or size == 1)
                    and label_first_operand[label] != oi
                    and label not in bound_labels_seen
                ):
                    # A size-1 label broadcasts across operands
                    # (issue #62): keep the larger size. A repeated
                    # label within one operand ('ii' with (1, 3))
                    # still conflicts, as it does in numpy.
                    label_size[label] = max(known, size)
                elif isinstance(known, int) and isinstance(size, int):
                    first_dims = label_first_dims.get(label, [])
                    ctx.api.fail(
                        f"einsum label '{label}' has conflicting sizes: "
                        f"{known} vs {size} (shapes {_dims_str(first_dims)} vs {_dims_str(dims)})",
                        ctx.context,
                        code=NUMTY_SHAPE,
                    )
                    return ctx.default_return_type
                elif _dims_provably_ne(known, size):
                    first_dims = label_first_dims.get(label, [])
                    ctx.api.fail(
                        f"einsum label '{label}' has conflicting sizes: "
                        f"{_dim_str(known)} vs {_dim_str(size)} (shapes {_dims_str(first_dims)} vs {_dims_str(dims)})",
                        ctx.context,
                        code=NUMTY_SHAPE,
                    )
                    return ctx.default_return_type
                else:
                    # Symbolic vs anything unprovable: bail out.
                    return ctx.default_return_type
            bound_labels_seen.add(label)
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
    if not dims_b:
        ctx.api.fail(
            "linalg.solve: b has 0 dimensions "
            f"(shapes {_dims_str(dims_a)} vs {_dims_str(dims_b)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
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
    dtype = _linalg_solve_dtype(ctx, dtype_a, dtype_b)
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
    if _reject_float16_linalg(ctx, dtype, op):
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


def _reject_float16_linalg(ctx: FunctionContext | MethodContext, dtype: Type, op: str) -> bool:
    """Fail for float16 linalg inputs (unsupported), else False."""
    if _dtype_fullname(dtype) == "numpy.float16":
        ctx.api.fail(
            f"array type float16 is unsupported in linalg.{op}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return True
    return False


def _linalg_solve_dtype(
    ctx: FunctionContext, dtype_a: Type, dtype_b: Type
) -> Type | None:
    """Promotion for solve/lstsq: ints become float64, float16 rejected."""
    if _reject_float16_linalg(ctx, dtype_a, "solve") or _reject_float16_linalg(
        ctx, dtype_b, "solve"
    ):
        return None
    name = _promote_dtypes(dtype_a, dtype_b)
    if name is None:
        return None
    kind = _kind_of(name)
    if kind in ("i", "u", "b"):
        return _construct_dtype(ctx, "numpy.float64")
    if name == _dtype_fullname(dtype_a):
        return dtype_a
    if name == _dtype_fullname(dtype_b):
        return dtype_b
    return _construct_dtype(ctx, name)


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
    # lstsq is not a gufunc: a must be 2-D and b at most 2-D (numpy
    # raises LinAlgError otherwise), so there is no batch to broadcast.
    if len(dims_a) != 2:
        ctx.api.fail(
            f"linalg.lstsq requires a 2-dimensional array, got shape "
            f"{_dims_str(dims_a)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    if len(dims_b) > 2:
        ctx.api.fail(
            f"linalg.lstsq requires b with at most 2 dimensions, got shape "
            f"{_dims_str(dims_b)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    rows, cols = dims_a[-2], dims_a[-1]
    if not dims_b:
        return ctx.default_return_type
    if len(dims_b) == 1:
        if not _check_dim_eq(
            ctx,
            dims_b[0],
            rows,
            f"linalg.lstsq inner dimensions disagree: {_dim_str(rows)} vs "
            f"{_dim_str(dims_b[0])} (shapes {_dims_str(dims_a)} vs {_dims_str(dims_b)})",
        ):
            return ctx.default_return_type
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
        core = [cols, dims_b[-1]]
    dtype = _linalg_solve_dtype(ctx, dtype_a, dtype_b)
    if dtype is None:
        return ctx.default_return_type
    # Only the solution is provable (residuals depend on runtime rank),
    # so the remaining items stay Any.
    rest = [AnyType(TypeOfAny.explicit)] * 3
    return _linalg_tuple(ctx, [_ndarray_type(ctx, core, dtype), *rest])


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


def _norm_signature_hook(ctx: FunctionSigContext) -> Type:
    """Widen ord/axis to Any when the call would fail _norm_ord_fits (issue #148).

    The function hook rejects every overload candidate for fro/nuc with a
    single axis (filtered errors), mypy finds no erased target for inline
    "fro" (plain str), and reports `No overload...` + 20 notes instead of
    our targeted message. When the actuals show an explicit ord that would
    reduce 1 dim with a string ord, or reduce >2 dims, return a copy with
    ord/axis as Any so one overload approximately matches; the fallback
    then runs the function hook unfiltered, which reports the single
    [numty-axis] error. All other calls keep the default signature.
    """
    try:
        sig = ctx.default_signature
        if not isinstance(sig, CallableType):
            return ctx.default_signature
        if len(ctx.args) != len(sig.arg_types):
            return ctx.default_signature
        names = list(sig.arg_names)
        try:
            x_idx = names.index("x")
            ord_idx = names.index("ord")
            axis_idx = names.index("axis")
        except ValueError:
            return ctx.default_signature
        if len(ctx.args[x_idx]) != 1:
            return ctx.default_signature
        x_expr = ctx.args[x_idx][0]
        try:
            x_tp = ctx.api.get_expression_type(x_expr)
        except Exception:
            _swallowed(ctx, "_norm_signature_hook")
            return ctx.default_signature
        part = _ndarray_parts(x_tp)
        if part is None:
            return ctx.default_signature
        dims, _ = part
        # ord: absent or None passes; otherwise need explicit value.
        ord_exprs = ctx.args[ord_idx]
        if not ord_exprs:
            return ctx.default_signature
        if len(ord_exprs) != 1:
            return ctx.default_signature
        ord_expr = ord_exprs[0]
        try:
            ord_tp = ctx.api.get_expression_type(ord_expr)
        except Exception:
            _swallowed(ctx, "_norm_signature_hook")
            return ctx.default_signature
        if _is_none_type(ord_tp):
            return ctx.default_signature
        order = _string_value(ord_tp)
        if order is None and _weak_name(ord_tp) not in ("builtins.int", "builtins.float"):
            return ctx.default_signature
        number = _int_value(ord_tp)
        # axis: absent/none reduces all dims; otherwise parse the list.
        axis_exprs = ctx.args[axis_idx]
        if not axis_exprs:
            reduced = len(dims)
        else:
            if len(axis_exprs) != 1:
                return ctx.default_signature
            # _parse_axis_arg needs (type, expr); get type for the axis expr.
            try:
                axis_tp = ctx.api.get_expression_type(axis_exprs[0])
            except Exception:
                _swallowed(ctx, "_norm_signature_hook")
                return ctx.default_signature
            parsed = _parse_axis_arg((axis_tp, axis_exprs[0]))
            if parsed is None:
                return ctx.default_signature
            if parsed in ("absent", "none"):
                reduced = len(dims)
            else:
                assert isinstance(parsed, list)
                # Duplicate axes bail in the function hook (no error), so
                # leave the signature alone.
                if len(set(parsed)) != len(parsed):
                    return ctx.default_signature
                reduced = len(parsed)
        would_fail = False
        if reduced > 2:
            # Explicit ord (we already know ord is explicit here).
            would_fail = True
        elif order is not None and reduced == 1:
            would_fail = True
        if not would_fail:
            return ctx.default_signature
        new_arg_types = list(sig.arg_types)
        new_arg_types[ord_idx] = AnyType(TypeOfAny.explicit)
        new_arg_types[axis_idx] = AnyType(TypeOfAny.explicit)
        return sig.copy_modified(arg_types=new_arg_types)
    except Exception:
        _swallowed(ctx, "_norm_signature_hook")
        return ctx.default_signature


def _linalg_cross_signature_hook(ctx: FunctionSigContext) -> Type:
    """Let `np.linalg.cross` take arrays of rank 3 and up.

    The stubs type both operands as 1-D or 2-D, so a batch of vectors
    (`F64[N, M, 3]`) is an [arg-type] error although numpy accepts any
    rank. When an operand is a numeric array of rank above 2, the two
    array formals become Any; the function hook then solves the shape
    and dtype and still reports a vector axis that is not 3.
    """
    try:
        sig = ctx.default_signature
        if not isinstance(sig, CallableType) or len(sig.arg_types) < 2:
            return ctx.default_signature
        if len(ctx.args) < 2 or len(ctx.args[0]) != 1 or len(ctx.args[1]) != 1:
            return ctx.default_signature
        ranks = []
        for exprs in ctx.args[:2]:
            parts = _ndarray_parts(ctx.api.get_expression_type(exprs[0]))
            if parts is None or _dtype_base(parts[1]) not in ("i", "u", "f", "c"):
                return ctx.default_signature
            ranks.append(len(parts[0]))
        if max(ranks) <= 2:
            return ctx.default_signature
        new_arg_types = list(sig.arg_types)
        new_arg_types[0] = AnyType(TypeOfAny.explicit)
        new_arg_types[1] = AnyType(TypeOfAny.explicit)
        return sig.copy_modified(arg_types=new_arg_types)
    except Exception:
        _swallowed(ctx, "_linalg_cross_signature_hook")
        return ctx.default_signature


def _reshape_signature_hook(ctx: MethodSigContext) -> Type:
    """Widen reshape formals to Any on provable int total mismatch (issue #148).

    The function hook reports `reshape changes total size` then mypy's
    fallback (empty_sequence overload with Sequence[Never]) reports a
    mechanical arg-type second error. When old and new are all ints with
    different totals (no -1), return Any formals so the fallback's arg
    check succeeds and only the single [numty-shape] error remains.
    All other calls keep the default signature.
    """
    try:
        sig = ctx.default_signature
        if not isinstance(sig, CallableType):
            return ctx.default_signature
        part = _ndarray_parts(ctx.type)
        if part is None:
            return ctx.default_signature
        dims_old, _ = part
        if not dims_old or not all(isinstance(d, int) for d in dims_old):
            return ctx.default_signature
        call = ctx.context
        if not isinstance(call, CallExpr):
            return ctx.default_signature
        pairs: list[tuple[Type, Expression | None]] = []
        for expr, kind in zip(call.args, call.arg_kinds):
            if kind != ARG_POS:
                continue
            try:
                tp = ctx.api.get_expression_type(expr)
            except Exception:
                _swallowed(ctx, "_reshape_signature_hook")
                return ctx.default_signature
            pairs.append((tp, expr))
        if not pairs:
            return ctx.default_signature
        raw = _reshape_new_dims(pairs)
        if raw is None:
            return ctx.default_signature
        if not raw or not all(isinstance(d, int) for d in raw):
            return ctx.default_signature
        if any(d == -1 for d in raw):  # type: ignore[comparison-overlap]
            return ctx.default_signature
        size_old = 1
        for d in dims_old:
            assert isinstance(d, int)
            size_old *= d
        size_new = 1
        for d in raw:
            assert isinstance(d, int)
            size_new *= d
        if size_new == size_old:
            return ctx.default_signature
        return sig.copy_modified(
            arg_types=[AnyType(TypeOfAny.explicit) for _ in sig.arg_types]
        )
    except Exception:
        _swallowed(ctx, "_reshape_signature_hook")
        return ctx.default_signature


def _broadcast_signature_hook(ctx: MethodSigContext) -> Type:
    """Widen other to Any on provable broadcast mismatch (issue #148).

    The method hook reports `broadcast shapes disagree` then mypy reports
    a mechanical [operator] duplicate. When both operands have known
    shapes that provably cannot broadcast (ints or N vs N-1 linear
    forms), return Any formals so the fallback's arg check succeeds and
    only the single [numty-broadcast] error remains. All other calls
    keep the default signature.
    """
    try:
        sig = ctx.default_signature
        if not isinstance(sig, CallableType):
            return ctx.default_signature
        call = ctx.context
        if isinstance(call, OpExpr) and call.op in ("//", "%"):
            # No complex loop for floor_divide/remainder: the method
            # hook reports it, so let the call through to reach it
            # (an inline complex operand otherwise matches the first
            # overload as `ndarray[Any, Any]` and nothing is said).
            for side in (call.left, call.right):
                side_part = _parts_or_unpack(ctx.api.get_expression_type(side))
                if side_part is not None and _dtype_base(side_part[1]) == "c":
                    return sig.copy_modified(
                        arg_types=[AnyType(TypeOfAny.explicit) for _ in sig.arg_types]
                    )
        part_a = _ndarray_parts(ctx.type)
        if part_a is None:
            return ctx.default_signature
        dims_a, _ = part_a
        if not dims_a:
            return ctx.default_signature
        # Other operand: OpExpr (a+b) has left/right; explicit
        # a.__add__(b) is a CallExpr with one arg.
        from mypy.nodes import CallExpr as _CallExpr
        from mypy.nodes import OpExpr as _OpExpr

        other_expr = None
        call = ctx.context
        if isinstance(call, _OpExpr):
            # a + b: left is self, right is other (or vice versa for __radd__?).
            # For __radd__, self is right operand (b in a+b where b.__radd__(a))?
            # To be safe, try both sides: if left matches self type, other is right, else vice versa.
            # Simpler: use right for __add__ etc., left for __radd__? Actually for `a+b`,
            # mypy tries both `a.__add__(b)` (self=a, other=b) and `b.__radd__(a)` (self=b, other=a).
            # In the first, ctx.type is a, other is b (right). In the second, ctx.type is b, other is a (left).
            # We don't know which hook invocation this is (fullname tells: __add__ vs __radd__).
            # For __add__ etc. (normal), other is right; for __radd__ etc. (reflected), other is left.
            # Detect via fullname? We don't have fullname here (hook doesn't receive it).
            # Instead, try right first: if right type has known dims and mismatches, use it;
            # otherwise try left. Simplest: check both, if either mismatches with self, trigger.
            try:
                right_tp = ctx.api.get_expression_type(call.right)
                right_part = _ndarray_parts(right_tp)
                if right_part is not None:
                    dims_b, _ = right_part
                    if dims_b:
                        _, msg = _broadcast_result(list(dims_a), list(dims_b), None)
                        if msg is not None:
                            return sig.copy_modified(
                                arg_types=[AnyType(TypeOfAny.explicit) for _ in sig.arg_types]
                            )
            except Exception:
                _swallowed(ctx, "_broadcast_signature_hook")
                pass
            try:
                left_tp = ctx.api.get_expression_type(call.left)
                left_part = _ndarray_parts(left_tp)
                if left_part is not None:
                    dims_b, _ = left_part
                    if dims_b:
                        _, msg = _broadcast_result(list(dims_a), list(dims_b), None)
                        if msg is not None:
                            return sig.copy_modified(
                                arg_types=[AnyType(TypeOfAny.explicit) for _ in sig.arg_types]
                            )
            except Exception:
                _swallowed(ctx, "_broadcast_signature_hook")
                pass
            return ctx.default_signature
        elif isinstance(call, _CallExpr) and call.args:
            # Explicit a.__add__(b): first arg is other.
            other_expr = call.args[0]
        else:
            return ctx.default_signature
        if other_expr is None:
            return ctx.default_signature
        try:
            other_tp = ctx.api.get_expression_type(other_expr)
        except Exception:
            _swallowed(ctx, "_broadcast_signature_hook")
            return ctx.default_signature
        part_b = _ndarray_parts(other_tp)
        if part_b is None:
            return ctx.default_signature
        dims_b, _ = part_b
        if not dims_b:
            return ctx.default_signature
        _, msg = _broadcast_result(list(dims_a), list(dims_b), None)
        if msg is None:
            return ctx.default_signature
        return sig.copy_modified(
            arg_types=[AnyType(TypeOfAny.explicit) for _ in sig.arg_types]
        )
    except Exception:
        _swallowed(ctx, "_broadcast_signature_hook")
        return ctx.default_signature


def _matmul_signature_hook(ctx: MethodSigContext) -> Type:
    """Widen formals to Any on provable matmul mismatch (issue #140).

    The method hook reports `matmul ... disagree` then mypy's overload
    fallback checks the actuals against an early (bool) overload and
    reports a mechanical [operator] duplicate. When the operands'
    true-order shapes provably clash (same conditions the method hook
    errors on: 0-d/scalar operands, inner or batch mismatch), return
    Any formals so the fallback's arg check succeeds and only the
    single [numty-shape] error remains. All other calls keep the
    default signature.
    """
    try:
        sig = ctx.default_signature
        if not isinstance(sig, CallableType):
            return ctx.default_signature
        from mypy.nodes import CallExpr as _CallExpr
        from mypy.nodes import OpExpr as _OpExpr

        call = ctx.context
        left_tp: Type | None = None
        right_tp: Type | None = None
        if isinstance(call, _OpExpr):
            # `a @ b`: true runtime order is (left, right) for both the
            # forward (__matmul__) and the reverse (__rmatmul__) check.
            # `a @= b` never arrives as an OpExpr.
            if call.op != "@":
                return ctx.default_signature
            try:
                left_tp = ctx.api.get_expression_type(call.left)
                right_tp = ctx.api.get_expression_type(call.right)
            except Exception:
                _swallowed(ctx, "_matmul_signature_hook")
                return ctx.default_signature
        elif isinstance(call, _CallExpr) and call.args:
            # Explicit `a.__matmul__(b)` / `a.__imatmul__(b)`: (self, arg).
            # Explicit `b.__rmatmul__(a)`: true order is (arg, self).
            callee = call.callee
            name = getattr(callee, "name", None)
            try:
                arg_tp = ctx.api.get_expression_type(call.args[0])
            except Exception:
                _swallowed(ctx, "_matmul_signature_hook")
                return ctx.default_signature
            if name == "__rmatmul__":
                left_tp, right_tp = arg_tp, ctx.type
            elif name in ("__matmul__", "__imatmul__"):
                left_tp, right_tp = ctx.type, arg_tp
            else:
                return ctx.default_signature
        else:
            return ctx.default_signature
        if left_tp is None or right_tp is None:
            return ctx.default_signature
        # 0-d arrays and scalars: the method hook rejects these outright.
        for tp in (left_tp, right_tp):
            parts = _ndarray_parts(tp)
            if parts is not None and not parts[0]:
                return sig.copy_modified(
                    arg_types=[AnyType(TypeOfAny.explicit) for _ in sig.arg_types]
                )
            if _scalar_key(tp) is not None:
                return sig.copy_modified(
                    arg_types=[AnyType(TypeOfAny.explicit) for _ in sig.arg_types]
                )
        parts_a, parts_b = _operand_parts(left_tp, right_tp)
        if parts_a is not None and parts_b is not None:
            (dims_a, _), (dims_b, _) = parts_a, parts_b
            _, message = _matmul_result(dims_a, dims_b, None)
            if message is None:
                return ctx.default_signature
        else:
            _, message, _, _ = _matmul_unpack_dims(left_tp, right_tp)
            if message is None:
                return ctx.default_signature
        return sig.copy_modified(
            arg_types=[AnyType(TypeOfAny.explicit) for _ in sig.arg_types]
        )
    except Exception:
        _swallowed(ctx, "_matmul_signature_hook")
        return ctx.default_signature


def _einsum_signature_hook(ctx: FunctionSigContext) -> Type:
    """Widen *operands to Any on provable int label clash (issue #148).

    The function hook reports the clash then mypy's fallback (bool
    overload) reports two mechanical arg-type follow-ons. When labels
    conflict on ints, return Any for *operands so the fallback's arg
    check succeeds and only the single [numty-shape] error remains.
    All other calls keep the default signature.
    """
    try:
        sig = ctx.default_signature
        if not isinstance(sig, CallableType):
            return ctx.default_signature
        call = ctx.context
        if not isinstance(call, CallExpr) or not call.args:
            return ctx.default_signature
        eq_expr = call.args[0]
        if not isinstance(eq_expr, StrExpr):
            return ctx.default_signature
        equation = "".join(eq_expr.value.split())
        if equation.count("->") != 1:
            return ctx.default_signature
        lhs, _ = equation.split("->")
        terms = lhs.split(",")
        operand_exprs = [a for a, k in zip(call.args[1:], call.arg_kinds[1:]) if k == ARG_POS]
        # *operands could also be passed as *args? Bail then (function hook bails too).
        if len(operand_exprs) != len(terms):
            return ctx.default_signature
        dims_list: list[list[int | Type]] = []
        for expr in operand_exprs:
            try:
                tp = ctx.api.get_expression_type(expr)
            except Exception:
                _swallowed(ctx, "_einsum_signature_hook")
                return ctx.default_signature
            part = _ndarray_parts(tp)
            if part is None:
                return ctx.default_signature
            dims, _ = part
            # Only all-int dims for the targeted suppression; symbolic
            # bails in the function hook (no error, no follow-on).
            if not all(isinstance(d, int) for d in dims):
                return ctx.default_signature
            dims_list.append(list(dims))  # type: ignore[arg-type]
        label_size: dict[str, int] = {}
        for dims, term in zip(dims_list, terms):
            parsed = _parse_einsum_term(term)
            if parsed is None:
                return ctx.default_signature
            pre, post, has_ell = parsed
            if has_ell:
                return ctx.default_signature
            if len(pre) != len(dims):
                return ctx.default_signature
            for label, size in zip(pre, dims):
                assert isinstance(size, int)
                if label not in label_size:
                    label_size[label] = size
                elif label_size[label] != size:
                    # Provable int clash: widen *operands.
                    for i, kind in enumerate(sig.arg_kinds):
                        if kind == ARG_STAR:
                            new_arg_types = list(sig.arg_types)
                            new_arg_types[i] = AnyType(TypeOfAny.explicit)
                            return sig.copy_modified(arg_types=new_arg_types)
                    return ctx.default_signature
        return ctx.default_signature
    except Exception:
        _swallowed(ctx, "_einsum_signature_hook")
        return ctx.default_signature


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
    if op == "det":
        return _linalg_det(ctx)
    if op == "slogdet":
        return _linalg_slogdet(ctx)
    if op == "qr":
        return _linalg_qr(ctx)
    if op == "matrix_power":
        return _linalg_matrix_power(ctx)
    if op == "cond":
        return _linalg_cond(ctx)
    if op == "tensorsolve":
        return _linalg_tensorsolve(ctx)
    return ctx.default_return_type


def _linalg_det(ctx: FunctionContext) -> Type:
    """linalg.det: square (..., N, N) in, scalar/batch out."""
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
            f"linalg.det requires a square (..., N, N) array, got shape "
            f"{_dims_str(dims)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    if not _check_dim_eq(
        ctx,
        dims[-1],
        dims[-2],
        f"linalg.det requires a square (..., N, N) array, got shape "
        f"{_dims_str(dims)}",
    ):
        return ctx.default_return_type
    if _reject_float16_linalg(ctx, dtype, "det"):
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
    return _maybe_scalar(ctx, dims[:-2], out_dtype)


def _linalg_slogdet(ctx: FunctionContext) -> Type:
    """linalg.slogdet: square in, (sign, logabsdet) tuple out."""
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
            f"linalg.slogdet requires a square (..., N, N) array, got shape "
            f"{_dims_str(dims)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    if not _check_dim_eq(
        ctx,
        dims[-1],
        dims[-2],
        f"linalg.slogdet requires a square (..., N, N) array, got shape "
        f"{_dims_str(dims)}",
    ):
        return ctx.default_return_type
    if _reject_float16_linalg(ctx, dtype, "slogdet"):
        return ctx.default_return_type
    ks = _dtype_kind_size(dtype)
    if ks is None:
        return ctx.default_return_type
    kind, size = ks
    if kind in ("i", "u", "b"):
        sign_dtype = _construct_dtype(ctx, "numpy.float64")
        log_dtype = _construct_dtype(ctx, "numpy.float64")
    elif kind == "f":
        sign_dtype = dtype
        log_dtype = dtype
    elif kind == "c":
        sign_dtype = dtype
        log_name = _linalg_real_name(kind, size)
        if log_name is None:
            return ctx.default_return_type
        log_dtype = _linalg_dtype(ctx, dtype, log_name)
    else:
        return ctx.default_return_type
    if sign_dtype is None or log_dtype is None:
        return ctx.default_return_type
    batch = dims[:-2]
    sign = _maybe_scalar(ctx, batch, sign_dtype)
    logabs = _maybe_scalar(ctx, batch, log_dtype)
    return _linalg_tuple(ctx, [sign, logabs])


def _linalg_qr(ctx: FunctionContext) -> Type:
    """linalg.qr: rank>=2 in, (Q, R) tuple (or R / raw tuple) out."""
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
            f"linalg.qr requires at least 2 dimensions, got shape {_dims_str(dims)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    mode_pair = _formal_pair(ctx, "mode")
    if mode_pair is None:
        mode = "reduced"
    else:
        mode = _string_value(mode_pair[0])
        if mode is None:
            return ctx.default_return_type
    if mode in ("full", "f"):
        mode = "reduced"
    elif mode in ("economic", "e"):
        mode = "r"
    if mode not in ("reduced", "complete", "r", "raw"):
        return ctx.default_return_type
    if _reject_float16_linalg(ctx, dtype, "qr"):
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
    batch = dims[:-2]
    rows, cols = dims[-2], dims[-1]
    if mode == "complete":
        return _linalg_tuple(
            ctx,
            [
                _ndarray_type(ctx, [*batch, rows, rows], out_dtype),
                _ndarray_type(ctx, [*batch, rows, cols], out_dtype),
            ],
        )
    if not isinstance(rows, int) or not isinstance(cols, int):
        return ctx.default_return_type
    inner = min(rows, cols)
    if mode == "reduced":
        return _linalg_tuple(
            ctx,
            [
                _ndarray_type(ctx, [*batch, rows, inner], out_dtype),
                _ndarray_type(ctx, [*batch, inner, cols], out_dtype),
            ],
        )
    if mode == "r":
        return _ndarray_type(ctx, [*batch, inner, cols], out_dtype)
    return _linalg_tuple(
        ctx,
        [
            _ndarray_type(ctx, [*batch, cols, rows], out_dtype),
            _ndarray_type(ctx, [*batch, inner], out_dtype),
        ],
    )


def _linalg_matrix_power(ctx: FunctionContext) -> Type:
    """linalg.matrix_power: square (..., N, N) in, same shape out."""
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
            f"linalg.matrix_power requires a square (..., N, N) array, got shape "
            f"{_dims_str(dims)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    if not _check_dim_eq(
        ctx,
        dims[-1],
        dims[-2],
        f"linalg.matrix_power requires a square (..., N, N) array, got shape "
        f"{_dims_str(dims)}",
    ):
        return ctx.default_return_type
    ks = _dtype_kind_size(dtype)
    if ks is None:
        return ctx.default_return_type
    kind, _size = ks
    out_dtype: Type | None = dtype
    if kind in ("i", "u", "b"):
        n_pair = _formal_pair(ctx, "n")
        if n_pair is None:
            return ctx.default_return_type
        n_value = _fold_single(n_pair)
        if n_value is None:
            return ctx.default_return_type
        if n_value < 0:
            out_dtype = _construct_dtype(ctx, "numpy.float64")
    if out_dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, out_dtype)


def _linalg_cond(ctx: FunctionContext) -> Type:
    """linalg.cond: rank>=2 in, scalar/batch real out (not square)."""
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
            f"linalg.cond requires at least 2 dimensions, got shape {_dims_str(dims)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    if _reject_float16_linalg(ctx, dtype, "cond"):
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
    return _maybe_scalar(ctx, dims[:-2], out_dtype)


def _linalg_tensorsolve(ctx: FunctionContext) -> Type:
    """linalg.tensorsolve: prod(a[b.ndim:]) == prod(a[:b.ndim]), shape Q out."""
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    part_a = _ndarray_parts(pairs[0][0])
    part_b = _ndarray_parts(pairs[1][0])
    if part_a is None or part_b is None:
        return ctx.default_return_type
    dims_a, dtype_a = part_a
    dims_b, dtype_b = part_b
    if _has_unpack(dims_a) or _has_unpack(dims_b):
        return ctx.default_return_type
    axes_pair = _formal_pair(ctx, "axes")
    if axes_pair is not None and not _is_none_type(axes_pair[0]):
        return ctx.default_return_type
    an, bn = len(dims_a), len(dims_b)
    if bn == 0:
        oldshape: list[int | Type] = list(dims_a)
    elif an - bn == 0:
        oldshape = list(dims_a)
    else:
        oldshape = list(dims_a[-(an - bn) :])
    if any(not isinstance(d, int) for d in dims_a) or any(
        not isinstance(d, int) for d in dims_b
    ):
        return ctx.default_return_type
    prod = 1
    for d in oldshape:
        assert isinstance(d, int)
        prod *= d
    size_a = 1
    for d in dims_a:
        assert isinstance(d, int)
        size_a *= d
    if size_a != prod * prod:
        ctx.api.fail(
            f"linalg.tensorsolve requires prod(a.shape[b.ndim:]) == "
            f"prod(a.shape[:b.ndim]) (shapes {_dims_str(dims_a)} vs "
            f"{_dims_str(dims_b)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    prod_b = 1
    for d in dims_b:
        assert isinstance(d, int)
        prod_b *= d
    if prod_b != prod:
        ctx.api.fail(
            f"linalg.tensorsolve inner dimensions disagree: {prod_b} vs "
            f"{prod} (shapes {_dims_str(dims_a)} vs {_dims_str(dims_b)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    dtype = _linalg_solve_dtype(ctx, dtype_a, dtype_b)
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, oldshape, dtype)


def _cross_int_value(
    ctx: FunctionContext, name: str, default: int
) -> int | None:
    """Folded int for a cross axis formal (default when absent), None to bail."""
    pair = _formal_pair(ctx, name)
    if pair is None:
        return default
    value = _fold_single(pair)
    return value


def cross_function_hook(ctx: FunctionContext) -> Type:
    """np.cross: broadcast (arrays of) 3-vectors, 3-vector size checked."""
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    part_a = _ndarray_parts(pairs[0][0])
    part_b = _ndarray_parts(pairs[1][0])
    if part_a is None or part_b is None:
        return ctx.default_return_type
    dims_a, dtype_a = part_a
    dims_b, dtype_b = part_b
    if _has_unpack(dims_a) or _has_unpack(dims_b):
        return ctx.default_return_type
    if not dims_a or not dims_b:
        ctx.api.fail(
            f"cross requires at least 1 dimension, got shapes {_dims_str(dims_a)} "
            f"vs {_dims_str(dims_b)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    axis_pair = _formal_pair(ctx, "axis")
    if axis_pair is not None and not _is_none_type(axis_pair[0]):
        axis_value = _fold_single(axis_pair)
        if axis_value is None:
            return ctx.default_return_type
        axisa = axisb = axisc = axis_value
    else:
        axisa = _cross_int_value(ctx, "axisa", -1)
        axisb = _cross_int_value(ctx, "axisb", -1)
        axisc = _cross_int_value(ctx, "axisc", -1)
        if axisa is None or axisb is None or axisc is None:
            return ctx.default_return_type
    assert isinstance(axisa, int) and isinstance(axisb, int) and isinstance(axisc, int)
    norm_a = _normalize_axis(axisa, len(dims_a))
    if norm_a is None:
        ctx.api.fail(
            f"cross axisa {axisa} is out of range for {len(dims_a)} dimensions "
            f"(shape {_dims_str(dims_a)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    norm_b = _normalize_axis(axisb, len(dims_b))
    if norm_b is None:
        ctx.api.fail(
            f"cross axisb {axisb} is out of range for {len(dims_b)} dimensions "
            f"(shape {_dims_str(dims_b)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    vec_a, vec_b = dims_a[norm_a], dims_b[norm_b]
    if not isinstance(vec_a, int) or not isinstance(vec_b, int):
        return ctx.default_return_type
    if vec_a != 3 or vec_b != 3:
        ctx.api.fail(
            f"cross requires 3-dimensional vectors, got {vec_a} and {vec_b} "
            f"(shapes {_dims_str(dims_a)} vs {_dims_str(dims_b)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    wo_a = [d for i, d in enumerate(dims_a) if i != norm_a]
    wo_b = [d for i, d in enumerate(dims_b) if i != norm_b]
    bcast, cause = _broadcast_result(wo_a, wo_b, ctx)
    if cause is not None:
        ctx.api.fail(f"cross {cause}", ctx.context, code=NUMTY_BROADCAST)
        return ctx.default_return_type
    assert bcast is not None
    out = [*bcast, 3]
    norm_c = _normalize_axis(axisc, len(out))
    if norm_c is None:
        ctx.api.fail(
            f"cross axisc {axisc} is out of range for {len(out)} dimensions "
            f"(shapes {_dims_str(dims_a)} vs {_dims_str(dims_b)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out.insert(norm_c, out.pop())
    dtype = _promote_all(ctx, [dtype_a, dtype_b])
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, out, dtype)


def linalg_cross_function_hook(ctx: FunctionContext) -> Type:
    """np.linalg.cross: single-axis variant of np.cross."""
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    part_a = _ndarray_parts(pairs[0][0])
    part_b = _ndarray_parts(pairs[1][0])
    if part_a is None or part_b is None:
        return ctx.default_return_type
    dims_a, dtype_a = part_a
    dims_b, dtype_b = part_b
    if _has_unpack(dims_a) or _has_unpack(dims_b):
        return ctx.default_return_type
    if not dims_a or not dims_b:
        ctx.api.fail(
            f"linalg.cross requires at least 1 dimension, got shapes "
            f"{_dims_str(dims_a)} vs {_dims_str(dims_b)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    axis_value = _cross_int_value(ctx, "axis", -1)
    if axis_value is None:
        return ctx.default_return_type
    norm_a = _normalize_axis(axis_value, len(dims_a))
    if norm_a is None:
        ctx.api.fail(
            f"linalg.cross axis {axis_value} is out of range for {len(dims_a)} "
            f"dimensions (shape {_dims_str(dims_a)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    norm_b = _normalize_axis(axis_value, len(dims_b))
    if norm_b is None:
        ctx.api.fail(
            f"linalg.cross axis {axis_value} is out of range for {len(dims_b)} "
            f"dimensions (shape {_dims_str(dims_b)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    vec_a, vec_b = dims_a[norm_a], dims_b[norm_b]
    # A symbolic vector axis is taken to be 3: the call fails otherwise,
    # and the stubs (written for 1-D and 2-D) would answer int8 here.
    if any(isinstance(v, int) and v != 3 for v in (vec_a, vec_b)):
        ctx.api.fail(
            f"linalg.cross requires 3-dimensional vectors, got "
            f"{_dim_str(vec_a)} and {_dim_str(vec_b)} "
            f"(shapes {_dims_str(dims_a)} vs {_dims_str(dims_b)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    wo_a = [d for i, d in enumerate(dims_a) if i != norm_a]
    wo_b = [d for i, d in enumerate(dims_b) if i != norm_b]
    bcast, cause = _broadcast_result(wo_a, wo_b, ctx)
    if cause is not None:
        ctx.api.fail(f"linalg.cross {cause}", ctx.context, code=NUMTY_BROADCAST)
        return ctx.default_return_type
    assert bcast is not None
    out = [*bcast, 3]
    norm_c = _normalize_axis(axis_value, len(out))
    if norm_c is None:
        ctx.api.fail(
            f"linalg.cross axis {axis_value} is out of range for {len(out)} "
            f"dimensions (shapes {_dims_str(dims_a)} vs {_dims_str(dims_b)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out.insert(norm_c, out.pop())
    dtype = _promote_all(ctx, [dtype_a, dtype_b])
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, out, dtype)


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
    if key is not None:
        return (None, key[0], key[1])
    # Numpy scalars (np.int64(3), x.max()) are strong (NEP 50).
    name = _scalar_ret_name(tp)
    if name is not None:
        return (None, "strong", name)
    return None


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
    # result_type promotes the arrays (and numpy scalars) among
    # themselves first and only then lets Python scalars in: bool with
    # 0 and a uint8 array is uint8, not the int64 that bool-with-0
    # alone gives.
    specs = sorted(specs, key=lambda spec: spec[0] == "weak")
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
    pairs = _all_positionals_expanded(ctx)
    if pairs is None:
        return ctx.default_return_type
    if not pairs:
        return ctx.default_return_type
    # Keyword bounds not already covered positionally still broadcast.
    # NumPy >= 2.1 also spells them min=/max=.
    for name, position in (("a_min", 2), ("a_max", 3), ("min", 2), ("max", 3)):
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
    # NumPy 2 promotes across all operands (like a ufunc).
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
    branches = [(kind, name) for _, kind, name in specs]
    array_dtypes = []
    for tp in types:
        part = _parts_or_unpack(tp)
        if part is not None:
            array_dtypes.append(part[1])
    dtype = _specs_dtype(ctx, branches, array_dtypes)
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


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


def _broadcast_to_compatible(
    ctx: FunctionContext,
    arr_dims: list[int | Type],
    dims: list[int | Type],
) -> bool:
    """Whether an array shape broadcasts to the target (issue #46).

    Right-aligns the two shapes: extra leading target dims are fine, a
    source rank larger than the target rank fails, and each aligned pair
    must be equal or have a 1 on the source side. Provable mismatches
    fail loudly; anything involving a symbolic dim that could be 1 bails
    (returns False) instead of guessing.
    """
    # A scalar target takes only a 0-d source: numpy refuses even (1,)
    # ("cannot broadcast a non-scalar to a scalar array").
    if not dims and arr_dims:
        ctx.api.fail(
            f"broadcast_to of shape {_dims_str(arr_dims)} cannot "
            f"broadcast to scalar shape {_dims_str(dims)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return False
    if len(arr_dims) > len(dims):
        ctx.api.fail(
            f"broadcast_to of shape {_dims_str(arr_dims)} cannot broadcast "
            f"to fewer dimensions {_dims_str(dims)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return False
    offset = len(dims) - len(arr_dims)
    for k, (s, t) in enumerate(zip(arr_dims, dims[offset:])):
        if _dim_eq(s, t):
            continue
        if isinstance(s, int) and s == 1:
            continue
        if isinstance(s, int) and isinstance(t, int):
            ctx.api.fail(
                f"broadcast_to of shape {_dims_str(arr_dims)} is incompatible "
                f"with target shape {_dims_str(dims)}: {s} vs {t} at "
                f"trailing axis {len(arr_dims) - 1 - k}",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return False
        return False
    return True


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
        if not _broadcast_to_compatible(ctx, arr_dims, dims):
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


def _apply_ndmin(
    ctx: FunctionContext | MethodContext, dims: list[int | Type]
) -> list[int | Type] | None:
    """Prepend 1s until the rank reaches `ndmin` (issue #46), else None.

    No `ndmin` argument leaves dims untouched. An unprovable `ndmin`
    (or one over a `*Ts` shape) bails instead of guessing; a negative
    one fails loudly.
    """
    pair = _formal_pair(ctx, "ndmin")
    if pair is None:
        return dims
    if _has_unpack(dims):
        return None
    value = _fold_single(pair)
    if value is None:
        return None
    if value < 0:
        ctx.api.fail(
            f"ndmin must be non-negative, got {value}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return None
    if len(dims) >= value:
        return dims
    return [1] * (value - len(dims)) + list(dims)


def _asarray_array_type(
    ctx: FunctionContext | MethodContext, tp: Type, named: Type | None
) -> Type:
    """asarray over one array operand: same shape, optional new dtype."""
    part = _parts_or_unpack(tp)
    if part is None:
        return ctx.default_return_type
    dims = _apply_ndmin(ctx, part[0])
    if dims is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, named if named is not None else part[1])


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
        dims = _apply_ndmin(ctx, part[0])
        if dims is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, named if named is not None else part[1])
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
    dims = _apply_ndmin(ctx, [len(elems)])
    if dims is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def _solve_ufunc(ctx: FunctionContext | MethodContext, group: str) -> Type:
    pairs = _all_positionals_expanded(ctx)
    if pairs is None:
        return ctx.default_return_type
    if len(pairs) != 1:
        return ctx.default_return_type
    if _formal_pair(ctx, "where") is not None:
        return ctx.default_return_type
    dist = _distribute_solve([pairs[0][0]], lambda t: _ufunc_types(ctx, group, t))
    if dist is not None:
        return _apply_out_dtype(ctx, dist)
    return _apply_out_dtype(ctx, _ufunc_types(ctx, group, pairs[0][0]))


def _smallest_float(name: str) -> str | None:
    """Smallest float holding an int/uint/bool dtype (issue #61).

    `result_type(name, float16)`: 1-byte -> float16, 2-byte -> float32,
    larger -> float64. Floats/complex pass through.
    """
    if name.startswith("numpy.float") or name.startswith("numpy.complex"):
        return name
    return _promote_strong(name, "numpy.float16")


def _ufunc_types(ctx: FunctionContext | MethodContext, group: str, tp: Type) -> Type:
    part = _parts_or_unpack(tp)
    if part is None:
        # Scalar input (np.sqrt(4.0)): map to the matching numpy scalar
        # instead of the stub's Any (issue #61).
        if group in ("float", "abs", "real"):
            key = _scalar_key(tp)
            if key is None:
                name = _scalar_ret_name(tp)
                if name is None:
                    return ctx.default_return_type
                key = ("strong", name)
            kind, sname = key
            if kind == "weak":
                # Python scalars: float -> float64, int/bool -> smallest float.
                if sname == "builtins.float":
                    out = "numpy.float64"
                elif sname in ("builtins.int", "builtins.bool"):
                    out = _smallest_float("numpy.int64")
                    if out is None:
                        return ctx.default_return_type
                elif sname == "builtins.complex":
                    out = "numpy.complex128"
                else:
                    return ctx.default_return_type
            else:
                if _kind_of(sname) in ("f", "c"):
                    out = sname
                else:
                    out = _smallest_float(sname)
                    if out is None:
                        return ctx.default_return_type
            dtype = _construct_dtype(ctx, out)
            if dtype is None:
                return ctx.default_return_type
            scalar = _scalar_of_dtype(dtype)
            return scalar if scalar is not None else ctx.default_return_type
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
        fullname = _dtype_fullname(dtype)
        if fullname is None:
            return ctx.default_return_type
        name = _smallest_float(fullname)
    if name is None:
        return ctx.default_return_type
    out_dtype = _linalg_dtype(ctx, dtype, name)
    if out_dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, out_dtype)
    if name is None:
        return ctx.default_return_type
    out_dtype = _linalg_dtype(ctx, dtype, name)
    if out_dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, out_dtype)


def ufunc_function_hook(ctx: FunctionContext, group: str) -> Type:
    return _solve_ufunc(ctx, group)


def _solve_round(
    ctx: FunctionContext | MethodContext, array: Type, function: bool
) -> Type:
    """round: the input's shape and dtype, except on a bool array.

    Every numeric dtype is kept. A bool array comes back float16 from
    `round()` and `round(0)`; with any other `decimals` numpy refuses
    (it cannot cast the scaled values back to bool), so nothing is
    claimed there. numty has no error code for a dtype failure.
    """
    part = _parts_or_unpack(array)
    if part is None:
        return ctx.default_return_type
    out = _formal_pair(ctx, "out")
    if out is not None and not _is_none_type(out[0]):
        return ctx.default_return_type
    dims, dtype = part
    base = _dtype_base(dtype)
    if base != "b":
        # The method's stub already returns Self; np.round's loses the shape.
        if function and base is not None:
            return _ndarray_type(ctx, dims, dtype)
        return ctx.default_return_type
    pair = _formal_pair(ctx, "decimals")
    if pair is not None:
        decimals = _fold_single(pair)
        if decimals is None:
            any_dtype = ctx.api.named_generic_type(
                DTYPE_FULLNAME, [AnyType(TypeOfAny.special_form)]
            )
            return _ndarray_type(ctx, dims, any_dtype)
        if decimals != 0:
            return AnyType(TypeOfAny.special_form)
    f16 = _construct_dtype(ctx, "numpy.float16")
    if f16 is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, f16)


def round_method_hook(ctx: MethodContext) -> Type:
    return _solve_round(ctx, ctx.type, function=False)


def round_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _solve_round(ctx, pairs[0][0], function=True)


def angle_function_hook(ctx: FunctionContext) -> Type:
    """np.angle: the input's shape, in the matching real float dtype.

    complex64 gives float32 and complex128 float64; a float keeps its
    dtype. Integer and bool inputs follow a table of their own (int8
    is float16, bool is float64), so those stay with the stubs.
    """
    pairs = _all_positionals(ctx)
    if len(pairs) != 1:
        return ctx.default_return_type
    part = _parts_or_unpack(pairs[0][0])
    if part is None or _dtype_base(part[1]) not in ("f", "c"):
        return ctx.default_return_type
    return _ufunc_types(ctx, "abs", pairs[0][0])


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
    # Division with no float/complex operand yields the default float;
    # weak Python ints count like strong ones (np.divide(7, 2) -> float64).
    # With a float present, normal promotion applies (f32/u8 -> f32).
    if not any(_kind_of(name) in ("f", "c") for name in names):
        if any(_kind_of(name) in ("i", "u", "b") for name in names):
            dtype = _construct_dtype(ctx, "numpy.float64")
            if dtype is None:
                return ctx.default_return_type
            return _maybe_scalar(ctx, dims, dtype)
    dtype = _specs_dtype(ctx, [(kind_l, name_l), (kind_r, name_r)], array_dtypes)
    if dtype is None:
        return ctx.default_return_type
    return _maybe_scalar(ctx, dims, dtype)


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


def _has_expr_var(tp: Type, seen: set[int] | None = None) -> bool:
    """Whether a type mentions a numty dim-expression var (issue #128)."""
    if seen is None:
        seen = set()
    if id(tp) in seen:
        return False
    seen.add(id(tp))
    proper = get_proper_type(tp)
    if isinstance(proper, TypeVarType):
        return proper.fullname.startswith(_EXPR_PREFIX)
    if isinstance(proper, TupleType):
        return any(_has_expr_var(item, seen) for item in proper.items)
    if isinstance(proper, Instance):
        return any(_has_expr_var(arg, seen) for arg in proper.args)
    if isinstance(proper, UnionType):
        return any(_has_expr_var(item, seen) for item in proper.relevant_items())
    return False


def _subst_expr_dim(
    ctx: AttributeContext, dim: int | Type, mapping: dict[str, Type]
) -> int | Type | None:
    """An attribute dim with class args substituted (issue #128).

    Expression vars (`N+2` stored on `self`) are rewritten from their
    stored form; `N := 4` turns `N+2` into `6`. Returns None when the
    result cannot be spelled (caller widens to a fresh unknown).
    """
    if isinstance(dim, int):
        return dim
    if not isinstance(dim, TypeVarType):
        return dim
    if not dim.fullname.startswith(_EXPR_PREFIX):
        return dim
    form = _form_of(dim)
    if not isinstance(form, _LinForm):
        return None
    const = form.const
    coeffs: list[tuple[str, int]] = []
    for name, coeff in form.coeffs:
        if _is_unknown_fullname(name):
            coeffs.append((name, coeff))
            continue
        actual = mapping.get(name)
        if actual is None:
            return None
        value = _dim_value(actual)
        if isinstance(value, int):
            const += coeff * value
        elif isinstance(value, TypeVarType):
            coeffs.append((value.fullname, coeff))
        else:
            return None
    coeffs = [(n, c) for n, c in coeffs if c != 0]
    if not coeffs:
        return const
    if len(coeffs) == 1 and coeffs[0][1] == 1 and const == 0:
        # A single substituted var: return the real object, not a lookalike.
        for tp in mapping.values():
            proper = get_proper_type(tp)
            if isinstance(proper, TypeVarType) and proper.fullname == coeffs[0][0]:
                return proper
    return _expr_var(ctx, "attr", 0, _LinForm(tuple(sorted(coeffs)), const))


def _subst_attr_type(
    ctx: AttributeContext, tp: Type, mapping: dict[str, Type]
) -> Type | None:
    """Attribute type with class args substituted; None to widen instead."""
    proper = get_proper_type(tp)
    if isinstance(proper, TypeVarType):
        if not proper.fullname.startswith(_EXPR_PREFIX):
            return proper
        dim = _subst_expr_dim(ctx, proper, mapping)
        if dim is None:
            return None
        if isinstance(dim, int):
            int_instance = ctx.api.named_generic_type(INT_FULLNAME, [])
            return LiteralType(dim, fallback=int_instance)
        return dim
    if isinstance(proper, TupleType):
        items: list[Type] = []
        for item in proper.items:
            if isinstance(item, UnpackType):
                return None
            sub = _subst_attr_type(ctx, item, mapping)
            if sub is None:
                return None
            items.append(sub)
        return proper.copy_modified(items=items)
    if isinstance(proper, Instance):
        args: list[Type] = []
        for arg in proper.args:
            if isinstance(arg, UnpackType):
                return None
            sub = _subst_attr_type(ctx, arg, mapping)
            if sub is None:
                return None
            args.append(sub)
        if args == list(proper.args):
            return proper
        return proper.copy_modified(args=args)
    if isinstance(proper, UnionType):
        items = []
        for item in proper.relevant_items():
            sub = _subst_attr_type(ctx, item, mapping)
            if sub is None:
                return None
            items.append(sub)
        return proper.copy_modified(items=items)
    return proper


def substituted_attribute_hook(ctx: AttributeContext) -> Type:
    """Generic attribute access: substitute class args into expr dims.

    Expression vars (`N+2`) mint a fresh id mypy never substitutes, so
    `Padded[Literal[4]].p` leaks an unbound `N+2` (issue #128). Rewrite
    from the stored form with the base's type arguments; widen to a
    fresh unknown when the result cannot be spelled.
    """
    default = ctx.default_attr_type
    if not _has_expr_var(default):
        return default
    base = get_proper_type(ctx.type)
    if not isinstance(base, Instance):
        return default
    try:
        tvars = base.type.defn.type_vars
    except Exception:
        _swallowed(ctx, "substituted_attribute_hook")
        return default
    if len(tvars) != len(base.args):
        return default
    mapping: dict[str, Type] = {}
    for tvar, arg in zip(tvars, base.args):
        fullname = getattr(tvar, "fullname", None)
        if isinstance(fullname, str):
            mapping[fullname] = arg
    try:
        sub = _subst_attr_type(ctx, default, mapping)
    except Exception:
        _swallowed(ctx, "substituted_attribute_hook")
        return default
    if sub is None:
        return default
    return sub


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
    items: list[Type] = []
    for v in dims:
        if isinstance(v, int):
            # mypy's widening form: a plain int carrying its literal
            # value. Fresh locals widen to int (so `k = x.shape[0]; k -= 1`
            # checks), while _int_value/_dim_value still read the value
            # back for direct uses like reshape(x.shape[0], 3) (issue #62).
            items.append(
                int_fallback.copy_modified(
                    last_known_value=LiteralType(v, fallback=int_fallback)
                )
            )
        else:
            items.append(v)
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
    """The .T view: full reversal, concrete or mixed shapes only.

    Unpack shapes bail: reversing would also reverse *Ts itself, which
    no annotation can spell (issue #48). Anything else stays on the stubs.
    """
    parts = _movable_parts(tp)
    if parts is not None:
        dims, dtype = parts
        return _ndarray_type(ctx, list(reversed(dims)), dtype)
    return ctx.default_attr_type


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
        if first.fullname.startswith("__numty_unknown__."):
            # Unknown (data-dependent) lengths are plain ints: nothing
            # nameable is lost, and `k = len(masked); k = max(k, 1)`
            # checks (issue #62). Named dims stay exact.
            return int_fallback
        return first
    if not isinstance(first, int):
        return ctx.default_return_type
    # Same widening form as .shape items (issue #62): fresh locals get
    # plain int (`k = len(x); k -= 1`), direct uses still read the value.
    return int_fallback.copy_modified(
        last_known_value=LiteralType(first, fallback=int_fallback)
    )


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
            # Along the axis the result is the index's size, even when
            # that size is 1 (issue #46: the argmax(keepdims=True) ->
            # take_along_axis idiom); symbolic index dims carry through.
            out.append(y)
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
    pairs = _all_positionals_expanded(ctx)
    if pairs is None:
        return ctx.default_return_type
    # Method form also accepts named bounds: a.clip(min=w, max=v).
    # One entry per slot; aliases resolve to the first spelling present.
    for aliases in (("a_min", "min"), ("a_max", "max")):
        for name in aliases:
            named = _formal_pair(ctx, name)
            if named is not None and not _is_none_type(named[0]):
                pairs.append(named)
                break
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
    return _apply_out_dtype(ctx, _ufunc_call_solve(ctx))


def _ufunc_call_solve(ctx: MethodContext) -> Type:
    proper = get_proper_type(ctx.type)
    if isinstance(proper, Instance) and "Nout2" in proper.type.fullname:
        # Two-output ufuncs (frexp, modf): the stubs own the tuple.
        return ctx.default_return_type
    name = _ufunc_name(ctx)
    if name is None:
        return ctx.default_return_type
    if name in _UFUNC_NO_COMPLEX:
        # No complex loop: numpy raises TypeError, so no shape to claim.
        operands = _all_positionals_expanded(ctx)
        for tp, _expr in operands or []:
            part = _parts_or_unpack(tp)
            if part is not None and _dtype_base(part[1]) == "c":
                ctx.api.fail(
                    f"ufunc '{name}' not supported for complex inputs",
                    ctx.context,
                    code=NUMTY_BROADCAST,
                )
                return ctx.default_return_type
    if name in _UFUNC_KEEP:
        # Per-loop tables (issue #61): bool square/reciprocal/conj ->
        # int8; bool sign/negative and float/complex invert have no loop.
        pairs = _all_positionals_expanded(ctx)
        if pairs is None:
            return ctx.default_return_type
        if len(pairs) == 1:
            tp = pairs[0][0]
            part = _parts_or_unpack(tp)
            if part is not None:
                base = _dtype_base(part[1])
                if name in ("square", "reciprocal", "conj", "conjugate") and base == "b":
                    dims = _broadcast_dims(ctx, tp, tp)
                    if dims is None:
                        return ctx.default_return_type
                    dtype = _construct_dtype(ctx, "numpy.int8")
                    if dtype is None:
                        return ctx.default_return_type
                    return _ndarray_type(ctx, dims, dtype)
                if name in ("sign", "negative") and base == "b":
                    ctx.api.fail(
                        f"ufunc '{name}' has no loop for bool inputs",
                        ctx.context,
                        code=NUMTY_BROADCAST,
                    )
                    return ctx.default_return_type
                if name in ("invert", "bitwise_not") and base in ("f", "c"):
                    ctx.api.fail(
                        f"ufunc '{name}' not supported for float inputs",
                        ctx.context,
                        code=NUMTY_BROADCAST,
                    )
                    return ctx.default_return_type
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
        pairs = _all_positionals_expanded(ctx)
        if pairs is None:
            return ctx.default_return_type
        if len(pairs) != 2:
            return ctx.default_return_type
        if not _matmul_has_dims(ctx, [pairs[0][0], pairs[1][0]]):
            return ctx.default_return_type
        return _solve_matmul(ctx, pairs[0][0], pairs[1][0])
    if name in {"divide", "true_divide"}:
        pairs = _all_positionals_expanded(ctx)
        if pairs is None:
            return ctx.default_return_type
        if len(pairs) != 2:
            return ctx.default_return_type
        return _solve_truediv_pair(ctx, pairs[0][0], pairs[1][0])
    if name in _UFUNC_BINARY_PROMOTE:
        pairs = _all_positionals_expanded(ctx)
        if pairs is None:
            return ctx.default_return_type
        if len(pairs) != 2:
            return ctx.default_return_type
        if name == "subtract" and _is_bool_operand(pairs[0][0]) and _is_bool_operand(pairs[1][0]):
            ctx.api.fail(
                "numpy boolean subtract is not supported, use bitwise_xor instead",
                ctx.context,
                code=NUMTY_BROADCAST,
            )
            return ctx.default_return_type
        if (
            name in ("floor_divide", "remainder", "fmod", "power")
            and _is_bool_operand(pairs[0][0])
            and _is_bool_operand(pairs[1][0])
        ):
            dims = _broadcast_dims(ctx, pairs[0][0], pairs[1][0])
            if dims is None:
                return ctx.default_return_type
            dtype = _construct_dtype(ctx, "numpy.int8")
            if dtype is None:
                return ctx.default_return_type
            return _ndarray_type(ctx, dims, dtype)
        return _solve_broadcast(ctx, pairs[0][0], pairs[1][0], name)
    if name in _UFUNC_BINARY_FLOAT:
        pairs = _all_positionals_expanded(ctx)
        if pairs is None:
            return ctx.default_return_type
        if len(pairs) != 2:
            return ctx.default_return_type
        return _solve_ufunc_binary_float(ctx, pairs[0][0], pairs[1][0])
    if name in _UFUNC_BINARY_BOOL:
        pairs = _all_positionals_expanded(ctx)
        if pairs is None:
            return ctx.default_return_type
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
        out = _smallest_float(combined)
        if kind_l == "strong" and kind_r == "strong":
            # numpy picks the first float loop both inputs cast to
            # safely, so each operand is lifted on its own: int8 with
            # uint8 is float16, although their common integer (int16)
            # would need float32.
            fl, fr = _smallest_float(name_l), _smallest_float(name_r)
            out = _promote_strong(fl, fr) if fl and fr and fl != fr else fl
        if out is None:
            return ctx.default_return_type
        out_name = out
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
    return _maybe_scalar(ctx, dims, dtype)


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
    return _maybe_scalar(ctx, dims, bool_dtype)


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
            "round": round_function_hook,
            "around": round_function_hook,
            "all": partial(allany_function_hook, op="all"),
            "any": partial(allany_function_hook, op="any"),
            "clip": clip_function_hook,
            "cumprod": partial(cumsum_function_hook, op="cumprod"),
            "cumsum": partial(cumsum_function_hook, op="cumsum"),
            "max": partial(reduction_function_hook, op="max"),
            "mean": partial(reduction_function_hook, op="mean"),
            "min": partial(reduction_function_hook, op="min"),
            "nonzero": nonzero_function_hook,
            "prod": partial(reduction_function_hook, op="prod"),
            "ravel": ravel_function_hook,
            "repeat": repeat_function_hook,
            "reshape": reshape_function_hook,
            "searchsorted": searchsorted_function_hook,
            "squeeze": squeeze_function_hook,
            "std": partial(reduction_function_hook, op="std"),
            "sum": partial(reduction_function_hook, op="sum"),
            "take": take_function_hook,
            "transpose": transpose_function_hook,
            "swapaxes": swapaxes_function_hook,
            "matrix_transpose": matrix_transpose_function_hook,
            "var": partial(reduction_function_hook, op="var"),
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
            "cross": cross_function_hook,
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
            "irfft": irfft_function_hook,
            "rfft2": rfft2_function_hook,
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
            "angle": angle_function_hook,
            "average": average_function_hook,
            "copy": copy_function_hook,
            "diff": diff_function_hook,
            "flip": flip_function_hook,
            "gradient": gradient_function_hook,
            "median": partial(reduction_function_hook, op="median"),
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
            "nancumprod": partial(cumsum_function_hook, op="nancumprod"),
            "nancumsum": partial(cumsum_function_hook, op="nancumsum"),
            "nanmax": partial(reduction_function_hook, op="nanmax"),
            "nanmean": partial(reduction_function_hook, op="nanmean"),
            "nanmedian": partial(reduction_function_hook, op="nanmedian"),
            "nanmin": partial(reduction_function_hook, op="nanmin"),
            "nanpercentile": percentile_function_hook,
            "nanprod": partial(reduction_function_hook, op="nanprod"),
            "nanquantile": percentile_function_hook,
            "nanstd": partial(reduction_function_hook, op="nanstd"),
            "nansum": partial(reduction_function_hook, op="nansum"),
            "nanvar": partial(reduction_function_hook, op="nanvar"),
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
            "cross": linalg_cross_function_hook,
            **{
                op: partial(linalg_function_hook, op=op)
                for op in (
                    "cholesky",
                    "cond",
                    "det",
                    "eig",
                    "eigh",
                    "inv",
                    "lstsq",
                    "matrix_power",
                    "norm",
                    "pinv",
                    "qr",
                    "slogdet",
                    "solve",
                    "svd",
                    "tensorsolve",
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
