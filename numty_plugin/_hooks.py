"""_hooks.py — sole dispatch owner (Plugin + get_* + _method_hook +
_numpy_function_hooks). (Phase 5, #199). Top-level imports go to earlier
layers only (order in tests/test_layout.py, _LAYERS). A function-local
facade import is used only for a monkeypatch seam (_LAZY_SEAMS).
"""
from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

from mypy.nodes import (
    ARG_POS, CallExpr, Decorator, Expression, FuncDef, OpExpr, OverloadedFuncDef, TypeInfo,
    TypeVarLikeExpr,
)
from mypy.plugin import (
    AnalyzeTypeContext, AttributeContext, FunctionContext, FunctionSigContext, MethodContext,
    MethodSigContext, Plugin,
)
from mypy.types import (
    AnyType, CallableType, FunctionLike, Instance, LiteralType, Overloaded, TupleType, Type,
    TypeOfAny, TypeVarLikeType, TypeVarType, UnionType, UnpackType, get_proper_type,
)

if TYPE_CHECKING:
    from mypy.tvar_scope import TypeVarLikeScope
    from mypy.typeanal import TypeAnalyser
    from mypy.types import RawExpressionType, UnboundType

from ._errors import NUMTY_ANNOTATION, NUMTY_BROADCAST, NUMTY_SHAPE, _CANARY_BY_MODULE, _swallowed, _warn_startup
from ._state import STATE
from ._forms import _LinForm, _is_unknown_fullname
from ._exprs import _EXPR_PREFIX, _LATE_PREFIXES, _expr_var, _form_of

from ._types import (
    _dim_value, _ndarray_parts, _parts_or_unpack, _distribute_solve, _conflict_pairs,
    _emit_call_site_conflicts, _callsite_formula_return, _mentions_ndarray, _info_mentions_array,
    _note_rank_mismatches, _ndarray_type, _union_members, INT_FULLNAME, TUPLE_FULLNAME,
    _OPERATOR_DUNDERS,
)
from ._names import _mask_pair_in, _named_pair_in, _names_enabled, _report_call_site_names
from ._builders import (
    _guard_mixed, zeros_like_hook, _named_dtype, like_function_hook, copy_function_hook,
    ascontiguousarray_function_hook, full_function_hook, eye_function_hook, linspace_function_hook,
    identity_function_hook, tri_function_hook, arange_function_hook, concatenate_function_hook,
    vstack_function_hook, hstack_function_hook, dstack_function_hook, column_stack_function_hook,
    stack_function_hook, r_getitem_hook,
)
from ._matmul import (
    _matmul_result, _matmul_unpack_dims, _operand_parts, matmul_hook, dot_function_hook,
    dot_method_hook, matmul_method_hook, rmatmul_method_hook, imatmul_method_hook,
    inner_function_hook, outer_function_hook, kron_function_hook, vdot_function_hook,
    tensordot_function_hook,
)
from ._ops import (
    flip_function_hook, fliplr_function_hook, flipud_function_hook, trace_function_hook,
    trace_method_hook, diag_function_hook, diagonal_function_hook, diagonal_method_hook,
    corrcoef_function_hook, cov_function_hook, roll_function_hook, gradient_function_hook,
    rot90_function_hook, convolve_function_hook, correlate_function_hook, tile_function_hook,
    repeat_function_hook, repeat_method_hook, pad_function_hook, split_function_hook,
    array_split_function_hook, nonzero_function_hook, count_nonzero_function_hook,
    flatnonzero_function_hook, argwhere_function_hook, unique_function_hook, isin_function_hook,
    searchsorted_function_hook, sliding_window_view_function_hook, atleast_function_hook,
)
from ._reshape import (
    _reshape_new_dims, _all_positionals, _scalar_key, reshape_method_hook, reshape_function_hook,
    transpose_method_hook, transpose_function_hook, swapaxes_function_hook, swapaxes_method_hook,
    moveaxis_function_hook, rollaxis_function_hook, matrix_transpose_function_hook,
    mt_attribute_hook, permute_dims_function_hook, append_function_hook, insert_function_hook,
    delete_function_hook, squeeze_method_hook, squeeze_function_hook, expand_dims_function_hook,
)
from ._reductions import (
    allany_method_hook, allany_function_hook, reduction_method_hook, ufunc_method_hook,
    arg_method_hook, arg_function_hook, rng_method_hook, legacy_rand_hook, percentile_function_hook,
    reduction_function_hook, average_function_hook, cumsum_function_hook, cumsum_method_hook,
    ravel_method_hook, flatten_method_hook, ravel_function_hook, diff_function_hook,
)

from ._fft import (
    fft2_function_hook, fft_function_hook, fftn_function_hook, fftshift_function_hook,
    irfft_function_hook, rfft2_function_hook, rfft_function_hook, rfftn_function_hook,
)
from ._einsum import _einsum_signature_hook, einsum_function_hook
from ._linalg import (
    _linalg_cross_signature_hook, _norm_signature_hook, cross_function_hook,
    linalg_cross_function_hook, linalg_function_hook,
)
from ._indexing import (
    getitem_method_hook, setitem_method_hook, take_along_axis_function_hook, take_function_hook,
    take_method_hook,
)
from ._broadcast import (
    angle_function_hook, asarray_function_hook, broadcast_method_hook, broadcast_to_function_hook,
    clip_function_hook, clip_method_hook, compare_method_hook, divmod_function_hook,
    divmod_method_hook, floordiv_mod_pow_method_hook, inplace_method_hook, isclose_function_hook,
    logical_method_hook, round_function_hook, round_method_hook, subtract_method_hook,
    truediv_method_hook, ufunc_call_hook, ufunc_function_hook, where_function_hook,
)
from ._frames import POLARS_FRAME_TO_NUMPY, POLARS_SERIES_TO_NUMPY, frame_to_numpy_hook
from ._heads import (
    _FAM_HEAD_DTYPES, _arr_head, _fam_applied, _fam_arg, _fam_bare, _fam_bind_words, _fam_ensure_implicit,
    _fam_expr, _fam_head, _fam_string,
)
from ._exprs import (
    _iter_tvarlikes,
)
from ._reshape import (
    _movable_parts,
)
from ._linalg import (
    _dtype_base,
)
from ._types import (
    _enum_member_count,
)
from ._unions import (
    _UNION_PAIR_METHODS, _reflected_name, _union_pair_hook, _union_reflected_hook,
    _union_signature_clash,
)

class NumtyPlugin(Plugin):
    def __init__(self, options) -> None:
        super().__init__(options)
        # Phase 4 (#198): process-global bundle; never NumtyState() here.
        self.state = STATE
        # Fullnames of TypeVars first bound from inside a string dim spec
        # (see _fam_bind_words). The pre-pass never saw them, so they are
        # missing from the analyzed function's variables; the call-site
        # hook adds them back so concrete calls still check and infer.
        self._fam_late_bound: set[str] = set()
        # Issue #164 step 2: stub canary state, once per module per run.
        self._startup_canary_checked: set[str] = set()
        # Bundle fields mirror the owning plugin's set objects (identity).
        self.state.fam_late_bound = self._fam_late_bound
        self.state.startup_canary_checked = self._startup_canary_checked

    def _ensure_startup_check(self, fullname: str) -> None:
        """Warn once about hooked names missing from the loaded stubs.

        Runs on hook lookup during type checking, when every reachable
        module is fully loaded: the requested `fullname` itself must
        resolve first, which proves its module is in the symbol table.
        Only canary entries under that same module are judged then, so
        a module the checked code never imports stays silent instead
        of false-positive warning. (Signature and type-analyze hooks
        run during semantic analysis, when modules are still loading,
        so they do not run this check.)
        """
        if not fullname.startswith("numpy."):
            return
        try:
            module, _, _ = fullname.rpartition(".")
            candidates = _CANARY_BY_MODULE.get(module)
            if not candidates:
                return
            unchecked = [c for c in candidates if c not in self._startup_canary_checked]
            if not unchecked:
                return
            try:
                sym = self.lookup_fully_qualified(fullname)
            except Exception:
                _swallowed(None, "_startup_canary")
                return
            if sym is None or getattr(sym, "node", None) is None:
                return  # module not loaded yet; retry on a later hook
            missing: list[str] = []
            for candidate in unchecked:
                try:
                    resolved = self.lookup_fully_qualified(candidate)
                except Exception:
                    _swallowed(None, "_startup_canary")
                    continue
                self._startup_canary_checked.add(candidate)
                if resolved is None or getattr(resolved, "node", None) is None:
                    missing.append(candidate)
            if missing:
                _warn_startup(
                    "numty: warning: "
                    f"{len(missing)} hooked name(s) missing from the loaded "
                    "numpy stubs, those hooks will silently never fire "
                    f"(see issue #164): {', '.join(missing)}"
                )
        except Exception:
            _swallowed(None, "startup_check")

    def get_function_signature_hook(
        self, fullname: str
    ) -> Callable[[FunctionSigContext], FunctionLike] | None:
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

    def callsite_signature_hook(self, ctx: FunctionSigContext) -> FunctionLike:
        try:
            sig = self._callsite_check(ctx)
        except Exception:
            _swallowed(ctx, "callsite_signature_hook")
            return ctx.default_signature
        _note_rank_mismatches(ctx, sig)
        return sig

    def get_method_signature_hook(
        self, fullname: str
    ) -> Callable[[MethodSigContext], FunctionLike] | None:
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

    def method_signature_hook(self, fullname: str, ctx: MethodSigContext) -> FunctionLike:
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
                # Issue #220: a method of a generic class takes an ndarray
                # with a method dim K and returns G[P, K]. When the receiver
                # uses a TypeVar from another scope (G[Q, H]), member access
                # already expanded the signature to mention caller Q. Adding
                # Q to variables makes it call-bound with no inference source
                # -> Never. Only late-bound/FAM vars need adding; receiver
                # tvars stay free.
                try:
                    receiver_names = {
                        tvar.fullname for tvar in _iter_tvarlikes(ctx.type)
                    }
                except Exception:
                    _swallowed(ctx, "method_signature_hook")
                    receiver_names = set()
                have |= receiver_names
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
                _report_call_site_names(sig, ctx)
        except Exception:
            _swallowed(ctx, "method_signature_hook")
            pass
        if isinstance(sig, CallableType) and not overloaded:
            sig = _callsite_formula_return(sig, ctx)
        _note_rank_mismatches(ctx, sig)
        return sig

    def _later_plugin_hook(
        self, ctx: MethodSigContext, getter: str, fullname: str
    ) -> Callable[[MethodSigContext], FunctionLike] | None:
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
            return any(v.fullname == fullname for v in sym.node.defn.type_vars)
        except Exception:
            _swallowed(None, "_is_class_tvar")
            return False

    def _callsite_check(self, ctx: FunctionSigContext) -> FunctionLike:
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
            _report_call_site_names(sig, ctx)
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
        if fullname == "numty.Arr":
            # The generic head (issue #278): the dtype rides the first
            # arg, the dims ride the rest flat.

            def arr_hook(ctx: AnalyzeTypeContext) -> Type:
                return self._arr_head(ctx)

            return arr_hook
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
        # Issues #219/#221: a generic NamedTuple (e.g. Block[int]) must keep
        # mypy's default TupleType handling, which carries per-field types for
        # _replace self-checks and tuple unpacking. Rebuilding via named_type
        # loses the TupleType and makes _replace reject and unpacking union
        # all fields. No hook for those: default handles both bare and
        # parameterized.
        if (
            getattr(info, "is_named_tuple", False)
            or getattr(info, "tuple_type", None) is not None
            or getattr(info, "typeddict_type", None) is not None
        ):
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

    # Phase 3 (#197) delegators: _fam_* bodies live in _heads.py as
    # module functions taking late_bound explicitly (same object as
    # self._fam_late_bound, mutated in place).
    def _fam_head(self, ctx: AnalyzeTypeContext, dtype_fullname: str) -> Type:
        return _fam_head(ctx, dtype_fullname, self._fam_late_bound)

    def _arr_head(self, ctx: AnalyzeTypeContext) -> Type:
        return _arr_head(ctx, self._fam_late_bound)

    def _fam_arg(
        self, ctx: AnalyzeTypeContext, head: str, arg: Type
    ) -> int | Type | list[int | Type]:
        return _fam_arg(ctx, head, arg, self._fam_late_bound)

    def _fam_string(
        self, ctx: AnalyzeTypeContext, head: str, text: str, node: RawExpressionType | UnboundType
    ) -> list[int | Type]:
        return _fam_string(ctx, head, text, node, self._fam_late_bound)

    def _fam_expr(
        self, ctx: AnalyzeTypeContext, head: str, word: str, node: RawExpressionType | UnboundType
    ) -> int | Type:
        return _fam_expr(ctx, head, word, node, self._fam_late_bound)

    def _fam_bind_words(
        self, ctx: AnalyzeTypeContext, words: list[str], node: RawExpressionType | UnboundType
    ) -> dict[str, TypeVarLikeType]:
        return _fam_bind_words(ctx, words, node, self._fam_late_bound)

    def _fam_ensure_implicit(
        self,
        ctx: AnalyzeTypeContext,
        analyzer: TypeAnalyser,
        scope: TypeVarLikeScope,
        word: str,
        node: RawExpressionType | UnboundType,
    ) -> TypeVarLikeType:
        return _fam_ensure_implicit(ctx, analyzer, scope, word, node, self._fam_late_bound)

    def _fam_bare(self, ctx: AnalyzeTypeContext, head: str, arg: UnboundType) -> int | Type:
        return _fam_bare(ctx, head, arg, self._fam_late_bound)

    def _fam_applied(self, ctx: AnalyzeTypeContext, head: str, arg: UnboundType) -> int | Type:
        return _fam_applied(ctx, head, arg, self._fam_late_bound)

    def get_function_hook(self, fullname: str) -> Callable[[FunctionContext], Type] | None:
        self._ensure_startup_check(fullname)
        return _guard_mixed(fullname, self._function_hook(fullname))

    def get_attribute_hook(
        self, fullname: str
    ) -> Callable[[AttributeContext], Type] | None:
        self._ensure_startup_check(fullname)
        return _guard_mixed(fullname, self._attribute_hook(fullname))

    def get_method_hook(self, fullname: str) -> Callable[[MethodContext], Type] | None:
        self._ensure_startup_check(fullname)
        hook = self._method_hook(fullname)
        short = fullname[len("numpy.ndarray."):] if fullname.startswith("numpy.ndarray.") else ""
        if hook is not None and short in _UNION_PAIR_METHODS:
            # Issue #265: one error for an operator on a union of arrays.
            hook = partial(_union_pair_hook, hook=hook, name=short)
        return _guard_mixed(fullname, hook)

    def _function_hook(self, fullname: str) -> Callable[..., Type] | None:
        if fullname == "builtins.len":
            return len_function_hook
        if fullname == "builtins.divmod":
            return divmod_function_hook
        reflected = _reflected_name(fullname)
        if reflected is not None:
            # Issues #252, #255, #256: a reflected operator on a union of arrays.
            return partial(
                _union_reflected_hook,
                name=reflected,
                solve=self._method_hook(f"numpy.ndarray.{reflected}"),
            )
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
        if fullname == POLARS_SERIES_TO_NUMPY:
            return partial(frame_to_numpy_hook, rank=1)
        if fullname == POLARS_FRAME_TO_NUMPY:
            return partial(frame_to_numpy_hook, rank=2)
        reflected = _reflected_name(fullname)
        if reflected is not None:
            # Issues #252, #255, #256: a reflected operator on a union of arrays.
            return partial(
                _union_reflected_hook,
                name=reflected,
                solve=self._method_hook(f"numpy.ndarray.{reflected}"),
            )
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
        if fullname == "numpy.lib._index_tricks_impl.RClass.__getitem__":
            return r_getitem_hook
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
        if fullname.startswith("numpy.ndarray.") and short == "trace":
            return trace_method_hook
        if fullname.startswith("numpy.ndarray.") and short == "diagonal":
            return diagonal_method_hook
        return None


def _reshape_signature_hook(ctx: MethodSigContext) -> FunctionLike:
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


def _broadcast_signature_hook(ctx: MethodSigContext) -> FunctionLike:
    """Widen other to Any on provable broadcast mismatch (issue #148).

    The method hook reports `broadcast shapes disagree` then mypy reports
    a mechanical [operator] duplicate. When both operands have known
    shapes that provably cannot broadcast (ints or N vs N-1 linear
    forms), return Any formals so the fallback's arg check succeeds and
    only the single [numty-broadcast] error remains. All other calls
    keep the default signature.
    """
    from numty_plugin import _broadcast_result
    try:
        sig = ctx.default_signature
        if not isinstance(sig, CallableType):
            return ctx.default_signature
        call = ctx.context
        # Issue #217: the method hook never runs when BOTH operands are
        # pure array unions, so a provable clash would pass silently.
        # Emit it here when every member pairing clashes (co-derived
        # unions share matching pairs, which must stay silent); the
        # widening below still suppresses the mechanical [operator]
        # duplicate, and any method-hook duplicate dedupes to one line.
        # This runs before the part_a bail: a union self has no single
        # dims list, so the per-operand path below cannot see it.
        try:
            from mypy.nodes import CallExpr as _CallExpr0
            from mypy.nodes import OpExpr as _OpExpr0
            uu_left: Type | None = None
            uu_right: Type | None = None
            if isinstance(call, _OpExpr0):
                try:
                    uu_left = ctx.api.get_expression_type(call.left)
                    uu_right = ctx.api.get_expression_type(call.right)
                except Exception:
                    _swallowed(ctx, "_broadcast_signature_hook")
                    uu_left = uu_right = None
            elif isinstance(call, _CallExpr0) and call.args:
                try:
                    uu_right = ctx.api.get_expression_type(call.args[0])
                    uu_left = ctx.type
                except Exception:
                    _swallowed(ctx, "_broadcast_signature_hook")
                    uu_left = uu_right = None
            if (
                uu_left is not None
                and uu_right is not None
                and _union_members(uu_left) is not None
                and _union_members(uu_right) is not None
            ):
                def _uu_dims(tp: Type) -> list[list[int | Type]] | None:
                    proper = get_proper_type(tp)
                    if not isinstance(proper, UnionType):
                        return None
                    out: list[list[int | Type]] = []
                    for item in proper.relevant_items():
                        part = _ndarray_parts(item)
                        if part is None:
                            part = _movable_parts(item)
                        if part is None:
                            return None
                        if part[0]:
                            out.append(list(part[0]))
                    return out or None

                llist0 = _uu_dims(uu_left)
                rlist0 = _uu_dims(uu_right)
                if llist0 is not None and rlist0 is not None:
                    first_msg: str | None = None
                    all_bad = True
                    for _da in llist0:
                        for _db in rlist0:
                            _, _m = _broadcast_result(list(_da), list(_db), None)
                            if _m is None:
                                all_bad = False
                                break
                            if first_msg is None:
                                first_msg = _m
                        if not all_bad:
                            break
                    if all_bad and first_msg is not None:
                        if isinstance(call, _OpExpr0) and call.op == "/":
                            # The message of the true-division hook: one
                            # line for the mismatch, not two (issue #252).
                            first_msg = f"true_divide shapes disagree: {first_msg}"
                        try:
                            ctx.api.fail(first_msg, ctx.context, code=NUMTY_BROADCAST)
                        except Exception:
                            _swallowed(ctx, "_broadcast_signature_hook")
                        return sig.copy_modified(
                            arg_types=[AnyType(TypeOfAny.explicit) for _ in sig.arg_types]
                        )
        except Exception:
            _swallowed(ctx, "_broadcast_signature_hook")
            pass
        try:
            if _union_signature_clash(ctx, matmul=False):
                return sig.copy_modified(
                    arg_types=[AnyType(TypeOfAny.explicit) for _ in sig.arg_types]
                )
        except Exception:
            _swallowed(ctx, "_broadcast_signature_hook")
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
            # Issue #227: mixed/plain-int shapes bail in _ndarray_parts but
            # the rank and literal axes are known via _movable_parts. Use
            # those for the provable-mismatch check so the [operator]
            # duplicate stays suppressed.
            part_a = _movable_parts(ctx.type)
        if part_a is None:
            return ctx.default_signature
        dims_a, _ = part_a
        if not dims_a:
            return ctx.default_signature
        # Other operand: OpExpr (a+b) has left/right; explicit
        # a.__add__(b) is a CallExpr with one arg.
        from mypy.nodes import CallExpr as _CallExpr
        from mypy.nodes import OpExpr as _OpExpr

        def _member_dims(tp: Type) -> list[list[int | Type]] | None:
            """Array dims per union member, else None.

            Issue #217: a union argument (e.g. F64[N] | F32[N]) must not
            hide a provable mismatch. Only pure array unions are handled
            (mirroring _union_members distribution); mixed unions bail.
            Issue #227: plain-int/mixed shapes fall back to _movable_parts
            so provable literal clashes still suppress [operator].
            """
            proper = get_proper_type(tp)
            if isinstance(proper, UnionType):
                out: list[list[int | Type]] = []
                for item in proper.relevant_items():
                    part = _ndarray_parts(item)
                    if part is None:
                        part = _movable_parts(item)
                    if part is None:
                        return None
                    dims, _ = part
                    if dims:
                        out.append(list(dims))
                return out or None
            part = _ndarray_parts(tp)
            if part is None:
                part = _movable_parts(tp)
            if part is None:
                return None
            dims, _ = part
            if not dims:
                return None
            return [list(dims)]

        def _any_mismatch(
            a_list: list[list[int | Type]], b_list: list[list[int | Type]]
        ) -> bool:
            try:
                names_on = _names_enabled(ctx)
            except Exception:
                _swallowed(ctx, "_any_mismatch")
                names_on = False
            for da in a_list:
                for db in b_list:
                    _, msg = _broadcast_result(list(da), list(db), None)
                    if msg is not None:
                        return True
                    # The opt-in reports for two different names (issue
                    # #280) and for selection counts (issue #288) are
                    # errors of the method hook as well.
                    if names_on and (_named_pair_in(da, db) or _mask_pair_in(da, db)):
                        return True
            return False

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
                right_list = _member_dims(right_tp)
                if right_list is not None and _any_mismatch([list(dims_a)], right_list):
                    return sig.copy_modified(
                        arg_types=[AnyType(TypeOfAny.explicit) for _ in sig.arg_types]
                    )
            except Exception:
                _swallowed(ctx, "_broadcast_signature_hook")
                pass
            try:
                left_tp = ctx.api.get_expression_type(call.left)
                left_list = _member_dims(left_tp)
                if left_list is not None and _any_mismatch([list(dims_a)], left_list):
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
        other_list = _member_dims(other_tp)
        if other_list is None:
            return ctx.default_signature
        if not _any_mismatch([list(dims_a)], other_list):
            return ctx.default_signature
        return sig.copy_modified(
            arg_types=[AnyType(TypeOfAny.explicit) for _ in sig.arg_types]
        )
    except Exception:
        _swallowed(ctx, "_broadcast_signature_hook")
        return ctx.default_signature


def _matmul_signature_hook(ctx: MethodSigContext) -> FunctionLike:
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
        if isinstance(call, _OpExpr) and _union_signature_clash(ctx, matmul=True):
            return sig.copy_modified(
                arg_types=[AnyType(TypeOfAny.explicit) for _ in sig.arg_types]
            )
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
        # UnionType has no copy_modified (it raised AttributeError here).
        return UnionType(items, proper.line, proper.column)
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
            "amax": partial(reduction_function_hook, op="amax"),
            "amin": partial(reduction_function_hook, op="amin"),
            "clip": clip_function_hook,
            "cumprod": partial(cumsum_function_hook, op="cumprod"),
            "cumsum": partial(cumsum_function_hook, op="cumsum"),
            "diagonal": diagonal_function_hook,
            "max": partial(reduction_function_hook, op="max"),
            "mean": partial(reduction_function_hook, op="mean"),
            "min": partial(reduction_function_hook, op="min"),
            "nonzero": nonzero_function_hook,
            "prod": partial(reduction_function_hook, op="prod"),
            "ptp": partial(reduction_function_hook, op="ptp"),
            "ravel": ravel_function_hook,
            "repeat": repeat_function_hook,
            "reshape": reshape_function_hook,
            "searchsorted": searchsorted_function_hook,
            "squeeze": squeeze_function_hook,
            "std": partial(reduction_function_hook, op="std"),
            "sum": partial(reduction_function_hook, op="sum"),
            "take": take_function_hook,
            "trace": trace_function_hook,
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
            "count_nonzero": count_nonzero_function_hook,
            "cross": cross_function_hook,
            "diagonal": diagonal_function_hook,
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
            "trace": trace_function_hook,
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
            "corrcoef": corrcoef_function_hook,
            "cov": cov_function_hook,
            "diag": diag_function_hook,
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
            "array_split": array_split_function_hook,
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
            "diag": diag_function_hook,
            "diagonal": diagonal_function_hook,
            "eye": eye_function_hook,
            "fliplr": fliplr_function_hook,
            "flipud": flipud_function_hook,
            "tri": tri_function_hook,
        },
        "numpy.lib._type_check_impl": {
            "imag": partial(ufunc_function_hook, group="real"),
            "real": partial(ufunc_function_hook, group="real"),
            "nan_to_num": copy_function_hook,
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
