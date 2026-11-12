"""Indexing hooks (Phase 3, #197). Top-level imports go to earlier layers only
(tests/test_layout.py, _LAYERS); a function-local facade import is used only
for a monkeypatch seam (_LAZY_SEAMS). _fresh_unknown only via
facade lazy (never from ._unknowns import) to preserve without_typevars
patch. NUMTY/_swallowed from _errors direct.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._errors import NUMTY_AXIS, NUMTY_BROADCAST, NUMTY_NAMES, NUMTY_SHAPE, _swallowed
from ._forms import (
    _LinForm,
)
from ._types import (
    INT_FULLNAME, _dim_eq, _dim_str, _dims_provably_ne, _distribute_solve, _fold_int,
    _has_unpack, _int_value, _named_sizes_differ, _names_enabled, _ndarray_parts, _ndarray_type,
    _parts_or_unpack, _report_mask_names,
    _report_names, _union_members,
)
from ._heads import (
    _index_record_exact, _index_recorded_exceeds, _recorded_at_site, _recorded_index_on,
)
from ._unknowns import _fresh_mask_count
from ._ops import flatnonzero_function_hook
from ._reshape import (
    _all_positionals, _dims_str, _formal_pair, _is_none_type, _known_rank_parts, _movable_parts,
    _normalize_axis, _normalize_axis_0d, _parse_axis_arg, _scalar_key,
)
from ._reductions import (
    _enum_member_value, _final_slice_call, _fold_index_bound,
)

if TYPE_CHECKING:
    from mypy.plugin import FunctionContext
    from mypy.plugin import FunctionSigContext
    from mypy.plugin import MethodContext
    

# Runtime-only (bodies check isinstance / read arg kinds).
from mypy.nodes import (
    ARG_POS, EllipsisExpr, Expression, IndexExpr, IntExpr, ListExpr, NameExpr, SliceExpr, StarExpr, TupleExpr,
)
from mypy.types import AnyType, Instance, TupleType, Type, TypeOfAny, TypeVarType, UnpackType, get_proper_type
from ._linalg import (
    _dtype_base, _kind_of, _scalar_of_dtype, _string_value,
)
from ._types import (
    _unpack_shape,
)


def _enclosing_func(
    ctx: MethodContext | FunctionSigContext | FunctionContext,
    use_line: int,
) -> tuple[str, object | None]:
    """Innermost function containing `use_line`: (fullname, node).

    Best-effort flow-insensitive scope for mask-var reuse (issue #288):
    the same mask variable in the same function reuses one count; a
    variable rebound between uses must not. Different scopes never share
    (documented limit: a global mask reused in two functions mints one
    count per function).
    """
    from mypy.nodes import ClassDef as _ClassDef
    from mypy.nodes import FuncDef as _FuncDef

    try:
        tree = getattr(ctx.api, "tree", None)
        module = getattr(tree, "fullname", "") or ""
        if tree is None or use_line <= 0:
            return f"{module}::<module>", None
        best: tuple[str, object | None] = (f"{module}::<module>", None)
        best_span = -1

        def _visit_func(node: object, prefix: str) -> None:
            nonlocal best, best_span
            line = getattr(node, "line", 0) or 0
            end = getattr(node, "end_line", 0) or line
            if not (line <= use_line <= end):
                return
            span = end - line
            fullname = getattr(node, "fullname", "") or f"{module}::{prefix}"
            if best[1] is None or span < best_span or best_span < 0:
                best = (fullname, node)
                best_span = span
            body = getattr(getattr(node, "body", None), "body", []) or []
            for stmt in body:
                if isinstance(stmt, _FuncDef):
                    _visit_func(stmt, getattr(stmt, "name", "?") or "?")

        defs = getattr(tree, "defs", []) or []
        for top in defs:
            if isinstance(top, _FuncDef):
                _visit_func(top, getattr(top, "name", "?") or "?")
            elif isinstance(top, _ClassDef):
                members = getattr(getattr(top, "defs", None), "body", []) or []
                for member in members:
                    if isinstance(member, _FuncDef):
                        _visit_func(member, getattr(member, "name", "?") or "?")
        return best
    except Exception:
        _swallowed(ctx, "_enclosing_func")
        tree = getattr(ctx.api, "tree", None)
        module = getattr(tree, "fullname", "") or "" if tree is not None else ""
        return f"{module}::<module>", None


def _count_var_assignments(func_node: object | None, varname: str) -> tuple[int, bool]:
    """(assignments to `varname`, is_param) in one function body.

    Flow-insensitive: any `Name = ...`, `for Name in ...`, `with ... as
    Name` or `+=` counts. Nested functions/classes are different scopes
    and are skipped (documented limit). `is_param` is whether the name is
    a parameter of the function.
    """
    from mypy.nodes import AssignmentStmt as _AssignmentStmt
    from mypy.nodes import ClassDef as _ClassDef
    from mypy.nodes import ForStmt as _ForStmt
    from mypy.nodes import FuncDef as _FuncDef
    from mypy.nodes import OperatorAssignmentStmt as _OpAssign
    from mypy.nodes import WithStmt as _WithStmt

    if func_node is None:
        return 0, False
    arg_names = getattr(func_node, "arg_names", []) or []
    is_param = varname in (arg_names or [])
    count = 0

    def _targets(lvalue: object) -> list[str]:
        if isinstance(lvalue, NameExpr):
            return [lvalue.name]
        if isinstance(lvalue, (TupleExpr, ListExpr)):
            out: list[str] = []
            for item in getattr(lvalue, "items", []) or []:
                out.extend(_targets(item))
            return out
        if isinstance(lvalue, StarExpr):
            return _targets(getattr(lvalue, "expr", None))
        return []

    def _walk(stmts: object) -> None:
        nonlocal count
        items = list(stmts) if isinstance(stmts, list) else []
        for stmt in items:
            if isinstance(stmt, (_FuncDef, _ClassDef)):
                continue
            if isinstance(stmt, _AssignmentStmt):
                for lv in getattr(stmt, "lvalues", []) or []:
                    if varname in _targets(lv):
                        count += 1
            elif isinstance(stmt, _OpAssign):
                if varname in _targets(getattr(stmt, "lvalue", None)):
                    count += 1
            elif isinstance(stmt, _ForStmt):
                if varname in _targets(getattr(stmt, "index", None)):
                    count += 1
            elif isinstance(stmt, _WithStmt):
                for item in getattr(stmt, "items", []) or []:
                    target = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else None
                    if target is not None and varname in _targets(target):
                        count += 1
            for attr in ("body", "else_body", "finally_body"):
                sub = getattr(stmt, attr, None)
                inner = getattr(sub, "body", None) if sub is not None else None
                if isinstance(inner, list) and inner:
                    _walk(inner)
            handlers = getattr(stmt, "handlers", None)
            if isinstance(handlers, list):
                for handler in handlers:
                    hbody = getattr(getattr(handler, "body", None), "body", None)
                    if isinstance(hbody, list) and hbody:
                        _walk(hbody)

    top_body = getattr(getattr(func_node, "body", None), "body", []) or []
    if isinstance(top_body, list):
        _walk(top_body)
    return count, is_param


def _own_index_annotation(func_node: object | None, varname: str, var_line: int) -> tuple[bool, bool, object | None]:
    """(has_ann, is_index, ann) for `varname`'s own annotation.

    Reads the annotation node off the enclosing function node: a
    parameter's `type_annotation`, or the annotated assignment on the
    matching line for a local. Positions identify the annotation site,
    so two Index parameters on one `def` line resolve independently
    (issue #286). `is_index` is true only for an `Index[...]`
    application; `ann` is its node (still unanalyzed). Never raises.
    """

    try:
        if func_node is None:
            return False, False, None
        ann: object | None = None
        for arg in getattr(func_node, "arguments", []) or []:
            variable = getattr(arg, "variable", None)
            if getattr(variable, "name", None) == varname:
                ann = getattr(arg, "type_annotation", None)
                break
        else:
            from mypy.nodes import AssignmentStmt as _AssignmentStmt

            for stmt in getattr(getattr(func_node, "body", None), "body", []) or []:
                if not isinstance(stmt, _AssignmentStmt):
                    continue
                if getattr(stmt, "line", -1) != var_line:
                    continue
                for lvalue in getattr(stmt, "lvalues", []) or []:
                    if isinstance(lvalue, NameExpr) and lvalue.name == varname:
                        ann = getattr(stmt, "type_annotation", None)
                        break
                if ann is not None:
                    break
        if ann is None:
            return False, False, None
        name = getattr(ann, "name", None)
        if not isinstance(name, str):
            return True, False, None
        if name != "Index" and not name.endswith(".Index"):
            return True, False, None
        return True, True, ann
    except Exception:
        _swallowed(None, "_own_index_annotation")
        return False, False, None


def _ann_recorded_name(ann: object | None) -> int | str | None:
    """The recorded size spelled in an Index[...] annotation, else None.

    A bare name stays a string (resolved against real dims by the
    caller, which knows the scope); ints stay ints; linear forms and
    anything computed stay silent. Never raises.
    """
    try:
        if ann is None:
            return None
        args = getattr(ann, "args", ()) or ()
        if not args:
            return None
        first = args[0]
        tname = type(first).__name__
        if tname == "UnboundType":
            sub = getattr(first, "args", ()) or ()
            name = getattr(first, "name", None)
            if sub or not isinstance(name, str):
                return None
            word = name.split()
            # Bare size names match by string; anything else (dotted
            # paths, linear forms) stays silent here — the analyzed
            # record covers those when present.
            if len(word) != 1 or not word[0].isidentifier():
                return None
            return word[0]
        if isinstance(first, bool):
            return None
        if isinstance(first, int):
            return first
        if tname == "RawExpressionType":
            try:
                lit = first.literal_value()  # type: ignore[union-attr]
            except Exception:
                _swallowed(None, "_ann_recorded_name")
                return None
            if isinstance(lit, bool):
                return None
            if isinstance(lit, int):
                return lit
            if isinstance(lit, str):
                word = lit.split()
                if len(word) != 1 or not word[0].isidentifier():
                    return None
                return word[0]
            return None
        if isinstance(first, TypeVarType):
            return first
        proper = get_proper_type(first)
        if isinstance(proper, TypeVarType):
            return proper
        return None
    except Exception:
        _swallowed(None, "_ann_recorded_name")
        return None


def _selection_base(
    ctx: MethodContext | FunctionSigContext, expr: IndexExpr
) -> tuple[Expression, list[int | Type]] | None:
    """The index array a 1-d selection was taken from, with its dims, else None.

    `rows[known]`, `rows[a:b]` and `rows[order]` hold fewer positions,
    or the same ones in another order, and every one of them is a value
    of `rows`: a selection changes how many positions there are, not the
    size they are positions on (issue #305). Only a 1-d integer array
    selected by a mask, a slice or an integer array is looked through.
    """
    lookup = getattr(ctx.api, "lookup_type_or_none", None)
    index = expr.index
    if lookup is None or isinstance(index, (TupleExpr, EllipsisExpr)):
        return None
    base_tp = lookup(expr.base)
    parts = _ndarray_parts(base_tp) if base_tp is not None else None
    if parts is None or len(parts[0]) != 1 or _dtype_base(parts[1]) not in ("i", "u"):
        return None
    if not isinstance(index, SliceExpr):
        index_tp = lookup(index)
        iparts = (_ndarray_parts(index_tp) or _known_rank_parts(index_tp)) if index_tp is not None else None
        if iparts is None or len(iparts[0]) != 1 or _dtype_base(iparts[1]) not in ("b", "i", "u"):
            return None
    return expr.base, list(parts[0])


def _recorded_for_operand(
    ctx: MethodContext | FunctionSigContext,
    module: str,
    func_node: object | None,
    expr: Expression | None,
    idims: list[int | Type],
    depth: int = 0,
) -> int | Type | str | None:
    """The size an index operand's values are positions on, else None.

    Order: an Index[...] annotation on the operand's own parameter or
    local wins (name-keyed, so same-line parameters stay distinct);
    a reassigned variable bails (its record is stale: `clip`/`- 1`
    clear the positions); otherwise the producer line entries apply
    (direct calls, single-assignment stores). Anything ambiguous stays
    silent. Never raises.
    """
    try:
        if isinstance(expr, IndexExpr) and depth < 4:
            base = _selection_base(ctx, expr)
            if base is not None:
                return _recorded_for_operand(ctx, module, func_node, base[0], base[1], depth + 1)
        if not isinstance(expr, NameExpr):
            return _recorded_index_on(ctx, idims, expr)
        varname = expr.name
        if not varname or varname in ("True", "False", "None"):
            return None
        node = getattr(expr, "node", None)
        if type(node).__name__ != "Var":
            return _recorded_index_on(ctx, idims, expr)
        var_line = getattr(node, "line", 0) or 0
        use_line = getattr(expr, "line", 0) or 0
        if func_node is None:
            _, func_node = _enclosing_func(ctx, use_line or var_line)
        count, is_param = _count_var_assignments(func_node, varname)
        if is_param:
            if count >= 1:
                return None
        elif count > 1:
            return None
        has_ann, is_index, ann = _own_index_annotation(func_node, varname, var_line or use_line)
        if is_index:
            # The annotation site is unique per parameter even on one
            # shared `def` line; the analyzed record was stored there.
            # Without one (warm cache, odd spelling), the spelled name
            # still matches by string.
            if ann is not None:
                aline = getattr(ann, "line", 0) or 0
                acol = getattr(ann, "column", 0) or 0
                rec = _recorded_at_site(ctx, module, aline, acol)
                if rec is not None:
                    return rec
            return _ann_recorded_name(ann)
        if has_ann:
            # A non-Index annotation carries no record (plain I64 stays
            # silent even where a producer record shares the line).
            return None
        return _recorded_index_on(ctx, idims, expr)
    except Exception:
        _swallowed(ctx, "_recorded_for_operand")
        try:
            return _recorded_index_on(ctx, idims, expr)
        except Exception:
            _swallowed(ctx, "_recorded_for_operand")
            return None


def _mask_var_key(ctx: MethodContext | FunctionContext, mask_expr: Expression | None) -> str | None:
    """Stable key for a mask variable use, else None to mint per site (issue #288).

    Same `Name` in the same function (module::func::name), not rebound
    between uses, reuses one unknown. Inline masks (`x[x > 0]`), member
    access, `flatnonzero` results and any rebinding bail to a fresh
    per-site unknown (documented limits).
    """
    if not isinstance(mask_expr, NameExpr):
        return None
    varname = mask_expr.name
    if not varname or varname in ("True", "False", "None"):
        return None
    try:
        use_line = getattr(mask_expr, "line", 0) or getattr(ctx.context, "line", 0) or 0
        func_fullname, func_node = _enclosing_func(ctx, use_line)
        count, is_param = _count_var_assignments(func_node, varname)
        if is_param:
            if count >= 1:
                return None
        elif count > 1:
            return None
        tree = getattr(ctx.api, "tree", None)
        module = getattr(tree, "fullname", "") or ""
        return f"{module}::{func_fullname}::{varname}"
    except Exception:
        _swallowed(ctx, "_mask_var_key")
        return None


def _mask_count_for(ctx: MethodContext, mask_expr: Expression | None, dim: int) -> Type:
    """The count dim for one boolean-mask use (issue #288).

    Same variable without rebinding shares one unknown; everything else
    mints a fresh per-site unknown as before.
    """
    from numty_plugin import _fresh_unknown

    key = _mask_var_key(ctx, mask_expr)
    if key is None:
        return _fresh_unknown(ctx, "mask", dim)
    try:
        return _fresh_mask_count(ctx, key)
    except Exception:
        _swallowed(ctx, "_mask_count_for")
        return _fresh_unknown(ctx, "mask", dim)


def flatnonzero_mask_hook(ctx: FunctionContext) -> Type:
    """`np.flatnonzero(mask)` of a boolean mask variable has the count of that mask.

    `x[keep]` and `np.flatnonzero(keep)` hold one entry per True of
    `keep`, so they share the unknown of the variable (issue #288), and
    under the opt-in code the positions are a size of their own against
    a name. Any other argument (an expression, a non-boolean array, a
    variable bound twice) gives a fresh unknown, as before.
    """
    count: Type | None = None
    try:
        if ctx.args and ctx.args[0] and ctx.arg_types and ctx.arg_types[0]:
            expr = ctx.args[0][0]
            parts = _ndarray_parts(ctx.arg_types[0][0])
            if isinstance(expr, NameExpr) and parts is not None and _dtype_base(parts[1]) == "b":
                key = _mask_var_key(ctx, expr)
                if key is not None:
                    count = _fresh_mask_count(ctx, key)
    except Exception:
        _swallowed(ctx, "flatnonzero_mask_hook")
        count = None
    return flatnonzero_function_hook(ctx, count)


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
    parts = _ndarray_parts(array) or _known_rank_parts(array)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    # Issue #183: validate the axis before the indices, so a literal
    # axis out of range for a known rank fails even when the index is
    # too complex to solve (e.g. nested [[0], [0]] bails on indices).
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if isinstance(axis, list):
        if len(axis) != 1 or not isinstance(axis[0], int):
            ctx.api.fail("np.take axis must be an integer or None", ctx.context, code=NUMTY_AXIS)
            return ctx.default_return_type
        if _normalize_axis_0d(axis[0], len(dims)) is None:
            ctx.api.fail(
                f"take axis {axis[0]} is out of range for {len(dims)} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
    idx_dims, is_scalar = _take_indices(pair)
    if idx_dims is None:
        return ctx.default_return_type
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
    if axis is None:
        # Unknown axis (e.g. symbolic): cannot judge, keep the stubs.
        return ctx.default_return_type
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
    if part is None:
        # Issue #226: Bool[int] (and other plain-int shapes) bails in
        # _ndarray_parts, but the shape is known via _movable_parts.
        # Masks/fancy indices there keep the dtype instead of Any.
        movable = _movable_parts(tp)
        if movable is not None:
            kind = _dtype_base(movable[1])
            if kind is None:
                return None
            if kind in ("i", "u"):
                return ("fancy0d",) if not movable[0] else ("fancy", list(movable[0]))
            if kind == "b":
                return ("mask", list(movable[0]))
            return ("badindex",)
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
    # Issue #231: x[[-1]] is a fancy index of length 1, not a bail.
    if isinstance(expr, (ListExpr, TupleExpr)) and expr.items:
        if all(_fold_int(e) is not None for e in expr.items):
            return ("fancy", [len(expr.items)])
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
        # Issue #231: numpy integer scalars (np.intp, np.int64, loop vars
        # like `for t in steps` which reveal as signedinteger[_64Bit])
        # are valid indices like Python int, not Any.
        if proper_tp.type.fullname in (
            "numpy.signedinteger",
            "numpy.unsignedinteger",
            "numpy.integer",
        ):
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
    from numty_plugin import _fresh_unknown
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
    exprs: list[Expression | None] | None = None,
) -> tuple[list[int | Type], list[int | Type]] | None:
    """Items before ... consume the known prefix; (kept, unmatched)."""
    out: list[int | Type] = []
    i = 0
    for pos, item in enumerate(items):
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
                # Issue #228: symbolic lengths that are provably different
                # (N-1 vs N) must be reported like literal mismatches.
                if _dims_provably_ne(md, pd):
                    ctx.api.fail(
                        f"boolean index did not match indexed array along axis {i + off}; "
                        f"size of axis is {_dims_str([pd])} but size of corresponding "
                        f"boolean axis is {_dims_str([md])} (shape {shape_repr})",
                        ctx.context,
                        code=NUMTY_AXIS,
                    )
                    return None
                _report_names(ctx, md, pd, f"boolean index on axis {i + off} (shape {shape_repr})")
            mask_expr = exprs[pos] if exprs is not None and pos < len(exprs) else None
            out.append(_mask_count_for(ctx, mask_expr, i))
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
            frag = slot[idx]
            assert frag is not None
            out.extend(frag)
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
        exprs = [p[1] for p in pairs]
        leading = _consume_prefix(ctx, items, prefix, shape_repr, exprs)
        if leading is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, leading[0] + leading[1] + [unpack] + suffix, dtype)
    pos = positions[0]
    left = _consume_prefix(ctx, items[:pos], prefix, shape_repr, [p[1] for p in pairs[:pos]])
    if left is None:
        return ctx.default_return_type
    right = _consume_suffix(ctx, items[pos + 1 :], suffix, shape_repr)
    if right is None:
        return ctx.default_return_type
    out = left[0] + left[1] + [unpack] + right[0] + right[1]
    return _ndarray_type(ctx, out, dtype)


def _check_index_positions(
    ctx: MethodContext,
    pair: tuple[Type, Expression | None],
    size: int | Type,
    axis: int,
    dims: list[int | Type],
) -> bool:
    """A recorded positions-on size must fit the axis it indexes (issues #285, #287).

    A recorded bound provably above the axis (N+1 positions on an N
    axis) is an error; a different *named* size is an opt-in
    [numty-names] error; equal, fitting, unknown, or unrecorded
    operands stay silent. Returns True when an error was reported, so
    the caller can keep the stubs' type: like every older index error,
    a failed check yields the default rather than a solved shape, which
    also keeps a mechanical [operator] duplicate off enclosing
    expressions.
    """
    tp, expr = pair
    parts = _ndarray_parts(tp)
    if parts is None:
        return False
    idims, _ = parts
    if len(idims) != 1:
        return False
    try:
        tree = getattr(ctx.api, "tree", None)
        module = getattr(tree, "fullname", "") or ""
        use_line = getattr(expr, "line", 0) or getattr(ctx.context, "line", 0) or 0
        _, func_node = _enclosing_func(ctx, use_line)
        recorded = _recorded_for_operand(ctx, module, func_node, expr, list(idims))
    except Exception:
        _swallowed(ctx, "_check_index_positions")
        return False
    if recorded is None:
        return False
    if isinstance(recorded, str):
        # The annotation spelled a bare size name the analyzed record
        # missed (warm cache, odd spelling): match by name. Same name
        # as the axis is fitting; a different named axis is the opt-in
        # report. Linear forms fall back to silent here; the analyzed
        # record covers them when present.
        if isinstance(size, int):
            return False
        proper_size = get_proper_type(size)
        if isinstance(proper_size, TypeVarType) and proper_size.name == recorded:
            return False
        if _names_enabled(ctx):
            if isinstance(proper_size, TypeVarType) and proper_size.name != recorded:
                ctx.api.fail(
                    f"index positions on axis {axis} (shape {_dims_str(dims)}): "
                    f"{recorded} and {_dim_str(size)} are different names",
                    ctx.context,
                    code=NUMTY_NAMES,
                )
                return True
        return False
    if _dim_eq(recorded, size):
        return False
    if _index_recorded_exceeds(recorded, size):
        # An error by default only when the overrun is certain: the
        # operand holds every position below the recorded size (`arange`,
        # `argsort`). `searchsorted`, `argmax`, `flatnonzero` and an
        # `Index[...]` annotation give an upper bound on the values, so
        # `a[np.searchsorted(a, v)]` is wrong only when a value is above
        # `a[-1]`, and code that clips, guards or fixes the positions in
        # place is right. That report is under the opt-in code.
        certain = _index_record_exact(ctx, expr, recorded)
        if not certain and not _names_enabled(ctx):
            return False
        ctx.api.fail(
            f"index positions on {_dim_str(recorded)} used on axis {axis} "
            f"of size {_dim_str(size)} (shape {_dims_str(dims)})"
            + ("" if certain else ": they may run past the axis"),
            ctx.context,
            code=NUMTY_AXIS if certain else NUMTY_NAMES,
        )
        return True
    _report_names(
        ctx,
        recorded,
        size,
        f"index positions on axis {axis} (shape {_dims_str(dims)})",
    )
    try:
        return bool(_names_enabled(ctx) and _named_sizes_differ(recorded, size))
    except Exception:
        _swallowed(ctx, "_check_index_positions")
        return True


def _solve_getitem(
    ctx: MethodContext,
    dims: list[int | Type],
    dtype: Type,
    pair: tuple[Type, Expression | None],
) -> Type:
    from numty_plugin import _broadcast_result, _fresh_unknown, _materialize_dim, _resolve_dim
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
    pos_failed = False
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
            # A failed positions check still runs the remaining axes
            # (all errors are reported); the result below is error-Any
            # like every older index error, so no mechanical [operator]
            # duplicate lands on an enclosing expression (issue #148's
            # widening, applied at the source instead of the operator).
            if _check_index_positions(ctx, pairs[n], dims[dim], dim, dims):
                pos_failed = True
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
                    # Issue #228: symbolic lengths that are provably
                    # different (N-1 vs N) report like literal mismatches.
                    if _dims_provably_ne(md, pd):
                        ctx.api.fail(
                            f"boolean index did not match indexed array along axis {dim + off}; "
                            f"size of axis is {_dims_str([pd])} but size of corresponding "
                            f"boolean axis is {_dims_str([md])} (shape {_dims_str(dims)})",
                            ctx.context,
                            code=NUMTY_AXIS,
                        )
                        return ctx.default_return_type
                    _report_names(ctx, md, pd, f"boolean index on axis {dim + off} (shape {_dims_str(dims)})")
                mask_expr = pairs[n][1] if n < len(pairs) else None
                unk = _mask_count_for(ctx, mask_expr, dim)
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
        # The dtype is known even when the bound is not (issue #230:
        # np.r_[x[0], x] needs the float64 scalar, not Any).
        if pos_failed:
            return AnyType(TypeOfAny.from_error)
        scalar = _scalar_of_dtype(dtype)
        return scalar if scalar is not None else ctx.default_return_type
    if pos_failed:
        return AnyType(TypeOfAny.from_error)
    return _ndarray_type(ctx, out, dtype)


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
    unpack and unknown targets bail. Returns the stub default (None) in
    every case.
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
        # The rule of a broadcast (issue 290): two literal sizes, or two
        # symbolic ones that no sizes make equal (`C-1` into `C`). Only
        # literal sizes were compared, so `out[i] = v[:-1]` was silent
        # while `out[i] + v[:-1]` was an error.
        if _dims_provably_ne(t, v):
            ctx.api.fail(
                f"cannot assign value of shape {_dims_str(value_dims)} "
                f"into target of shape {_dims_str(target_dims)}",
                ctx.context,
                code=NUMTY_BROADCAST,
            )
            return ctx.default_return_type
        # Unknown, or two names that may be equal: unprovable. The
        # other axes are still compared.
        _report_names(
            ctx, v, t, f"assignment of shape {_dims_str(value_dims)} into target of shape {_dims_str(target_dims)}"
        )
        _report_mask_names(
            ctx, v, t, f"assignment of shape {_dims_str(value_dims)} into target of shape {_dims_str(target_dims)}"
        )
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
    # Issue #226: F64[int]/Bool[int] have plain-int dims; _ndarray_parts
    # bails but the rank is known via _movable_parts. Solve with those
    # dims (plain ints stay dynamic, masks mint unknowns) instead of Any.
    movable = _movable_parts(tp)
    if movable is not None:
        dims, dtype = movable
        return _solve_getitem(ctx, list(dims), dtype, pair)
    split = _unpack_shape(tp)
    if split is None:
        return ctx.default_return_type
    (prefix, unpack, suffix), dtype = split
    shape_repr = f"({_dims_str(prefix + [unpack] + suffix)})"
    return _solve_getitem_unpack(ctx, prefix, unpack, suffix, dtype, pair, shape_repr)


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
