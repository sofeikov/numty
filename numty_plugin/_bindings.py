"""_bindings.py — how a name is bound in the function that uses it: the
enclosing function of a line, how many times a name is assigned there,
whether a loop rebinds it, and the value of a name bound once. Split out
of _indexing.py (near the 2000-line limit) so the size readers of earlier
layers (`_fold_dim_expr`) can follow a local too (issue 359). Top-level
imports go to earlier layers only (order in tests/test_layout.py, _LAYERS).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.nodes import Expression, ListExpr, NameExpr, StarExpr, TupleExpr

from ._errors import _swallowed

if TYPE_CHECKING:
    from mypy.plugin import AttributeContext, FunctionContext, FunctionSigContext, MethodContext


def _enclosing_func(
    ctx: MethodContext | FunctionSigContext | FunctionContext | AttributeContext,
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


def _nested_bodies(stmt: object) -> list[list]:
    """The statement lists inside a compound statement.

    `body` is one block (`for`, `while`, `with`, `try`) or a list of
    blocks (`if` / `elif`); `handlers` is the list of `except` blocks.
    """
    out: list[list] = []
    for attr in ("body", "else_body", "finally_body", "handlers"):
        sub = getattr(stmt, attr, None)
        for block in sub if isinstance(sub, list) else [sub]:
            inner = getattr(block, "body", None)
            if isinstance(inner, list) and inner:
                out.append(inner)
    return out


def _shallow_bodies(stmt: object) -> list[list]:
    """`_nested_bodies` without the branches of an `if`: what the counts of issue #288 were built on.

    The records of index positions rely on it (a name assigned in both
    branches of an `if` keeps its record, `bad_index_overrun.py`).
    """
    out: list[list] = []
    for attr in ("body", "else_body", "finally_body"):
        sub = getattr(stmt, attr, None)
        inner = getattr(sub, "body", None) if sub is not None else None
        if isinstance(inner, list) and inner:
            out.append(inner)
    handlers = getattr(stmt, "handlers", None)
    if isinstance(handlers, list):
        for handler in handlers:
            hbody = getattr(getattr(handler, "body", None), "body", None)
            if isinstance(hbody, list) and hbody:
                out.append(hbody)
    return out


def _count_var_assignments(func_node: object | None, varname: str, deep: bool = False) -> tuple[int, bool]:
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
            for inner in _nested_bodies(stmt) if deep else _shallow_bodies(stmt):
                _walk(inner)

    top_body = getattr(getattr(func_node, "body", None), "body", []) or []
    if isinstance(top_body, list):
        _walk(top_body)
    return count, is_param


def _bound_in_loop(func_node: object | None, varname: str) -> bool:
    """Whether `varname` may hold a value per run of a loop (issue 328).

    A binding inside a `for` / `while` body, or as a `for` target, does;
    so does a name with no binding of its own in the function that is
    not a parameter (a comprehension variable, a walrus, a global).
    """
    from mypy.nodes import AssignmentStmt as _AssignmentStmt
    from mypy.nodes import ClassDef as _ClassDef
    from mypy.nodes import ForStmt as _ForStmt
    from mypy.nodes import FuncDef as _FuncDef
    from mypy.nodes import WhileStmt as _WhileStmt

    count, is_param = _count_var_assignments(func_node, varname, deep=True)
    if count == 0:
        return not is_param

    def _names(lvalue: object) -> list[str]:
        if isinstance(lvalue, NameExpr):
            return [lvalue.name]
        if isinstance(lvalue, (TupleExpr, ListExpr)):
            return [n for item in getattr(lvalue, "items", []) or [] for n in _names(item)]
        if isinstance(lvalue, StarExpr):
            return _names(getattr(lvalue, "expr", None))
        return []

    def _walk(stmts: object, in_loop: bool) -> bool:
        for stmt in stmts if isinstance(stmts, list) else []:
            if isinstance(stmt, (_FuncDef, _ClassDef)):
                continue
            loop = in_loop or isinstance(stmt, (_ForStmt, _WhileStmt))
            if loop:
                if isinstance(stmt, _ForStmt) and varname in _names(stmt.index):
                    return True
                if isinstance(stmt, _AssignmentStmt) and any(varname in _names(lv) for lv in stmt.lvalues):
                    return True
            for inner in _nested_bodies(stmt):
                if _walk(inner, loop):
                    return True
        return False

    return _walk(getattr(getattr(func_node, "body", None), "body", None), False)


def _single_rvalue(func_node: object | None, varname: str) -> Expression | None:
    """The value of `varname = <expr>` when that is the only binding of the name in the function."""
    from mypy.nodes import AssignmentStmt as _AssignmentStmt
    from mypy.nodes import ClassDef as _ClassDef
    from mypy.nodes import FuncDef as _FuncDef

    count, is_param = _count_var_assignments(func_node, varname, deep=True)
    if is_param or count != 1:
        return None
    found: list[Expression] = []

    def _walk(stmts: object) -> None:
        for stmt in stmts if isinstance(stmts, list) else []:
            if isinstance(stmt, (_FuncDef, _ClassDef)):
                continue
            if isinstance(stmt, _AssignmentStmt) and len(stmt.lvalues) == 1:
                lv = stmt.lvalues[0]
                if isinstance(lv, NameExpr) and lv.name == varname:
                    found.append(stmt.rvalue)
            for inner in _nested_bodies(stmt):
                _walk(inner)

    _walk(getattr(getattr(func_node, "body", None), "body", None))
    return found[0] if len(found) == 1 else None


def _assignment_lines(func_node: object | None, varname: str) -> list[int]:
    """The lines of the plain assignments `varname = ...` in the function, at any depth."""
    from mypy.nodes import AssignmentStmt as _AssignmentStmt
    from mypy.nodes import ClassDef as _ClassDef
    from mypy.nodes import FuncDef as _FuncDef

    lines: list[int] = []

    def _walk(stmts: object) -> None:
        for stmt in stmts if isinstance(stmts, list) else []:
            if isinstance(stmt, (_FuncDef, _ClassDef)):
                continue
            if isinstance(stmt, _AssignmentStmt) and any(
                isinstance(lv, NameExpr) and lv.name == varname for lv in stmt.lvalues
            ):
                lines.append(stmt.line)
            for inner in _nested_bodies(stmt):
                _walk(inner)

    _walk(getattr(getattr(func_node, "body", None), "body", None))
    return lines


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


def _single_rvalue_of(ctx: MethodContext | FunctionContext, expr: NameExpr) -> Expression | None:
    """The value of `expr`'s variable when `name = <value>` is its only binding in the enclosing function."""
    use_line = getattr(expr, "line", 0) or getattr(ctx.context, "line", 0) or 0
    return _single_rvalue(_enclosing_func(ctx, use_line)[1], expr.name)


def _value_of_rebound_name(
    ctx: MethodContext | FunctionContext | AttributeContext, expr: object
) -> bool:
    """Whether `expr` is the whole value of `name = expr` for a name bound elsewhere too.

    mypy types a local from its first binding, so a size there (`N`,
    `K*N`, `?K@14`) refuses a later `total += y.size` or `n //= 2` that
    is correct code. The readers that give a size (`.size`, `len()`, an
    `int`-list index) give what main gave instead when the name is a
    parameter or is bound more than once in the function.
    """
    from mypy.nodes import AssignmentStmt as _AssignmentStmt
    from mypy.nodes import ClassDef as _ClassDef
    from mypy.nodes import FuncDef as _FuncDef

    try:
        _, func_node = _enclosing_func(ctx, getattr(expr, "line", 0) or 0)
        if func_node is None:
            return False
        names: list[str] = []

        def _walk(stmts: object) -> None:
            for stmt in stmts if isinstance(stmts, list) else []:
                if isinstance(stmt, (_FuncDef, _ClassDef)):
                    continue
                if isinstance(stmt, _AssignmentStmt) and stmt.rvalue is expr:
                    names.extend(lv.name for lv in stmt.lvalues if isinstance(lv, NameExpr))
                for inner in _nested_bodies(stmt):
                    _walk(inner)

        _walk(getattr(getattr(func_node, "body", None), "body", None))
        for name in names:
            count, is_param = _count_var_assignments(func_node, name, deep=True)
            if is_param or count > 1:
                return True
        return False
    except Exception:
        _swallowed(ctx, "_value_of_rebound_name")
        return False


def _is_conditional_branch(ctx: MethodContext | FunctionContext, expr: object) -> bool:
    """Whether `expr` is a whole branch of a conditional expression (`a if c else expr`).

    The conditional is looked for as the value of an assignment, a
    `return` or an expression statement of the enclosing function.
    """
    from mypy.nodes import ConditionalExpr as _ConditionalExpr
    from mypy.nodes import ClassDef as _ClassDef
    from mypy.nodes import FuncDef as _FuncDef

    def _branch_of(value: object) -> bool:
        if not isinstance(value, _ConditionalExpr):
            return False
        return (
            value.if_expr is expr
            or value.else_expr is expr
            or _branch_of(value.if_expr)
            or _branch_of(value.else_expr)
        )

    def _walk(stmts: object) -> bool:
        for stmt in stmts if isinstance(stmts, list) else []:
            if isinstance(stmt, (_FuncDef, _ClassDef)):
                continue
            if _branch_of(getattr(stmt, "rvalue", None)) or _branch_of(getattr(stmt, "expr", None)):
                return True
            if any(_walk(inner) for inner in _nested_bodies(stmt)):
                return True
        return False

    try:
        _, func_node = _enclosing_func(ctx, getattr(expr, "line", 0) or 0)
        if func_node is None:
            return False
        return _walk(getattr(getattr(func_node, "body", None), "body", None))
    except Exception:
        _swallowed(ctx, "_is_conditional_branch")
        return False


# Per module tree: (tree, binding counts of attributes, of module names,
# names declared `global`, targets per assignment value id). Rebuilt when
# the tree is a new object (a daemon re-parse).
_TargetIndex = tuple[
    dict[tuple[str, str], int], dict[str, int], set[str], dict[int, list[tuple[str, str, str]]]
]
_TARGET_INDEX: dict[int, tuple[object, _TargetIndex]] = {}
# The count an annotated binding adds: it marks the target declared.
_DECLARED = 1 << 20


def _target_index(tree: object) -> _TargetIndex:
    """Bindings of attributes (per class) and module names across one module.

    See _value_of_rebound_target. An attribute is keyed by the innermost
    class around its binding, "" outside every class.
    """
    from mypy.nodes import AssignmentStmt as _AssignmentStmt
    from mypy.nodes import ClassDef as _ClassDef
    from mypy.nodes import Decorator as _Decorator
    from mypy.nodes import ForStmt as _ForStmt
    from mypy.nodes import FuncDef as _FuncDef
    from mypy.nodes import GlobalDecl as _GlobalDecl
    from mypy.nodes import MemberExpr as _MemberExpr
    from mypy.nodes import OperatorAssignmentStmt as _OpAssign
    from mypy.nodes import OverloadedFuncDef as _OverloadedFuncDef
    from mypy.nodes import WithStmt as _WithStmt

    cached = _TARGET_INDEX.get(id(tree))
    if cached is not None and cached[0] is tree:
        return cached[1]
    attrs: dict[tuple[str, str], int] = {}
    names: dict[str, int] = {}
    declared: set[str] = set()
    values: dict[int, list[tuple[str, str, str]]] = {}

    def _flat(lvalue: object) -> list[object]:
        if isinstance(lvalue, (TupleExpr, ListExpr)):
            return [leaf for item in lvalue.items for leaf in _flat(item)]
        if isinstance(lvalue, StarExpr):
            return _flat(lvalue.expr)
        return [lvalue]

    def _walk(stmts: object, level: str, owner: str = "") -> None:
        for stmt in stmts if isinstance(stmts, list) else []:
            if isinstance(stmt, _FuncDef):
                _walk(stmt.body.body, "func", owner)
                continue
            if isinstance(stmt, _Decorator):
                _walk(stmt.func.body.body, "func", owner)
                continue
            if isinstance(stmt, _OverloadedFuncDef):
                for item in stmt.items:
                    func = item.func if isinstance(item, _Decorator) else item
                    _walk(func.body.body, "func", owner)
                if stmt.impl is not None:
                    impl = stmt.impl.func if isinstance(stmt.impl, _Decorator) else stmt.impl
                    _walk(impl.body.body, "func", owner)
                continue
            if isinstance(stmt, _ClassDef):
                _walk(stmt.defs.body, "class", stmt.fullname)
                continue
            if isinstance(stmt, _GlobalDecl):
                declared.update(stmt.names)
            lvalues: list[object] = []
            if isinstance(stmt, _AssignmentStmt):
                lvalues = list(stmt.lvalues)
            elif isinstance(stmt, _OpAssign):
                lvalues = [stmt.lvalue]
            elif isinstance(stmt, _ForStmt):
                lvalues = [stmt.index]
            elif isinstance(stmt, _WithStmt):
                lvalues = [target for target in stmt.target if target is not None]
            found: list[tuple[str, str, str]] = []
            # An annotated binding declares the type, which every binding
            # is checked against: such a target is never "bound again".
            annotated = isinstance(stmt, _AssignmentStmt) and stmt.unanalyzed_type is not None
            for lvalue in lvalues:
                for leaf in _flat(lvalue):
                    if isinstance(leaf, _MemberExpr) or (isinstance(leaf, NameExpr) and level == "class"):
                        key = (owner, leaf.name)
                        attrs[key] = attrs.get(key, 0) + (_DECLARED if annotated else 1)
                        found.append(("attr", owner, leaf.name))
                    elif isinstance(leaf, NameExpr):
                        if level == "module" or leaf.name in declared:
                            names[leaf.name] = names.get(leaf.name, 0) + (_DECLARED if annotated else 1)
                        found.append((level, owner, leaf.name))
            if isinstance(stmt, _AssignmentStmt):
                values[id(stmt.rvalue)] = found
            for inner in _nested_bodies(stmt):
                _walk(inner, level, owner)

    _walk(getattr(tree, "defs", None), "module")
    if len(_TARGET_INDEX) > 64:
        _TARGET_INDEX.clear()
    index = (attrs, names, declared, values)
    _TARGET_INDEX[id(tree)] = (tree, index)
    return index


def _value_of_rebound_target(ctx: MethodContext | FunctionContext, expr: object) -> bool:
    """Whether `expr` is the whole value of an attribute or a module name bound elsewhere too.

    _value_of_rebound_name for the bindings it does not see: mypy types
    `self.Q = np.array([[1.0, 0.0], [0.0, 1.0]])` of an `__init__` from
    that binding, so a method's `self.Q = np.eye(n)` is refused, and a
    module's `g = np.array(0.0)` refuses a function's `global g; g =
    np.zeros(n)`. Flow-insensitive: another binding of `<anything>.Q`
    in the same class or outside every class, or of `g` at module level
    or under `global g`, counts. A target with an annotated binding is
    declared, and every binding is checked against the declaration.
    """
    try:
        tree = getattr(ctx.api, "tree", None)
        if tree is None:
            return False
        attrs, names, declared, values = _target_index(tree)
        for kind, owner, name in values.get(id(expr), []):
            if kind == "attr":
                count = attrs.get((owner, name), 0) + (attrs.get(("", name), 0) if owner else 0)
            elif kind == "module":
                count = names.get(name, 0) + (2 if name in declared else 0)
            else:
                continue
            if 1 < count < _DECLARED:
                return True
        return False
    except Exception:
        _swallowed(ctx, "_value_of_rebound_target")
        return False
