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
    from mypy.plugin import FunctionContext, FunctionSigContext, MethodContext


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
