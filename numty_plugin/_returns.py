"""_returns.py — return-boundary dynamic dims (issues #269/#271).

A plain-`int` or bare-unknown dim in a directly returned array value (or in
an unannotated local that value is returned through) cannot be proven for
or against a declared named size, so at the `return` it is `Any`. Top-level
imports go to earlier layers only (order in tests/test_layout.py, _LAYERS).
"""

from __future__ import annotations

import ast
import os

from mypy.plugin import AttributeContext, FunctionContext, MethodContext
from mypy.types import (
    AnyType,
    Instance,
    TupleType,
    Type,
    TypeOfAny,
    TypeVarType,
    UnionType,
    get_proper_type,
)

from ._errors import _swallowed
from ._forms import _is_unknown_fullname
from ._types import INT_FULLNAME, NDARRAY_FULLNAME


def _is_dynamic_return_dim(d: int | Type) -> bool:
    """A dim the return-boundary rewrite turns into Any (issues #269/#271).

    Plain `int` (a builder's dynamic axis, e.g. `n` in `np.zeros(n)` or
    `len()` of a non-array) and a bare fresh unknown (e.g. a mask count)
    are unprovable sizes: at a `return` they are written as `Any`, which
    fits any declared named size. Literal ints (including `(0,)`) and
    named or formula TypeVars keep checking, so provable mismatches
    still fail.
    """
    if isinstance(d, int):
        return False
    if isinstance(d, Instance):
        return d.type.fullname == INT_FULLNAME and d.last_known_value is None
    return isinstance(d, TypeVarType) and _is_unknown_fullname(d.fullname)


_RETURN_SITES_CACHE: dict[str, tuple[float, list, list]] = {}


def _return_sites_of(path: str) -> tuple[list, list] | None:
    """`(returns, assigns)` of the file at `path`, each with a func stack.

    Parsed with stdlib `ast` (mypy's own visitor classes are compiled and
    cannot be subclassed from interpreted plugin code). Entries are
    `(node, stack)` where `stack` is the tuple of enclosing
    `FunctionDef`/`AsyncFunctionDef` nodes, innermost last, so an
    assigned-then-returned name is scoped to its own function. Cached by
    modification time; any read or parse failure yields None (no rewrite,
    exactly today's behavior).
    """
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    hit = _RETURN_SITES_CACHE.get(path)
    if hit is not None and hit[0] == mtime:
        return hit[1], hit[2]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            mod = ast.parse(fh.read())
    except (OSError, SyntaxError, ValueError):
        return None
    returns: list = []
    assigns: list = []
    stack: list = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.append(node)
            for child in ast.iter_child_nodes(node):
                visit(child)
            stack.pop()
            return
        if isinstance(node, ast.Return):
            returns.append((node, tuple(stack)))
        elif isinstance(node, ast.Assign):
            assigns.append((node, tuple(stack)))
        for child in ast.iter_child_nodes(node):
            visit(child)

    try:
        visit(mod)
    except Exception:
        _swallowed(None, "_return_sites_of")
        return None
    _RETURN_SITES_CACHE[path] = (mtime, returns, assigns)
    return returns, assigns


def _ast_directly_returns(node: ast.AST, line: int, col: int) -> bool:
    """Whether the value `node` returns starts at `(line, col)`.

    The returned expression itself starts where the hooked call does;
    items of a returned tuple/list display and branches of a returned
    conditional do too. Anything nested deeper (an argument of a
    returned call, an operand of a returned operator) starts elsewhere,
    so a dynamic size there keeps its precise type.
    """
    if (getattr(node, "lineno", None), getattr(node, "col_offset", None)) == (
        line,
        col,
    ):
        return True
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(_ast_directly_returns(e, line, col) for e in node.elts)
    if isinstance(node, ast.Starred):
        return _ast_directly_returns(node.value, line, col)
    if isinstance(node, ast.IfExp):
        return _ast_directly_returns(node.body, line, col) or _ast_directly_returns(node.orelse, line, col)
    if isinstance(node, ast.NamedExpr):
        return _ast_directly_returns(node.value, line, col)
    return False


def _ast_returns_name(node: ast.AST, name: str) -> bool:
    """Whether the value `node` returns the name `name` (same positions)."""
    if isinstance(node, ast.Name):
        return node.id == name
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(_ast_returns_name(e, name) for e in node.elts)
    if isinstance(node, ast.Starred):
        return _ast_returns_name(node.value, name)
    if isinstance(node, ast.IfExp):
        return _ast_returns_name(node.body, name) or _ast_returns_name(node.orelse, name)
    if isinstance(node, ast.NamedExpr):
        return _ast_returns_name(node.value, name)
    return False


def _assigned_name_returned(returns: list, assigns: list, line: int, col: int) -> str | None:
    """The local name assigned from `(line, col)`, if it is returned.

    `out = <call>` (unannotated, plain names) followed by `return out`
    in the same function is the named twin of `return <call>`: the call
    is the value the return check sees, through the name. Returns the
    name, else None. Annotated assignments (`out: F64[N] = ...`, an
    `AnnAssign`, never a plain `Assign`) are left alone: the assignment
    itself checks those.
    """
    for assign, fstack in assigns:
        try:
            targets = assign.targets
            if not targets or not all(isinstance(t, ast.Name) for t in targets):
                continue
            names = [t.id for t in targets]
            value = assign.value
            if (value.lineno, value.col_offset) != (line, col):
                continue
            for ret, rfstack in returns:
                if fstack != rfstack:
                    continue
                if ret.value is not None and any(
                    _ast_returns_name(ret.value, name) for name in names
                ):
                    return names[0]
        except Exception:
            _swallowed(None, "_assigned_name_returned")
            continue
    return None


def _return_any_for_dynamic(ctx: FunctionContext | MethodContext | AttributeContext, tp: Type) -> Type:
    """`tp` with returned dynamic dims written as Any (issues #269/#271).

    A plain-`int` or bare-unknown dim in a directly returned array value
    (or in an unannotated local that value is returned through) cannot
    be proven for or against a declared named size, so it is `Any`,
    which fits any declared size. Runs on the hook result (or the kept
    stub default) in `_guard_mixed`, so every producer shares it; mypy
    still reports the check itself as [return-value] when a literal or
    named dim disagrees. Literals, named and formula TypeVars, unpacks
    and dtypes are untouched.
    """
    proper = get_proper_type(tp)
    if isinstance(proper, UnionType):
        members = [_return_any_for_dynamic(ctx, item) for item in proper.items]
        if all(new is old for new, old in zip(members, proper.items)):
            return tp
        try:
            return UnionType.make_union(members)
        except Exception:
            _swallowed(ctx, "_return_any_for_dynamic")
            return tp
    if not isinstance(proper, Instance) or proper.type.fullname != NDARRAY_FULLNAME or len(proper.args) != 2:
        return tp
    shape = get_proper_type(proper.args[0])
    if not isinstance(shape, TupleType):
        return tp
    if not any(_is_dynamic_return_dim(get_proper_type(item)) for item in shape.items):
        return tp
    try:
        tree = getattr(ctx.api, "tree", None)
        target = ctx.context
        path = getattr(tree, "path", None) if tree is not None else None
        line = getattr(target, "line", None)
        col = getattr(target, "column", None)
        if path is None or line is None or col is None:
            return tp
        sites = _return_sites_of(path)
        if sites is None:
            return tp
        returns, assigns = sites
        directly = any(ret.value is not None and _ast_directly_returns(ret.value, line, col) for ret, _ in returns)
        if not directly and _assigned_name_returned(returns, assigns, line, col) is None:
            return tp
    except Exception:
        _swallowed(ctx, "_return_any_for_dynamic")
        return tp
    new_items: list[Type] = [
        AnyType(TypeOfAny.explicit) if _is_dynamic_return_dim(get_proper_type(item)) else item for item in shape.items
    ]
    try:
        new_shape = shape.copy_modified(items=new_items)
    except Exception:
        _swallowed(ctx, "_return_any_for_dynamic")
        return tp
    try:
        return proper.copy_modified(args=[new_shape, proper.args[1]])
    except Exception:
        _swallowed(ctx, "_return_any_for_dynamic")
        return tp
