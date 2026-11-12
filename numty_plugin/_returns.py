"""_returns.py — return-boundary dynamic dims (issues #269/#271).

A plain-`int` dim in a directly returned array value (or in an
unannotated local that value is returned through) cannot be proven for
or against a declared named size, so at the `return` it is `Any`. Mask
counts and other unknowns keep checking: only the unprovable `int`
is rewritten. Top-level imports go to earlier layers only (order in
tests/test_layout.py, _LAYERS).
"""

from __future__ import annotations

import ast
import os
from collections.abc import Callable

from mypy.plugin import AttributeContext, FunctionContext, FunctionSigContext, MethodContext, MethodSigContext
from mypy.types import (
    AnyType,
    CallableType,
    Instance,
    TupleType,
    Type,
    TypeOfAny,
    TypeVarType,
    UnionType,
    get_proper_type,
)

from mypy.nodes import LambdaExpr
from mypy.types import LiteralType

from ._errors import _swallowed
from ._exprs import _EXPR_PREFIX, _iter_tvarlikes, _resolve_dim
from ._forms import _form_has_unknown, _is_unknown_fullname
from ._types import _OWN_NAME_OF, INT_FULLNAME, NDARRAY_FULLNAME


def _is_dynamic_return_dim(d: int | Type) -> bool:
    """A dim the return-boundary rewrite turns into Any (issues #269/#271).

    A plain `int` (a builder's dynamic axis, e.g. `n` in `np.zeros(n)` or
    `len()` of a non-array) is an unprovable size: at a `return` it is
    written as `Any`, which fits any declared named size. Literal ints
    (including `(0,)`), mask counts and other unknowns, and named or
    formula TypeVars keep checking, so provable mismatches still fail.
    """
    if isinstance(d, int):
        return False
    if isinstance(d, Instance):
        return d.type.fullname == INT_FULLNAME and d.last_known_value is None
    return False


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


_MODULE_AST_CACHE: dict[str, tuple[float, ast.Module]] = {}


def _module_ast(path: str) -> ast.Module | None:
    try:
        mtime = os.path.getmtime(path)
        hit = _MODULE_AST_CACHE.get(path)
        if hit is not None and hit[0] == mtime:
            return hit[1]
        with open(path, "r", encoding="utf-8") as fh:
            mod = ast.parse(fh.read())
    except (OSError, SyntaxError, ValueError):
        return None
    _MODULE_AST_CACHE[path] = (mtime, mod)
    return mod


def _holder_key(path: str | None, use_line: int, name: str) -> str | None:
    """A key for the object the variable `name` holds at `use_line`, else None.

    The key is the innermost function around the line and the line of
    the last binding of the name before it (0 for a parameter or a name
    from outside), so `ctx = ctx or Ctx(...)` starts a new object and
    the reads after it share one. Not flow-sensitive: with a binding
    inside a loop and another one anywhere, there is no key.
    """
    mod = _module_ast(path) if path else None
    if mod is None or use_line <= 0:
        return None
    func: ast.AST = mod
    found = True
    while found:
        found = False
        todo = list(ast.iter_child_nodes(func))
        while todo:
            node = todo.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.lineno <= use_line <= (node.end_lineno or node.lineno):
                    func, found = node, True
                    break
                continue
            todo.extend(ast.iter_child_nodes(node))
    stores: list[tuple[int, bool]] = []

    def walk(node: ast.AST, in_loop: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            if isinstance(child, ast.Name) and child.id == name and isinstance(child.ctx, (ast.Store, ast.Del)):
                stores.append((child.lineno, in_loop))
            walk(child, in_loop or isinstance(child, (ast.For, ast.AsyncFor, ast.While)))

    walk(func, False)
    args = getattr(func, "args", None)
    is_param = args is not None and any(
        a is not None and a.arg == name
        for a in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
    )
    if len(stores) + (1 if is_param else 0) > 1 and any(loop for _, loop in stores):
        return None
    last = max((line for line, _ in stores if line < use_line), default=0)
    return f"{getattr(func, 'name', '')}@{getattr(func, 'lineno', 0)}::{name}::{last}"


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


def _declared_return_shapes(ctx: object, rank: int) -> list[list[Type]]:
    """Shapes of the given rank that the function being checked declares it returns.

    Read from the checker's stack of declared return types. A union
    (`F64[N, int] | None`) gives each array member of that rank. Anything
    else (a tuple of arrays, no annotation, another rank) gives nothing.
    """
    stack = getattr(getattr(ctx, "api", None), "return_types", None)
    if not stack:
        return []
    declared = get_proper_type(stack[-1])
    members = declared.items if isinstance(declared, UnionType) else [declared]
    out: list[list[Type]] = []
    for member in members:
        proper = get_proper_type(member)
        if not isinstance(proper, Instance) or proper.type.fullname != NDARRAY_FULLNAME or len(proper.args) != 2:
            continue
        shape = get_proper_type(proper.args[0])
        if isinstance(shape, TupleType) and len(shape.items) == rank:
            out.append([get_proper_type(item) for item in shape.items])
    return out


def _same_named_size(a: Type, b: Type) -> bool:
    return isinstance(a, TypeVarType) and isinstance(b, TypeVarType) and a.fullname == b.fullname


_ANNOTATED_VALUES_CACHE: dict[str, tuple[ast.Module, set[tuple[int, int]]]] = {}


def _is_annotated_value(path: str, line: int, col: int) -> bool:
    """Whether the expression at `(line, col)` is the value of an annotated assignment (`x: T = <here>`)."""
    mod = _module_ast(path)
    if mod is None:
        return False
    hit = _ANNOTATED_VALUES_CACHE.get(path)
    if hit is None or hit[0] is not mod:
        spots = {
            (node.value.lineno, node.value.col_offset)
            for node in ast.walk(mod)
            if isinstance(node, ast.AnnAssign) and node.value is not None
        }
        hit = (mod, spots)
        _ANNOTATED_VALUES_CACHE[path] = hit
    return (line, col) in hit[1]


def _misplaced_name_axes(ctx: object, items: list[Type], shapes: list[list[Type]] | None = None) -> set[int]:
    """Axes whose plain `int` must not be written as Any: the value is transposed (issue #297).

    The rewrite lets `int` fit a declared name, and a name always fits a
    declared `int`, so `(int, N)` passed where `(N, int)` is declared: a
    matrix allocated with its axes swapped. An `int` axis keeps its type
    when the name declared there is a size of the value on another axis,
    one that is not declared with that name. mypy then reports the
    return as it did before the rewrite. `(int, N)` where `(N, N)` is
    declared still passes, and so does `(S, int)` where `(S, C)` is.
    """
    try:
        value = [get_proper_type(item) for item in items]
        if shapes is None:
            shapes = _declared_return_shapes(ctx, len(value))
        if not shapes:
            return set()
        per_shape: list[set[int]] = []
        for declared in shapes:
            axes: set[int] = set()
            for i, (have, want) in enumerate(zip(value, declared)):
                if not _is_dynamic_return_dim(have) or not isinstance(want, TypeVarType):
                    continue
                if any(
                    k != i and _same_named_size(value[k], want) and not _same_named_size(declared[k], want)
                    for k in range(len(value))
                ):
                    axes.add(i)
            per_shape.append(axes)
        # With several declared members, an axis is kept only when it
        # is misplaced for every one of them.
        return set.intersection(*per_shape)
    except Exception:
        _swallowed(None, "_misplaced_name_axes")
        return set()


_FREE_NAMES_CACHE: dict[int, tuple[object, dict[str, TypeVarType]]] = {}


def _free_return_names(ctx: object) -> dict[str, TypeVarType]:
    """Sizes the function being checked names in its return alone, by fullname (issue 307).

    `def bounds(seg: I64[N]) -> tuple[I64[K], I64[K]]`: no parameter
    binds `K`, so no caller chooses it. It is the function's to choose:
    a size born in the body. A name of the class, a name a parameter of
    an enclosing function binds, a formula and an unknown are not such
    names. A lambda has none.
    """
    scope = getattr(getattr(ctx, "api", None), "scope", None)
    func = scope.current_function() if scope is not None else None
    if func is None or isinstance(func, LambdaExpr):
        return {}
    hit = _FREE_NAMES_CACHE.get(id(func))
    if hit is not None and hit[0] is func:
        return hit[1]
    out: dict[str, TypeVarType] = {}
    try:
        tp = get_proper_type(func.type)
        if isinstance(tp, CallableType):
            bound: set[str] = set()
            for item in getattr(scope, "stack", []):
                item_tp = get_proper_type(getattr(item, "type", None))
                if isinstance(item_tp, CallableType):
                    for arg in item_tp.arg_types:
                        bound.update(tv.fullname for tv in _iter_tvarlikes(arg))
                defn = getattr(item, "defn", None)  # a class on the stack
                for tv in getattr(defn, "type_vars", None) or []:
                    bound.add(tv.fullname)
            info = getattr(func, "info", None)
            for tv in getattr(getattr(info, "defn", None), "type_vars", None) or []:
                bound.add(tv.fullname)
            for tv in _iter_tvarlikes(tp.ret_type):
                if (
                    isinstance(tv, TypeVarType)
                    and tv.fullname not in bound
                    and not _is_unknown_fullname(tv.fullname)
                    and not tv.fullname.startswith(_EXPR_PREFIX)
                    and not tv.values
                ):
                    upper = get_proper_type(tv.upper_bound)
                    if isinstance(upper, Instance) and upper.type.fullname == INT_FULLNAME:
                        out[tv.fullname] = tv
    except Exception:
        _swallowed(None, "_free_return_names")
        out = {}
    _FREE_NAMES_CACHE[id(func)] = (func, out)
    return out


def _is_literal_dim(d: Type) -> bool:
    if isinstance(d, LiteralType):
        return isinstance(d.value, int)
    return isinstance(d, Instance) and d.type.fullname == INT_FULLNAME and d.last_known_value is not None


def _is_born_dim(d: Type) -> bool:
    """A size with no name of the signature in it: `int`, an unknown, a formula over an unknown."""
    if _is_dynamic_return_dim(d):
        return True
    if not isinstance(d, TypeVarType):
        return False
    if _is_unknown_fullname(d.fullname):
        return True
    if d.fullname.startswith(_EXPR_PREFIX):
        form = _resolve_dim(d)
        return form is not None and _form_has_unknown(form)
    return False


def _array_shapes(tp: Type | None, rank: int) -> list[list[Type]]:
    """Shapes of the given rank among the array members of `tp` (an array, or a union with one)."""
    if tp is None:
        return []
    proper = get_proper_type(tp)
    members = proper.items if isinstance(proper, UnionType) else [proper]
    out: list[list[Type]] = []
    for member in members:
        inst = get_proper_type(member)
        if not isinstance(inst, Instance) or inst.type.fullname != NDARRAY_FULLNAME or len(inst.args) != 2:
            continue
        shape = get_proper_type(inst.args[0])
        if isinstance(shape, TupleType) and len(shape.items) == rank:
            out.append([get_proper_type(item) for item in shape.items])
    return out


def _return_positions(node: ast.AST, match: Callable[[ast.AST], bool]) -> list[int | None]:
    """Where the returned value `node` holds what `match` accepts: None for the value itself, else the tuple index."""
    if isinstance(node, ast.IfExp):
        return _return_positions(node.body, match) + _return_positions(node.orelse, match)
    if isinstance(node, ast.NamedExpr):
        return _return_positions(node.value, match)
    if match(node):
        return [None]
    if isinstance(node, ast.Tuple):
        return [i for i, elt in enumerate(node.elts) if match(elt)]
    return []


def _bound_once(func: ast.AST, name: str) -> bool:
    """Whether `name` is bound exactly once in the body of `func` (its own scope) and is not a parameter."""
    count = 0
    args = getattr(func, "args", None)
    if args is not None:
        every = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
        if any(a is not None and a.arg == name for a in every):
            return False
    todo = list(ast.iter_child_nodes(func))
    while todo:
        node = todo.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, (ast.Store, ast.Del)):
            count += 1
        todo.extend(ast.iter_child_nodes(node))
    return count == 1


# Sizes already given to a free name in one `return`, by position:
# (path, line and column of the return, name) -> {(tuple index, axis): size}.
_OWN_RETURN_SIZES: dict[tuple[str, int, int, str], dict[tuple[int, int], Type]] = {}


def _fits_first(key: tuple[str, int, int, str], where: tuple[int, int], dim: Type) -> bool:
    """Record `dim` for this position, and say whether it is the size the first position has.

    Every position a `return` declares with one free name must hold one
    size. The position checked first is the reference (a local that is
    returned, then the items of the `return` from the left). `int` is
    not followed, so it fits anything, as it fits a name a parameter
    binds.
    """
    seen = _OWN_RETURN_SIZES.setdefault(key, {})
    seen[where] = dim
    # The position recorded first, which keeps its place when it is
    # recorded again: the same reference in every pass.
    first = next(iter(seen))
    if first == where:
        return True
    ref = seen[first]
    if _is_dynamic_return_dim(ref) or _is_dynamic_return_dim(dim):
        return True
    return bool(ref == dim)


def _name_born_sizes(
    ctx: FunctionContext | MethodContext | AttributeContext | FunctionSigContext | MethodSigContext, tp: Type
) -> Type:
    """`tp` with the sizes born here written as the free name declared for them (issues 306, 307).

    A free name is a size the function names in its return alone (see
    `_free_return_names`). Where a value meets a position declared with
    one, a size with no name in it (`int`, a selection count, any other
    unknown) takes the name:

    - a value under a declared type: `ex: F64[E] = col.to_numpy()`, a
      directly returned call, an item of a returned tuple;
    - a call assigned to a plain local that is bound once and returned,
      `starts = np.flatnonzero(...)` then `return starts, ends`. The
      local then has the name for its whole life, so what is computed
      from it is in the name's terms: `np.r_[starts[1:], n]` is `(K,)`
      and `np.r_[starts, n]` is `(K+1,)`, which mypy refuses for `K`.

    In a `return`, a literal size takes the name too (`return
    np.zeros(0), np.zeros(0)`), and every position declared with one
    name must hold the size the first one checked holds; a position that
    does not keeps its own size, and mypy reports the return. A size with a
    name of the signature in it (`N`, `N+1`) is never renamed.
    """
    proper = get_proper_type(tp)
    if not isinstance(proper, Instance) or proper.type.fullname != NDARRAY_FULLNAME or len(proper.args) != 2:
        return tp
    shape = get_proper_type(proper.args[0])
    if not isinstance(shape, TupleType) or not shape.items:
        return tp
    free = _free_return_names(ctx)
    if not free:
        return tp
    items = [get_proper_type(item) for item in shape.items]
    if not any(_is_born_dim(item) or _is_literal_dim(item) for item in items):
        return tp
    try:
        tree = getattr(ctx.api, "tree", None)
        path = getattr(tree, "path", None) if tree is not None else None
        line = getattr(ctx.context, "line", None)
        col = getattr(ctx.context, "column", None)
        if path is None or line is None or col is None:
            return tp
        sites = _return_sites_of(path)
        if sites is None:
            return tp
        returns, assigns = sites

        def here(node: ast.AST) -> bool:
            return (getattr(node, "lineno", None), getattr(node, "col_offset", None)) == (line, col)

        # (declared shape, the return and tuple index it is a position of)
        declared: list[Type] | None = None
        direct: list[tuple[ast.Return, int]] = []
        for ret, _ in returns:
            if ret.value is not None:
                direct.extend((ret, -1 if p is None else p) for p in _return_positions(ret.value, here))
        stack = getattr(getattr(ctx.api, "expr_checker", None), "type_context", None)
        context = stack[-1] if stack else None
        shapes = _array_shapes(context, len(items))
        if len(shapes) == 1:
            declared = shapes[0]
        elif not direct:
            # A plain local, bound once, that is returned.
            ret_stack = getattr(ctx.api, "return_types", None)
            for assign, fstack in assigns:
                targets = assign.targets
                if len(targets) != 1 or not isinstance(targets[0], ast.Name) or not here(assign.value):
                    continue
                name = targets[0].id
                if not fstack or not ret_stack or not _bound_once(fstack[-1], name):
                    break
                want = get_proper_type(ret_stack[-1])
                for ret, rstack in returns:
                    if rstack != fstack or ret.value is None:
                        continue
                    for p in _return_positions(ret.value, lambda n: isinstance(n, ast.Name) and n.id == name):
                        if p is None:
                            found = _array_shapes(want, len(items))
                        elif isinstance(want, TupleType) and p < len(want.items):
                            found = _array_shapes(want.items[p], len(items))
                        else:
                            found = []
                        if len(found) == 1 and (declared is None or declared == found[0]):
                            declared = found[0]
                            direct.append((ret, -1 if p is None else p))
                break
        if declared is None:
            return tp
        new_items: list[Type] = list(shape.items)
        for axis, (have, want) in enumerate(zip(items, declared)):
            # By id: a callee's variable of the same name (the formal an
            # argument is checked under) is not this function's name.
            if not isinstance(want, TypeVarType) or want.fullname not in free or want.id != free[want.fullname].id:
                continue
            if not (_is_born_dim(have) or (direct and _is_literal_dim(have))):
                continue
            if all(
                _fits_first((path, ret.lineno, ret.col_offset, want.fullname), (p, axis), have) for ret, p in direct
            ):
                new_items[axis] = want
                if isinstance(have, TypeVarType) and _is_unknown_fullname(have.fullname):
                    # The name is this unknown: one count is not two
                    # sizes when it meets itself under the name.
                    _OWN_NAME_OF.setdefault(want.id, set()).add(have.fullname)
        if all(new is old for new, old in zip(new_items, shape.items)):
            return tp
        return proper.copy_modified(args=[shape.copy_modified(items=new_items), proper.args[1]])
    except Exception:
        _swallowed(ctx, "_name_born_sizes")
        return tp


def _return_any_for_dynamic(
    ctx: FunctionContext | MethodContext | AttributeContext | FunctionSigContext | MethodSigContext, tp: Type
) -> Type:
    """`tp` with returned dynamic dims written as Any (issues #269/#271).

    A plain-`int` dim in a directly returned array value (or in an
    unannotated local that value is returned through) cannot be proven
    for or against a declared named size, so it is `Any`, which fits
    any declared size. Runs on the hook result (or the kept stub
    default) in `_guard_mixed`, so every producer shares it; mypy
    still reports the check itself as [return-value] when a literal,
    an unknown, or a named dim disagrees. Literals, unknowns, named
    and formula TypeVars, unpacks and dtypes are untouched.
    """
    tp = _name_born_sizes(ctx, tp)
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
        # An annotated local (`out: F64[S, C] = np.full((len(steps), k), 0.0)`):
        # the annotation is the author's word for the `int` size, as a
        # declared return is. Only an axis declared with a name takes it.
        local: list[list[Type]] | None = None
        if not directly and _assigned_name_returned(returns, assigns, line, col) is None:
            if not _is_annotated_value(path, line, col):
                return tp
            stack = getattr(getattr(ctx.api, "expr_checker", None), "type_context", None)
            local = _array_shapes(stack[-1] if stack else None, len(shape.items))
            if len(local) != 1:
                return tp
    except Exception:
        _swallowed(ctx, "_return_any_for_dynamic")
        return tp
    keep = _misplaced_name_axes(ctx, list(shape.items), local)
    if local is not None:
        keep |= {i for i, want in enumerate(local[0]) if not isinstance(want, TypeVarType)}
    new_items: list[Type] = [
        AnyType(TypeOfAny.explicit) if i not in keep and _is_dynamic_return_dim(get_proper_type(item)) else item
        for i, item in enumerate(shape.items)
    ]
    if all(new is old for new, old in zip(new_items, shape.items)):
        return tp
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


def _return_sig_for_dynamic(sig: CallableType, ctx: FunctionSigContext | MethodSigContext) -> CallableType:
    """`sig` for a call that is returned, with the result's plain `int` sizes as Any.

    `return np.zeros(n)` fits a declared `F64[N]` through the rewrite of
    the hook's result. `return helper(n)`, with `helper` a function of
    the project that returns `F64[int]`, has no hook for its result, so
    it was still an error: the same value passed or failed by where it
    was built. The call's signature is where its result type can be
    reached, so the rewrite is done on the return type there, at the
    same sites (a direct `return`, or one unannotated local that is
    returned).
    """
    try:
        ret = _return_any_for_dynamic(ctx, sig.ret_type)
        if ret is sig.ret_type:
            return sig
        return sig.copy_modified(ret_type=ret)
    except Exception:
        _swallowed(ctx, "_return_sig_for_dynamic")
        return sig
