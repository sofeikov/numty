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

from mypy.nodes import (
    AssignmentStmt, Block, ClassDef, FuncDef, IndexExpr, LambdaExpr, ListExpr, NameExpr, TupleExpr, Var,
)
from mypy.types import LiteralType

from ._errors import _swallowed
from ._exprs import _EXPR_PREFIX, _implicit_scope_vars, _iter_tvarlikes, _ns_of_ctx, _resolve_dim
from ._forms import _form_has_unknown, _is_unknown_fullname
from ._types import _OWN_NAME_OF, INT_FULLNAME, NDARRAY_FULLNAME
from ._unknowns import _AST_REPEATED, _module_ast


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


def _holder_key(path: str | None, use_line: int, name: str) -> str | None:
    """A key for the object the variable `name` holds at `use_line`, else None.

    The key is the innermost function around the line and the line of
    the last binding of the name before it (0 for a parameter or a name
    from outside), so `ctx = ctx or Ctx(...)` starts a new object and
    the reads after it share one. Not flow-sensitive: with a binding
    inside a loop and another one anywhere, there is no key. A binding
    in a loop or a comprehension holds an object per run: the key then
    ends with `::loop` (issue 328).
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
            walk(child, in_loop or isinstance(child, _AST_REPEATED))

    walk(func, False)
    args = getattr(func, "args", None)
    is_param = args is not None and any(
        a is not None and a.arg == name
        for a in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
    )
    if len(stores) + (1 if is_param else 0) > 1 and any(loop for _, loop in stores):
        return None
    last = max((line for line, _ in stores if line < use_line), default=0)
    loop = "::loop" if any(in_loop for _, in_loop in stores) else ""
    return f"{getattr(func, 'name', '')}@{getattr(func, 'lineno', 0)}::{name}::{last}{loop}"


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
    """Whether the expression at `(line, col)` is the value of an annotated assignment (`x: T = <here>`).

    Also of a plain assignment to a name its function declared on an
    earlier line (`d: F64[N, N]` then `d = <here>`, or a parameter
    rebound, issue 359): mypy checks it under the declared type too, and
    the annotation means the same wherever it is written.
    """
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
        for func in ast.walk(mod):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            declared = _annotated_names(func) | {
                arg.arg
                for arg in (*func.args.posonlyargs, *func.args.args, *func.args.kwonlyargs)
                if arg.annotation is not None
            }
            todo: list[ast.AST] = list(func.body)
            while todo:
                node = todo.pop()
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                    continue
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in declared
                ):
                    spots.add((node.value.lineno, node.value.col_offset))
                todo.extend(ast.iter_child_nodes(node))
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


def _signature_names(scope: object, func: object) -> set[str]:
    """Fullnames of the names a parameter (of `func` or an enclosing function) or a class binds."""
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
    return bound


_BODY_NAMES_CACHE: dict[int, tuple[object, set[str]]] = {}


def _is_body_name(ctx: object, want: TypeVarType) -> bool:
    """Whether `want` is a size the function being checked names in its body only (issues 321, 330).

    `Uhat: F64[n_basis, n_rank]` in a function whose signature binds
    neither: the plugin binds such a name for the body (`_heads`), once
    per function, and keeps the object by (function, name). Like a name
    of the return alone it is the function's to choose, so a size born
    in the body (an unknown, an `int`) takes it where it is declared. A
    name a parameter binds is the caller's and never does.
    """
    scope = getattr(getattr(ctx, "api", None), "scope", None)
    func = scope.current_function() if scope is not None else None
    if func is None or isinstance(func, LambdaExpr) or not isinstance(want, TypeVarType):
        return False
    if _is_unknown_fullname(want.fullname) or want.fullname.startswith(_EXPR_PREFIX) or want.values:
        return False
    upper = get_proper_type(want.upper_bound)
    if not (isinstance(upper, Instance) and upper.type.fullname == INT_FULLNAME):
        return False
    ns = _ns_of_ctx(ctx)
    cached = _implicit_scope_vars.get((ns, want.fullname)) or _implicit_scope_vars.get((ns, want.name))
    if cached is None or cached.fullname != want.fullname or cached.id != want.id:
        return False
    hit = _BODY_NAMES_CACHE.get(id(func))
    if hit is None or hit[0] is not func:
        hit = (func, _signature_names(scope, func))
        _BODY_NAMES_CACHE[id(func)] = hit
    return want.fullname not in hit[1]


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
            bound = _signature_names(scope, func)
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


def _annotated_names(func: ast.AST) -> set[str]:
    """Names given an annotation (`clock: F64[E] = ...`, `clock: F64[E]`) in the body of `func`, its own scope."""
    out: set[str] = set()
    todo = list(ast.iter_child_nodes(func))
    while todo:
        node = todo.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
        todo.extend(ast.iter_child_nodes(node))
    return out


def _value_matches(node: ast.AST, match: Callable[[ast.AST], bool]) -> bool:
    """Whether the bound value `node` is what `match` accepts, or a branch of a conditional that is."""
    if isinstance(node, ast.IfExp):
        return _value_matches(node.body, match) or _value_matches(node.orelse, match)
    if isinstance(node, ast.NamedExpr):
        return _value_matches(node.value, match)
    return match(node)


def _const_index(node: ast.AST, length: int) -> int | None:
    """The item a constant subscript `[i]` (or `[-i]`) picks from a tuple of `length` items, else None."""
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        sign, node = -1, node.operand
    if not isinstance(node, ast.Constant) or not isinstance(node.value, int) or isinstance(node.value, bool):
        return None
    index = sign * node.value
    if index < 0:
        index += length
    return index if 0 <= index < length else None


def _bound_names(
    assign: ast.Assign,
    match: Callable[[ast.AST], bool],
    whole: Callable[[ast.AST], bool],
    item: int | None,
    length: int,
) -> list[str]:
    """Plain names `assign` binds to the value `match` accepts (issue 332).

    `v = <value>` and its conditional (`v = a if c else b`); `v, z =
    <value>, 0`, item by item; and, for the item `item` of a hooked
    tuple (`whole` accepts the hooked call itself), `t, _ = pick(x)`.
    """
    if len(assign.targets) != 1:
        return []
    target, value = assign.targets[0], assign.value
    if isinstance(target, ast.Name):
        return [target.id] if _value_matches(value, match) else []
    if not isinstance(target, (ast.Tuple, ast.List)) or any(isinstance(e, ast.Starred) for e in target.elts):
        return []
    if isinstance(value, (ast.Tuple, ast.List)):
        if len(value.elts) != len(target.elts) or any(isinstance(e, ast.Starred) for e in value.elts):
            return []
        return [
            elt.id
            for elt, sub in zip(target.elts, value.elts)
            if isinstance(elt, ast.Name) and _value_matches(sub, match)
        ]
    if item is not None and len(target.elts) == length and whole(value):
        elt = target.elts[item]
        return [elt.id] if isinstance(elt, ast.Name) else []
    return []


def _shapes_at(declared: Type, p: int | None, rank: int) -> list[list[Type]]:
    """Array shapes of the given rank declared at a position of a return.

    The position is None for the value itself, else the tuple index. A
    union of tuples (a function that also returns a null system of
    rank-1 arrays) gives the shapes each member declares there.
    """
    if p is None:
        return _array_shapes(declared, rank)
    proper = get_proper_type(declared)
    members = proper.items if isinstance(proper, UnionType) else [proper]
    out: list[list[Type]] = []
    for member in members:
        tup = get_proper_type(member)
        if isinstance(tup, TupleType) and p < len(tup.items):
            out.extend(shape for shape in _array_shapes(tup.items[p], rank) if shape not in out)
    return out


# Sizes already given to a free name in one `return`, by position:
# (path, line and column of the return, name) -> {position: size}. A
# position is (tuple index, axis) for an item of the `return` itself,
# and (tuple index, axis, line, column, item) for a binding of a local
# that is returned there: each binding holds a size of its own.
_OWN_RETURN_SIZES: dict[tuple[str, int, int, str], dict[tuple[int, ...], Type]] = {}


def _fits_first(key: tuple[str, int, int, str], where: tuple[int, ...], dim: Type) -> bool:
    """Record `dim` for this position, and say whether it is the size the first position has.

    Every position a `return` declares with one free name must hold one
    size, and so must every binding of a local returned there. The
    position checked first is the reference (a local that is returned,
    then the items of the `return` from the left). `int` is not
    followed, so it fits anything, as it fits a name a parameter binds,
    and it is not the reference: `int`, then two selection counts, is
    still two sizes.
    """
    seen = _OWN_RETURN_SIZES.setdefault(key, {})
    seen[where] = dim
    if _is_dynamic_return_dim(dim):
        return True
    # The position recorded first, which keeps its place when it is
    # recorded again: the same reference in every pass.
    first = next((w for w, d in seen.items() if not _is_dynamic_return_dim(d)), where)
    if first == where:
        return True
    return bool(seen[first] == dim)


def _plain_tuple(proper: TupleType) -> bool:
    """A tuple of arrays: a plain one, or a NamedTuple result (`SVDResult`, issue 330)."""
    fallback = get_proper_type(proper.partial_fallback)
    return isinstance(fallback, Instance) and (
        fallback.type.fullname == "builtins.tuple" or fallback.type.is_named_tuple
    )


def _assign_of(block: Block, value: object) -> AssignmentStmt | None:
    """The assignment in `block` (nested blocks too, not nested functions) whose value is
    `value` or a subscript of it."""
    todo: list[object] = [block]
    while todo:
        node = todo.pop()
        if isinstance(node, Block):
            todo.extend(node.body)
        elif isinstance(node, list):
            todo.extend(node)
        elif isinstance(node, AssignmentStmt):
            rvalue = node.rvalue
            if rvalue is value or (isinstance(rvalue, IndexExpr) and rvalue.base is value):
                return node
        elif not isinstance(node, (FuncDef, ClassDef)):
            for attr in ("body", "else_body", "finally_body", "handlers", "bodies"):
                sub = getattr(node, attr, None)
                if isinstance(sub, (Block, list)):
                    todo.append(sub)
    return None


def _target_declared(
    ctx: FunctionContext | MethodContext | AttributeContext | FunctionSigContext | MethodSigContext, item: int
) -> Type | None:
    """The declared type of the local bound to item `item` of the hooked value, else None.

    By a tuple target (`u, s, vt = svd(a)`) or a constant subscript
    (`s = svd(a)[1]`): mypy checks either with no type context.
    """
    scope = getattr(ctx.api, "scope", None)
    func = scope.current_function() if scope is not None else None
    if not isinstance(func, FuncDef):
        return None
    stmt = _assign_of(func.body, ctx.context)
    if stmt is None or len(stmt.lvalues) != 1:
        return None
    elt = stmt.lvalues[0]
    if isinstance(elt, (TupleExpr, ListExpr)):
        if stmt.rvalue is not ctx.context or item >= len(elt.items):
            return None
        elt = elt.items[item]
    if not isinstance(elt, NameExpr) or not isinstance(elt.node, Var) or elt.node.is_inferred:
        return None
    return elt.node.type


def _name_born_sizes(
    ctx: FunctionContext | MethodContext | AttributeContext | FunctionSigContext | MethodSigContext, tp: Type
) -> Type:
    """`tp` with the sizes born here written as the free name declared for them (issues 306, 307, 332).

    A free name is a size the function names in its return alone (see
    `_free_return_names`). Where a value meets a position declared with
    one, a size with no name in it (`int`, a selection count, any other
    unknown) takes the name:

    - a value under a declared type: `ex: F64[E] = col.to_numpy()`, a
      directly returned call, an item of a returned tuple;
    - a value bound to a plain local that is returned, `starts =
      np.flatnonzero(...)` then `return starts, ends`. The local then
      has the name, so what is computed from it is in the name's terms:
      `np.r_[starts[1:], n]` is `(K,)` and `np.r_[starts, n]` is
      `(K+1,)`, which mypy refuses for `K`. The local may be bound more
      than once (a rebound parameter, both branches of an `if`, a
      conditional, a tuple target): every binding must hold the size
      the first one holds, and one that does not keeps its own size, so
      mypy reports it there (issue 332).

    A tuple (`np.nonzero(m)`, a call of a function that returns two
    arrays) is followed into the item a constant subscript or a tuple
    target picks (issue 332).

    In a `return`, a literal size takes the name too (`return
    np.zeros(0), np.zeros(0)`), and every position declared with one
    name must hold the size the first one checked holds; a position that
    does not keeps its own size, and mypy reports the return. A size with a
    name of the signature in it (`N`, `N+1`) is never renamed.

    A name the body alone declares (`_is_body_name`) is taken the same
    way by a local annotated with it, also through a tuple target
    (`Uhat, s, Vt = np.linalg.svd(B)`, issue 330).
    """
    proper = get_proper_type(tp)
    if isinstance(proper, TupleType):
        if not proper.items or not _plain_tuple(proper):
            return tp
        try:
            items = [
                _name_born_array(ctx, item, index, len(proper.items)) for index, item in enumerate(proper.items)
            ]
            if all(new is old for new, old in zip(items, proper.items)):
                return tp
            return proper.copy_modified(items=items)
        except Exception:
            _swallowed(ctx, "_name_born_sizes")
            return tp
    return _name_born_array(ctx, tp, None, 0)


def _name_born_array(
    ctx: FunctionContext | MethodContext | AttributeContext | FunctionSigContext | MethodSigContext,
    tp: Type,
    item: int | None,
    length: int,
) -> Type:
    """`_name_born_sizes` for one array: the hooked value, or its item `item` of `length` items."""
    proper = get_proper_type(tp)
    if not isinstance(proper, Instance) or proper.type.fullname != NDARRAY_FULLNAME or len(proper.args) != 2:
        return tp
    shape = get_proper_type(proper.args[0])
    if not isinstance(shape, TupleType) or not shape.items:
        return tp
    free = _free_return_names(ctx)
    items = [get_proper_type(i) for i in shape.items]
    if not any(_is_born_dim(i) or (free and _is_literal_dim(i)) for i in items):
        # Without a name of the return alone, only a size born here can
        # take a name: one the body declares (issue 330).
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
            if (getattr(node, "lineno", None), getattr(node, "col_offset", None)) == (line, col):
                return True
            # mypy places an operator at its left operand inside the
            # parentheses: `(S**2) / n` is where `S**2` is, one column
            # past the `(` (issue 330).
            while isinstance(node, ast.BinOp):
                node = node.left
                if (node.lineno, node.col_offset) == (line, col):
                    return True
            return False

        def whole(node: ast.AST) -> bool:
            # A tuple display starts where its first item does.
            return here(node) and not isinstance(node, (ast.Tuple, ast.List))

        def match(node: ast.AST) -> bool:
            if item is None:
                return whole(node)
            return (
                isinstance(node, ast.Subscript)
                and whole(node.value)
                and _const_index(node.slice, length) == item
            )

        ret_stack = getattr(ctx.api, "return_types", None)
        want_ret = get_proper_type(ret_stack[-1]) if ret_stack else None
        # (declared shape, and the positions it is declared at: the
        # return, its tuple index, and the binding for a local)
        declared: list[Type] | None = None
        uses: list[tuple[ast.Return, int, tuple[int, ...]]] = []
        for ret, _ in returns:
            if ret.value is not None:
                uses.extend((ret, -1 if p is None else p, ()) for p in _return_positions(ret.value, match))
        stack = getattr(getattr(ctx.api, "expr_checker", None), "type_context", None)
        context = stack[-1] if stack else None
        shapes = _array_shapes(context, len(items))
        if uses:
            if len(shapes) == 1:
                declared = shapes[0]
            elif item is not None and want_ret is not None:
                # `return np.nonzero(m)[0]`: the call is checked with no
                # context, so the declared type is read from the return.
                for ret, p, _ in uses:
                    found = _shapes_at(want_ret, None if p < 0 else p, len(items))
                    if len(found) != 1 or (declared is not None and declared != found[0]):
                        return tp
                    declared = found[0]
        else:
            bound: list[tuple[str, tuple]] = []
            for assign, fstack in assigns:
                bound.extend((name, fstack) for name in _bound_names(assign, match, whole, item, length))
            # A plain local is checked under the type its first binding
            # gave it: that is no declaration, so only a local the
            # function annotates takes a name from its context.
            plain = [(name, fstack) for name, fstack in bound if fstack and name not in _annotated_names(fstack[-1])]
            if not plain:
                if len(shapes) == 1:
                    declared = shapes[0]
                elif item is not None and bound:
                    # `Uhat, s, Vt = np.linalg.svd(B)` into annotated
                    # locals: mypy checks a tuple target with no context,
                    # so the declared type is read off the target.
                    found = _array_shapes(_target_declared(ctx, item), len(items))
                    if len(found) == 1:
                        declared = found[0]
            elif want_ret is not None:
                site = (line, col, -1 if item is None else item)
                for name, fstack in plain:
                    for ret, rstack in returns:
                        if rstack != fstack or ret.value is None:
                            continue
                        for p in _return_positions(ret.value, lambda n: isinstance(n, ast.Name) and n.id == name):
                            found = _shapes_at(want_ret, p, len(items))
                            if len(found) == 1 and (declared is None or declared == found[0]):
                                declared = found[0]
                                uses.append((ret, -1 if p is None else p, site))
        if declared is None:
            return tp
        new_items: list[Type] = list(shape.items)
        for axis, (have, want) in enumerate(zip(items, declared)):
            # By id: a callee's variable of the same name (the formal an
            # argument is checked under) is not this function's name.
            if not isinstance(want, TypeVarType):
                continue
            if want.fullname in free:
                if want.id != free[want.fullname].id:
                    continue
            elif not _is_body_name(ctx, want):
                continue
            if not (_is_born_dim(have) or (uses and _is_literal_dim(have))):
                continue
            if all(
                _fits_first((path, ret.lineno, ret.col_offset, want.fullname), (p, axis, *site), have)
                for ret, p, site in uses
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
