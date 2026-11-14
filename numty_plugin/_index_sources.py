"""_index_sources.py — sizes and records an index reads from its source code
(issue 333): the size of a slice whose bounds or step are sizes, and the
value a name was given by a tuple target (`r, c = a, b`). Kept out of
_indexing.py, which is near the 2000-line limit. Top-level imports go to
earlier layers only (tests/test_layout.py, _LAYERS); a function-local
facade import is used only for a monkeypatch seam (_LAZY_SEAMS).
"""
from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from mypy.nodes import AssignmentStmt, Block, Expression, ListExpr, NameExpr, StarExpr, TupleExpr
from mypy.types import Type

from ._errors import _trial_failed
from ._forms import _LinForm, _MonoForm, _canon_div, _form_has_unknown, _form_mul, _lin_add, _lin_scale

if TYPE_CHECKING:
    from mypy.plugin import FunctionContext, MethodContext


def _never_negative(form: _LinForm) -> bool:
    """A linear form that is at least 0 for all sizes: no negative coefficient or constant."""
    return form.const >= 0 and all(c >= 0 for _, c in form.coeffs)


def _fits_long_axis(form: _LinForm) -> bool:
    """At least 0 once the sizes are large enough, with no name subtracted.

    `M-1` qualifies (the "axis long enough" assumption `x[1:]` makes too);
    `M-N` does not (N may be the larger), nor does `-1`.
    """
    if any(c < 0 for _, c in form.coeffs):
        return False
    return form.const >= 0 or any(c > 0 for _, c in form.coeffs)


class _FromEnd:
    """A slice bound written `-k` with `k` an `int` variable (`x[:-k]`, issue 359).

    It counts from the end, as a negative literal does, when `k` is
    positive (`x[:-0]` is empty, which nothing about `?k` rules out: such
    a size is an unknown, compared only under the numty-names code).
    """

    __slots__ = ("dim",)

    def __init__(self, dim: Type) -> None:
        self.dim = dim


def _is_size_bound(bound: object) -> bool:
    """A slice bound that is a size, or one counted from the end."""
    return isinstance(bound, (Type, _FromEnd))


def _symbolic_slice(
    ctx: MethodContext | FunctionContext, dim: int, size: int | Type, begin: object, end: object, stride: object
) -> int | Type | None:
    """The size of `a[b:e]` with a bound that is a size, else None.

    `e - b`, with the axis size for a missing end and a negative literal
    counted from the end. This holds when the bounds are inside the
    axis; numpy clips them otherwise. So the answer is given when an
    `int` variable is left in it (such a size is an unknown, nothing is
    proved from it, and it is compared only under the numty-names code:
    `N-?k` against `N-?k+1` are 1 apart; `x[i:i + 1]` is 1, and `x[:-k]`
    is `N-?k`), or when the bounds are over
    names alone and inside the axis (issue 333): `s[:n]` with
    `n = X.shape[1]` on an `M+1` axis is `(M,)`, since `0 <= M <= M+1`.
    As for `x[1:]`, the axis is assumed long enough for a constant
    (`x[0:n - 1]` is `(M-1,)` for `n: M`), but a name is never assumed
    larger than another: `x[:n]` with `n: N` on an `M` axis is unknown.
    """
    from numty_plugin import _materialize_dim, _resolve_dim

    if not _is_size_bound(begin) and not _is_size_bound(end):
        return None
    if stride not in ("missing", 1):
        return None
    whole = _resolve_dim(size)
    if not isinstance(whole, _LinForm):
        return None

    def position(bound: object, missing: _LinForm) -> _LinForm | None:
        if bound == "missing":
            return missing
        if isinstance(bound, bool):
            return None
        if isinstance(bound, int):
            return _LinForm((), bound) if bound >= 0 else _lin_add(whole, _LinForm((), bound))
        if isinstance(bound, Type):
            form = _resolve_dim(bound)
            return form if isinstance(form, _LinForm) else None
        if isinstance(bound, _FromEnd):
            form = _resolve_dim(bound.dim)
            if not isinstance(form, _LinForm) or not _form_has_unknown(form):
                return None
            return _lin_add(whole, _lin_scale(form, -1))
        return None

    lo = position(begin, _LinForm((), 0))
    hi = position(end, whole)
    if lo is None or hi is None:
        return None
    form = _lin_add(hi, _lin_scale(lo, -1))
    # An `int` variable in a bound (`x[i:i + 1]`, `R[s:s + len(z)]`, issue
    # 359) is assumed inside the axis, as for `x[m:]`, also where it
    # cancels out of the size.
    if not (_form_has_unknown(form) or _form_has_unknown(lo) or _form_has_unknown(hi)):
        inside = _lin_add(whole, _lin_scale(hi, -1))
        if not (_fits_long_axis(lo) and _fits_long_axis(form) and _fits_long_axis(inside)):
            return None
    return _materialize_dim(ctx, "slice", dim, form)


def _strided_count(ctx: MethodContext | FunctionContext, dim: int, size: int | Type, step: Type) -> int | Type | None:
    """The size of `a[::s]` for a step `s` that is a size, else None (issue 333).

    numpy takes `ceil(L / s)` items of an axis of size `L`. Two cases are
    exact over names: `L` a multiple of `s` (`L / s`), and `L = (s-1)**2`,
    where `ceil` is `s-1` for every `s >= 1`: the flat diagonal idiom
    `m.reshape(c, -1)[:, ::f + 1]` of an `(f, f)` block is `(c, f)`.
    """
    from numty_plugin import _materialize_dim, _resolve_dim

    whole, s = _resolve_dim(size), _resolve_dim(step)
    # A step of 0 raises, so a step over names is at least 1 where the call runs.
    if not isinstance(s, _LinForm) or not _never_negative(s) or _form_has_unknown(s) or s == _LinForm((), 0):
        return None
    if not isinstance(whole, (_LinForm, _MonoForm)) or _form_has_unknown(whole):
        return None
    quotient = _canon_div(whole, s)
    if quotient is not None:
        return _materialize_dim(ctx, "slice", dim, quotient)
    less = _lin_add(s, _LinForm((), -1))
    if _form_mul(less, less) == whole:
        return _materialize_dim(ctx, "slice", dim, less)
    return None


def _tuple_targets(stmts: object, out: dict[int, list[AssignmentStmt]]) -> None:
    """Collect the assignments with a tuple target, by line, at any depth.

    A plain walk: mypy's compiled `TraverserVisitor` cannot be subclassed
    from a plugin. Statement bodies are reached through the attributes
    mypy's nodes keep them in (functions, classes, branches, loops,
    `with`, `try`, decorators).
    """
    if isinstance(stmts, Block):
        stmts = stmts.body
    if not isinstance(stmts, list):
        return
    for stmt in stmts:
        if isinstance(stmt, AssignmentStmt) and any(isinstance(lv, (TupleExpr, ListExpr)) for lv in stmt.lvalues):
            out.setdefault(stmt.line, []).append(stmt)
        for attr in ("body", "else_body", "finally_body", "defs", "handlers", "bodies"):
            inner = getattr(stmt, attr, None)
            for block in inner if isinstance(inner, list) else [inner]:
                if isinstance(block, Block):
                    _tuple_targets(block, out)
        func = getattr(stmt, "func", None)
        if func is not None and isinstance(getattr(func, "body", None), Block):
            _tuple_targets(func.body, out)


_TARGETS: dict[int, tuple[object, dict[int, list[AssignmentStmt]]]] = {}


def _tuple_target_value(ctx: object, node: object) -> Expression | None:
    """The value a tuple target gave a variable, from a display on the right (issue 333).

    `r, c = np.array([...]), np.arange(5)` gives `r` the first item, so
    `r` holds none of the positions `np.arange(5)` does. None when the
    variable was not bound by a tuple target, or the value is no display
    of as many items.
    """
    tree = getattr(getattr(ctx, "api", None), "tree", None)
    line = getattr(node, "line", 0) or 0
    if tree is None or line <= 0:
        return None
    cached = _TARGETS.get(id(tree))
    if cached is None or cached[0] is not tree:
        by_line: dict[int, list[AssignmentStmt]] = {}
        _tuple_targets(getattr(tree, "defs", None), by_line)
        cached = (tree, by_line)
        _TARGETS[id(tree)] = cached
    for stmt in cached[1].get(line, []):
        rvalue = stmt.rvalue
        if not isinstance(rvalue, (TupleExpr, ListExpr)):
            continue
        for lv in stmt.lvalues:
            if not isinstance(lv, (TupleExpr, ListExpr)) or len(lv.items) != len(rvalue.items):
                continue
            if any(isinstance(i, StarExpr) for i in (*lv.items, *rvalue.items)):
                continue
            for target, value in zip(lv.items, rvalue.items):
                if isinstance(target, NameExpr) and target.node is node:
                    return value
    return None


@contextmanager
def _past_overload_trial(ctx: object) -> Generator[None]:
    """Report the errors of an index hook past mypy's overload trial (issue 333).

    A key with a part that is not an array (`range(k)`, a list, a slice)
    fits several `__getitem__` overloads, so mypy tries them with errors
    filtered and drops a candidate that raised one: an error the hook
    reports made every candidate fail, and mypy added `[index] Invalid
    index type` to it. The hook's errors are about the shapes, not about
    which overload fits, so while a trial that has not failed is running
    they go past its watchers (as `_report_unsolved` does). A trial whose
    arguments already failed is one mypy throws away: nothing changes.
    """
    errors = getattr(getattr(getattr(ctx, "api", None), "msg", None), "errors", None)
    watchers = getattr(errors, "_watchers", None)
    if not isinstance(watchers, list) or not watchers or _trial_failed(ctx):
        yield
        return
    setattr(errors, "_watchers", [])
    try:
        yield
    finally:
        setattr(errors, "_watchers", watchers)
