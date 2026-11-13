"""_matrix.py — square-matrix builders and readers on symbolic shapes:
eye/identity, diag/diagonal, outer and cov (issue #330). Top-level
imports go to earlier layers only (order in tests/test_layout.py,
_LAYERS). A function-local facade import is used only for a monkeypatch
seam (_LAZY_SEAMS).
"""
from __future__ import annotations

from mypy.nodes import Expression, ListExpr, StarExpr, TupleExpr
from mypy.plugin import FunctionContext, MethodContext
from mypy.types import AnyType, Instance, Type, TypeVarType, get_proper_type

from ._errors import NUMTY_AXIS, NUMTY_SHAPE, _swallowed
from ._types import _dim_add_const, _dim_eq, _ndarray_parts, _ndarray_type
from ._reshape import (
    _all_positionals, _construct_dtype, _dims_str, _dtype_kind_size, _fold_flag, _formal_pair, _int_or_none,
    _is_none_type, _normalize_axis, _picture_complete,
)
from ._builders import _builder_shape, _explicit_dtype, _fold_single, _named_dtype, _unknown_dtype
from ._matmul import _matmul_promoted, _two_arrays
from ._matmul import outer_function_hook as _outer_literal_hook
from ._reductions import _flatten_total_dim
from ._ops import _hook_pair, _rank_check_hook


def _size_arg(
    ctx: FunctionContext, pair: tuple[Type, Expression | None]
) -> int | Type | None:
    """One builder size (eye's N or M, identity's n), else None to bail.

    A literal folds to an int; otherwise the size is read like a 1-d
    builder shape: a size TypeVar (`X.shape[1]`), arithmetic over sizes
    (`k + 1`), or a plain `int` that stays `int`.
    """
    value = _fold_single(pair)
    if value is not None:
        return value
    tp, expr = pair
    if isinstance(expr, (TupleExpr, ListExpr, StarExpr)):
        return None
    dims = _builder_shape(ctx, tp, expr, ctx.default_return_type, shaped_stub=False)
    if dims is None or len(dims) != 1:
        return None
    return dims[0]


def _builder_dtype(ctx: FunctionContext) -> Type | None:
    """eye/identity dtype: the named dtype=, else the stubs' own, else float64.

    A dtype= the plugin cannot name (`x.dtype`) is still resolved by the
    stubs; only their shape is lost on a symbolic size, so their dtype
    is reused.
    """
    if not _explicit_dtype(ctx):
        return _construct_dtype(ctx, "numpy.float64")
    named = _named_dtype(ctx) or _unknown_dtype(ctx)
    if named is not None:
        return named
    default = get_proper_type(ctx.default_return_type)
    if isinstance(default, Instance) and len(default.args) == 2:
        if _dtype_kind_size(default.args[1]) is not None:
            return default.args[1]
    return None


def _square_builder(ctx: FunctionContext, op: str, n: int | Type, m: int | Type) -> Type:
    if (isinstance(n, int) and n < 0) or (isinstance(m, int) and m < 0):
        ctx.api.fail(
            f"{op} dimensions must be non-negative, got {_dims_str([n, m])}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    dtype = _builder_dtype(ctx)
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, [n, m], dtype)


def eye_function_hook(ctx: FunctionContext) -> Type:
    """np.eye(N, M=None, k=0): (N, M), or (N, N) without M; k never resizes.

    N and M may be literals, size TypeVars or arithmetic over sizes, so
    `np.eye(X.shape[1])` is (d, d) and `np.eye(k + 1)` is (d+1, d+1).
    """
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    n = _size_arg(ctx, pairs[0])
    if n is None:
        return ctx.default_return_type
    m_pair = _formal_pair(ctx, "M")
    if m_pair is None or _is_none_type(m_pair[0]):
        m: int | Type | None = n
    else:
        m = _size_arg(ctx, m_pair)
    if m is None:
        return ctx.default_return_type
    return _square_builder(ctx, "eye", n, m)


def identity_function_hook(ctx: FunctionContext) -> Type:
    """np.identity(n): (n, n), with eye's size rules (issue #58, #330)."""
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    n = _size_arg(ctx, pairs[0])
    if n is None:
        return ctx.default_return_type
    return _square_builder(ctx, "identity", n, n)


def _offset(ctx: FunctionContext | MethodContext, name: str, index: int) -> int | None:
    """A literal diagonal offset (k= / offset=), 0 when absent, else None."""
    if isinstance(ctx, FunctionContext):
        found = _hook_pair(ctx, "v" if name == "k" else "a", name, index)
        if isinstance(found, str):
            return None
    else:
        found = _formal_pair(ctx, name)
    value = _int_or_none(found)
    if value is None:
        return 0
    return value if isinstance(value, int) else None


def _diagonal_length(
    ctx: FunctionContext | MethodContext, op: str, n: int | Type, m: int | Type, k: int
) -> int | Type:
    """Length of the k-th diagonal of an (n, m) matrix.

    Literal sizes give numpy's exact count (0 when the offset leaves
    the matrix). A square (n, n) main diagonal is n. Anything else
    (non-square symbolic sizes, an offset on a symbolic size) is
    min(...) clamped at 0, which the dim algebra cannot spell: a fresh
    unknown.
    """
    from numty_plugin import _fresh_unknown
    if isinstance(n, int) and isinstance(m, int):
        rows, cols = (n, m - k) if k >= 0 else (n + k, m)
        return max(0, min(rows, cols))
    if k == 0 and _dim_eq(n, m):
        return n
    return _fresh_unknown(ctx, op, 0)


def _diag_solve(ctx: FunctionContext, array: Type) -> Type:
    parts = _ndarray_parts(array)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    k = _offset(ctx, "k", 1)
    if k is None:
        return ctx.default_return_type
    if len(dims) == 1:
        # A vector becomes the k-th diagonal of a square (n+|k|, n+|k|).
        size = dims[0] if k == 0 else _dim_add_const(ctx, "diag", 0, dims[0], abs(k))
        return _ndarray_type(ctx, [size, size], dtype)
    if len(dims) == 2:
        return _ndarray_type(ctx, [_diagonal_length(ctx, "diag", dims[0], dims[1], k)], dtype)
    return ctx.default_return_type


def diag_function_hook(ctx: FunctionContext) -> Type:
    """np.diag: 1-d (n,) -> (n, n); 2-d (n, m) -> its k-th diagonal.

    Other known ranks always raise (issue #183).
    """
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    if not _rank_check_hook(ctx, pairs[0][0], "diag", min_rank=1, max_rank=2):
        return ctx.default_return_type
    try:
        return _diag_solve(ctx, pairs[0][0])
    except Exception:
        _swallowed(ctx, "diag_function_hook")
        return ctx.default_return_type


def _diagonal_solve(ctx: FunctionContext | MethodContext, array: Type, first: int) -> Type:
    """diagonal: drop axis1/axis2 and append the diagonal's length.

    `first` is the positional index of offset (1 for np.diagonal(a, ...),
    0 for a.diagonal(...)). Out-of-range or equal axes always raise.
    """
    parts = _ndarray_parts(array)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    if len(dims) < 2:
        return ctx.default_return_type
    if isinstance(ctx, FunctionContext) and not _picture_complete(ctx, "a"):
        return ctx.default_return_type
    k = _offset(ctx, "offset", first)
    if k is None:
        return ctx.default_return_type
    axes: list[int] = []
    for name, index, default in (("axis1", first + 1, 0), ("axis2", first + 2, 1)):
        found = (
            _hook_pair(ctx, "a", name, index) if isinstance(ctx, FunctionContext) else _formal_pair(ctx, name)
        )
        if isinstance(found, str):
            return ctx.default_return_type
        value = _int_or_none(found)
        if value is None:
            value = default
        if not isinstance(value, int):
            return ctx.default_return_type
        norm = _normalize_axis(value, len(dims))
        if norm is None:
            ctx.api.fail(
                f"diagonal {name} {value} is out of range for {len(dims)} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        axes.append(norm)
    if axes[0] == axes[1]:
        ctx.api.fail(
            f"diagonal axis1 and axis2 cannot be the same (both {axes[0]})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    length = _diagonal_length(ctx, "diagonal", dims[axes[0]], dims[axes[1]], k)
    rest = [d for i, d in enumerate(dims) if i not in axes]
    return _ndarray_type(ctx, [*rest, length], dtype)


def diagonal_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    if not _rank_check_hook(ctx, pairs[0][0], "diagonal", min_rank=2):
        return ctx.default_return_type
    try:
        return _diagonal_solve(ctx, pairs[0][0], 1)
    except Exception:
        _swallowed(ctx, "diagonal_function_hook")
        return ctx.default_return_type


def diagonal_method_hook(ctx: MethodContext) -> Type:
    if not _rank_check_hook(ctx, ctx.type, "diagonal", min_rank=2):
        return ctx.default_return_type
    try:
        return _diagonal_solve(ctx, ctx.type, 0)
    except Exception:
        _swallowed(ctx, "diagonal_method_hook")
        return ctx.default_return_type


def _outer_size(ctx: FunctionContext, dims: list[int | Type], index: int) -> int | Type | None:
    """np.outer ravels each operand: a vector keeps its size, else the product."""
    if len(dims) == 1:
        return dims[0]
    return _flatten_total_dim(ctx, dims, "outer")


def outer_function_hook(ctx: FunctionContext) -> Type:
    """np.outer(a, b): (a.size, b.size), each operand raveled.

    Literal shapes keep their solver (_matmul); symbolic ones give
    (n, m) for vectors and the size product for higher ranks.
    """
    pair = _two_arrays(ctx)
    if pair is None:
        return ctx.default_return_type
    parts_a, parts_b = _ndarray_parts(pair[0]), _ndarray_parts(pair[1])
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    (dims_a, dtype_a), (dims_b, dtype_b) = parts_a, parts_b
    if all(isinstance(d, int) for d in [*dims_a, *dims_b]):
        return _outer_literal_hook(ctx)
    size_a = _outer_size(ctx, dims_a, 0)
    size_b = _outer_size(ctx, dims_b, 1)
    if size_a is None or size_b is None:
        return ctx.default_return_type
    return _matmul_promoted(ctx, [size_a, size_b], dtype_a, dtype_b)


def _cov_dtype(ctx: FunctionContext, dtype: Type) -> Type | None:
    """np.cov computes in result_type(m, float64): complex stays complex128."""
    if _explicit_dtype(ctx):
        return _named_dtype(ctx)
    ks = _dtype_kind_size(dtype)
    if ks is None:
        return None
    kind, size = ks
    if kind in ("b", "i", "u") or (kind == "f" and size <= 8):
        return _construct_dtype(ctx, "numpy.float64")
    if kind == "c" and size <= 16:
        return _construct_dtype(ctx, "numpy.complex128")
    return None


def cov_function_hook(ctx: FunctionContext) -> Type:
    """np.cov(m): rows are variables, so (d, n) -> (d, d); rowvar=False
    reads columns, so (n, d) -> (d, d). A vector is one variable: ().

    numpy squeezes the result, so a literal single variable is () as
    well. y=, a non-literal rowvar and ranks above 2 (which raise) are
    left to the stubs or the rank check.
    """
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    if not _rank_check_hook(ctx, pairs[0][0], "cov", max_rank=2):
        return ctx.default_return_type
    try:
        if not _picture_complete(ctx, "m"):
            return ctx.default_return_type
        y_pair = _hook_pair(ctx, "m", "y", 1)
        if isinstance(y_pair, str) or (y_pair is not None and not _is_none_type(y_pair[0])):
            return ctx.default_return_type
        rowvar_pair = _hook_pair(ctx, "m", "rowvar", 2)
        if isinstance(rowvar_pair, str):
            return ctx.default_return_type
        rowvar = True if rowvar_pair is None else _fold_flag(rowvar_pair)
        if rowvar is None:
            return ctx.default_return_type
        parts = _ndarray_parts(pairs[0][0])
        if parts is None:
            return ctx.default_return_type
        dims, dtype = parts
        out_dtype = _cov_dtype(ctx, dtype)
        if out_dtype is None or isinstance(get_proper_type(out_dtype), AnyType):
            return ctx.default_return_type
        if len(dims) < 2:
            return _ndarray_type(ctx, [], out_dtype)
        size = dims[0] if rowvar else dims[1]
        if isinstance(size, int) and size == 1:
            return _ndarray_type(ctx, [], out_dtype)
        if not isinstance(size, (int, TypeVarType)):
            return ctx.default_return_type
        return _ndarray_type(ctx, [size, size], out_dtype)
    except Exception:
        _swallowed(ctx, "cov_function_hook")
        return ctx.default_return_type
