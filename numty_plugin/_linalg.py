"""numpy.linalg hooks (Phase 3, #197). Top-level imports go to earlier layers
only (tests/test_layout.py, _LAYERS); a function-local facade import is used
only for a monkeypatch seam (_LAZY_SEAMS).
NUMTY/_bail/_swallowed from _errors direct. No direct _fresh_unknown
(preserves facade patch seam).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._errors import NUMTY_AXIS, NUMTY_BROADCAST, NUMTY_SHAPE, _swallowed
from ._dtypes import (
    _NAME_OF_KIND_SIZE,
)
from ._types import (
    TUPLE_FULLNAME, _check_dim_eq, _dim_str, _has_unpack, _int_value, _ndarray_parts, _ndarray_type,
)
from ._reshape import (
    _all_positionals, _dims_str, _dtype_fullname, _dtype_kind_size, _fold_flag, _formal_pair,
    _is_none_type, _normalize_axis, _parse_axis_arg, _promote_all, _promote_dtypes, _weak_name,
)
from ._builders import (
    _fold_single,
)

if TYPE_CHECKING:
    from mypy.plugin import FunctionContext
    from mypy.plugin import FunctionSigContext
    from mypy.plugin import MethodContext
    from mypy.types import FunctionLike
    

# Runtime-only (bodies check isinstance / read arg kinds).
from mypy.types import AnyType, CallableType, TupleType, Type, TypeOfAny
from ._reshape import (
    _construct_dtype,
)
from ._dtypes import (
    _DTYPE_INFO,
)
from mypy.types import (
    Instance, LiteralType, get_proper_type,
)

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
    from numty_plugin import _broadcast_result
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
    rows, cols = dims[-2], dims[-1]
    if not isinstance(rows, int) or not isinstance(cols, int):
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
    # _linalg_dtype returns `dtype` when the name is its own fullname;
    # a None name is that same case.
    uv_dtype = dtype if uv_name is None else _linalg_dtype(ctx, dtype, uv_name)
    s_dtype = _linalg_dtype(ctx, dtype, s_name)
    if uv_dtype is None or s_dtype is None:
        return ctx.default_return_type
    batch = dims[:-2]
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


def _norm_signature_hook(ctx: FunctionSigContext) -> FunctionLike:
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


def _linalg_cross_signature_hook(ctx: FunctionSigContext) -> FunctionLike:
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
    from numty_plugin import _broadcast_result
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
    # Issue #183: fail on a literal vector size that can never be 3,
    # even when the other side is symbolic (e.g. 2 vs P, where 2 != 3
    # always raises whatever P is). Both-symbolic (N vs N) stays silent:
    # N could be 3, and cross-position conflicts (N must be 3 for the
    # vector but 2 for the broadcast) need follow-up reasoning.
    if isinstance(vec_a, int) and vec_a != 3:
        ctx.api.fail(
            f"cross requires 3-dimensional vectors, got {vec_a} and {_dim_str(vec_b)} "
            f"(shapes {_dims_str(dims_a)} vs {_dims_str(dims_b)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    if isinstance(vec_b, int) and vec_b != 3:
        ctx.api.fail(
            f"cross requires 3-dimensional vectors, got {_dim_str(vec_a)} and {vec_b} "
            f"(shapes {_dims_str(dims_a)} vs {_dims_str(dims_b)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
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
    from numty_plugin import _broadcast_result
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


def _string_value(tp: Type) -> str | None:
    """String of a literal mode/kind argument, else None."""
    proper = get_proper_type(tp)
    if isinstance(proper, LiteralType) and isinstance(proper.value, str):
        return proper.value
    if isinstance(proper, Instance) and proper.last_known_value is not None:
        return _string_value(proper.last_known_value)
    return None


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
