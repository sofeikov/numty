"""Runtime behavior of the numty helpers: shapes, values, narrowing."""
import dataclasses
import os
import subprocess
import sys
import typing
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from numty import F32, F64, I32, I64, Bool, NarrowError, batch, ident, matmul, narrow

ROOT = Path(__file__).resolve().parent.parent


@dataclasses.dataclass
class Batch:
    x: F64["batch features"]
    mask: Bool[4]


def test_zeros_have_expected_shapes() -> None:
    assert np.zeros((2, 3)).shape == (2, 3)
    assert np.zeros((3, 2)).shape == (3, 2)
    assert np.zeros((3, 4)).shape == (3, 4)


def test_zeros_are_float64() -> None:
    assert np.zeros((2, 3)).dtype == np.dtype(np.float64)


def test_matmul_computes() -> None:
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.arange(12, dtype=np.float64).reshape(3, 4)
    got = matmul(a, b)
    assert got.shape == (2, 4)
    np.testing.assert_allclose(got, np.matmul(a, b))


def test_matmul_of_known_shapes_has_chained_shape() -> None:
    assert matmul(np.zeros((2, 3)), np.zeros((3, 4))).shape == (2, 4)


def test_matmul_solves_unblessed_shapes_at_runtime() -> None:
    assert matmul(np.zeros((2, 3)), np.zeros((3, 2))).shape == (2, 2)


def test_add_broadcasts_ones_at_runtime() -> None:
    assert (np.zeros((2, 4)) + np.zeros((2, 1))).shape == (2, 4)


def test_end_to_end_matmul_plus_bias() -> None:
    assert (matmul(np.zeros((2, 3)), np.zeros((3, 4))) + np.zeros((2, 4))).shape == (2, 4)
    assert (np.zeros((2, 4)) * 2.0).shape == (2, 4)


def test_narrow_accepts_matching() -> None:
    m = np.zeros((2, 3))
    got = narrow(m, np.zeros((2, 3)))
    assert got is m
    assert got.shape == (2, 3)


def test_narrow_rejects_mismatch() -> None:
    with pytest.raises(NarrowError, match=r"shape \(3, 2\) does not match witness \(2, 3\)"):
        narrow(np.zeros((3, 2)), np.zeros((2, 3)))


def test_narrow_rejects_dtype_mismatch() -> None:
    with pytest.raises(NarrowError, match="dtype float32 does not match witness float64"):
        narrow(np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3)))


def test_narrow_error_is_value_and_assertion_error() -> None:
    # ValueError is the documented type; AssertionError keeps callers of
    # the old assert-based check working.
    with pytest.raises(ValueError):
        narrow(np.zeros((3, 2)), np.zeros((2, 3)))
    with pytest.raises(AssertionError):
        narrow(np.zeros((3, 2)), np.zeros((2, 3)))


@pytest.mark.parametrize(
    "call, message",
    [
        (
            "narrow(np.zeros((3, 2)), np.zeros((2, 3)))",
            "shape (3, 2) does not match witness (2, 3)",
        ),
        (
            "narrow(np.zeros((2, 3), dtype=np.int64), np.zeros((2, 3)))",
            "dtype int64 does not match witness float64",
        ),
    ],
)
def test_narrow_rejects_mismatch_under_O(call: str, message: str) -> None:
    # python -O strips asserts; narrow must still refuse the array.
    code = f"import numpy as np\nfrom numty import narrow\nassert False\n{call}\n"
    proc = subprocess.run(
        [sys.executable, "-O", "-c", code],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "NarrowError" in proc.stderr
    assert message in proc.stderr


@pytest.mark.parametrize(
    "head, dtype",
    [(F64, np.float64), (F32, np.float32), (I64, np.int64), (I32, np.int32), (Bool, np.bool_)],
)
def test_isinstance_against_head_checks_ndarray_and_dtype(head: Any, dtype: Any) -> None:
    assert isinstance(np.zeros(3, dtype=dtype), head)
    assert isinstance(np.zeros((2, 3, 4), dtype=dtype), head)
    assert not isinstance(np.zeros(3, dtype=np.complex128), head)
    assert not isinstance([0.0, 1.0], head)
    assert not isinstance(dtype(0), head)


def test_isinstance_picks_array_branch() -> None:
    def describe(x: F64[3] | list[float]) -> str:
        if isinstance(x, F64):
            return "array"
        return "list"

    assert describe(np.zeros(3)) == "array"
    assert describe([1.0, 2.0, 3.0]) == "list"


def test_head_is_not_constructible() -> None:
    with pytest.raises(TypeError, match="shape annotation"):
        F64()


def test_heads_still_work_as_runtime_annotations() -> None:
    def f(x: F64["batch features"], y: F64[batch, 3], z: I64[3]) -> None: ...

    hints = typing.get_type_hints(f)
    assert hints["x"] is F64
    assert typing.get_origin(hints["y"]) is F64
    assert typing.get_args(hints["y"]) == (batch, 3)
    assert typing.get_origin(hints["z"]) is I64
    assert isinstance(np.zeros((2, 3)), typing.get_origin(hints["y"]))
    assert not isinstance(np.zeros((2, 3)), typing.get_origin(hints["z"]))

    assert [f.name for f in dataclasses.fields(Batch)] == ["x", "mask"]
    batch_hints = typing.get_type_hints(Batch)
    assert batch_hints["x"] is F64
    assert typing.get_origin(batch_hints["mask"]) is Bool
    assert typing.get_args(batch_hints["mask"]) == (4,)


def test_ident_returns_same_object() -> None:
    a = np.zeros((2, 3))
    assert ident(a) is a


def test_reshape_keeps_values() -> None:
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    np.testing.assert_array_equal(a.reshape((3, 2)), np.arange(6, dtype=np.float64).reshape(3, 2))
    assert a.reshape(3, 2).shape == (3, 2)
    assert np.reshape(a, (6,)).shape == (6,)


def test_transpose_reorders() -> None:
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    np.testing.assert_array_equal(a.transpose(), a.T)
    assert a.transpose(1, 0).shape == (3, 2)
    np.testing.assert_array_equal(np.transpose(a, (1, 0)), a.T)


def test_squeeze_and_expand_dims() -> None:
    assert np.zeros((1, 2, 1, 3)).squeeze().shape == (2, 3)
    assert np.zeros((1, 2, 3)).squeeze(0).shape == (2, 3)
    assert np.expand_dims(np.zeros((3,)), 0).shape == (1, 3)
    assert np.expand_dims(np.zeros((2, 3)), (0, -1)).shape == (1, 2, 3, 1)


def test_reductions_drop_or_keep() -> None:
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    np.testing.assert_allclose(a.sum(axis=1), np.array([3.0, 12.0]))
    assert a.sum(axis=1, keepdims=True).shape == (2, 1)
    assert a.mean(axis=(0, 1)).shape == ()
    assert isinstance(a.sum(), np.float64)


def test_like_copies_shape_and_dtype() -> None:
    proto = np.zeros((2, 3), dtype=np.float32)
    got = np.zeros_like(proto)
    assert got.shape == (2, 3) and got.dtype == np.dtype(np.float32)
    assert np.zeros_like(proto, shape=(3, 2)).shape == (3, 2)


def test_full_eye_arange_values() -> None:
    assert np.full((2, 3), 5.0).shape == (2, 3)
    assert np.full((2, 3), 5.0).dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(np.eye(2, 3), np.eye(2, 3))
    np.testing.assert_array_equal(np.arange(0, 10, 3), [0, 3, 6, 9])


def test_comparisons_broadcast_bool() -> None:
    r = np.zeros((2, 3)) == np.zeros((2, 1))
    assert r.shape == (2, 3) and r.dtype == np.dtype(bool)
    assert (np.arange(3) < 2).tolist() == [True, True, False]


def test_arg_family_shapes() -> None:
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    np.testing.assert_array_equal(a.argmax(axis=1), [2, 2])
    assert a.argsort(axis=1).shape == (2, 3)
    np.testing.assert_array_equal(np.sort(a[:, ::-1], axis=1), a)
    assert np.partition(a.ravel(), 2)[:3].shape == (3,)


def test_list_stack_values() -> None:
    a = np.zeros((2, 3))
    assert np.stack([a, a], axis=0).shape == (2, 2, 3)
    assert np.concatenate([a, np.zeros((2, 4))], axis=1).shape == (2, 7)


def test_rng_shapes() -> None:
    g = np.random.default_rng(0)
    assert g.normal(size=(2, 3)).shape == (2, 3)
    assert g.normal(size=(2, 3)).dtype == np.dtype(np.float64)
    got = g.integers(0, 10, size=(4,))
    assert got.shape == (4,) and np.issubdtype(got.dtype, np.integer)
    assert g.choice(np.zeros((5,)), size=(3,)).shape == (3,)
    assert g.permutation(np.zeros((4, 2))).shape == (4, 2)


def test_subscript_tracks_shapes() -> None:
    a = np.arange(12, dtype=np.float64).reshape(4, 3)
    assert a[0].shape == (3,)
    assert a[:, 1].shape == (4,)
    assert a[1:3].shape == (2, 3)
    assert a[::2].shape == (2, 3)
    assert a[None, :, :].shape == (1, 4, 3)
    assert a[..., 1].shape == (4,)
    assert a[-1].shape == (3,)
    assert a[1:99].shape == (3, 3)
    np.testing.assert_array_equal(a[1:3], np.arange(12, dtype=np.float64).reshape(4, 3)[1:3])
    assert isinstance(a[0, 1], np.float64)


def test_promotion_matches_numpy() -> None:
    import numty_plugin

    names = [
        "bool",
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float16", "float32", "float64",
        "complex64", "complex128",
    ]
    for a in names:
        for b in names:
            want = "numpy." + np.result_type(a, b).name
            assert numty_plugin._promote_strong("numpy." + a, "numpy." + b) == want, (a, b)
    weak_cases: list[tuple[str, object]] = [
        ("builtins.int", 1),
        ("builtins.float", 1.5),
        ("builtins.complex", 1j),
        ("builtins.bool", True),
    ]
    for scalar, value in weak_cases:
        for b in names:
            want = "numpy." + np.result_type(value, np.dtype(b)).name
            assert numty_plugin._promote_weak(scalar, "numpy." + b) == want, (scalar, b)


def test_concat_stack_values() -> None:
    a = np.zeros((2, 3))
    b = np.zeros((2, 4))
    assert np.concatenate((a, b), axis=1).shape == (2, 7)
    assert np.stack((a, a), axis=0).shape == (2, 2, 3)


def test_matmul_broadcasts_batch_dims() -> None:
    assert matmul(np.zeros((1, 2, 3)), np.zeros((5, 3, 4))).shape == (5, 2, 4)
    assert matmul(np.zeros((5, 2, 3)), np.zeros((3, 4))).shape == (5, 2, 4)


def test_reshape_minus1_infers() -> None:
    a = np.zeros((2, 3))
    assert a.reshape((3, -1)).shape == (3, 2)
    assert a.reshape(-1).shape == (6,)


def test_einsum_values() -> None:
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.arange(12, dtype=np.float64).reshape(3, 4)
    got = np.einsum("ij,jh->ih", a, b)
    assert got.shape == (2, 4)
    np.testing.assert_allclose(got, a @ b)
    assert np.einsum("ij,j->i", a, np.zeros(3)).shape == (2,)
    assert np.einsum("ij,jh", a, b).shape == (4, 2)
    assert np.einsum("ii", np.eye(3)).shape == ()
    assert np.einsum("ii->i", np.eye(3)).shape == (3,)
    assert np.einsum("bij,bj->bi", np.zeros((7, 2, 3)), np.zeros((7, 3))).shape == (7, 2)
    assert np.einsum("...ij,j->...i", np.zeros((7, 2, 3)), np.zeros(3)).shape == (7, 2)


def test_linalg_shapes_and_dtypes() -> None:
    a = np.eye(4)
    assert np.linalg.solve(a, np.ones(4)).shape == (4,)
    assert np.linalg.solve(a, np.ones((4, 2))).shape == (4, 2)
    assert np.linalg.inv(a).shape == (4, 4)
    assert np.linalg.inv(np.eye(2, dtype=np.int64)).dtype == np.dtype(np.float64)
    vals, vecs = np.linalg.eig(a)
    assert vals.shape == (4,) and vals.dtype == np.dtype(np.complex128)
    assert vecs.shape == (4, 4) and vecs.dtype == np.dtype(np.complex128)
    w, v = np.linalg.eigh(a)
    assert w.shape == (4,) and w.dtype == np.dtype(np.float64)
    assert v.shape == (4, 4)
    u, s, vh = np.linalg.svd(np.zeros((5, 3)))
    assert (u.shape, s.shape, vh.shape) == ((5, 5), (3,), (3, 3))
    u, s, vh = np.linalg.svd(np.zeros((5, 3)), full_matrices=False)
    assert (u.shape, s.shape, vh.shape) == ((5, 3), (3,), (3, 3))
    assert np.linalg.svd(np.zeros((5, 3)), compute_uv=False).shape == (3,)
    sol, _, _, _ = np.linalg.lstsq(np.eye(3), np.ones(3), rcond=None)
    assert sol.shape == (3,)
    assert np.linalg.norm(np.zeros((2, 3)), axis=1).shape == (2,)
    assert np.linalg.norm(np.zeros((2, 3)), axis=0).shape == (3,)
    assert np.linalg.norm(np.zeros((2, 3)), axis=(0, 1), keepdims=True).shape == (1, 1)
    assert np.linalg.norm(np.eye(2, dtype=np.int32), axis=0).dtype == np.dtype(np.float64)


def test_ufunc_values_and_dtypes() -> None:
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    i = np.arange(6).reshape(2, 3)
    f32 = np.zeros((2, 3), dtype=np.float32)
    assert np.sqrt(a).shape == (2, 3)
    assert np.sqrt(i).dtype == np.dtype(np.float64)
    assert np.exp(a).dtype == np.dtype(np.float64)
    assert np.sin(a).shape == (2, 3)
    assert np.floor(i).dtype == i.dtype
    assert np.abs(-i).dtype == i.dtype
    assert np.abs(np.array([-1 + 1j])).dtype == np.dtype(np.float64)
    assert np.isfinite(a).dtype == np.dtype(np.bool_)
    assert np.real(np.zeros(3, dtype=np.complex64)).dtype == np.dtype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        for name in [
            "sqrt", "cbrt", "exp", "exp2", "expm1", "log", "log2", "log10",
            "log1p", "sin", "cos", "tan", "arcsin", "arccos", "arctan",
            "sinh", "cosh", "tanh", "rint", "sinc",
        ]:
            fn = getattr(np, name)
            assert fn(i).dtype == np.dtype(np.float64), name
            assert fn(a).dtype == np.dtype(np.float64), name
            assert fn(f32).dtype == np.dtype(np.float32), name
    for name in ["square", "floor", "ceil", "trunc", "sign", "conjugate", "positive", "negative"]:
        fn = getattr(np, name)
        assert fn(i).dtype == i.dtype, name
        assert fn(a).dtype == a.dtype, name
    for name in ["isfinite", "isinf", "isnan", "signbit", "logical_not"]:
        assert getattr(np, name)(a).dtype == np.dtype(np.bool_), name
    assert np.add(a, a).shape == (2, 3)
    assert (a - a).shape == (2, 3)
    assert (i - i).dtype == np.dtype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        assert (a / a).dtype == np.dtype(np.float64)
        assert (i / i).dtype == np.dtype(np.float64)
    assert (a**2).dtype == np.dtype(np.float64)
    assert (i**2).dtype == np.dtype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        assert np.divide(i, i).dtype == np.dtype(np.float64)
    assert np.hypot(a, a).dtype == np.dtype(np.float64)
    assert np.equal(a, a).dtype == np.dtype(np.bool_)
    assert np.minimum(a, 1.0).shape == (2, 3)
    w = np.where(a > 2, a, -1.0)
    assert w.shape == (2, 3) and w.dtype == np.dtype(np.float64)
    assert np.where(a > 2, i, 1).dtype == np.dtype(np.int64)
    assert np.clip(a, 0.0, 5.0).shape == (2, 3)
    assert a.clip(0.0, 5.0).shape == (2, 3)
    assert np.clip(i, 0, 5).dtype == np.dtype(np.int64)
    assert np.broadcast_to(np.zeros(3), (2, 3)).shape == (2, 3)
    assert np.asarray(a).shape == (2, 3)
    assert np.matmul(a, np.arange(12, dtype=np.float64).reshape(3, 4)).shape == (2, 4)


def test_named_dtype_and_helpers() -> None:
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    assert np.zeros((2, 3), dtype=np.float32).dtype == np.dtype(np.float32)
    assert np.asarray(a, dtype=np.float32).dtype == np.dtype(np.float32)
    assert np.asarray(a, dtype="float32").dtype == np.dtype(np.float32)
    assert a.astype(np.float32).dtype == np.dtype(np.float32)
    assert np.array([105.0, 68.0]).shape == (2,)
    assert np.array([105.0, 68.0]).dtype == np.dtype(np.float64)
    assert np.array([1, 2, 3]).dtype == np.dtype(np.int64)
    assert (-a).shape == (2, 3) and (+a).shape == (2, 3)
    assert len(a) == 2
    assert a.shape == (2, 3)
    assert np.broadcast_to(np.zeros(3), a.shape).shape == (2, 3)


def test_fancy_index_and_take_values() -> None:
    a = np.zeros((5, 10, 2))
    ids = np.zeros((5, 3), dtype=np.int64)
    assert np.take_along_axis(a, ids[..., None], axis=1).shape == (5, 3, 2)
    assert np.take_along_axis(np.zeros(10), np.zeros(5, dtype=np.int64), axis=None).shape == (5,)
    assert (~np.zeros((5, 10), dtype=np.bool_)).dtype == np.dtype(np.bool_)
    b = np.zeros((5, 22))
    actor = np.zeros(5, dtype=np.int64)
    assert b[np.arange(5), actor].shape == (5,)
    assert b[actor, 1:3].shape == (5, 2)
    assert b[:, actor].shape == (5, 5)
    assert np.zeros((7, 4))[np.zeros(5, dtype=np.int64)].shape == (5, 4)
    assert a[ids].shape == (5, 3, 10, 2)
    assert a[ids[:, 0]].shape == (5, 10, 2)
    assert np.zeros((5, 6))[np.array(3)].shape == (6,)


def test_numpy_hook_table_matches_installed_numpy() -> None:
    # The plugin dispatches on exact stub fullnames (#59). If a numpy
    # release moves a function, its hook would silently stop firing; fail
    # loudly here instead, so the table is updated with the move.
    import importlib

    import numty_plugin

    missing = []
    for fullname in numty_plugin._NUMPY_FUNCTION_HOOKS:
        module, _, name = fullname.rpartition(".")
        try:
            found = hasattr(importlib.import_module(module), name)
        except ImportError:
            found = False
        if not found:
            missing.append(fullname)
    assert len(numty_plugin._NUMPY_FUNCTION_HOOKS) > 100
    assert missing == [], f"hooked numpy names not found at runtime: {missing}"
