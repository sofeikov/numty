"""Runtime behavior of the numty helpers: shapes, values, narrowing."""
import numpy as np
import pytest

from numty import ident, matmul, narrow


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
    assert narrow(np.zeros((2, 3)), np.zeros((2, 3))).shape == (2, 3)


def test_narrow_rejects_mismatch() -> None:
    with pytest.raises(AssertionError):
        narrow(np.zeros((3, 2)), np.zeros((2, 3)))


def test_narrow_rejects_dtype_mismatch() -> None:
    with pytest.raises(AssertionError):
        narrow(np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3)))


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
