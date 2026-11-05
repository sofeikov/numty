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


def test_ident_returns_same_object() -> None:
    a = np.zeros((2, 3))
    assert ident(a) is a


def test_reshape_keeps_values() -> None:
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    np.testing.assert_array_equal(a.reshape((3, 2)), np.arange(6, dtype=np.float64).reshape(3, 2))
    assert a.reshape(3, 2).shape == (3, 2)
    assert np.reshape(a, (6,)).shape == (6,)


def test_reshape_minus1_infers() -> None:
    a = np.zeros((2, 3))
    assert a.reshape((3, -1)).shape == (3, 2)
    assert a.reshape(-1).shape == (6,)
