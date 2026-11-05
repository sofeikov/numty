"""Runtime behavior of the numty helpers: shapes, values, narrowing."""
import numpy as np
import pytest

from numty import assert_is_2x3, ident, matmul, zeros_2x3, zeros_3x2, zeros_3x4


def test_zeros_wrappers_have_advertised_shapes() -> None:
    assert zeros_2x3().shape == (2, 3)
    assert zeros_3x2().shape == (3, 2)
    assert zeros_3x4().shape == (3, 4)


def test_zeros_wrappers_are_float64() -> None:
    assert zeros_2x3().dtype == np.dtype(np.float64)


def test_matmul_computes() -> None:
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.arange(12, dtype=np.float64).reshape(3, 4)
    got = matmul(a, b)
    assert got.shape == (2, 4)
    np.testing.assert_allclose(got, np.matmul(a, b))


def test_matmul_of_wrappers_has_chained_shape() -> None:
    assert matmul(zeros_2x3(), zeros_3x4()).shape == (2, 4)


def test_assert_narrower_accepts_matching() -> None:
    assert assert_is_2x3(np.zeros((2, 3))).shape == (2, 3)


def test_assert_narrower_rejects_mismatch() -> None:
    with pytest.raises(AssertionError):
        assert_is_2x3(np.zeros((3, 2)))


def test_ident_returns_same_object() -> None:
    a = zeros_2x3()
    assert ident(a) is a
