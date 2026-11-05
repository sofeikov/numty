"""Benchmark: realistic bugs, each annotated with the expected static error.

main() executes every bug and records whether it raised at runtime or ran
silently wrong. The static checker should flag all of them before running.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def transposed_matmul() -> np.ndarray:
    a = np.zeros((2, 3))
    b = np.zeros((2, 3))
    return a @ b.T  # this one is fine; the bug is below


def transposed_matmul_bug() -> np.ndarray:
    # expect: matmul inner dimensions disagree: 3 vs 2
    return np.matmul(np.zeros((2, 3)), np.zeros((2, 3)))


def bias_broadcast_bug() -> np.ndarray:
    # expect: broadcast shapes disagree at trailing axis 1
    return np.zeros((2, 4)) + np.zeros((3, 4))


def wrong_axis_sum(x: np.ndarray) -> np.ndarray:
    # expect: axis 5 is out of range (author meant axis=1 on a 2-D array)
    return x.sum(axis=5)


def reshape_size_bug() -> np.ndarray:
    # expect: reshape changes total size: 12 -> 10
    return np.zeros((3, 4)).reshape((2, 5))


def reshape_minus1_bug() -> np.ndarray:
    # expect: reshape cannot infer -1
    return np.zeros((2, 3)).reshape((4, -1))


def concat_dim_bug() -> np.ndarray:
    # expect: concatenate dimension 1 differs
    return np.concatenate((np.zeros((2, 3)), np.zeros((2, 5))), axis=0)


def stack_shape_bug() -> np.ndarray:
    # expect: stack shapes differ
    return np.stack((np.zeros((2, 3)), np.zeros((3, 2))))


def index_range_bug() -> np.ndarray:
    # expect: index 7 is out of range for axis 0
    return np.zeros((4, 4))[7]


def too_many_indices_bug() -> np.ndarray:
    # expect: too many indices
    return np.zeros((2, 2))[0, 0, 0]


def transpose_axes_bug() -> np.ndarray:
    # expect: not a permutation
    return np.zeros((2, 3)).transpose(0, 0)


def squeeze_size_bug() -> np.ndarray:
    # expect: squeeze axis 0 has size 2, not 1
    return np.zeros((2, 3)).squeeze(0)


def eye_negative_bug() -> np.ndarray:
    # expect: eye dimensions must be non-negative
    return np.eye(-3)


def arange_step_bug() -> np.ndarray:
    # expect: arange step is zero
    return np.arange(0, 10, 0)


def batch_matmul_bug() -> np.ndarray:
    # expect: matmul batch dimensions disagree
    return np.matmul(np.zeros((5, 2, 3)), np.zeros((4, 3, 4)))


def compare_broadcast_bug() -> np.ndarray:
    # expect: broadcast shapes disagree
    return np.zeros((2, 3)) == np.zeros((3, 2))


def argmax_axis_bug() -> np.ndarray:
    # expect: axis 9 is out of range
    return np.zeros((2, 3)).argmax(axis=9)


def einsum_labels_bug() -> np.ndarray:
    # expect: label 'j' has conflicting sizes
    return np.einsum("ij,jh->ih", np.zeros((2, 3)), np.zeros((5, 4)))


def solve_square_bug() -> np.ndarray:
    # expect: requires a square matrix
    return np.linalg.solve(np.zeros((2, 3)), np.zeros(3))


def main() -> None:
    x = np.zeros((4, 4))
    cases: list[tuple[str, Callable[[], object]]] = [
        ("transposed_matmul_bug", lambda: transposed_matmul_bug()),
        ("bias_broadcast_bug", lambda: bias_broadcast_bug()),
        ("wrong_axis_sum", lambda: wrong_axis_sum(x)),
        ("reshape_size_bug", lambda: reshape_size_bug()),
        ("reshape_minus1_bug", lambda: reshape_minus1_bug()),
        ("concat_dim_bug", lambda: concat_dim_bug()),
        ("stack_shape_bug", lambda: stack_shape_bug()),
        ("index_range_bug", lambda: index_range_bug()),
        ("too_many_indices_bug", lambda: too_many_indices_bug()),
        ("transpose_axes_bug", lambda: transpose_axes_bug()),
        ("squeeze_size_bug", lambda: squeeze_size_bug()),
        ("eye_negative_bug", lambda: eye_negative_bug()),
        ("arange_step_bug", lambda: arange_step_bug()),
        ("batch_matmul_bug", lambda: batch_matmul_bug()),
        ("compare_broadcast_bug", lambda: compare_broadcast_bug()),
        ("argmax_axis_bug", lambda: argmax_axis_bug()),
        ("einsum_labels_bug", lambda: einsum_labels_bug()),
        ("solve_square_bug", lambda: solve_square_bug()),
    ]
    for name, run in cases:
        try:
            run()
            print(f"{name}: ran SILENT (wrong result, no crash)")
        except Exception as exc:  # noqa: BLE001 - benchmark records everything
            print(f"{name}: raised {type(exc).__name__}")


if __name__ == "__main__":
    main()
