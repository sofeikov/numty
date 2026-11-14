"""Issues 349-351, 360: idioms from scipy, statsmodels and filterpy that stay clean."""
from typing import TypeVar

import numpy as np
from numpy import real
from numpy.lib.stride_tricks import sliding_window_view
from numty import F64

S = TypeVar("S", bound=int)
N = TypeVar("N", bound=int)
W = TypeVar("W", bound=int)
k_ar = TypeVar("k_ar", bound=int)
neqs = TypeVar("neqs", bound=int)
n_eig = TypeVar("n_eig", bound=int)
n_vec = TypeVar("n_vec", bound=int)
nvar = TypeVar("nvar", bound=int)


def abcd(A: F64[S, S] | None, B: F64[S, N] | None) -> F64[S, S]:
    # An empty stand-in in a conditional has `int` sizes.
    A2 = np.atleast_2d(np.asarray(A)) if A is not None else np.zeros((0, 0), dtype=np.float64)
    B2 = np.atleast_2d(np.asarray(B)) if B is not None else np.zeros((0, 0), dtype=np.float64)
    n = A2.shape[0] or B2.shape[0]
    A3: F64[S, S] = np.zeros((n, n), dtype=np.float64) if A2.size == 0 else A2
    return A3


def simplex(n: int) -> None:
    # A local grown in a loop keeps main's answer for its first binding.
    istar = np.array([[-1.0, 1.0]])
    for d in range(2, n + 1):
        row = np.ones((1, istar.shape[1] + 1))
        istar = np.r_[np.c_[istar, np.zeros(istar.shape[0])], row]


def sigma(s: float | F64[N] | F64[N, N], n: N) -> F64[N] | None:
    # `.squeeze()` of a union is narrowed on `.ndim` afterwards.
    arr = np.asarray(s).squeeze()
    if arr.ndim == 0:
        arr = np.repeat(arr, n)
    if arr.ndim == 1:
        vec: F64[N] = arr
        return vec
    return None


def convolve(x: F64["c n"], weights: F64[W]) -> F64["c n"]:
    n_w = weights.shape[0]
    padded: F64["c n_padded"] = np.pad(x, ((0, 0), (n_w // 2, n_w - 1 - n_w // 2)))
    windows = sliding_window_view(padded, n_w, axis=1)
    return windows @ weights[::-1]


def svd_eig(x: F64["nobs", nvar], full: bool) -> tuple[F64[n_eig], F64[nvar, n_vec]]:
    u, s, v = np.linalg.svd(x, full_matrices=full)
    return s**2.0, v.T


def comp_matrix(coefs: F64[k_ar, neqs, neqs]) -> F64["kp kp"]:
    p, k1, k2 = coefs.shape
    kp = k1 * p
    result = np.zeros((kp, kp))
    result[:k1] = np.concatenate(coefs, axis=1)
    return result


def response_times(A: F64[S, S]) -> float:
    vals = np.linalg.eigvals(A)
    r = min(abs(real(vals)))
    if r == 0.0:
        r = 1.0
    return 1.0 / r
