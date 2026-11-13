"""Issue 330: np.linalg.svd and np.linalg.qr on symbolic shapes.

K = min(M, N) is one fresh unknown per call, shared by the factors:
q (M, ?K), r (?K, N); u (M, ?K), s (?K,), vt (?K, N). With
full_matrices=True u and vt are exact, and a square input is exact.
"""
from typing import TypeVar, reveal_type

import numpy as np
from numty import F64

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)
D = TypeVar("D", bound=int)
B = TypeVar("B", bound=int)
R = TypeVar("R", bound=int)


def factors(a: F64[M, N], sq: F64[D, D], st: F64[3, M, N]) -> None:
    q, r = np.linalg.qr(a, mode="reduced")
    reveal_type(q)
    reveal_type(r)
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    reveal_type(u)
    reveal_type(s)
    reveal_type(vt)
    reveal_type(np.linalg.svd(a))
    reveal_type(np.linalg.svd(a, compute_uv=False))
    reveal_type(np.linalg.svd(sq))
    reveal_type(np.linalg.svd(st, full_matrices=False))
    reveal_type(q.T @ a)
    reveal_type((u * s) @ vt)
    reveal_type(np.linalg.svd(a).S)  # the NamedTuple fields stay readable


def wide(a: F64[M, N]) -> None:
    # (M, ?K) @ (M, N) runs when M <= N (then ?K is M): not provably
    # wrong, so it is not reported, and the rank stays.
    q, _ = np.linalg.qr(a)
    reveal_type(q @ a)


def declared(a: F64[M, N], n: int) -> F64[M, N]:
    # Locals declared with a name the signature does not bind take the
    # unknown, like a name of the return alone does.
    Q = np.zeros((a.shape[0], n))
    Uhat: F64[B, R]
    s: F64[R]
    Vt: F64[R, N]
    Uhat, s, Vt = np.linalg.svd(Q.T @ a, full_matrices=False)
    return (Q @ Uhat) * s @ Vt


def scaled(X: F64[M, N], reg: float) -> tuple[F64[R], F64[N, R]]:
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    scaling = (S**2) / X.shape[0]
    scaling = ((1 - reg) * scaling) + reg
    return scaling, Vt.T


if __name__ == "__main__":
    a = np.arange(15.0).reshape(5, 3)
    factors(a, np.eye(3), np.ones((3, 5, 3)))
    wide(a.T)
    q, r = np.linalg.qr(a, mode="reduced")
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    print(q.shape, r.shape, u.shape, s.shape, vt.shape)
    print(np.linalg.svd(a)[0].shape, np.linalg.svd(a)[2].shape, np.linalg.svd(a).S.shape)
    print(declared(a, 2).shape, scaled(a, 0.1)[1].shape)
