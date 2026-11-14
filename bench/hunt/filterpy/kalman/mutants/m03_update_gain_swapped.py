# MUTANT: KalmanFilter.update: K = SI @ PHT (operands swapped)
"""Linear Kalman filter from filterpy, extracted and shape-annotated for numty.

Upstream: filterpy v2.0.0, commit 1337082 (https://github.com/rlabbe/filterpy)

- ``filterpy/kalman/kalman_filter.py``
    ``KalmanFilter`` (class, lines 133-1297): ``__init__``, ``predict``,
    ``update``, ``predict_steadystate``, ``update_steadystate``,
    ``update_correlated``, ``update_sequential``, ``batch_filter``,
    ``rts_smoother``, ``get_prediction``, ``get_update``, ``residual_of``,
    ``measurement_of_state``, ``log_likelihood`` / ``likelihood`` /
    ``mahalanobis`` properties, ``log_likelihood_of``;
    module functions ``update`` (1401), ``update_steadystate`` (1511),
    ``predict`` (1571), ``batch_filter`` (1664), ``rts_smoother`` (1792).
- ``filterpy/common/helpers.py`` ``reshape_z`` (324), column-vector state only.
- ``filterpy/stats/stats.py`` ``logpdf`` (131): wraps
  ``scipy.stats.multivariate_normal.logpdf(..., allow_singular=True)``;
  re-implemented here with numpy (eigh pseudo-inverse / pseudo-determinant,
  the same cut-off scipy's ``_PSD`` uses).

Sizes: ``n_x`` state (dim_x), ``n_z`` measurement (dim_z), ``n_u`` control
(dim_u), ``n_steps`` epochs of a batch, ``n_i`` measurements in one sequential
update. The state is a column vector ``(n_x, 1)`` throughout (filterpy's
documented default); the 1-D and 0-D state branches of ``reshape_z`` and
``batch_filter`` are left out, see NOTES.md.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from math import exp, log, sqrt
from typing import Any, Generic, Literal, Protocol, Sequence, TypeVar, overload

import numpy as np
import numpy.linalg as linalg
from numpy import dot, eye, isscalar, zeros

from numty import F64

n_x = TypeVar("n_x", bound=int)
n_z = TypeVar("n_z", bound=int)
n_u = TypeVar("n_u", bound=int)
n_i = TypeVar("n_i", bound=int)
k = TypeVar("k", bound=int)


class Inverse(Protocol):
    """A square-matrix inverse such as ``np.linalg.inv`` or ``np.linalg.pinv``."""

    def __call__(self, a: F64[k, k], /) -> F64[k, k]: ...


# ---------------------------------------------------------------- helpers


def reshape_z(z: F64[n_z, 1] | F64[n_z] | F64[1, n_z] | float, dim_z: n_z) -> F64[n_z, 1]:
    """Ensure z is a (dim_z, 1) shaped vector (filterpy.common.reshape_z, ndim=2)."""
    z2 = np.atleast_2d(z)
    if z2.shape[1] == dim_z:
        z2 = z2.T

    if z2.shape != (dim_z, 1):
        raise ValueError(
            "z (shape {}) must be convertible to shape ({}, 1)".format(z2.shape, dim_z)
        )
    return z2


def logpdf(x: F64[k, 1] | F64[k], mean: F64[k, 1] | F64[k] | None, cov: F64[k, k]) -> float:
    """log N(x; mean, cov), as scipy's multivariate_normal.logpdf(allow_singular=True)."""
    flat_x: F64[k] = np.asarray(x).flatten()
    if mean is not None:
        flat_mean: F64[k] = np.asarray(mean).flatten()
    else:
        flat_mean = np.zeros_like(flat_x)
    dev = flat_x - flat_mean

    s, u = np.linalg.eigh(cov)
    eps = 1e6 * np.finfo(np.float64).eps * np.max(np.abs(s))
    if np.min(s) < -eps:
        raise ValueError("the input matrix must be symmetric positive semidefinite")
    d = s[s > eps]
    s_pinv = np.array([0.0 if abs(v) <= eps else 1.0 / v for v in s])
    U = u * np.sqrt(s_pinv)
    log_pdet = float(np.sum(np.log(d)))
    rank = len(d)
    maha = float(np.sum(np.square(dot(dev, U))))
    out = -0.5 * (rank * log(2 * np.pi) + log_pdet + maha)
    if rank < len(s):
        # outside the support of a singular normal
        V = u[:, s <= eps]
        if not float(np.linalg.norm(dot(dev, V))) < 1e3 * eps:
            out = -np.inf
    return out


# ---------------------------------------------------------------- the filter


class KalmanFilter(Generic[n_x, n_z, n_u]):
    """Linear Kalman filter with state (n_x, 1), measurement (n_z, 1), control (n_u, 1)."""

    @overload
    def __init__(self: KalmanFilter[n_x, n_z, Literal[0]], dim_x: n_x, dim_z: n_z) -> None: ...
    @overload
    def __init__(self, dim_x: n_x, dim_z: n_z, dim_u: n_u) -> None: ...
    def __init__(self, dim_x: n_x, dim_z: n_z, dim_u: int = 0) -> None:
        if dim_x < 1:
            raise ValueError("dim_x must be 1 or greater")
        if dim_z < 1:
            raise ValueError("dim_z must be 1 or greater")
        if dim_u < 0:
            raise ValueError("dim_u must be 0 or greater")

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x: F64[n_x, 1] = zeros((dim_x, 1))  # state
        self.P: F64[n_x, n_x] = eye(dim_x)  # uncertainty covariance
        self.Q: F64[n_x, n_x] = eye(dim_x)  # process uncertainty
        self.B: F64[n_x, n_u] | None = None  # control transition matrix
        self.F: F64[n_x, n_x] = eye(dim_x)  # state transition matrix
        self.H: F64[n_z, n_x] = zeros((dim_z, dim_x))  # measurement function
        self.R: F64[n_z, n_z] = eye(dim_z)  # measurement uncertainty
        self._alpha_sq = 1.0  # fading memory control
        self.M: F64[n_x, n_z] = np.zeros((dim_x, dim_z))  # process-measurement cross correlation
        self.z: np.ndarray[tuple[n_z, Literal[1]], np.dtype[Any]] = np.array([[None] * self.dim_z]).T

        self.K: F64[n_x, n_z] = np.zeros((dim_x, dim_z))  # kalman gain
        self.y: F64[n_z, 1] = zeros((dim_z, 1))
        self.S: F64[n_z, n_z] = np.zeros((dim_z, dim_z))  # system uncertainty
        self.SI: F64[n_z, n_z] = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I: F64[n_x, n_x] = np.eye(dim_x)

        self.x_prior: F64[n_x, 1] = self.x.copy()
        self.P_prior: F64[n_x, n_x] = self.P.copy()
        self.x_post: F64[n_x, 1] = self.x.copy()
        self.P_post: F64[n_x, n_x] = self.P.copy()

        self._log_likelihood: float | None = log(sys.float_info.min)
        self._likelihood: float | None = sys.float_info.min
        self._mahalanobis: float | None = None

        self.inv: Inverse = np.linalg.inv

    # ------------------------------------------------------------ predict

    def predict(
        self,
        u: F64[n_u, 1] | float | None = None,
        B: F64[n_x, n_u] | None = None,
        F: F64[n_x, n_x] | None = None,
        Q: F64[n_x, n_x] | float | None = None,
    ) -> None:
        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = eye(self.dim_x) * Q

        # x = Fx + Bu
        if B is not None and u is not None:
            self.x = dot(F, self.x) + dot(B, u)
        else:
            self.x = dot(F, self.x)

        # P = FPF' + Q
        self.P = self._alpha_sq * dot(dot(F, self.P), F.T) + Q

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def predict_steadystate(self, u: F64[n_u, 1] | float = 0, B: F64[n_x, n_u] | None = None) -> None:
        if B is None:
            B = self.B

        if B is not None:
            self.x = dot(self.F, self.x) + dot(B, u)
        else:
            self.x = dot(self.F, self.x)

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    # ------------------------------------------------------------ update

    def update(
        self,
        z: F64[n_z, 1] | None,
        R: F64[n_z, n_z] | float | None = None,
        H: F64[n_z, n_x] | None = None,
    ) -> None:
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = zeros((self.dim_z, 1))
            return

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        if H is None:
            z = reshape_z(z, self.dim_z)
            H = self.H

        # y = z - Hx
        self.y = z - dot(H, self.x)

        PHT = dot(self.P, H.T)

        # S = HPH' + R
        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)
        # K = PH'inv(S)
        self.K = dot(self.SI, PHT)

        # x = x + Ky
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)

        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def update_steadystate(self, z: F64[n_z, 1] | F64[n_z] | float | None) -> None:
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = zeros((self.dim_z, 1))
            return

        zc = reshape_z(z, self.dim_z)

        self.y = zc - dot(self.H, self.x)
        self.x = self.x + dot(self.K, self.y)

        self.z = deepcopy(zc)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def update_correlated(
        self,
        z: F64[n_z, 1] | None,
        R: F64[n_z, n_z] | float | None = None,
        H: F64[n_z, n_x] | None = None,
    ) -> None:
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = zeros((self.dim_z, 1))
            return

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        if H is None:
            z = reshape_z(z, self.dim_z)
            H = self.H

        # upstream's two scalar special cases (x.ndim == 1 with z of shape (1, 1),
        # and shape(z) == ()) are dead for a column state and a column z

        self.y = z - dot(H, self.x)

        PHT = dot(self.P, H.T)

        self.S = dot(H, PHT) + dot(H, self.M) + dot(self.M.T, H.T) + R
        self.SI = self.inv(self.S)

        self.K = dot(PHT + self.M, self.SI)

        self.x = self.x + dot(self.K, self.y)
        self.P = self.P - dot(self.K, dot(H, self.P) + self.M.T)

        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def update_sequential(
        self,
        start: int,
        z_i: F64[n_i] | float,
        R_i: F64[int, int] | float | None = None,
        H_i: F64[int, n_x] | None = None,
    ) -> None:
        if isscalar(z_i):
            length = 1
        else:
            length = len(z_i)
        zc: F64[n_i, 1] = np.reshape(z_i, [length, 1])
        stop = start + length

        if R_i is None:
            R_i = self.R[start:stop, start:stop]
        elif isscalar(R_i):
            R_i = eye(length) * R_i

        if H_i is None:
            H_i = self.H[start:stop]

        Hc: F64[n_i, n_x] = np.reshape(H_i, [length, self.dim_x])

        # y_i = z_i - H_i @ x
        y_i = zc - dot(Hc, self.x)
        self.y[start:stop] = y_i

        PHT = dot(self.P, Hc.T)

        S_i = dot(Hc, PHT) + R_i

        if length == 1:
            K_i = PHT * (1.0 / S_i)
        else:
            K_i = dot(PHT, linalg.inv(S_i))

        self.K[:, start:stop] = K_i
        I_KH = self._I - np.dot(K_i, Hc)

        self.x += dot(K_i, y_i)

        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(K_i, R_i), K_i.T)  # type: ignore[assignment]  # NUMTY: S_i = (n_i, n_i) + R_i (int, int) is (?U, ?U) (#333); the unknowns spread through K_i and I_KH and P comes out (?U, ?U)

        self.z[start:stop] = zc
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    # ------------------------------------------------------------ batch

    def batch_filter(
        self,
        zs: Sequence[F64[n_z, 1] | None],
        Fs: Sequence[F64[n_x, n_x]] | None = None,
        Qs: Sequence[F64[n_x, n_x] | float] | None = None,
        Hs: Sequence[F64[n_z, n_x]] | None = None,
        Rs: Sequence[F64[n_z, n_z] | float] | None = None,
        Bs: Sequence[F64[n_x, n_u] | None] | None = None,
        us: Sequence[F64[n_u, 1] | float | None] | None = None,
        update_first: bool = False,
    ) -> tuple[
        F64["n_steps n_x 1"], F64["n_steps n_x n_x"], F64["n_steps n_x 1"], F64["n_steps n_x n_x"]
    ]:
        n = len(zs)
        if Fs is None:
            Fs = [self.F] * n
        if Qs is None:
            Qs = [self.Q] * n
        if Hs is None:
            Hs = [self.H] * n
        if Rs is None:
            Rs = [self.R] * n
        if Bs is None:
            Bs = [self.B] * n
        if us is None:
            us = [0] * n

        # mean estimates from Kalman Filter (column state: the ndim == 1 branch is left out)
        means = zeros((n, self.dim_x, 1))
        means_p = zeros((n, self.dim_x, 1))

        # state covariances from Kalman Filter
        covariances = zeros((n, self.dim_x, self.dim_x))
        covariances_p = zeros((n, self.dim_x, self.dim_x))

        if update_first:
            for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):
                self.update(z, R=R, H=H)
                means[i, :] = self.x
                covariances[i, :, :] = self.P

                self.predict(u=u, B=B, F=F, Q=Q)
                means_p[i, :] = self.x
                covariances_p[i, :, :] = self.P
        else:
            for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):
                self.predict(u=u, B=B, F=F, Q=Q)
                means_p[i, :] = self.x
                covariances_p[i, :, :] = self.P

                self.update(z, R=R, H=H)
                means[i, :] = self.x
                covariances[i, :, :] = self.P

        return (means, covariances, means_p, covariances_p)

    def rts_smoother(
        self,
        Xs: F64["n_steps n_x 1"],
        Ps: F64["n_steps n_x n_x"],
        Fs: Sequence[F64[n_x, n_x]] | None = None,
        Qs: Sequence[F64[n_x, n_x]] | None = None,
        inv: Inverse = np.linalg.inv,
    ) -> tuple[
        F64["n_steps n_x 1"], F64["n_steps n_x n_x"], F64["n_steps n_x n_x"], F64["n_steps n_x n_x"]
    ]:
        if len(Xs) != len(Ps):
            raise ValueError("length of Xs and Ps must be the same")

        n = Xs.shape[0]
        dim_x = Xs.shape[1]

        if Fs is None:
            Fs = [self.F] * n
        if Qs is None:
            Qs = [self.Q] * n

        # smoother gain
        K = zeros((n, dim_x, dim_x))

        x, P, Pp = Xs.copy(), Ps.copy(), Ps.copy()
        for k in range(n - 2, -1, -1):
            Pp[k] = dot(dot(Fs[k + 1], P[k]), Fs[k + 1].T) + Qs[k + 1]

            K[k] = dot(dot(P[k], Fs[k + 1].T), inv(Pp[k]))
            x[k] += dot(K[k], x[k + 1] - dot(Fs[k + 1], x[k]))
            P[k] += dot(dot(K[k], P[k + 1] - Pp[k]), K[k].T)

        return (x, P, K, Pp)

    # ------------------------------------------------------------ queries

    def get_prediction(
        self,
        u: F64[n_u, 1] | float | None = None,
        B: F64[n_x, n_u] | None = None,
        F: F64[n_x, n_x] | None = None,
        Q: F64[n_x, n_x] | float | None = None,
    ) -> tuple[F64[n_x, 1], F64[n_x, n_x]]:
        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = eye(self.dim_x) * Q

        if B is not None and u is not None:
            x = dot(F, self.x) + dot(B, u)
        else:
            x = dot(F, self.x)

        P = self._alpha_sq * dot(dot(F, self.P), F.T) + Q

        return x, P

    def get_update(
        self, z: F64[n_z, 1] | F64[n_z] | float | None = None
    ) -> tuple[F64[n_x, 1], F64[n_x, n_x]]:
        if z is None:
            return self.x, self.P
        zc = reshape_z(z, self.dim_z)

        R = self.R
        H = self.H
        P = self.P
        x = self.x

        y = zc - dot(H, x)

        PHT = dot(P, H.T)

        S = dot(H, PHT) + R

        K = dot(PHT, self.inv(S))

        x = x + dot(K, y)

        I_KH = self._I - dot(K, H)
        P = dot(dot(I_KH, P), I_KH.T) + dot(dot(K, R), K.T)

        return x, P

    def residual_of(self, z: F64[n_z, 1] | F64[n_z] | float) -> F64[n_z, 1]:
        zc = reshape_z(z, self.dim_z)
        return zc - dot(self.H, self.x_prior)

    def measurement_of_state(self, x: F64[n_x, 1]) -> F64[n_z, 1]:
        return dot(self.H, x)

    @property
    def log_likelihood(self) -> float:
        if self._log_likelihood is None:
            self._log_likelihood = logpdf(x=self.y, mean=None, cov=self.S)
        return self._log_likelihood

    @property
    def likelihood(self) -> float:
        if self._likelihood is None:
            self._likelihood = exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self) -> float:
        if self._mahalanobis is None:
            self._mahalanobis = sqrt(dot(dot(self.y.T, self.SI), self.y).item())
        return self._mahalanobis

    def log_likelihood_of(self, z: F64[n_z, 1] | None) -> float:
        if z is None:
            return log(sys.float_info.min)
        return logpdf(z, dot(self.H, self.x), self.S)


# ---------------------------------------------------------------- functions


@overload
def update(
    x: F64[n_x, 1],
    P: F64[n_x, n_x],
    z: F64[n_z, 1] | F64[n_z] | float | None,
    R: F64[n_z, n_z],
    H: F64[n_z, n_x],
    return_all: Literal[False] = False,
) -> tuple[F64[n_x, 1], F64[n_x, n_x]]: ...
@overload
def update(
    x: F64[n_x, 1],
    P: F64[n_x, n_x],
    z: F64[n_z, 1] | F64[n_z] | float | None,
    R: F64[n_z, n_z],
    H: F64[n_z, n_x],
    return_all: Literal[True],
) -> tuple[F64[n_x, 1], F64[n_x, n_x], F64[n_z, 1] | None, F64[n_x, n_z] | None, F64[n_z, n_z] | None, float | None]: ...
def update(
    x: F64[n_x, 1],
    P: F64[n_x, n_x],
    z: F64[n_z, 1] | F64[n_z] | float | None,
    R: F64[n_z, n_z],
    H: F64[n_z, n_x],
    return_all: bool = False,
) -> (
    tuple[F64[n_x, 1], F64[n_x, n_x]]
    | tuple[F64[n_x, 1], F64[n_x, n_x], F64[n_z, 1] | None, F64[n_x, n_z] | None, F64[n_z, n_z] | None, float | None]
):
    """filterpy.kalman.update for a column state and a matrix H (H=None, the
    scalar/univariate form, is left out)."""
    if z is None:
        if return_all:
            return x, P, None, None, None, None
        return x, P

    Hx = np.atleast_1d(dot(H, x))
    zc = reshape_z(z, Hx.shape[0])

    # error (residual) between measurement and prediction
    y = zc - Hx

    # project system uncertainty into measurement space
    S = dot(dot(H, P), H.T) + R

    # map system uncertainty into kalman gain
    try:
        K = dot(dot(P, H.T), linalg.inv(S))
    except:  # noqa: E722  (upstream: a bare except, also taken for a singular S)
        K = dot(dot(P, H.T), 1.0 / S)

    # predict new x with residual scaled by the kalman gain
    x = x + dot(K, y)

    # P = (I-KH)P(I-KH)' + KRK'
    KH = dot(K, H)
    I_KH = np.eye(KH.shape[0]) - KH
    P = dot(dot(I_KH, P), I_KH.T) + dot(dot(K, R), K.T)

    if return_all:
        # compute log likelihood
        log_likelihood = logpdf(zc, dot(H, x), S)
        return x, P, y, K, S, log_likelihood
    return x, P


def update_steadystate(
    x: F64[n_x, 1], z: F64[n_z, 1] | F64[n_z] | float | None, K: F64[n_x, n_z], H: F64[n_z, n_x]
) -> F64[n_x, 1]:
    if z is None:
        return x

    Hx = np.atleast_1d(dot(H, x))
    zc = reshape_z(z, Hx.shape[0])

    y = zc - Hx

    return x + dot(K, y)


def predict(
    x: F64[n_x, 1],
    P: F64[n_x, n_x],
    F: F64[n_x, n_x],
    Q: F64[n_x, n_x] | float = 0,
    u: F64[n_u, 1] | float = 0,
    B: F64[n_x, n_u] | float = 1,
    alpha: float = 1.0,
) -> tuple[F64[n_x, 1], F64[n_x, n_x]]:
    """filterpy.kalman.predict for a matrix F (the scalar F form is left out)."""
    x = dot(F, x) + dot(B, u)
    P = (alpha * alpha) * dot(dot(F, P), F.T) + Q

    return x, P


def batch_filter(
    x: F64[n_x, 1],
    P: F64[n_x, n_x],
    zs: Sequence[F64[n_z, 1] | F64[n_z] | float | None],
    Fs: Sequence[F64[n_x, n_x]],
    Qs: Sequence[F64[n_x, n_x] | float],
    Hs: Sequence[F64[n_z, n_x]],
    Rs: Sequence[F64[n_z, n_z]],
    Bs: Sequence[F64[n_x, n_u] | float] | None = None,
    us: Sequence[F64[n_u, 1] | float] | None = None,
    update_first: bool = False,
) -> tuple[F64["n_steps n_x 1"], F64["n_steps n_x n_x"], F64["n_steps n_x 1"], F64["n_steps n_x n_x"]]:
    n = len(zs)
    dim_x = x.shape[0]

    # mean estimates from Kalman Filter (column state: the ndim == 1 branch is left out)
    means = zeros((n, dim_x, 1))
    means_p = zeros((n, dim_x, 1))

    # state covariances from Kalman Filter
    covariances = zeros((n, dim_x, dim_x))
    covariances_p = zeros((n, dim_x, dim_x))

    if us is None:
        us = [0.0] * n
        Bs = [0.0] * n

    if update_first:
        for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):  # type: ignore[arg-type]  # UPSTREAM: Bs stays None when us is given (TypeError upstream too)
            x, P = update(x, P, z, R=R, H=H)
            means[i, :] = x
            covariances[i, :, :] = P

            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i, :] = x
            covariances_p[i, :, :] = P
    else:
        for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):  # type: ignore[arg-type]  # UPSTREAM: Bs stays None when us is given (TypeError upstream too)
            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i, :] = x
            covariances_p[i, :, :] = P

            x, P = update(x, P, z, R=R, H=H)
            means[i, :] = x
            covariances[i, :, :] = P

    return (means, covariances, means_p, covariances_p)


def rts_smoother(
    Xs: F64["n_steps n_x 1"],
    Ps: F64["n_steps n_x n_x"],
    Fs: Sequence[F64[n_x, n_x]],
    Qs: Sequence[F64[n_x, n_x]],
) -> tuple[F64["n_steps n_x 1"], F64["n_steps n_x n_x"], F64["n_steps n_x n_x"], F64["n_steps n_x n_x"]]:
    if len(Xs) != len(Ps):
        raise ValueError("length of Xs and Ps must be the same")

    n = Xs.shape[0]
    dim_x = Xs.shape[1]

    # smoother gain
    K = zeros((n, dim_x, dim_x))

    x, P, pP = Xs.copy(), Ps.copy(), Ps.copy()
    for k in range(n - 2, -1, -1):
        pP[k] = dot(dot(Fs[k], P[k]), Fs[k].T) + Qs[k]

        K[k] = dot(dot(P[k], Fs[k].T), linalg.inv(pP[k]))
        x[k] += dot(K[k], x[k + 1] - dot(Fs[k], x[k]))
        P[k] += dot(dot(K[k], P[k + 1] - pP[k]), K[k].T)

    return (x, P, K, pP)
