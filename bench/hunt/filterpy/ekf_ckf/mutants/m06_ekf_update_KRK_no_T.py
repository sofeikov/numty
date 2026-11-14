# MUTANT: EKF.update: Joseph form K R K instead of K R K' (forgotten .T)
"""Extended and cubature Kalman filters, extracted from filterpy for the numty bug hunt.

Upstream: filterpy v2.0.0, commit 1337082 (https://github.com/rlabbe/filterpy)

- filterpy/kalman/EKF.py:32-410
  ExtendedKalmanFilter: __init__, predict_update, update, predict_x, predict,
  log_likelihood, likelihood, mahalanobis
- filterpy/kalman/CubatureKalmanFilter.py:32-428
  spherical_radial_sigmas, ckf_transform,
  CubatureKalmanFilter: __init__, predict, update, log_likelihood,
  likelihood, mahalanobis
- inlined helpers: filterpy/common/helpers.py:386-436 (outer_product_sum),
  filterpy/stats/stats.py:131-154 (logpdf, via scipy multivariate_normal)

scipy routines are replaced by numpy equivalents: scipy.linalg.inv ->
np.linalg.inv, scipy.linalg.cholesky (upper factor) -> np.linalg.cholesky(P).T,
scipy.stats.multivariate_normal.logpdf (zero mean, non-singular cov) ->
_mvn_logpdf below. __repr__/pretty_str are dropped.

Shape conventions: n_x states, n_z measurements, n_u control inputs. The EKF
keeps x as a column (n_x, 1); the CKF keeps x as a column after the first
predict (ckf_transform returns a column), and its cubature points are
(2*n_x, n_x).
"""

from __future__ import annotations

import sys
from copy import deepcopy
from math import exp, log, pi, sqrt
from typing import Any, Callable, Generic, TypeVar

import numpy as np
from numpy import dot, eye, zeros

from numty import F64, Arr

n_x = TypeVar("n_x", bound=int)
n_z = TypeVar("n_z", bound=int)
n_u = TypeVar("n_u", bound=int)


# ---------------------------------------------------------------- helpers


def _inv(a: F64["k k"]) -> F64["k k"]:
    """scipy.linalg.inv"""
    return np.linalg.inv(a)


def _cholesky_upper(P: F64["n n"]) -> F64["n n"]:
    """scipy.linalg.cholesky(P) (upper factor U with P = U.T @ U)"""
    return np.linalg.cholesky(P).T


def _mvn_logpdf(x: F64["k 1"], cov: F64["k k"]) -> float:
    """filterpy.stats.logpdf(x=x, cov=cov): zero-mean multivariate normal."""
    flat_x = np.asarray(x).flatten()
    k = flat_x.shape[0]
    _, logdet = np.linalg.slogdet(cov)
    maha = float(flat_x @ np.linalg.solve(cov, flat_x))
    return -0.5 * (k * log(2 * pi) + float(logdet) + maha)


def outer_product_sum(A: F64["n_pts a"], B: F64["n_pts b"]) -> F64["a b"]:
    """Sum over i of np.outer(A[i], B[i]) (filterpy.common.outer_product_sum)."""
    outer = np.einsum("ij,ik->ijk", A, B)
    return np.sum(outer, axis=0)


# ---------------------------------------------------------------- EKF


class ExtendedKalmanFilter(Generic[n_x, n_z]):
    """Extended Kalman filter (filterpy.kalman.ExtendedKalmanFilter)."""

    def __init__(self, dim_x: n_x, dim_z: n_z, dim_u: int = 0) -> None:
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x: F64[n_x, 1] = zeros((dim_x, 1))  # state
        self.P: F64[n_x, n_x] = eye(dim_x)  # uncertainty covariance
        self.B: F64[n_x, int] | float = 0  # control transition matrix
        self.F: F64[n_x, n_x] = np.eye(dim_x)  # state transition matrix
        self.R: F64[n_z, n_z] = eye(dim_z)  # state uncertainty
        self.Q: F64[n_x, n_x] = eye(dim_x)  # process uncertainty
        self.y: F64[n_z, 1] = zeros((dim_z, 1))  # residual

        # upstream: reshape_z(np.array([None]*dim_z), dim_z, self.x.ndim),
        # which for a column state is this object column
        self.z: Arr[Any, n_z, 1] | F64[n_z, 1] | F64[1] | float = np.array([[None] * self.dim_z]).T

        # upstream initialises K as np.zeros(self.x.shape), i.e. (n_x, 1);
        # the gain is (n_x, n_z), which is what update() stores
        self.K: F64[n_x, n_z] = np.zeros((dim_x, dim_z))  # kalman gain
        self.y = zeros((dim_z, 1))
        self.S: F64[n_z, n_z] = np.zeros((dim_z, dim_z))  # system uncertainty
        self.SI: F64[n_z, n_z] = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I: F64[n_x, n_x] = np.eye(dim_x)

        self._log_likelihood: float | None = log(sys.float_info.min)
        self._likelihood: float | None = sys.float_info.min
        self._mahalanobis: float | None = None

        # these will always be a copy of x,P after predict() is called
        self.x_prior: F64[n_x, 1] = self.x.copy()
        self.P_prior: F64[n_x, n_x] = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post: F64[n_x, 1] = self.x.copy()
        self.P_post: F64[n_x, n_x] = self.P.copy()

    def predict_update(
        self,
        z: F64[n_z, 1] | float,
        HJacobian: Callable[..., F64[n_z, n_x]],
        Hx: Callable[..., F64[n_z, 1]],
        args: object = (),
        hx_args: object = (),
        u: F64[n_u, 1] | float = 0,
    ) -> None:
        """Predict/update innovation of the extended Kalman filter."""
        if not isinstance(args, tuple):
            args = (args,)

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        z_arr: F64[n_z, 1] | F64[1] | float = z
        if np.isscalar(z) and self.dim_z == 1:
            z_arr = np.asarray([z], float)

        F = self.F
        B = self.B
        P = self.P
        Q = self.Q
        R = self.R
        x = self.x

        H = HJacobian(x, *args)

        # predict step
        x = dot(F, x) + dot(B, u)
        P = dot(F, P).dot(F.T) + Q

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

        # update step
        PHT = dot(P, H.T)
        self.S = dot(H, PHT) + R
        self.SI = _inv(self.S)
        self.K = dot(PHT, self.SI)

        self.y = z_arr - Hx(x, *hx_args)
        self.x = x + dot(self.K, self.y)

        I_KH = self._I - dot(self.K, H)
        self.P = dot(I_KH, P).dot(I_KH.T) + dot(self.K, R).dot(self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z_arr)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def update(
        self,
        z: F64[n_z, 1] | float | None,
        HJacobian: Callable[..., F64[n_z, n_x]],
        Hx: Callable[..., F64[n_z, 1]],
        R: F64[n_z, n_z] | float | None = None,
        args: object = (),
        hx_args: object = (),
        residual: Callable[[F64[n_z, 1] | F64[1] | float, F64[n_z, 1]], F64[n_z, 1]] = np.subtract,
    ) -> None:
        """Update innovation of the extended Kalman filter."""
        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if not isinstance(args, tuple):
            args = (args,)

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        R_mat: F64[n_z, n_z]
        if R is None:
            R_mat = self.R
        elif np.isscalar(R):
            R_mat = eye(self.dim_z) * R
        else:
            R_mat = R

        z_arr: F64[n_z, 1] | F64[1] | float = z
        if np.isscalar(z) and self.dim_z == 1:
            z_arr = np.asarray([z], float)

        H = HJacobian(self.x, *args)

        PHT = dot(self.P, H.T)
        self.S = dot(H, PHT) + R_mat
        self.SI = _inv(self.S)
        self.K = PHT.dot(self.SI)

        hx = Hx(self.x, *hx_args)
        self.y = residual(z_arr, hx)
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK' is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
        I_KH = self._I - dot(self.K, H)
        self.P = dot(I_KH, self.P).dot(I_KH.T) + dot(self.K, R_mat).dot(self.K)

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        # save measurement and posterior state
        self.z = deepcopy(z_arr)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict_x(self, u: F64[n_u, 1] | float = 0) -> None:
        """Predicts the next state of X."""
        self.x = dot(self.F, self.x) + dot(self.B, u)

    def predict(self, u: F64[n_u, 1] | float = 0) -> None:
        """Predict next state (prior) using the state propagation equations."""
        self.predict_x(u)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    @property
    def log_likelihood(self) -> float:
        """log-likelihood of the last measurement."""
        if self._log_likelihood is None:
            self._log_likelihood = _mvn_logpdf(x=self.y, cov=self.S)
        return self._log_likelihood

    @property
    def likelihood(self) -> float:
        """likelihood of the last measurement, floored at sys.float_info.min."""
        if self._likelihood is None:
            self._likelihood = exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self) -> float:
        """Mahalanobis distance of the innovation."""
        if self._mahalanobis is None:
            self._mahalanobis = sqrt(dot(dot(self.y.T, self.SI), self.y).item())
        return self._mahalanobis


# ---------------------------------------------------------------- CKF


def spherical_radial_sigmas(x: F64[n_x] | F64[n_x, 1], P: F64[n_x, n_x]) -> F64["2*n_x n_x"]:
    """Cubature points for state x and covariance P (Arasaratnam & Haykin 2009)."""
    n, _ = P.shape
    xf = x.flatten()

    sigmas = np.empty((2 * n, n))
    U = _cholesky_upper(P) * sqrt(n)
    for k in range(n):
        sigmas[k] = xf + U[k]
        sigmas[n + k] = xf - U[k]

    return sigmas


def ckf_transform(Xs: F64["n_pts dim"], Q: F64["dim dim"]) -> tuple[F64["dim 1"], F64["dim dim"]]:
    """Mean (column) and covariance of an array of cubature points."""
    m, n = Xs.shape

    x = sum(Xs, 0)[:, None] / m
    P = np.zeros((n, n))
    xf = x.flatten()
    for k in range(m):
        P += np.outer(Xs[k], Xs[k]) - np.outer(xf, xf)

    P *= 1 / m
    P += Q

    return x, P


class CubatureKalmanFilter(Generic[n_x, n_z]):
    """Cubature Kalman filter (filterpy.kalman.CubatureKalmanFilter)."""

    def __init__(
        self,
        dim_x: n_x,
        dim_z: n_z,
        dt: float,
        hx: Callable[..., F64[n_z]],
        fx: Callable[..., F64[n_x]],
        x_mean_fn: Callable[..., F64[n_x]] | None = None,
        z_mean_fn: Callable[..., F64[n_z]] | None = None,
        residual_x: Callable[[F64[n_x, 1], F64[n_x, 1]], F64[n_x, 1]] | None = None,
        residual_z: Callable[[F64[n_z, 1], F64[n_z, 1]], F64[n_z, 1]] | None = None,
    ) -> None:
        self.Q: F64[n_x, n_x] = eye(dim_x)
        self.R: F64[n_z, n_z] = eye(dim_z)
        # upstream: zeros(dim_x), a flat vector; predict() and update() keep
        # x as a column (ckf_transform returns one), so start with a column
        self.x: F64[n_x, 1] = zeros((dim_x, 1))
        self.P: F64[n_x, n_x] = eye(dim_x)
        # upstream: K = 0 and y = 0 until the first update
        self.K: F64[n_x, n_z] = zeros((dim_x, dim_z))
        self.dim_x = dim_x
        self.dim_z = dim_z
        self._dt = dt
        self._num_sigmas = 2 * dim_x
        self.hx = hx
        self.fx = fx
        self.x_mean = x_mean_fn
        self.z_mean = z_mean_fn
        self.y: F64[n_z, 1] = zeros((dim_z, 1))
        self.z: Arr[Any, n_z, 1] | F64[n_z, 1] = np.array([[None] * self.dim_z]).T
        self.S: F64[n_z, n_z] = np.zeros((dim_z, dim_z))  # system uncertainty
        self.SI: F64[n_z, n_z] = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        self.residual_x: Callable[[F64[n_x, 1], F64[n_x, 1]], F64[n_x, 1]]
        if residual_x is None:
            self.residual_x = np.subtract
        else:
            self.residual_x = residual_x

        self.residual_z: Callable[[F64[n_z, 1], F64[n_z, 1]], F64[n_z, 1]]
        if residual_z is None:
            self.residual_z = np.subtract
        else:
            self.residual_z = residual_z

        # sigma points transformed through f(x) and h(x)
        self.sigmas_f: F64["2*n_x n_x"] = zeros((2 * self.dim_x, self.dim_x))
        self.sigmas_h: F64["2*n_x n_z"] = zeros((2 * self.dim_x, self.dim_z))

        # Only computed only if requested via property
        self._log_likelihood: float | None = log(sys.float_info.min)
        self._likelihood: float | None = sys.float_info.min
        self._mahalanobis: float | None = None

        # these will always be a copy of x,P after predict() is called
        self.x_prior: F64[n_x, 1] = self.x.copy()
        self.P_prior: F64[n_x, n_x] = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post: F64[n_x, 1] = self.x.copy()
        self.P_post: F64[n_x, n_x] = self.P.copy()

    def predict(self, dt: float | None = None, fx_args: object = ()) -> None:
        """Predict step of the CKF."""
        if dt is None:
            dt = self._dt

        if not isinstance(fx_args, tuple):
            fx_args = (fx_args,)

        sigmas = spherical_radial_sigmas(self.x, self.P)

        # evaluate cubature points
        for k in range(self._num_sigmas):
            self.sigmas_f[k] = self.fx(sigmas[k], dt, *fx_args)

        self.x, self.P = ckf_transform(self.sigmas_f, self.Q)

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(
        self,
        z: F64[n_z, 1] | None,
        R: F64[n_z, n_z] | float | None = None,
        hx_args: object = (),
    ) -> None:
        """Update the CKF with the measurement z (a (n_z, 1) column)."""
        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        R_mat: F64[n_z, n_z]
        if R is None:
            R_mat = self.R
        elif np.isscalar(R):
            R_mat = eye(self.dim_z) * R
        else:
            R_mat = R

        for k in range(self._num_sigmas):
            self.sigmas_h[k] = self.hx(self.sigmas_f[k], *hx_args)

        # mean and covariance of prediction passed through unscented transform
        zp, self.S = ckf_transform(self.sigmas_h, R_mat)
        self.SI = _inv(self.S)

        # compute cross variance of the state and the measurements
        m = self._num_sigmas  # literaure uses m for scaling factor
        xf = self.x.flatten()
        zpf = zp.flatten()
        Pxz = outer_product_sum(self.sigmas_f - xf, self.sigmas_h - zpf) / m

        self.K = dot(Pxz, self.SI)  # Kalman gain
        self.y = self.residual_z(z, zp)  # residual

        self.x = self.x + dot(self.K, self.y)
        self.P = self.P - dot(self.K, self.S).dot(self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    @property
    def log_likelihood(self) -> float:
        """log-likelihood of the last measurement."""
        if self._log_likelihood is None:
            self._log_likelihood = _mvn_logpdf(x=self.y, cov=self.S)
        return self._log_likelihood

    @property
    def likelihood(self) -> float:
        """likelihood of the last measurement, floored at sys.float_info.min."""
        if self._likelihood is None:
            self._likelihood = exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self) -> float:
        """Mahalanobis distance of the innovation."""
        if self._mahalanobis is None:
            self._mahalanobis = sqrt(dot(dot(self.y.T, self.SI), self.y).item())
        return self._mahalanobis
