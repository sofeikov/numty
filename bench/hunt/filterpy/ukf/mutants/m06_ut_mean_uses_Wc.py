# MUTANT: unscented_transform: mean weighted with Wc instead of Wm
"""Unscented Kalman filter, sigma points and unscented transform (filterpy).

Extracted from filterpy v2.0.0 (commit 1337082, https://github.com/rlabbe/filterpy):

- filterpy/kalman/unscented_transform.py:22-128   unscented_transform
- filterpy/kalman/sigma_points.py:24-208          MerweScaledSigmaPoints
- filterpy/kalman/sigma_points.py:211-383         JulierSigmaPoints
- filterpy/kalman/sigma_points.py:386-534         SimplexSigmaPoints
- filterpy/kalman/UKF.py:32-777                   UnscentedKalmanFilter
  (__init__, predict, update, cross_variance, compute_process_sigmas,
  batch_filter, rts_smoother, log_likelihood, likelihood, mahalanobis)
- filterpy/stats/stats.py:131-154                 logpdf (numpy rewrite)

numpy-only: scipy.linalg.cholesky (upper factor) is `cholesky` below, and
scipy.stats.multivariate_normal.logpdf is `logpdf` below. Dropped: __repr__,
the `saver` argument of batch_filter, the **fx_args / **hx_args pass-through.

Sizes: n_x states (dim_x), n_z measurements (dim_z), n_sig sigma points
(2*n_x+1 for Merwe and Julier, n_x+1 for Simplex), n_steps time steps.
"""

import sys
from copy import deepcopy
from math import exp, log, sqrt
from typing import Callable, Generic, Literal, Protocol, Sequence, TypeVar

import numpy as np

from numty import F64, Arr

n_x = TypeVar("n_x", bound=int)
n_z = TypeVar("n_z", bound=int)
n_sig = TypeVar("n_sig", bound=int)
n_dim = TypeVar("n_dim", bound=int)
n_steps = TypeVar("n_steps", bound=int)


# --------------------------------------------------------------------------
# numpy stand-ins for scipy
# --------------------------------------------------------------------------


def cholesky(a: F64[n_dim, n_dim]) -> F64[n_dim, n_dim]:
    """scipy.linalg.cholesky(a): the upper factor U with U.T @ U == a."""
    return np.linalg.cholesky(a).T


def logpdf(x: F64[n_dim], cov: F64[n_dim, n_dim]) -> float:
    """filterpy.stats.logpdf(x, cov=cov): log N(x; 0, cov), cov positive definite."""
    flat_x = np.asarray(x).flatten()
    sign, logdet = np.linalg.slogdet(cov)
    maha = float(flat_x @ np.linalg.solve(cov, flat_x))
    return -0.5 * (flat_x.shape[0] * log(2 * np.pi) + float(logdet) + maha)


# --------------------------------------------------------------------------
# unscented_transform.py
# --------------------------------------------------------------------------


def unscented_transform(
    sigmas: F64[n_sig, n_dim],
    Wm: F64[n_sig],
    Wc: F64[n_sig],
    noise_cov: F64[n_dim, n_dim] | None = None,
    mean_fn: Callable[[F64[n_sig, n_dim], F64[n_sig]], F64[n_dim]] | None = None,
    residual_fn: Callable[[F64[n_dim], F64[n_dim]], F64[n_dim]] | None = None,
) -> tuple[F64[n_dim], F64[n_dim, n_dim]]:
    """Mean and covariance of sigma points (rows) under weights Wm, Wc."""
    kmax, n = sigmas.shape

    if mean_fn is None:
        x = np.dot(Wc, sigmas)  # dot = \Sigma^n_1 (W[k]*Xi[k])
    else:
        x = mean_fn(sigmas, Wm)

    if residual_fn is np.subtract or residual_fn is None:
        y = sigmas - x[np.newaxis, :]
        P = np.dot(y.T, np.dot(np.diag(Wc), y))
    else:
        P = np.zeros((n, n))
        for k in range(kmax):
            yk = residual_fn(sigmas[k], x)
            P += Wc[k] * np.outer(yk, yk)

    if noise_cov is not None:
        P += noise_cov

    return (x, P)


class UnscentedTransformFn(Protocol):
    """The signature of `unscented_transform`, for the UT= argument."""

    def __call__(
        self,
        sigmas: F64[n_sig, n_dim],
        Wm: F64[n_sig],
        Wc: F64[n_sig],
        noise_cov: F64[n_dim, n_dim] | None = None,
        mean_fn: Callable[[F64[n_sig, n_dim], F64[n_sig]], F64[n_dim]] | None = None,
        residual_fn: Callable[[F64[n_dim], F64[n_dim]], F64[n_dim]] | None = None,
    ) -> tuple[F64[n_dim], F64[n_dim, n_dim]]: ...


# --------------------------------------------------------------------------
# sigma_points.py
# --------------------------------------------------------------------------


class MerweScaledSigmaPoints(Generic[n_x]):
    """Van der Merwe's scaled sigma points: 2*n+1 points of an n-state."""

    def __init__(
        self,
        n: int,
        alpha: float,
        beta: float,
        kappa: float,
        sqrt_method: Callable[[F64[n_x, n_x]], F64[n_x, n_x]] | None = None,
        subtract: Callable[[F64[n_x], F64[n_x]], F64[n_x]] | None = None,
    ) -> None:
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.sqrt: Callable[[F64[n_x, n_x]], F64[n_x, n_x]]
        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method

        self.subtract: Callable[[F64[n_x], F64[n_x]], F64[n_x]]
        if subtract is None:
            self.subtract = np.subtract
        else:
            self.subtract = subtract

        self._compute_weights()

    def num_sigmas(self) -> int:
        return 2 * self.n + 1

    def sigma_points(self, x: F64[n_x] | float, P: F64[n_x, n_x] | float) -> F64["2*n_x+1 n_x"]:
        if self.n != np.size(x):
            raise ValueError("expected size(x) {}, but size is {}".format(self.n, np.size(x)))

        n = self.n

        xv: F64[n_x] = np.asarray([x]) if np.isscalar(x) else x  # type: ignore[assignment]  # NUMTY: a scalar x passed the size check above, so n_x is 1 here; no annotation can tie (1,) to n_x in this branch

        Pm: F64[n_x, n_x] = np.eye(n) * P if np.isscalar(P) else np.atleast_2d(P)

        lambda_ = self.alpha**2 * (n + self.kappa) - n
        U = self.sqrt((lambda_ + n) * Pm)

        sigmas: F64["2*n_x+1 n_x"] = np.zeros((2 * n + 1, n))
        sigmas[0] = xv
        for k in range(n):
            sigmas[k + 1] = self.subtract(xv, -U[k])
            sigmas[n + k + 1] = self.subtract(xv, U[k])

        return sigmas

    def _compute_weights(self) -> None:
        n = self.n
        lambda_ = self.alpha**2 * (n + self.kappa) - n

        c = 0.5 / (n + lambda_)
        self.Wc: F64["2*n_x+1"] = np.full(2 * n + 1, c)
        self.Wm: F64["2*n_x+1"] = np.full(2 * n + 1, c)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        self.Wm[0] = lambda_ / (n + lambda_)


class JulierSigmaPoints(Generic[n_x]):
    """Julier's sigma points: 2*n+1 points of an n-state."""

    def __init__(
        self,
        n: int,
        kappa: float = 0.0,
        sqrt_method: Callable[[F64[n_x, n_x]], F64[n_x, n_x]] | None = None,
        subtract: Callable[[F64[n_x], F64[n_x]], F64[n_x]] | None = None,
    ) -> None:
        self.n = n
        self.kappa = kappa
        self.sqrt: Callable[[F64[n_x, n_x]], F64[n_x, n_x]]
        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method

        self.subtract: Callable[[F64[n_x], F64[n_x]], F64[n_x]]
        if subtract is None:
            self.subtract = np.subtract
        else:
            self.subtract = subtract

        self._compute_weights()

    def num_sigmas(self) -> int:
        return 2 * self.n + 1

    def sigma_points(self, x: F64[n_x] | float, P: F64[n_x, n_x] | float) -> F64["2*n_x+1 n_x"]:
        if self.n != np.size(x):
            raise ValueError("expected size(x) {}, but size is {}".format(self.n, np.size(x)))

        xv: F64[n_x] = np.asarray([x]) if np.isscalar(x) else x  # type: ignore[assignment]  # NUMTY: a scalar x passed the size check above, so n_x is 1 here; no annotation can tie (1,) to n_x in this branch

        n = np.size(xv)  # dimension of problem

        Pm: F64[n_x, n_x] = np.eye(n) * P if np.isscalar(P) else np.atleast_2d(P)

        sigmas: F64["2*n_x+1 n_x"] = np.zeros((2 * n + 1, n))

        # implements U'*U = (n+kappa)*P. Returns upper triangular matrix.
        U = self.sqrt((n + self.kappa) * Pm)

        sigmas[0] = xv
        for k in range(n):
            sigmas[k + 1] = self.subtract(xv, -U[k])
            sigmas[n + k + 1] = self.subtract(xv, U[k])
        return sigmas

    def _compute_weights(self) -> None:
        n = self.n
        k = self.kappa

        self.Wm: F64["2*n_x+1"] = np.full(2 * n + 1, 0.5 / (n + k))
        self.Wm[0] = k / (n + k)
        self.Wc: F64["2*n_x+1"] = self.Wm


class SimplexSigmaPoints(Generic[n_x]):
    """Moireau and Chapelle's simplex sigma points: n+1 points of an n-state."""

    def __init__(
        self,
        n: int,
        alpha: float = 1,
        sqrt_method: Callable[[F64[n_x, n_x]], F64[n_x, n_x]] | None = None,
        subtract: Callable[[F64[n_x, Literal[1]], F64["n_x n_x+1"]], F64["n_x n_x+1"]] | None = None,
    ) -> None:
        self.n = n
        self.alpha = alpha
        self.sqrt: Callable[[F64[n_x, n_x]], F64[n_x, n_x]]
        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method

        self.subtract: Callable[[F64[n_x, Literal[1]], F64["n_x n_x+1"]], F64["n_x n_x+1"]]
        if subtract is None:
            self.subtract = np.subtract
        else:
            self.subtract = subtract

        self._compute_weights()

    def num_sigmas(self) -> int:
        return self.n + 1

    def sigma_points(self, x: F64[n_x] | float, P: F64[n_x, n_x] | float) -> F64["n_x+1 n_x"]:
        if self.n != np.size(x):
            raise ValueError("expected size(x) {}, but size is {}".format(self.n, np.size(x)))

        n = self.n

        xv: F64[n_x] = np.asarray([x]) if np.isscalar(x) else x  # type: ignore[assignment]  # NUMTY: a scalar x passed the size check above, so n_x is 1 here; no annotation can tie (1,) to n_x in this branch
        xc = xv.reshape(-1, 1)

        Pm: F64[n_x, n_x] = np.eye(n) * P if np.isscalar(P) else np.atleast_2d(P)

        U = self.sqrt(Pm)

        lambda_ = n / (n + 1)
        Istar = np.array([[-1 / np.sqrt(2 * lambda_), 1 / np.sqrt(2 * lambda_)]])

        for d in range(2, n + 1):
            row = np.ones((1, Istar.shape[1] + 1)) * 1.0 / np.sqrt(lambda_ * d * (d + 1))
            row[0, -1] = -d / np.sqrt(lambda_ * d * (d + 1))
            Istar = np.r_[np.c_[Istar, np.zeros((Istar.shape[0]))], row]

        I: F64["n_x n_x+1"] = np.sqrt(n) * Istar
        scaled_unitary = (U.T).dot(I)

        sigmas = self.subtract(xc, -scaled_unitary)
        return sigmas.T

    def _compute_weights(self) -> None:
        n = self.n
        c = 1.0 / (n + 1)
        self.Wm: F64["n_x+1"] = np.full(n + 1, c)
        self.Wc: F64["n_x+1"] = self.Wm


class InverseFn(Protocol):
    """A matrix inverse, for the `inv` attribute (np.linalg.inv by default)."""

    def __call__(self, a: F64[n_dim, n_dim], /) -> F64[n_dim, n_dim]: ...


class SigmaPointsFn(Protocol[n_x, n_sig]):
    """What UnscentedKalmanFilter needs of a sigma point scheme."""

    Wm: F64[n_sig]
    Wc: F64[n_sig]

    def num_sigmas(self) -> int: ...

    def sigma_points(self, x: F64[n_x], P: F64[n_x, n_x]) -> F64[n_sig, n_x]: ...


# --------------------------------------------------------------------------
# UKF.py
# --------------------------------------------------------------------------


class UnscentedKalmanFilter(Generic[n_x, n_z, n_sig]):
    """Unscented Kalman filter with n_x states and n_z measurements."""

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        dt: float,
        hx: Callable[[F64[n_x]], F64[n_z]],
        fx: Callable[[F64[n_x], float], F64[n_x]],
        points: SigmaPointsFn[n_x, n_sig],
        sqrt_fn: Callable[[F64[n_x, n_x]], F64[n_x, n_x]] | None = None,
        x_mean_fn: Callable[[F64[n_sig, n_x], F64[n_sig]], F64[n_x]] | None = None,
        z_mean_fn: Callable[[F64[n_sig, n_z], F64[n_sig]], F64[n_z]] | None = None,
        residual_x: Callable[[F64[n_x], F64[n_x]], F64[n_x]] | None = None,
        residual_z: Callable[[F64[n_z], F64[n_z]], F64[n_z]] | None = None,
        state_add: Callable[[F64[n_x], F64[n_x]], F64[n_x]] | None = None,
    ) -> None:
        self.x: F64[n_x] = np.zeros(dim_x)
        self.P: F64[n_x, n_x] = np.eye(dim_x)
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
        self.Q: F64[n_x, n_x] = np.eye(dim_x)
        self.R: F64[n_z, n_z] = np.eye(dim_z)
        self._dim_x = dim_x
        self._dim_z = dim_z
        self.points_fn = points
        self._dt = dt
        self._num_sigmas = points.num_sigmas()
        self.hx = hx
        self.fx = fx
        self.x_mean = x_mean_fn
        self.z_mean = z_mean_fn

        # Only computed only if requested via property
        self._log_likelihood: float | None = log(sys.float_info.min)
        self._likelihood: float | None = sys.float_info.min
        self._mahalanobis: float | None = None

        self.msqrt: Callable[[F64[n_x, n_x]], F64[n_x, n_x]]
        if sqrt_fn is None:
            self.msqrt = cholesky
        else:
            self.msqrt = sqrt_fn

        # weights for the means and covariances.
        self.Wm, self.Wc = points.Wm, points.Wc

        self.residual_x: Callable[[F64[n_x], F64[n_x]], F64[n_x]]
        if residual_x is None:
            self.residual_x = np.subtract
        else:
            self.residual_x = residual_x

        self.residual_z: Callable[[F64[n_z], F64[n_z]], F64[n_z]]
        if residual_z is None:
            self.residual_z = np.subtract
        else:
            self.residual_z = residual_z

        self.state_add: Callable[[F64[n_x], F64[n_x]], F64[n_x]]
        if state_add is None:
            self.state_add = np.add
        else:
            self.state_add = state_add

        # sigma points transformed through f(x) and h(x)
        self.sigmas_f: F64[n_sig, n_x] = np.zeros((self._num_sigmas, self._dim_x))
        self.sigmas_h: F64[n_sig, n_z] = np.zeros((self._num_sigmas, self._dim_z))

        self.K: F64[n_x, n_z] = np.zeros((dim_x, dim_z))  # Kalman gain
        self.y: F64[n_z] = np.zeros((dim_z))  # residual
        self.z: Arr[np.object_, n_z, Literal[1]] | F64[n_z] = np.array([[None] * dim_z]).T  # measurement
        self.S: F64[n_z, n_z] = np.zeros((dim_z, dim_z))  # system uncertainty
        self.SI: F64[n_z, n_z] = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        self.inv: InverseFn = np.linalg.inv

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(
        self,
        dt: float | None = None,
        UT: UnscentedTransformFn | None = None,
        fx: Callable[[F64[n_x], float], F64[n_x]] | None = None,
    ) -> None:
        if dt is None:
            dt = self._dt

        if UT is None:
            UT = unscented_transform

        # calculate sigma points for given mean and covariance
        self.compute_process_sigmas(dt, fx)

        # and pass sigmas through the unscented transform to compute prior
        self.x, self.P = UT(self.sigmas_f, self.Wm, self.Wc, self.Q, self.x_mean, self.residual_x)

        # update sigma points to reflect the new variance of the points
        self.sigmas_f = self.points_fn.sigma_points(self.x, self.P)

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(
        self,
        z: F64[n_z] | None,
        R: F64[n_z, n_z] | float | None = None,
        UT: UnscentedTransformFn | None = None,
        hx: Callable[[F64[n_x]], F64[n_z]] | None = None,
    ) -> None:
        if z is None:
            self.z = np.array([[None] * self._dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if hx is None:
            hx = self.hx

        if UT is None:
            UT = unscented_transform

        # upstream rebinds the parameter R; numty refuses np.eye(int) there
        # (a parameter is not an annotated local), so the matrix gets its own name
        Rm: F64[n_z, n_z]
        if R is None:
            Rm = self.R
        elif np.isscalar(R):
            Rm = np.eye(self._dim_z) * R  # type: ignore[assignment]  # NUMTY: an int-sized builder fits a named size only in `x: T = value`, not in a later assignment to a declared local
        else:
            Rm = R

        # pass prior sigmas through h(x) to get measurement sigmas
        sigmas_h: list[F64[n_z]] = []
        for s in self.sigmas_f:
            sigmas_h.append(hx(s))

        self.sigmas_h = np.atleast_2d(sigmas_h)

        # mean and covariance of prediction passed through unscented transform
        zp, self.S = UT(self.sigmas_h, self.Wm, self.Wc, Rm, self.z_mean, self.residual_z)
        self.SI = self.inv(self.S)

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)

        self.K = np.dot(Pxz, self.SI)  # Kalman gain
        self.y = self.residual_z(z, zp)  # residual

        # update Gaussian state estimate (x, P)
        self.x = self.state_add(self.x, np.dot(self.K, self.y))
        self.P = self.P - np.dot(self.K, np.dot(self.S, self.K.T))

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def cross_variance(
        self,
        x: F64[n_x],
        z: F64[n_z],
        sigmas_f: F64[n_sig, n_x],
        sigmas_h: F64[n_sig, n_z],
    ) -> F64[n_x, n_z]:
        """Cross variance Pxz of the state and the measurement sigma points."""
        Pxz = np.zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
        N = sigmas_f.shape[0]
        for i in range(N):
            dx = self.residual_x(sigmas_f[i], x)
            dz = self.residual_z(sigmas_h[i], z)
            Pxz += self.Wc[i] * np.outer(dx, dz)
        return Pxz

    def compute_process_sigmas(
        self,
        dt: float,
        fx: Callable[[F64[n_x], float], F64[n_x]] | None = None,
    ) -> None:
        """Sigma points of (x, P) passed through fx, into sigmas_f."""
        if fx is None:
            fx = self.fx

        # calculate sigma points for given mean and covariance
        sigmas = self.points_fn.sigma_points(self.x, self.P)

        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = fx(s, dt)

    def batch_filter(
        self,
        zs: F64[n_steps, n_z],
        Rs: Sequence[F64[n_z, n_z]] | None = None,
        dts: Sequence[float] | None = None,
        UT: UnscentedTransformFn | None = None,
    ) -> tuple[F64[n_steps, n_x], F64[n_steps, n_x, n_x]]:
        """Run predict/update over every measurement; posterior means and covariances."""
        try:
            z = zs[0]
        except TypeError:
            raise TypeError("zs must be list-like")

        if self._dim_z == 1:
            if not (np.isscalar(z) or (z.ndim == 1 and len(z) == 1)):
                raise TypeError("zs must be a list of scalars or 1D, 1 element arrays")
        else:
            if len(z) != self._dim_z:
                raise TypeError("each element in zs must be a 1D array of length {}".format(self._dim_z))

        z_n = len(zs)

        if Rs is None:
            Rs = [self.R] * z_n

        if dts is None:
            dts = [self._dt] * z_n

        # mean estimates from Kalman Filter (x is always 1-D here)
        means: F64[n_steps, n_x] = np.zeros((z_n, self._dim_x))

        # state covariances from Kalman Filter
        covariances: F64[n_steps, n_x, n_x] = np.zeros((z_n, self._dim_x, self._dim_x))

        for i, (zi, r, dt) in enumerate(zip(zs, Rs, dts)):
            self.predict(dt=dt, UT=UT)
            self.update(zi, r, UT=UT)
            means[i, :] = self.x
            covariances[i, :, :] = self.P

        return (means, covariances)

    def rts_smoother(
        self,
        Xs: F64[n_steps, n_x],
        Ps: F64[n_steps, n_x, n_x],
        Qs: Sequence[F64[n_x, n_x]] | None = None,
        dts: Sequence[float] | float | None = None,
        UT: UnscentedTransformFn | None = None,
    ) -> tuple[F64[n_steps, n_x], F64[n_steps, n_x, n_x], F64[n_steps, n_x, n_x]]:
        """Rauch-Tung-Striebel smoother: smoothed means, covariances and gains."""
        if len(Xs) != len(Ps):
            raise ValueError("Xs and Ps must have the same length")

        n, dim_x = Xs.shape

        if dts is None:
            dts = [self._dt] * n
        elif np.isscalar(dts):
            dts = [dts] * n

        if Qs is None:
            Qs = [self.Q] * n

        if UT is None:
            UT = unscented_transform

        # smoother gain
        Ks = np.zeros((n, dim_x, dim_x))

        num_sigmas = self._num_sigmas

        xs, ps = Xs.copy(), Ps.copy()
        sigmas_f: F64[n_sig, n_x] = np.zeros((num_sigmas, dim_x))

        for k in reversed(range(n - 1)):
            # create sigma points from state estimate, pass through state func
            sigmas = self.points_fn.sigma_points(xs[k], ps[k])
            for i in range(num_sigmas):
                sigmas_f[i] = self.fx(sigmas[i], dts[k])

            xb, Pb = UT(sigmas_f, self.Wm, self.Wc, self.Q, self.x_mean, self.residual_x)

            # compute cross variance (upstream starts from the int 0)
            Pxb = np.zeros((dim_x, dim_x))
            for i in range(num_sigmas):
                y = self.residual_x(sigmas_f[i], xb)
                z = self.residual_x(sigmas[i], Xs[k])
                Pxb += self.Wc[i] * np.outer(z, y)

            # compute gain
            K = np.dot(Pxb, self.inv(Pb))

            # update the smoothed estimates
            xs[k] += np.dot(K, self.residual_x(xs[k + 1], xb))
            ps[k] += np.dot(K, ps[k + 1] - Pb).dot(K.T)
            Ks[k] = K

        return (xs, ps, Ks)

    @property
    def log_likelihood(self) -> float:
        """log-likelihood of the last measurement."""
        if self._log_likelihood is None:
            self._log_likelihood = logpdf(x=self.y, cov=self.S)
        return self._log_likelihood

    @property
    def likelihood(self) -> float:
        """Computed from the log-likelihood."""
        if self._likelihood is None:
            self._likelihood = exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self) -> float:
        """Mahalanobis distance of the innovation."""
        if self._mahalanobis is None:
            self._mahalanobis = sqrt(np.dot(np.dot(self.y.T, self.SI), self.y).item())
        return self._mahalanobis
