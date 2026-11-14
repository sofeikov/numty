# MUTANT: S1_2 slices r[0:dim_z, 0:dim_z + 1] (off by one column)
"""Information filter and square-root Kalman filter, extracted from filterpy.

Upstream: filterpy v2.0.0, commit 1337082 ("Declare long description content
type"), https://github.com/rlabbe/filterpy

- filterpy/kalman/information_filter.py:30-379  InformationFilter
  (__init__, update, predict, batch_filter, F, P)
- filterpy/kalman/square_root.py:28-340  SquareRootKalmanFilter
  (__init__, update, predict, residual_of, measurement_of_state and the
  Q/P/R/S square-root properties)
- inlined helpers: filterpy/common/helpers.py:324 reshape_z,
  filterpy/stats/stats.py:131 logpdf (scipy.stats.multivariate_normal.logpdf
  with allow_singular=True, re-implemented with numpy).
- scipy.linalg.qr / cholesky / pinv are replaced by numpy wrappers with the
  same results (``_qr_full``, ``_cholesky_lower``, ``_pinv``).

Sizes: n_x = dim_x (state), n_z = dim_z (measurement), n_u = dim_u (control).
Vectors are column matrices (n, 1), as in filterpy.

Deliberate differences from upstream (see NOTES.md):
- ``InformationFilter.update`` computes the log-likelihood of the residual
  under its own covariance H P H' + R (n_z x n_z).  Upstream passes the
  (n_x, n_x) information matrix as the covariance of the (n_z,) residual,
  which raises for dim_z != dim_x (upstream bug).
- ``_F`` / ``_F_inv`` start as zero matrices instead of the float 0.
  (upstream predict() raises either way before F is set).
- bare ``except:`` is narrowed to ``np.linalg.LinAlgError`` (the only error
  ``np.linalg.inv`` raises on a well-shaped input).
- ``np.isscalar(R)`` is spelled ``isinstance(R, (int, float, np.generic))``.
- ``batch_filter`` raises ``NotImplementedError`` first thing upstream; the
  unreachable body is dropped.
- ``InformationFilter.inv`` is a method calling ``np.linalg.inv`` (upstream:
  the attribute ``self.inv = np.linalg.inv``; numty types calls through a
  function held in an attribute as ``ndarray[Any, Any]``).
- the size TypeVars are bound by ``dim_x: n_x`` and ``dim_z: n_z``;
  ``dim_u`` stays ``int = 0`` (a TypeVar parameter cannot default to 0).

Upstream behaviour kept as is although it looks wrong (see NOTES.md):
``SquareRootKalmanFilter.P_post`` reads the prior factor, its constructor
checks dim_z twice and never dim_x, and ``predict()`` with a matrix ``B`` and
the default ``u=0`` broadcasts ``x`` to (n_x, n_u) in both filters.
"""

from __future__ import annotations

import math
import sys
from copy import deepcopy
from typing import Any, Generic, TypeVar

import numpy as np

from numty import F64, Arr

n_x = TypeVar("n_x", bound=int)
n_z = TypeVar("n_z", bound=int)
n_u = TypeVar("n_u", bound=int)
d = TypeVar("d", bound=int)  # size of one measurement vector in reshape_z


# ---------------------------------------------------------------------------
# helpers (filterpy.common / filterpy.stats / scipy.linalg stand-ins)
# ---------------------------------------------------------------------------


def reshape_z(z: F64["d 1"], dim_z: d) -> F64["d 1"]:
    """ensure z is a (dim_z, 1) shaped vector (filterpy.common.reshape_z, ndim=2)"""
    z2 = np.atleast_2d(z)
    if z2.shape[1] == dim_z:
        z2 = z2.T  # type: ignore[assignment]  # NUMTY: a shape test does not narrow; the check below makes the result (d, 1)
    if z2.shape != (dim_z, 1):
        raise ValueError(
            "z (shape {}) must be convertible to shape ({}, 1)".format(z2.shape, dim_z)
        )
    return z2


def logpdf(x: F64["k 1"], cov: F64["k k"]) -> float:
    """log N(x; 0, cov), allow_singular=True (scipy multivariate_normal.logpdf)."""
    flat_x = x.flatten()
    s, u = np.linalg.eigh(cov)
    eps = 1e6 * np.finfo(np.float64).eps * np.max(np.abs(s))
    if np.min(s) < -eps:
        raise ValueError("the input matrix must be positive semidefinite")
    keep = s > eps
    s_pinv = np.array([0.0 if v <= eps else 1.0 / v for v in s])
    U = u * np.sqrt(s_pinv)
    log_pdet = float(np.sum(np.log(s[keep])))
    rank = int(np.count_nonzero(keep))
    maha = float(np.sum(np.square(flat_x @ U)))
    return -0.5 * (rank * math.log(2 * np.pi) + log_pdet + maha)


def _qr_full(a: F64["m n"]) -> tuple[F64["m m"], F64["m n"]]:
    """scipy.linalg.qr(a) (mode='full'): Q (m, m), R (m, n)."""
    q, r = np.linalg.qr(a, mode="complete")
    return q, r


def _cholesky_lower(a: F64["d d"]) -> F64["d d"]:
    """scipy.linalg.cholesky(a, lower=True)."""
    return np.linalg.cholesky(a)


def _pinv(a: F64["m n"]) -> F64["n m"]:
    """scipy.linalg.pinv(a)."""
    return np.linalg.pinv(a)


# ---------------------------------------------------------------------------
# InformationFilter (filterpy/kalman/information_filter.py)
# ---------------------------------------------------------------------------


class InformationFilter(Generic[n_x, n_z, n_u]):
    """Linear information filter: carries the inverse covariance P_inv."""

    def __init__(
        self, dim_x: n_x, dim_z: n_z, dim_u: int = 0, compute_log_likelihood: bool = True
    ) -> None:
        if dim_x < 1:
            raise ValueError("dim_x must be 1 or greater")
        if dim_z < 1:
            raise ValueError("dim_z must be 1 or greater")
        if dim_u < 0:
            raise ValueError("dim_u must be 0 or greater")

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x: F64["n_x 1"] = np.zeros((dim_x, 1))  # state
        self.P_inv: F64["n_x n_x"] = np.eye(dim_x)  # uncertainty covariance
        self.Q: F64["n_x n_x"] = np.eye(dim_x)  # process uncertainty
        self.B: F64["n_x n_u"] | float = 0.0  # control transition matrix
        self._F: F64["n_x n_x"] = np.zeros((dim_x, dim_x))  # state transition matrix
        self._F_inv: F64["n_x n_x"] = np.zeros((dim_x, dim_x))
        self.H: F64["n_z n_x"] = np.zeros((dim_z, dim_x))  # measurement function
        self.R_inv: F64["n_z n_z"] = np.eye(dim_z)  # state uncertainty

        # gain and residual are computed during the innovation step
        self.K: F64["n_x n_z"] | float = 0.0  # kalman gain
        self.y: F64["n_z 1"] = np.zeros((dim_z, 1))
        self.z: F64["n_z 1"] | None = np.zeros((dim_z, 1))
        self.S: F64["n_x n_x"] | float = 0.0  # system uncertainty

        self._I: F64["n_x n_x"] = np.eye(dim_x)
        self._no_information = False

        self.compute_log_likelihood = compute_log_likelihood
        self.log_likelihood = math.log(sys.float_info.min)
        self.likelihood = sys.float_info.min

        # save priors and posteriors
        self.x_prior: F64["n_x 1"] = np.copy(self.x)
        self.P_inv_prior: F64["n_x n_x"] = np.copy(self.P_inv)
        self.x_post: F64["n_x 1"] = np.copy(self.x)
        self.P_inv_post: F64["n_x n_x"] = np.copy(self.P_inv)

    def inv(self, a: F64["d d"]) -> F64["d d"]:
        """upstream: ``self.inv = np.linalg.inv`` (swappable for pinv)."""
        return np.linalg.inv(a)

    def update(
        self, z: F64["n_z 1"] | None, R_inv: F64["n_z n_z"] | float | None = None
    ) -> None:
        """Add a new measurement (z) to the filter. If z is None, nothing changes."""
        if z is None:
            self.z = None
            self.x_post = self.x.copy()
            self.P_inv_post = self.P_inv.copy()
            return

        if R_inv is None:
            R_inv = self.R_inv
        elif isinstance(R_inv, (int, float, np.generic)):
            R_inv = np.eye(self.dim_z) * R_inv

        H = self.H
        H_T = H.T
        P_inv = self.P_inv
        x = self.x

        if self._no_information:
            self.x = np.dot(P_inv, x) + np.dot(H_T, R_inv).dot(z)
            self.P_inv = P_inv + np.dot(H_T, R_inv).dot(H)
            self.log_likelihood = math.log(sys.float_info.min)
            self.likelihood = sys.float_info.min

        else:
            # y = z - Hx: residual between measurement and prediction
            self.y = z - np.dot(H, x)

            # posterior information matrix (upstream calls it S)
            S = P_inv + np.dot(H_T, R_inv).dot(H)
            self.S = S
            K = np.dot(self.inv(S), H_T).dot(R_inv)
            self.K = K

            # x = x + Ky
            self.x = x + np.dot(K, self.y)
            self.P_inv = P_inv + np.dot(H_T, R_inv).dot(H)

            self.z = np.copy(reshape_z(z, self.dim_z))

            if self.compute_log_likelihood:
                # covariance of the residual: H P H' + R
                S_y = np.dot(H, self.inv(P_inv)).dot(H_T) + self.inv(R_inv)
                self.log_likelihood = logpdf(x=self.y, cov=S_y)
                self.likelihood = math.exp(self.log_likelihood)
                if self.likelihood == 0:
                    self.likelihood = sys.float_info.min

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_inv_post = self.P_inv.copy()

    def predict(self, u: F64["n_u 1"] | float = 0) -> None:
        """Predict next position."""
        # x = Fx + Bu
        A = np.dot(self._F_inv.T, self.P_inv).dot(self._F_inv)
        try:
            AI = self.inv(A)
            invertable = True
            if self._no_information:
                try:
                    self.x = np.dot(self.inv(self.P_inv), self.x)
                except np.linalg.LinAlgError:
                    self.x = np.dot(0, self.x)
                self._no_information = False
        except np.linalg.LinAlgError:
            invertable = False
            self._no_information = True

        if invertable:
            self.x = np.dot(self._F, self.x) + np.dot(self.B, u)
            self.P_inv = self.inv(AI + self.Q)

            # save priors
            self.P_inv_prior = np.copy(self.P_inv)
            self.x_prior = np.copy(self.x)
        else:
            I_PF = self._I - np.dot(self.P_inv, self._F_inv)
            FTI = self.inv(self._F.T)
            FTIX = np.dot(FTI, self.x)
            AQI = self.inv(A + self.Q)
            self.x = np.dot(FTI, np.dot(I_PF, AQI).dot(FTIX))

            # save priors
            self.x_prior = np.copy(self.x)
            self.P_inv_prior = np.copy(AQI)

    def batch_filter(
        self,
        zs: list[F64["n_z 1"] | None],
        Rs: list[F64["n_z n_z"] | None] | None = None,
        update_first: bool = False,
        saver: Any = None,
    ) -> tuple[F64["n_steps n_x 1"], F64["n_steps n_x n_x"]]:
        """Batch processes a sequence of measurements (not implemented upstream)."""
        raise NotImplementedError("this is not implemented yet")

    @property
    def F(self) -> F64["n_x n_x"]:
        """State Transition matrix"""
        return self._F

    @F.setter
    def F(self, value: F64["n_x n_x"]) -> None:
        self._F = value
        self._F_inv = self.inv(self._F)

    @property
    def P(self) -> F64["n_x n_x"]:
        """State covariance matrix"""
        return self.inv(self.P_inv)


# ---------------------------------------------------------------------------
# SquareRootKalmanFilter (filterpy/kalman/square_root.py)
# ---------------------------------------------------------------------------


class SquareRootKalmanFilter(Generic[n_x, n_z, n_u]):
    """Kalman filter that propagates the Cholesky factor of P."""

    def __init__(self, dim_x: n_x, dim_z: n_z, dim_u: int = 0) -> None:
        # upstream checks dim_z twice and never dim_x (kept as is)
        if dim_z < 1:
            raise ValueError("dim_x must be 1 or greater")
        if dim_z < 1:
            raise ValueError("dim_x must be 1 or greater")
        if dim_u < 0:
            raise ValueError("dim_x must be 0 or greater")

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x: F64["n_x 1"] = np.zeros((dim_x, 1))  # state
        self._P: F64["n_x n_x"] = np.eye(dim_x)  # uncertainty covariance
        self._P1_2: F64["n_x n_x"] = np.eye(dim_x)  # sqrt uncertainty covariance
        self._Q: F64["n_x n_x"] = np.eye(dim_x)  # process uncertainty
        self._Q1_2: F64["n_x n_x"] = np.eye(dim_x)  # sqrt process uncertainty
        self.B: F64["n_x n_u"] | float = 0.0  # control transition matrix
        self.F: F64["n_x n_x"] = np.eye(dim_x)  # state transition matrix
        self.H: F64["n_z n_x"] = np.zeros((dim_z, dim_x))  # measurement function
        self._R1_2: F64["n_z n_z"] = np.eye(dim_z)  # sqrt state uncertainty
        self._R: F64["n_z n_z"] = np.eye(dim_z)  # state uncertainty
        self.z: Arr[Any, n_z, 1] = np.array([[None] * self.dim_z]).T

        self.K: F64["n_x n_z"] = np.zeros((dim_x, dim_z))  # kalman gain
        self.S1_2: F64["n_z n_z"] = np.zeros((dim_z, dim_z))  # sqrt system uncertainty
        self.SI1_2: F64["n_z n_z"] = np.zeros((dim_z, dim_z))  # inverse sqrt

        self.y: F64["n_z 1"] = np.zeros((dim_z, 1))

        self._I: F64["n_x n_x"] = np.eye(dim_x)

        self.M: F64["n_z+n_x n_z+n_x"] = np.zeros((dim_z + dim_x, dim_z + dim_x))

        # copy prior and posterior
        self.x_prior: F64["n_x 1"] = np.copy(self.x)
        self._P1_2_prior: F64["n_x n_x"] = np.copy(self._P1_2)
        self.x_post: F64["n_x 1"] = np.copy(self.x)
        self._P1_2_post: F64["n_x n_x"] = np.copy(self._P1_2)

    def update(
        self, z: F64["n_z 1"] | None, R2: F64["n_z n_z"] | float | None = None
    ) -> None:
        """Add a new measurement (z) to the filter. If z is None, nothing changes.

        R2 is the square root of the measurement noise for this one call.
        """
        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self._P1_2_post = np.copy(self._P1_2)
            return

        if R2 is None:
            R2 = self._R1_2
        elif isinstance(R2, (int, float, np.generic)):
            R2 = np.eye(self.dim_z) * R2

        dim_z = self.dim_z
        M = self.M

        M[0:dim_z, 0:dim_z] = R2.T
        M[dim_z:, 0:dim_z] = np.dot(self.H, self._P1_2).T
        M[dim_z:, dim_z:] = self._P1_2.T

        _, r_decomp = _qr_full(M)
        self.S1_2 = r_decomp[0:dim_z, 0:dim_z + 1].T  # type: ignore[assignment]  # NUMTY: [0:n_z] of an n_z+n_x axis is ?U, not n_z
        self.SI1_2 = _pinv(self.S1_2)
        self.K = np.dot(r_decomp[0:dim_z, dim_z:].T, self.SI1_2)  # type: ignore[assignment]  # NUMTY: [n_z:] of an n_z+n_x axis is ?U, not n_x

        # y = z - Hx
        self.y = z - np.dot(self.H, self.x)

        # x = x + Ky
        self.x += np.dot(self.K, self.y)
        self._P1_2 = r_decomp[dim_z:, dim_z:].T  # type: ignore[assignment]  # NUMTY: [n_z:] of an n_z+n_x axis is ?U, not n_x

        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self._P1_2_post = np.copy(self._P1_2)

    def predict(self, u: F64["n_u 1"] | float = 0) -> None:
        """Predict next state (prior) with the square-root propagation."""
        # x = Fx + Bu
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)

        # P = FPF' + Q
        _, P2 = _qr_full(np.hstack([np.dot(self.F, self._P1_2), self._Q1_2]).T)
        self._P1_2 = P2[: self.dim_x, : self.dim_x].T  # type: ignore[assignment]  # NUMTY: [:n_x] of a 2*n_x axis is ?U, not n_x

        # copy prior
        self.x_prior = np.copy(self.x)
        self._P1_2_prior = np.copy(self._P1_2)

    def residual_of(self, z: F64["n_z 1"]) -> F64["n_z 1"]:
        """returns the residual for the given measurement (z)."""
        return z - np.dot(self.H, self.x)

    def measurement_of_state(self, x: F64["n_x 1"]) -> F64["n_z 1"]:
        """converts a state into a measurement."""
        return np.dot(self.H, x)

    @property
    def Q(self) -> F64["n_x n_x"]:
        """Process uncertainty"""
        return np.dot(self._Q1_2, self._Q1_2.T)

    @Q.setter
    def Q(self, value: F64["n_x n_x"]) -> None:
        self._Q = value
        self._Q1_2 = _cholesky_lower(self._Q)

    @property
    def Q1_2(self) -> F64["n_x n_x"]:
        """Sqrt Process uncertainty"""
        return self._Q1_2

    @property
    def P(self) -> F64["n_x n_x"]:
        """covariance matrix"""
        return np.dot(self._P1_2, self._P1_2.T)

    @P.setter
    def P(self, value: F64["n_x n_x"]) -> None:
        self._P = value
        self._P1_2 = _cholesky_lower(self._P)

    @property
    def P_prior(self) -> F64["n_x n_x"]:
        """covariance matrix of the prior"""
        return np.dot(self._P1_2_prior, self._P1_2_prior.T)

    @property
    def P_post(self) -> F64["n_x n_x"]:
        """covariance matrix of the posterior (upstream reads the prior factor)"""
        return np.dot(self._P1_2_prior, self._P1_2_prior.T)

    @property
    def P1_2(self) -> F64["n_x n_x"]:
        """sqrt of covariance matrix"""
        return self._P1_2

    @property
    def R(self) -> F64["n_z n_z"]:
        """measurement uncertainty"""
        return np.dot(self._R1_2, self._R1_2.T)

    @R.setter
    def R(self, value: F64["n_z n_z"]) -> None:
        self._R = value
        self._R1_2 = _cholesky_lower(self._R)

    @property
    def R1_2(self) -> F64["n_z n_z"]:
        """sqrt of measurement uncertainty"""
        return self._R1_2

    @property
    def S(self) -> F64["n_z n_z"]:
        """system uncertainty (P projected to measurement space)"""
        return np.dot(self.S1_2, self.S1_2.T)

    @property
    def SI(self) -> F64["n_z n_z"]:
        """inverse system uncertainty"""
        return np.dot(self.SI1_2.T, self.SI1_2)
