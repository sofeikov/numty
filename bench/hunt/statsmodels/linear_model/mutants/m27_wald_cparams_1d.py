# MUTANT: wald_test: cparams = R @ params without params[:, None]
"""GLS, WLS, OLS and GLSAR fitting from statsmodels, annotated for numty.

Extracted from statsmodels v0.15.0 (commit 278ff99):

- statsmodels/regression/linear_model.py
    - ``_get_sigma`` (lines 184-235)
    - ``RegressionModel`` initialize, df_model, df_resid, fit (pinv and qr),
      predict (lines 238-466)
    - ``GLS`` __init__, whiten, loglike, hessian_factor (lines 585-734)
    - ``WLS`` __init__, whiten, loglike, hessian_factor (lines 829-941)
    - ``OLS`` loglike, whiten, score, _setup_score_hess, hessian,
      hessian_factor, _fit_ridge (lines 1047-1203, 1312-1349)
    - ``GLSAR`` __init__, iterative_fit, whiten (lines 1412-1536)
    - ``yule_walker`` (lines 1547-1709), the plain-tuple return
    - ``RegressionResults`` fittedvalues, wresid, resid, scale, ssr,
      centered_tss, uncentered_tss, ess, rsquared, rsquared_adj, mse_model,
      mse_resid, mse_total, fvalue, bse, aic, bic, eigenvals,
      condition_number, _HCCM, _abat_diagonal, cov_HC0..3, HC0..3_se,
      resid_pearson (lines 1846-2345)
- statsmodels/base/model.py ``LikelihoodModelResults`` bse, tvalues,
  cov_params, t_test, f_test / wald_test (lines 1615-2145, the numeric
  core: the constraint matrix is taken as an array, as
  ``FormulaManager.get_linear_constraints`` returns it with zero constants)
- statsmodels/base/covtype.py ``get_robustcov_results`` for HC0..HC3
  (lines 70-170: ``cov_params_default = cov_HCx``, ``use_t`` False)
- statsmodels/base/data.py ``ModelData._handle_constant`` (lines 129-188)
- statsmodels/tools/tools.py ``pinv_extended`` (lines 188-224), ``recipr``
  (lines 227-249)

numpy only: ``scipy.linalg.cholesky`` is ``np.linalg.cholesky`` (both lower),
LAPACK ``dtrtri`` is a triangular solve against the identity, and
``scipy.linalg.toeplitz`` of a real vector is a gather by ``|i - j|``.
p-values and confidence intervals (scipy.stats quantiles) are left out.

Sizes: ``nobs`` observations, ``k_exog`` regressors, ``n_w`` whitened rows
(``nobs`` for GLS/WLS/OLS, ``nobs - p`` for GLSAR with AR order ``p``),
``q`` linear restrictions (rows of ``r_matrix``), ``m`` columns of a 2-d
array being whitened.
"""

from __future__ import annotations

import warnings
from typing import Generic, Literal, TypeVar, overload

import numpy as np
from typing_extensions import TypeIs, TypeVarTuple, Unpack

from numty import F64, Intp

nobs = TypeVar("nobs", bound=int)
k_exog = TypeVar("k_exog", bound=int)
n_w = TypeVar("n_w", bound=int)
q = TypeVar("q", bound=int)
q2 = TypeVar("q2", bound=int)
m = TypeVar("m", bound=int)
n = TypeVar("n", bound=int)
p = TypeVar("p", bound=int)
c = TypeVar("c", bound=int)
n_new = TypeVar("n_new", bound=int)
n_sv = TypeVar("n_sv", bound=int)
Ts = TypeVarTuple("Ts")


class SingularMatrixWarning(UserWarning):
    pass


class ValueWarning(UserWarning):
    pass


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _is_vec(x: F64[nobs] | F64[nobs, m]) -> TypeIs[F64[nobs]]:
    """``x.ndim == 1``, as a narrowing test for a 1-d / 2-d union."""
    return x.ndim == 1


def toeplitz(c_: F64[n]) -> F64[n, n]:
    """``scipy.linalg.toeplitz(c)`` for a real 1-d ``c`` (symmetric)."""
    i = np.arange(c_.shape[0])
    return c_[np.abs(i[:, None] - i[None, :])]


def _chol_inv_lower(sigma: F64[n, n]) -> F64[n, n]:
    """``dtrtri(cholesky(sigma, lower=True), lower=True)``: the inverse of the
    lower Cholesky factor, lower triangular."""
    chol = np.linalg.cholesky(sigma)
    if np.any(np.diag(chol) == 0):
        raise np.linalg.LinAlgError(
            "Cholesky decomposition of sigma yields a singular matrix"
        )
    return np.tril(np.linalg.solve(chol, np.eye(chol.shape[0])))


def pinv_extended(
    x: F64["n_rows n_cols"], rcond: float = 1e-15
) -> tuple[F64["n_cols n_rows"], F64["n_sv"]]:
    """The pinv of ``x`` and the singular values used to compute it."""
    x = x.conjugate()
    u, s, vt = np.linalg.svd(x, False)
    s_orig = np.copy(s)
    m_ = u.shape[0]
    n_ = vt.shape[1]
    cutoff = rcond * np.maximum.reduce(s)
    for i in range(min(n_, m_)):
        if s[i] > cutoff:
            s[i] = 1.0 / s[i]
        else:
            s[i] = 0.0
    res = np.dot(np.transpose(vt), np.multiply(s[:, np.newaxis], np.transpose(u)))
    return res, s_orig


def recipr(x: F64[Unpack[Ts]]) -> F64[Unpack[Ts]]:
    """Reciprocal of an array with entries less than or equal to 0 set to 0."""
    out = np.zeros_like(x, dtype=np.float64)
    nans = np.isnan(x)
    pos = ~nans
    pos[pos] = pos[pos] & (x[pos] > 0)
    out[pos] = 1.0 / x[pos]
    out[nans] = np.nan
    return out


def _handle_constant(
    exog: F64[nobs, k_exog], hasconst: bool | None
) -> tuple[int, int | None]:
    """``ModelData._handle_constant``: ``(k_constant, const_idx)``."""
    if hasconst is False:
        return 0, None
    check_implicit = False
    k_constant = 0
    const_idx_out: int | None = None
    exog_max = np.max(exog, axis=0)
    if not np.isfinite(exog_max).all():
        raise ValueError("exog contains inf or nans")
    exog_min = np.min(exog, axis=0)
    const_idx = np.where(exog_max == exog_min)[0].squeeze()
    k_constant = const_idx.size
    if k_constant == 1:
        if exog[:, const_idx].mean() != 0:
            const_idx_out = int(const_idx)
        else:
            check_implicit = True
    elif k_constant > 1:
        values = []
        for idx in const_idx:
            value = exog[:, idx].mean()
            if value == 1:
                k_constant = 1
                const_idx_out = int(idx)
                break
            values.append(value)
        else:
            pos = np.array(values) != 0
            if pos.any():
                k_constant = 1
                const_idx_out = int(const_idx[pos.argmax()])
            else:
                check_implicit = True
    else:
        check_implicit = True

    if check_implicit and not hasconst:
        augmented_exog = np.column_stack((np.ones(exog.shape[0]), exog))
        rank_augm = np.linalg.matrix_rank(augmented_exog)
        rank_orig = np.linalg.matrix_rank(exog)
        k_constant = int(rank_orig == rank_augm)
        const_idx_out = None
    elif hasconst:
        k_constant = 1
    return k_constant, const_idx_out


def _get_sigma(
    sigma: float | F64[nobs] | F64[nobs, nobs] | None, nobs_: nobs
) -> tuple[F64[nobs], F64[nobs]] | tuple[F64[nobs, nobs], F64[nobs, nobs]] | tuple[None, None]:
    """Validate sigma and compute the inverse of its Cholesky factor."""
    if sigma is None:
        return None, None
    arr = np.asarray(sigma).squeeze()
    if arr.ndim == 0:
        arr = np.repeat(arr, nobs_)
    if arr.ndim == 1:
        if arr.shape != (nobs_,):
            raise ValueError(
                f"Sigma must be a scalar, 1d of length {nobs_} or a 2d "
                f"array of shape {nobs_} x {nobs_}"
            )
        vec: F64[nobs] = arr
        return vec, 1 / np.sqrt(vec)
    if arr.shape != (nobs_, nobs_):
        raise ValueError(
            f"Sigma must be a scalar, 1d of length {nobs_} or a 2d "
            f"array of shape {nobs_} x {nobs_}"
        )
    mat: F64[nobs, nobs] = arr
    return mat, _chol_inv_lower(mat)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


class RegressionModel(Generic[nobs, n_w, k_exog]):
    """Base class for linear regression models."""

    def __init__(
        self, endog: F64[nobs], exog: F64[nobs, k_exog], hasconst: bool | None = None
    ) -> None:
        self.endog = endog
        self.exog = exog
        self.k_constant, self.const_idx = _handle_constant(exog, hasconst)
        self.pinv_wexog: F64[k_exog, n_w] | None = None
        self.wexog_singular_values: F64[int] | None = None
        self.normalized_cov_params: F64[k_exog, k_exog] | None = None
        self.initialize()

    def initialize(self) -> None:
        self.wexog: F64[n_w, k_exog] = self.whiten(self.exog)
        self.wendog: F64[n_w] = self.whiten(self.endog)
        self.nobs = float(self.wexog.shape[0])
        self._df_model: float | None = None
        self._df_resid: float | None = None
        self.rank: int | None = None

    @property
    def df_model(self) -> float:
        if self._df_model is None:
            if self.rank is None:
                self.rank = int(np.linalg.matrix_rank(self.exog))
            self._df_model = float(self.rank - self.k_constant)
        return self._df_model

    @df_model.setter
    def df_model(self, value: float) -> None:
        self._df_model = value

    @property
    def df_resid(self) -> float:
        if self._df_resid is None:
            if self.rank is None:
                self.rank = int(np.linalg.matrix_rank(self.exog))
            self._df_resid = self.nobs - self.rank
        return self._df_resid

    @df_resid.setter
    def df_resid(self, value: float) -> None:
        self._df_resid = value

    @overload
    def whiten(self, x: F64[nobs]) -> F64[n_w]: ...
    @overload
    def whiten(self, x: F64[nobs, m]) -> F64[n_w, m]: ...
    def whiten(self, x: F64[nobs] | F64[nobs, m]) -> F64[n_w] | F64[n_w, m]:
        raise NotImplementedError("Subclasses must implement.")

    def loglike(self, params: F64[k_exog]) -> float:
        raise NotImplementedError

    def fit(
        self,
        method: Literal["pinv", "qr"] = "pinv",
        cov_type: Literal["nonrobust", "HC0", "HC1", "HC2", "HC3"] = "nonrobust",
        use_t: bool | None = None,
    ) -> RegressionResults[nobs, n_w, k_exog]:
        if method == "pinv":
            pinv_wexog, singular_values = pinv_extended(self.wexog)
            self.pinv_wexog = pinv_wexog
            self.normalized_cov_params = np.dot(pinv_wexog, np.transpose(pinv_wexog))
            self.wexog_singular_values = singular_values
            self.rank = int(np.linalg.matrix_rank(np.diag(singular_values)))
            if self.rank < self.wexog.shape[1]:
                warnings.warn(
                    "The design matrix is rank-deficient. "
                    "The model parameters are not uniquely determined.",
                    SingularMatrixWarning,
                    stacklevel=2,
                )
            beta = np.dot(pinv_wexog, self.wendog)
        else:
            Q, R = np.linalg.qr(self.wexog)
            self.normalized_cov_params = np.linalg.inv(np.dot(R.T, R))
            self.wexog_singular_values = np.linalg.svd(R, False, False)
            self.rank = int(np.linalg.matrix_rank(R))
            self.pinv_wexog = np.linalg.pinv(self.wexog)
            self.effects = effects = np.dot(Q.T, self.wendog)
            beta = np.linalg.solve(R, effects)

        if self._df_model is None:
            self._df_model = float(self.rank - self.k_constant)
        if self._df_resid is None:
            self.df_resid = self.nobs - self.rank

        return RegressionResults(
            self, beta, self.normalized_cov_params, cov_type=cov_type, use_t=use_t
        )

    @overload
    def predict(self, params: F64[k_exog], exog: None = None) -> F64[nobs]: ...
    @overload
    def predict(self, params: F64[k_exog], exog: F64[n_new, k_exog]) -> F64[n_new]: ...
    def predict(
        self, params: F64[k_exog], exog: F64[n_new, k_exog] | None = None
    ) -> F64[nobs] | F64[n_new]:
        if exog is None:
            return np.dot(self.exog, params)
        return np.dot(exog, params)


class GLS(RegressionModel[nobs, nobs, k_exog]):
    """Generalized least squares with a known error covariance ``sigma``."""

    def __init__(
        self,
        endog: F64[nobs],
        exog: F64[nobs, k_exog],
        sigma: float | F64[nobs] | F64[nobs, nobs] | None = None,
        hasconst: bool | None = None,
    ) -> None:
        self.sigma: F64[nobs] | F64[nobs, nobs] | None
        self.cholsigmainv: F64[nobs] | F64[nobs, nobs] | None
        self.sigma, self.cholsigmainv = _get_sigma(sigma, endog.shape[0])
        super().__init__(endog, exog, hasconst=hasconst)

    @overload
    def whiten(self, x: F64[nobs]) -> F64[nobs]: ...
    @overload
    def whiten(self, x: F64[nobs, m]) -> F64[nobs, m]: ...
    def whiten(self, x: F64[nobs] | F64[nobs, m]) -> F64[nobs] | F64[nobs, m]:
        cholsigmainv = self.cholsigmainv
        if self.sigma is None or self.sigma.shape == () or cholsigmainv is None:
            return x
        elif _is_vec(cholsigmainv):
            if _is_vec(x):
                return x * cholsigmainv
            else:
                return x * cholsigmainv[:, None]
        else:
            return np.dot(cholsigmainv, x)

    def loglike(self, params: F64[k_exog]) -> float:
        nobs2 = self.nobs / 2.0
        SSR = np.sum((self.wendog - np.dot(self.wexog, params)) ** 2, axis=0)
        llf = -np.log(SSR) * nobs2
        llf -= (1 + np.log(np.pi / nobs2)) * nobs2
        sigma = self.sigma
        if sigma is not None and np.any(sigma):
            if not _is_vec(sigma):
                det = np.linalg.slogdet(sigma)
                llf -= 0.5 * det[1]
            else:
                llf -= 0.5 * np.sum(np.log(sigma))
        return float(llf)

    def hessian_factor(self, params: F64[k_exog]) -> F64[nobs]:
        cholsigmainv = self.cholsigmainv
        if self.sigma is None or self.sigma.shape == () or cholsigmainv is None:
            return np.ones(self.exog.shape[0])
        elif _is_vec(cholsigmainv):
            return cholsigmainv**2
        else:
            raise NotImplementedError(
                "hessian_factor is not available for GLS with a full "
                "(non-diagonal) sigma"
            )


class WLS(RegressionModel[nobs, nobs, k_exog]):
    """Weighted least squares: diagonal error covariance ``1 / weights``."""

    def __init__(
        self,
        endog: F64[nobs],
        exog: F64[nobs, k_exog],
        weights: float | F64[nobs] = 1.0,
        hasconst: bool | None = None,
    ) -> None:
        w = np.array(weights)
        if w.shape == ():
            w = np.repeat(w, len(endog))
        # handle case that endog might be of len == 1
        if len(w) == 1:
            w = np.array([w.squeeze()])
        else:
            w = w.squeeze()
        self.weights: F64[nobs] = w
        super().__init__(endog, exog, hasconst=hasconst)
        n_obs = self.exog.shape[0]
        if self.weights.size != n_obs and self.weights.shape[0] != n_obs:
            raise ValueError("Weights must be scalar or same length as design")

    @overload
    def whiten(self, x: F64[nobs]) -> F64[nobs]: ...
    @overload
    def whiten(self, x: F64[nobs, m]) -> F64[nobs, m]: ...
    def whiten(self, x: F64[nobs] | F64[nobs, m]) -> F64[nobs] | F64[nobs, m]:
        if _is_vec(x):
            return x * np.sqrt(self.weights)
        elif x.ndim == 2:
            return np.sqrt(self.weights)[:, None] * x
        else:
            raise ValueError("x must be 1 or 2 dimensional")

    def loglike(self, params: F64[k_exog]) -> float:
        nobs2 = self.nobs / 2.0
        SSR = np.sum((self.wendog - np.dot(self.wexog, params)) ** 2, axis=0)
        llf = -np.log(SSR) * nobs2
        llf -= (1 + np.log(np.pi / nobs2)) * nobs2
        llf += 0.5 * np.sum(np.log(self.weights))
        return float(llf)

    def hessian_factor(self, params: F64[k_exog]) -> F64[nobs]:
        return self.weights


class OLS(WLS[nobs, k_exog]):
    """Ordinary least squares."""

    def __init__(
        self, endog: F64[nobs], exog: F64[nobs, k_exog], hasconst: bool | None = None
    ) -> None:
        super().__init__(endog, exog, hasconst=hasconst)

    def loglike(self, params: F64[k_exog], scale: float | None = None) -> float:
        nobs2 = self.nobs / 2.0
        nobs_f = float(self.nobs)
        resid = self.endog - np.dot(self.exog, params)
        ssr = np.sum(resid**2)
        if scale is None:
            llf = -nobs2 * np.log(2 * np.pi) - nobs2 * np.log(ssr / nobs_f) - nobs2
        else:
            llf = -nobs2 * np.log(2 * np.pi * scale) - ssr / (2 * scale)
        return float(llf)

    @overload
    def whiten(self, x: F64[nobs]) -> F64[nobs]: ...
    @overload
    def whiten(self, x: F64[nobs, m]) -> F64[nobs, m]: ...
    def whiten(self, x: F64[nobs] | F64[nobs, m]) -> F64[nobs] | F64[nobs, m]:
        return x

    def _setup_score_hess(self) -> None:
        y = self.wendog
        self._wendog_xprod = np.sum(y * y)
        self._wexog_xprod: F64[k_exog, k_exog] = np.dot(self.wexog.T, self.wexog)
        self._wexog_x_wendog: F64[k_exog] = np.dot(self.wexog.T, y)

    def score(self, params: F64[k_exog], scale: float | None = None) -> F64[k_exog]:
        if not hasattr(self, "_wexog_xprod"):
            self._setup_score_hess()
        xtxb = np.dot(self._wexog_xprod, params)
        sdr = -self._wexog_x_wendog + xtxb
        if scale is None:
            ssr = self._wendog_xprod - 2 * np.dot(self._wexog_x_wendog.T, params)
            ssr += np.dot(params, xtxb)
            return -self.nobs * sdr / ssr
        else:
            return -sdr / scale

    def hessian(
        self, params: F64[k_exog], scale: float | None = None
    ) -> F64[k_exog, k_exog]:
        if not hasattr(self, "_wexog_xprod"):
            self._setup_score_hess()
        xtxb = np.dot(self._wexog_xprod, params)
        if scale is None:
            ssr = self._wendog_xprod - 2 * np.dot(self._wexog_x_wendog.T, params)
            ssr += np.dot(params, xtxb)
            ssrp = -2 * self._wexog_x_wendog + 2 * xtxb
            hm = self._wexog_xprod / ssr - np.outer(ssrp, ssrp) / ssr**2
            return -self.nobs * hm / 2
        else:
            return -self._wexog_xprod / scale

    def hessian_factor(self, params: F64[k_exog]) -> F64[nobs]:
        return np.ones(self.exog.shape[0])

    def fit_ridge(self, alpha: float | F64[k_exog]) -> F64[k_exog]:
        """``OLS._fit_ridge``: the ridge parameters (``RegularizedResults.params``)."""
        u, s, vt = np.linalg.svd(self.exog, False)
        v = vt.T
        q_ = np.dot(u.T, self.endog) * s
        s2 = s * s
        if isinstance(alpha, float):
            sd = s2 + alpha * self.nobs
            params = q_ / sd
            return np.dot(v, params)
        else:
            alpha = np.asarray(alpha)
            vtav = self.nobs * np.dot(vt, alpha[:, None] * v)
            d = np.diag(vtav) + s2
            np.fill_diagonal(vtav, d)
            r = np.linalg.solve(vtav, q_)
            return np.dot(v, r)


def yule_walker(
    x: F64[n], order: p, method: Literal["adjusted", "mle"] = "adjusted",
    df: int | None = None, demean: bool = True,
) -> tuple[F64[p], float]:
    """``yule_walker(..., inv=False, result_object=False)``: ``(rho, sigma)``."""
    x = np.array(x, dtype=np.float64)
    if demean:
        x -= x.mean()
    n_ = df or x.shape[0]
    adj_needed = method == "adjusted"
    # the dtype argument drops the p+1 the bare call gives (see NOTES.md)
    r: F64["p+1"] = np.zeros(order + 1, np.float64)
    r[0] = (x**2).sum() / n_
    for k in range(1, order + 1):
        r[k] = (x[0:-k] * x[k:]).sum() / (n_ - k * adj_needed)
    R = toeplitz(r[:-1])
    try:
        rho = np.linalg.solve(R, r[1:])
    except np.linalg.LinAlgError as err:
        if "Singular matrix" in str(err):
            warnings.warn("Matrix is singular. Using pinv.", SingularMatrixWarning,
                          stacklevel=2)
            rho = np.linalg.pinv(R) @ r[1:]
        else:
            raise
    sigmasq = r[0] - (r[1:] * rho).sum()
    if not np.isnan(sigmasq) and sigmasq > 0:
        sigma = float(np.sqrt(sigmasq))
    else:
        sigma = np.nan
    return rho, sigma


class GLSAR(RegressionModel[nobs, int, k_exog], Generic[nobs, k_exog, p]):
    """GLS with AR(p) errors: whitening drops the first ``p`` rows."""

    def __init__(
        self, endog: F64[nobs], exog: F64[nobs, k_exog], rho: F64[p],
        hasconst: bool | None = None,
    ) -> None:
        self.rho = rho
        self.order: p = rho.shape[0]
        super().__init__(endog, exog, hasconst=hasconst)

    @overload
    def whiten(self, x: F64[nobs]) -> F64["nobs-p"]: ...
    @overload
    def whiten(self, x: F64[nobs, m]) -> F64["nobs-p", m]: ...
    def whiten(self, x: F64[nobs] | F64[nobs, m]) -> F64["nobs-p"] | F64["nobs-p", m]:
        x = np.asarray(x, np.float64)
        _x = x.copy()
        for i in range(self.order):
            _x[(i + 1):] = _x[(i + 1):] - self.rho[i] * x[0:-(i + 1)]
        return _x[self.order:]  # type: ignore[return-value]  # NUMTY: x[p:] with p a size TypeVar is ?U, not nobs-p (#333)

    def loglike(self, params: F64[k_exog]) -> float:
        """``GLS.loglike`` with ``sigma=None``."""
        nobs2 = self.nobs / 2.0
        SSR = np.sum((self.wendog - np.dot(self.wexog, params)) ** 2, axis=0)
        llf = -np.log(SSR) * nobs2
        llf -= (1 + np.log(np.pi / nobs2)) * nobs2
        return float(llf)

    def iterative_fit(
        self, maxiter: int = 3, rtol: float = 1e-4
    ) -> tuple[RegressionResults[nobs, int, k_exog], list[F64[p]], list[F64[k_exog]], int, bool]:
        """Returns ``(results, rho history, params history, iter, converged)``."""
        converged = False
        i = -1
        history_params: list[F64[k_exog]] = []
        history_rho: list[F64[p]] = [self.rho]
        last = None
        for i in range(maxiter - 1):
            self.initialize()
            results = self.fit()
            history_params.append(results.params)
            if i == 0:
                last = results.params
            else:
                assert last is not None
                diff = np.max(np.abs(last - results.params) / np.abs(last))
                if diff < rtol:
                    converged = True
                    break
                last = results.params
            self.rho, _ = yule_walker(results.resid, order=self.order)
            history_rho.append(self.rho)
        if not converged and maxiter > 0:
            self.initialize()
        results = self.fit()
        n_iter = i + 1
        if not converged:
            history_params.append(results.params)
            n_iter += 1
        return results, history_rho, history_params, n_iter, converged


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


class RegressionResults(Generic[nobs, n_w, k_exog]):
    """The linear-algebra part of ``RegressionResults``."""

    def __init__(
        self,
        model: RegressionModel[nobs, n_w, k_exog],
        params: F64[k_exog],
        normalized_cov_params: F64[k_exog, k_exog],
        cov_type: Literal["nonrobust", "HC0", "HC1", "HC2", "HC3"] = "nonrobust",
        use_t: bool | None = None,
    ) -> None:
        self.model = model
        self.params = params
        self.normalized_cov_params = normalized_cov_params
        self.k_constant = model.k_constant
        self._wexog_singular_values = model.wexog_singular_values
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self.cov_type = cov_type
        self.cov_params_default: F64[k_exog, k_exog] | None = None
        if cov_type == "nonrobust":
            self.use_t = True if use_t is None else use_t
        else:
            self.use_t = False if use_t is None else use_t
            self.cov_params_default = {
                "HC0": lambda: self.cov_HC0,
                "HC1": lambda: self.cov_HC1,
                "HC2": lambda: self.cov_HC2,
                "HC3": lambda: self.cov_HC3,
            }[cov_type]()

    # -- residuals and sums of squares --------------------------------------

    @property
    def nobs(self) -> float:
        return float(self.model.wexog.shape[0])

    # The property above hides the size TypeVar ``nobs`` from bare ``F64[nobs]``
    # in the rest of the class body; the quoted spec still names the TypeVar.
    @property
    def fittedvalues(self) -> F64["nobs"]:
        return self.model.predict(self.params, self.model.exog)

    @property
    def wresid(self) -> F64[n_w]:
        return self.model.wendog - np.dot(self.model.wexog, self.params)

    @property
    def resid(self) -> F64["nobs"]:
        return self.model.endog - self.model.predict(self.params, self.model.exog)

    @property
    def scale(self) -> float:
        wresid = self.wresid
        return float(np.dot(wresid, wresid) / self.df_resid)

    @property
    def ssr(self) -> float:
        wresid = self.wresid
        return float(np.dot(wresid, wresid))

    @property
    def centered_tss(self) -> float:
        # isinstance narrows ``self.model`` to ``WLS[Any, Any]`` / ``GLS[Any, Any]``
        # (mypy cannot map ``n_w`` onto the subclass), so the arrays are read
        # through the typed ``model`` and the narrowed name is only tested.
        model = self.model
        narrowed = self.model
        if isinstance(narrowed, WLS):
            weights: F64["nobs"] = narrowed.weights
            mean = np.average(model.endog, weights=weights)
            return float(np.sum(weights * (model.endog - mean) ** 2))
        elif isinstance(narrowed, GLS) and narrowed.sigma is not None:
            iota = np.ones_like(model.endog)
            w_iota = model.whiten(iota)
            mean = model.wendog.dot(w_iota) / w_iota.dot(w_iota)
            err = model.endog - mean
            w_err = model.whiten(err)
            return float(np.sum(w_err**2))
        else:
            centered_endog = model.wendog - model.wendog.mean()
            return float(np.dot(centered_endog, centered_endog))

    @property
    def uncentered_tss(self) -> float:
        wendog = self.model.wendog
        return float(np.dot(wendog, wendog))

    @property
    def ess(self) -> float:
        if self.k_constant:
            return self.centered_tss - self.ssr
        else:
            return self.uncentered_tss - self.ssr

    @property
    def rsquared(self) -> float:
        if self.k_constant:
            return 1 - self.ssr / self.centered_tss
        else:
            return 1 - self.ssr / self.uncentered_tss

    @property
    def rsquared_adj(self) -> float:
        return float(
            1 - (np.divide(self.nobs - self.k_constant, self.df_resid) * (1 - self.rsquared))
        )

    @property
    def mse_model(self) -> float:
        if np.all(self.df_model == 0.0):
            return np.nan
        return self.ess / self.df_model

    @property
    def mse_resid(self) -> float:
        if np.all(self.df_resid == 0.0):
            return np.nan
        return self.ssr / self.df_resid

    @property
    def mse_total(self) -> float:
        if np.all(self.df_resid + self.df_model == 0.0):
            return np.nan
        if self.k_constant:
            return self.centered_tss / (self.df_resid + self.df_model)
        else:
            return self.uncentered_tss / (self.df_resid + self.df_model)

    @property
    def fvalue(self) -> float:
        if self.cov_type != "nonrobust":
            k_params = self.normalized_cov_params.shape[0]
            mat = np.eye(k_params)
            const_idx = self.model.const_idx
            if self.model.k_constant == 1:
                if const_idx is None:
                    return np.nan
                idx = list(range(k_params))
                idx.pop(const_idx)
                restr = mat[idx]
                if restr.size == 0:
                    return np.nan
                return self.f_test(restr)
            return self.f_test(mat)
        else:
            return self.mse_model / self.mse_resid

    # -- likelihood and information criteria --------------------------------

    @property
    def llf(self) -> float:
        return self.model.loglike(self.params)

    @property
    def aic(self) -> float:
        k_params = self.df_model + self.k_constant
        return -2 * self.llf + 2 * k_params

    @property
    def bic(self) -> float:
        k_params = self.df_model + self.k_constant
        return float(-2 * self.llf + np.log(self.nobs) * k_params)

    @property
    def eigenvals(self) -> F64[int]:
        if self._wexog_singular_values is not None:
            eigvals = self._wexog_singular_values**2
        else:
            wx = self.model.wexog
            eigvals = np.linalg.eigvalsh(wx.T @ wx)
        return np.sort(eigvals)[::-1]

    @property
    def condition_number(self) -> float:
        eigvals = self.eigenvals
        if eigvals[-1] > 0:
            return float(np.sqrt(eigvals[0] / eigvals[-1]))
        else:
            return np.inf

    # -- heteroscedasticity robust covariances ------------------------------

    def _HCCM(self, scale: F64[n_w]) -> F64[k_exog, k_exog]:
        pinv_wexog = self.model.pinv_wexog
        assert pinv_wexog is not None
        return np.dot(pinv_wexog, scale[:, None] * pinv_wexog.T)

    def _abat_diagonal(self, a: F64[n, k_exog], b: F64[k_exog, k_exog]) -> F64[n]:
        # equivalent to np.diag(a @ b @ a.T)
        return np.einsum("ij,ik,kj->i", a, a, b)

    @property
    def cov_HC0(self) -> F64[k_exog, k_exog]:
        self.het_scale = self.wresid**2
        return self._HCCM(self.het_scale)

    @property
    def cov_HC1(self) -> F64[k_exog, k_exog]:
        self.het_scale = self.nobs / (self.df_resid) * (self.wresid**2)
        return self._HCCM(self.het_scale)

    @property
    def cov_HC2(self) -> F64[k_exog, k_exog]:
        wexog = self.model.wexog
        h = self._abat_diagonal(wexog, self.normalized_cov_params)
        self.het_scale = self.wresid**2 / (1 - h)
        return self._HCCM(self.het_scale)

    @property
    def cov_HC3(self) -> F64[k_exog, k_exog]:
        wexog = self.model.wexog
        h = self._abat_diagonal(wexog, self.normalized_cov_params)
        self.het_scale = (self.wresid / (1 - h)) ** 2
        return self._HCCM(self.het_scale)

    @property
    def HC0_se(self) -> F64[k_exog]:
        return np.sqrt(np.diag(self.cov_HC0))

    @property
    def HC1_se(self) -> F64[k_exog]:
        return np.sqrt(np.diag(self.cov_HC1))

    @property
    def HC2_se(self) -> F64[k_exog]:
        return np.sqrt(np.diag(self.cov_HC2))

    @property
    def HC3_se(self) -> F64[k_exog]:
        return np.sqrt(np.diag(self.cov_HC3))

    @property
    def resid_pearson(self) -> F64[n_w]:
        eps = np.finfo(self.wresid.dtype).eps
        if np.sqrt(self.scale) < 10 * eps * self.model.endog.mean():
            warnings.warn("All residuals are 0, cannot compute normed residuals.",
                          RuntimeWarning, stacklevel=2)
            return self.wresid
        else:
            return self.wresid / np.sqrt(self.scale)

    # -- parameter covariance and linear restrictions -----------------------

    @overload
    def cov_params(
        self, r_matrix: None = None, column: None = None, scale: float | None = None,
        cov_p: F64[k_exog, k_exog] | None = None, other: None = None,
    ) -> F64[k_exog, k_exog]: ...
    @overload
    def cov_params(
        self, r_matrix: None = None, *, column: int, scale: float | None = None,
        cov_p: F64[k_exog, k_exog] | None = None, other: None = None,
    ) -> float: ...
    @overload
    def cov_params(
        self, r_matrix: None = None, *, column: Intp[c], scale: float | None = None,
        cov_p: F64[k_exog, k_exog] | None = None, other: None = None,
    ) -> F64[c, c]: ...
    @overload
    def cov_params(
        self, r_matrix: F64[q, k_exog], column: None = None, scale: float | None = None,
        cov_p: F64[k_exog, k_exog] | None = None, other: None = None,
    ) -> F64[q, q]: ...
    @overload
    def cov_params(
        self, r_matrix: F64[q, k_exog], column: None = None, scale: float | None = None,
        cov_p: F64[k_exog, k_exog] | None = None, *, other: F64[q2, k_exog],
    ) -> F64[q, q2]: ...
    def cov_params(
        self,
        r_matrix: F64[q, k_exog] | None = None,
        column: int | Intp[c] | None = None,
        scale: float | None = None,
        cov_p: F64[k_exog, k_exog] | None = None,
        other: F64[q2, k_exog] | None = None,
    ) -> F64[k_exog, k_exog] | float | F64[c, c] | F64[q, q] | F64[q, q2]:
        if column is not None and (r_matrix is not None or other is not None):
            raise ValueError("Column should be specified without other arguments.")
        if other is not None and r_matrix is None:
            raise ValueError("other can only be specified with r_matrix")
        if cov_p is None:
            if self.cov_params_default is not None:
                cov_p = self.cov_params_default
            else:
                if scale is None:
                    scale = self.scale
                cov_p = self.normalized_cov_params * scale
        if column is not None:
            if isinstance(column, int):
                return float(cov_p[column, column])
            else:
                return cov_p[column[:, None], column]
        elif r_matrix is not None:
            if other is None:
                return np.dot(r_matrix, np.dot(cov_p, np.transpose(r_matrix)))
            return np.dot(r_matrix, np.dot(cov_p, np.transpose(other)))
        else:
            return cov_p

    @property
    def bse(self) -> F64[k_exog]:
        return np.sqrt(np.diag(self.cov_params()))

    @property
    def tvalues(self) -> F64[k_exog]:
        return self.params / self.bse

    def t_test(
        self, r_matrix: F64[q, k_exog], q_matrix: F64[q] | None = None,
        cov_p: F64[k_exog, k_exog] | None = None,
    ) -> tuple[F64[q], F64[q] | F64[q, q], F64[q] | F64[q, q]]:
        """``(effect, sd, tvalue)`` of ``t_test``.

        With a single restriction ``sd`` and ``tvalue`` are ``(1, 1)``, as
        upstream: the ``num_ttests > 1`` branch takes the diagonal only then.
        """
        num_ttests = r_matrix.shape[0]
        num_params = r_matrix.shape[1]
        params = self.params.ravel(order="F")
        if num_params != params.shape[0]:
            raise ValueError("r_matrix and params are not aligned")
        if q_matrix is None:
            q_matrix = np.zeros(num_ttests)
        q_vec = np.asarray(q_matrix).squeeze()
        if q_vec.size > 1:
            if q_vec.shape[0] != num_ttests:
                raise ValueError(
                    "r_matrix and q_matrix must have the same number of rows"
                )
        _effect = np.dot(r_matrix, params)
        if num_ttests > 1:
            _sd = np.sqrt(np.diag(self.cov_params(r_matrix=r_matrix, cov_p=cov_p)))
            _t = (_effect - q_vec) * recipr(_sd)
            return _effect, _sd, _t
        else:
            _sd1 = np.sqrt(self.cov_params(r_matrix=r_matrix, cov_p=cov_p))
            _t1 = (_effect - q_vec) * recipr(_sd1)
            return _effect, _sd1, _t1

    def wald_test(
        self, r_matrix: F64[q, k_exog], q_matrix: F64[q] | None = None,
        cov_p: F64[k_exog, k_exog] | None = None,
        invcov: F64[q, q] | None = None, use_f: bool | None = None,
        df_constraints: float | None = None,
    ) -> tuple[float, float]:
        """``(statistic, df)``: ``(F, df_num)`` with ``use_f``, else ``(chi2, J)``."""
        if use_f is None:
            use_f = self.use_t
        params = self.params.ravel(order="F")
        cparams = np.dot(r_matrix, params)
        J = float(r_matrix.shape[0])
        if q_matrix is None:
            q_matrix = np.zeros(r_matrix.shape[0])
        q_col = q_matrix[:, None]
        if q_col.shape[0] != J:
            raise ValueError("r_matrix and q_matrix must have the same number of rows")
        Rbq = cparams - q_col
        if invcov is None:
            cov_r = self.cov_params(r_matrix=r_matrix, cov_p=cov_p)
            if np.isnan(cov_r).max():
                raise ValueError(
                    "r_matrix performs f_test for using dimensions that are "
                    "asymptotically non-normal"
                )
            invcov = np.linalg.pinv(cov_r)
            J_ = np.linalg.matrix_rank(cov_r)
            if J_ < J:
                warnings.warn(
                    "covariance of constraints does not have full "
                    f"rank. The number of constraints is {int(J):d}, but "
                    f"rank is {J_:d}",
                    ValueWarning,
                    stacklevel=2,
                )
                J = float(J_)
        if df_constraints is not None:
            J = df_constraints
        F = float(np.squeeze(np.dot(np.dot(Rbq.T, invcov), Rbq)))
        if use_f:
            return F / J, J
        return F, J

    def f_test(
        self, r_matrix: F64[q, k_exog], q_matrix: F64[q] | None = None,
        cov_p: F64[k_exog, k_exog] | None = None, invcov: F64[q, q] | None = None,
    ) -> float:
        """The F statistic of ``f_test``."""
        return self.wald_test(r_matrix, q_matrix, cov_p=cov_p, invcov=invcov,
                              use_f=True)[0]
