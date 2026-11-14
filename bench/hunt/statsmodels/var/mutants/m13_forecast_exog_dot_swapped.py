# MUTANT: forecast: np.dot(trend_coefs, exog) instead of np.dot(exog, trend_coefs)
"""VAR(p) estimation, MA representation, forecasts and ACF from statsmodels.

Extracted from statsmodels v0.15.0 (commit 278ff99), numpy only:

- statsmodels/tsa/vector_ar/var_model.py
    - ``ma_rep`` (50-88), ``is_stable`` (91-114), ``var_acf`` (117-155),
      ``_var_acf`` (158-186), ``forecast_cov``/``mse`` (189-223),
      ``forecast`` (226-283), ``_forecast_vars`` (286-308),
      ``forecast_interval`` (311-348), ``var_loglike`` (351-383),
      ``orth_ma_rep`` (434-456)
    - ``VAR.__init__`` / ``fit`` / ``_estimate_var`` (595-844, without
      ``ic=`` order selection, names and pandas bookkeeping)
    - ``VARProcess`` (903-1355: ``__init__``, ``is_stable``,
      ``intercept_longrun``, ``mean``, ``ma_rep``, ``orth_ma_rep``,
      ``long_run_effects``, ``_chol_sigma_u``, ``_char_mat``, ``acf``,
      ``acorr``, ``forecast``, ``mse``/``forecast_cov``, ``_forecast_vars``,
      ``forecast_interval``)
    - ``VARResults`` (1382-2560: ``__init__``, ``df_model``, ``df_resid``,
      ``fittedvalues``, ``resid``, ``sample_acov``, ``sample_acorr``,
      ``resid_acov``, ``resid_acorr``, ``resid_corr``, ``sigma_u_mle``,
      ``cov_params``, ``cov_ybar``, ``_zz``, ``_cov_alpha``, ``_cov_sigma``,
      ``llf``, ``stderr``, ``stderr_endog_lagged``, ``stderr_dt``,
      ``tvalues``, ``forecast_cov``, ``_omega_forc_cov``, ``_bmat_forc_cov``,
      ``detomega``, ``info_criteria``, ``roots``)
    - ``_compute_acov`` (2698-2710), ``_acovs_to_acorrs`` (2713-2715)
- statsmodels/tsa/vector_ar/util.py: ``get_var_endog`` (20-58),
  ``get_trendorder`` (61-98), ``comp_matrix`` (165-200),
  ``norm_signif_level`` (274-289), ``acf_to_acorr`` (292-309)
- statsmodels/tsa/tsatools.py: ``add_trend`` (42-171, ndarray path),
  ``vec`` (601-623), ``unvec`` (670-692), ``unvech`` (695-727),
  ``duplication_matrix`` (730-747)
- statsmodels/tools/linalg.py: ``logdet_symm`` (10-33)

numpy only: ``scipy.linalg.cho_factor`` in ``logdet_symm`` is
``np.linalg.cholesky`` (same diagonal), ``scipy.stats.norm.ppf`` is
``statistics.NormalDist().inv_cdf``. ``cache_readonly`` is
``functools.cached_property``.

Sizes: ``n_totobs`` rows of endog, ``neqs`` equations, ``k_ar`` lags,
``k_exog`` deterministic + user exog columns (``k_trend + k_exog_user``),
``n_steps`` forecast horizon. The regressor matrix ``z`` is
``(n_totobs - k_ar, neqs*k_ar + k_exog)`` and ``params`` is its column count
by ``neqs``; a product of two names has no spelling in an annotation, so that
column count is ``int``.
"""

from __future__ import annotations

import statistics
from functools import cached_property
from typing import Any, Generic, Literal, NamedTuple, TypeVar, overload

import numpy as np

from numty import F64, Integer

n_totobs = TypeVar("n_totobs", bound=int)
neqs = TypeVar("neqs", bound=int)
k_ar = TypeVar("k_ar", bound=int)
k_exog = TypeVar("k_exog", bound=int)
k_exog_user = TypeVar("k_exog_user", bound=int)
n_steps = TypeVar("n_steps", bound=int)
n_y = TypeVar("n_y", bound=int)
n_lags = TypeVar("n_lags", bound=int)
n = TypeVar("n", bound=int)
m = TypeVar("m", bound=int)


# ---------------------------------------------------------------------------
# tsatools / tools.linalg helpers


def vec(mat: F64[n, m]) -> F64[int]:
    """Stack the columns of ``mat`` (length ``n*m``: a product, no spelling)."""
    return mat.ravel("F")


def unvec(v: F64[n]) -> F64["k k"]:
    """Square matrix whose stacked columns are ``v`` (``k*k == n``)."""
    k = int(np.sqrt(len(v)))
    assert k * k == len(v)
    return v.reshape((k, k), order="F")


def unvech(v: F64[n]) -> F64["r r"]:
    """Symmetric matrix from its stacked lower triangle (``r*(r+1)/2 == n``)."""
    # quadratic formula, correct fp error
    rows = 0.5 * (-1 + np.sqrt(1 + 8 * len(v)))
    r = int(np.round(rows))

    result = np.zeros((r, r))
    result[np.triu_indices(r)] = v
    result = result + result.T

    # divide diagonal elements by 2
    result[np.diag_indices(r)] /= 2

    return result


def duplication_matrix(size: int) -> F64[int, int]:
    """D_n with vec(S) = D_n vech(S): ``(size*size, size*(size+1)//2)``."""
    tmp = np.eye(size * (size + 1) // 2)
    return np.array([unvech(x).ravel() for x in tmp]).T


def logdet_symm(mat: F64[n, n]) -> float:
    """log(det(mat)) of a positive definite matrix."""
    c = np.linalg.cholesky(mat)
    return float(2 * np.sum(np.log(c.diagonal())))


def norm_signif_level(alpha: float = 0.05) -> float:
    """Two-sided critical value of the standard normal."""
    return statistics.NormalDist().inv_cdf(1 - alpha / 2)


def get_trendorder(trend: str = "c") -> int:
    """Number of deterministic terms implied by ``trend``."""
    if trend == "c":
        trendorder = 1
    elif trend in ("n", "nc"):
        trendorder = 0
    elif trend == "ct":
        trendorder = 2
    elif trend == "ctt":
        trendorder = 3
    else:
        raise ValueError(f"Unkown trend: {trend}")
    return trendorder


def add_trend(
    x: F64[n, m], trend: str = "c", has_constant: str = "skip"
) -> F64[n, int]:
    """Prepend constant / trend columns to ``x`` (ndarray path, prepend=True)."""
    if trend not in ("n", "c", "t", "ct", "ctt"):
        raise ValueError(f"trend must be one of n, c, t, ct, ctt, got {trend}")
    if has_constant not in ("raise", "add", "skip"):
        raise ValueError(f"has_constant must be raise, add or skip, got {has_constant}")
    if trend == "n":
        return x.copy()
    elif trend == "c":
        trendorder = 0
    elif trend == "ct" or trend == "t":
        trendorder = 1
    else:  # ctt
        trendorder = 2

    nobs = len(x)
    trendarr = np.vander(np.arange(1, nobs + 1, dtype=np.float64), trendorder + 1)
    # put in order ctt
    trendarr = np.fliplr(trendarr)
    if trend == "t":
        trendarr = trendarr[:, 1:2]

    if "c" in trend:
        ptp0 = np.ptp(x, axis=0)
        col_is_const = ptp0 == 0
        nz_const = col_is_const & (x[0] != 0)
        if np.any(nz_const):
            if has_constant == "raise":
                columns = np.arange(x.shape[1])[nz_const]
                const_cols = ", ".join([str(c) for c in columns])
                raise ValueError(
                    "x contains one or more constant columns. Column(s) "
                    f"{const_cols} are constant. Adding a constant with "
                    f"trend='{trend}' is not allowed."
                )
            elif has_constant == "skip":
                trendarr = trendarr[:, 1:]

    # NUMTY: the rows of x (n) are lost against trendarr's int rows: (?U, ?U)
    return np.column_stack([trendarr, x])  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# vector_ar.util


def get_var_endog(
    y: F64[n_totobs, neqs], lags: k_ar, trend: str = "c", has_constant: str = "skip"
) -> F64["n_totobs-k_ar", int]:
    """Lagged regressor matrix Z, rows ``[trend, y_{t-1}, ..., y_{t-p}]``."""
    nobs = len(y)
    # Ravel C order, need to put in descending order
    Z = np.array([y[t - lags : t][::-1].ravel() for t in range(lags, nobs)])

    # Add constant, trend, etc.
    if trend != "n":
        Z = add_trend(Z, trend=trend, has_constant=has_constant)

    return Z


def comp_matrix(coefs: F64[k_ar, neqs, neqs]) -> F64["kp kp"]:
    """Companion matrix of the VAR(1) form, ``kp = neqs * k_ar``."""
    p, k1, k2 = coefs.shape
    if k1 != k2:
        raise ValueError("coefs must be 3-d with shape (p, k, k).")

    kp = k1 * p

    result = np.zeros((kp, kp))
    result[:k1] = np.concatenate(coefs, axis=1)

    # Set I_K matrices
    if p > 1:
        result[np.arange(k1, kp), np.arange(kp - k1)] = 1

    return result


def acf_to_acorr(acf: F64[n_lags, neqs, neqs]) -> F64[n_lags, neqs, neqs]:
    """Scale an autocovariance function into an autocorrelation function."""
    diag = np.diag(acf[0])
    # numpy broadcasting sufficient
    return acf / np.sqrt(np.outer(diag, diag))


# ---------------------------------------------------------------------------
# var_model: process routines


def ma_rep(coefs: F64[k_ar, neqs, neqs], maxn: int = 10) -> F64[int, neqs, neqs]:
    """MA(infinity) matrices Phi_0 .. Phi_maxn of a VAR(p)."""
    p, k, _ = coefs.shape
    phis = np.zeros((maxn + 1, k, k))
    phis[0] = np.eye(k)

    # recursively compute Phi matrices
    for i in range(1, maxn + 1):
        for j in range(1, i + 1):
            if j > p:
                break

            phis[i] += np.dot(phis[i - j], coefs[j - 1])

    return phis


def is_stable(coefs: F64[k_ar, neqs, neqs], verbose: bool = False) -> bool:
    """Every eigenvalue of the companion matrix inside the unit circle."""
    A_var1 = comp_matrix(coefs)
    eigs = np.linalg.eigvals(A_var1)

    if verbose:
        print("Eigenvalues of VAR(1) rep")
        for val in np.abs(eigs):
            print(val)

    return bool((np.abs(eigs) <= 1).all())


def var_acf(
    coefs: F64[k_ar, neqs, neqs], sig_u: F64[neqs, neqs], nlags: int | None = None
) -> F64[int, neqs, neqs]:
    """Autocovariances ACF_y(0..nlags) of a stable VAR(p)."""
    p, k, _ = coefs.shape
    if nlags is None:
        nlags = p

    # p x k x k, ACF for lags 0, ..., p-1
    result = np.zeros((nlags + 1, k, k))
    result[:p] = _var_acf(coefs, sig_u)

    # yule-walker equations
    for h in range(p, nlags + 1):
        # compute ACF for lag=h
        # G(h) = A_1 G(h-1) + ... + A_p G(h-p)

        for j in range(p):
            result[h] += np.dot(coefs[j], result[h - j - 1])

    return result


def _var_acf(coefs: F64[k_ar, neqs, neqs], sig_u: F64[neqs, neqs]) -> F64[k_ar, neqs, neqs]:
    """ACF_y(h) for h = 0, ..., p-1 from vec(ACF) = (I - A kron A)^-1 vec(Sigma_U)."""
    p, k, k2 = coefs.shape
    assert k == k2

    A = comp_matrix(coefs)
    # construct VAR(1) noise covariance
    SigU = np.zeros((k * p, k * p))
    SigU[:k, :k] = sig_u

    # vec(ACF) = (I_(kp)^2 - kron(A, A))^-1 vec(Sigma_U)
    vecACF = np.linalg.solve(np.eye((k * p) ** 2) - np.kron(A, A), vec(SigU))

    acf_full = unvec(vecACF)
    blocks = [acf_full[:k, k * i : k * (i + 1)] for i in range(p)]
    acf = np.array(blocks)

    # NUMTY: a comprehension over range(p) has p items; numty mints ?U
    return acf  # type: ignore[return-value]


def forecast_cov(
    ma_coefs: F64[int, neqs, neqs], sigma_u: F64[neqs, neqs], steps: n_steps
) -> F64[n_steps, neqs, neqs]:
    """MSE(h) = sum_{i<h} Phi_i Sigma_u Phi_i' for h = 1 .. steps."""
    k = len(sigma_u)
    forc_covs = np.zeros((steps, k, k))

    prior = np.zeros((k, k))
    for h in range(steps):
        # Sigma(h) = Sigma(h-1) + Phi Sig_u Phi'
        phi = ma_coefs[h]
        var = phi @ sigma_u @ phi.T
        forc_covs[h] = prior = prior + var

    return forc_covs


mse = forecast_cov


def forecast(
    y: F64[n_y, neqs],
    coefs: F64[k_ar, neqs, neqs],
    trend_coefs: F64[k_exog, neqs] | F64[neqs] | None,
    steps: n_steps,
    exog: F64[n_steps, k_exog] | float | None = None,
) -> F64[n_steps, neqs]:
    """Linear minimum-MSE forecast, Lutkepohl p. 37."""
    coefs = np.asarray(coefs)
    if coefs.ndim != 3:
        raise ValueError("coefs must be an array with 3 dimensions")
    p, k = coefs.shape[:2]
    if y.shape[0] < p:
        raise ValueError(
            f"y must by have at least order ({p}) observations. Got {y.shape[0]}."
        )
    # initial value
    forcs = np.zeros((steps, k))
    if exog is not None and trend_coefs is not None:
        forcs += np.dot(trend_coefs, exog)
    # to make existing code (with trend_coefs=intercept and without exog) work:
    elif exog is None and trend_coefs is not None:
        forcs += trend_coefs

    # make indices easier to think about
    for h in range(1, steps + 1):
        # y_t(h) = intercept + sum_1^p A_i y_t_(h-i)
        f = forcs[h - 1]
        for i in range(1, p + 1):
            # slightly hackish
            if h - i <= 0:
                # e.g., when h=1, h-1 = 0, which is y[-1]
                prior_y = y[h - i - 1]
            else:
                # e.g., when h=2, h-1=1, which is forcs[0]
                prior_y = forcs[h - i - 1]

            # i=1 is coefs[0]
            f = f + np.dot(coefs[i - 1], prior_y)

        forcs[h - 1] = f

    return forcs


def _forecast_vars(
    steps: n_steps, ma_coefs: F64[int, neqs, neqs], sig_u: F64[neqs, neqs]
) -> F64[n_steps, neqs]:
    """Diagonals of the forecast error covariances (used by VECMResults)."""
    covs = mse(ma_coefs, sig_u, steps)
    # Take diagonal for each cov
    k = len(sig_u)
    inds = np.arange(k)
    return covs[:, inds, inds]


class ForecastInterval(NamedTuple, Generic[n_steps, neqs]):
    point_forecast: F64[n_steps, neqs]
    forc_lower: F64[n_steps, neqs]
    forc_upper: F64[n_steps, neqs]


def forecast_interval(
    y: F64[n_y, neqs],
    coefs: F64[k_ar, neqs, neqs],
    trend_coefs: F64[k_exog, neqs] | F64[neqs] | None,
    sig_u: F64[neqs, neqs],
    steps: n_steps,
    alpha: float = 0.05,
    exog: F64[n_steps, k_exog] | float | None = 1,
) -> ForecastInterval[n_steps, neqs]:
    """Gaussian forecast intervals around ``forecast``."""
    assert 0 < alpha < 1
    q = norm_signif_level(alpha)

    point_forecast = forecast(y, coefs, trend_coefs, steps, exog)
    ma_coefs = ma_rep(coefs, steps)
    sigma = np.sqrt(_forecast_vars(steps, ma_coefs, sig_u))

    forc_lower = point_forecast - q * sigma
    forc_upper = point_forecast + q * sigma

    # NUMTY: np.sqrt(f(steps, ...)) loses the int-TypeVar size n_steps (Any)
    return ForecastInterval(point_forecast, forc_lower, forc_upper)  # type: ignore[arg-type]


def var_loglike(resid: F64[n, neqs], omega: F64[neqs, neqs], nobs: int) -> float:
    """Gaussian VAR(p) log-likelihood at the MLE covariance ``omega``."""
    logdet = logdet_symm(np.asarray(omega))
    k = len(omega)
    part1 = -(nobs * k / 2) * np.log(2 * np.pi)
    part2 = -(nobs / 2) * (logdet + k)
    return float(part1 + part2)


def _compute_acov(x: F64[n, neqs], nlags: int = 1) -> F64[int, neqs, neqs]:
    x = x - x.mean(0)

    result = []
    for lag in range(nlags + 1):
        if lag > 0:
            r = np.dot(x[lag:].T, x[:-lag])
        else:
            r = np.dot(x.T, x)

        result.append(r)

    return np.array(result) / len(x)


def _acovs_to_acorrs(acovs: F64[n_lags, neqs, neqs]) -> F64[n_lags, neqs, neqs]:
    sd = np.sqrt(np.diag(acovs[0]))
    return acovs / np.outer(sd, sd)


# ---------------------------------------------------------------------------
# VARProcess


class VARProcess(Generic[k_ar, neqs]):
    """A known VAR(p) process.

    ``coefs`` are A_1..A_p, ``coefs_exog`` the deterministic/exog coefficients,
    one row per equation (or the intercept alone as a vector).
    """

    # Set by VARResults only; a bare VARProcess has none of them (upstream too).
    exog: F64[int, int] | None
    trend: str
    n_totobs: int

    def __init__(
        self,
        coefs: F64[k_ar, neqs, neqs],
        coefs_exog: F64[neqs, int] | F64[neqs] | None,
        sigma_u: F64[neqs, neqs],
        _params_info: dict[str, int] | None = None,
    ) -> None:
        self.k_ar = len(coefs)
        self.neqs = coefs.shape[1]
        self.coefs = coefs
        self.coefs_exog = coefs_exog
        # Note reshaping 1-D coefs_exog to 2_D makes unit tests fail
        self.sigma_u = sigma_u

        if _params_info is None:
            _params_info = {}
        self.k_exog_user = _params_info.get("k_exog_user", 0)
        if self.coefs_exog is not None:
            k_ex = self.coefs_exog.shape[0] if self.coefs_exog.ndim != 1 else 1
            k_c = k_ex - self.k_exog_user
        else:
            k_c = 0
        self.k_trend = _params_info.get("k_trend", k_c)
        self.k_exog = self.k_trend + self.k_exog_user

        self.intercept: F64[neqs]
        if self.k_trend > 0:
            assert coefs_exog is not None
            if coefs_exog.ndim == 2:
                # NUMTY: no narrowing on .ndim: the 1-D member is indexed too
                self.intercept = coefs_exog[:, 0]  # type: ignore[index, numty-axis]
            else:
                # NUMTY: no narrowing on .ndim: the 2-D member is assigned too
                self.intercept = coefs_exog  # type: ignore[assignment]
        else:
            self.intercept = np.zeros(self.neqs)

    def is_stable(self, verbose: bool = False) -> bool:
        return is_stable(self.coefs, verbose=verbose)

    def intercept_longrun(self) -> F64[neqs]:
        r"""mu = (I - A_1 - ... - A_p)^{-1} alpha"""
        return np.linalg.solve(self._char_mat, self.intercept)

    def mean(self) -> F64[neqs]:
        return self.intercept_longrun()

    def ma_rep(self, maxn: int = 10) -> F64[int, neqs, neqs]:
        return ma_rep(self.coefs, maxn=maxn)

    def orth_ma_rep(self, maxn: int = 10, P: F64[neqs, neqs] | None = None) -> F64[int, neqs, neqs]:
        if P is None:
            P = self._chol_sigma_u

        ma_mats = self.ma_rep(maxn=maxn)
        return np.array([np.dot(coefs, P) for coefs in ma_mats])

    def long_run_effects(self) -> F64[neqs, neqs]:
        return np.linalg.inv(self._char_mat)

    @cached_property
    def _chol_sigma_u(self) -> F64[neqs, neqs]:
        return np.linalg.cholesky(self.sigma_u)

    @cached_property
    def _char_mat(self) -> F64[neqs, neqs]:
        """Characteristic matrix of the VAR"""
        return np.eye(self.neqs) - self.coefs.sum(0)

    def acf(self, nlags: int | None = None) -> F64[int, neqs, neqs]:
        return var_acf(self.coefs, self.sigma_u, nlags=nlags)

    def acorr(self, nlags: int | None = None) -> F64[int, neqs, neqs]:
        return acf_to_acorr(self.acf(nlags=nlags))

    def forecast(
        self,
        y: F64[n_y, neqs],
        steps: n_steps,
        exog_future: F64[n_steps, k_exog_user] | None = None,
    ) -> F64[n_steps, neqs]:
        """Forecasts ``steps`` ahead from the last ``k_ar`` rows of ``y``."""
        if self.exog is None and exog_future is not None:
            raise ValueError(
                "No exog in model, so no exog_future supported in forecast method."
            )
        if self.exog is not None and exog_future is None:
            raise ValueError(
                "Please provide an exog_future argument to the forecast method."
            )

        if exog_future is not None:
            exog_future = np.asarray(exog_future, dtype=float)
            if exog_future.ndim != 2:
                raise ValueError("exog_future is required to have ndim 2.")
            if exog_future.shape[0] != steps:
                err_msg = f"""\
exog_future only has {exog_future.shape[0]} observations. It must have \
steps ({steps}) observations.
"""
                raise ValueError(err_msg)
        coefs_exog = self.coefs_exog
        assert coefs_exog is not None
        trend_coefs = None if coefs_exog.size == 0 else coefs_exog.T

        exogs: list[F64[n_steps] | Integer[n_steps] | F64[n_steps, k_exog_user]] = []
        if self.trend.startswith("c"):  # constant term
            exogs.append(np.ones(steps))
        exog_lin_trend: Integer[n_steps] = np.arange(
            self.n_totobs + 1, self.n_totobs + 1 + steps
        )
        if "t" in self.trend:
            exogs.append(exog_lin_trend)
        if "tt" in self.trend:
            exogs.append(exog_lin_trend**2)
        if exog_future is not None:
            exogs.append(exog_future)

        if not exogs:
            exog_all = None
        else:
            # NUMTY: items of a union of 1-D and 2-D shapes bail with an error
            exog_all = np.column_stack(exogs)  # type: ignore[numty-bail]
        return forecast(y, self.coefs, trend_coefs, steps, exog_all)

    def mse(self, steps: n_steps) -> F64[n_steps, neqs, neqs]:
        """Forecast error covariances MSE(1) .. MSE(steps)."""
        ma_coefs = self.ma_rep(steps)

        k = len(self.sigma_u)
        forc_covs = np.zeros((steps, k, k))

        prior = np.zeros((k, k))
        for h in range(steps):
            # Sigma(h) = Sigma(h-1) + Phi Sig_u Phi'
            phi = ma_coefs[h]
            var = phi @ self.sigma_u @ phi.T
            forc_covs[h] = prior = prior + var

        return forc_covs

    forecast_cov = mse

    def _forecast_vars(self, steps: n_steps) -> F64[n_steps, neqs]:
        covs = self.forecast_cov(steps)

        # Take diagonal for each cov
        inds = np.arange(self.neqs)
        return covs[:, inds, inds]

    def forecast_interval(
        self,
        y: F64[n_y, neqs],
        steps: n_steps,
        alpha: float = 0.05,
        exog_future: F64[n_steps, k_exog_user] | None = None,
    ) -> ForecastInterval[n_steps, neqs]:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        q = norm_signif_level(alpha)

        point_forecast = self.forecast(y, steps, exog_future=exog_future)
        sigma = np.sqrt(self._forecast_vars(steps))

        forc_lower = point_forecast - q * sigma
        forc_upper = point_forecast + q * sigma

        # NUMTY: np.sqrt(f(steps)) loses the int-TypeVar size n_steps (Any)
        return ForecastInterval(point_forecast, forc_lower, forc_upper)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# VARResults


class VARResults(VARProcess[k_ar, neqs], Generic[n_totobs, neqs, k_ar]):
    """Estimated VAR(p).

    ``endog_lagged`` is ``z``, ``(n_totobs - k_ar, neqs*k_ar + k_exog)``;
    ``params`` stacks the deterministic rows over A_1' .. A_p'.
    """

    def __init__(
        self,
        endog: F64[n_totobs, neqs],
        endog_lagged: F64["n_totobs-k_ar", int],
        params: F64[int, neqs],
        sigma_u: F64[neqs, neqs],
        lag_order: k_ar,
        trend: str = "c",
        exog: F64[int, int] | None = None,
    ) -> None:
        self.endog = endog
        self.endog_lagged = endog_lagged

        self.n_totobs, n_eq = self.endog.shape
        self.nobs = self.n_totobs - lag_order
        self.trend = trend
        k_trend = get_trendorder(trend)
        self.params = params
        self.exog = exog

        # construct coefficient matrices; each matrix needs to be transposed
        endog_start = k_trend
        if exog is not None:
            k_exog_user = exog.shape[1]
            endog_start += k_exog_user
        else:
            k_exog_user = 0
        reshaped = self.params[endog_start:]
        lagged: F64[k_ar, neqs, neqs] = reshaped.reshape((lag_order, n_eq, n_eq))
        # Need to transpose each coefficient matrix
        coefs = lagged.swapaxes(1, 2).copy()

        coefs_exog = params[:endog_start].T
        self.k_exog_user = k_exog_user

        _params_info: dict[str, int] = {
            "k_trend": k_trend,
            "k_exog_user": k_exog_user,
            "k_ar": lag_order,
        }
        super().__init__(coefs, coefs_exog, sigma_u, _params_info=_params_info)
        self.coefs_exog = coefs_exog
        self.k_exog = coefs_exog.shape[1]

    @property
    def df_model(self) -> int:
        """Free parameters per equation: neqs * k_ar + k_exog."""
        return self.neqs * self.k_ar + self.k_exog

    @property
    def df_resid(self) -> int:
        return self.nobs - self.df_model

    @cached_property
    def fittedvalues(self) -> F64["n_totobs-k_ar", neqs]:
        return np.dot(self.endog_lagged, self.params)

    @cached_property
    def resid(self) -> F64["n_totobs-k_ar", neqs]:
        # NUMTY: endog[k_ar:] is ?U, not n_totobs-k_ar (slice start of a size type)
        return self.endog[self.k_ar :] - self.fittedvalues  # type: ignore[return-value]

    def sample_acov(self, nlags: int = 1) -> F64[int, neqs, neqs]:
        return _compute_acov(self.endog[self.k_ar :], nlags=nlags)

    def sample_acorr(self, nlags: int = 1) -> F64[int, neqs, neqs]:
        acovs = self.sample_acov(nlags=nlags)
        return _acovs_to_acorrs(acovs)

    def resid_acov(self, nlags: int = 1) -> F64[int, neqs, neqs]:
        return _compute_acov(self.resid, nlags=nlags)

    def resid_acorr(self, nlags: int = 1) -> F64[int, neqs, neqs]:
        acovs = self.resid_acov(nlags=nlags)
        return _acovs_to_acorrs(acovs)

    @cached_property
    def resid_corr(self) -> F64[neqs, neqs]:
        return self.resid_acorr(0)[0]

    @cached_property
    def sigma_u_mle(self) -> F64[neqs, neqs]:
        """(Biased) maximum likelihood estimate of noise process covariance"""
        if not self.df_resid:
            return np.zeros_like(self.sigma_u)
        return self.sigma_u * self.df_resid / self.nobs

    def cov_params(self) -> F64[int, int]:
        """Covariance of vec(B): kron((Z'Z)^-1, Sigma_u)."""
        z = self.endog_lagged
        return np.kron(np.linalg.inv(z.T @ z), self.sigma_u)

    def cov_ybar(self) -> F64[neqs, neqs]:
        Ainv = np.linalg.inv(np.eye(self.neqs) - self.coefs.sum(0))
        return Ainv @ self.sigma_u @ Ainv.T

    @cached_property
    def _zz(self) -> F64[int, int]:
        # Z'Z
        return np.dot(self.endog_lagged.T, self.endog_lagged)

    @property
    def _cov_alpha(self) -> F64[int, int]:
        """Covariance of the lag coefficients, exog rows dropped."""
        kn = self.k_exog * self.neqs
        return self.cov_params()[kn:, kn:]

    @cached_property
    def _cov_sigma(self) -> F64[int, int]:
        """Estimated covariance matrix of vech(sigma_u)"""
        D_K = duplication_matrix(self.neqs)
        D_Kinv = np.linalg.pinv(D_K)

        sigxsig = np.kron(self.sigma_u, self.sigma_u)
        return 2 * D_Kinv @ sigxsig @ D_Kinv.T

    @cached_property
    def llf(self) -> float:
        return var_loglike(self.resid, self.sigma_u_mle, self.nobs)

    @cached_property
    def stderr(self) -> F64[int, neqs]:
        """Standard errors of coefficients, reshaped to match in size"""
        stderr = np.sqrt(np.diag(self.cov_params()))
        return stderr.reshape((self.df_model, self.neqs), order="C")

    @cached_property
    def stderr_endog_lagged(self) -> F64[int, neqs]:
        start = self.k_exog
        return self.stderr[start:]

    @cached_property
    def stderr_dt(self) -> F64[int, neqs]:
        end = self.k_exog
        return self.stderr[:end]

    @cached_property
    def tvalues(self) -> F64[int, neqs]:
        return self.params / self.stderr

    @overload
    def forecast_cov(self) -> F64[Literal[1], neqs, neqs]: ...
    @overload
    def forecast_cov(self, steps: n_steps, method: str = "mse") -> F64[n_steps, neqs, neqs]: ...
    def forecast_cov(self, steps: int = 1, method: str = "mse") -> F64[int, neqs, neqs]:
        r"""Sigma_yhat(h) = Sigma_y(h) + Omega(h) / T (method="auto")."""
        fc_cov = self.mse(steps)
        if method == "mse":
            pass
        elif method == "auto":
            if self.k_exog == 1 and self.k_trend < 2:
                # currently only supported if no exog and trend in ['n', 'c']
                fc_cov += self._omega_forc_cov(steps) / self.nobs
        else:
            raise ValueError("method has to be either 'mse' or 'auto'")

        return fc_cov

    def _omega_forc_cov(self, steps: int) -> F64[int, neqs, neqs]:
        # Approximate MSE matrix \Omega(h) as defined in Lut p97
        G = self._zz
        Ginv = np.linalg.inv(G)

        B = self._bmat_forc_cov()
        _B: dict[int, F64[int, int]] = {}

        def bpow(i: int) -> F64[int, int]:
            if i not in _B:
                _B[i] = np.linalg.matrix_power(B, i)

            return _B[i]

        phis = self.ma_rep(steps)
        sig_u = self.sigma_u

        omegas = np.zeros((steps, self.neqs, self.neqs))
        for h in range(1, steps + 1):
            if h == 1:
                omegas[h - 1] = self.df_model * self.sigma_u
                continue

            om = omegas[h - 1]
            for i in range(h):
                for j in range(h):
                    Bi = bpow(h - 1 - i)
                    Bj = bpow(h - 1 - j)
                    mult = np.trace(Bi.T @ Ginv @ Bj @ G)
                    om += mult * phis[i] @ sig_u @ phis[j].T
            omegas[h - 1] = om

        return omegas

    def _bmat_forc_cov(self) -> F64[int, int]:
        # B as defined on p. 96 of Lut
        upper = np.zeros((self.k_exog, self.df_model))
        upper[:, : self.k_exog] = np.eye(self.k_exog)

        lower_dim = self.neqs * (self.k_ar - 1)
        eye = np.eye(lower_dim)
        lower = np.column_stack(
            (
                np.zeros((lower_dim, self.k_exog)),
                eye,
                np.zeros((lower_dim, self.neqs)),
            )
        )

        return np.vstack((upper, self.params.T, lower))

    @cached_property
    def detomega(self) -> float:
        return float(np.linalg.det(self.sigma_u))

    @cached_property
    def info_criteria(self) -> dict[str, float]:
        "information criteria for lagorder selection"
        nobs = self.nobs
        k = self.neqs
        lag_order = self.k_ar
        free_params = lag_order * k**2 + k * self.k_exog
        if self.df_resid:
            ld = logdet_symm(self.sigma_u_mle)
        else:
            ld = -np.inf

        aic = ld + (2.0 / nobs) * free_params
        bic = ld + (np.log(nobs) / nobs) * free_params
        hqic = ld + (2.0 * np.log(np.log(nobs)) / nobs) * free_params
        fpe: float
        if self.df_resid:
            fpe = ((nobs + self.df_model) / self.df_resid) ** k * np.exp(ld)
        else:
            fpe = np.inf

        return {"aic": aic, "bic": bic, "hqic": hqic, "fpe": float(fpe)}

    @property
    def aic(self) -> float:
        return self.info_criteria["aic"]

    @property
    def fpe(self) -> float:
        return self.info_criteria["fpe"]

    @property
    def hqic(self) -> float:
        return self.info_criteria["hqic"]

    @property
    def bic(self) -> float:
        return self.info_criteria["bic"]

    @cached_property
    def roots(self) -> Any:
        """Inverse roots of det(I - A_1 z - ... - A_p z^p), by decreasing modulus."""
        k = self.neqs
        p = k * self.k_ar
        arr = np.zeros((p, p))
        arr[:k, :] = np.column_stack(list(self.coefs))
        arr[k:, :-k] = np.eye(p - k)
        roots = np.linalg.eig(arr)[0] ** -1
        idx = np.argsort(np.abs(roots))[::-1]  # sort by reverse modulus
        return roots[idx]


# ---------------------------------------------------------------------------
# VAR model


class VAR(Generic[n_totobs, neqs]):
    """Fit a VAR(p) by OLS, equation by equation (Lutkepohl pp. 146-153)."""

    def __init__(
        self, endog: F64[n_totobs, neqs], exog: F64[n_totobs, int] | None = None
    ) -> None:
        self.endog = endog
        self.exog = exog
        if self.endog.ndim == 1:
            raise ValueError("Only gave one variable to VAR")
        self.neqs = self.endog.shape[1]
        self.n_totobs = len(endog)
        # Set by fit()/_estimate_var(), once the lag order is known
        self.nobs: int | None = None
        self.k_trend: int | None = None

    def fit(self, maxlags: int | None = None, trend: str = "c") -> VARResults[int, neqs, int]:
        lags = maxlags
        if trend not in ("c", "ct", "ctt", "n"):
            raise ValueError(f"trend must be one of c, ct, ctt, n, got {trend}")
        if lags is None:
            lags = 1

        self.nobs = self.n_totobs - lags
        return self._estimate_var(lags, trend=trend)

    def _estimate_var(
        self, lags: k_ar, offset: int = 0, trend: str = "c"
    ) -> VARResults[int, neqs, k_ar]:
        """OLS of endog[lags:] on the lagged regressors."""
        # have to do this again because select_order does not call fit
        self.k_trend = k_trend = get_trendorder(trend)

        if offset < 0:  # pragma: no cover
            raise ValueError("offset must be >= 0")

        nobs = self.n_totobs - lags - offset
        endog = self.endog[offset:]
        exog = None if self.exog is None else self.exog[offset:]
        z = get_var_endog(endog, lags, trend=trend, has_constant="raise")
        if exog is not None:
            # TODO: currently only deterministic terms supported (exoglags==0)
            # and since exoglags==0, x will be an array of size 0.
            x = get_var_endog(exog[-nobs:], 0, trend="n", has_constant="raise")
            x_inst = exog[-nobs:]
            x = np.column_stack((x, x_inst))
            temp_z = z
            # NUMTY: z's rows are one ?U, the new rows another: rebinding refused
            z = np.empty((x.shape[0], x.shape[1] + z.shape[1]))  # type: ignore[assignment]
            z[:, : self.k_trend] = temp_z[:, : self.k_trend]
            z[:, self.k_trend : self.k_trend + x.shape[1]] = x
            z[:, self.k_trend + x.shape[1] :] = temp_z[:, self.k_trend :]
            del temp_z, x  # free memory
        # the following modification of z is necessary to get the same results
        # as JMulTi for the constant-term-parameter...
        for i in range(self.k_trend):
            if (np.diff(z[:, i]) == 1).all():  # modify the trend-column
                z[:, i] += lags
            # make the same adjustment for the quadratic term
            if (np.diff(np.sqrt(z[:, i])) == 1).all():
                z[:, i] = (np.sqrt(z[:, i]) + lags) ** 2

        y_sample = endog[lags:]
        # Lutkepohl p75, about 5x faster than stated formula
        params = np.linalg.lstsq(z, y_sample, rcond=1e-15)[0]
        resid = y_sample - np.dot(z, params)

        avobs = len(y_sample)
        if exog is not None:
            k_trend += exog.shape[1]
        df_resid = avobs - (self.neqs * lags + k_trend)

        sse = np.dot(resid.T, resid)
        if df_resid:
            omega = sse / df_resid
        else:
            omega = np.full_like(sse, np.nan)

        # NUMTY: VARResults[-?offset+n_totobs, ...] does not fit VARResults[int, ...]
        return VARResults(endog, z, params, omega, lags, trend=trend, exog=self.exog)  # type: ignore[return-value]
