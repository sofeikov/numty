# MUTANT: _compute_eig: svd dispatch passes x.T
"""Principal Component Analysis from statsmodels, annotated for numty.

Extracted from statsmodels v0.15.0 (commit 278ff99):

- statsmodels/multivariate/pca.py
    - ``_norm`` (lines 20-34)
    - ``PCA.__init__`` (lines 212-329, without the pandas index/columns)
    - ``PCA._adjust_missing`` with ``keep_col`` / ``keep_row`` (lines 331-383)
    - ``PCA._compute_gls_weights`` (lines 385-410)
    - ``PCA._pca`` (lines 412-418)
    - ``PCA._prepare_data`` (lines 445-463)
    - ``PCA._compute_eig`` / ``_compute_using_svd`` / ``_compute_using_eig`` /
      ``_compute_using_nipals`` (lines 465-522)
    - ``PCA._fill_missing_em`` (lines 524-580)
    - ``PCA._compute_pca_from_eig`` (lines 582-619)
    - ``PCA._compute_rsquare_and_ic`` (lines 621-664)
    - ``PCA.project`` (lines 666-710, without the DataFrame wrapping)
    - ``pca`` (lines 843-932)

numpy only. Dropped: ``__repr__`` / ``__str__``, ``_to_pandas``, the plots,
and the ``statsmodels.tools.validation`` converters (``array_like`` and
friends): the inputs here are already a 2-d float array, bools, floats and
ints, and the checks that remain (``tol`` range, ``weights`` length, the
``method`` / ``missing`` options) are kept.

The numerical core of every method is a module-level function over named
sizes; the ``PCA`` class keeps upstream's state machine and calls them.

Sizes: ``nobs`` observations (rows), ``nvar`` variables (columns),
``n_comp`` components kept (a size the code chooses: ``min(nobs, nvar)`` by
default, fewer when asked or when eigenvalues are not positive), ``n_eig``
eigenvalues an eigen-routine returns before truncation (``nvar`` for
``eig``, ``min(nobs, nvar)`` for ``svd``, ``ncomp`` for ``nipals``) and
``n_vec`` the matching eigenvector columns (``nvar`` for ``svd`` with
``full_matrices=True``, otherwise ``n_eig``).
"""

from __future__ import annotations

import warnings
from typing import TypeVar, Unpack

import numpy as np
from typing_extensions import TypeVarTuple

from numty import F64, Bool, Intp

nobs = TypeVar("nobs", bound=int)
nvar = TypeVar("nvar", bound=int)
n_eig = TypeVar("n_eig", bound=int)
n_vec = TypeVar("n_vec", bound=int)
n_comp = TypeVar("n_comp", bound=int)
Ts = TypeVarTuple("Ts")


class ValueWarning(UserWarning):
    pass


class EstimationWarning(UserWarning):
    pass


def _norm(x: F64[Unpack[Ts]]) -> np.float64:
    """Euclidean (L2) norm of an array."""
    return np.sqrt(np.sum(x * x))


# ---------------------------------------------------------------------------
# Missing data (PCA._adjust_missing)
# ---------------------------------------------------------------------------


def _keep_col(x: F64[nobs, nvar]) -> tuple[F64["nobs n_keep"], Bool[nvar]]:
    index = np.logical_not(np.any(np.isnan(x), 0))
    return x[:, index], index


def _keep_row(x: F64[nobs, nvar]) -> tuple[F64["n_keep nvar"], Bool[nobs]]:
    index = np.logical_not(np.any(np.isnan(x), 1))
    return x[index, :], index


# ---------------------------------------------------------------------------
# Data preparation (PCA._prepare_data)
# ---------------------------------------------------------------------------


def _prepare_data(
    adj_data: F64[nobs, nvar], weights: F64[nvar], standardize: bool, demean: bool
) -> tuple[F64[nobs, nvar], F64[nvar], F64[nvar]]:
    """Standardize or demean data; returns ``(data, mu, sigma)``.

    Upstream's all-NaN branch (which returns a 1-d ``(nvar,)`` array of NaN
    instead of the 2-d data) is handled by the only caller that can reach
    it, ``_fill_missing_em``.
    """
    mu = np.nanmean(adj_data, axis=0)
    sigma = np.sqrt(np.nanmean((adj_data - mu) ** 2.0, axis=0))
    if standardize:
        data = (adj_data - mu) / sigma
    elif demean:
        data = adj_data - mu
    else:
        data = adj_data
    return data / np.sqrt(weights), mu, sigma


# ---------------------------------------------------------------------------
# Eigen-routines (PCA._compute_eig and friends)
# ---------------------------------------------------------------------------


def _compute_using_svd(
    x: F64[nobs, nvar], full_matrices: bool
) -> tuple[F64[n_eig], F64[nvar, n_vec]]:
    """SVD method to compute eigenvalues and eigenvecs."""
    u, s, v = np.linalg.svd(x, full_matrices=full_matrices)
    return s**2.0, v.T


def _compute_using_eig(x: F64[nobs, nvar]) -> tuple[F64[nvar], F64[nvar, nvar]]:
    """Eigenvalue decomposition of ``x'x``."""
    vals, vecs = np.linalg.eigh(x.T.dot(x))
    return vals, vecs


def _compute_using_nipals(
    x: F64[nobs, nvar], ncomp: int, tol: float, max_iter: int
) -> tuple[F64[n_comp], F64[nvar, n_comp]]:
    """NIPALS: the first ``ncomp`` eigenvalues and eigenvectors."""
    if ncomp > 1:
        x = x + 0.0  # Copy

    vals: F64[n_comp] = np.zeros(ncomp)
    vecs: F64[nvar, n_comp] = np.zeros((x.shape[1], ncomp))
    for i in range(ncomp):
        max_var_ind = np.argmax(x.var(0))
        factor: F64[nobs, 1] = x[:, [max_var_ind]]
        _iter = 0
        diff = 1.0
        while diff > tol and _iter < max_iter:
            vec = x.T.dot(factor) / (factor.T.dot(factor))
            vec = vec / np.sqrt(vec.T.dot(vec))
            factor_last = factor
            factor = x.dot(vec) / (vec.T.dot(vec))
            diff = float(_norm(factor - factor_last) / _norm(factor))
            _iter += 1
        vals[i] = (factor**2).sum()
        vecs[:, [i]] = vec
        if ncomp > 1:
            x -= factor.dot(vec.T)
    return vals, vecs


def _compute_eig(
    method: str,
    x: F64[nobs, nvar],
    ncomp: int,
    tol: float,
    max_iter: int,
    svd_full_matrices: bool,
) -> tuple[F64[int], F64[nvar, int]]:
    """Dispatch on ``method``; returns ``(eigenvals, eigenvecs)``.

    ``(nvar,)`` / ``(nvar, nvar)`` for ``eig``, ``(n_eig,)`` / ``(nvar, n_vec)``
    for ``svd``, ``(ncomp,)`` / ``(nvar, ncomp)`` for ``nipals``.
    """
    if method == "eig":
        return _compute_using_eig(x)
    elif method == "svd":
        return _compute_using_svd(x.T, svd_full_matrices)
    else:  # method == 'nipals'
        return _compute_using_nipals(x, ncomp, tol, max_iter)


# ---------------------------------------------------------------------------
# Factors, loadings, projection (PCA._compute_pca_from_eig, PCA.project)
# ---------------------------------------------------------------------------


def _compute_pca_from_eig(
    eigenvals: F64[n_eig],
    eigenvecs: F64[nvar, n_vec],
    transformed: F64[nobs, nvar],
    ncomp: int,
    normalize: bool,
) -> tuple[int, F64[n_comp], F64[nvar, n_comp], F64[nobs, n_comp], F64[n_comp, nvar]]:
    """Sort, truncate to ``ncomp`` and build the factors.

    Returns ``(ncomp, eigenvals, eigenvecs, factors, coeff)``; ``ncomp``
    shrinks when fewer eigenvalues are positive. ``loadings`` is
    ``eigenvecs``.
    """
    # Ensure sorted largest to smallest
    indices = np.argsort(eigenvals)
    indices = indices[::-1]
    vals = eigenvals[indices]
    vecs = eigenvecs[:, indices]
    if (vals <= 0).any():
        # Discard and warn
        num_good = vals.shape[0] - int((vals <= 0).sum())
        if num_good < ncomp:
            warnings.warn(
                f"Only {num_good:d} eigenvalues are positive.  "
                "This is the maximum number of components "
                "that can be extracted.",
                EstimationWarning,
                stacklevel=2,
            )

            ncomp = num_good
            vals[num_good:] = np.finfo(np.float64).tiny
    # Use ncomp for the remaining calculations
    vals_k = vals[:ncomp]
    vecs_k = vecs[:, :ncomp]
    # Select correct number of components to return
    factors = transformed.dot(vecs_k)
    coeff = vecs_k.T
    if normalize:
        coeff = (coeff.T * np.sqrt(vals_k)).T
        factors /= np.sqrt(vals_k)
    return ncomp, vals_k, vecs_k, factors, coeff


def _project(
    factors: F64[nobs, n_comp],
    coeff: F64[n_comp, nvar],
    weights: F64[nvar],
    mu: F64[nvar],
    sigma: F64[nvar],
    standardize: bool,
    demean: bool,
    ncomp: int | None = None,
    transform: bool = True,
    unweight: bool = True,
) -> F64[nobs, nvar]:
    """Project series onto the first ``ncomp`` factors (``PCA.project``)."""
    n_total = factors.shape[1]
    ncomp = n_total if ncomp is None else ncomp
    if ncomp > n_total:
        raise ValueError(
            "ncomp must be smaller than the number of components computed."
        )
    projection = factors[:, :ncomp].dot(coeff[:ncomp, :])
    if transform or unweight:
        projection *= np.sqrt(weights)
    if transform:
        # Remove the weights, which do not depend on transformation
        if standardize:
            projection *= sigma
        if standardize or demean:
            projection += mu
    return projection


# ---------------------------------------------------------------------------
# GLS weights, R-square and information criteria
# ---------------------------------------------------------------------------


def _compute_gls_weights(
    transformed: F64[nobs, nvar],
    factors: F64[nobs, n_comp],
    coeff: F64[n_comp, nvar],
    weights: F64[nvar],
    mu: F64[nvar],
    sigma: F64[nvar],
    standardize: bool,
    demean: bool,
) -> F64[nvar]:
    """GLS weights from the residual variance of each series."""
    projection = _project(
        factors, coeff, weights, mu, sigma, standardize, demean, transform=False
    )
    errors = transformed - projection
    if factors.shape[1] == transformed.shape[1]:
        raise ValueError(
            "gls can only be used when ncomp < nvar "
            "so that residuals have non-zero variance"
        )
    var = (errors**2.0).mean(0)
    new_weights = 1.0 / var
    new_weights = new_weights / np.sqrt((new_weights**2.0).mean())
    n_var = transformed.shape[1]
    eff_series_perc = (1.0 / sum((new_weights / new_weights.sum()) ** 2.0)) / n_var
    if eff_series_perc < 0.1:
        eff_series = int(np.round(eff_series_perc * n_var))

        warn = f"""\
Many series are being down weighted by GLS. Of the {n_var} series, the GLS
estimates are based on only {eff_series} (effective) series."""
        warnings.warn(warn, EstimationWarning, stacklevel=2)

    return new_weights


def _compute_rsquare_and_ic(
    transformed: F64[nobs, nvar],
    factors: F64[nobs, n_comp],
    coeff: F64[n_comp, nvar],
    weights: F64[nvar],
    mu: F64[nvar],
    sigma: F64[nvar],
    standardize: bool,
    demean: bool,
) -> tuple[F64["n_comp+1"], F64["n_ic 3"], F64["n_comp+1"], F64["n_comp+1 nvar"], F64[nvar]]:
    """Returns ``(rsquare, ic, ess, ess_indiv, tss_indiv)``."""
    ss_data = transformed * np.sqrt(weights)
    tss_indiv = np.sum(ss_data**2, 0)
    tss = np.sum(tss_indiv)
    ncomp = factors.shape[1]
    n_var = transformed.shape[1]
    ess = np.zeros(ncomp + 1)
    ess_indiv = np.zeros((ncomp + 1, n_var))
    for i in range(ncomp + 1):
        # Projection in the same space as transformed_data
        projection = _project(
            factors, coeff, weights, mu, sigma, standardize, demean,
            ncomp=i, transform=False, unweight=False,
        )
        indiv_rss = (projection**2).sum(axis=0)
        rss = indiv_rss.sum()
        ess[i] = tss - rss
        ess_indiv[i, :] = tss_indiv - indiv_rss
    rsquare = 1.0 - ess / tss
    # Information Criteria
    ess_ic: F64[int] = ess
    invalid = ess <= 0  # Prevent log issues of 0
    if invalid.any():
        last_obs = (np.where(invalid)[0]).min()
        ess_ic = ess[:last_obs]

    log_ess = np.log(ess_ic)
    r = np.arange(ess_ic.shape[0])

    n_obs = transformed.shape[0]
    sum_to_prod = (n_obs + n_var) / (n_obs * n_var)
    min_dim = min(n_obs, n_var)
    penalties = np.array(
        [
            sum_to_prod * np.log(1.0 / sum_to_prod),
            sum_to_prod * np.log(min_dim),
            np.log(min_dim) / min_dim,
        ]
    )
    penalty_col = penalties[:, None]
    ic = log_ess + r * penalty_col
    return rsquare, ic.T, ess, ess_indiv, tss_indiv


# ---------------------------------------------------------------------------
# EM fill of missing values (PCA._fill_missing_em)
# ---------------------------------------------------------------------------


def _fill_missing_em(
    data: F64[nobs, nvar],
    weights: F64[nvar],
    ncomp: int,
    standardize: bool,
    demean: bool,
    normalize: bool,
    method: str,
    tol: float,
    max_iter: int,
    tol_em: float,
    max_em_iter: int,
    svd_full_matrices: bool,
) -> tuple[F64[nobs, nvar], int]:
    """EM algorithm to fill missing values; returns ``(filled, ncomp)``.

    ``ncomp`` comes back smaller when an EM step finds fewer positive
    eigenvalues (upstream updates ``self._ncomp`` in place).
    """
    non_missing = np.logical_not(np.isnan(data))

    # If nothing missing, return without altering the data
    if np.all(non_missing):
        return data, ncomp

    # 1. Standardized data as needed (upstream's all-NaN branch of
    # _prepare_data gives a 1-d array that fails the check of step 2)
    if np.all(np.isnan(data)):
        raise ValueError(
            "Implementation requires that all columns and "
            "all rows have at least ncomp non-missing values"
        )
    prepared, mu, sigma = _prepare_data(data, weights, standardize, demean)

    # 2. Check for all nans
    col_non_missing = np.sum(non_missing, 1)
    row_non_missing = np.sum(non_missing, 0)
    if np.any(col_non_missing < ncomp) or np.any(row_non_missing < ncomp):
        raise ValueError(
            "Implementation requires that all columns and "
            "all rows have at least ncomp non-missing values"
        )
    # 3. Get mask
    mask = np.isnan(prepared)

    # 4. Compute mean
    mu_fill = np.nanmean(prepared, 0)

    # 5. Replace missing with mean
    projection = np.ones((prepared.shape[0], 1)) * mu_fill
    projection_masked = projection[mask]
    prepared[mask] = projection_masked

    # 6. Compute eigenvalues and fit
    diff = 1.0
    _iter = 0
    while diff > tol_em and _iter < max_em_iter:
        last_projection_masked = projection_masked
        vals, vecs = _compute_eig(
            method, prepared, ncomp, tol, max_iter, svd_full_matrices
        )
        ncomp, _, _, factors, coeff = _compute_pca_from_eig(
            vals, vecs, prepared, ncomp, normalize
        )
        projection = _project(
            factors, coeff, weights, mu, sigma, standardize, demean,
            transform=False, unweight=False,
        )
        projection_masked = projection[mask]
        prepared[mask] = projection_masked
        delta = last_projection_masked - projection_masked
        diff = float(_norm(delta) / _norm(projection_masked))
        _iter += 1
    # Must copy to avoid overwriting original data since replacing values
    filled = data + 0.0
    projection = _project(factors, coeff, weights, mu, sigma, standardize, demean)
    filled[mask] = projection[mask]

    return filled, ncomp


# ---------------------------------------------------------------------------
# The PCA class
# ---------------------------------------------------------------------------


class PCA:
    """Principal Component Analysis (numpy-only, see ``statsmodels.multivariate.pca.PCA``).

    Shapes after fitting (``nobs``, ``nvar`` after the missing-value
    adjustment): ``factors`` / ``scores`` ``(nobs, ncomp)``, ``loadings``
    ``(nvar, ncomp)``, ``coeff`` ``(ncomp, nvar)``, ``projection``
    ``(nobs, nvar)``, ``eigenvals`` ``(ncomp,)``, ``eigenvecs``
    ``(nvar, ncomp)``, ``rsquare`` ``(ncomp+1,)``, ``ic`` ``(<= ncomp+1, 3)``,
    ``weights`` ``(nvar,)``.
    """

    data: F64[int, int]
    weights: F64[int]
    rows: Intp[int]
    cols: Intp[int]
    transformed_data: F64[int, int]
    eigenvals: F64[int]
    eigenvecs: F64[int, int]
    factors: F64[int, int]
    scores: F64[int, int]
    loadings: F64[int, int]
    coeff: F64[int, int]
    projection: F64[int, int]
    rsquare: F64[int]
    ic: F64[int, 3]
    _adjusted_data: F64[int, int]
    _mu: F64[int]
    _sigma: F64[int]
    _ess: F64[int]
    _ess_indiv: F64[int, int]
    _tss_indiv: F64[int]

    def __init__(
        self,
        data: F64[int, int],
        ncomp: int | None = None,
        standardize: bool = True,
        demean: bool = True,
        normalize: bool = True,
        gls: bool = False,
        weights: F64[int] | None = None,
        method: str = "svd",
        missing: str | None = None,
        tol: float = 5e-8,
        max_iter: int = 1000,
        tol_em: float = 5e-8,
        max_em_iter: int = 100,
        svd_full_matrices: bool = False,
    ) -> None:
        if data.ndim != 2:
            raise ValueError("data is required to have ndim 2")
        self.data = data
        # Store inputs
        self._gls = gls
        self._normalize = normalize
        self._svd_full_matrices = svd_full_matrices
        self._tol = tol
        if not 0 < self._tol < 1:
            raise ValueError("tol must be strictly between 0 and 1")
        self._max_iter = max_iter
        self._max_em_iter = max_em_iter
        self._tol_em = tol_em

        # Prepare data
        self._standardize = standardize
        self._demean = demean

        self._nobs, self._nvar = self.data.shape
        if weights is None:
            w = np.ones(self._nvar)
        else:
            w = np.array(weights).flatten()
            if w.shape[0] != self._nvar:
                raise ValueError("weights should have nvar elements")
            w = w / np.sqrt((w**2.0).mean())
        self.weights = w

        # Check ncomp against maximum
        min_dim = min(self._nobs, self._nvar)
        self._ncomp = min_dim if ncomp is None else ncomp
        if self._ncomp > min_dim:
            warn = (
                "The requested number of components is more than can be "
                "computed from data. The maximum number of components is "
                "the minimum of the number of observations or variables"
            )
            warnings.warn(warn, ValueWarning, stacklevel=2)
            self._ncomp = min_dim

        if method not in ("eig", "svd", "nipals"):
            raise ValueError("method must be one of eig, svd, nipals")
        self._method = method

        self.rows = np.arange(self._nobs)
        self.cols = np.arange(self._nvar)
        # Handle missing
        self._missing = None if missing is None else missing.lower()
        self._adjusted_data = self.data
        self._adjust_missing()

        # Update size
        self._nobs, self._nvar = self._adjusted_data.shape
        if self._ncomp == np.min(self.data.shape):
            self._ncomp = int(np.min(self._adjusted_data.shape))
        elif self._ncomp > np.min(self._adjusted_data.shape):
            raise ValueError(
                "When adjusting for missing values, user "
                "provided ncomp must be no larger than the "
                "smallest dimension of the "
                "missing-value-adjusted data size."
            )

        # Prepare data
        self._prepare()
        # Perform the PCA
        self._pca()
        if gls:
            self._compute_gls_weights()
            self._prepare()
            self._pca()

        # Final calculations
        self._compute_rsquare_and_ic()

    def _adjust_missing(self) -> None:
        """Implements alternatives for handling missing values."""
        if self._missing == "drop-col":
            self._adjusted_data, index = _keep_col(self.data)
            self.cols = np.where(index)[0]
            self.weights = self.weights[index]
        elif self._missing == "drop-row":
            self._adjusted_data, index = _keep_row(self.data)
            self.rows = np.where(index)[0]
        elif self._missing == "drop-min":
            drop_col, drop_col_index = _keep_col(self.data)
            drop_col_size = drop_col.size

            drop_row, drop_row_index = _keep_row(self.data)
            drop_row_size = drop_row.size

            if drop_row_size > drop_col_size:
                self._adjusted_data = drop_row
                self.rows = np.where(drop_row_index)[0]
            else:
                self._adjusted_data = drop_col
                self.weights = self.weights[drop_col_index]
                self.cols = np.where(drop_col_index)[0]
        elif self._missing == "fill-em":
            self._adjusted_data, self._ncomp = _fill_missing_em(
                self.data,
                self.weights,
                self._ncomp,
                self._standardize,
                self._demean,
                self._normalize,
                self._method,
                self._tol,
                self._max_iter,
                self._tol_em,
                self._max_em_iter,
                self._svd_full_matrices,
            )
        elif self._missing is None:
            if not np.isfinite(self._adjusted_data).all():
                raise ValueError(
                    """\
data contains non-finite values (inf, NaN). You should drop these values or
use one of the methods for adjusting data for missing-values."""
                )
        else:
            raise ValueError("missing method is not known.")

        # Check adjusted data size
        if self._adjusted_data.size == 0:
            raise ValueError("Removal of missing values has eliminated all data.")

    def _prepare(self) -> None:
        self.transformed_data, self._mu, self._sigma = _prepare_data(
            self._adjusted_data, self.weights, self._standardize, self._demean
        )

    def _pca(self) -> None:
        """Main PCA routine."""
        vals, vecs = _compute_eig(
            self._method,
            self.transformed_data,
            self._ncomp,
            self._tol,
            self._max_iter,
            self._svd_full_matrices,
        )
        self._ncomp, self.eigenvals, self.eigenvecs, self.factors, self.coeff = (
            _compute_pca_from_eig(
                vals, vecs, self.transformed_data, self._ncomp, self._normalize
            )
        )
        self.scores = self.factors
        self.loadings = self.eigenvecs
        self.projection = self.project()

    def _compute_gls_weights(self) -> None:
        self.weights = _compute_gls_weights(
            self.transformed_data,
            self.factors,
            self.coeff,
            self.weights,
            self._mu,
            self._sigma,
            self._standardize,
            self._demean,
        )

    def _compute_rsquare_and_ic(self) -> None:
        self.rsquare, self.ic, self._ess, self._ess_indiv, self._tss_indiv = (
            _compute_rsquare_and_ic(
                self.transformed_data,
                self.factors,
                self.coeff,
                self.weights,
                self._mu,
                self._sigma,
                self._standardize,
                self._demean,
            )
        )
        self._tss = float(np.sum(self._tss_indiv))

    def project(
        self, ncomp: int | None = None, transform: bool = True, unweight: bool = True
    ) -> F64[int, int]:
        """The ``(nobs, nvar)`` projection of the data onto ``ncomp`` factors."""
        return _project(
            self.factors,
            self.coeff,
            self.weights,
            self._mu,
            self._sigma,
            self._standardize,
            self._demean,
            ncomp,
            transform,
            unweight,
        )


def pca(
    data: F64[int, int],
    ncomp: int | None = None,
    standardize: bool = True,
    demean: bool = True,
    normalize: bool = True,
    gls: bool = False,
    weights: F64[int] | None = None,
    method: str = "svd",
) -> tuple[
    F64[int, int], F64[int, int], F64[int, int], F64[int], F64[int, 3], F64[int], F64[int, int]
]:
    """Returns ``(factors, loadings, projection, rsquare, ic, eigenvals, eigenvecs)``."""
    pc = PCA(
        data,
        ncomp=ncomp,
        standardize=standardize,
        demean=demean,
        normalize=normalize,
        gls=gls,
        weights=weights,
        method=method,
    )

    return (
        pc.factors,
        pc.loadings,
        pc.projection,
        pc.rsquare,
        pc.ic,
        pc.eigenvals,
        pc.eigenvecs,
    )
