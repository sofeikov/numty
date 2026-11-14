"""Runtime harness for the statsmodels PCA extract.

Sizes never coincide: the main data set is nobs=25 observations of nvar=6
variables with ncomp=3 components. A wide set (nobs=4 < nvar=6) exercises
``svd_full_matrices=True`` (4 eigenvalues, 6x6 eigenvectors) and ``eig`` with
more eigenvalues than components; a 50x30 set drives the GLS warning.
"""

from __future__ import annotations

import warnings

import numpy as np

NOBS, NVAR, NCOMP = 25, 6, 3


def _factor_data(rng: np.random.Generator, nobs: int, nvar: int, k: int) -> np.ndarray:
    f = rng.normal(size=(nobs, k))
    lam = rng.normal(size=(k, nvar))
    return f @ lam + 0.5 * rng.normal(size=(nobs, nvar)) + rng.normal(size=nvar)


def _summ(pc, tag: str) -> dict:
    return {
        f"{tag}.factors": pc.factors,
        f"{tag}.loadings": pc.loadings,
        f"{tag}.coeff": pc.coeff,
        f"{tag}.projection": pc.projection,
        f"{tag}.eigenvals": pc.eigenvals,
        f"{tag}.eigenvecs": pc.eigenvecs,
        f"{tag}.rsquare": pc.rsquare,
        f"{tag}.ic": pc.ic,
        f"{tag}.weights": pc.weights,
        f"{tag}.transformed": pc.transformed_data,
        f"{tag}.rows": pc.rows,
        f"{tag}.cols": pc.cols,
        f"{tag}.ncomp": pc._ncomp,
        f"{tag}.ess_indiv": pc._ess_indiv,
        f"{tag}.tss_indiv": pc._tss_indiv,
    }


def _err(fn) -> str:
    try:
        fn()
    except ValueError as e:
        return "ValueError: " + str(e)
    return "no error"


def run(mod) -> dict:
    rng = np.random.default_rng(20250340)
    out: dict = {}
    x = _factor_data(rng, NOBS, NVAR, NCOMP)
    w = rng.uniform(0.5, 2.0, size=NVAR)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        # Methods and transformations
        for method in ("svd", "eig", "nipals"):
            out.update(_summ(mod.PCA(x, ncomp=NCOMP, method=method), f"{method}"))
        out.update(_summ(mod.PCA(x, ncomp=NCOMP, standardize=False), "demean"))
        out.update(
            _summ(mod.PCA(x, ncomp=NCOMP, standardize=False, demean=False), "raw")
        )
        out.update(_summ(mod.PCA(x, ncomp=NCOMP, normalize=False), "nonorm"))
        out.update(
            _summ(
                mod.PCA(x, ncomp=NCOMP, method="nipals", normalize=False, weights=w),
                "nipals_w",
            )
        )
        out.update(_summ(mod.PCA(x, ncomp=1, method="nipals"), "nipals1"))
        out.update(_summ(mod.PCA(x, weights=w), "full_w"))
        out.update(_summ(mod.PCA(x, ncomp=NCOMP, gls=True), "gls"))
        out.update(_summ(mod.PCA(x, ncomp=NCOMP, gls=True, method="eig"), "gls_eig"))

        # project() variants
        pc = mod.PCA(x, ncomp=NCOMP, weights=w)
        out["project.2"] = pc.project(ncomp=2)
        out["project.notrans"] = pc.project(transform=False)
        out["project.raw"] = pc.project(ncomp=1, transform=False, unweight=False)
        out["project.too_many"] = _err(lambda: pc.project(ncomp=NCOMP + 1))

        # Wide data: nobs < nvar
        xw = _factor_data(rng, 4, NVAR, 2)
        out.update(_summ(mod.PCA(xw, svd_full_matrices=True), "wide_fm"))
        out.update(_summ(mod.PCA(xw, ncomp=3, method="eig"), "wide_eig"))
        out.update(_summ(mod.PCA(xw, ncomp=2, method="nipals"), "wide_nipals"))

        # Non-positive eigenvalues: a zero column, no demeaning -> ncomp shrinks
        xz = x.copy()
        xz[:, 2] = 0.0
        out.update(
            _summ(mod.PCA(xz, method="eig", standardize=False, demean=False), "zero_col")
        )

        # Missing data
        xm = x.copy()
        xm[3, 1] = np.nan
        xm[7, 4] = np.nan
        xm[11, 1] = np.nan
        out.update(_summ(mod.PCA(xm, ncomp=NCOMP, missing="drop-row"), "drop_row"))
        out.update(
            _summ(mod.PCA(xm, ncomp=NCOMP, missing="drop-col", weights=w), "drop_col")
        )
        out.update(_summ(mod.PCA(xm, ncomp=NCOMP, missing="drop-min"), "drop_min_row"))
        xc = x.copy()
        xc[::2, 5] = np.nan
        out.update(
            _summ(mod.PCA(xc, ncomp=NCOMP, missing="drop-min", weights=w), "drop_min_col")
        )
        out.update(_summ(mod.PCA(xc, missing="drop-col"), "drop_col_default_ncomp"))
        out.update(_summ(mod.PCA(xm, ncomp=NCOMP, missing="fill-em"), "fill_em"))
        out.update(
            _summ(
                mod.PCA(xm, ncomp=2, missing="fill-em", method="nipals", weights=w),
                "fill_em_nipals",
            )
        )
        out.update(
            _summ(
                mod.PCA(xm, ncomp=NCOMP, missing="fill-em", method="eig", demean=False,
                        standardize=False),
                "fill_em_eig",
            )
        )
        out.update(_summ(mod.PCA(x, ncomp=NCOMP, missing="fill-em"), "fill_em_none"))

        # GLS that down-weights most series: two near-copies of one factor
        # among 28 low-variance noise series (no standardization)
        rng2 = np.random.default_rng(7)
        f = rng2.normal(size=(50, 1))
        xg = 0.1 * rng2.normal(size=(50, 30))
        xg[:, 0] = f[:, 0] + 1e-3 * rng2.normal(size=50)
        xg[:, 1] = f[:, 0] + 1e-3 * rng2.normal(size=50)
        out.update(_summ(mod.PCA(xg, ncomp=1, gls=True, standardize=False), "gls_warn"))

        # More components than the data allow
        out.update(_summ(mod.PCA(x, ncomp=NVAR + 2), "ncomp_too_big"))

        out["warnings"] = sorted(
            f"{type(c.message).__name__}: {c.message}" for c in caught
        )

    # Errors
    xm_bad = x.copy()
    xm_bad[0, 0] = np.nan
    out["err.nan"] = _err(lambda: mod.PCA(xm_bad))
    out["err.missing"] = _err(lambda: mod.PCA(x, missing="bogus"))
    out["err.weights"] = _err(lambda: mod.PCA(x, weights=np.ones(NVAR + 1)))
    out["err.tol"] = _err(lambda: mod.PCA(x, tol=2.0))
    out["err.gls_full"] = _err(lambda: mod.PCA(x, gls=True))
    x10 = _factor_data(np.random.default_rng(3), 10, 8, 2)
    x10[1, [0, 3, 6]] = np.nan
    out["err.adjusted_ncomp"] = _err(lambda: mod.PCA(x10, ncomp=7, missing="drop-col"))
    out["err.all_nan_em"] = _err(
        lambda: mod.PCA(np.full((5, 3), np.nan), missing="fill-em")
    )
    out["err.all_dropped"] = _err(
        lambda: mod.PCA(np.full((5, 3), np.nan), missing="drop-row")
    )
    xsparse = x.copy()
    xsparse[:, 0] = np.nan
    xsparse[0, 0] = 1.0
    out["err.em_sparse"] = _err(lambda: mod.PCA(xsparse, ncomp=NCOMP, missing="fill-em"))

    # Functional interface
    res = mod.pca(x, ncomp=NCOMP, method="eig", weights=w)
    for name, val in zip(
        ("factors", "loadings", "projection", "rsquare", "ic", "eigenvals", "eigenvecs"),
        res,
    ):
        out[f"pca.{name}"] = val
    return out
