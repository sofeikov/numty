"""Runtime harness for the GLS / WLS / OLS / GLSAR extract.

Sizes never coincide: nobs=30, k_exog=4, q=2 restrictions (and a q=1 and a
q=3 case), m=3 columns for the 2-d whitening, AR order p=2, 7 new rows for
predict, 5 column indices for cov_params(column=...).
"""

from __future__ import annotations

import warnings

import numpy as np

NOBS, K, M, P, NNEW = 30, 4, 3, 2, 7


def _design(rng: np.random.Generator, nobs: int = NOBS, k: int = K) -> np.ndarray:
    return np.column_stack([np.ones(nobs), rng.normal(size=(nobs, k - 1))])


def _ar1_cov(nobs: int, rho: float) -> np.ndarray:
    i = np.arange(nobs)
    return rho ** np.abs(i[:, None] - i[None, :]) / (1 - rho**2)


def _results(res, R: np.ndarray, R1: np.ndarray, Xnew: np.ndarray, tag: str) -> dict:
    out: dict[str, object] = {}
    for name in [
        "params", "bse", "tvalues", "resid", "fittedvalues", "wresid", "ssr",
        "scale", "centered_tss", "uncentered_tss", "ess", "rsquared",
        "rsquared_adj", "mse_model", "mse_resid", "mse_total", "fvalue", "llf",
        "aic", "bic", "eigenvals", "condition_number", "cov_HC0", "cov_HC1",
        "cov_HC2", "cov_HC3", "HC0_se", "HC1_se", "HC2_se", "HC3_se",
        "resid_pearson", "nobs", "df_model", "df_resid",
    ]:
        out[f"{tag}.{name}"] = getattr(res, name)
    out[f"{tag}.normalized_cov_params"] = res.normalized_cov_params
    out[f"{tag}.cov_params"] = res.cov_params()
    out[f"{tag}.cov_params_col"] = res.cov_params(column=2)
    out[f"{tag}.cov_params_cols"] = res.cov_params(column=np.array([2, 0, 2, 1, 0]))
    out[f"{tag}.cov_params_r"] = res.cov_params(r_matrix=R)
    out[f"{tag}.cov_params_r_scale"] = res.cov_params(r_matrix=R, scale=2.5)
    out[f"{tag}.cov_params_r_other"] = res.cov_params(
        r_matrix=R, other=np.vstack([R, R1])
    )
    out[f"{tag}.t_test"] = list(res.t_test(R))
    out[f"{tag}.t_test_q"] = list(res.t_test(R, q_matrix=np.array([0.5, -1.0])))
    out[f"{tag}.t_test_1"] = list(res.t_test(R1))
    out[f"{tag}.f_test"] = res.f_test(R)
    out[f"{tag}.f_test_q"] = res.f_test(R, q_matrix=np.array([0.5, -1.0]))
    out[f"{tag}.wald_chi2"] = list(res.wald_test(R, use_f=False))
    out[f"{tag}.wald_df"] = list(res.wald_test(R, df_constraints=5.0))
    cov_r = res.cov_params(r_matrix=R)
    out[f"{tag}.wald_invcov"] = list(res.wald_test(R, invcov=np.linalg.inv(cov_r)))
    out[f"{tag}.wald_covp"] = list(
        res.wald_test(R, cov_p=res.normalized_cov_params * 3.0)
    )
    out[f"{tag}.predict_new"] = res.model.predict(res.params, Xnew)
    out[f"{tag}.predict"] = res.model.predict(res.params)
    return out


def run(mod) -> dict:
    rng = np.random.default_rng(20251113)
    X = _design(rng)
    beta = np.array([1.0, -2.0, 0.5, 3.0])
    y = X @ beta + rng.normal(size=NOBS)
    Xnew = _design(rng, NNEW)
    R = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])
    R1 = np.array([[0.0, 1.0, 1.0, 0.0]])
    R3 = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    Z = rng.normal(size=(NOBS, M))
    out: dict[str, object] = {}

    # ---- helpers --------------------------------------------------------
    out["toeplitz"] = mod.toeplitz(np.array([3.0, 1.0, 0.5, 0.25, 0.1]))
    pinv, sv = mod.pinv_extended(X)
    out["pinv_extended"] = [pinv, sv]
    out["pinv_extended_wide"] = list(mod.pinv_extended(X[:3].copy()))
    out["recipr"] = mod.recipr(np.array([2.0, 0.0, -1.0, np.nan, 0.5]))
    out["recipr_2d"] = mod.recipr(np.array([[2.0, 0.0, 4.0], [np.nan, -3.0, 0.25]]))
    out["const"] = [
        list(mod._handle_constant(X, None)),
        list(mod._handle_constant(X, False)),
        list(mod._handle_constant(X[:, 1:].copy(), None)),
        list(mod._handle_constant(X[:, 1:].copy(), True)),
        list(mod._handle_constant(np.column_stack([np.zeros(NOBS), X[:, 1:]]), None)),
        list(mod._handle_constant(np.column_stack([2 * np.ones(NOBS), X]), None)),
        list(mod._handle_constant(np.column_stack([2 * np.ones(NOBS), X[:, 1:]]), None)),
        list(mod._handle_constant(
            np.column_stack([np.zeros(NOBS), np.zeros(NOBS), X[:, 1:]]), None)),
    ]
    dummies = (rng.integers(0, 2, NOBS) == 1).astype(float)
    Ximpl = np.column_stack([dummies, 1 - dummies, rng.normal(size=(NOBS, 2))])
    out["const_implicit"] = list(mod._handle_constant(Ximpl, None))

    # ---- OLS ------------------------------------------------------------
    ols = mod.OLS(y, X)
    for method in ("pinv", "qr"):
        res = ols.fit(method=method)
        out.update(_results(res, R, R1, Xnew, f"ols_{method}"))
        out[f"ols_{method}.rank"] = ols.rank
    out["ols_qr.effects"] = ols.effects
    out["ols.t_test_3"] = list(ols.fit().t_test(R3))
    out["ols.loglike"] = ols.loglike(beta)
    out["ols.loglike_scale"] = ols.loglike(beta, scale=1.7)
    out["ols.score"] = ols.score(beta)
    out["ols.score_scale"] = ols.score(beta, scale=1.7)
    out["ols.hessian"] = ols.hessian(beta)
    out["ols.hessian_scale"] = ols.hessian(beta, scale=1.7)
    out["ols.hessian_factor"] = ols.hessian_factor(beta)
    out["ols.ridge_scalar"] = ols.fit_ridge(0.3)
    out["ols.ridge_vector"] = ols.fit_ridge(np.array([0.0, 0.1, 0.5, 1.0]))
    for cov_type in ("HC0", "HC1", "HC2", "HC3"):
        rres = ols.fit(cov_type=cov_type)
        out[f"ols_{cov_type}.bse"] = rres.bse
        out[f"ols_{cov_type}.fvalue"] = rres.fvalue
        out[f"ols_{cov_type}.t_test"] = list(rres.t_test(R))
        out[f"ols_{cov_type}.use_t"] = rres.use_t
    # no constant: uncentered sums of squares, robust F over all columns
    ols_nc = mod.OLS(y, X[:, 1:].copy())
    out.update(_results(ols_nc.fit(), R[:, 1:].copy(), R1[:, 1:].copy(),
                        Xnew[:, 1:].copy(), "ols_nc"))
    out["ols_nc.fvalue_HC1"] = ols_nc.fit(cov_type="HC1").fvalue
    out["ols_nc.k_constant"] = ols_nc.k_constant
    # implicit constant (two dummies): robust F is nan
    ols_impl = mod.OLS(y, Ximpl)
    out["ols_impl.fvalue_HC1"] = ols_impl.fit(cov_type="HC1").fvalue
    out["ols_impl.fvalue"] = ols_impl.fit().fvalue
    # only a constant: robust F has no restriction left
    out["ols_const.fvalue_HC0"] = mod.OLS(y, np.ones((NOBS, 1))).fit(cov_type="HC0").fvalue
    # rank deficient design: the pinv path warns
    Xdef = np.column_stack([X, X[:, 1] + X[:, 2]])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res_def = mod.OLS(y, Xdef).fit()
    out["ols_def"] = [res_def.params, res_def.bse, res_def.df_model, res_def.df_resid]
    out["ols_def.warned"] = len(w)
    # the df properties of an unfitted model
    fresh = mod.OLS(y, X)
    out["ols.df_unfitted"] = [fresh.df_model, fresh.df_resid]
    fresh.df_model = 2.0
    fresh.df_resid = 20.0
    out["ols.df_set"] = [fresh.df_model, fresh.df_resid, fresh.fit().scale]

    # ---- WLS ------------------------------------------------------------
    weights = rng.uniform(0.5, 2.0, size=NOBS)
    wls = mod.WLS(y, X, weights=weights)
    out.update(_results(wls.fit(), R, R1, Xnew, "wls"))
    out.update(_results(wls.fit(method="qr"), R, R1, Xnew, "wls_qr"))
    out["wls.whiten_1d"] = wls.whiten(y)
    out["wls.whiten_2d"] = wls.whiten(Z)
    out["wls.loglike"] = wls.loglike(beta)
    out["wls.hessian_factor"] = wls.hessian_factor(beta)
    out["wls.HC2_fvalue"] = wls.fit(cov_type="HC2").fvalue
    wls_s = mod.WLS(y, X, weights=2.0)
    out["wls_scalar.weights"] = wls_s.weights
    out["wls_scalar.params"] = wls_s.fit().params
    out["wls_scalar.loglike"] = wls_s.loglike(beta)
    try:
        wls.whiten(np.zeros((NOBS, 2, 2)))
        out["wls.whiten_3d"] = "no error"
    except ValueError as e:
        out["wls.whiten_3d"] = str(e)
    try:
        mod.WLS(y, X, weights=np.ones(NOBS - 1))
        out["wls.bad_weights"] = "no error"
    except ValueError as e:
        out["wls.bad_weights"] = str(e)

    # ---- GLS ------------------------------------------------------------
    sigma_full = _ar1_cov(NOBS, 0.6)
    sigma_diag = rng.uniform(0.5, 2.0, size=NOBS)
    for tag, sig in [
        ("gls_none", None),
        ("gls_scalar", 2.0),
        ("gls_diag", sigma_diag),
        ("gls_col", sigma_diag[:, None].copy()),
        ("gls_full", sigma_full),
    ]:
        gls = mod.GLS(y, X, sigma=sig)
        out.update(_results(gls.fit(), R, R1, Xnew, tag))
        out[f"{tag}.params_qr"] = gls.fit(method="qr").params
        out[f"{tag}.sigma"] = gls.sigma if gls.sigma is not None else "none"
        out[f"{tag}.cholsigmainv"] = (
            gls.cholsigmainv if gls.cholsigmainv is not None else "none"
        )
        out[f"{tag}.whiten_1d"] = gls.whiten(y)
        out[f"{tag}.whiten_2d"] = gls.whiten(Z)
        out[f"{tag}.loglike"] = gls.loglike(beta)
        try:
            out[f"{tag}.hessian_factor"] = gls.hessian_factor(beta)
        except NotImplementedError:
            out[f"{tag}.hessian_factor"] = "not implemented"
        out[f"{tag}.HC3_fvalue"] = gls.fit(cov_type="HC3").fvalue
    for bad in (np.ones(NOBS - 1), np.eye(NOBS - 1)):
        try:
            mod.GLS(y, X, sigma=bad)
            out[f"gls.bad_sigma_{bad.ndim}"] = "no error"
        except ValueError as e:
            out[f"gls.bad_sigma_{bad.ndim}"] = str(e)
    out["get_sigma_none"] = list(mod._get_sigma(None, NOBS))

    # ---- yule_walker and GLSAR -----------------------------------------
    e = np.zeros(NOBS + 40)
    shocks = rng.normal(size=NOBS + 40)
    for t in range(2, NOBS + 40):
        e[t] = 0.5 * e[t - 1] - 0.3 * e[t - 2] + shocks[t]
    out["yw_adjusted"] = list(mod.yule_walker(e, order=3))
    out["yw_mle"] = list(mod.yule_walker(e, order=P, method="mle"))
    out["yw_df"] = list(mod.yule_walker(e, order=P, df=50))
    out["yw_nodemean"] = list(mod.yule_walker(e + 1.0, order=P, demean=False))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out["yw_singular"] = list(mod.yule_walker(np.zeros(12), order=P, demean=False))
    out["yw_singular.warned"] = len(w)
    ye = X @ beta + e[40:]
    glsar = mod.GLSAR(ye, X, rho=np.array([0.4, -0.2]))
    out["glsar.whiten_1d"] = glsar.whiten(ye)
    out["glsar.whiten_2d"] = glsar.whiten(Z)
    out.update(_results(glsar.fit(), R, R1, Xnew, "glsar"))
    res_it, rhos, params_hist, n_iter, conv = glsar.iterative_fit(maxiter=6)
    out["glsar.iter"] = [res_it.params, res_it.bse, rhos, params_hist, n_iter, conv]
    glsar2 = mod.GLSAR(ye, X, rho=np.array([0.4, -0.2]))
    res2 = glsar2.iterative_fit(maxiter=50, rtol=1e-3)
    out["glsar.iter_conv"] = [res2[0].params, res2[1], res2[3], res2[4]]
    glsar3 = mod.GLSAR(ye, X, rho=np.array([0.1, 0.0, 0.05]))
    res3 = glsar3.iterative_fit(maxiter=1)
    out["glsar.iter_one"] = [res3[0].params, res3[1], res3[3], res3[4]]
    return out
