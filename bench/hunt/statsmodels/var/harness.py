"""Runtime harness for the VAR extract.

Sizes never coincide: neqs=3 equations, k_ar=2 lags, n_totobs=60 rows,
steps=5, 2 user exog columns; a second fit uses k_ar=3 and a VAR(1) process
covers the ``p == 1`` branch of ``comp_matrix``. A tiny sample (9 rows) hits
``df_resid == 0``.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

NEQS, K_AR, NOBS, STEPS = 3, 2, 60, 5


def _simulate(rng: np.random.Generator, n: int, neqs: int) -> tuple[np.ndarray, np.ndarray]:
    A1 = np.array([[0.5, 0.1, 0.0], [0.05, 0.3, 0.1], [0.0, 0.2, 0.4]])[:neqs, :neqs]
    A2 = np.array([[-0.2, 0.0, 0.05], [0.1, -0.1, 0.0], [0.0, 0.05, -0.15]])[:neqs, :neqs]
    y = np.zeros((n, neqs))
    y[:2] = rng.normal(size=(2, neqs))
    for t in range(2, n):
        y[t] = 0.3 + A1 @ y[t - 1] + A2 @ y[t - 2] + rng.normal(scale=0.5, size=neqs)
    return y, np.stack([A1, A2])


def _err(fn) -> str:
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return type(e).__name__
    return "no error"


def _results(res, tag: str, y_last) -> dict:
    out = {
        f"{tag}.params": res.params,
        f"{tag}.sigma_u": res.sigma_u,
        f"{tag}.coefs": res.coefs,
        f"{tag}.coefs_exog": res.coefs_exog,
        f"{tag}.intercept": res.intercept,
        f"{tag}.endog_lagged": res.endog_lagged,
        f"{tag}.resid": res.resid,
        f"{tag}.fittedvalues": res.fittedvalues,
        f"{tag}.df": [res.df_model, res.df_resid, res.k_exog, res.k_trend, res.nobs],
        f"{tag}.llf": res.llf,
        f"{tag}.cov_params": res.cov_params(),
        f"{tag}.stderr": res.stderr,
        f"{tag}.stderr_endog_lagged": res.stderr_endog_lagged,
        f"{tag}.stderr_dt": res.stderr_dt,
        f"{tag}.tvalues": res.tvalues,
        f"{tag}.cov_ybar": res.cov_ybar(),
        f"{tag}.cov_alpha": res._cov_alpha,
        f"{tag}.cov_sigma": res._cov_sigma,
        f"{tag}.sigma_u_mle": res.sigma_u_mle,
        f"{tag}.ic": [res.aic, res.bic, res.hqic, res.fpe],
        f"{tag}.roots": res.roots,
        f"{tag}.detomega": res.detomega,
        f"{tag}.sample_acov": res.sample_acov(3),
        f"{tag}.sample_acorr": res.sample_acorr(3),
        f"{tag}.resid_acov": res.resid_acov(2),
        f"{tag}.resid_acorr": res.resid_acorr(2),
        f"{tag}.resid_corr": res.resid_corr,
        f"{tag}.is_stable": res.is_stable(),
        f"{tag}.mean": res.mean(),
        f"{tag}.long_run_effects": res.long_run_effects(),
        f"{tag}.ma_rep": res.ma_rep(STEPS),
        f"{tag}.orth_ma_rep": res.orth_ma_rep(STEPS),
        f"{tag}.acf": res.acf(),
        f"{tag}.acf6": res.acf(6),
        f"{tag}.acorr": res.acorr(4),
        f"{tag}.mse": res.mse(STEPS),
        f"{tag}.forecast_cov": res.forecast_cov(STEPS),
        f"{tag}.forecast_cov1": res.forecast_cov(),
        f"{tag}.forecast_vars": res._forecast_vars(STEPS),
        f"{tag}.bmat": res._bmat_forc_cov(),
    }
    if res.exog is None:
        out[f"{tag}.forecast"] = res.forecast(y_last, STEPS)
        fi = res.forecast_interval(y_last, STEPS, alpha=0.1)
        out[f"{tag}.forecast_interval"] = list(fi)
    return out


def run(mod) -> dict:
    rng = np.random.default_rng(1234)
    y, true_coefs = _simulate(rng, NOBS, NEQS)
    sig = np.array([[1.0, 0.3, 0.1], [0.3, 0.8, 0.2], [0.1, 0.2, 0.6]])
    out: dict = {}

    # --- helpers
    M = rng.normal(size=(4, 3))
    out["vec"] = mod.vec(M)
    out["unvec"] = mod.unvec(mod.vec(np.arange(9.0).reshape(3, 3)))
    out["unvech"] = mod.unvech(np.arange(1.0, 7.0))
    out["duplication_matrix"] = mod.duplication_matrix(3)
    out["logdet_symm"] = mod.logdet_symm(sig)
    out["norm_signif_level"] = mod.norm_signif_level(0.1)
    out["trendorder"] = [mod.get_trendorder(t) for t in ("c", "n", "nc", "ct", "ctt")]
    out["trendorder_bad"] = _err(lambda: mod.get_trendorder("x"))
    X = rng.normal(size=(7, 2))
    out["add_trend_ctt"] = mod.add_trend(X, trend="ctt")
    out["add_trend_t"] = mod.add_trend(X, trend="t")
    out["add_trend_n"] = mod.add_trend(X, trend="n")
    Xc = np.column_stack([np.full(7, 2.0), X])
    out["add_trend_skip"] = mod.add_trend(Xc, trend="ct", has_constant="skip")
    out["add_trend_add"] = mod.add_trend(Xc, trend="c", has_constant="add")
    out["add_trend_raise"] = _err(lambda: mod.add_trend(Xc, trend="c", has_constant="raise"))
    out["get_var_endog_c"] = mod.get_var_endog(y[:9], K_AR)
    out["get_var_endog_n"] = mod.get_var_endog(y[:9], 3, trend="n")
    out["get_var_endog_ct"] = mod.get_var_endog(y[:9], K_AR, trend="ct")
    out["comp_matrix"] = mod.comp_matrix(true_coefs)
    out["comp_matrix_p1"] = mod.comp_matrix(true_coefs[:1])
    out["comp_matrix_bad"] = _err(lambda: mod.comp_matrix(np.zeros((2, 3, 4))))

    # --- module-level process routines
    out["ma_rep"] = mod.ma_rep(true_coefs, 6)
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        out["is_stable"] = mod.is_stable(true_coefs, verbose=True)
    out["is_stable_print_lines"] = len(buf.getvalue().splitlines())
    out["is_unstable"] = mod.is_stable(true_coefs * 4.0)
    out["var_acf"] = mod.var_acf(true_coefs, sig)
    out["var_acf7"] = mod.var_acf(true_coefs, sig, nlags=7)
    out["_var_acf"] = mod._var_acf(true_coefs, sig)
    ma = mod.ma_rep(true_coefs, STEPS)
    out["forecast_cov"] = mod.forecast_cov(ma, sig, STEPS)
    out["mse_alias"] = mod.mse(ma, sig, 3)
    out["_forecast_vars"] = mod._forecast_vars(STEPS, ma, sig)
    intercept = np.array([0.3, -0.1, 0.2])
    out["forecast_1d"] = mod.forecast(y[-4:], true_coefs, intercept, STEPS)
    out["forecast_none"] = mod.forecast(y[-4:], true_coefs, None, STEPS)
    tc = rng.normal(size=(2, NEQS))
    ex = rng.normal(size=(STEPS, 2))
    out["forecast_exog"] = mod.forecast(y[-4:], true_coefs, tc, STEPS, ex)
    out["forecast_list_coefs"] = mod.forecast(y[-4:], true_coefs.tolist(), None, STEPS)
    out["forecast_bad_ndim"] = _err(lambda: mod.forecast(y[-4:], true_coefs[0], None, STEPS))
    out["forecast_short_y"] = _err(lambda: mod.forecast(y[-1:], true_coefs, None, STEPS))
    fi = mod.forecast_interval(y[-4:], true_coefs, intercept[None, :], sig, steps=STEPS)
    out["forecast_interval"] = list(fi)
    fi2 = mod.forecast_interval(y[-4:], true_coefs, tc, sig, steps=STEPS, alpha=0.2, exog=ex)
    out["forecast_interval_exog"] = list(fi2)
    resid = rng.normal(size=(40, NEQS))
    out["var_loglike"] = mod.var_loglike(resid, resid.T @ resid / 40, 40)
    out["compute_acov"] = mod._compute_acov(resid, nlags=3)
    out["acovs_to_acorrs"] = mod._acovs_to_acorrs(mod._compute_acov(resid, nlags=2))
    out["acf_to_acorr"] = mod.acf_to_acorr(mod.var_acf(true_coefs, sig, nlags=4))

    # --- VARProcess built directly
    p2d = mod.VARProcess(true_coefs, np.column_stack([intercept, [0.1, 0.0, -0.1]]), sig,
                         _params_info={"k_trend": 1, "k_exog_user": 1})
    out["proc2d"] = [p2d.intercept, p2d.k_trend, p2d.k_exog, p2d.mean(), p2d.long_run_effects()]
    p1d = mod.VARProcess(true_coefs, intercept, sig)
    out["proc1d"] = [p1d.intercept, p1d.k_trend, p1d.k_exog, p1d.mean()]
    pnone = mod.VARProcess(true_coefs, None, sig)
    out["procnone"] = [pnone.intercept, pnone.k_trend, pnone.k_exog, pnone.is_stable()]
    p2d_noinfo = mod.VARProcess(true_coefs, np.column_stack([intercept, intercept]), sig)
    out["proc2d_noinfo"] = [p2d_noinfo.k_trend, p2d_noinfo.k_exog]
    out["proc_mse"] = p1d.mse(4)
    out["proc_orth_P"] = p1d.orth_ma_rep(3, P=np.linalg.cholesky(sig) * 2.0)
    out["proc_acorr"] = p1d.acorr()
    out["proc_forecast_bare"] = _err(lambda: p1d.forecast(y[-2:], STEPS))

    # --- VAR fits
    for trend in ("c", "n", "ct", "ctt"):
        res = mod.VAR(y).fit(K_AR, trend=trend)
        out.update(_results(res, f"fit_{trend}", y[-K_AR:]))
    res3 = mod.VAR(y).fit(3, trend="c")
    out.update(_results(res3, "fit3", y[-3:]))
    out["fit3.forecast_cov_auto"] = res3.forecast_cov(STEPS, method="auto")
    out["fit3.omega"] = res3._omega_forc_cov(STEPS)
    res_n = mod.VAR(y).fit(K_AR, trend="n")
    out["fit_n.forecast_cov_auto"] = res_n.forecast_cov(STEPS, method="auto")
    out["fit_bad_method"] = _err(lambda: res3.forecast_cov(STEPS, method="x"))
    out["fit_bad_trend"] = _err(lambda: mod.VAR(y).fit(K_AR, trend="x"))
    out["fit_default_lags"] = mod.VAR(y).fit().params
    out["offset"] = mod.VAR(y)._estimate_var(K_AR, offset=3).params
    out["offset_bad"] = _err(lambda: mod.VAR(y)._estimate_var(K_AR, offset=-1))
    out["fit_1d"] = _err(lambda: mod.VAR(y[:, 0]))

    exog = np.column_stack([np.sin(np.arange(NOBS) / 4.0), rng.normal(size=NOBS)])
    vx = mod.VAR(y, exog=exog)
    resx = vx.fit(K_AR, trend="c")
    out.update(_results(resx, "fitx", y[-K_AR:]))
    ef = rng.normal(size=(STEPS, 2))
    out["fitx.forecast"] = resx.forecast(y[-K_AR:], STEPS, exog_future=ef)
    out["fitx.forecast_interval"] = list(resx.forecast_interval(y[-K_AR:], STEPS, exog_future=ef))
    out["fitx.forecast_cov_auto"] = resx.forecast_cov(STEPS, method="auto")
    out["fitx.no_exog_future"] = _err(lambda: resx.forecast(y[-K_AR:], STEPS))
    out["fitx.short_exog_future"] = _err(lambda: resx.forecast(y[-K_AR:], STEPS, exog_future=ef[:3]))
    out["fitx.ndim_exog_future"] = _err(lambda: resx.forecast(y[-K_AR:], STEPS, exog_future=ef[:, 0]))
    resxt = vx.fit(K_AR, trend="ct")
    out["fitxt.params"] = resxt.params
    out["fitxt.forecast"] = resxt.forecast(y[-K_AR:], STEPS, exog_future=ef)
    resxn = vx.fit(K_AR, trend="n")
    out["fitxn.forecast"] = resxn.forecast(y[-K_AR:], STEPS, exog_future=ef)
    out["fit_c.exog_future_without_exog"] = _err(
        lambda: mod.VAR(y).fit(K_AR).forecast(y[-K_AR:], STEPS, exog_future=ef)
    )

    # --- df_resid == 0: 9 rows, 2 lags, 3 equations, constant: 7 obs, 7 params
    tiny = mod.VAR(y[:9]).fit(K_AR, trend="c")
    out["tiny"] = [tiny.df_resid, tiny.sigma_u, tiny.sigma_u_mle, tiny.info_criteria["fpe"],
                   tiny.info_criteria["aic"]]
    return out
