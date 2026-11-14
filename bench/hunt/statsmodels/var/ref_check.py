"""Compare clean.py against the installed statsmodels (0.15.0).

Prints PASS/FAIL per case, exits 1 on any failure.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
from statsmodels.tools.linalg import logdet_symm
from statsmodels.tsa import tsatools
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar import util, var_model
from statsmodels.tsa.vector_ar.var_model import VARProcess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("clean", HERE / "clean.py")
assert spec is not None and spec.loader is not None
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

FAILS: list[str] = []


def check(name: str, ours, ref, rtol: float = 1e-9, atol: float = 1e-11) -> None:
    try:
        if isinstance(ours, (list, tuple)):
            assert len(ours) == len(ref), f"length {len(ours)} vs {len(ref)}"
            for a, b in zip(ours, ref):
                a, b = np.asarray(a), np.asarray(b)
                assert a.shape == b.shape, f"shape {a.shape} vs {b.shape}"
                np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
        else:
            a, b = np.asarray(ours), np.asarray(ref)
            assert a.shape == b.shape, f"shape {a.shape} vs {b.shape}"
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    except AssertionError as e:
        FAILS.append(name)
        print(f"FAIL {name}: {str(e).strip().splitlines()[0]}")
        return
    print(f"PASS {name}")


def same_error(name: str, f_ours, f_ref) -> None:
    def kind(f):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            return type(e).__name__
        return "no error"

    a, b = kind(f_ours), kind(f_ref)
    if a == b:
        print(f"PASS {name} ({a})")
    else:
        FAILS.append(name)
        print(f"FAIL {name}: {a} vs {b}")


def simulate(rng, n, neqs, k_ar):
    coefs = rng.normal(scale=0.15, size=(k_ar, neqs, neqs))
    while not var_model.is_stable(coefs):
        coefs *= 0.7
    y = np.zeros((n, neqs))
    y[:k_ar] = rng.normal(size=(k_ar, neqs))
    for t in range(k_ar, n):
        y[t] = 0.2 + sum(coefs[j] @ y[t - j - 1] for j in range(k_ar)) + rng.normal(size=neqs)
    return y, coefs


def compare_fit(tag, y, lags, trend, exog=None, steps=5, rtol=1e-8):
    ours = C.VAR(y, exog=exog).fit(lags, trend=trend)
    ref = VAR(y, exog=exog).fit(lags, trend=trend)
    check(f"{tag} params", ours.params, ref.params, rtol)
    check(f"{tag} endog_lagged", ours.endog_lagged, ref.endog_lagged, rtol)
    check(f"{tag} sigma_u", ours.sigma_u, ref.sigma_u, rtol)
    check(f"{tag} coefs", ours.coefs, ref.coefs, rtol)
    check(f"{tag} coefs_exog", ours.coefs_exog, ref.coefs_exog, rtol)
    check(f"{tag} intercept", ours.intercept, ref.intercept, rtol)
    check(f"{tag} resid", ours.resid, ref.resid, rtol)
    check(f"{tag} fittedvalues", ours.fittedvalues, ref.fittedvalues, rtol)
    check(
        f"{tag} counts",
        [ours.df_model, ours.df_resid, ours.k_exog, ours.k_trend, ours.k_exog_user, ours.nobs],
        [ref.df_model, ref.df_resid, ref.k_exog, ref.k_trend, ref.k_exog_user, ref.nobs],
    )
    check(f"{tag} llf", ours.llf, ref.llf, rtol)
    check(f"{tag} cov_params", ours.cov_params(), ref.cov_params(), rtol)
    check(f"{tag} stderr", ours.stderr, ref.stderr, rtol)
    check(f"{tag} stderr_endog_lagged", ours.stderr_endog_lagged, ref.stderr_endog_lagged, rtol)
    check(f"{tag} stderr_dt", ours.stderr_dt, ref.stderr_dt, rtol)
    check(f"{tag} tvalues", ours.tvalues, ref.tvalues, rtol)
    check(f"{tag} cov_ybar", ours.cov_ybar(), ref.cov_ybar(), rtol)
    check(f"{tag} _cov_alpha", ours._cov_alpha, ref._cov_alpha, rtol)
    check(f"{tag} _cov_sigma", ours._cov_sigma, ref._cov_sigma, rtol)
    check(f"{tag} sigma_u_mle", ours.sigma_u_mle, ref.sigma_u_mle, rtol)
    check(f"{tag} info_criteria", [ours.aic, ours.bic, ours.hqic, ours.fpe],
          [ref.aic, ref.bic, ref.hqic, ref.fpe], rtol)
    check(f"{tag} roots", ours.roots, ref.roots, rtol)
    check(f"{tag} detomega", ours.detomega, ref.detomega, rtol)
    check(f"{tag} sample_acov", ours.sample_acov(3), ref.sample_acov(3), rtol)
    check(f"{tag} sample_acorr", ours.sample_acorr(3), ref.sample_acorr(3), rtol)
    check(f"{tag} resid_acov", ours.resid_acov(2), ref.resid_acov(2), rtol)
    check(f"{tag} resid_acorr", ours.resid_acorr(2), ref.resid_acorr(2), rtol)
    check(f"{tag} resid_corr", ours.resid_corr, ref.resid_corr, rtol)
    check(f"{tag} is_stable", ours.is_stable(), ref.is_stable())
    check(f"{tag} mean", ours.mean(), ref.mean(), rtol)
    check(f"{tag} long_run_effects", ours.long_run_effects(), ref.long_run_effects(), rtol)
    check(f"{tag} ma_rep", ours.ma_rep(steps), ref.ma_rep(steps), rtol)
    check(f"{tag} orth_ma_rep", ours.orth_ma_rep(steps), ref.orth_ma_rep(steps), rtol)
    check(f"{tag} acf", ours.acf(), ref.acf(), rtol)
    check(f"{tag} acf(7)", ours.acf(7), ref.acf(7), rtol)
    check(f"{tag} acorr", ours.acorr(4), ref.acorr(4), rtol)
    check(f"{tag} mse", ours.mse(steps), ref.mse(steps), rtol)
    check(f"{tag} forecast_cov", ours.forecast_cov(steps), ref.forecast_cov(steps), rtol)
    check(f"{tag} forecast_cov()", ours.forecast_cov(), ref.forecast_cov(), rtol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check(f"{tag} forecast_cov auto", ours.forecast_cov(steps, method="auto"),
              ref.forecast_cov(steps, method="auto"), rtol)
    check(f"{tag} _forecast_vars", ours._forecast_vars(steps), ref._forecast_vars(steps), rtol)
    check(f"{tag} _bmat_forc_cov", ours._bmat_forc_cov(), ref._bmat_forc_cov(), rtol)
    if ours.k_ar > 1 or ours.k_exog > 0:
        check(f"{tag} _omega_forc_cov", ours._omega_forc_cov(steps), ref._omega_forc_cov(steps), 1e-7)
    y0 = y[-lags:]
    ef = None if exog is None else np.random.default_rng(7).normal(size=(steps, exog.shape[1]))
    check(f"{tag} forecast", ours.forecast(y0, steps, exog_future=ef),
          ref.forecast(y0, steps, exog_future=ef), rtol)
    check(f"{tag} forecast_interval",
          list(ours.forecast_interval(y0, steps, alpha=0.1, exog_future=ef)),
          list(ref.forecast_interval(y0, steps, alpha=0.1, exog_future=ef)), rtol)


def main() -> None:
    rng = np.random.default_rng(20251113)

    # helpers
    M = rng.normal(size=(4, 3))
    check("vec", C.vec(M), tsatools.vec(M))
    v9 = rng.normal(size=9)
    check("unvec", C.unvec(v9), tsatools.unvec(v9))
    v10 = rng.normal(size=10)
    check("unvech", C.unvech(v10), tsatools.unvech(v10))
    check("duplication_matrix(3)", C.duplication_matrix(3), tsatools.duplication_matrix(3))
    check("duplication_matrix(4)", C.duplication_matrix(4), tsatools.duplication_matrix(4))
    S = np.cov(rng.normal(size=(4, 20)))
    check("logdet_symm", C.logdet_symm(S), logdet_symm(S))
    check("norm_signif_level", C.norm_signif_level(0.07), util.norm_signif_level(0.07), 1e-12)
    for t in ("c", "n", "nc", "ct", "ctt"):
        check(f"get_trendorder {t}", C.get_trendorder(t), util.get_trendorder(t))
    same_error("get_trendorder bad", lambda: C.get_trendorder("q"), lambda: util.get_trendorder("q"))
    X = rng.normal(size=(9, 3))
    Xc = np.column_stack([X, np.full(9, 3.0)])
    for t in ("c", "t", "ct", "ctt", "n"):
        check(f"add_trend {t}", C.add_trend(X, trend=t), tsatools.add_trend(X, trend=t, prepend=True))
        check(f"add_trend {t} skip", C.add_trend(Xc, trend=t, has_constant="skip"),
              tsatools.add_trend(Xc, trend=t, prepend=True, has_constant="skip"))
        check(f"add_trend {t} add", C.add_trend(Xc, trend=t, has_constant="add"),
              tsatools.add_trend(Xc, trend=t, prepend=True, has_constant="add"))
    same_error("add_trend raise", lambda: C.add_trend(Xc, "ct", has_constant="raise"),
               lambda: tsatools.add_trend(Xc, "ct", prepend=True, has_constant="raise"))

    y, coefs = simulate(rng, 60, 3, 2)
    for lags in (1, 2, 4):
        for t in ("c", "n", "ct", "ctt"):
            check(f"get_var_endog lags={lags} {t}", C.get_var_endog(y, lags, trend=t),
                  util.get_var_endog(y, lags, trend=t))
    check("comp_matrix p=2", C.comp_matrix(coefs), util.comp_matrix(coefs))
    check("comp_matrix p=1", C.comp_matrix(coefs[:1]), util.comp_matrix(coefs[:1]))
    same_error("comp_matrix non-square", lambda: C.comp_matrix(np.zeros((2, 3, 4))),
               lambda: util.comp_matrix(np.zeros((2, 3, 4))))
    sig = np.cov(rng.normal(size=(3, 50)))

    # module-level routines, rectangular in (k_ar, neqs, steps)
    for k_ar, neqs in ((2, 3), (3, 2), (1, 4)):
        _, cf = simulate(rng, 10, neqs, k_ar)
        sg = np.cov(rng.normal(size=(neqs, 50)))
        tag = f"k_ar={k_ar} neqs={neqs}"
        check(f"{tag} ma_rep", C.ma_rep(cf, 6), var_model.ma_rep(cf, 6))
        check(f"{tag} is_stable", C.is_stable(cf), var_model.is_stable(cf))
        check(f"{tag} _var_acf", C._var_acf(cf, sg), var_model._var_acf(cf, sg))
        check(f"{tag} var_acf", C.var_acf(cf, sg), var_model.var_acf(cf, sg))
        check(f"{tag} var_acf 7", C.var_acf(cf, sg, 7), var_model.var_acf(cf, sg, 7))
        ma = C.ma_rep(cf, 5)
        check(f"{tag} forecast_cov", C.forecast_cov(ma, sg, 5), var_model.forecast_cov(ma, sg, 5))
        check(f"{tag} mse", C.mse(ma, sg, 4), var_model.mse(ma, sg, 4))
        check(f"{tag} _forecast_vars", C._forecast_vars(5, ma, sg), var_model._forecast_vars(5, ma, sg))
        y0 = rng.normal(size=(k_ar + 2, neqs))
        icpt = rng.normal(size=neqs)
        tc = rng.normal(size=(2, neqs))
        ex = rng.normal(size=(5, 2))
        check(f"{tag} forecast 1-D trend", C.forecast(y0, cf, icpt, 5), var_model.forecast(y0, cf, icpt, 5))
        check(f"{tag} forecast None", C.forecast(y0, cf, None, 5), var_model.forecast(y0, cf, None, 5))
        check(f"{tag} forecast exog", C.forecast(y0, cf, tc, 5, ex), var_model.forecast(y0, cf, tc, 5, ex))
        same_error(f"{tag} forecast short y", lambda: C.forecast(y0[:k_ar - 1], cf, None, 5),
                   lambda: var_model.forecast(y0[:k_ar - 1], cf, None, 5))
        check(f"{tag} forecast_interval exog=1",
              list(C.forecast_interval(y0, cf, icpt[None, :], sg, steps=5)),
              list(var_model.forecast_interval(y0, cf, icpt[None, :], sg, steps=5)))
        check(f"{tag} forecast_interval exog",
              list(C.forecast_interval(y0, cf, tc, sg, steps=5, alpha=0.2, exog=ex)),
              list(var_model.forecast_interval(y0, cf, tc, sg, steps=5, alpha=0.2, exog=ex)))
        r = rng.normal(size=(30, neqs))
        om = r.T @ r / 30
        check(f"{tag} var_loglike", C.var_loglike(r, om, 30), var_model.var_loglike(r, om, 30))
        check(f"{tag} _compute_acov", C._compute_acov(r, 3), var_model._compute_acov(r, 3))
        acv = C._compute_acov(r, 2)
        check(f"{tag} _acovs_to_acorrs", C._acovs_to_acorrs(acv), var_model._acovs_to_acorrs(acv))
        check(f"{tag} acf_to_acorr", C.acf_to_acorr(acv), util.acf_to_acorr(acv))

    # VARProcess built directly
    icpt = np.array([0.3, -0.1, 0.2])
    for name, ce, info in (
        ("2-D with info", np.column_stack([icpt, [0.1, 0.0, -0.1]]), {"k_trend": 1, "k_exog_user": 1}),
        ("1-D", icpt, None),
        ("None", None, None),
        ("2-D no info", np.column_stack([icpt, icpt]), None),
    ):
        ours = C.VARProcess(coefs, ce, sig, _params_info=info)
        ref = VARProcess(coefs, ce, sig, _params_info=info)
        check(f"VARProcess {name} intercept/k", [ours.intercept, ours.k_trend, ours.k_exog],
              [ref.intercept, ref.k_trend, ref.k_exog])
        check(f"VARProcess {name} mean", ours.mean(), ref.mean())
        check(f"VARProcess {name} mse", ours.mse(4), ref.mse(4))
        check(f"VARProcess {name} acorr", ours.acorr(), ref.acorr())
        check(f"VARProcess {name} orth_ma_rep P", ours.orth_ma_rep(3, P=sig), ref.orth_ma_rep(3, P=sig))
        same_error(f"VARProcess {name} forecast", lambda: ours.forecast(y[-2:], 3),
                   lambda: ref.forecast(y[-2:], 3))
    same_error("VARProcess 2-D (neqs, 0) no info",
               lambda: C.VARProcess(coefs, np.zeros((3, 0)), sig),
               lambda: VARProcess(coefs, np.zeros((3, 0)), sig))

    # fits: neqs=3, k_ar=2 and 3, all trends; exog with 2 columns
    for t in ("c", "n", "ct", "ctt"):
        compare_fit(f"fit k_ar=2 {t}", y, 2, t)
    compare_fit("fit k_ar=3 c", y, 3, "c")
    y4, _ = simulate(rng, 45, 4, 3)
    compare_fit("fit neqs=4 k_ar=3 ct", y4, 3, "ct", steps=6)
    exog = np.column_stack([np.sin(np.arange(60) / 4.0), rng.normal(size=60)])
    compare_fit("fit exog c", y, 2, "c", exog=exog)
    compare_fit("fit exog ct", y, 2, "ct", exog=exog)
    compare_fit("fit exog n", y, 2, "n", exog=exog)
    check("default lags", C.VAR(y).fit().params, VAR(y).fit().params)
    check("offset=3", C.VAR(y)._estimate_var(2, offset=3).params, VAR(y)._estimate_var(2, offset=3).params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tiny_o = C.VAR(y[:9]).fit(2)
        tiny_r = VAR(y[:9]).fit(2)
    check("df_resid=0 sigma_u", tiny_o.sigma_u, tiny_r.sigma_u)
    check("df_resid=0 sigma_u_mle", tiny_o.sigma_u_mle, tiny_r.sigma_u_mle)
    check("df_resid=0 ic", [tiny_o.aic, tiny_o.fpe], [tiny_r.aic, tiny_r.fpe])
    same_error("VAR 1-D endog", lambda: C.VAR(y[:, 0]), lambda: VAR(y[:, 0]))
    same_error("bad method", lambda: C.VAR(y).fit(2).forecast_cov(3, method="x"),
               lambda: VAR(y).fit(2).forecast_cov(3, method="x"))

    if FAILS:
        print(f"{len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
