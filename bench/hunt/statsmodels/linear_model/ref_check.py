"""Compare clean.py against the installed statsmodels (0.15.0).

Prints PASS/FAIL per case, exits 1 on any failure.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.linalg
import statsmodels.api as sm
from statsmodels.regression import linear_model as lm
from statsmodels.regression.linear_model import yule_walker
from statsmodels.tools.tools import pinv_extended, recipr

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("clean", HERE / "clean.py")
assert spec is not None and spec.loader is not None
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

FAILS: list[str] = []


def check(name: str, ours, ref, rtol: float = 1e-8, atol: float = 1e-10) -> None:
    try:
        if isinstance(ours, (list, tuple)):
            assert len(ours) == len(ref), f"length {len(ours)} vs {len(ref)}"
            for a, b in zip(ours, ref):
                if a is None or b is None:
                    assert a is None and b is None, f"{a} vs {b}"
                else:
                    a_, b_ = np.asarray(a), np.asarray(b)
                    assert a_.shape == b_.shape, f"shape {a_.shape} vs {b_.shape}"
                    np.testing.assert_allclose(a_, b_, rtol=rtol, atol=atol)
        else:
            a, b = np.asarray(ours), np.asarray(ref)
            assert a.shape == b.shape, f"shape {a.shape} vs {b.shape}"
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
        print(f"PASS {name}")
    except Exception as e:  # noqa: BLE001
        FAILS.append(name)
        print(f"FAIL {name}: {str(e)[:300]}")


RESULT_ATTRS = [
    "params", "bse", "tvalues", "resid", "fittedvalues", "wresid", "ssr", "scale",
    "centered_tss", "uncentered_tss", "ess", "rsquared", "rsquared_adj",
    "mse_model", "mse_resid", "mse_total", "fvalue", "llf", "aic", "bic",
    "eigenvals", "condition_number", "cov_HC0", "cov_HC1", "cov_HC2", "cov_HC3",
    "HC0_se", "HC1_se", "HC2_se", "HC3_se", "resid_pearson", "nobs", "df_model",
    "df_resid", "normalized_cov_params",
]


def compare_results(tag: str, ours, ref, R: np.ndarray, R1: np.ndarray,
                    Xnew: np.ndarray) -> None:
    for a in RESULT_ATTRS:
        check(f"{tag}.{a}", getattr(ours, a), getattr(ref, a))
    check(f"{tag}.cov_params", ours.cov_params(), ref.cov_params())
    check(f"{tag}.cov_params(column=int)", ours.cov_params(column=1),
          ref.cov_params(column=1))
    cols = np.array([2, 0, 1, 2])
    check(f"{tag}.cov_params(column=array)", ours.cov_params(column=cols),
          ref.cov_params(column=cols))
    check(f"{tag}.cov_params(r)", ours.cov_params(r_matrix=R), ref.cov_params(r_matrix=R))
    check(f"{tag}.cov_params(r, scale)", ours.cov_params(r_matrix=R, scale=2.5),
          ref.cov_params(r_matrix=R, scale=2.5))
    other = np.vstack([R, R1])
    check(f"{tag}.cov_params(r, other)", ours.cov_params(r_matrix=R, other=other),
          ref.cov_params(r_matrix=R, other=other))
    for rname, r in (("R(q=2)", R), ("R1(q=1)", R1)):
        t = ref.t_test(r)
        check(f"{tag}.t_test {rname}", list(ours.t_test(r)), [t.effect, t.sd, t.tvalue])
        f = ref.f_test(r)
        check(f"{tag}.f_test {rname}", ours.f_test(r), f.fvalue)
    qv = np.array([0.5, -1.0])
    t = ref.t_test((R, qv))
    check(f"{tag}.t_test q", list(ours.t_test(R, q_matrix=qv)), [t.effect, t.sd, t.tvalue])
    check(f"{tag}.f_test q", ours.f_test(R, q_matrix=qv), ref.f_test((R, qv)).fvalue)
    w = ref.wald_test(R, use_f=False, scalar=True)
    check(f"{tag}.wald chi2", list(ours.wald_test(R, use_f=False)),
          [w.statistic, w.df_denom])
    w = ref.wald_test(R, use_f=True, df_constraints=5.0, scalar=True)
    check(f"{tag}.wald df_constraints", list(ours.wald_test(R, use_f=True,
          df_constraints=5.0)), [w.fvalue, w.df_num])
    inv = np.linalg.inv(ref.cov_params(r_matrix=R))
    w = ref.wald_test(R, invcov=inv, use_f=True, scalar=True)
    check(f"{tag}.wald invcov", list(ours.wald_test(R, invcov=inv, use_f=True)),
          [w.fvalue, w.df_num])
    cp = ref.normalized_cov_params * 3.0
    w = ref.wald_test(R, cov_p=cp, use_f=True, scalar=True)
    check(f"{tag}.wald cov_p", list(ours.wald_test(R, cov_p=cp, use_f=True)),
          [w.fvalue, w.df_num])
    check(f"{tag}.predict(new)", ours.model.predict(ours.params, Xnew),
          ref.model.predict(ref.params, Xnew))


def main() -> int:
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(7)
    nobs, k, m = 30, 4, 3
    X = np.column_stack([np.ones(nobs), rng.normal(size=(nobs, k - 1))])
    beta = np.array([1.0, -2.0, 0.5, 3.0])
    y = X @ beta + rng.normal(size=nobs)
    Y = np.column_stack([y, X @ beta[::-1] + rng.normal(size=nobs),
                         rng.normal(size=nobs)])
    Z = rng.normal(size=(nobs, m))
    Xnew = np.column_stack([np.ones(7), rng.normal(size=(7, k - 1))])
    R = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])
    R1 = np.array([[0.0, 1.0, 1.0, 0.0]])

    # helpers
    c = rng.normal(size=6)
    check("toeplitz", C.toeplitz(c), scipy.linalg.toeplitz(c))
    check("pinv_extended tall", list(C.pinv_extended(X)), list(pinv_extended(X)))
    W = rng.normal(size=(3, 5))
    check("pinv_extended wide", list(C.pinv_extended(W)), list(pinv_extended(W)))
    v = np.array([[2.0, 0.0, -1.0], [np.nan, 0.5, 4.0]])
    check("recipr 2d", C.recipr(v), recipr(v))
    dummies = (rng.integers(0, 2, nobs) == 1).astype(float)
    designs = {
        "explicit": X,
        "none": X[:, 1:].copy(),
        "zero col": np.column_stack([np.zeros(nobs), X[:, 1:]]),
        "two consts": np.column_stack([2 * np.ones(nobs), X]),
        "const 2": np.column_stack([2 * np.ones(nobs), X[:, 1:]]),
        "implicit": np.column_stack([dummies, 1 - dummies, X[:, 1:3]]),
    }
    for name, D in designs.items():
        for hasconst in (None, False, True):
            mod = sm.OLS(y, D, hasconst=hasconst)
            check(f"_handle_constant {name} hasconst={hasconst}",
                  [np.nan if v is None else v for v in C._handle_constant(D, hasconst)],
                  [mod.k_constant, np.nan if mod.data.const_idx is None
                   else mod.data.const_idx])
    sig_full = 0.6 ** np.abs(np.subtract.outer(np.arange(nobs), np.arange(nobs)))
    sig_diag = rng.uniform(0.5, 2.0, nobs)
    for name, s in (("scalar", 1.5), ("1d", sig_diag), ("col", sig_diag[:, None]),
                    ("2d", sig_full)):
        check(f"_get_sigma {name}", list(C._get_sigma(s, nobs)), list(lm._get_sigma(s, nobs)))

    # OLS / WLS / GLS fits
    weights = rng.uniform(0.5, 2.0, nobs)
    cases = [
        ("OLS", C.OLS(y, X), sm.OLS(y, X)),
        ("OLS no const", C.OLS(y, X[:, 1:].copy()), sm.OLS(y, X[:, 1:])),
        ("WLS", C.WLS(y, X, weights=weights), sm.WLS(y, X, weights=weights)),
        ("WLS scalar", C.WLS(y, X, weights=2.0), sm.WLS(y, X, weights=2.0)),
        ("GLS none", C.GLS(y, X), sm.GLS(y, X)),
        ("GLS scalar", C.GLS(y, X, sigma=2.0), sm.GLS(y, X, sigma=2.0)),
        ("GLS 1d", C.GLS(y, X, sigma=sig_diag), sm.GLS(y, X, sigma=sig_diag)),
        ("GLS 2d", C.GLS(y, X, sigma=sig_full), sm.GLS(y, X, sigma=sig_full)),
    ]
    for name, ours, ref in cases:
        Ri, R1i, Xn = (R, R1, Xnew) if ours.exog.shape[1] == k else (
            R[:, 1:].copy(), R1[:, 1:].copy(), Xnew[:, 1:].copy())
        for method in ("pinv", "qr"):
            compare_results(f"{name} {method}", ours.fit(method=method),
                            ref.fit(method=method), Ri, R1i, Xn)
            check(f"{name} {method} rank", ours.rank, ref.rank)
        b = ref.fit().params * 1.1
        check(f"{name} loglike", ours.loglike(b), ref.loglike(b))
        check(f"{name} whiten 1d", ours.whiten(y), ref.whiten(y))
        check(f"{name} whiten 2d (nobs, m)", ours.whiten(Z), ref.whiten(Z))
        try:
            hf_ref = ref.hessian_factor(b)
        except NotImplementedError:
            hf_ref = None
        if hf_ref is not None:
            check(f"{name} hessian_factor", ours.hessian_factor(b), hf_ref)
        for cov_type in ("HC0", "HC1", "HC2", "HC3"):
            ro, rr = ours.fit(cov_type=cov_type), ref.fit(cov_type=cov_type)
            check(f"{name} {cov_type} bse", ro.bse, rr.bse)
            check(f"{name} {cov_type} fvalue", ro.fvalue, rr.fvalue)
            check(f"{name} {cov_type} use_t", ro.use_t, rr.use_t)
            t = rr.t_test(Ri)
            check(f"{name} {cov_type} t_test", list(ro.t_test(Ri)),
                  [t.effect, t.sd, t.tvalue])

    ours, ref = C.OLS(y, X), sm.OLS(y, X)
    b = beta * 0.9
    for scale in (None, 1.7):
        check(f"OLS loglike scale={scale}", ours.loglike(b, scale=scale),
              ref.loglike(b, scale=scale))
        check(f"OLS score scale={scale}", ours.score(b, scale=scale),
              ref.score(b, scale=scale))
        check(f"OLS hessian scale={scale}", ours.hessian(b, scale=scale),
              ref.hessian(b, scale=scale))
    for alpha in (0.3, np.array([0.0, 0.1, 0.5, 1.0])):
        check(f"OLS ridge alpha={alpha}", ours.fit_ridge(alpha),
              ref._fit_ridge(alpha).params)
    impl = designs["implicit"]
    check("OLS implicit const HC1 fvalue", C.OLS(y, impl).fit(cov_type="HC1").fvalue,
          sm.OLS(y, impl).fit(cov_type="HC1").fvalue)
    Xdef = np.column_stack([X, X[:, 1] + X[:, 2]])
    compare_results("OLS rank deficient", C.OLS(y, Xdef).fit(), sm.OLS(y, Xdef).fit(),
                    np.column_stack([R, np.zeros(2)]), np.column_stack([R1, np.zeros(1)]),
                    np.column_stack([Xnew, Xnew[:, 1] + Xnew[:, 2]]))

    # multi-output endog: upstream fits all columns at once; params agree with
    # one fit per column (the results statistics fail upstream, see NOTES.md)
    for name, mk_ours, ref in (
        ("OLS", lambda yy: C.OLS(yy, X), sm.OLS(Y, X)),
        ("WLS", lambda yy: C.WLS(yy, X, weights=weights), sm.WLS(Y, X, weights=weights)),
        ("GLS 2d", lambda yy: C.GLS(yy, X, sigma=sig_full), sm.GLS(Y, X, sigma=sig_full)),
        ("GLS 1d", lambda yy: C.GLS(yy, X, sigma=sig_diag), sm.GLS(Y, X, sigma=sig_diag)),
    ):
        for method in ("pinv", "qr"):
            ours_p = np.column_stack([mk_ours(Y[:, j].copy()).fit(method=method).params
                                      for j in range(Y.shape[1])])
            check(f"{name} multi-output params {method} (k, n_y)", ours_p,
                  ref.fit(method=method).params)

    # yule_walker / GLSAR
    e = np.zeros(nobs + 40)
    shocks = rng.normal(size=nobs + 40)
    for t in range(2, nobs + 40):
        e[t] = 0.5 * e[t - 1] - 0.3 * e[t - 2] + shocks[t]
    for kw in ({"order": 3}, {"order": 2, "method": "mle"}, {"order": 2, "df": 50},
               {"order": 4, "demean": False}):
        check(f"yule_walker {kw}", list(C.yule_walker(e, **kw)),
              list(yule_walker(e, result_object=False, **kw)))
    check("yule_walker singular", list(C.yule_walker(np.zeros(12), order=2, demean=False)),
          list(yule_walker(np.zeros(12), order=2, demean=False, result_object=False)))
    ye = X @ beta + e[40:]
    rho0 = np.array([0.4, -0.2])
    ours, ref = C.GLSAR(ye, X, rho=rho0.copy()), sm.GLSAR(ye, X, rho=rho0.copy())
    check("GLSAR whiten 1d", ours.whiten(ye), ref.whiten(ye))
    check("GLSAR whiten 2d (nobs, m)", ours.whiten(Z), ref.whiten(Z))
    compare_results("GLSAR fit", ours.fit(), ref.fit(), R, R1, Xnew)
    for maxiter, rtol in ((6, 1e-4), (50, 1e-3), (1, 1e-4)):
        ours = C.GLSAR(ye, X, rho=rho0.copy())
        ref = sm.GLSAR(ye, X, rho=rho0.copy())
        ro, rhos, ph, n_iter, conv = ours.iterative_fit(maxiter=maxiter, rtol=rtol)
        rr = ref.iterative_fit(maxiter=maxiter, rtol=rtol)
        tag = f"GLSAR iterative_fit maxiter={maxiter}"
        check(f"{tag} params", ro.params, rr.params)
        check(f"{tag} bse", ro.bse, rr.bse)
        check(f"{tag} rho", ours.rho, ref.rho)
        check(f"{tag} history rho", rhos, rr.history["rho"])
        check(f"{tag} history params", ph, rr.history["params"])
        check(f"{tag} iter/converged", [n_iter, conv], [rr.iter, rr.converged])

    print(f"{len(FAILS)} failures")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
