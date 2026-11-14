"""Compare clean.py against the installed statsmodels (0.15.0).

Prints PASS/FAIL per case, exits 1 on any failure.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.special
import scipy.stats
import statsmodels.api as sm
from statsmodels.tools.tools import matrix_rank

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
                if a is None or b is None:
                    assert a is None and b is None, f"{a} vs {b}"
                else:
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


def design(rng, nobs, k):
    return np.column_stack([np.ones(nobs), rng.normal(size=(nobs, k - 1))])


def mnlogit_case(tag, rng, nobs, k, labels, dummies_upstream=False, check_rank=True):
    X = design(rng, nobs, k)
    y = labels[rng.integers(0, len(labels), nobs)]
    D = C._numpy_to_dummies(y)
    ours = C.MNLogit(D, X, check_rank=check_rank)
    ref = sm.MNLogit(D if dummies_upstream else y, X, check_rank=check_rank)
    J = D.shape[1]
    check(f"{tag} dummies", D, ref.wendog)
    check(f"{tag} J/K/endog", [ours.J, ours.K, ours.endog], [ref.J, ref.K, ref.endog])
    check(f"{tag} df", [ours.df_model, ours.df_resid], [ref.df_model, ref.df_resid])
    flat = rng.normal(scale=0.3, size=k * (J - 1))
    p2d = flat.reshape(k, -1, order="F")
    check(f"{tag} cdf", ours.cdf(X @ p2d), ref.cdf(X @ p2d))
    check(f"{tag} loglike", ours.loglike(flat), ref.loglike(flat))
    check(f"{tag} loglikeobs", ours.loglikeobs(flat), ref.loglikeobs(flat))
    check(f"{tag} score", ours.score(flat), ref.score(flat))
    check(f"{tag} loglike_and_score", ours.loglike_and_score(flat), ref.loglike_and_score(flat))
    check(f"{tag} score_obs", ours.score_obs(flat), ref.score_obs(flat))
    check(f"{tag} hessian", ours.hessian(flat), ref.hessian(flat))
    check(f"{tag} score_factor", ours.score_factor(flat), ref.score_factor(flat))
    check(f"{tag} hessian_factor", ours.hessian_factor(flat), ref.hessian_factor(flat))
    Xnew = design(rng, 7, k)
    for which in ("mean", "linear", "var"):
        check(f"{tag} predict {which}", ours.predict(p2d, which=which), ref.predict(p2d, which=which))
        check(
            f"{tag} predict new {which}",
            ours.predict(p2d, Xnew, which=which),
            ref.predict(p2d, Xnew, which=which),
        )
    check(f"{tag} predict 1-d row", ours.predict(p2d, Xnew[3]), ref.predict(p2d, Xnew[3]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for start, maxiter in ((None, 3), (None, 35), (flat, 50)):
            res = ref.fit(method="newton", start_params=start, maxiter=maxiter, disp=0)
            params, fopt, it, warn = ours.fit(start_params=start, maxiter=maxiter)
            label = f"{tag} fit start={'zeros' if start is None else 'given'} maxiter={maxiter}"
            check(f"{label} params", params, res.params, rtol=1e-8)
            check(
                f"{label} fopt/iterations/warnflag",
                [fopt, it, warn],
                [res.mle_retvals["fopt"], res.mle_retvals["iterations"], res.mle_retvals["warnflag"]],
                rtol=1e-8,
            )


def logit_case(rng):
    nobs, k = 40, 3
    X = design(rng, nobs, k)
    y = (rng.uniform(size=nobs) < 1 / (1 + np.exp(-X @ np.array([0.2, -0.8, 0.5])))).astype(float)
    off = rng.normal(scale=0.1, size=nobs)
    for tag, ours, ref in (
        ("logit", C.Logit(y, X), sm.Logit(y, X)),
        ("logit+offset", C.Logit(y, X, offset=off), sm.Logit(y, X, offset=off)),
        ("logit continuous", C.Logit(y * 0.7 + 0.1, X), sm.Logit(y * 0.7 + 0.1, X)),
    ):
        p = np.array([0.1, -0.5, 0.3])
        lin = X @ p
        Xnew = design(rng, 7, k)
        check(f"{tag} df", [ours.df_model, ours.df_resid], [ref.df_model, ref.df_resid])
        check(f"{tag} cdf/pdf", [ours.cdf(lin), ours.pdf(lin)], [ref.cdf(lin), ref.pdf(lin)])
        for meth in ("loglike", "loglikeobs", "score", "score_obs", "score_factor", "hessian", "hessian_factor"):
            check(f"{tag} {meth}", getattr(ours, meth)(p), getattr(ref, meth)(p))
        for which in ("mean", "linear", "var"):
            check(f"{tag} predict {which}", ours.predict(p, which=which), ref.predict(p, which=which))
            check(
                f"{tag} predict new {which}",
                ours.predict(p, Xnew, which=which, offset=off[:7]),
                ref.predict(p, Xnew, which=which, offset=off[:7]),
            )
        check(f"{tag} predict offset=0.5", ours.predict(p, offset=0.5), ref.predict(p, offset=0.5))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for start, maxiter in ((None, 35), (p, 2)):
                res = ref.fit(method="newton", start_params=start, maxiter=maxiter, disp=0)
                params, fopt, it, warn = ours.fit(start_params=start, maxiter=maxiter)
                check(f"{tag} fit maxiter={maxiter} params", params, res.params, rtol=1e-8)
                check(
                    f"{tag} fit maxiter={maxiter} fopt/iterations/warnflag",
                    [fopt, it, warn],
                    [res.mle_retvals["fopt"], res.mle_retvals["iterations"], res.mle_retvals["warnflag"]],
                    rtol=1e-8,
                )


def poisson_case(rng):
    nobs, k = 40, 3
    X = design(rng, nobs, k)
    y = rng.poisson(np.exp(X @ np.array([0.4, 0.3, -0.2]))).astype(float)
    off = rng.normal(scale=0.1, size=nobs)
    expo = rng.uniform(0.5, 2.0, size=nobs)
    cases = (
        ("poisson", C.Poisson(y, X), sm.Poisson(y, X)),
        ("poisson+offset+exposure", C.Poisson(y, X, offset=off, exposure=expo), sm.Poisson(y, X, offset=off, exposure=expo)),
        ("poisson no const", C.Poisson(y, X[:, 1:]), sm.Poisson(y, X[:, 1:])),
    )
    for tag, ours, ref in cases:
        kk = ours.exog.shape[1]
        p = np.array([0.3, 0.2, -0.1])[:kk]
        lin = ours.exog @ p
        Xnew = design(rng, 7, 3)[:, 3 - kk :]
        check(f"{tag} df", [ours.df_model, ours.df_resid], [ref.df_model, ref.df_resid])
        check(f"{tag} cdf/pdf", [ours.cdf(lin), ours.pdf(lin)], [ref.cdf(lin), ref.pdf(lin)])
        for meth in ("loglike", "loglikeobs", "score", "score_obs", "score_factor", "hessian", "hessian_factor"):
            check(f"{tag} {meth}", getattr(ours, meth)(p), getattr(ref, meth)(p))
        check(f"{tag} start null", ours._get_start_params_null(), ref._get_start_params_null())
        check(f"{tag} const_idx", [C._const_idx(ours.exog)], [ref.data.const_idx])
        for which in ("mean", "linear", "lin", "var", "prob"):
            check(f"{tag} predict {which}", ours.predict(p, which=which), ref.predict(p, which=which))
            check(
                f"{tag} predict new {which}",
                ours.predict(p, Xnew, exposure=expo[:7], offset=off[:7], which=which),
                ref.predict(p, Xnew, exposure=expo[:7], offset=off[:7], which=which),
            )
        yv = np.array([0.0, 2.0, 5.0])
        check(
            f"{tag} predict prob y_values",
            ours.predict(p, Xnew, which="prob", y_values=yv),
            ref.predict(p, Xnew, which="prob", y_values=yv),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for start, maxiter in ((None, 35), (p, 2)):
                res = ref.fit(method="newton", start_params=start, maxiter=maxiter, disp=0)
                params, fopt, it, warn = ours.fit(start_params=start, maxiter=maxiter)
                check(f"{tag} fit maxiter={maxiter} params", params, res.params, rtol=1e-8)
                check(
                    f"{tag} fit maxiter={maxiter} fopt/iterations/warnflag",
                    [fopt, it, warn],
                    [res.mle_retvals["fopt"], res.mle_retvals["iterations"], res.mle_retvals["warnflag"]],
                    rtol=1e-8,
                )


def helpers_case(rng):
    x = np.array([0.5, 1.0, 3.0, 10.5, 101.0])
    check("gammaln", C.gammaln(x), scipy.special.gammaln(x))
    k = np.array([0.0, 1.0, 2.0, 5.0, 3.0, 7.0])
    mu = np.array([0.5, 1.5, 2.0, 4.0, 0.1, 9.0])
    check("poisson logpmf", C._poisson_logpmf(k, mu), scipy.stats.poisson.logpmf(k, mu))
    check("poisson cdf", C._poisson_cdf(k + 0.5, mu), scipy.stats.poisson.cdf(k + 0.5, mu))
    check("poisson pmf grid", C._poisson_pmf(k[None, :], mu[:, None]), scipy.stats.poisson.pmf(k[None, :], mu[:, None]))
    A = rng.normal(size=(9, 4))
    for name, m in (
        ("full", A),
        ("deficient", np.column_stack([A, A[:, 0] + A[:, 1]])),
        ("wide", A.T),
        ("scaled col", np.column_stack([A, 1e-9 * A[:, 0]])),
        ("zeros", np.zeros((3, 2))),
    ):
        check(f"matrix_rank qr {name}", [C._matrix_rank_qr(m)], [matrix_rank(m, method="qr")])
    X = design(rng, 12, 3)
    for name, ex in (
        ("const first", X),
        ("no const", X[:, 1:]),
        ("two consts, one is 1", np.column_stack([X[:, 1:], 2 * X[:, :1], X[:, :1]])),
        ("two consts, 0 and 3", np.column_stack([X[:, 1:], 0 * X[:, :1], 3 * X[:, :1]])),
        ("zero const", np.column_stack([X[:, 1:], 0 * X[:, :1]])),
    ):
        ref = sm.Poisson(np.ones(12), ex)
        check(f"const_idx {name}", [C._const_idx(ex)], [ref.data.const_idx])
    # MNLogit with a rank-deficient design: df_model / df_resid
    Xd = np.column_stack([X, X[:, 1] - X[:, 2]])
    y = np.array([0, 1, 2, 3] * 3, dtype=float)
    ours = C.MNLogit(C._numpy_to_dummies(y), Xd)
    ref = sm.MNLogit(y, Xd)
    check("mnlogit rank-deficient df", [ours.df_model, ours.df_resid], [ref.df_model, ref.df_resid])


def main() -> int:
    rng = np.random.default_rng(2024)
    mnlogit_case("mnlogit J=4 k=3", rng, 40, 3, np.array([1.0, 2.0, 5.0, 7.0]))
    mnlogit_case("mnlogit J=5 k=4 (2-d endog)", rng, 60, 4, np.arange(5.0), dummies_upstream=True)
    mnlogit_case("mnlogit J=3 k=5 no rank check", rng, 50, 5, np.array([3.0, 8.0, 9.0]), check_rank=False)
    logit_case(rng)
    poisson_case(rng)
    helpers_case(rng)
    print(f"{len(FAILS)} failures")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
