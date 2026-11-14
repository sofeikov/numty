"""Runtime harness for the MNLogit / Logit / Poisson extract.

Sizes never coincide: nobs=40, k_exog=3, J=4 (3 non-base classes), and a
prediction set of 7 rows. A second MNLogit case uses J=5, k_exog=4.
"""

from __future__ import annotations

import warnings

import numpy as np


def _design(rng: np.random.Generator, nobs: int, k: int) -> np.ndarray:
    return np.column_stack([np.ones(nobs), rng.normal(size=(nobs, k - 1))])


def _mnlogit(mod, rng, nobs: int, k: int, labels: np.ndarray, tag: str) -> dict:
    X = _design(rng, nobs, k)
    y = labels[rng.integers(0, len(labels), nobs)]
    D = mod._numpy_to_dummies(y)
    m = mod.MNLogit(D, X)
    m_nr = mod.MNLogit(D, X, check_rank=False)
    J = D.shape[1]
    flat = rng.normal(scale=0.3, size=k * (J - 1))
    p2d = flat.reshape(k, -1, order="F")
    Xnew = _design(rng, 7, k)
    out = {
        "dummies": D,
        "endog": m.endog,
        "J": m.J,
        "K": m.K,
        "df": [m.df_model, m.df_resid, m_nr.df_model, m_nr.df_resid],
        "cdf": m.cdf(X @ p2d),
        "loglike": m.loglike(flat),
        "loglikeobs": m.loglikeobs(flat),
        "score": m.score(flat),
        "loglike_and_score": list(m.loglike_and_score(flat)),
        "score_obs": m.score_obs(flat),
        "hessian": m.hessian(flat),
        "score_factor": m.score_factor(flat),
        "hessian_factor": m.hessian_factor(flat),
        "predict_mean": m.predict(p2d),
        "predict_linear": m.predict(p2d, which="linear"),
        "predict_var": m.predict(p2d, which="var"),
        "predict_new": m.predict(p2d, Xnew),
        "predict_new_linear": m.predict(p2d, Xnew, which="linear"),
        "predict_row": m.predict(p2d, Xnew[2]),
    }
    try:
        m.predict(p2d, which="bogus")
    except ValueError as e:
        out["predict_bad_which"] = str(e)
    try:
        m.pdf(X @ p2d)
    except NotImplementedError:
        out["pdf"] = "NotImplementedError"
    params, fopt, it, warn = m.fit(maxiter=4)
    out["fit4"] = [params, fopt, it, warn]
    params, fopt, it, warn = m.fit(start_params=flat, maxiter=50)
    out["fit"] = [params, fopt, it, warn]
    return {tag + "_" + k_: v for k_, v in out.items()}


def _logit(mod, rng) -> dict:
    nobs, k = 40, 3
    X = _design(rng, nobs, k)
    beta = np.array([0.2, -0.8, 0.5])
    y = (rng.uniform(size=nobs) < 1 / (1 + np.exp(-X @ beta))).astype(float)
    off = rng.normal(scale=0.1, size=nobs)
    m = mod.Logit(y, X)
    mo = mod.Logit(y, X, offset=off)
    p = np.array([0.1, -0.5, 0.3])
    Xnew = _design(rng, 7, k)
    lin = X @ p
    out = {
        "df": [m.df_model, m.df_resid],
        "cdf": m.cdf(lin),
        "pdf": m.pdf(lin),
        "loglike": [m.loglike(p), mo.loglike(p)],
        "loglikeobs": m.loglikeobs(p),
        "loglikeobs_off": mo.loglikeobs(p),
        "score": m.score(p),
        "score_obs": m.score_obs(p),
        "score_factor": m.score_factor(p),
        "hessian": m.hessian(p),
        "hessian_factor": m.hessian_factor(p),
        "predict": [m.predict(p), m.predict(p, which="linear"), m.predict(p, which="var")],
        "predict_off": [mo.predict(p), mo.predict(p, offset=0.5)],
        "predict_new": [m.predict(p, Xnew), mo.predict(p, Xnew, which="linear", offset=off[:7])],
    }
    try:
        m.predict(p, which="bogus")
    except ValueError as e:
        out["predict_bad_which"] = str(e)
    for bad in (y + 2.0, y * 0.5):
        try:
            mod.Logit(bad, X)
            out.setdefault("bad_endog", []).append("ok")
        except ValueError as e:
            out.setdefault("bad_endog", []).append(str(e))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["fit"] = list(m.fit())
        out["fit_off"] = list(mo.fit(start_params=p, maxiter=3))
    # perfect prediction: the callback warns when fitted values equal endog
    ysep = (X[:, 1] > 0).astype(float)
    msep = mod.Logit(ysep, X)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        msep._check_perfect_pred(np.array([0.0, 1e6, 0.0]))
        msep._check_perfect_pred(p)
    out["perfect_pred"] = [str(x.message) for x in w if "Perfect" in str(x.message)]
    return {"logit_" + k_: v for k_, v in out.items()}


def _binary_model_errors(mod, rng) -> dict:
    X = _design(rng, 12, 3)
    out = []
    for y in (np.r_[np.zeros(6), np.ones(6)] * 0.5, np.r_[np.zeros(6), np.ones(6)]):
        try:
            mod.BinaryModel(y, X)
            out.append("ok")
        except ValueError as e:
            out.append(str(e))
    try:
        mod.BinaryModel(np.ones(12), X).predict(np.zeros(3))
    except NotImplementedError:
        out.append("NotImplementedError")
    return {"binary_errors": out}


def _poisson(mod, rng) -> dict:
    nobs, k = 40, 3
    X = _design(rng, nobs, k)
    beta = np.array([0.4, 0.3, -0.2])
    y = rng.poisson(np.exp(X @ beta)).astype(float)
    off = rng.normal(scale=0.1, size=nobs)
    expo = rng.uniform(0.5, 2.0, size=nobs)
    m = mod.Poisson(y, X)
    mo = mod.Poisson(y, X, offset=off, exposure=expo)
    p = np.array([0.3, 0.2, -0.1])
    Xnew = _design(rng, 7, k)
    lin = X @ p
    out = {
        "df": [m.df_model, m.df_resid],
        "cdf": m.cdf(lin),
        "pdf": m.pdf(lin),
        "loglike": [m.loglike(p), mo.loglike(p)],
        "loglikeobs": [m.loglikeobs(p), mo.loglikeobs(p)],
        "start_null": [m._get_start_params_null()[0], mo._get_start_params_null()[0]],
        "score": [m.score(p), mo.score(p)],
        "score_obs": mo.score_obs(p),
        "score_factor": mo.score_factor(p),
        "hessian": [m.hessian(p), mo.hessian(p)],
        "hessian_factor": mo.hessian_factor(p),
        "predict": [m.predict(p), mo.predict(p), mo.predict(p, which="linear")],
        "predict_new": [
            m.predict(p, Xnew),
            mo.predict(p, Xnew, exposure=expo[:7], offset=off[:7]),
            mo.predict(p, Xnew, which="lin"),
        ],
        "const_idx": [
            mod._const_idx(X),
            mod._const_idx(X[:, 1:]),
            mod._const_idx(np.column_stack([X, 2 * X[:, :1]])),
            mod._const_idx(np.column_stack([X[:, 1:], 0 * X[:, :1], 3 * X[:, :1]])),
            mod._const_idx(np.column_stack([X[:, 1:], 0 * X[:, :1]])),
        ],
    }
    out["predict_var"] = mo.predict(p, which="var")
    out["predict_prob"] = m.predict(p, which="prob")
    out["predict_prob_new"] = mo.predict(
        p, Xnew, exposure=expo[:7], offset=off[:7], which="prob", y_values=np.arange(5.0)
    )
    try:
        m.predict(p, which="bogus")  # type: ignore[call-overload]
    except ValueError as e:
        out["predict_bad_which"] = str(e)
    try:
        m._predict_count(p, which="var")
    except ValueError as e:
        out["predict_count_bad_which"] = str(e)
    for kw in ({"offset": off[:5]}, {"exposure": expo[:5]}):
        try:
            mod.Poisson(y, X, **kw)
        except ValueError as e:
            out.setdefault("bad_len", []).append(str(e))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["fit"] = list(m.fit())
        out["fit_off"] = list(mo.fit())
        out["fit_noconst"] = list(mod.Poisson(y, X[:, 1:]).fit())
        out["fit_start"] = list(m.fit(start_params=p, maxiter=2))
    m0 = mod.Poisson(np.zeros(nobs), X)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m0._check_perfect_pred(np.array([-60.0, 0.0, 0.0]))
        m0._check_perfect_pred(p)
    out["perfect_pred"] = [str(x.message) for x in w if "Perfect" in str(x.message)]
    return {"poisson_" + k_: v for k_, v in out.items()}


def _helpers(mod, rng) -> dict:
    A = rng.normal(size=(9, 4))
    deficient = np.column_stack([A, A[:, 0] + A[:, 1]])
    k = np.array([0.0, 1.0, 2.0, 5.0, 3.0])
    mu = np.array([0.5, 1.5, 2.0, 4.0, 0.1])
    return {
        "h_rank": [
            mod._matrix_rank_qr(A),
            mod._matrix_rank_qr(deficient),
            mod._matrix_rank_qr(A.T),
            mod._matrix_rank_qr(np.zeros((3, 2))),
        ],
        "h_gammaln": mod.gammaln(np.array([0.5, 1.0, 3.0, 10.5])),
        "h_logpmf": mod._poisson_logpmf(k, mu),
        "h_cdf": mod._poisson_cdf(k + 0.5, mu),
        "h_cdf_empty": mod._poisson_cdf(np.zeros(0), np.zeros(0)),
    }


def run(mod) -> dict:
    rng = np.random.default_rng(12345)
    out: dict = {}
    out.update(_mnlogit(mod, rng, 40, 3, np.array([1.0, 2.0, 5.0, 7.0]), "mn"))
    out.update(_mnlogit(mod, rng, 60, 4, np.array([0.0, 1.0, 2.0, 3.0, 4.0]), "mn5"))
    out.update(_logit(mod, rng))
    out.update(_binary_model_errors(mod, rng))
    out.update(_poisson(mod, rng))
    out.update(_helpers(mod, rng))
    return out
