"""Deterministic harness for the filterpy linear Kalman filter extract.

``run(mod)`` uses 5 states, 3 measurements, 2 controls and 7 epochs (no two
sizes coincide). ``scenario(mod, n_x, n_z, n_u, n_steps, seed)`` is reused by
ref_check.py with other sizes and with an adapter over the upstream library.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _spd(rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.standard_normal((n, n))
    return a @ a.T / n + np.eye(n)


def _setup(mod: Any, rng: np.random.Generator, n_x: int, n_z: int, n_u: int) -> Any:
    kf = mod.KalmanFilter(n_x, n_z, n_u)
    kf.F = np.eye(n_x) + 0.1 * rng.standard_normal((n_x, n_x))
    kf.H = rng.standard_normal((n_z, n_x))
    kf.R = _spd(rng, n_z)
    kf.Q = 0.1 * _spd(rng, n_x)
    kf.P = _spd(rng, n_x)
    kf.x = rng.standard_normal((n_x, 1))
    if n_u:
        kf.B = rng.standard_normal((n_x, n_u))
    return kf


def _state(kf: Any) -> dict[str, Any]:
    return {
        "x": kf.x.copy(),
        "P": kf.P.copy(),
        "K": kf.K.copy(),
        "y": kf.y.copy(),
        "S": kf.S.copy(),
        "x_prior": kf.x_prior.copy(),
        "P_post": kf.P_post.copy(),
    }


def scenario(mod: Any, n_x: int, n_z: int, n_u: int, n_steps: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}

    # ---------------------------------------------------------- __init__ checks
    for bad in ((0, n_z, 0), (n_x, 0, 0), (n_x, n_z, -1)):
        try:
            mod.KalmanFilter(*bad)
            out[f"init_err_{bad}"] = "no error"
        except ValueError as e:
            out[f"init_err_{bad}"] = str(e)
    fresh = mod.KalmanFilter(n_x, n_z)
    out["init_shapes"] = [fresh.x.shape, fresh.H.shape, fresh.R.shape, fresh.M.shape, fresh.z.shape]

    # ---------------------------------------------------------- reshape_z
    zrow = rng.standard_normal((1, n_z))
    out["reshape_row"] = mod.reshape_z(zrow, n_z)
    out["reshape_flat"] = mod.reshape_z(zrow[0], n_z)
    out["reshape_col"] = mod.reshape_z(zrow.T, n_z)
    try:
        mod.reshape_z(np.zeros((n_z + 1, 1)), n_z)
        out["reshape_err"] = "no error"
    except ValueError as e:
        out["reshape_err"] = str(e)

    # ---------------------------------------------------------- predict / update
    kf = _setup(mod, rng, n_x, n_z, n_u)
    u = rng.standard_normal((n_u, 1))
    if n_u:
        kf.predict(u=u)
    out["predict_u"] = _state(kf)
    kf.predict()
    out["predict_plain"] = _state(kf)
    kf.predict(Q=0.3)
    out["predict_Qscalar"] = _state(kf)
    F2 = np.eye(n_x) + 0.05 * rng.standard_normal((n_x, n_x))
    if n_u:
        B2 = rng.standard_normal((n_x, n_u))
        kf.predict(u=u, B=B2, F=F2, Q=kf.Q * 2)
    else:
        kf.predict(F=F2, Q=kf.Q * 2)
    out["predict_override"] = _state(kf)

    z = rng.standard_normal((n_z, 1))
    kf.update(z)
    out["update"] = _state(kf)
    out["log_likelihood"] = kf.log_likelihood
    out["likelihood"] = kf.likelihood
    out["mahalanobis"] = kf.mahalanobis
    out["log_likelihood_of"] = kf.log_likelihood_of(rng.standard_normal((n_z, 1)))
    out["log_likelihood_of_none"] = kf.log_likelihood_of(None)

    kf.predict()
    kf.update(rng.standard_normal((n_z, 1)), R=0.7)
    out["update_Rscalar"] = _state(kf)
    kf.predict()
    H2 = rng.standard_normal((n_z, n_x))
    kf.update(rng.standard_normal((n_z, 1)), R=_spd(rng, n_z), H=H2)
    out["update_H"] = _state(kf)
    kf.predict()
    kf.update(None)
    out["update_none"] = [kf.x_post.copy(), kf.P_post.copy(), kf.y.copy(), kf.z.shape]

    # ---------------------------------------------------------- queries
    xg, Pg = kf.get_prediction(u=u if n_u else None)
    out["get_prediction"] = [xg, Pg]
    out["get_prediction_override"] = list(kf.get_prediction(F=F2, Q=0.2))
    out["get_update"] = list(kf.get_update(rng.standard_normal((n_z, 1))))
    out["get_update_none"] = list(kf.get_update(None))
    out["residual_of"] = kf.residual_of(rng.standard_normal((n_z, 1)))
    out["measurement_of_state"] = kf.measurement_of_state(rng.standard_normal((n_x, 1)))

    # ---------------------------------------------------------- steady state
    ks = _setup(mod, rng, n_x, n_z, n_u)
    ks.K = 0.1 * rng.standard_normal((n_x, n_z))
    if n_u:
        ks.predict_steadystate(u=u)
    else:
        ks.predict_steadystate()
    ks.update_steadystate(rng.standard_normal((n_z, 1)))
    out["steady"] = [ks.x.copy(), ks.y.copy(), ks.P.copy()]
    ks.B = None  # predict_steadystate() with B set and the default u=0 corrupts x for n_u > 1
    ks.predict_steadystate()
    ks.update_steadystate(rng.standard_normal(n_z))
    out["steady_flat"] = [ks.x.copy(), ks.y.copy()]
    ks.update_steadystate(None)
    out["steady_none"] = [ks.x_post.copy(), ks.y.copy()]

    # ---------------------------------------------------------- correlated
    kc = _setup(mod, rng, n_x, n_z, n_u)
    kc.M = 0.05 * rng.standard_normal((n_x, n_z))
    kc.predict()
    kc.update_correlated(rng.standard_normal((n_z, 1)))
    out["correlated"] = _state(kc)
    kc.predict()
    kc.update_correlated(rng.standard_normal((n_z, 1)), R=0.9, H=rng.standard_normal((n_z, n_x)))
    out["correlated_RH"] = _state(kc)
    kc.update_correlated(None)
    out["correlated_none"] = [kc.y.copy(), kc.z.shape]

    # ---------------------------------------------------------- sequential
    kq = _setup(mod, rng, n_x, n_z, n_u)
    kq.predict()
    kq.z = np.zeros((n_z, 1))
    kq.update_sequential(0, float(rng.standard_normal()))
    out["seq_scalar"] = _state(kq)
    if n_z >= 3:
        kq.update_sequential(1, rng.standard_normal(2))
        out["seq_block_default"] = _state(kq)
        kq.update_sequential(1, rng.standard_normal(2), R_i=0.5, H_i=rng.standard_normal((2, n_x)))
        out["seq_block_scalarR_H"] = _state(kq)
        kq.update_sequential(0, rng.standard_normal(2), R_i=_spd(rng, 2))
        out["seq_block_R"] = _state(kq)
    out["seq_z"] = kq.z.astype(float)

    # ---------------------------------------------------------- batch + smoother (class)
    kb = _setup(mod, rng, n_x, n_z, n_u)
    zs: list[Any] = [rng.standard_normal((n_z, 1)) for _ in range(n_steps)]
    zs[2] = None
    us = [rng.standard_normal((n_u, 1)) for _ in range(n_steps)] if n_u else None
    m, c, mp, cp = kb.batch_filter(zs, us=us)
    out["batch"] = [m, c, mp, cp]
    Fs = [np.eye(n_x) + 0.1 * rng.standard_normal((n_x, n_x)) for _ in range(n_steps)]
    Qs = [0.1 * _spd(rng, n_x) for _ in range(n_steps)]
    Hs = [rng.standard_normal((n_z, n_x)) for _ in range(n_steps)]
    Rs = [_spd(rng, n_z) for _ in range(n_steps)]
    kb2 = _setup(mod, rng, n_x, n_z, n_u)
    m2, c2, mp2, cp2 = kb2.batch_filter(zs, Fs=Fs, Qs=Qs, Hs=Hs, Rs=Rs, us=us, update_first=True)
    out["batch_varying_update_first"] = [m2, c2, mp2, cp2]
    kb3 = _setup(mod, rng, n_x, n_z, 0)
    kb3.B = None
    out["batch_no_control"] = list(kb3.batch_filter(zs))

    out["rts_class"] = list(kb.rts_smoother(m, c))
    out["rts_class_varying"] = list(kb.rts_smoother(m2, c2, Fs=Fs, Qs=Qs, inv=np.linalg.pinv))
    try:
        kb.rts_smoother(m, c[:-1])
        out["rts_len_err"] = "no error"
    except ValueError as e:
        out["rts_len_err"] = str(e)

    # ---------------------------------------------------------- module functions
    x0 = rng.standard_normal((n_x, 1))
    P0 = _spd(rng, n_x)
    H = rng.standard_normal((n_z, n_x))
    R = _spd(rng, n_z)
    zc = rng.standard_normal((n_z, 1))
    out["f_update"] = list(mod.update(x0, P0, zc, R, H))
    out["f_update_all"] = list(mod.update(x0, P0, zc[:, 0], R, H, return_all=True))
    out["f_update_none"] = list(mod.update(x0, P0, None, R, H))
    out["f_update_none_all"] = [v for v in mod.update(x0, P0, None, R, H, return_all=True) if v is not None]
    K = 0.1 * rng.standard_normal((n_x, n_z))
    out["f_update_steadystate"] = mod.update_steadystate(x0, zc, K, H)
    out["f_update_steadystate_none"] = mod.update_steadystate(x0, None, K, H)
    F = np.eye(n_x) + 0.1 * rng.standard_normal((n_x, n_x))
    Q = 0.1 * _spd(rng, n_x)
    out["f_predict"] = list(mod.predict(x0, P0, F, Q))
    if n_u:
        Bm = rng.standard_normal((n_x, n_u))
        out["f_predict_u"] = list(mod.predict(x0, P0, F, Q, u=u, B=Bm, alpha=1.02))
    fb = mod.batch_filter(x0, P0, zs, Fs, Qs, Hs, Rs)
    out["f_batch"] = list(fb)
    if n_u:
        Bs = [rng.standard_normal((n_x, n_u)) for _ in range(n_steps)]
        out["f_batch_u_first"] = list(
            mod.batch_filter(x0, P0, zs, Fs, Qs, Hs, Rs, Bs=Bs, us=us, update_first=True)
        )
    out["f_rts"] = list(mod.rts_smoother(fb[0], fb[1], Fs, Qs))
    try:
        mod.rts_smoother(fb[0][:-1], fb[1], Fs, Qs)
        out["f_rts_len_err"] = "no error"
    except ValueError as e:
        out["f_rts_len_err"] = str(e)

    # logpdf directly (singular covariance included)
    v = rng.standard_normal((n_z, 1))
    out["logpdf"] = mod.logpdf(v, None, R)
    Ssing = H @ H.T if n_z > n_x else np.diag(np.r_[np.ones(n_z - 1), 0.0])
    out["logpdf_singular"] = mod.logpdf(v[:, 0], np.zeros(n_z), Ssing)
    vin = v[:, 0].copy()
    vin[-1] = 0.0
    out["logpdf_singular_in"] = mod.logpdf(vin, np.zeros(n_z), np.diag(np.r_[np.ones(n_z - 1), 0.0]))
    return out


def run(mod: Any) -> dict[str, Any]:
    return scenario(mod, n_x=5, n_z=3, n_u=2, n_steps=7, seed=20261113)
