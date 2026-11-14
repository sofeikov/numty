"""Runtime harness for bench/hunt/filterpy/ekf_ckf/clean.py.

Sizes never coincide: 5 states, 3 measurements, 2 control inputs, plus a
1-measurement filter for the scalar-z paths. Seeds are fixed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

N_X, N_Z, N_U = 5, 3, 2


# ------------------------------------------------------------ EKF models


def hx_ekf(x: Any, scale: float = 1.0) -> Any:
    """(N_X, 1) state -> (N_Z, 1) measurement, nonlinear."""
    x = x.ravel()
    return scale * np.array([[np.hypot(x[0], x[1])], [x[2] * x[3]], [np.sin(x[4])]])


def H_ekf(x: Any, scale: float = 1.0) -> Any:
    """Jacobian of hx_ekf, (N_Z, N_X)."""
    x = x.ravel()
    r = np.hypot(x[0], x[1])
    H = np.zeros((N_Z, N_X))
    H[0, 0] = x[0] / r
    H[0, 1] = x[1] / r
    H[1, 2] = x[3]
    H[1, 3] = x[2]
    H[2, 4] = np.cos(x[4])
    return scale * H


def hx_ekf1(x: Any, offset: float = 0.0) -> Any:
    x = x.ravel()
    return np.array([[np.hypot(x[0], x[2]) + offset]])


def H_ekf1(x: Any, offset: float = 0.0) -> Any:
    x = x.ravel()
    r = np.hypot(x[0], x[2])
    H = np.zeros((1, N_X))
    H[0, 0] = x[0] / r
    H[0, 2] = x[2] / r
    return H


def angle_residual(a: Any, b: Any) -> Any:
    y = np.asarray(a - b, dtype=float)
    y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi
    return y


def _spd(rng: np.random.Generator, n: int, scale: float) -> Any:
    A = rng.normal(size=(n, n))
    return scale * (A @ A.T / n + np.eye(n))


def _setup_ekf(mod: Any, rng: np.random.Generator, dim_z: int) -> Any:
    ekf = mod.ExtendedKalmanFilter(N_X, dim_z, N_U)
    ekf.x = np.array([[3.0], [4.0], [1.5], [-0.5], [0.3]])
    ekf.P = _spd(rng, N_X, 0.5)
    ekf.F = np.eye(N_X) + 0.1 * rng.normal(size=(N_X, N_X))
    ekf.Q = _spd(rng, N_X, 0.01)
    ekf.R = _spd(rng, dim_z, 0.1)
    # B stays the scalar 0 default here: upstream predict() with a matrix B
    # and the default u=0 turns x into (N_X, N_U) (see NOTES.md), so the
    # matrix B is only set before calls that pass u
    return ekf


def _ekf_state(ekf: Any) -> dict[str, Any]:
    return {
        "x": ekf.x.copy(), "P": ekf.P.copy(), "K": ekf.K.copy(), "y": np.copy(ekf.y),
        "S": ekf.S.copy(), "SI": ekf.SI.copy(), "x_prior": ekf.x_prior.copy(),
        "P_prior": ekf.P_prior.copy(), "x_post": ekf.x_post.copy(),
        "P_post": ekf.P_post.copy(), "z": np.asarray(ekf.z, dtype=float).copy(),
    }


def run_ekf(mod: Any) -> dict[str, Any]:
    rng = np.random.default_rng(1234)
    out: dict[str, Any] = {}

    # multi-output filter: predict (no control, control), update variants
    ekf = _setup_ekf(mod, rng, N_Z)
    out["init_K"] = ekf.K.copy()
    ekf.predict()
    out["predict_noctl"] = _ekf_state(ekf)
    ekf.B = rng.normal(size=(N_X, N_U))
    u = rng.normal(size=(N_U, 1))
    ekf.predict(u)
    out["predict_ctl"] = _ekf_state(ekf)
    ekf.predict_x(u)
    out["predict_x"] = ekf.x.copy()

    z = hx_ekf(ekf.x) + 0.05 * rng.normal(size=(N_Z, 1))
    ekf.update(z, H_ekf, hx_ekf)
    out["update_plain"] = _ekf_state(ekf)
    out["loglik"] = ekf.log_likelihood
    out["lik"] = ekf.likelihood
    out["maha"] = ekf.mahalanobis

    z = hx_ekf(ekf.x, 1.1) + 0.05 * rng.normal(size=(N_Z, 1))
    ekf.update(z, H_ekf, hx_ekf, R=0.2, args=1.1, hx_args=(1.1,))
    out["update_Rscalar_args"] = _ekf_state(ekf)

    Rm = _spd(rng, N_Z, 0.3)
    z = hx_ekf(ekf.x) + 0.05 * rng.normal(size=(N_Z, 1))
    ekf.update(z, H_ekf, hx_ekf, R=Rm, residual=angle_residual)
    out["update_Rmatrix_residual"] = _ekf_state(ekf)
    out["maha2"] = ekf.mahalanobis

    ekf.update(None, H_ekf, hx_ekf)
    out["update_none_xpost"] = ekf.x_post.copy()
    out["update_none_z_shape"] = np.array(np.shape(ekf.z))

    z = hx_ekf(ekf.x) + 0.05 * rng.normal(size=(N_Z, 1))
    ekf.predict_update(z, H_ekf, hx_ekf, u=rng.normal(size=(N_U, 1)))
    out["predict_update_ctl"] = _ekf_state(ekf)
    z = hx_ekf(ekf.x, 0.9) + 0.05 * rng.normal(size=(N_Z, 1))
    ekf.predict_update(z, H_ekf, hx_ekf, args=0.9, hx_args=0.9, u=0.5 * u)
    out["predict_update_args"] = _ekf_state(ekf)
    out["loglik2"] = ekf.log_likelihood

    # single-output filter: scalar z paths
    ekf1 = _setup_ekf(mod, rng, 1)
    ekf1.predict()
    zs = float(hx_ekf1(ekf1.x)[0, 0] + 0.1)
    ekf1.update(zs, H_ekf1, hx_ekf1)
    out["ekf1_update_scalar"] = _ekf_state(ekf1)
    ekf1.update(zs + 0.2, H_ekf1, hx_ekf1, R=0.3, args=(0.1,), hx_args=0.1)
    out["ekf1_update_scalar_R"] = _ekf_state(ekf1)
    ekf1.B = rng.normal(size=(N_X, N_U))
    ekf1.predict_update(zs - 0.1, H_ekf1, hx_ekf1, u=rng.normal(size=(N_U, 1)))
    out["ekf1_predict_update_scalar"] = _ekf_state(ekf1)
    out["ekf1_maha"] = ekf1.mahalanobis
    out["ekf1_lik"] = ekf1.likelihood

    # an innovation far outside S: exp(log-likelihood) underflows to 0 and
    # the likelihood is floored at sys.float_info.min
    ekf1.update(zs + 1e4, H_ekf1, hx_ekf1, R=1e-6)
    out["ekf1_far_loglik"] = ekf1.log_likelihood
    out["ekf1_far_lik"] = ekf1.likelihood
    return out


# ------------------------------------------------------------ CKF models


def fx_ckf(x: Any, dt: float, damping: float = 0.0) -> Any:
    """(N_X,) -> (N_X,), nonlinear constant-turn-like model."""
    x = np.asarray(x, dtype=float)
    return np.array([
        x[0] + dt * x[2] * np.cos(x[4]),
        x[1] + dt * x[2] * np.sin(x[4]),
        x[2] * (1.0 - damping * dt),
        x[3] + dt * x[2] * 0.1,
        x[4] + dt * 0.05 * np.tanh(x[3]),
    ])


def hx_ckf(x: Any, bias: float = 0.0) -> Any:
    """(N_X,) -> (N_Z,)"""
    return np.array([np.hypot(x[0], x[1]) + bias, np.arctan2(x[1], x[0]), x[2] + 0.5 * x[3]])


def bearing_residual(a: Any, b: Any) -> Any:
    y = np.asarray(a - b, dtype=float)
    y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
    return y


def hx_ckf1(x: Any) -> Any:
    return np.array([x[0] + 0.3 * x[1] * x[1]])


def run_ckf(mod: Any) -> dict[str, Any]:
    rng = np.random.default_rng(99)
    out: dict[str, Any] = {}

    P = _spd(rng, N_X, 0.4)
    x_flat = np.array([3.0, 4.0, 1.0, 0.2, 0.4])
    out["sigmas_flat"] = mod.spherical_radial_sigmas(x_flat, P)
    out["sigmas_col"] = mod.spherical_radial_sigmas(x_flat[:, None], P)
    Xs = rng.normal(size=(2 * N_X, N_Z))
    m, C = mod.ckf_transform(Xs, _spd(rng, N_Z, 0.1))
    out["transform_mean"] = m
    out["transform_cov"] = C

    ckf = mod.CubatureKalmanFilter(N_X, N_Z, 0.1, hx_ckf, fx_ckf)
    ckf.x = x_flat[:, None].copy()
    ckf.P = P
    ckf.Q = _spd(rng, N_X, 0.01)
    ckf.R = _spd(rng, N_Z, 0.05)

    ckf.predict()
    out["predict"] = (ckf.x.copy(), ckf.P.copy(), ckf.sigmas_f.copy())
    z = hx_ckf(ckf.x.ravel())[:, None] + 0.05 * rng.normal(size=(N_Z, 1))
    ckf.update(z)
    out["update"] = (ckf.x.copy(), ckf.P.copy(), ckf.K.copy(), ckf.y.copy(), ckf.S.copy(), ckf.sigmas_h.copy())
    out["loglik"] = ckf.log_likelihood
    out["lik"] = ckf.likelihood
    out["maha"] = ckf.mahalanobis

    ckf.predict(dt=0.2, fx_args=0.1)
    out["predict_dt_args"] = (ckf.x.copy(), ckf.P.copy())
    z = hx_ckf(ckf.x.ravel(), 0.2)[:, None] + 0.05 * rng.normal(size=(N_Z, 1))
    ckf.update(z, R=0.07, hx_args=0.2)
    out["update_Rscalar"] = (ckf.x.copy(), ckf.P.copy(), ckf.K.copy(), ckf.y.copy())

    ckf.predict(fx_args=(0.05,))
    z = hx_ckf(ckf.x.ravel(), 0.1)[:, None] + 0.05 * rng.normal(size=(N_Z, 1))
    ckf.update(z, R=_spd(rng, N_Z, 0.2), hx_args=(0.1,))
    out["update_Rmatrix"] = (ckf.x.copy(), ckf.P.copy(), ckf.K.copy())
    out["maha2"] = ckf.mahalanobis

    ckf.update(None)
    out["update_none"] = (ckf.x_post.copy(), np.array(np.shape(ckf.z)))

    # single-measurement filter
    ckf1 = mod.CubatureKalmanFilter(N_X, 1, 0.1, hx_ckf1, fx_ckf)
    ckf1.x = x_flat[:, None].copy()
    ckf1.P = P
    ckf1.Q = _spd(rng, N_X, 0.02)
    ckf1.R = np.array([[0.1]])
    for i in range(3):
        ckf1.predict()
        ckf1.update(np.array([[3.0 + 0.1 * i]]))
    out["ckf1"] = (ckf1.x.copy(), ckf1.P.copy(), ckf1.K.copy(), ckf1.y.copy())
    out["ckf1_lik"] = ckf1.likelihood
    ckf1.update(np.array([[1e4]]), R=1e-6)
    out["ckf1_far_lik"] = (ckf1.log_likelihood, ckf1.likelihood)

    # user residuals (the bearing component of z is wrapped)
    ckf2 = mod.CubatureKalmanFilter(N_X, N_Z, 0.1, hx_ckf, fx_ckf,
                                    residual_x=np.subtract, residual_z=bearing_residual)
    ckf2.x = x_flat[:, None].copy()
    ckf2.P = P
    ckf2.R = _spd(rng, N_Z, 0.05)
    ckf2.predict()
    z = hx_ckf(ckf2.x.ravel())[:, None] + np.array([[0.1], [2 * np.pi - 0.05], [0.0]])
    ckf2.update(z)
    out["ckf2_residual"] = (ckf2.x.copy(), ckf2.P.copy(), ckf2.y.copy())
    return out


def run(mod: Any) -> dict[str, Any]:
    out = {}
    out.update({"ekf_" + k: v for k, v in run_ekf(mod).items()})
    out.update({"ckf_" + k: v for k, v in run_ckf(mod).items()})
    return out


if __name__ == "__main__":
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location("clean", os.path.join(os.path.dirname(__file__), "clean.py"))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res = run(mod)
    for k, v in res.items():
        print(k, np.shape(v) if not isinstance(v, tuple) else [np.shape(a) for a in v])
