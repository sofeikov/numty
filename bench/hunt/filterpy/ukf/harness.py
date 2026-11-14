"""Deterministic harness for the filterpy UKF extract (clean.py and its mutants).

Sizes never coincide: n_x = 4 states, n_z = 2 measurements, 9 Merwe/Julier
sigma points (2*n_x+1), 5 Simplex sigma points (n_x+1), 7 time steps.
A second filter has n_z = 1 to reach the dim_z == 1 branch of batch_filter.
"""

from __future__ import annotations

import numpy as np

DIM_X = 4
DIM_Z = 2
N_STEPS = 7
DT = 0.5


def _F(dt):
    return np.array([[1.0, dt, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, dt],
                     [0.0, 0.0, 0.0, 1.0]])


def fx(x, dt):
    """Constant velocity in 2-D, state (px, vx, py, vy), with a slight drag on vy."""
    out = _F(dt) @ x
    out[3] = out[3] - 0.01 * dt * x[3] ** 2
    return out


def fx_alt(x, dt):
    return _F(dt) @ x


def hx(x):
    """Range and bearing from the origin."""
    return np.array([np.hypot(x[0], x[2]), np.arctan2(x[2], x[0])])


def hx1(x):
    """Range only (n_z = 1)."""
    return np.array([np.hypot(x[0], x[2])])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def residual_z(a, b):
    y = a - b
    y[1] = _wrap(y[1])
    return y


def z_mean(sigmas, Wm):
    z = np.zeros(sigmas.shape[1])
    z[0] = np.dot(Wm, sigmas[:, 0])
    z[1] = np.arctan2(np.dot(Wm, np.sin(sigmas[:, 1])), np.dot(Wm, np.cos(sigmas[:, 1])))
    return z


def x_mean(sigmas, Wm):
    return np.dot(Wm, sigmas)


def residual_x(a, b):
    return a - b


def state_add(a, b):
    return a + b


def sqrt_sym(P):
    """A symmetric square root (eigen decomposition), returned as rows like cholesky's U."""
    w, V = np.linalg.eigh(P)
    return (V * np.sqrt(np.maximum(w, 0.0))) @ V.T


def _truth_and_measurements(seed, n_z):
    rng = np.random.default_rng(seed)
    x = np.array([10.0, 1.0, 5.0, -0.5])
    xs, zs = [], []
    for _ in range(N_STEPS):
        x = fx_alt(x, DT) + rng.normal(0.0, 0.05, DIM_X)
        xs.append(x.copy())
        z = hx(x) if n_z == 2 else hx1(x)
        z = z + rng.normal(0.0, 0.02, n_z)
        zs.append(z)
    return np.array(xs), np.array(zs)


def _setup(f, n_z):
    f.x = np.array([9.5, 0.8, 5.2, -0.4])
    f.P = np.diag([0.5, 0.2, 0.5, 0.2])
    f.Q = np.diag([0.01, 0.02, 0.01, 0.02])
    f.R = np.diag([0.04, 0.001])[:n_z, :n_z]
    return f


def _filter_run(mod, points, *, custom=False, n_z=DIM_Z, seed=0):
    kw = {}
    if custom:
        kw = dict(sqrt_fn=sqrt_sym, x_mean_fn=x_mean, z_mean_fn=z_mean,
                  residual_x=residual_x, residual_z=residual_z, state_add=state_add)
    f = mod.UnscentedKalmanFilter(DIM_X, n_z, DT, hx if n_z == 2 else hx1, fx, points, **kw)
    _setup(f, n_z)
    _, zs = _truth_and_measurements(seed, n_z)
    means, covs = f.batch_filter(zs)
    out = {
        "means": means, "covs": covs,
        "K": f.K, "S": f.S, "SI": f.SI, "y": f.y, "sigmas_f": f.sigmas_f,
        "sigmas_h": f.sigmas_h, "x_prior": f.x_prior, "P_prior": f.P_prior,
        "mahalanobis": f.mahalanobis, "log_likelihood": f.log_likelihood,
        "likelihood": f.likelihood,
    }
    xs_s, ps_s, ks_s = f.rts_smoother(means, covs)
    out.update(rts_x=xs_s, rts_P=ps_s, rts_K=ks_s)
    xs2, ps2, ks2 = f.rts_smoother(means, covs, Qs=[f.Q * 3.0] * N_STEPS, dts=DT, UT=mod.unscented_transform)
    out.update(rts2_x=xs2, rts2_P=ps2, rts2_K=ks2)
    xs3, _, _ = f.rts_smoother(means, covs, dts=[DT] * N_STEPS)
    out.update(rts3_x=xs3)
    return out


def run(mod) -> dict:
    out: dict = {}

    # --- sigma point schemes on their own (rectangular: 9x4 and 5x4) ---
    x0 = np.array([1.0, -2.0, 0.5, 3.0])
    P0 = np.array([[2.0, 0.3, 0.1, 0.0],
                   [0.3, 1.0, 0.2, 0.1],
                   [0.1, 0.2, 1.5, 0.4],
                   [0.0, 0.1, 0.4, 0.8]])
    merwe = mod.MerweScaledSigmaPoints(DIM_X, alpha=0.3, beta=2.0, kappa=-1.0)
    julier = mod.JulierSigmaPoints(DIM_X, kappa=0.5)
    simplex = mod.SimplexSigmaPoints(DIM_X)
    for name, pts in (("merwe", merwe), ("julier", julier), ("simplex", simplex)):
        out[f"{name}_n"] = pts.num_sigmas()
        out[f"{name}_Wm"] = pts.Wm
        out[f"{name}_Wc"] = pts.Wc
        s = pts.sigma_points(x0, P0)
        out[f"{name}_sigmas"] = s
        out[f"{name}_scalarP"] = pts.sigma_points(x0, 0.7)
        mu, cov = mod.unscented_transform(s, pts.Wm, pts.Wc)
        out[f"{name}_ut"] = [mu, cov]
        mu2, cov2 = mod.unscented_transform(s, pts.Wm, pts.Wc, noise_cov=np.eye(DIM_X) * 0.1,
                                            mean_fn=x_mean, residual_fn=residual_x)
        out[f"{name}_ut_custom"] = [mu2, cov2]
        try:
            pts.sigma_points(x0[:3], P0[:3, :3])
            out[f"{name}_size_err"] = "no error"
        except ValueError as e:
            out[f"{name}_size_err"] = str(e)

    # scalar x and P with a 1-state scheme
    for name, pts in (("merwe1", mod.MerweScaledSigmaPoints(1, 0.5, 2.0, 0.0)),
                      ("julier1", mod.JulierSigmaPoints(1, kappa=2.0)),
                      ("simplex1", mod.SimplexSigmaPoints(1))):
        out[f"{name}_sigmas"] = pts.sigma_points(1.5, 0.4)

    # custom sqrt and subtract
    merwe_c = mod.MerweScaledSigmaPoints(DIM_X, 0.5, 2.0, 0.0, sqrt_method=sqrt_sym, subtract=residual_x)
    julier_c = mod.JulierSigmaPoints(DIM_X, 1.0, sqrt_method=sqrt_sym, subtract=residual_x)
    simplex_c = mod.SimplexSigmaPoints(DIM_X, sqrt_method=sqrt_sym, subtract=residual_x)
    out["merwe_custom"] = merwe_c.sigma_points(x0, P0)
    out["julier_custom"] = julier_c.sigma_points(x0, P0)
    out["simplex_custom"] = simplex_c.sigma_points(x0, P0)

    # --- full filter runs, rectangular: n_x = 4, n_z = 2 ---
    out["ukf_merwe"] = _filter_run(mod, mod.MerweScaledSigmaPoints(DIM_X, 0.3, 2.0, -1.0))
    out["ukf_julier"] = _filter_run(mod, mod.JulierSigmaPoints(DIM_X, kappa=0.5), seed=1)
    out["ukf_simplex"] = _filter_run(mod, mod.SimplexSigmaPoints(DIM_X), seed=2)
    out["ukf_custom"] = _filter_run(mod, mod.MerweScaledSigmaPoints(DIM_X, 0.3, 2.0, -1.0), custom=True, seed=3)
    out["ukf_z1"] = _filter_run(mod, mod.MerweScaledSigmaPoints(DIM_X, 0.3, 2.0, -1.0), n_z=1, seed=4)

    # --- step by step: predict/update branches ---
    f = _setup(mod.UnscentedKalmanFilter(DIM_X, DIM_Z, DT, hx, fx, mod.MerweScaledSigmaPoints(DIM_X, 0.3, 2.0, -1.0)), DIM_Z)
    _, zs = _truth_and_measurements(5, DIM_Z)
    f.predict()
    f.update(zs[0])
    out["step_x1"], out["step_P1"] = f.x.copy(), f.P.copy()
    f.predict(dt=0.25, fx=fx_alt, UT=mod.unscented_transform)
    f.update(zs[1], R=0.05, UT=mod.unscented_transform)  # scalar R
    out["step_x2"], out["step_P2"], out["step_S2"] = f.x.copy(), f.P.copy(), f.S.copy()
    f.predict()
    f.update(zs[2], R=np.diag([0.1, 0.002]), hx=hx)  # matrix R
    out["step_x3"], out["step_P3"], out["step_K3"] = f.x.copy(), f.P.copy(), f.K.copy()
    f.predict()
    f.update(None)  # no measurement
    out["step_x4"], out["step_post4"] = f.x_post.copy(), f.P_post.copy()
    out["step_z4_shape"] = list(np.shape(f.z))
    out["cross"] = f.cross_variance(f.x, hx(f.x), f.sigmas_f, np.array([hx(s) for s in f.sigmas_f]))
    f.compute_process_sigmas(DT)
    out["proc_sigmas"] = f.sigmas_f.copy()
    f.compute_process_sigmas(DT, fx=fx_alt)
    out["proc_sigmas_alt"] = f.sigmas_f.copy()

    # --- batch_filter validation branches ---
    g = _setup(mod.UnscentedKalmanFilter(DIM_X, DIM_Z, DT, hx, fx, mod.JulierSigmaPoints(DIM_X)), DIM_Z)
    try:
        g.batch_filter(zs[:, :1])
        out["batch_err_len"] = "no error"
    except TypeError as e:
        out["batch_err_len"] = str(e)
    try:
        g.batch_filter(3)
        out["batch_err_type"] = "no error"
    except TypeError as e:
        out["batch_err_type"] = str(e)
    g1 = _setup(mod.UnscentedKalmanFilter(DIM_X, 1, DT, hx1, fx, mod.JulierSigmaPoints(DIM_X)), 1)
    try:
        g1.batch_filter(zs)
        out["batch_err_z1"] = "no error"
    except TypeError as e:
        out["batch_err_z1"] = str(e)
    Rs = [np.diag([0.04 * (i + 1), 0.001]) for i in range(N_STEPS)]
    dts = [DT * (1 + 0.1 * i) for i in range(N_STEPS)]
    m2, c2 = g.batch_filter(zs, Rs=Rs, dts=dts, UT=mod.unscented_transform)
    out["batch_Rs_means"], out["batch_Rs_covs"] = m2, c2

    # --- rts_smoother validation ---
    try:
        g.rts_smoother(m2, c2[:-1])
        out["rts_err"] = "no error"
    except ValueError as e:
        out["rts_err"] = str(e)

    # --- logpdf / cholesky helpers through the filter ---
    out["chol"] = mod.cholesky(P0)
    return out
