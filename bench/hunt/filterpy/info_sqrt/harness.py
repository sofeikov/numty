"""Harness for the information / square-root filter extract.

Sizes never coincide: n_x = 5 states, n_z = 3 measurements, n_u = 2 controls,
7 time steps.  Every kept branch is exercised:

InformationFilter: constructor validation, F setter, predict (invertible and
non-invertible A, the no-information -> information switch, control input),
update (z None, R_inv None / scalar / matrix, informed and no-information
paths, log-likelihood on and off), P, batch_filter (raises).
SquareRootKalmanFilter: constructor validation, Q/P/R setters, predict with
and without control, update (z None, R2 None / scalar / matrix), residual_of,
measurement_of_state and every property.
"""

from __future__ import annotations

from typing import Any

import numpy as np

NX, NZ, NU, STEPS = 5, 3, 2, 7


def _system(rng: np.random.Generator) -> dict[str, np.ndarray]:
    F = np.eye(NX) + 0.1 * np.diag(np.ones(NX - 1), 1) + 0.02 * rng.normal(size=(NX, NX))
    H = rng.normal(size=(NZ, NX))
    B = rng.normal(size=(NX, NU))
    A = rng.normal(size=(NX, NX))
    Q = 0.05 * (A @ A.T) + 0.1 * np.eye(NX)
    C = rng.normal(size=(NZ, NZ))
    R = 0.3 * (C @ C.T) + 0.5 * np.eye(NZ)
    zs = rng.normal(size=(STEPS, NZ, 1))
    us = rng.normal(size=(STEPS, NU, 1))
    return dict(F=F, H=H, B=B, Q=Q, R=R, zs=zs, us=us)


def _errors(make: Any) -> list[str]:
    out = []
    for args in ((0, NZ, NU), (NX, 0, NU), (NX, NZ, -1)):
        try:
            make(*args)
            out.append("ok")
        except ValueError as e:
            out.append(str(e))
    return out


def run_info(mod: Any) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    s = _system(rng)
    res: dict[str, Any] = {"info_errors": _errors(mod.InformationFilter)}

    # informed filter, log-likelihood on, every R_inv form, control input
    f = mod.InformationFilter(NX, NZ, NU)
    f.F = s["F"]
    f.H = s["H"]
    f.Q = s["Q"]
    f.B = s["B"]
    f.R_inv = np.linalg.inv(s["R"])
    xs, lls, Ks, Ss, ys = [], [], [], [], []
    for k in range(STEPS):
        # B is a matrix here, so u is always given: upstream predict() with
        # the default u=0 broadcasts x to (n_x, n_u) (see NOTES.md)
        f.predict(s["us"][k])
        if k == 3:
            f.update(None)
        elif k == 4:
            f.update(s["zs"][k], 2.5)
        elif k == 5:
            f.update(s["zs"][k], np.linalg.inv(2.0 * s["R"]))
        else:
            f.update(s["zs"][k])
        xs.append(f.x.copy())
        lls.append(f.log_likelihood)
        Ks.append(np.copy(f.K))
        Ss.append(np.copy(f.S))
        ys.append(f.y.copy())
    res.update(
        info_x=np.array(xs), info_ll=np.array(lls), info_K=np.array(Ks),
        info_S=np.array(Ss), info_y=np.array(ys), info_P=f.P, info_P_inv=f.P_inv,
        info_x_prior=f.x_prior, info_P_inv_prior=f.P_inv_prior,
        info_x_post=f.x_post, info_P_inv_post=f.P_inv_post, info_z=f.z,
        info_likelihood=f.likelihood, info_F=f.F,
    )

    # log-likelihood off
    g = mod.InformationFilter(NX, NZ, NU, compute_log_likelihood=False)
    g.F, g.H, g.Q = s["F"], s["H"], s["Q"]
    g.predict()
    g.update(s["zs"][0])
    res.update(info_noll_x=g.x, info_noll_ll=g.log_likelihood)

    # no-information start: diagonal F keeps P_inv exactly singular, so the
    # non-invertible predict branch and the information update both run
    # until two partial measurement models make P_inv full rank
    h = mod.InformationFilter(NX, NZ, NU)
    h.F = np.diag([1.0, 1.1, 0.9, 1.2, 0.8])
    h.Q = 0.1 * np.eye(NX)
    h.P_inv = np.zeros((NX, NX))
    h.R_inv = np.linalg.inv(s["R"])
    H1 = np.zeros((NZ, NX))
    H1[0, 0] = H1[1, 1] = H1[2, 2] = 1.0
    H2 = np.zeros((NZ, NX))
    H2[0, 3] = H2[1, 4] = H2[2, 0] = 1.0
    hx, hP, flags = [], [], []
    for k in range(4):
        h.H = H1 if k % 2 == 0 else H2
        h.predict()
        flags.append(h._no_information)
        h.update(s["zs"][k])
        hx.append(h.x.copy())
        hP.append(h.P_inv.copy())
    res.update(noinfo_x=np.array(hx), noinfo_P_inv=np.array(hP),
               noinfo_flags=np.array(flags), noinfo_prior=h.P_inv_prior)

    try:
        f.batch_filter([z for z in s["zs"]])
        res["batch"] = "ran"
    except NotImplementedError as e:
        res["batch"] = str(e)
    return res


def run_sqrt(mod: Any) -> dict[str, Any]:
    rng = np.random.default_rng(1)
    s = _system(rng)
    res: dict[str, Any] = {"sqrt_errors": _errors(mod.SquareRootKalmanFilter)}

    f = mod.SquareRootKalmanFilter(NX, NZ, NU)
    f.F = s["F"]
    f.H = s["H"]
    f.Q = s["Q"]
    f.R = s["R"]
    f.P = 2.0 * np.eye(NX) + 0.1 * np.ones((NX, NX))
    f.B = s["B"]
    xs, Ps, Ks, S12, ys = [], [], [], [], []
    for k in range(STEPS):
        f.predict(s["us"][k])
        if k == 2:
            f.update(None)
        elif k == 3:
            f.update(s["zs"][k], 0.7)
        elif k == 4:
            f.update(s["zs"][k], np.linalg.cholesky(1.5 * s["R"]))
        else:
            f.update(s["zs"][k])
        xs.append(f.x.copy())
        Ps.append(f.P.copy())
        Ks.append(f.K.copy())
        S12.append(f.S1_2.copy())
        ys.append(f.y.copy())
    res.update(
        sqrt_x=np.array(xs), sqrt_P=np.array(Ps), sqrt_K=np.array(Ks),
        sqrt_S1_2=np.array(S12), sqrt_y=np.array(ys),
        sqrt_Q=f.Q, sqrt_Q1_2=f.Q1_2, sqrt_P1_2=f.P1_2, sqrt_P_prior=f.P_prior,
        sqrt_P_post=f.P_post, sqrt_R=f.R, sqrt_R1_2=f.R1_2, sqrt_S=f.S,
        sqrt_SI=f.SI, sqrt_M=f.M, sqrt_x_prior=f.x_prior, sqrt_x_post=f.x_post,
        sqrt_z=f.z,
        sqrt_resid=f.residual_of(s["zs"][0]),
        sqrt_meas=f.measurement_of_state(s["zs"][0][[0, 1, 2, 0, 1]]),
    )
    # z None right after construction keeps the object column of Nones
    g = mod.SquareRootKalmanFilter(NX, NZ, NU)
    g.update(None)
    res["sqrt_none_z"] = [repr(v) for v in g.z.ravel()]
    # scalar B = 0. and the default u = 0
    g.F, g.H, g.Q, g.R = s["F"], s["H"], s["Q"], s["R"]
    g.predict()
    g.update(s["zs"][0])
    res.update(sqrt_default_u_x=g.x, sqrt_default_u_P=g.P)
    return res


def run(mod: Any) -> dict[str, Any]:
    out = run_info(mod)
    out.update(run_sqrt(mod))
    return out


if __name__ == "__main__":
    import importlib.util
    import pathlib
    import time

    here = pathlib.Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("clean", here / "clean.py")
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    t = time.time()
    r = run(m)
    for k, v in r.items():
        print(k, np.shape(v) if isinstance(v, np.ndarray) else v)
    print("seconds", round(time.time() - t, 3))
