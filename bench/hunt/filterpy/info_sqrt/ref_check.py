"""Compare clean.py against the installed filterpy 2.0.0.

Prints PASS/FAIL per case and exits non-zero on any failure.  The one
intended difference is InformationFilter's log-likelihood (clean.py uses the
residual covariance H P H' + R; upstream passes the (n_x, n_x) information
matrix and raises for dim_z != dim_x): those cases check that upstream raises
or, for dim_z == dim_x, that clean's ``logpdf`` reproduces upstream's number
on upstream's own inputs.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any, Callable

import numpy as np
from filterpy.kalman import InformationFilter, SquareRootKalmanFilter
from scipy.stats import multivariate_normal

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("clean", HERE / "clean.py")
assert spec and spec.loader
clean = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean)

FAILED: list[str] = []


def close(a: Any, b: Any) -> bool:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and np.allclose(a, b, rtol=1e-8, atol=1e-10)


def case(name: str, fn: Callable[[], list[str]]) -> None:
    try:
        bad = fn()
    except Exception as e:  # noqa: BLE001
        bad = [f"raised {type(e).__name__}: {e}"]
    print(("PASS " if not bad else "FAIL ") + name + ("" if not bad else ": " + "; ".join(bad)))
    if bad:
        FAILED.append(name)


def system(nx: int, nz: int, nu: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(nx, nx))
    C = rng.normal(size=(nz, nz))
    return dict(
        F=np.eye(nx) + 0.05 * rng.normal(size=(nx, nx)),
        H=rng.normal(size=(nz, nx)),
        B=rng.normal(size=(nx, nu)),
        Q=0.05 * A @ A.T + 0.1 * np.eye(nx),
        R=0.3 * C @ C.T + 0.5 * np.eye(nz),
        zs=rng.normal(size=(8, nz, 1)),
        us=rng.normal(size=(8, nu, 1)),
    )


def compare(up: Any, cl: Any, attrs: list[str]) -> list[str]:
    return [a for a in attrs if not close(getattr(up, a), getattr(cl, a))]


INFO_ATTRS = ["x", "P_inv", "K", "S", "y", "x_prior", "P_inv_prior", "x_post", "P_inv_post", "P"]
SQRT_ATTRS = ["x", "P", "K", "S1_2", "SI1_2", "y", "M", "P_prior", "P_post", "P1_2",
              "Q", "Q1_2", "R", "R1_2", "S", "SI", "x_prior", "x_post"]


def info_run(nx: int, nz: int, nu: int, seed: int, ll: bool) -> Callable[[], list[str]]:
    def go() -> list[str]:
        s = system(nx, nz, nu, seed)
        up = InformationFilter(nx, nz, nu, compute_log_likelihood=ll)
        cl = clean.InformationFilter(nx, nz, nu, compute_log_likelihood=ll)
        bad: list[str] = []
        for f in (up, cl):
            f.F, f.H, f.Q, f.B = s["F"], s["H"], s["Q"], s["B"]
            f.R_inv = np.linalg.inv(s["R"])
        for k in range(8):
            for f in (up, cl):
                f.predict(s["us"][k])
            r: Any = [None, 1.7, np.linalg.inv(2 * s["R"])][k % 3]
            z = None if k == 5 else s["zs"][k]
            up_err = None
            try:
                up.update(z, r)
            except ValueError as e:
                up_err = e
            P_prior = np.linalg.inv(cl.P_inv)
            cl.update(z, r)
            if up_err is not None:
                if not ll or nz == nx:
                    return [f"step {k}: upstream raised {up_err}"]
                # upstream bug: logpdf(y (n_z,), cov (n_x, n_x)) raises; check
                # clean's likelihood independently, then stop comparing
                Rk = np.linalg.inv(cl.R_inv if r is None else (np.eye(nz) * r if np.isscalar(r) else r))
                want = multivariate_normal.logpdf(cl.y.ravel(), cov=s["H"] @ P_prior @ s["H"].T + Rk,
                                                  allow_singular=True)
                if not close(want, cl.log_likelihood):
                    bad.append(f"step {k}: loglik {cl.log_likelihood} vs scipy {want}")
                return bad + ["(upstream raised as expected)"] if bad else []
            bad += [f"step {k}: {a}" for a in compare(up, cl, INFO_ATTRS)]
            if ll and z is not None:
                # same number as upstream when fed upstream's own (y, S)
                if not close(clean.logpdf(up.y, up.S), up.log_likelihood):
                    bad.append(f"step {k}: logpdf wrapper")
            if not ll and up.log_likelihood != cl.log_likelihood:
                bad.append(f"step {k}: log_likelihood default")
        return bad
    return go


def info_noinfo() -> list[str]:
    nx, nz = 5, 3
    s = system(nx, nz, 2, 11)
    up = InformationFilter(nx, nz, compute_log_likelihood=False)
    cl = clean.InformationFilter(nx, nz, compute_log_likelihood=False)
    H1 = np.zeros((nz, nx)); H1[0, 0] = H1[1, 1] = H1[2, 2] = 1.0
    H2 = np.zeros((nz, nx)); H2[0, 3] = H2[1, 4] = H2[2, 1] = 1.0
    bad = []
    for f in (up, cl):
        f.F = np.diag([1.0, 1.1, 0.9, 1.2, 0.8])
        f.Q = 0.1 * np.eye(nx)
        f.P_inv = np.zeros((nx, nx))
    for k in range(5):
        for f in (up, cl):
            f.H = H1 if k % 2 == 0 else H2
            f.predict()
            f.update(s["zs"][k])
        if up._no_information != cl._no_information:
            bad.append(f"step {k}: flag")
        bad += [f"step {k}: {a}" for a in compare(up, cl, ["x", "P_inv", "x_prior", "P_inv_prior"])]
    return bad


def sqrt_run(nx: int, nz: int, nu: int, seed: int) -> list[str]:
    s = system(nx, nz, nu, seed)
    up = SquareRootKalmanFilter(nx, nz, nu)
    cl = clean.SquareRootKalmanFilter(nx, nz, nu)
    bad = []
    for f in (up, cl):
        f.F, f.H, f.B = s["F"], s["H"], s["B"]
        f.Q, f.R = s["Q"], s["R"]
        f.P = 3.0 * np.eye(nx) + 0.2 * np.ones((nx, nx))
    for k in range(8):
        r: Any = [None, 0.8, np.linalg.cholesky(1.3 * s["R"])][k % 3]
        z = None if k == 4 else s["zs"][k]
        for f in (up, cl):
            f.predict(s["us"][k])
            f.update(z, r)
        bad += [f"step {k}: {a}" for a in compare(up, cl, SQRT_ATTRS)]
        if not close(up.residual_of(s["zs"][0]), cl.residual_of(s["zs"][0])):
            bad.append("residual_of")
        if not close(up.measurement_of_state(up.x), cl.measurement_of_state(cl.x)):
            bad.append("measurement_of_state")
    return bad


def sqrt_default_u() -> list[str]:
    s = system(4, 2, 1, 5)
    up, cl = SquareRootKalmanFilter(4, 2), clean.SquareRootKalmanFilter(4, 2)
    for f in (up, cl):
        f.F, f.H, f.Q, f.R = s["F"], s["H"], s["Q"], s["R"]
        f.predict()
        f.update(s["zs"][0])
        f.update(None)
    bad = compare(up, cl, SQRT_ATTRS)
    if [repr(v) for v in up.z.ravel()] != [repr(v) for v in cl.z.ravel()]:
        bad.append("z")
    return bad


def validation() -> list[str]:
    bad = []
    for up_cls, cl_cls in ((InformationFilter, clean.InformationFilter),
                           (SquareRootKalmanFilter, clean.SquareRootKalmanFilter)):
        for args in ((0, 2, 1), (3, 0, 1), (3, 2, -1)):
            got = []
            for cls in (up_cls, cl_cls):
                try:
                    cls(*args)
                    got.append("ok")
                except ValueError as e:
                    got.append(str(e))
            if got[0] != got[1]:
                bad.append(f"{up_cls.__name__}{args}: {got}")
    return bad


def harness_runs() -> list[str]:
    hs = importlib.util.spec_from_file_location("harness", HERE / "harness.py")
    assert hs and hs.loader
    h = importlib.util.module_from_spec(hs)
    hs.loader.exec_module(h)
    h.run(clean)
    return []


case("info rect 5x3x2, loglik off", info_run(5, 3, 2, 0, False))
case("info rect 5x3x2, loglik on (upstream raises; clean vs scipy)", info_run(5, 3, 2, 1, True))
case("info tall 2x4x1, loglik on (upstream raises; clean vs scipy)", info_run(2, 4, 1, 2, True))
case("info square 4x4x2, loglik on (logpdf wrapper vs upstream)", info_run(4, 4, 2, 3, True))
case("info dim_z=1 3x1x1, loglik off", info_run(3, 1, 1, 4, False))
case("info no-information start", info_noinfo)
case("sqrt rect 5x3x2", lambda: sqrt_run(5, 3, 2, 6))
case("sqrt tall 2x4x3", lambda: sqrt_run(2, 4, 3, 7))
case("sqrt dim_z=1 4x1x2", lambda: sqrt_run(4, 1, 2, 8))
case("sqrt default u, z None", sqrt_default_u)
case("constructor validation", validation)
case("harness runs", harness_runs)

sys.exit(1 if FAILED else 0)
