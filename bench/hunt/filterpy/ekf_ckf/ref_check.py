"""Compare clean.py against the installed filterpy 2.0.0.

Prints PASS/FAIL per case and exits non-zero on any failure. Known,
documented deviations of clean.py (see NOTES.md) are checked as such:
- ExtendedKalmanFilter.K starts as (n_x, n_z) zeros (upstream: (n_x, 1));
- CubatureKalmanFilter.x starts as an (n_x, 1) column (upstream: (n_x,));
- CubatureKalmanFilter.K / .y start as zero arrays (upstream: the int 0).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Callable

import numpy as np
from filterpy.kalman import CubatureKalmanFilter as UpCKF
from filterpy.kalman import ExtendedKalmanFilter as UpEKF
from filterpy.kalman.CubatureKalmanFilter import ckf_transform as up_ckf_transform
from filterpy.kalman.CubatureKalmanFilter import spherical_radial_sigmas as up_sigmas
from filterpy.stats import logpdf as up_logpdf

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


clean = _load("clean")
harness = _load("harness")

FAILS: list[str] = []


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=1e-9, atol=1e-11, equal_nan=True))


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


class Upstream:
    """Upstream filterpy behind the clean.py module interface."""

    ExtendedKalmanFilter = UpEKF
    CubatureKalmanFilter = UpCKF
    spherical_radial_sigmas = staticmethod(up_sigmas)
    ckf_transform = staticmethod(up_ckf_transform)


# ------------------------------------------------------------------ cases


def case_harness() -> None:
    a = harness.run(clean)
    b = harness.run(Upstream)
    # deliberate deviation: initial K shape
    check("harness: ekf init K is (n_x, n_z) in clean, (n_x, 1) upstream",
          np.shape(a.pop("ekf_init_K")) == (5, 3) and np.shape(b.pop("ekf_init_K")) == (5, 1))
    # before the first update the stored K is still the initial one
    for k in ("ekf_predict_noctl", "ekf_predict_ctl"):
        a[k].pop("K")
        b[k].pop("K")
    bad = [k for k in a if not _close(a[k], b[k])]
    check("harness: all outputs (5 states, 3/1 meas, 2 inputs)", not bad and a.keys() == b.keys(), str(bad))


def _linear_ekf_run(make: Callable[..., Any], n_x: int, n_z: int, n_u: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    H = rng.normal(size=(n_z, n_x))
    F = np.eye(n_x) + 0.1 * rng.normal(size=(n_x, n_x))
    B = rng.normal(size=(n_x, n_u))
    A = rng.normal(size=(n_x, n_x))
    C = rng.normal(size=(n_z, n_z))

    def hj(x: Any, s: float = 1.0) -> Any:
        return s * H

    def hx(x: Any, s: float = 1.0) -> Any:
        return s * (H @ x) + 0.01 * np.tanh(H @ x)

    f = make(n_x, n_z, n_u)
    f.x = rng.normal(size=(n_x, 1))
    f.P = A @ A.T / n_x + np.eye(n_x)
    f.F = F
    f.Q = 0.01 * np.eye(n_x)
    f.R = C @ C.T / n_z + np.eye(n_z)
    out: dict[str, Any] = {}
    f.predict()
    f.B = B
    for i in range(3):
        u = rng.normal(size=(n_u, 1))
        f.predict(u)
        z = H @ f.x + 0.1 * rng.normal(size=(n_z, 1))
        f.update(z, hj, hx, R=None if i != 1 else 0.5, args=(1.0,), hx_args=1.0)
        out[f"step{i}"] = (f.x.copy(), f.P.copy(), f.K.copy(), f.y.copy(), f.S.copy(),
                           f.x_prior.copy(), f.P_prior.copy(), f.log_likelihood, f.mahalanobis)
    z = H @ (F @ f.x) + 0.1 * rng.normal(size=(n_z, 1))
    f.predict_update(z, hj, hx, args=1.0, hx_args=(1.0,), u=rng.normal(size=(n_u, 1)))
    out["pu"] = (f.x.copy(), f.P.copy(), f.K.copy(), f.y.copy(), f.x_prior.copy(), f.likelihood)
    return out


def case_ekf_sizes() -> None:
    for n_x, n_z, n_u in [(5, 3, 2), (4, 2, 3), (6, 1, 2), (3, 4, 1)]:
        a = _linear_ekf_run(clean.ExtendedKalmanFilter, n_x, n_z, n_u, 7)
        b = _linear_ekf_run(UpEKF, n_x, n_z, n_u, 7)
        check(f"ekf linear-ish model n_x={n_x} n_z={n_z} n_u={n_u}", _close(a, b))


def case_ekf_upstream_quirks() -> None:
    """Upstream behaviour clean.py keeps on purpose (see upstream suspicions)."""
    for mod_name, make in (("clean", clean.ExtendedKalmanFilter), ("upstream", UpEKF)):
        f = make(5, 3, 2)
        f.B = np.ones((5, 2))
        f.predict()  # default u=0 with a matrix B
        check(f"{mod_name}: predict() with matrix B and default u gives x (5, 2)", f.x.shape == (5, 2))
    for mod_name, make in (("clean", clean.ExtendedKalmanFilter), ("upstream", UpEKF)):
        f = make(2, 1)
        f.x = np.array([[1.0], [2.0]])
        f.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        f.predict_update(np.array([[3.0]]), lambda x: np.array([[1.0, 0.0]]), lambda x: x[:1])
        check(f"{mod_name}: predict_update stores the old posterior as x_prior",
              np.allclose(f.x_prior.ravel(), [1.0, 2.0]))


def _ckf_run(make: Callable[..., Any], n_x: int, n_z: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n_z, n_x))
    A = rng.normal(size=(n_x, n_x))

    def fx(x: Any, dt: float, k: float = 0.1) -> Any:
        return x + dt * k * np.sin(x[::-1])

    def hx(x: Any, s: float = 1.0) -> Any:
        return s * (W @ x) + 0.05 * (W @ x) ** 2

    f = make(n_x, n_z, 0.1, hx, fx)
    f.x = rng.normal(size=(n_x, 1))
    f.P = A @ A.T / n_x + np.eye(n_x)
    f.Q = 0.02 * np.eye(n_x)
    f.R = 0.3 * np.eye(n_z)
    out: dict[str, Any] = {}
    for i in range(3):
        f.predict(dt=None if i == 0 else 0.05 * i, fx_args=() if i == 0 else 0.2)
        z = W @ f.x + 0.1 * rng.normal(size=(n_z, 1))
        f.update(z, R=None if i != 2 else 0.4, hx_args=() if i == 0 else (1.0,))
        out[f"step{i}"] = (f.x.copy(), f.P.copy(), f.K.copy(), f.y.copy(), f.S.copy(),
                           f.sigmas_f.copy(), f.sigmas_h.copy(), f.log_likelihood, f.mahalanobis)
    f.update(None)
    out["none"] = (f.x_post.copy(), np.asarray(f.z, dtype=float))
    return out


def case_ckf_sizes() -> None:
    for n_x, n_z in [(5, 3), (4, 1), (3, 5), (6, 2)]:
        a = _ckf_run(clean.CubatureKalmanFilter, n_x, n_z, 11)
        b = _ckf_run(UpCKF, n_x, n_z, 11)
        check(f"ckf model n_x={n_x} n_z={n_z}", _close(a, b))


def case_ckf_init() -> None:
    c = clean.CubatureKalmanFilter(5, 3, 0.1, lambda x: x[:3], lambda x, dt: x)
    u = UpCKF(5, 3, 0.1, lambda x: x[:3], lambda x, dt: x)
    check("ckf init x: clean (5, 1) column, upstream (5,)", c.x.shape == (5, 1) and u.x.shape == (5,))
    c.predict()
    u.predict()
    check("ckf predict from the initial state agrees", _close((c.x, c.P), (u.x, u.P)))


def case_free_functions() -> None:
    rng = np.random.default_rng(5)
    for n in (1, 3, 6):
        A = rng.normal(size=(n, n))
        P = A @ A.T + np.eye(n)
        x = rng.normal(size=n)
        check(f"spherical_radial_sigmas n={n} flat x", _close(clean.spherical_radial_sigmas(x, P), up_sigmas(x, P)))
        check(f"spherical_radial_sigmas n={n} column x",
              _close(clean.spherical_radial_sigmas(x[:, None], P), up_sigmas(x[:, None], P)))
    for m, n in ((10, 5), (8, 3), (4, 7)):
        Xs = rng.normal(size=(m, n))
        Q = np.diag(rng.uniform(0.1, 1.0, size=n))
        check(f"ckf_transform Xs=({m}, {n})", _close(clean.ckf_transform(Xs, Q), up_ckf_transform(Xs, Q)))
        Bm = rng.normal(size=(m, 2))
        ref = sum(np.outer(Xs[i], Bm[i]) for i in range(m))
        check(f"outer_product_sum ({m}, {n}) x ({m}, 2)", _close(clean.outer_product_sum(Xs, Bm), ref))
    for k in (1, 3, 4):
        A = rng.normal(size=(k, k))
        S = A @ A.T + np.eye(k)
        y = rng.normal(size=(k, 1))
        check(f"logpdf k={k}", _close(clean._mvn_logpdf(y, S), up_logpdf(x=y, cov=S)))


def main() -> int:
    case_harness()
    case_ekf_sizes()
    case_ekf_upstream_quirks()
    case_ckf_sizes()
    case_ckf_init()
    case_free_functions()
    print(f"{len(FAILS)} failure(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
