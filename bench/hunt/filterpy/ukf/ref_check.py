"""Compare clean.py against the installed filterpy 2.0.0 (UKF, sigma points, UT).

Prints PASS or FAIL per case; exits 1 on any failure.

Case "harness": the full harness.run output (every branch it reaches) with
clean.py versus with a shim that exposes filterpy's own classes.
Cases "dims-*": filters of other rectangular sizes (n_x, n_z) for each
sigma-point scheme: batch_filter, rts_smoother, likelihoods.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import scipy.linalg
from filterpy.kalman import (JulierSigmaPoints, MerweScaledSigmaPoints, SimplexSigmaPoints,
                             UnscentedKalmanFilter, unscented_transform)

HERE = Path(__file__).resolve().parent


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


clean = load(HERE / "clean.py", "ukf_clean")
harness = load(HERE / "harness.py", "ukf_harness")

upstream = types.SimpleNamespace(
    MerweScaledSigmaPoints=MerweScaledSigmaPoints,
    JulierSigmaPoints=JulierSigmaPoints,
    SimplexSigmaPoints=SimplexSigmaPoints,
    UnscentedKalmanFilter=UnscentedKalmanFilter,
    unscented_transform=unscented_transform,
    cholesky=scipy.linalg.cholesky,
)


def compare(a, b, path="") -> list[str]:
    if isinstance(a, dict):
        if set(a) != set(b):
            return [f"{path}: keys {sorted(set(a) ^ set(b))}"]
        return [m for k in a for m in compare(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, (list, tuple)) and not isinstance(a, np.ndarray):
        if len(a) != len(b):
            return [f"{path}: len {len(a)} vs {len(b)}"]
        return [m for i, (x, y) in enumerate(zip(a, b)) for m in compare(x, y, f"{path}[{i}]")]
    if isinstance(a, str) or isinstance(b, str):
        return [] if a == b else [f"{path}: {a!r} vs {b!r}"]
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return [f"{path}: shape {a.shape} vs {b.shape}"]
    if a.dtype == object or b.dtype == object:
        return [] if a.tolist() == b.tolist() else [f"{path}: object arrays differ"]
    if not np.allclose(a, b, rtol=1e-9, atol=1e-10, equal_nan=True):
        return [f"{path}: max abs diff {np.max(np.abs(a - b)):.3e}"]
    return []


def dims_case(mod, n_x: int, n_z: int, scheme: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    A = np.eye(n_x) + 0.1 * rng.standard_normal((n_x, n_x))
    H = rng.standard_normal((n_z, n_x))

    def fx(x, dt):
        return A @ x + dt * 0.05 * np.sin(x)

    def hx(x):
        return H @ x + 0.1 * np.tanh(H @ x)

    if scheme == "merwe":
        pts = mod.MerweScaledSigmaPoints(n_x, alpha=0.5, beta=2.0, kappa=3.0 - n_x)
    elif scheme == "julier":
        pts = mod.JulierSigmaPoints(n_x, kappa=1.0)
    else:
        pts = mod.SimplexSigmaPoints(n_x)
    f = mod.UnscentedKalmanFilter(n_x, n_z, 0.2, hx, fx, pts)
    f.x = rng.standard_normal(n_x)
    M = rng.standard_normal((n_x, n_x))
    f.P = M @ M.T + n_x * np.eye(n_x)
    f.Q = 0.01 * np.eye(n_x)
    f.R = 0.1 * np.eye(n_z)
    n_steps = 6
    x = f.x.copy()
    zs = []
    for _ in range(n_steps):
        x = fx(x, 0.2)
        zs.append(hx(x) + 0.05 * rng.standard_normal(n_z))
    zs = np.array(zs)
    means, covs = f.batch_filter(zs)
    out = {"means": means, "covs": covs, "K": f.K, "y": f.y, "S": f.S,
           "maha": f.mahalanobis, "ll": f.log_likelihood, "lik": f.likelihood}
    xs, ps, ks = f.rts_smoother(means, covs)
    out.update(xs=xs, ps=ps, ks=ks)
    return out


def main() -> int:
    failures = 0

    def report(name, a, b):
        nonlocal failures
        diffs = compare(a, b)
        if diffs:
            failures += 1
            print(f"FAIL {name}: " + "; ".join(diffs[:5]))
        else:
            print(f"PASS {name}")

    report("harness", harness.run(clean), harness.run(upstream))
    for n_x, n_z in ((5, 3), (3, 1), (6, 2), (4, 7)):
        for scheme in ("merwe", "julier", "simplex"):
            seed = 100 * n_x + 10 * n_z
            report(f"dims-{scheme}-nx{n_x}-nz{n_z}",
                   dims_case(clean, n_x, n_z, scheme, seed),
                   dims_case(upstream, n_x, n_z, scheme, seed))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
