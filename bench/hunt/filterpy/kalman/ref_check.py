"""Compare clean.py against the installed filterpy (2.0.0) on several size sets.

Run with the hunt venv's python. Prints PASS/FAIL per case, exits non-zero on
any failure. The upstream adapter maps the two helpers whose signatures the
extract narrowed: ``reshape_z(z, dim_z)`` is upstream ``reshape_z(z, dim_z, 2)``
(column state) and ``logpdf`` is ``filterpy.stats.logpdf`` (scipy).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent


def load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def upstream() -> Any:
    from filterpy.common import reshape_z
    from filterpy.kalman import kalman_filter as kfmod
    from filterpy.stats import logpdf

    return SimpleNamespace(
        KalmanFilter=kfmod.KalmanFilter,
        update=kfmod.update,
        update_steadystate=kfmod.update_steadystate,
        predict=kfmod.predict,
        batch_filter=kfmod.batch_filter,
        rts_smoother=kfmod.rts_smoother,
        reshape_z=lambda z, dim_z: reshape_z(z, dim_z, 2),
        logpdf=lambda x, mean, cov: logpdf(x, mean, cov),
    )


def same(a: Any, b: Any, path: str, bad: list[str]) -> None:
    if isinstance(a, dict):
        if set(a) != set(b):
            bad.append(f"{path}: keys {sorted(set(a) ^ set(b))}")
            return
        for k in a:
            same(a[k], b[k], f"{path}.{k}", bad)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            bad.append(f"{path}: len {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]", bad)
    elif isinstance(a, str):
        if a != b:
            bad.append(f"{path}: {a!r} vs {b!r}")
    else:
        xa, xb = np.asarray(a), np.asarray(b)
        if xa.shape != xb.shape:
            bad.append(f"{path}: shape {xa.shape} vs {xb.shape}")
        elif xa.dtype == object or xb.dtype == object:
            if not all(p is q or p == q for p, q in zip(xa.ravel(), xb.ravel())):
                bad.append(f"{path}: object values differ")
        elif not np.allclose(xa, xb, rtol=1e-9, atol=1e-10, equal_nan=True):
            bad.append(f"{path}: max diff {np.nanmax(np.abs(xa - xb))}")


def main() -> int:
    harness = load(HERE / "harness.py", "_harness")
    clean = load(HERE / "clean.py", "_clean")
    up = upstream()
    failed = 0

    cases = [
        ("5x3x2 (harness sizes)", dict(n_x=5, n_z=3, n_u=2, n_steps=7, seed=20261113)),
        ("4x1x1 single output", dict(n_x=4, n_z=1, n_u=1, n_steps=6, seed=1)),
        ("6x4x3 multi output", dict(n_x=6, n_z=4, n_u=3, n_steps=9, seed=2)),
        ("3x5x0 more outputs than states, no control", dict(n_x=3, n_z=5, n_u=0, n_steps=8, seed=3)),
        ("2x3x4 more controls than states", dict(n_x=2, n_z=3, n_u=4, n_steps=5, seed=4)),
    ]
    for name, kw in cases:
        bad: list[str] = []
        try:
            a = harness.scenario(clean, **kw)
            b = harness.scenario(up, **kw)
            same(a, b, "out", bad)
        except Exception as e:  # noqa: BLE001
            bad.append(f"raised {type(e).__name__}: {e}")
        print(("PASS " if not bad else "FAIL ") + name)
        for line in bad[:10]:
            print("    " + line)
        failed += bool(bad)

    # Extra: 1-D z through the H=None path of KalmanFilter.update (reshape_z).
    bad = []
    rng = np.random.default_rng(7)
    outs = []
    for mod in (clean, up):
        r = np.random.default_rng(7)
        kf = mod.KalmanFilter(5, 3)
        kf.H = r.standard_normal((3, 5))
        kf.F = np.eye(5) + 0.1 * r.standard_normal((5, 5))
        for _ in range(4):
            kf.predict()
            kf.update(r.standard_normal(3))
        outs.append([kf.x, kf.P, kf.log_likelihood])
    same(outs[0], outs[1], "flat_z", bad)
    print(("PASS " if not bad else "FAIL ") + "1-D z through reshape_z (H=None)")
    for line in bad:
        print("    " + line)
    failed += bool(bad)
    del rng

    print("ALL PASS" if not failed else f"{failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
