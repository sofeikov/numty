"""Compare clean.py against the installed scipy.signal (1.18.1).

Runs every harness case through clean.py and through an adapter over
scipy.signal / scipy.linalg, then a few extra randomized MIMO cases, and
prints PASS / FAIL per case. Exits non-zero on any failure.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import scipy.linalg
import scipy.signal as signal

HERE = Path(__file__).resolve().parent


def load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


clean = load(HERE / "clean.py", "ltisys_clean")
harness = load(HERE / "harness.py", "ltisys_harness")

upstream = types.SimpleNamespace(
    StateSpace=signal.StateSpace,
    lsim=signal.lsim,
    impulse=signal.impulse,
    step=signal.step,
    dlsim=signal.dlsim,
    place_poles=signal.place_poles,
    expm=scipy.linalg.expm,
)


def same(a: Any, b: Any, rtol: float = 1e-7, atol: float = 1e-9) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if isinstance(a, (list, tuple)):
        return (isinstance(b, (list, tuple)) and len(a) == len(b)
                and all(same(x, y, rtol, atol) for x, y in zip(a, b)))
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.size == 0:
        return True
    return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))


def describe(a: Any) -> str:
    if isinstance(a, (list, tuple)):
        return "[" + ", ".join(describe(x) for x in a) + "]"
    if isinstance(a, str):
        return a
    return str(np.shape(a))


def main() -> int:
    print(f"scipy {scipy.__version__}, numpy {np.__version__}")
    failures = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mine = harness.run(clean)
        ref = harness.run(upstream)
    for key in mine:
        ok = same(mine[key], ref[key])
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} harness:{key}"
              + ("" if ok else f"  clean={describe(mine[key])} upstream={describe(ref[key])}"))

    # Extra randomized MIMO cases: n states, m inputs, p outputs, T samples.
    for seed, (n, m, p, T) in enumerate([(4, 2, 3, 11), (6, 3, 2, 17), (5, 1, 4, 9),
                                         (3, 4, 5, 13)]):
        rng = np.random.default_rng(100 + seed)
        A = rng.standard_normal((n, n)) - 2.0 * np.eye(n)
        B = rng.standard_normal((n, m))
        C = rng.standard_normal((p, n))
        D = rng.standard_normal((p, m))
        t = np.linspace(0.0, 2.0, T)
        U = rng.standard_normal((T, m))
        x0 = rng.standard_normal(n)
        tag = f"n={n},m={m},p={p},T={T}"
        mine_ss = clean.StateSpace(A, B, C, D)
        ref_ss = signal.StateSpace(A, B, C, D)
        for interp in (True, False):
            ok = same(list(clean.lsim(mine_ss, U, t, X0=x0, interp=interp)),
                      list(signal.lsim(ref_ss, U, t, X0=x0, interp=interp)))
            failures += not ok
            print(f"{'PASS' if ok else 'FAIL'} lsim[{tag},interp={interp}]")
        dts = (0.05,)
        for dt in dts:
            md = clean.StateSpace(0.2 * A, B, C, D, dt=dt)
            rd = signal.StateSpace(0.2 * A, B, C, D, dt=dt)
            ok = same(list(clean.dlsim(md, U, t=t, x0=x0)),
                      list(signal.dlsim(rd, U, t=t, x0=x0)))
            failures += not ok
            print(f"{'PASS' if ok else 'FAIL'} dlsim_t[{tag},dt={dt}]")
        if m > 1 and m < n:
            poles = -np.arange(1.0, n + 1.0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for method in ("YT", "KNV0"):
                    a = clean.place_poles(A, B, poles, method=method)
                    b = signal.place_poles(A, B, poles, method=method)
                    ok = same([a.gain_matrix, a.computed_poles, a.X],
                              [b.gain_matrix, b.computed_poles, b.X], rtol=1e-6)
                    failures += not ok
                    print(f"{'PASS' if ok else 'FAIL'} place_poles[{tag},{method}]")
        # series and parallel connections against a 2nd system
        A2 = rng.standard_normal((n + 1, n + 1))
        B2 = rng.standard_normal((n + 1, m))
        C2 = rng.standard_normal((p, n + 1))
        D2 = rng.standard_normal((p, m))
        par_m = clean.StateSpace(A2, B2, C2, D2) + mine_ss
        par_r = signal.StateSpace(A2, B2, C2, D2) + ref_ss
        ok = same([par_m.A, par_m.B, par_m.C, par_m.D], [par_r.A, par_r.B, par_r.C, par_r.D])
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} parallel[{tag}]")
        A3 = rng.standard_normal((2, 2))
        B3 = rng.standard_normal((2, 7))
        C3 = rng.standard_normal((m, 2))
        D3 = rng.standard_normal((m, 7))
        ser_m = mine_ss * clean.StateSpace(A3, B3, C3, D3)
        ser_r = ref_ss * signal.StateSpace(A3, B3, C3, D3)
        ok = same([ser_m.A, ser_m.B, ser_m.C, ser_m.D], [ser_r.A, ser_r.B, ser_r.C, ser_r.D])
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} series[{tag}]")

    print("FAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
