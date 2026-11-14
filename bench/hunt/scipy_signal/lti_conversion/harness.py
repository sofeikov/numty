"""Harness for the LTI conversion extract: ``run(mod) -> dict``.

Sizes never coincide: 4 states, 2 inputs, 3 outputs (n+m = 6, n+2m = 8,
n+1 = 5 coefficients); transfer functions have 3 outputs over a degree-4
denominator; zpk systems have 2 zeros and 3 poles.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np


def _try(f: Callable[[], Any]) -> Any:
    try:
        return f()
    except (ValueError, IndexError, np.linalg.LinAlgError) as e:
        return f"raises {type(e).__name__}"


def run(mod: Any) -> dict[str, object]:
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(20240611)
    n, m, p = 4, 2, 3
    A = rng.standard_normal((n, n)) - 2.0 * np.eye(n)
    B = rng.standard_normal((n, m))
    C = rng.standard_normal((p, n))
    D = rng.standard_normal((p, m))
    Dz = np.zeros((p, m))
    out: dict[str, object] = {}

    # helpers
    out["poly"] = mod._poly(np.array([1.0, -2.0, 0.5]))
    out["poly_empty"] = mod._poly(np.zeros(0))
    out["trim"] = mod._trim_zeros_f(np.array([0.0, 0.0, 1.0, 0.0, 2.0]))
    out["block_diag"] = mod._block_diag2(B, C)
    out["expm_small"] = mod._expm(0.1 * A)
    out["expm_big"] = mod._expm(9.0 * np.hstack((A, B))[:, :n] + rng.standard_normal((n, n)))

    # normalize / tf2zpk / zpk2tf
    num3 = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 4.0, 1.0], [0.0, 2.0, 0.0, 5.0]])
    den5 = np.array([0.0, 2.0, 1.0, 3.0, 4.0, 1.5])
    out["normalize_1d"] = mod.normalize(np.array([0.0, 3.0, 6.0]), np.array([2.0, -5.0, 4.0]))
    out["normalize_2d"] = mod.normalize(num3, den5)
    out["normalize_row"] = mod.normalize(num3[1:2], den5)
    out["normalize_zero_den"] = _try(lambda: mod.normalize(np.array([1.0]), np.zeros(3)))
    z, pz, k = mod.tf2zpk(np.array([3.0, 0.0, 1.0]), np.array([1.0, 5.0, 13.0, 2.0]))
    out["tf2zpk"] = (np.sort_complex(z), np.sort_complex(pz), k)
    zr, pr = np.array([2.0, -6.0]), np.array([-1.0, -8.0, -3.0])
    zr2 = np.array([[1.0, -2.0], [3.0, 4.0], [-1.0, 0.5]])
    out["zpk2tf_1d"] = mod.zpk2tf(zr, pr, 5.0)
    out["zpk2tf_2d"] = mod.zpk2tf(zr2, pr, 2.0)

    # tf2ss: SISO, SIMO, short numerator, static gain, null, improper
    out["tf2ss_siso"] = mod.tf2ss(np.array([1.0, 3.0, 3.0]), np.array([1.0, 2.0, 1.0]))
    out["tf2ss_simo"] = mod.tf2ss(num3, den5)
    out["tf2ss_short"] = mod.tf2ss(np.array([2.0, 1.0]), np.array([1.0, 0.5, 3.0, 1.0, 0.2]))
    out["tf2ss_gain"] = mod.tf2ss(np.array([[2.0], [3.0], [-1.0]]), np.array([4.0]))
    out["tf2ss_null"] = mod.tf2ss(np.zeros(0), np.array([1.0]))
    out["tf2ss_improper"] = _try(lambda: mod.tf2ss(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])))

    # abcd_normalize: full, each missing matrix, size-0 A, bad shape, undefined dims
    out["abcd_full"] = mod.abcd_normalize(A, B, C, D)
    out["abcd_noC"] = mod.abcd_normalize(A=A, B=B, D=D)
    out["abcd_noAD"] = mod.abcd_normalize(B=B, C=C)
    out["abcd_noB"] = mod.abcd_normalize(A=A, C=C, D=D)
    out["abcd_emptyA"] = mod.abcd_normalize(np.zeros((0, 0)), B, C, D)
    out["abcd_badC"] = _try(lambda: mod.abcd_normalize(A, B, C[:, :3], D))
    out["abcd_badD"] = _try(lambda: mod.abcd_normalize(A, B, C, D.T))
    out["abcd_noBD"] = _try(lambda: mod.abcd_normalize(A=A, C=C))

    # ss2tf / ss2zpk / zpk2ss
    out["ss2tf_in0"] = mod.ss2tf(A, B, C, D)
    out["ss2tf_in1"] = mod.ss2tf(A, B, C, D, input=1)
    out["ss2tf_badin"] = _try(lambda: mod.ss2tf(A, B, C, D, input=m))
    out["ss2tf_gain"] = mod.ss2tf(np.zeros((0, 0)), np.zeros((0, m)), np.zeros((p, 0)), D, input=1)
    out["ss2tf_empty"] = mod.ss2tf(np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((0, 0)), np.zeros((0, 1)))
    zz, pp, kk = mod.ss2zpk(A, B[:, 1:], C[2:], D[2:, 1:])
    out["ss2zpk"] = (np.sort_complex(zz), np.sort_complex(pp), kk)
    out["zpk2ss_1d"] = mod.zpk2ss(zr, pr, 5.0)
    out["zpk2ss_2d"] = mod.zpk2ss(zr2, pr, 2.0)

    # cont2discrete: every method on a MIMO system, tf and zpk inputs
    sys_ss = (A, B, C, D)
    for method in ("zoh", "foh", "bilinear", "tustin", "euler", "forward_diff", "backward_diff"):
        out[f"c2d_{method}"] = mod.cont2discrete(sys_ss, 0.1, method=method)
    out["c2d_impulse"] = mod.cont2discrete((A, B, C, Dz), 0.1, method="impulse")
    out["c2d_impulse_D"] = _try(lambda: mod.cont2discrete(sys_ss, 0.1, method="impulse"))
    out["c2d_gbt"] = mod.cont2discrete(sys_ss, 0.25, method="gbt", alpha=0.3)
    out["c2d_gbt_noalpha"] = _try(lambda: mod.cont2discrete(sys_ss, 0.1, method="gbt"))
    out["c2d_gbt_badalpha"] = _try(lambda: mod.cont2discrete(sys_ss, 0.1, method="gbt", alpha=-0.5))
    out["c2d_unknown"] = _try(lambda: mod.cont2discrete(sys_ss, 0.1, method="nope"))
    out["c2d_zoh_bigdt"] = mod.cont2discrete(sys_ss, 6.0)
    tf_num = np.array([[1.0, 2.0, 0.5], [3.0, -1.0, 2.0], [0.5, 4.0, -2.0]])
    tf_den = np.array([1.0, 3.0, 2.0, 0.7, 0.1])
    out["c2d_tf_zoh"] = mod.cont2discrete((tf_num, tf_den), 0.1)
    out["c2d_tf_foh"] = mod.cont2discrete((tf_num, tf_den), 0.1, method="foh")
    out["c2d_tf_siso"] = mod.cont2discrete((np.array([1.0, 2.0]), np.array([1.0, 3.0, 2.0])), 0.1, method="bilinear")
    zd, pd, kd, dtd = mod.cont2discrete((zr[:1], pr, 3.0), 0.1, method="backward_diff")
    out["c2d_zpk"] = (np.sort_complex(zd), np.sort_complex(pd), kd, dtd)
    return out
