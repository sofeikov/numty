"""Harness for the scipy.signal LTI simulation / pole placement extract.

Sizes never coincide: 4 states, 2 inputs, 3 outputs, 11 time samples for
the main system; 5 states / 6 inputs for the series partner; 7 rows for an
output scaling; 15 resampled points in dlsim.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

N_X, N_U, N_Y, N_T = 4, 2, 3, 11


def _system(rng: np.random.Generator, n: int, m: int, p: int) -> tuple[np.ndarray, ...]:
    A = rng.standard_normal((n, n)) - 2.5 * np.eye(n)
    B = rng.standard_normal((n, m))
    C = rng.standard_normal((p, n))
    D = rng.standard_normal((p, m))
    return A, B, C, D


def _ss(sys: Any) -> list[np.ndarray]:
    return [sys.A, sys.B, sys.C, sys.D]


def _err(fn: Any) -> str:
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return type(e).__name__
    return "no error"


def run(mod: Any) -> dict[str, object]:
    rng = np.random.default_rng(1234)
    out: dict[str, object] = {}

    A, B, C, D = _system(rng, N_X, N_U, N_Y)
    sys1 = mod.StateSpace(A, B, C, D)

    # --- expm / block diag -------------------------------------------------
    out["expm_small"] = mod.expm(0.1 * A)
    out["expm_large"] = mod.expm(4.0 * A)
    out["expm_rect_block"] = mod.expm(rng.standard_normal((6, 6)))

    # --- StateSpace arithmetic -------------------------------------------
    A2, B2, C2, D2 = _system(rng, 5, 6, N_U)       # series: outputs = sys1 inputs
    sys2 = mod.StateSpace(A2, B2, C2, D2)
    out["series"] = _ss(sys1 * sys2)
    A3, B3, C3, D3 = _system(rng, 5, N_U, N_Y)     # parallel: same inputs/outputs
    sys3 = mod.StateSpace(A3, B3, C3, D3)
    out["parallel"] = _ss(sys1 + sys3)
    out["difference"] = _ss(sys1 - sys3)
    out["neg"] = _ss(-sys1)
    K_in = rng.standard_normal((N_U, 6))
    out["mul_matrix"] = _ss(sys1 * K_in)
    out["mul_scalar"] = _ss(sys1 * 2.5)
    K_out = rng.standard_normal((7, N_Y))
    out["rmul_matrix"] = _ss(K_out * sys1)
    out["rmul_scalar"] = _ss(1.5 * sys1)
    D_static = rng.standard_normal((N_Y, N_U))
    out["add_matrix"] = _ss(sys1 + D_static)
    out["radd_matrix"] = _ss(D_static + sys1)
    out["rsub_matrix"] = _ss(D_static - sys1)
    out["truediv"] = _ss(sys1 / 4.0)
    As, Bs, Cs, Ds = _system(rng, N_X, 1, 1)
    siso = mod.StateSpace(As, Bs, Cs, Ds)
    out["add_scalar_siso"] = _ss(siso + 0.75)
    out["sub_scalar_siso"] = _ss(siso - 0.75)
    disc1 = mod.StateSpace(A * 0.1, B, C, D, dt=0.1)
    disc3 = mod.StateSpace(A3 * 0.1, B3, C3, D3, dt=0.1)
    out["parallel_discrete"] = _ss(disc1 + disc3)
    out["err_add_dims"] = _err(lambda: sys1 + np.ones((2, 2)))
    out["err_add_dt"] = _err(lambda: disc1 + mod.StateSpace(A3, B3, C3, D3, dt=0.2))
    out["err_mul_mixed"] = _err(lambda: disc1 * sys2)
    out["err_div_array"] = _err(lambda: sys1 / np.ones(2))
    out["err_validate"] = _err(lambda: mod.StateSpace(A, B, C[:, :3], D))

    # --- lsim ----------------------------------------------------------------
    T = np.linspace(0.0, 1.0, N_T)
    U = rng.standard_normal((N_T, N_U))
    X0 = rng.standard_normal(N_X)
    out["lsim_interp"] = list(mod.lsim(sys1, U, T, X0=X0))
    out["lsim_zoh"] = list(mod.lsim(sys1, U, T, X0=X0, interp=False))
    out["lsim_none"] = list(mod.lsim(sys1, None, T, X0=X0))
    out["lsim_zero"] = list(mod.lsim(sys1, 0.0, T, X0=X0))
    out["lsim_t0_pos"] = list(mod.lsim(sys1, U, T + 0.3, X0=X0))
    out["lsim_one_step"] = list(mod.lsim(sys1, U[:1], T[:1], X0=X0))
    out["lsim_one_step_noinput"] = list(mod.lsim(sys1, None, T[:1]))
    out["lsim_siso_1d"] = list(mod.lsim(siso, U[:, 0], T))
    out["lsim_noX0"] = list(mod.lsim(sys1, U, T))
    out["err_lsim_rows"] = _err(lambda: mod.lsim(sys1, U[:-1], T))
    out["err_lsim_inputs"] = _err(lambda: mod.lsim(sys1, U[:, :1], T))
    out["err_lsim_spacing"] = _err(lambda: mod.lsim(sys1, U, T ** 2))
    out["err_lsim_negative"] = _err(lambda: mod.lsim(sys1, U, T - 1.0))

    # --- impulse / step (single input) -----------------------------------
    Ai, Bi, Ci, Di = _system(rng, N_X, 1, N_Y)
    sisoish = mod.StateSpace(Ai, Bi, Ci, Di)
    out["impulse_T"] = list(mod.impulse(sisoish, T=T))
    out["impulse_X0_scalar"] = list(mod.impulse(sisoish, X0=0.5, T=T))
    out["impulse_X0_column"] = list(mod.impulse(sisoish, X0=np.full((N_X, 1), 0.25), T=T))
    out["impulse_default_T"] = list(mod.impulse(sisoish, N=13))
    out["step_T"] = list(mod.step(sisoish, T=T))
    out["step_X0"] = list(mod.step(sisoish, X0=X0, T=T))
    out["step_default_T"] = list(mod.step(sisoish, N=9))

    # --- place_poles --------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Ap = rng.standard_normal((N_X, N_X))
        Bp = rng.standard_normal((N_X, N_U))
        res = []
        for poles, method in (
            (np.array([-1.0, -2.0, -3.0, -4.0]), "YT"),
            (np.array([-1.0, -2.0, -3.0, -4.0]), "KNV0"),
            (np.array([-1.0 + 1.0j, -1.0 - 1.0j, -2.0 + 0.5j, -2.0 - 0.5j]), "YT"),
            (np.array([-3.0, -1.0 + 1.0j, -1.0 - 1.0j, -4.0]), "YT"),
        ):
            fsf = mod.place_poles(Ap, Bp, poles, method=method)
            res.append([fsf.gain_matrix, np.sort_complex(fsf.computed_poles),
                        fsf.requested_poles, fsf.nb_iter])
        out["place_mimo"] = res

        # one real pole + two complex pairs: hnb == 0 path, 5 states, 3 inputs
        A5 = rng.standard_normal((5, 5))
        B5 = rng.standard_normal((5, 3))
        fsf = mod.place_poles(A5, B5, np.array([-2.0, -1 + 2j, -1 - 2j, -3 + 1j, -3 - 1j]))
        out["place_hnb0"] = [fsf.gain_matrix, np.sort_complex(fsf.computed_poles)]

        # odd number of real poles (3) with 5 states, 2 inputs
        B52 = rng.standard_normal((5, 2))
        fsf = mod.place_poles(A5, B52, np.array([-1.0, -2.0, -3.0, -1 + 1j, -1 - 1j]))
        out["place_odd_real"] = [fsf.gain_matrix, np.sort_complex(fsf.computed_poles)]

        # single input: nothing to optimise
        B1 = rng.standard_normal((N_X, 1))
        fsf = mod.place_poles(Ap, B1, np.array([-1.0, -2.0, -0.5 + 1j, -0.5 - 1j]))
        out["place_siso"] = [fsf.gain_matrix, np.sort_complex(fsf.computed_poles), fsf.rtol]

        # B has as many rows as its rank: lstsq path (wide B, 6 inputs)
        Bw = rng.standard_normal((N_X, 6))
        fsf = mod.place_poles(Ap, Bw, np.array([-1.0, -2.0, -0.5 + 1j, -0.5 - 1j]))
        out["place_lstsq"] = [fsf.gain_matrix, np.sort_complex(fsf.computed_poles), fsf.X]

        out["err_place_knv_complex"] = _err(lambda: mod.place_poles(
            Ap, Bp, np.array([-1.0, -2.0, -1 + 1j, -1 - 1j]), method="KNV0"))
        out["err_place_count"] = _err(lambda: mod.place_poles(Ap, Bp, np.array([-1.0, -2.0])))
        out["err_place_repeat"] = _err(lambda: mod.place_poles(
            Ap, Bp, np.array([-1.0, -1.0, -1.0, -2.0])))
        out["err_place_conj"] = _err(lambda: mod.place_poles(
            Ap, Bp, np.array([-1.0, -2.0, -1 + 1j, -3 - 1j])))
        out["err_place_method"] = _err(lambda: mod.place_poles(
            Ap, Bp, np.array([-1.0, -2.0, -3.0, -4.0]), method="XX"))

    # --- dlsim ---------------------------------------------------------------
    Ad = 0.3 * rng.standard_normal((N_X, N_X))
    sysd = mod.StateSpace(Ad, B, C, D, dt=0.07)
    out["dlsim_u2d"] = list(mod.dlsim(sysd, U))
    out["dlsim_x0"] = list(mod.dlsim(sysd, U, x0=X0))
    out["dlsim_t"] = list(mod.dlsim(sysd, U, t=T, x0=X0))
    sisod = mod.StateSpace(Ad, Bs, Cs, Ds, dt=0.07)
    out["dlsim_siso_1d"] = list(mod.dlsim(sisod, U[:, 1]))
    out["dlsim_siso_1d_t"] = list(mod.dlsim(sisod, U[:, 1], t=T))
    out["err_dlsim_continuous"] = _err(lambda: mod.dlsim(sys1, U))
    return out
