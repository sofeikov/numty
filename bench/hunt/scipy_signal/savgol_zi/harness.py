"""Deterministic harness for the savgol_zi area: run(mod) -> dict of outputs.

Sizes never coincide inside a call: 3 channels; 29 samples for Savitzky-Golay
(windows 11 and 12, n_coef 4 and 5, halflen 5 and 6); 41 samples for the
filters; numerator 6 taps / denominator 3 taps (n_zi 5); 4 second-order
sections (n_sections 4, last axis 2, 6 coefficients).
"""

from __future__ import annotations

import warnings

import numpy as np


def _err(f) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f()
    except Exception as e:  # noqa: BLE001
        return type(e).__name__
    return "no error"


def run(mod) -> dict[str, object]:
    rng = np.random.default_rng(20251113)
    out: dict[str, object] = {}

    n_ch, n_sg, n_f = 3, 29, 41
    x = rng.normal(size=(n_ch, n_sg)).cumsum(axis=1)

    # --- savgol_coeffs: odd/even window, pos, use, deriv > polyorder.
    out["float_factorial"] = [mod.float_factorial(4), mod.float_factorial(200)]
    out["coeffs_odd"] = mod.savgol_coeffs(11, 3)
    out["coeffs_even"] = mod.savgol_coeffs(12, 4)
    out["coeffs_deriv"] = mod.savgol_coeffs(11, 3, deriv=2, delta=0.5)
    out["coeffs_pos_dot"] = mod.savgol_coeffs(12, 4, deriv=1, pos=9, use="dot")
    out["coeffs_zero"] = mod.savgol_coeffs(11, 3, deriv=5)
    out["coeffs_errors"] = [
        _err(lambda: mod.savgol_coeffs(5, 5)),
        _err(lambda: mod.savgol_coeffs(5, 2, pos=5)),
        _err(lambda: mod.savgol_coeffs(5, 2, use="corr")),
    ]

    # --- polynomial helpers on (window 11, 3 channels) columns.
    t = np.arange(11, dtype=np.float64)
    cols = rng.normal(size=(11, n_ch))
    pc = mod._polyfit(t, cols, 3)  # (4, 3)
    out["polyfit"] = pc
    out["polyval"] = mod._polyval(pc, np.linspace(-1.0, 2.0, 7).reshape(-1, 1))  # (7, 3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # rank-deficient fit: every sample at t=1, one usable column of lhs
        out["polyfit_rank"] = mod._polyfit(np.ones(11), cols, 3)
    out["polyfit_rank_warned"] = len(caught)
    out["polyder_0"] = mod._polyder(pc, 0)
    out["polyder_2"] = mod._polyder(pc, 2)  # (2, 3)
    out["polyder_big"] = mod._polyder(pc, 6)  # (1, 3)

    # --- convolve1d replacement, every mode, odd and even weights.
    w_odd = rng.normal(size=11)
    w_even = rng.normal(size=12)
    for mode in ("constant", "mirror", "nearest", "wrap"):
        out[f"conv_{mode}_odd"] = mod._convolve1d(x, w_odd, mode=mode, cval=0.75)
        out[f"conv_{mode}_even"] = mod._convolve1d(x, w_even, mode=mode, cval=-0.5)

    # --- edges and the full filter.
    y0 = rng.normal(size=(n_ch, n_sg))
    out["fit_edge"] = mod._fit_edge(x, 0, 11, 0, 5, 3, 1, 0.5, y0.copy())
    out["fit_edges"] = mod._fit_edges_polyfit(x, 12, 4, 0, 1.0, y0.copy())
    out["savgol_interp"] = mod.savgol_filter(x, 11, 3)
    out["savgol_interp_even"] = mod.savgol_filter(x, 12, 4, deriv=1, delta=0.25)
    out["savgol_interp_d2"] = mod.savgol_filter(x, 11, 3, deriv=2)
    out["savgol_mirror"] = mod.savgol_filter(x, 11, 3, mode="mirror")
    out["savgol_nearest"] = mod.savgol_filter(x, 12, 4, mode="nearest")
    out["savgol_wrap"] = mod.savgol_filter(x, 11, 3, mode="wrap")
    out["savgol_constant"] = mod.savgol_filter(x, 11, 3, mode="constant", cval=2.0)
    out["savgol_errors"] = [
        _err(lambda: mod.savgol_filter(x, 11, 3, mode="reflect")),
        _err(lambda: mod.savgol_filter(x[:, :9], 11, 3)),
    ]

    # --- lfilter_zi / sosfilt_zi.
    b = np.array([0.05, 0.12, 0.2, 0.2, 0.12, 0.05])  # 6 taps
    a = np.array([2.0, -0.9, 0.3])  # 3 taps, a[0] != 1
    out["lfilter_zi"] = mod.lfilter_zi(b, a)  # (5,)
    out["lfilter_zi_long_a"] = mod.lfilter_zi(np.array([0.4, 0.3]), np.array([1.0, -0.5, 0.2, 0.1]))
    out["lfilter_zi_errors"] = [
        _err(lambda: mod.lfilter_zi(b, np.array([0.0, 1.0]))),
        _err(lambda: mod.lfilter_zi(b, np.array([1.0, -1.0]))),
    ]
    # 4 sections of a band-pass Butterworth (order 4, normalized sos[:, 3] == 1).
    sos = np.array([
        [0.00482434, 0.00964869, 0.00482434, 1.0, -1.04859958, 0.29614036],
        [1.0, 2.0, 1.0, 1.0, -1.32091343, 0.63273879],
        [1.0, -2.0, 1.0, 1.0, -1.74089886, 0.77837542],
        [1.0, -2.0, 1.0, 1.0, -1.8973236, 0.92312628],
    ])
    out["sosfilt_zi"] = mod.sosfilt_zi(sos)  # (4, 2)
    out["sosfilt_zi_error"] = _err(lambda: mod.sosfilt_zi(sos[:, :5]))

    # --- extensions.
    xf = rng.normal(size=(n_ch, n_f))
    for name in ("odd_ext", "even_ext", "const_ext"):
        out[name] = getattr(mod, name)(xf, 7)
        out[name + "_0"] = getattr(mod, name)(xf, 0)
    out["ext_error"] = _err(lambda: mod.odd_ext(xf[:, :5], 7))

    # --- lfilter / sosfilt replacements.
    zi = rng.normal(size=(n_ch, 5))
    out["lfilter"] = list(mod._lfilter(b, a, xf, zi))
    out["lfilter_static"] = list(mod._lfilter(np.array([2.0]), np.array([4.0]), xf, np.zeros((n_ch, 0))))
    zis = rng.normal(size=(4, n_ch, 2))
    out["sosfilt"] = list(mod._sosfilt(sos, xf, zis))

    # --- padding validation and forward-backward filtering.
    for pt, pl in (("odd", None), ("even", 9), ("constant", 4), (None, None), ("odd", 0)):
        edge, ext = mod._validate_pad(pt, pl, xf, 7)
        out[f"validate_pad_{pt}_{pl}"] = [edge, ext]
    out["validate_pad_errors"] = [
        _err(lambda: mod._validate_pad("zero", None, xf, 7)),
        _err(lambda: mod._validate_pad("odd", None, xf, 14)),
    ]
    out["validate_x"] = mod._validate_x(xf)
    out["validate_sos"] = list(mod._validate_sos(sos))
    bad = sos.copy()
    bad[1, 3] = 2.0
    out["validate_sos_error"] = _err(lambda: mod._validate_sos(bad))

    out["filtfilt"] = mod.filtfilt(b, a, xf)
    out["filtfilt_even"] = mod.filtfilt(b, a, xf, padtype="even", padlen=11)
    out["filtfilt_const"] = mod.filtfilt(b, a, xf, padtype="constant", padlen=6)
    out["filtfilt_nopad"] = mod.filtfilt(b, a, xf, padtype=None)
    out["filtfilt_long_a"] = mod.filtfilt(np.array([0.4, 0.3]), np.array([1.0, -0.5, 0.2, 0.1]), xf)
    out["sosfiltfilt"] = mod.sosfiltfilt(sos, xf)
    out["sosfiltfilt_even"] = mod.sosfiltfilt(sos, xf, padtype="even", padlen=8)
    out["sosfiltfilt_const"] = mod.sosfiltfilt(sos, xf, padtype="constant", padlen=6)
    out["sosfiltfilt_nopad"] = mod.sosfiltfilt(sos, xf, padtype=None)
    sos_first = sos.copy()
    sos_first[0] = [0.2, 0.2, 0.0, 1.0, -0.6, 0.0]  # a first-order section
    out["sosfiltfilt_first_order"] = mod.sosfiltfilt(sos_first, xf)
    return out
