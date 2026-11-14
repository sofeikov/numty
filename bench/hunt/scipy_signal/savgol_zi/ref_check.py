"""Compare clean.py against the installed scipy (1.18.1) on several cases.

Prints PASS/FAIL per case, exits non-zero on any failure.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.signal as ss
from scipy.signal import _savitzky_golay as sg
from scipy.signal import _arraytools as at

HERE = Path(__file__).resolve().parent


def load():
    spec = importlib.util.spec_from_file_location("savgol_zi_clean", HERE / "clean.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FAILS: list[str] = []


def check(name: str, got, want, rtol: float = 1e-9, atol: float = 1e-11) -> None:
    got = np.asarray(got)
    want = np.asarray(want)
    ok = got.shape == want.shape and np.allclose(got, want, rtol=rtol, atol=atol)
    if not ok:
        FAILS.append(name)
        detail = f"shape {got.shape} vs {want.shape}"
        if got.shape == want.shape:
            detail += f", max abs diff {np.max(np.abs(got - want)):.3g}"
        print(f"FAIL {name}: {detail}")
    else:
        print(f"PASS {name}")


def main() -> int:
    m = load()
    rng = np.random.default_rng(1234)

    # savgol_coeffs: odd/even windows, pos, use, derivatives, deriv > polyorder.
    for wl, po, kw in [
        (7, 2, {}),
        (8, 3, {}),
        (9, 4, {"deriv": 1, "delta": 0.5}),
        (11, 3, {"deriv": 2, "pos": 2}),
        (6, 2, {"pos": 4, "use": "dot"}),
        (5, 1, {"deriv": 3}),
        (4, 2, {"pos": 3, "deriv": 1, "use": "dot"}),
    ]:
        check(f"savgol_coeffs({wl},{po},{kw})", m.savgol_coeffs(wl, po, **kw),
              ss.savgol_coeffs(wl, po, **kw))

    # savgol_filter on (3 channels, 23 samples), along the last axis.
    x = rng.normal(size=(3, 23))
    for wl, po, kw in [
        (7, 2, {}),
        (8, 3, {}),
        (9, 3, {"deriv": 1, "delta": 0.3}),
        (5, 2, {"deriv": 2}),
        (11, 4, {"deriv": 5}),
        (7, 2, {"mode": "mirror"}),
        (6, 2, {"mode": "nearest"}),
        (9, 3, {"mode": "wrap"}),
        (7, 3, {"mode": "constant", "cval": 1.5}),
        (23, 4, {}),
    ]:
        check(f"savgol_filter({wl},{po},{kw})", m.savgol_filter(x, wl, po, **kw),
              ss.savgol_filter(x, wl, po, axis=-1, **kw))

    # 1 channel, and the per-row result equals the 1-D upstream call.
    x1 = rng.normal(size=(1, 17))
    check("savgol_filter 1-row", m.savgol_filter(x1, 9, 2, deriv=1)[0],
          ss.savgol_filter(x1[0], 9, 2, deriv=1))

    # Edge helpers in isolation.
    y0 = rng.normal(size=(3, 23))
    check("_fit_edges_polyfit", m._fit_edges_polyfit(x, 7, 3, 1, 0.5, y0.copy()),
          sg._fit_edges_polyfit(x, 7, 3, 1, 0.5, -1, y0.copy()))
    p = rng.normal(size=(5, 4))
    for mm in (0, 2, 5, 6):
        check(f"_polyder(m={mm})", m._polyder(p, mm), sg._polyder(p, mm, xp=np))

    # lfilter_zi: len(a) != len(b) both ways, a[0] != 1.
    for b, a in [
        (rng.normal(size=5), np.r_[1.0, 0.2, -0.1]),
        (rng.normal(size=2), np.r_[2.0, 0.3, 0.1, -0.05]),
        (*ss.butter(4, 0.2),),
    ]:
        check(f"lfilter_zi(n_b={b.size}, n_a={a.size})", m.lfilter_zi(b, a), ss.lfilter_zi(b, a))

    # sosfilt_zi: 3 sections.
    sos = ss.butter(6, [0.1, 0.4], btype="band", output="sos")
    check("sosfilt_zi(3 sections)", m.sosfilt_zi(sos[:3]), ss.sosfilt_zi(sos[:3]))
    check("sosfilt_zi(6 sections)", m.sosfilt_zi(sos), ss.sosfilt_zi(sos))

    # Extensions.
    xe = rng.normal(size=(3, 11))
    for name in ("odd_ext", "even_ext", "const_ext"):
        for n in (0, 4):
            check(f"{name}(n={n})", getattr(m, name)(xe, n), getattr(at, name)(xe, n, axis=-1))

    # lfilter / sosfilt replacements with zi.
    b, a = ss.butter(3, 0.25)
    zi = rng.normal(size=(3, 3))
    got_y, got_z = m._lfilter(b, a, xe, zi)
    want_y, want_z = ss.lfilter(b, a, xe, axis=-1, zi=zi)
    check("_lfilter y", got_y, want_y)
    check("_lfilter zf", got_z, want_z)
    b2, a2 = np.r_[0.5, 0.25, 0.1, 0.05], np.r_[2.0, -0.4]
    zi2 = rng.normal(size=(3, 3))
    got_y, got_z = m._lfilter(b2, a2, xe, zi2)
    want_y, want_z = ss.lfilter(b2, a2, xe, axis=-1, zi=zi2)
    check("_lfilter n_b>n_a, a0!=1 y", got_y, want_y)
    check("_lfilter n_b>n_a, a0!=1 zf", got_z, want_z)
    zis = rng.normal(size=(sos.shape[0], 3, 2))
    got_y, got_z = m._sosfilt(sos, xe, zis)
    want_y, want_z = ss.sosfilt(sos, xe, axis=-1, zi=zis)
    check("_sosfilt y", got_y, want_y)
    check("_sosfilt zf", got_z, want_z)

    # filtfilt / sosfiltfilt on (3 channels, 40 samples).
    xs = rng.normal(size=(3, 40))
    b, a = ss.butter(3, 0.25)
    for kw in [{}, {"padtype": "even"}, {"padtype": "constant", "padlen": 7},
               {"padtype": None}, {"padlen": 0}]:
        check(f"filtfilt({kw})", m.filtfilt(b, a, xs, **kw), ss.filtfilt(b, a, xs, axis=-1, **kw))
    bf = ss.firwin(5, 0.3)
    check("filtfilt FIR (a=[1])", m.filtfilt(bf, np.r_[1.0], xs), ss.filtfilt(bf, [1.0], xs))
    check("filtfilt n_a>n_b", m.filtfilt(np.r_[0.3, 0.2], np.r_[1.0, -0.3, 0.1, 0.05], xs),
          ss.filtfilt([0.3, 0.2], [1.0, -0.3, 0.1, 0.05], xs))
    for kw in [{}, {"padtype": "even"}, {"padtype": "constant", "padlen": 5}, {"padtype": None}]:
        check(f"sosfiltfilt({kw})", m.sosfiltfilt(sos, xs, **kw),
              ss.sosfiltfilt(sos, xs, axis=-1, **kw))
    # sos with a zero in sos[:, 2] and sos[:, 5] (reduces ntaps).
    sos_z = ss.butter(3, 0.2, output="sos")  # odd order: a first-order section
    check("sosfiltfilt odd order", m.sosfiltfilt(sos_z, xs), ss.sosfiltfilt(sos_z, xs))

    # Error paths agree.
    for name, f_clean, f_up in [
        ("savgol polyorder>=window", lambda: m.savgol_coeffs(3, 3), lambda: ss.savgol_coeffs(3, 3)),
        ("savgol interp window>n", lambda: m.savgol_filter(x[:, :5], 7, 2),
         lambda: ss.savgol_filter(x[:, :5], 7, 2)),
        ("lfilter_zi a0==0", lambda: m.lfilter_zi(np.r_[1.0], np.r_[0.0, 1.0]),
         lambda: ss.lfilter_zi([1.0], [0.0, 1.0])),
        ("filtfilt padlen too long", lambda: m.filtfilt(b, a, xs[:, :8]),
         lambda: ss.filtfilt(b, a, xs[:, :8])),
    ]:
        errs = []
        for f in (f_clean, f_up):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    f()
                errs.append(None)
            except Exception as e:  # noqa: BLE001
                errs.append(type(e).__name__)
        if errs[0] is not None and errs[0] == errs[1]:
            print(f"PASS error {name}: {errs[0]}")
        else:
            FAILS.append(name)
            print(f"FAIL error {name}: clean {errs[0]} vs upstream {errs[1]}")

    print(f"{len(FAILS)} failure(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
