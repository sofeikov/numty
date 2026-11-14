"""Compare clean.py against the installed scipy.signal (and scipy.linalg.expm).

Prints PASS or FAIL per case and exits non-zero on any failure.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import linalg, signal

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lti_clean", HERE / "clean.py")
assert spec is not None and spec.loader is not None
clean: Any = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean)

warnings.simplefilter("ignore")

FAILED: list[str] = []


def same(x: Any, y: Any, path: str = "") -> None:
    if isinstance(x, (tuple, list)) and isinstance(y, (tuple, list)):
        if len(x) != len(y):
            raise AssertionError(f"{path}: length {len(x)} vs {len(y)}")
        for i, (a, b) in enumerate(zip(x, y)):
            same(a, b, f"{path}[{i}]")
        return
    xa, ya = np.asarray(x), np.asarray(y)
    if xa.shape != ya.shape:
        raise AssertionError(f"{path}: shape {xa.shape} vs {ya.shape}")
    if type(x) is not type(y) and not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
        if not (np.isscalar(x) and np.isscalar(y)):
            raise AssertionError(f"{path}: type {type(x).__name__} vs {type(y).__name__}")
    if xa.dtype.kind in "fc" or ya.dtype.kind in "fc":
        if xa.dtype.kind != ya.dtype.kind and xa.size:
            raise AssertionError(f"{path}: dtype {xa.dtype} vs {ya.dtype}")
        if not np.allclose(xa, ya, rtol=1e-9, atol=1e-11):
            raise AssertionError(f"{path}: values differ\n{xa}\n{ya}")
    elif not np.array_equal(xa, ya):
        raise AssertionError(f"{path}: values differ\n{xa}\n{ya}")


def sort_roots(t: tuple[Any, ...]) -> tuple[Any, ...]:
    z, p, *rest = t
    return (np.sort_complex(np.asarray(z)), np.sort_complex(np.asarray(p)), *rest)


def case(name: str, ours: Callable[[], Any], ref: Callable[[], Any],
         post: Callable[[Any], Any] = lambda v: v) -> None:
    try:
        try:
            r = ("ok", post(ref()))
        except Exception as e:  # noqa: BLE001
            r = ("raise", type(e).__name__)
        try:
            o = ("ok", post(ours()))
        except Exception as e:  # noqa: BLE001
            o = ("raise", type(e).__name__)
        if r[0] != o[0]:
            raise AssertionError(f"upstream {r}, clean {o}")
        if r[0] == "raise":
            if r[1] != o[1]:
                raise AssertionError(f"upstream raises {r[1]}, clean raises {o[1]}")
        else:
            same(o[1], r[1])
        print(f"PASS {name}" + (f" (both raise {r[1]})" if r[0] == "raise" else ""))
    except AssertionError as e:
        FAILED.append(name)
        print(f"FAIL {name}: {e}")


rng = np.random.default_rng(1234)
n, m, p = 4, 2, 3
A = rng.standard_normal((n, n)) - 2.0 * np.eye(n)
B = rng.standard_normal((n, m))
C = rng.standard_normal((p, n))
D = rng.standard_normal((p, m))
Dz = np.zeros((p, m))

# --- normalize / tf2zpk / zpk2tf
case("normalize 1-D", lambda: clean.normalize(np.array([0.0, 3.0, 6.0]), np.array([2.0, -5.0, 4.0])),
     lambda: signal.normalize(np.array([0.0, 3.0, 6.0]), np.array([2.0, -5.0, 4.0])))
num2 = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 4.0, 1.0], [0.0, 2.0, 0.0, 5.0]])
case("normalize 2-D (3 rows)", lambda: clean.normalize(num2, np.array([0.0, 2.0, 1.0, 3.0, 4.0])),
     lambda: signal.normalize(num2, np.array([0.0, 2.0, 1.0, 3.0, 4.0])))
case("normalize 2-D single row squeezed", lambda: clean.normalize(num2[:1], np.array([1.0, 3.0])),
     lambda: signal.normalize(num2[:1], np.array([1.0, 3.0])))
case("normalize all-zero den", lambda: clean.normalize(np.array([1.0]), np.array([0.0, 0.0])),
     lambda: signal.normalize(np.array([1.0]), np.array([0.0, 0.0])))
case("tf2zpk", lambda: clean.tf2zpk(np.array([3.0, 0.0, 1.0]), np.array([1.0, 5.0, 13.0, 2.0])),
     lambda: signal.tf2zpk(np.array([3.0, 0.0, 1.0]), np.array([1.0, 5.0, 13.0, 2.0])), sort_roots)
z1, p1 = np.array([2.0, 6.0]), np.array([1.0, 8.0, -3.0])
case("zpk2tf 1-D", lambda: clean.zpk2tf(z1, p1, 5.0), lambda: signal.zpk2tf(z1, p1, 5.0))
z2 = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]])
case("zpk2tf 2-D (3 outputs)", lambda: clean.zpk2tf(z2, p1, 2.0), lambda: signal.zpk2tf(z2, p1, 2.0))
case("zpk2tf no zeros", lambda: clean.zpk2tf(np.zeros(0), p1, 2.0), lambda: signal.zpk2tf(np.zeros(0), p1, 2.0))

# --- tf2ss
case("tf2ss SISO", lambda: clean.tf2ss(np.array([1.0, 3.0, 3.0]), np.array([1.0, 2.0, 1.0])),
     lambda: signal.tf2ss(np.array([1.0, 3.0, 3.0]), np.array([1.0, 2.0, 1.0])))
case("tf2ss SIMO 3 outputs, 4 states", lambda: clean.tf2ss(num2, np.array([1.0, 2.0, 1.0, 3.0, 4.0])),
     lambda: signal.tf2ss(num2, np.array([1.0, 2.0, 1.0, 3.0, 4.0])))
case("tf2ss strictly proper short num", lambda: clean.tf2ss(np.array([2.0, 1.0]), np.array([1.0, 0.5, 3.0, 1.0])),
     lambda: signal.tf2ss(np.array([2.0, 1.0]), np.array([1.0, 0.5, 3.0, 1.0])))
case("tf2ss static gain K==1", lambda: clean.tf2ss(np.array([[2.0], [3.0]]), np.array([4.0])),
     lambda: signal.tf2ss(np.array([[2.0], [3.0]]), np.array([4.0])))
case("tf2ss null system", lambda: clean.tf2ss(np.zeros(0), np.array([1.0])),
     lambda: signal.tf2ss(np.zeros(0), np.array([1.0])))
case("tf2ss improper", lambda: clean.tf2ss(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])),
     lambda: signal.tf2ss(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])))

# --- abcd_normalize
case("abcd_normalize full MIMO", lambda: clean.abcd_normalize(A, B, C, D), lambda: signal.abcd_normalize(A, B, C, D))
case("abcd_normalize C=None", lambda: clean.abcd_normalize(A=A, B=B, D=D), lambda: signal.abcd_normalize(A=A, B=B, D=D))
case("abcd_normalize A=None D=None", lambda: clean.abcd_normalize(B=B, C=C), lambda: signal.abcd_normalize(B=B, C=C))
case("abcd_normalize B=None", lambda: clean.abcd_normalize(A=A, C=C, D=D), lambda: signal.abcd_normalize(A=A, C=C, D=D))
case("abcd_normalize size-0 A", lambda: clean.abcd_normalize(np.zeros((0, 0)), B, C, D),
     lambda: signal.abcd_normalize(np.zeros((0, 0)), B, C, D))
case("abcd_normalize bad C", lambda: clean.abcd_normalize(A, B, C[:, :3], D),
     lambda: signal.abcd_normalize(A, B, C[:, :3], D))
case("abcd_normalize B=D=None", lambda: clean.abcd_normalize(A=A, C=C), lambda: signal.abcd_normalize(A=A, C=C))

# --- ss2tf / ss2zpk / zpk2ss
for inp in range(m):
    case(f"ss2tf MIMO input={inp}", lambda inp=inp: clean.ss2tf(A, B, C, D, input=inp),
         lambda inp=inp: signal.ss2tf(A, B, C, D, input=inp))
case("ss2tf bad input", lambda: clean.ss2tf(A, B, C, D, input=m), lambda: signal.ss2tf(A, B, C, D, input=m))
case("ss2tf static gain (0 states)",
     lambda: clean.ss2tf(np.zeros((0, 0)), np.zeros((0, m)), np.zeros((p, 0)), D, input=1),
     lambda: signal.ss2tf(np.zeros((0, 0)), np.zeros((0, m)), np.zeros((p, 0)), D, input=1))
case("ss2tf empty system",
     lambda: clean.ss2tf(np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((0, 0)), np.zeros((0, 1))),
     lambda: signal.ss2tf(np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((0, 0)), np.zeros((0, 1))))
case("ss2zpk SISO", lambda: clean.ss2zpk(A, B[:, :1], C[:1], D[:1, :1]),
     lambda: signal.ss2zpk(A, B[:, :1], C[:1], D[:1, :1]), sort_roots)
case("ss2zpk MIMO", lambda: clean.ss2zpk(A, B, C, D, input=1),
     lambda: signal.ss2zpk(A, B, C, D, input=1), sort_roots)
case("zpk2ss 1-D", lambda: clean.zpk2ss(z1, p1, 5.0), lambda: signal.zpk2ss(z1, p1, 5.0))
case("zpk2ss 2-D (3 outputs)", lambda: clean.zpk2ss(z2, p1, 2.0), lambda: signal.zpk2ss(z2, p1, 2.0))

# --- expm stand-in
for scale in (0.01, 1.0, 30.0):
    M = rng.standard_normal((6, 6)) * scale
    case(f"expm scale={scale}", lambda M=M: clean._expm(M), lambda M=M: linalg.expm(M))
nil = np.diag(np.ones(5), 1)
case("expm nilpotent", lambda: clean._expm(nil), lambda: linalg.expm(nil))
case("block_diag", lambda: clean._block_diag2(B, C), lambda: linalg.block_diag(B, C))

# --- cont2discrete
dt = 0.1
for method in ("zoh", "foh", "bilinear", "tustin", "euler", "forward_diff", "backward_diff"):
    case(f"cont2discrete ss {method}", lambda method=method: clean.cont2discrete((A, B, C, D), dt, method=method),
         lambda method=method: signal.cont2discrete((A, B, C, D), dt, method=method))
case("cont2discrete ss impulse", lambda: clean.cont2discrete((A, B, C, Dz), dt, method="impulse"),
     lambda: signal.cont2discrete((A, B, C, Dz), dt, method="impulse"))
case("cont2discrete ss impulse D!=0", lambda: clean.cont2discrete((A, B, C, D), dt, method="impulse"),
     lambda: signal.cont2discrete((A, B, C, D), dt, method="impulse"))
case("cont2discrete ss gbt 0.3", lambda: clean.cont2discrete((A, B, C, D), 0.2, method="gbt", alpha=0.3),
     lambda: signal.cont2discrete((A, B, C, D), 0.2, method="gbt", alpha=0.3))
case("cont2discrete gbt no alpha", lambda: clean.cont2discrete((A, B, C, D), dt, method="gbt"),
     lambda: signal.cont2discrete((A, B, C, D), dt, method="gbt"))
case("cont2discrete gbt alpha>1", lambda: clean.cont2discrete((A, B, C, D), dt, method="gbt", alpha=1.5),
     lambda: signal.cont2discrete((A, B, C, D), dt, method="gbt", alpha=1.5))
case("cont2discrete unknown method", lambda: clean.cont2discrete((A, B, C, D), dt, method="nope"),
     lambda: signal.cont2discrete((A, B, C, D), dt, method="nope"))
case("cont2discrete large dt zoh", lambda: clean.cont2discrete((A, B, C, D), 7.0),
     lambda: signal.cont2discrete((A, B, C, D), 7.0))
tf_num, tf_den = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]]), np.array([1.0, 3.0, 2.0, 0.7])
for method in ("zoh", "foh", "bilinear", "impulse"):
    case(f"cont2discrete tf SIMO {method}", lambda method=method: clean.cont2discrete((tf_num, tf_den), dt, method=method),
         lambda method=method: signal.cont2discrete((tf_num, tf_den), dt, method=method))
case("cont2discrete tf SISO", lambda: clean.cont2discrete((np.array([1.0, 2.0]), np.array([1.0, 3.0, 2.0])), dt),
     lambda: signal.cont2discrete((np.array([1.0, 2.0]), np.array([1.0, 3.0, 2.0])), dt))
for method in ("zoh", "backward_diff"):
    case(f"cont2discrete zpk {method}", lambda method=method: clean.cont2discrete((z1[:1], p1, 3.0), dt, method=method),
         lambda method=method: signal.cont2discrete((z1[:1], p1, 3.0), dt, method=method), sort_roots)

if FAILED:
    print(f"{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1)
print("all cases PASS")
