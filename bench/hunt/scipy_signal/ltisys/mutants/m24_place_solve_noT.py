# MUTANT: place_poles: m = solve(X.T, diag(P) X.T) without the outer .T (square)
"""LTI simulation and pole placement, extracted from scipy.signal.

Upstream: scipy v1.18.1 (commit e4e854e), file ``scipy/signal/_ltisys.py``:

* ``StateSpace`` arithmetic (``__mul__``, ``__rmul__``, ``__neg__``,
  ``__add__``, ``__sub__``, ``__radd__``, ``__rsub__``, ``__truediv__``),
  lines 1495-1810, minus the TF/ZPK conversion machinery and the
  continuous/discrete subclass split (``dt`` stays an attribute);
* ``lsim`` (2066-2278), ``_default_response_times`` (2281-2309),
  ``impulse`` (2312-2379), ``step`` (2382-2446), state-space path only;
* ``place_poles`` and its helpers ``_valid_inputs``,
  ``_order_complex_poles``, ``_KNV0``, ``_YT_real``, ``_YT_complex``,
  ``_YT_loop``, ``_KNV0_loop`` (2600-3338);
* ``dlsim`` (3341-3458), state-space input only.

The shape checks of ``abcd_normalize`` (``scipy/signal/_lti_conversion.py``
118-230) are kept for fully specified float64 2-D matrices.

numpy-only replacements for scipy routines:

* ``scipy.linalg.expm`` -> ``expm`` below (Pade 13 scaling and squaring,
  Higham 2005);
* ``scipy.linalg.block_diag`` of two blocks -> ``_block_diag2``;
* ``scipy.linalg.qr(mode="full")`` -> ``np.linalg.qr(mode="complete")``
  (the commented-out debug line upstream);
* ``scipy.interpolate.make_interp_spline(t, u, k=1)(tout)`` ->
  ``_interp_linear`` (piecewise linear, extrapolating like the spline).

Systems are real (float64); upstream also accepts complex matrices.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Generic, Literal, TypeVar, overload

import numpy as np
from numpy import atleast_1d, dot, linspace, ones, real, squeeze, transpose, zeros

from numty import F64, Arr

n_states = TypeVar("n_states", bound=int, covariant=True)
n_inputs = TypeVar("n_inputs", bound=int, covariant=True)
n_outputs = TypeVar("n_outputs", bound=int, covariant=True)
n_s2 = TypeVar("n_s2", bound=int)
n_in2 = TypeVar("n_in2", bound=int)
k_cols = TypeVar("k_cols", bound=int)
k_rows = TypeVar("k_rows", bound=int)
n_steps = TypeVar("n_steps", bound=int)
n_samples = TypeVar("n_samples", bound=int)
n_out = TypeVar("n_out", bound=int)
n_poles = TypeVar("n_poles", bound=int)
PT = TypeVar("PT", bound=np.inexact[Any])

Inexact = np.inexact[Any]
UpdateLoop = Callable[..., tuple[bool, float, int]]


# ---------------------------------------------------------------------------
# numpy stand-ins for scipy.linalg / scipy.interpolate
# ---------------------------------------------------------------------------

_PADE13 = (64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
           1187353796428800.0, 129060195264000.0, 10559470521600.0,
           670442572800.0, 33522128640.0, 1323241920.0, 40840800.0,
           960960.0, 16380.0, 182.0, 1.0)


def expm(A: F64["k k"]) -> F64["k k"]:
    """Matrix exponential (stand-in for scipy.linalg.expm)."""
    b = _PADE13
    norm = float(np.linalg.norm(A, 1))
    s = 0
    if norm > 5.371920351148152:
        s = int(np.ceil(np.log2(norm / 5.371920351148152)))
    As = A / 2.0**s
    ident = np.eye(A.shape[0])
    A2 = As @ As
    A4 = A2 @ A2
    A6 = A4 @ A2
    U = As @ (A6 @ (b[13] * A6 + b[11] * A4 + b[9] * A2)
              + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = (A6 @ (b[12] * A6 + b[10] * A4 + b[8] * A2)
         + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident)
    R = np.linalg.solve(V - U, V + U)
    for _ in range(s):
        R = R @ R
    return R


def _block_diag2(a: F64["r1 r1"], b: F64["r2 r2"]) -> F64["r1+r2 r1+r2"]:
    """scipy.linalg.block_diag(a, b) for two square float blocks."""
    r1 = a.shape[0]
    r2 = b.shape[0]
    out = np.zeros((r1 + r2, r1 + r2))
    out[:r1, :r1] = a
    out[r1:, r1:] = b
    return out


def _interp_linear(
    t: F64["n_samples"], u: F64["n_samples n_inputs"], tout: F64["n_out"]
) -> F64["n_out n_inputs"]:
    """make_interp_spline(t, u, k=1)(tout): piecewise linear, extrapolating."""
    idx = np.clip(np.searchsorted(t, tout, side="right") - 1, 0, len(t) - 2)
    w = (tout - t[idx]) / (t[idx + 1] - t[idx])
    return u[idx] + w[:, np.newaxis] * (u[idx + 1] - u[idx])


# ---------------------------------------------------------------------------
# StateSpace
# ---------------------------------------------------------------------------

def _abcd_validate(
    A: F64["n_states n_states"],
    B: F64["n_states n_inputs"],
    C: F64["n_outputs n_states"],
    D: F64["n_outputs n_inputs"],
) -> None:
    """The shape checks of abcd_normalize for four given 2-D matrices."""
    n = A.shape[0] or B.shape[0] or C.shape[1]
    p = B.shape[1] or D.shape[1]
    q = C.shape[0] or D.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"Parameter A has shape {A.shape} but should be ({n}, {n})!")
    if B.shape != (n, p):
        raise ValueError(f"Parameter B has shape {B.shape} but should be ({n}, {p})!")
    if C.shape != (q, n):
        raise ValueError(f"Parameter C has shape {C.shape} but should be ({q}, {n})!")
    if D.shape != (q, p):
        raise ValueError(f"Parameter D has shape {D.shape} but should be ({q}, {p})!")


class StateSpace(Generic[n_states, n_inputs, n_outputs]):
    """Linear time invariant system in state-space form (A, B, C, D)."""

    __array_priority__ = 100.0
    __array_ufunc__ = None

    def __init__(
        self,
        A: F64["n_states n_states"],
        B: F64["n_states n_inputs"],
        C: F64["n_outputs n_states"],
        D: F64["n_outputs n_inputs"],
        dt: float | None = None,
    ) -> None:
        _abcd_validate(A, B, C, D)
        self.A = np.atleast_2d(A)
        self.B = np.atleast_2d(B)
        self.C = np.atleast_2d(C)
        self.D = np.atleast_2d(D)
        self.inputs = self.B.shape[-1]
        self.outputs = self.C.shape[0]
        self.dt = dt

    @property
    def _dt_dict(self) -> dict[str, float]:
        if self.dt is None:
            return {}
        return {"dt": self.dt}

    def _check_binop_other(self, other: object) -> bool:
        return isinstance(other, StateSpace | np.ndarray | float | complex
                          | np.number | int)

    # -- series connection / input scaling ---------------------------------

    def _series(
        self, other: "StateSpace[n_s2, n_in2, n_inputs]"
    ) -> "StateSpace[int, n_in2, n_outputs]":
        if self.dt != other.dt:
            raise TypeError("Cannot multiply systems with different `dt`.")

        n1 = self.A.shape[0]
        n2 = other.A.shape[0]

        # [x1']   [A1 B1*C2 ] [x1]   [B1*D2]
        # [x2'] = [0  A2    ] [x2] + [B2   ] u2
        #                    [x1]
        #  y2   = [C1 D1*C2] [x2] + D1*D2 u2
        a = np.vstack((np.hstack((self.A, np.dot(self.B, other.C))),
                       np.hstack((zeros((n2, n1)), other.A))))
        b = np.vstack((np.dot(self.B, other.D), other.B))
        c = np.hstack((self.C, np.dot(self.D, other.C)))
        d = np.dot(self.D, other.D)
        return StateSpace(a, b, c, d, **self._dt_dict)

    def _scale_input(
        self, other: F64["n_inputs k_cols"]
    ) -> "StateSpace[n_states, k_cols, n_outputs]":
        # For post multiplication the input gets scaled
        a = self.A
        b = np.dot(self.B, other)
        c = self.C
        d = np.dot(self.D, other)
        return StateSpace(a, b, c, d, **self._dt_dict)

    def _scale_input_scalar(
        self, other: float
    ) -> "StateSpace[n_states, n_inputs, n_outputs]":
        a = self.A
        b = np.dot(self.B, other)
        c = self.C
        d = np.dot(self.D, other)
        return StateSpace(a, b, c, d, **self._dt_dict)

    @overload
    def __mul__(
        self, other: "StateSpace[n_s2, n_in2, n_inputs]"
    ) -> "StateSpace[int, n_in2, n_outputs]": ...
    @overload
    def __mul__(
        self, other: F64["n_inputs k_cols"]
    ) -> "StateSpace[n_states, k_cols, n_outputs]": ...
    @overload
    def __mul__(self, other: float) -> "StateSpace[n_states, n_inputs, n_outputs]": ...
    def __mul__(self, other: Any) -> Any:
        """Post-multiply another system (series connection) or a matrix."""
        if not self._check_binop_other(other):
            return NotImplemented
        if isinstance(other, StateSpace):
            # Disallow mix of discrete and continuous systems.
            if (other.dt is None) != (self.dt is None):
                return NotImplemented
            return self._series(other)
        if isinstance(other, np.ndarray):
            return self._scale_input(other)
        return self._scale_input_scalar(other)

    def _scale_output(
        self, other: F64["k_rows n_outputs"]
    ) -> "StateSpace[n_states, n_inputs, k_rows]":
        # For pre-multiplication only the output gets scaled
        a = self.A
        b = self.B
        c = np.dot(other, self.C)
        d = np.dot(other, self.D)
        return StateSpace(a, b, c, d, **self._dt_dict)

    @overload
    def __rmul__(
        self, other: F64["k_rows n_outputs"]
    ) -> "StateSpace[n_states, n_inputs, k_rows]": ...
    @overload
    def __rmul__(self, other: float) -> "StateSpace[n_states, n_inputs, n_outputs]": ...
    def __rmul__(self, other: Any) -> Any:
        """Pre-multiply a scalar or matrix (but not StateSpace)."""
        if not self._check_binop_other(other) or isinstance(other, StateSpace):
            return NotImplemented
        if isinstance(other, np.ndarray):
            return self._scale_output(other)
        a = self.A
        b = self.B
        c = np.dot(other, self.C)
        d = np.dot(other, self.D)
        return StateSpace(a, b, c, d, **self._dt_dict)

    def __neg__(self) -> "StateSpace[n_states, n_inputs, n_outputs]":
        """Negate the system (equivalent to pre-multiplying by -1)."""
        return StateSpace(self.A, self.B, -self.C, -self.D, **self._dt_dict)

    # -- parallel connection / static gain ---------------------------------

    def _parallel(
        self, other: "StateSpace[n_s2, n_inputs, n_outputs]"
    ) -> "StateSpace[int, n_inputs, n_outputs]":
        if (other.dt is None) != (self.dt is None):
            raise TypeError(f"Cannot add {type(self)} and {type(other)}")
        if self.dt != other.dt:
            raise TypeError("Cannot add systems with different `dt`.")

        # [x1']   [A1 0 ] [x1]   [B1]
        # [x2'] = [0  A2] [x2] + [B2] u
        #                 [x1]
        #  y    = [C1 C2] [x2] + [D1 + D2] u
        a = _block_diag2(self.A, other.A)
        b = np.vstack((self.B, other.B))
        c = np.hstack((self.C, other.C))
        d = self.D + other.D
        return StateSpace(a, b, c, d, **self._dt_dict)

    def _add_static(
        self, other: F64["n_outputs n_inputs"] | float
    ) -> "StateSpace[n_states, n_inputs, n_outputs]":
        other2d = np.atleast_2d(other)
        if self.D.shape == other2d.shape:
            # A scalar/matrix is really just a static system (A=0, B=0, C=0)
            a = self.A
            b = self.B
            c = self.C
            d = self.D + other2d
        else:
            raise ValueError("Cannot add systems with incompatible "
                             f"dimensions ({self.D.shape} and {other2d.shape})")
        return StateSpace(a, b, c, d, **self._dt_dict)

    @overload
    def __add__(
        self, other: "StateSpace[n_s2, n_inputs, n_outputs]"
    ) -> "StateSpace[int, n_inputs, n_outputs]": ...
    @overload
    def __add__(
        self, other: F64["n_outputs n_inputs"] | float
    ) -> "StateSpace[n_states, n_inputs, n_outputs]": ...
    def __add__(self, other: Any) -> Any:
        """Adds two systems in the sense of frequency domain addition."""
        if not self._check_binop_other(other):
            return NotImplemented
        if isinstance(other, StateSpace):
            return self._parallel(other)
        return self._add_static(other)

    def __sub__(self, other: Any) -> Any:
        if not self._check_binop_other(other):
            return NotImplemented
        return self.__add__(-other)

    def __radd__(self, other: Any) -> Any:
        if not self._check_binop_other(other):
            return NotImplemented
        return self.__add__(other)

    def __rsub__(self, other: Any) -> Any:
        if not self._check_binop_other(other):
            return NotImplemented
        return (-self).__add__(other)

    def __truediv__(self, other: float) -> "StateSpace[n_states, n_inputs, n_outputs]":
        """Divide by a scalar."""
        if not self._check_binop_other(other) or isinstance(other, StateSpace):
            return NotImplemented
        if isinstance(other, np.ndarray) and other.ndim > 0:
            raise ValueError("Cannot divide StateSpace by non-scalar numpy arrays")
        return self._scale_input_scalar(1 / other)


# ---------------------------------------------------------------------------
# Continuous-time simulation
# ---------------------------------------------------------------------------

def lsim(
    system: StateSpace[n_states, n_inputs, n_outputs],
    U: F64["n_steps n_inputs"] | F64["n_steps"] | float | None,
    T: F64["n_steps"],
    X0: F64["n_states"] | None = None,
    interp: bool = True,
) -> tuple[F64["n_steps"], np.ndarray[Any, np.dtype[np.float64]],
           np.ndarray[Any, np.dtype[np.float64]]]:
    """Simulate output of a continuous-time linear system."""
    sys = system
    T = atleast_1d(T)
    if len(T.shape) != 1:
        raise ValueError("T must be a rank-1 array.")

    # upstream: map(np.asarray, (sys.A, sys.B, sys.C, sys.D)); the display
    # keeps each matrix's own shape (map joins them to (n_states, int)).
    A, B, C, D = (np.asarray(sys.A), np.asarray(sys.B), np.asarray(sys.C),
                  np.asarray(sys.D))
    n_states = A.shape[0]
    n_inputs = B.shape[1]

    n_steps = T.size
    if X0 is None:
        X0 = zeros(n_states, sys.A.dtype)
    xout = np.empty((n_steps, n_states), sys.A.dtype)

    if T[0] == 0:
        xout[0] = X0
    elif T[0] > 0:
        # step forward to initial time, with zero input
        xout[0] = dot(X0, expm(transpose(A) * T[0]))
    else:
        raise ValueError("Initial time must be nonnegative")

    no_input = (U is None or
                (isinstance(U, int | float) and U == 0.) or
                not np.any(U))

    if n_steps == 1:
        yout = squeeze(xout @ C.T)
        if not no_input:
            assert isinstance(U, np.ndarray)  # a scalar U here is 0 (no input)
            yout += squeeze(U @ D.T)
        return T, yout, squeeze(xout)

    dt = T[1] - T[0]
    if not np.allclose(np.diff(T), dt):
        raise ValueError("Time steps are not equally spaced.")

    if no_input:
        # Zero input: just use matrix exponential
        # take transpose because state is a row vector
        expAT_dt = expm(A.T * dt)
        for i in range(1, n_steps):
            xout[i] = xout[i-1] @ expAT_dt
        yout = squeeze(xout @ C.T)
        return T, yout, squeeze(xout)

    # Nonzero input
    assert U is not None  # None is no_input
    U_in = atleast_1d(U)
    # atleast_1d of the union is ndarray[Any, Any] in numty, so this is
    # accepted; the `ndim == 1` test could not narrow the union anyway.
    U2: F64["n_steps n_inputs"] = (
        U_in[:, np.newaxis] if U_in.ndim == 1 else U_in)

    if U2.shape[0] != n_steps:
        raise ValueError("U must have the same number of rows "
                         "as elements in T.")

    if U2.shape[1] != n_inputs:
        raise ValueError("System does not define that many inputs.")

    Ad: F64["n_states n_states"]
    if not interp:
        # Zero-order hold
        #   [ x(dt) ]       [ A*dt   B*dt ] [ x0 ]
        #   [ u(dt) ] = exp [  0     0    ] [ u0 ]
        M = np.vstack([np.hstack([A * dt, B * dt]),
                       np.zeros((n_inputs, n_states + n_inputs))])
        # transpose everything because the state and input are row vectors
        expMT = expm(M.T)
        Ad = expMT[:n_states, :n_states]  # type: ignore[assignment]  # NUMTY: a slice of an n_states+n_inputs axis bounded by n_states is ?U
        Bd: F64["n_inputs n_states"] = expMT[n_states:, :n_states]  # type: ignore[assignment]  # NUMTY: a slice of an n_states+n_inputs axis bounded by n_states is ?U
        for i in range(1, n_steps):
            xout[i] = xout[i-1] @ Ad + U2[i-1] @ Bd
    else:
        # Linear interpolation between steps
        #   [ x(dt) ]       [ A*dt  B*dt  0 ] [  x0   ]
        #   [ u(dt) ] = exp [  0     0    I ] [  u0   ]
        #   [u1 - u0]       [  0     0    0 ] [u1 - u0]
        M2 = np.vstack([np.hstack([A * dt, B * dt,
                                  np.zeros((n_states, n_inputs))]),
                       np.hstack([np.zeros((n_inputs, n_states + n_inputs)),
                                  np.identity(n_inputs)]),
                       np.zeros((n_inputs, n_states + 2 * n_inputs))])
        expMT2 = expm(M2.T)
        Ad = expMT2[:n_states, :n_states]  # type: ignore[assignment]  # NUMTY: a slice of an n_states+n_inputs axis bounded by n_states is ?U
        Bd1: F64["n_inputs n_states"] = expMT2[n_states+n_inputs:, :n_states]  # type: ignore[assignment]  # NUMTY: a slice of an n_states+n_inputs axis bounded by n_states is ?U
        Bd0: F64["n_inputs n_states"] = expMT2[n_states:n_states + n_inputs, :n_states] - Bd1  # type: ignore[assignment]  # NUMTY: a slice of an n_states+n_inputs axis bounded by n_states is ?U
        for i in range(1, n_steps):
            xout[i] = xout[i-1] @ Ad + U2[i-1] @ Bd0 + U2[i] @ Bd1

    yout = squeeze(xout @ C.T) + squeeze(U2 @ D.T)
    return T, yout, squeeze(xout)


def _default_response_times(A: F64["n_states n_states"], n: int) -> F64[int]:
    """Compute a reasonable set of time samples for the response time."""
    vals = np.linalg.eigvals(A)
    r = min(abs(real(vals)))
    if r == 0.0:
        r = 1.0
    tc = 1.0 / r
    t = linspace(0.0, 7 * tc, n)
    return t


def impulse(
    system: StateSpace[n_states, Literal[1], n_outputs],
    X0: F64["n_states 1"] | float | None = None,
    T: F64[int] | None = None,
    N: int | None = None,
) -> tuple[F64[int], np.ndarray[Any, np.dtype[np.float64]]]:
    """Impulse response of a continuous-time single-input system."""
    sys = system
    if X0 is None:
        X = squeeze(sys.B)
    else:
        X = squeeze(sys.B + X0)
    if N is None:
        N = 100
    if T is None:
        T = _default_response_times(sys.A, N)
    else:
        T = np.asarray(T)

    _, h, _ = lsim(sys, 0., T, X, interp=False)
    return T, h


def step(
    system: StateSpace[n_states, Literal[1], n_outputs],
    X0: F64["n_states"] | None = None,
    T: F64[int] | None = None,
    N: int | None = None,
) -> tuple[F64[int], np.ndarray[Any, np.dtype[np.float64]]]:
    """Step response of a continuous-time single-input system."""
    sys = system
    if N is None:
        N = 100
    if T is None:
        T = _default_response_times(sys.A, N)
    else:
        T = np.asarray(T)
    U = ones(T.shape, sys.A.dtype)
    vals = lsim(sys, U, T, X0=X0, interp=False)
    return vals[0], vals[1]


# ---------------------------------------------------------------------------
# Pole placement
# ---------------------------------------------------------------------------

@dataclass
class PlacePolesResult(Generic[n_inputs, n_states, PT]):
    """The ``Bunch`` place_poles returns."""

    gain_matrix: F64["n_inputs n_states"]
    computed_poles: Arr[Inexact, "n_states"]
    requested_poles: Arr[PT, "n_states"]
    X: Arr[Inexact, "n_states n_states"]
    rtol: float
    nb_iter: float


def _valid_inputs(
    A: F64["n_states n_states"],
    B: F64["n_states n_inputs"],
    poles: Arr[PT, "n_states"],
    method: str,
    rtol: float,
    maxiter: int,
) -> tuple[UpdateLoop, Arr[PT, "n_states"]]:
    """Check the poles come from a valid set and return the update loop."""
    poles = np.asarray(poles)
    if poles.ndim > 1:
        raise ValueError("Poles must be a 1D array like.")
    # Will raise ValueError if poles do not come in complex conjugates pairs
    poles = _order_complex_poles(poles)
    if A.ndim > 2:
        raise ValueError("A must be a 2D array/matrix.")
    if B.ndim > 2:
        raise ValueError("B must be a 2D array/matrix")
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if len(poles) > A.shape[0]:
        raise ValueError(
            f"maximum number of poles is {A.shape[0]} but you asked for {len(poles)}"
        )
    if len(poles) < A.shape[0]:
        raise ValueError(
            f"number of poles is {len(poles)} but you should provide {A.shape[0]}"
        )
    r = np.linalg.matrix_rank(B)
    for p in poles:
        if sum(p == poles) > r:
            raise ValueError("at least one of the requested pole is repeated "
                             "more than rank(B) times")
    # Choose update method
    update_loop: UpdateLoop = _YT_loop
    if method not in ('KNV0', 'YT'):
        raise ValueError("The method keyword must be one of 'YT' or 'KNV0'")

    if method == "KNV0":
        update_loop = _KNV0_loop
        if not all(np.isreal(poles)):
            raise ValueError("Complex poles are not supported by KNV0")

    if maxiter < 1:
        raise ValueError("maxiter must be at least equal to 1")

    # We do not check rtol <= 0 as the user can use a negative rtol to
    # force maxiter iterations
    if rtol > 1:
        raise ValueError("rtol can not be greater than 1")

    return update_loop, poles


def _order_complex_poles(poles: Arr[PT, "n_poles"]) -> Arr[PT, "n_poles"]:
    """Real poles first (sorted), then each complex pole and its conjugate."""
    ordered_poles = np.sort(poles[np.isreal(poles)])
    im_poles: list[Any] = []
    for p in np.sort(poles[np.imag(poles) < 0]):
        if np.conj(p) in poles:
            im_poles.extend((p, np.conj(p)))

    ordered_poles = np.hstack((ordered_poles, im_poles))

    if poles.shape[0] != len(ordered_poles):
        raise ValueError("Complex poles must come with their conjugates")
    return ordered_poles


def _KNV0(
    B: F64["n_states n_inputs"],
    ker_pole: list[Arr[Inexact, "n_states n_ker"]],
    transfer_matrix: Arr[Inexact, "n_states n_states"],
    j: int,
    poles: Arr[Inexact, "n_states"],
) -> None:
    """KNV update method 0 for column j of the transfer matrix (in place)."""
    # Remove xj form the base
    transfer_matrix_not_j = np.delete(transfer_matrix, j, axis=1)
    # If we QR this matrix in full mode Q=Q0|Q1
    # then Q1 will be a single column orthogonal to
    # Q0, that's what we are looking for !
    Q, R = np.linalg.qr(transfer_matrix_not_j, mode="complete")

    mat_ker_pj = np.dot(ker_pole[j], ker_pole[j].T)
    yj = np.dot(mat_ker_pj, Q[:, -1])

    # If Q[:, -1] is "almost" orthogonal to ker_pole[j] its
    # projection into ker_pole[j] will yield a vector
    # close to 0.  As we are looking for a vector in ker_pole[j]
    # simply stick with transfer_matrix[:, j]
    if not np.allclose(yj, 0):
        xj = yj/np.linalg.norm(yj)
        transfer_matrix[:, j] = xj


def _YT_real(
    ker_pole: list[Arr[Inexact, "n_states n_ker"]],
    Q: Arr[Inexact, "n_states n_states"],
    transfer_matrix: Arr[Inexact, "n_states n_states"],
    i: int,
    j: int,
) -> None:
    """Applies algorithm from YT section 6.1 page 19 related to real pairs."""
    # step 1 page 19
    u = Q[:, -2, np.newaxis]
    v = Q[:, -1, np.newaxis]

    # step 2 page 19
    m = np.dot(np.dot(ker_pole[i].T, np.dot(u, v.T) -
        np.dot(v, u.T)), ker_pole[j])

    # step 3 page 19
    um, sm, vm = np.linalg.svd(m)
    # mu1, mu2 two first columns of U => 2 first lines of U.T
    mu1, mu2 = um.T[:2, :, np.newaxis]
    # VM is V.T with numpy we want the first two lines of V.T
    nu1, nu2 = vm[:2, :, np.newaxis]

    # what follows is a rough python translation of the formulas
    # in section 6.2 page 20 (step 4)
    transfer_matrix_j_mo_transfer_matrix_j = np.vstack((
            transfer_matrix[:, i, np.newaxis],
            transfer_matrix[:, j, np.newaxis]))

    if not np.allclose(sm[0], sm[1]):
        ker_pole_imo_mu1 = np.dot(ker_pole[i], mu1)
        ker_pole_i_nu1 = np.dot(ker_pole[j], nu1)
        ker_pole_mu_nu = np.vstack((ker_pole_imo_mu1, ker_pole_i_nu1))
    else:
        ker_pole_ij = np.vstack((
                                np.hstack((ker_pole[i],
                                           np.zeros(ker_pole[i].shape))),
                                np.hstack((np.zeros(ker_pole[j].shape),
                                                    ker_pole[j]))
                                ))
        mu_nu_matrix = np.vstack(
            (np.hstack((mu1, mu2)), np.hstack((nu1, nu2)))
            )
        ker_pole_mu_nu = np.dot(ker_pole_ij, mu_nu_matrix)
    transfer_matrix_ij = np.dot(np.dot(ker_pole_mu_nu, ker_pole_mu_nu.T),
                             transfer_matrix_j_mo_transfer_matrix_j)
    if not np.allclose(transfer_matrix_ij, 0):
        transfer_matrix_ij = (np.sqrt(2)*transfer_matrix_ij /
                              np.linalg.norm(transfer_matrix_ij))
        transfer_matrix[:, i] = transfer_matrix_ij[
            :transfer_matrix[:, i].shape[0], 0
            ]
        transfer_matrix[:, j] = transfer_matrix_ij[
            transfer_matrix[:, i].shape[0]:, 0
            ]
    else:
        # As in knv0 if transfer_matrix_j_mo_transfer_matrix_j is orthogonal
        # to Vect{ker_pole_mu_nu} assign transfer_matrixi/transfer_matrix_j
        # to ker_pole_mu_nu and iterate.
        transfer_matrix[:, i] = ker_pole_mu_nu[
            :transfer_matrix[:, i].shape[0], 0
            ]
        transfer_matrix[:, j] = ker_pole_mu_nu[
            transfer_matrix[:, i].shape[0]:, 0
            ]


def _YT_complex(
    ker_pole: list[Arr[Inexact, "n_states n_ker"]],
    Q: Arr[Inexact, "n_states n_states"],
    transfer_matrix: Arr[Inexact, "n_states n_states"],
    i: int,
    j: int,
) -> None:
    """Applies algorithm from YT section 6.2 page 20 related to complex pairs."""
    # step 1 page 20
    ur = np.sqrt(2)*Q[:, -2, np.newaxis]
    ui = np.sqrt(2)*Q[:, -1, np.newaxis]
    u = ur + 1j*ui

    # step 2 page 20
    ker_pole_ij = ker_pole[i]
    m = np.dot(np.dot(np.conj(ker_pole_ij.T), np.dot(u, np.conj(u).T) -
               np.dot(np.conj(u), u.T)), ker_pole_ij)

    # step 3 page 20
    e_val, e_vec = np.linalg.eig(m)
    # sort eigenvalues according to their module
    e_val_idx = np.argsort(np.abs(e_val))
    mu1 = e_vec[:, e_val_idx[-1], np.newaxis]
    mu2 = e_vec[:, e_val_idx[-2], np.newaxis]

    # remember transfer_matrix_i has been split as
    # transfer_matrix[i]=real(transfer_matrix_i) and
    # transfer_matrix[j]=imag(transfer_matrix_i)
    transfer_matrix_j_mo_transfer_matrix_j = (
        transfer_matrix[:, i, np.newaxis] +
        1j*transfer_matrix[:, j, np.newaxis]
        )
    if not np.allclose(np.abs(e_val[e_val_idx[-1]]),
                              np.abs(e_val[e_val_idx[-2]])):
        ker_pole_mu = np.dot(ker_pole_ij, mu1)
    else:
        mu1_mu2_matrix = np.hstack((mu1, mu2))
        ker_pole_mu = np.dot(ker_pole_ij, mu1_mu2_matrix)
    transfer_matrix_i_j = np.dot(np.dot(ker_pole_mu, np.conj(ker_pole_mu.T)),
                              transfer_matrix_j_mo_transfer_matrix_j)

    if not np.allclose(transfer_matrix_i_j, 0):
        transfer_matrix_i_j = (transfer_matrix_i_j /
            np.linalg.norm(transfer_matrix_i_j))
        transfer_matrix[:, i] = np.real(transfer_matrix_i_j[:, 0])
        transfer_matrix[:, j] = np.imag(transfer_matrix_i_j[:, 0])
    else:
        # same idea as in YT_real
        transfer_matrix[:, i] = np.real(ker_pole_mu[:, 0])
        transfer_matrix[:, j] = np.imag(ker_pole_mu[:, 0])


def _YT_loop(
    ker_pole: list[Arr[Inexact, "n_states n_ker"]],
    transfer_matrix: Arr[Inexact, "n_states n_states"],
    poles: Arr[Inexact, "n_states"],
    B: F64["n_states n_inputs"],
    maxiter: int,
    rtol: float,
) -> tuple[bool, float, int]:
    """Algorithm "YT" Tits, Yang. Globally Convergent Algorithms for Robust
    Pole Assignment by State Feedback."""
    # The IEEE edition of the YT paper gives useful information on the
    # optimal update order for the real poles in order to minimize the number
    # of times we have to loop over all poles, see page 1442
    nb_real = poles[np.isreal(poles)].shape[0]
    # hnb => Half Nb Real
    hnb = nb_real // 2

    # Stick to the indices in the paper and then remove one to get numpy array
    # index.  The paper is unclear about what should be done when there is
    # only one real pole => use KNV0 on this real pole seem to work
    update_order: list[list[int]]
    if nb_real > 0:
        #update the biggest real pole with the smallest one
        update_order = [[nb_real], [1]]
    else:
        update_order = [[], []]

    r_comp = np.arange(nb_real+1, len(poles)+1, 2)
    # step 1.a
    r_p = np.arange(1, hnb+nb_real % 2)
    update_order[0].extend(2*r_p)
    update_order[1].extend(2*r_p+1)
    # step 1.b
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp+1)
    # step 1.c
    r_p = np.arange(1, hnb+1)
    update_order[0].extend(2*r_p-1)
    update_order[1].extend(2*r_p)
    # step 1.d
    if hnb == 0 and np.isreal(poles[0]):
        update_order[0].append(1)
        update_order[1].append(1)
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp+1)
    # step 2.a
    r_j = np.arange(2, hnb+nb_real % 2)
    for j in r_j:
        for i in range(1, hnb+1):
            update_order[0].append(i)
            update_order[1].append(i+j)
    # step 2.b
    if hnb == 0 and np.isreal(poles[0]):
        update_order[0].append(1)
        update_order[1].append(1)
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp+1)
    # step 2.c
    r_j = np.arange(2, hnb+nb_real % 2)
    for j in r_j:
        for i in range(hnb+1, nb_real+1):
            idx_1 = i+j
            if idx_1 > nb_real:
                idx_1 = i+j-nb_real
            update_order[0].append(i)
            update_order[1].append(idx_1)
    # step 2.d
    if hnb == 0 and np.isreal(poles[0]):
        update_order[0].append(1)
        update_order[1].append(1)
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp+1)
    # step 3.a
    for i in range(1, hnb+1):
        update_order[0].append(i)
        update_order[1].append(i+hnb)
    # step 3.b
    if hnb == 0 and np.isreal(poles[0]):
        update_order[0].append(1)
        update_order[1].append(1)
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp+1)

    update_order_arr = np.array(update_order).T-1
    stop = False
    nb_try = 0
    cur_rtol = 0.0
    while nb_try < maxiter and not stop:
        det_transfer_matrixb = np.abs(np.linalg.det(transfer_matrix))
        for i, j in update_order_arr:
            if i == j:
                assert i == 0, "i!=0 for KNV call in YT"
                assert np.isreal(poles[i]), "calling KNV on a complex pole"
                _KNV0(B, ker_pole, transfer_matrix, i, poles)
            else:
                transfer_matrix_not_i_j = np.delete(transfer_matrix, (i, j),
                                                    axis=1)
                Q, _ = np.linalg.qr(transfer_matrix_not_i_j, mode="complete")

                if np.isreal(poles[i]):
                    assert np.isreal(poles[j]), "mixing real and complex " + \
                        "in YT_real" + str(poles)
                    _YT_real(ker_pole, Q, transfer_matrix, i, j)
                else:
                    assert ~np.isreal(poles[i]), "mixing real and complex " + \
                        "in YT_real" + str(poles)
                    _YT_complex(ker_pole, Q, transfer_matrix, i, j)

        det_transfer_matrix = np.max((np.sqrt(np.spacing(1)),
                                  np.abs(np.linalg.det(transfer_matrix))))
        cur_rtol = np.abs(
            (det_transfer_matrix -
             det_transfer_matrixb) /
            det_transfer_matrix)
        if cur_rtol < rtol and det_transfer_matrix > np.sqrt(np.spacing(1)):
            # Convergence test from YT page 21
            stop = True
        nb_try += 1
    return stop, cur_rtol, nb_try


def _KNV0_loop(
    ker_pole: list[Arr[Inexact, "n_states n_ker"]],
    transfer_matrix: Arr[Inexact, "n_states n_states"],
    poles: Arr[Inexact, "n_states"],
    B: F64["n_states n_inputs"],
    maxiter: int,
    rtol: float,
) -> tuple[bool, float, int]:
    """Loop over all poles one by one and apply KNV method 0 algorithm."""
    stop = False
    nb_try = 0
    cur_rtol = 0.0
    while nb_try < maxiter and not stop:
        det_transfer_matrixb = np.abs(np.linalg.det(transfer_matrix))
        for j in range(B.shape[0]):
            _KNV0(B, ker_pole, transfer_matrix, j, poles)

        det_transfer_matrix = np.max((np.sqrt(np.spacing(1)),
                                  np.abs(np.linalg.det(transfer_matrix))))
        cur_rtol = np.abs((det_transfer_matrix - det_transfer_matrixb) /
                       det_transfer_matrix)
        if cur_rtol < rtol and det_transfer_matrix > np.sqrt(np.spacing(1)):
            # Convergence test from YT page 21
            stop = True

        nb_try += 1
    return stop, cur_rtol, nb_try


def place_poles(
    A: F64["n_states n_states"],
    B: F64["n_states n_inputs"],
    poles: Arr[PT, "n_states"],
    method: str = "YT",
    rtol: float = 1e-3,
    maxiter: int = 30,
) -> PlacePolesResult[n_inputs, n_states, PT]:
    """Compute K such that eigenvalues (A - dot(B, K))=poles."""
    # Move away all the inputs checking, it only adds noise to the code
    update_loop, poles = _valid_inputs(A, B, poles, method, rtol, maxiter)

    # The current value of the relative tolerance we achieved
    cur_rtol: float = 0
    # The number of iterations needed before converging
    nb_iter: float = 0

    # Step A: QR decomposition of B page 1132 KN
    u, z = np.linalg.qr(B, mode="complete")
    rankB = np.linalg.matrix_rank(B)
    u0 = u[:, :rankB]
    u1 = u[:, rankB:]
    z_r = z[:rankB, :]

    transfer_matrix: Arr[Inexact, "n_states n_states"]
    # If we can use the identity matrix as X the solution is obvious
    if B.shape[0] == rankB:
        # if B is square and full rank there is only one solution
        # such as (A+BK)=inv(X)*diag(P)*X with X=eye(A.shape[0])
        # i.e K=inv(B)*(diag(P)-A)
        # if B has as many lines as its rank (but not square) there are many
        # solutions and we can choose one using least squares
        # => use lstsq in both cases.
        #
        # for complex poles we use the following trick
        #
        # |a -b| has for eigenvalues a+b and a-b
        # |b a|
        #
        # |a+bi 0| has the obvious eigenvalues a+bi and a-bi
        # |0 a-bi|
        diag_poles = np.zeros(A.shape)
        idx = 0
        while idx < poles.shape[0]:
            p = poles[idx]
            diag_poles[idx, idx] = np.real(p)
            if ~np.isreal(p):
                diag_poles[idx, idx+1] = -np.imag(p)
                diag_poles[idx+1, idx+1] = np.real(p)
                diag_poles[idx+1, idx] = np.imag(p)
                idx += 1  # skip next one
            idx += 1
        gain_matrix = np.linalg.lstsq(B, diag_poles-A, rcond=-1)[0]
        transfer_matrix = np.eye(A.shape[0])
        cur_rtol = np.nan
        nb_iter = np.nan
    else:
        # step A (p1144 KNV) and beginning of step F: decompose
        # dot(U1.T, A-P[i]*I).T and build our set of transfer_matrix vectors
        # in the same loop
        ker_pole: list[Arr[Inexact, "n_states n_ker"]] = []

        # flag to skip the conjugate of a complex pole
        skip_conjugate = False
        # select orthonormal base ker_pole for each Pole and vectors for
        # transfer_matrix
        for j in range(B.shape[0]):
            if skip_conjugate:
                skip_conjugate = False
                continue
            pole_space_j = np.dot(u1.T, A-poles[j]*np.eye(B.shape[0])).T

            # after QR Q=Q0|Q1
            # only Q0 is used to reconstruct  the qr'ed (dot Q, R) matrix.
            # Q1 is orthogonal to Q0 and will be multiplied by the zeros in
            # R when using mode "complete".
            Q, _ = np.linalg.qr(pole_space_j, mode="complete")

            ker_pole_j = Q[:, pole_space_j.shape[1]:]

            # Summing all vectors in ker_pole_j gives a valid column of the
            # transfer matrix with no zero lines (see upstream comment).
            transfer_matrix_j = np.sum(ker_pole_j, axis=1)[:, np.newaxis]
            transfer_matrix_j = (transfer_matrix_j /
                                 np.linalg.norm(transfer_matrix_j))
            if ~np.isreal(poles[j]):  # complex pole
                transfer_matrix_j = np.hstack([np.real(transfer_matrix_j),
                                               np.imag(transfer_matrix_j)])
                ker_pole.extend([ker_pole_j, ker_pole_j])

                # Skip next pole as it is the conjugate
                skip_conjugate = True
            else:  # real pole, nothing to do
                ker_pole.append(ker_pole_j)

            if j == 0:
                transfer_matrix = transfer_matrix_j
            else:
                transfer_matrix = np.hstack((transfer_matrix, transfer_matrix_j))

        if rankB > 1:  # otherwise there is nothing we can optimize
            stop, cur_rtol, nb_iter = update_loop(ker_pole, transfer_matrix,
                                                  poles, B, maxiter, rtol)
            if not stop and rtol > 0:
                # if rtol<=0 the user has probably done that on purpose,
                # don't annoy them
                err_msg = (
                    "Convergence was not reached after maxiter iterations.\n"
                    f"You asked for a tolerance of {rtol}, we got {cur_rtol}."
                    )
                warnings.warn(err_msg, stacklevel=2)

        # reconstruct transfer_matrix to match complex conjugate pairs,
        # ie transfer_matrix_j/transfer_matrix_j+1 are
        # Re(Complex_pole), Im(Complex_pole) now and will be Re-Im/Re+Im after
        transfer_matrix = transfer_matrix.astype(complex)
        idx = 0
        while idx < poles.shape[0]-1:
            if ~np.isreal(poles[idx]):
                rel = transfer_matrix[:, idx].copy()
                img = transfer_matrix[:, idx+1]
                # rel will be an array referencing a column of transfer_matrix
                # if we don't copy() it will changer after the next line and
                # and the line after will not yield the correct value
                transfer_matrix[:, idx] = rel-1j*img
                transfer_matrix[:, idx+1] = rel+1j*img
                idx += 1  # skip next one
            idx += 1

        try:
            m = np.linalg.solve(transfer_matrix.T, np.dot(np.diag(poles),
                                                          transfer_matrix.T))
            gain_matrix = np.linalg.solve(z_r, np.dot(u0.T, m-A))
        except np.linalg.LinAlgError as e:
            raise ValueError("The poles you've chosen can't be placed. "
                             "Check the controllability matrix and try "
                             "another set of poles") from e

    # Beware: Kautsky solves A+BK but the usual form is A-BK
    gain_matrix = -gain_matrix
    # K still contains complex with ~=0j imaginary parts, get rid of them
    gain_matrix = np.real(gain_matrix)

    return PlacePolesResult(
        gain_matrix=gain_matrix,
        computed_poles=_order_complex_poles(
            np.linalg.eig(A - np.dot(B, gain_matrix))[0]
        ),
        requested_poles=poles,
        X=transfer_matrix,
        rtol=cur_rtol,
        nb_iter=nb_iter,
    )


# ---------------------------------------------------------------------------
# Discrete-time simulation
# ---------------------------------------------------------------------------

def dlsim(
    system: StateSpace[n_states, n_inputs, n_outputs],
    u: F64["n_samples n_inputs"] | F64["n_samples"],
    t: F64["n_samples"] | None = None,
    x0: F64["n_states"] | None = None,
) -> tuple[F64["n_out"], F64["n_out n_outputs"], F64["n_out n_states"]]:
    """Simulate output of a discrete-time linear system (state-space input)."""
    if system.dt is None:
        raise AttributeError('dlsim can only be used with discrete-time dlti '
                             'systems.')
    dt = system.dt

    u_in = np.atleast_1d(u)

    # atleast_1d of the union is ndarray[Any, Any] in numty (see NOTES.md)
    u2: F64["n_samples n_inputs"] = u_in
    if u_in.ndim == 1:
        u2 = np.atleast_2d(u_in).T

    if t is None:
        out_samples: int = len(u2)
        stoptime = (out_samples - 1) * dt
    else:
        stoptime = t[-1]
        out_samples = int(np.floor(stoptime / dt)) + 1

    # Pre-build output arrays
    xout = np.zeros((out_samples, system.A.shape[0]))
    yout = np.zeros((out_samples, system.C.shape[0]))
    tout = np.linspace(0.0, stoptime, num=out_samples)

    # Check initial condition
    if x0 is None:
        xout[0, :] = np.zeros((system.A.shape[1],))
    else:
        xout[0, :] = np.asarray(x0)

    # Pre-interpolate inputs into the desired time steps
    u_dt: F64[int, n_inputs]
    if t is None:
        u_dt = u2
    else:
        if len(u2.shape) == 1:
            u2 = u2[:, np.newaxis]

        u_dt = _interp_linear(t, u2, tout)

    # Simulate the system
    for i in range(0, out_samples - 1):
        xout[i+1, :] = (np.dot(system.A, xout[i, :]) +
                        np.dot(system.B, u_dt[i, :]))
        yout[i, :] = (np.dot(system.C, xout[i, :]) +
                      np.dot(system.D, u_dt[i, :]))

    # Last point
    yout[out_samples-1, :] = (np.dot(system.C, xout[out_samples-1, :]) +
                              np.dot(system.D, u_dt[out_samples-1, :]))

    return tout, yout, xout
