# MUTANT: impulse: cd = c @ ad instead of c (domain mistake, same (p, n) shape)
"""scipy.signal LTI conversions, extracted and annotated for the numty bug hunt.

Upstream: scipy v1.18.1 (commit e4e854eaa8f18d807cd3496028e257e36caa93cc)

- scipy/signal/_lti_conversion.py: tf2ss (21-115), abcd_normalize (118-230),
  ss2tf (233-331), zpk2ss (334-351), ss2zpk (354-381), cont2discrete (384-583)
- scipy/signal/_filter_design.py: tf2zpk (1158-1242), zpk2tf (1245-1318),
  normalize (1847-1936)
- scipy/signal/_polyutils.py: _trim_zeros (56-74, 'f' only), poly (140-166)
- scipy.linalg.expm and scipy.linalg.block_diag (two blocks) are re-implemented
  below with numpy (Pade 13 scaling and squaring; zero-filled block layout).

The array-API ``xp.`` calls are translated to numpy. The typed API takes
float64 arrays (2-D state-space matrices, 1-D or 2-D numerators, 1-D
denominators, real zeros and poles) where upstream takes any array_like; for
those inputs the runtime behaviour is that of the numpy backend upstream.
Sizes: ``n_states`` (n), ``n_inputs`` (upstream's p in abcd_normalize, m in
cont2discrete), ``n_outputs`` (upstream's q).
"""

from __future__ import annotations

import warnings
from typing import Any, TypeVar, overload

import numpy as np

from numty import F64, Arr

n_states = TypeVar("n_states", bound=int)
n_inputs = TypeVar("n_inputs", bound=int)
n_outputs = TypeVar("n_outputs", bound=int)
n_rows = TypeVar("n_rows", bound=int)
n_b = TypeVar("n_b", bound=int)
n_a = TypeVar("n_a", bound=int)
n_num = TypeVar("n_num", bound=int)
n_den = TypeVar("n_den", bound=int)
n_coef = TypeVar("n_coef", bound=int)
n_trim = TypeVar("n_trim", bound=int)
n_seq = TypeVar("n_seq", bound=int)
n_zeros = TypeVar("n_zeros", bound=int)
n_poles = TypeVar("n_poles", bound=int)
n_sq = TypeVar("n_sq", bound=int)

# Roots of a real polynomial: float64 when every root is real, else complex.
Roots = Arr[np.inexact[Any], n_zeros]


class BadCoefficients(UserWarning):
    """Warning about badly conditioned filter coefficients"""


# ---------------------------------------------------------------- helpers


def _trim_zeros_f(filt: F64[n_coef]) -> F64[n_trim]:
    """Leading zeros removed (scipy/signal/_polyutils.py::_trim_zeros, 'f')."""
    first = 0
    for i in filt:
        if i != 0.0:
            break
        else:
            first = first + 1
    last = filt.shape[0]
    return filt[first:last]


def _poly(seq_of_zeros: F64[n_seq]) -> F64["n_seq+1"] | F64[()]:
    """Coefficients of the monic polynomial with the given real roots.

    scipy/signal/_polyutils.py::poly (1-D only); an empty sequence gives 1.0.
    """
    if seq_of_zeros.shape[0] == 0:
        return np.asarray(1.0, dtype=np.float64)
    a: F64[int] = np.ones((1,), dtype=seq_of_zeros.dtype)
    one = np.ones_like(seq_of_zeros[0])
    for zero in seq_of_zeros:
        a = np.convolve(a, np.stack((one, -zero)), mode="full")
    return a  # type: ignore[return-value]  # NUMTY: a size grown in a loop has no name; the F64[int] accumulator is refused where n_seq+1 is declared


def _block_diag2(a: F64["r1 c1"], b: F64["r2 c2"]) -> F64["r1+r2 c1+c2"]:
    """scipy.linalg.block_diag for two 2-D blocks."""
    r1, c1 = a.shape
    r2, c2 = b.shape
    out = np.zeros((r1 + r2, c1 + c2), dtype=np.result_type(a, b))
    out[:r1, :c1] = a
    out[r1:, c1:] = b
    return out


_PADE13 = (
    64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
    1187353796428800.0, 129060195264000.0, 10559470521600.0,
    670442572800.0, 33522128640.0, 1323241920.0, 40840800.0,
    960960.0, 16380.0, 182.0, 1.0,
)
_THETA13 = 5.371920351148152


def _expm(a: F64[n_sq, n_sq]) -> F64[n_sq, n_sq]:
    """Matrix exponential (stand-in for scipy.linalg.expm).

    Pade(13) with scaling and squaring, Higham (2005).
    """
    n = a.shape[0]
    if n == 0:
        return a.copy()
    norm1 = float(np.linalg.norm(a, 1))
    s = 0
    if norm1 > _THETA13:
        s = max(0, int(np.ceil(np.log2(norm1 / _THETA13))))
    a_s = a / 2.0**s
    ident = np.eye(n)
    b = _PADE13
    a2 = a_s @ a_s
    a4 = a2 @ a2
    a6 = a2 @ a4
    u = a_s @ (a6 @ (b[13] * a6 + b[11] * a4 + b[9] * a2)
               + b[7] * a6 + b[5] * a4 + b[3] * a2 + b[1] * ident)
    v = (a6 @ (b[12] * a6 + b[10] * a4 + b[8] * a2)
         + b[6] * a6 + b[4] * a4 + b[2] * a2 + b[0] * ident)
    r = np.linalg.solve(v - u, v + u)
    for _ in range(s):
        r = r @ r
    return r


# -------------------------------------------- transfer-function helpers


@overload
def normalize(
    b: F64[n_b], a: F64[n_a] | int | list[float]
) -> tuple[F64[n_num], F64[n_den]]: ...
@overload
def normalize(
    b: F64[n_outputs, n_b], a: F64[n_a] | int | list[float]
) -> tuple[F64[n_outputs, n_num] | F64[n_num], F64[n_den]]: ...
def normalize(
    b: F64[n_b] | F64[n_outputs, n_b], a: F64[n_a] | int | list[float]
) -> tuple[F64[n_rows, n_num] | F64[n_num], F64[n_den]]:
    """Normalize numerator/denominator of a continuous-time transfer function.

    The numerator keeps one row per output and is squeezed to 1-D when it
    has a single row; the denominator is 1-D with its leading zeros trimmed.
    """
    den = np.atleast_1d(np.asarray(a))
    num = np.atleast_2d(np.asarray(b))

    if den.ndim != 1:
        raise ValueError("Denominator polynomial must be rank-1 array.")
    if num.ndim > 2:
        raise ValueError("Numerator polynomial must be rank-1 or rank-2 array.")
    if np.all(den == 0):
        raise ValueError("Denominator must have at least on nonzero element.")

    # Trim leading zeros in denominator, leave at least one.
    den_t = _trim_zeros_f(den)

    # Normalize transfer function
    num_n, den_n = num / den_t[0], den_t / den_t[0]

    # Count numerator columns that are all zero
    leading_zeros = 0
    for j in range(num_n.shape[-1]):
        col = num_n[:, j]
        if np.all(np.abs(col) <= 1e-14):
            leading_zeros += 1
        else:
            break

    # Trim leading zeros of numerator
    if leading_zeros > 0:
        warnings.warn("Badly conditioned filter coefficients (numerator): the "
                      "results may be meaningless",
                      BadCoefficients, stacklevel=2)
        # Make sure at least one column remains
        if leading_zeros == num_n.shape[1]:
            leading_zeros -= 1
        num_n = num_n[:, leading_zeros:]

    # Squeeze first dimension if singular
    if num_n.shape[0] == 1:
        return num_n[0, :], den_n

    return num_n, den_n


def tf2zpk(
    b: F64[n_b] | F64[n_outputs, n_b], a: F64[n_a] | int | list[float]
) -> tuple[Roots[n_zeros], Roots[n_poles], np.float64 | F64[n_num]]:
    """Zeros, poles and gain of a transfer function (scipy.signal.tf2zpk)."""
    bn, an = normalize(b, a)
    k = bn[0]
    bn = bn / bn[0]
    z = np.roots(bn)
    p = np.roots(an)
    return z, p, k


@overload
def zpk2tf(
    z: F64[n_zeros], p: F64[n_poles], k: float
) -> tuple[F64["n_zeros+1"], F64["n_poles+1"]]: ...
@overload
def zpk2tf(
    z: F64[n_outputs, n_zeros], p: F64[n_poles], k: float
) -> tuple[F64["n_outputs n_zeros+1"], F64["n_poles+1"]]: ...
def zpk2tf(
    z: F64[n_zeros] | F64[n_outputs, n_zeros], p: F64[n_poles], k: float
) -> tuple[F64["n_zeros+1"] | F64["n_outputs n_zeros+1"], F64["n_poles+1"]]:
    """Polynomial transfer function from zeros, poles and gain.

    Real zeros and poles only (the typed API); a 2-D ``z`` gives one
    numerator row per output.
    """
    k_a = np.asarray(k, dtype=np.result_type(np.real(z), np.real(p), k))
    k_1 = np.atleast_1d(k_a)
    if z.ndim > 1:
        z2 = np.atleast_2d(z)
        b2 = np.empty((z2.shape[0], z2.shape[1] + 1), dtype=np.float64)
        ks: list[np.float64] | F64[int] = k_1
        if k_1.shape[0] == 1:
            ks = [k_1[0]] * z2.shape[0]
        for i in range(z2.shape[0]):
            k_i = np.asarray(ks[i], dtype=np.float64)
            poly_i = np.asarray(_poly(z2[i, ...]), dtype=np.float64)
            b2[i, ...] = k_i * poly_i
        b: F64["n_zeros+1"] | F64["n_outputs n_zeros+1"] = b2
    else:
        b = np.multiply(k_1, _poly(np.atleast_1d(z)))
    a = np.atleast_1d(np.asarray(_poly(p)))
    return b, a


# ------------------------------------------------------- state space


@overload
def tf2ss(
    num: F64[n_b], den: F64[n_a]
) -> tuple[F64[n_states, n_states], F64[n_states, 1], F64[1, n_states], F64[1, 1]]: ...
@overload
def tf2ss(
    num: F64[n_outputs, n_b], den: F64[n_a]
) -> tuple[
    F64[n_states, n_states], F64[n_states, 1], F64[n_outputs, n_states], F64[n_outputs, 1]
]: ...
def tf2ss(
    num: F64[n_b] | F64[n_outputs, n_b], den: F64[n_a]
) -> (
    tuple[F64[n_states, n_states], F64[n_states, 1], F64[n_rows, n_states], F64[n_rows, 1]]
    | tuple[F64[0], F64[0], F64[0], F64[0]]
):
    """Transfer function to state-space representation (controller canonical form).

    A numerator with no coefficients is the null system, returned as four
    empty 1-D arrays (as upstream does).
    """
    num_n, den_n = normalize(num, den)  # Strips zeros, checks arrays
    num2 = np.atleast_2d(num_n)
    M = num2.shape[1]
    K = len(den_n)
    if M > K:
        msg = "Improper transfer function. `num` is longer than `den`."
        raise ValueError(msg)
    if M == 0 or K == 0:  # Null system
        return (np.array([], float), np.array([], float), np.array([], float),
                np.array([], float))

    # pad numerator to have same number of columns has denominator
    num_p = np.hstack((np.zeros((num2.shape[0], K - M), dtype=num2.dtype), num2))

    if num_p.shape[-1] > 0:
        D = np.atleast_2d(num_p[:, 0])
    else:
        # We don't assign it an empty array because this system
        # is not 'null'. It just doesn't have a non-zero D
        # matrix. Thus, it should have a non-zero shape so that
        # it can be operated on by functions like 'ss2tf'
        D = np.array([[0]], float)

    if K == 1:
        D1 = D.reshape(num_p.shape)

        return (np.zeros((1, 1)), np.zeros((1, D1.shape[1])),  # type: ignore[return-value, misc]  # NUMTY: D.reshape(num_p.shape) is typed 0-d when num_p comes from np.hstack of an untracked shape
                np.zeros((D1.shape[0], 1)), D1)  # type: ignore[misc]  # NUMTY: same 0-d reshape as the line above

    frow = -np.array([den_n[1:]])
    A = np.r_[frow, np.eye(K - 2, K - 1)]
    B = np.eye(K - 1, 1)
    C = num_p[:, 1:] - np.outer(num_p[:, 0], den_n[1:])
    D2 = D.reshape((C.shape[0], B.shape[1]))

    return A, B, C, D2


def abcd_normalize(
    A: F64[n_states, n_states] | None = None,
    B: F64[n_states, n_inputs] | None = None,
    C: F64[n_outputs, n_states] | None = None,
    D: F64[n_outputs, n_inputs] | None = None,
) -> tuple[
    F64[n_states, n_states], F64[n_states, n_inputs],
    F64[n_outputs, n_states], F64[n_outputs, n_inputs],
]:
    """Check state-space matrix compatibility and make them 2-D float arrays.

    A missing (None) or size-zero matrix is replaced by zeros of the shape
    the others imply.
    """
    if A is None and B is None and C is None:
        raise ValueError("Dimension n is undefined for parameters A = B = C = None!")
    if B is None and D is None:
        raise ValueError("Dimension p is undefined for parameters B = D = None!")
    if C is None and D is None:
        raise ValueError("Dimension q is undefined for parameters C = D = None!")

    dtype = np.float64

    # convert inputs into 2d arrays (zero-size 2d array if None):
    A2 = np.atleast_2d(np.asarray(A)) if A is not None else np.zeros((0, 0), dtype=dtype)
    B2 = np.atleast_2d(np.asarray(B)) if B is not None else np.zeros((0, 0), dtype=dtype)
    C2 = np.atleast_2d(np.asarray(C)) if C is not None else np.zeros((0, 0), dtype=dtype)
    D2 = np.atleast_2d(np.asarray(D)) if D is not None else np.zeros((0, 0), dtype=dtype)

    n = A2.shape[0] or B2.shape[0] or C2.shape[1]  # try finding non-zero dimensions
    p = B2.shape[1] or D2.shape[1]
    q = C2.shape[0] or D2.shape[0]

    # Create zero matrices as needed:
    A3: F64[n_states, n_states] = np.zeros((n, n), dtype=dtype) if A2.size == 0 else A2
    B3: F64[n_states, n_inputs] = np.zeros((n, p), dtype=dtype) if B2.size == 0 else B2
    C3: F64[n_outputs, n_states] = np.zeros((q, n), dtype=dtype) if C2.size == 0 else C2
    D3: F64[n_outputs, n_inputs] = np.zeros((q, p), dtype=dtype) if D2.size == 0 else D2

    if A3.shape != (n, n):
        raise ValueError(f"Parameter A has shape {A3.shape} but should be ({n}, {n})!")
    if B3.shape != (n, p):
        raise ValueError(f"Parameter B has shape {B3.shape} but should be ({n}, {p})!")
    if C3.shape != (q, n):
        raise ValueError(f"Parameter C has shape {C3.shape} but should be ({q}, {n})!")
    if D3.shape != (q, p):
        raise ValueError(f"Parameter D has shape {D3.shape} but should be ({q}, {p})!")

    return A3, B3, C3, D3


def ss2tf(
    A: F64[n_states, n_states],
    B: F64[n_states, n_inputs],
    C: F64[n_outputs, n_states],
    D: F64[n_outputs, n_inputs],
    input: int = 0,
) -> tuple[
    F64["n_outputs n_states+1"] | F64[n_outputs], F64["n_states+1"] | int | list[float]
]:
    """State-space to transfer function, for one input.

    One numerator row per output over the common denominator. A system
    with no states is a static gain: its numerator is 1-D and its
    denominator the int 1 (or [] when there are no outputs either).
    """
    # transfer function is C (sI - A)**(-1) B + D

    # Check consistency and make them all floating point 2d arrays:
    A, B, C, D = abcd_normalize(A, B, C, D)

    nout, nin = D.shape
    if input >= nin:
        raise ValueError("System does not have the input specified.")

    # make SIMO from possibly MIMO system.
    B1 = B[:, input:input + 1]
    D1 = D[:, input:input + 1]

    den: F64["n_states+1"] | int
    try:
        den = np.poly(A)  # type: ignore[assignment]  # NUMTY: np.poly is not solved: (n, n) gives (n+1,) float64, the stub says floating of any shape
    except ValueError:
        den = 1

    if (B1.size == 0) and (C.size == 0):
        num_gain = np.ravel(D1)
        if (D1.size == 0) and (A.size == 0):
            return num_gain, []
        return num_gain, den

    num_states = A.shape[0]
    type_test = A[:, 0] + B1[:, 0] + C[0, :] + D1 + 0.0
    num = np.empty((nout, num_states + 1), type_test.dtype)
    for k in range(nout):
        Ck = np.atleast_2d(C[k, :])
        num[k] = np.poly(A - np.dot(B1, Ck)) + (D1[k] - 1) * den

    return num, den


@overload
def zpk2ss(
    z: F64[n_zeros], p: F64[n_poles], k: float
) -> tuple[F64[n_states, n_states], F64[n_states, 1], F64[1, n_states], F64[1, 1]]: ...
@overload
def zpk2ss(
    z: F64[n_outputs, n_zeros], p: F64[n_poles], k: float
) -> tuple[
    F64[n_states, n_states], F64[n_states, 1], F64[n_outputs, n_states], F64[n_outputs, 1]
]: ...
def zpk2ss(
    z: F64[n_zeros] | F64[n_outputs, n_zeros], p: F64[n_poles], k: float
) -> (
    tuple[F64[n_states, n_states], F64[n_states, 1], F64[n_rows, n_states], F64[n_rows, 1]]
    | tuple[F64[0], F64[0], F64[0], F64[0]]
):
    """Zero-pole-gain representation to state-space representation."""
    return tf2ss(*zpk2tf(z, p, k))


def ss2zpk(
    A: F64[n_states, n_states],
    B: F64[n_states, n_inputs],
    C: F64[n_outputs, n_states],
    D: F64[n_outputs, n_inputs],
    input: int = 0,
) -> tuple[Roots[n_zeros], Roots[n_poles], np.float64 | F64[n_num]]:
    """State-space representation to zero-pole-gain representation."""
    return tf2zpk(*ss2tf(A, B, C, D, input=input))  # type: ignore[return-value]  # NUMTY: forwarding a call's tuple with chosen sizes (and a union gain) is refused


SS = tuple[
    F64[n_states, n_states], F64[n_states, n_inputs],
    F64[n_outputs, n_states], F64[n_outputs, n_inputs],
]


@overload
def cont2discrete(
    system: tuple[F64[n_b] | F64[n_outputs, n_b], F64[n_a]],
    dt: float, method: str = ..., alpha: float | None = ...,
) -> tuple[F64[n_rows, n_coef] | F64[n_rows], F64[n_coef] | int | list[float], float]: ...
@overload
def cont2discrete(
    system: tuple[F64[n_zeros], F64[n_poles], float],
    dt: float, method: str = ..., alpha: float | None = ...,
) -> tuple[Roots[n_zeros], Roots[n_poles], np.float64 | F64[n_num], float]: ...
@overload
def cont2discrete(
    system: tuple[
        F64[n_states, n_states], F64[n_states, n_inputs],
        F64[n_outputs, n_states], F64[n_outputs, n_inputs],
    ],
    dt: float, method: str = ..., alpha: float | None = ...,
) -> tuple[
    F64[n_states, n_states], F64[n_states, n_inputs],
    F64[n_outputs, n_states], F64[n_outputs, n_inputs], float,
]: ...
def cont2discrete(
    system: (
        tuple[F64[n_b] | F64[n_outputs, n_b], F64[n_a]]
        | tuple[F64[n_zeros], F64[n_poles], float]
        | tuple[
            F64[n_states, n_states], F64[n_states, n_inputs],
            F64[n_outputs, n_states], F64[n_outputs, n_inputs],
        ]
    ),
    dt: float,
    method: str = "zoh",
    alpha: float | None = None,
) -> (
    tuple[F64[n_rows, n_coef] | F64[n_rows], F64[n_coef] | int | list[float], float]
    | tuple[Roots[n_zeros], Roots[n_poles], np.float64 | F64[n_num], float]
    | tuple[
        F64[n_states, n_states], F64[n_states, n_inputs],
        F64[n_outputs, n_states], F64[n_outputs, n_inputs], float,
    ]
):
    """Transform a continuous to a discrete system (tf, zpk or ss tuple)."""
    if len(system) == 2:
        sysd_tf = cont2discrete(tf2ss(system[0], system[1]), dt, method=method,
                                alpha=alpha)
        return ss2tf(sysd_tf[0], sysd_tf[1], sysd_tf[2], sysd_tf[3]) + (dt,)  # type: ignore[return-value]  # NUMTY: chosen sizes from a call are refused inside union members of the returned tuple
    elif len(system) == 3:
        sysd = cont2discrete(zpk2ss(system[0], system[1], system[2]), dt,
                             method=method, alpha=alpha)
        return ss2zpk(sysd[0], sysd[1], sysd[2], sysd[3]) + (dt,)  # type: ignore[return-value]  # NUMTY: forwarding a call's tuple with chosen sizes (and a union gain) is refused
    elif len(system) == 4:
        a, b, c, d = system
    else:
        raise ValueError("First argument must either be a tuple of 2 (tf), "
                         "3 (zpk), or 4 (ss) arrays.")

    if method == 'gbt':
        if alpha is None:
            raise ValueError("Alpha parameter must be specified for the "
                             "generalized bilinear transform (gbt) method")
        elif alpha < 0 or alpha > 1:
            raise ValueError("Alpha parameter must be within the interval "
                             "[0,1] for the gbt method")

    ad: F64[n_states, n_states]
    bd: F64[n_states, n_inputs]
    cd: F64[n_outputs, n_states]
    dd: F64[n_outputs, n_inputs]

    if method == 'gbt':
        assert alpha is not None
        # This parameter is used repeatedly - compute once here
        ima = np.eye(a.shape[0]) - alpha * dt * a
        ad = np.linalg.solve(ima, np.eye(a.shape[0]) + (1.0 - alpha) * dt * a)
        bd = np.linalg.solve(ima, dt * b)

        # Similarly solve for the output equation matrices
        cd_t = np.linalg.solve(ima.transpose(), c.transpose())
        cd = cd_t.transpose()
        dd = d + alpha * np.dot(c, bd)

    elif method == 'bilinear' or method == 'tustin':
        return cont2discrete(system, dt, method="gbt", alpha=0.5)

    elif method == 'euler' or method == 'forward_diff':
        return cont2discrete(system, dt, method="gbt", alpha=0.0)

    elif method == 'backward_diff':
        return cont2discrete(system, dt, method="gbt", alpha=1.0)

    elif method == 'zoh':
        # Build an exponential matrix
        em_upper = np.hstack((a, b))

        # Need to stack zeros under the a and b matrices
        em_lower = np.hstack((np.zeros((b.shape[1], a.shape[0])),
                              np.zeros((b.shape[1], b.shape[1]))))

        em = np.vstack((em_upper, em_lower))
        ms_full = _expm(dt * em)

        # Dispose of the lower rows
        ms = ms_full[:a.shape[0], :]

        ad = ms[:, 0:a.shape[1]]  # type: ignore[assignment]  # NUMTY: a slice bounded by a.shape[k] on an n+m axis is ?U, not n
        bd = ms[:, a.shape[1]:]  # type: ignore[assignment]  # NUMTY: a slice from a.shape[1] on an n+m axis is ?U, not m

        cd = c
        dd = d

    elif method == 'foh':
        # Size parameters for convenience
        n = a.shape[0]
        m = b.shape[1]

        # Build an exponential matrix similar to 'zoh' method
        em_upper_f = _block_diag2(np.block([a, b]) * dt, np.eye(m))
        em_lower_f = np.zeros((m, n + 2 * m))
        em_f = np.block([[em_upper_f], [em_lower_f]])

        ms_f = _expm(em_f)

        # Get the three blocks from upper rows
        ms11 = ms_f[:n, 0:n]
        ms12 = ms_f[:n, n:n + m]
        ms13 = ms_f[:n, n + m:]

        ad = ms11  # type: ignore[assignment]  # NUMTY: np.block is not solved, and ms[:n, 0:n] on an n+2m axis is ?U
        bd = ms12 - ms13 + ms11 @ ms13  # type: ignore[assignment]  # NUMTY: np.block is not solved, and ms[:n, n:n + m] is ?U
        cd = c
        dd = d + c @ ms13  # type: ignore[assignment]  # NUMTY: np.block is not solved, and ms[:n, n + m:] is ?U

    elif method == 'impulse':
        if not np.allclose(d, 0):
            raise ValueError("Impulse method is only applicable "
                             "to strictly proper systems")

        ad = _expm(a * dt)
        bd = ad @ b * dt
        cd = c @ ad
        dd = c @ b * dt

    else:
        raise ValueError(f"Unknown transformation method '{method}'")

    return ad, bd, cd, dd, dt
