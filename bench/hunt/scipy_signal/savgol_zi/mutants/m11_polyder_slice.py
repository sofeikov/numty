# MUTANT: _polyder: drops m + 1 trailing coefficients instead of m (p[:-m - 1])
"""Savitzky-Golay filtering and filter initial conditions, extracted from SciPy.

Upstream: scipy v1.18.1 (commit e4e854e), files

- scipy/signal/_savitzky_golay.py: savgol_coeffs (L11-160), _polyder (L163-186),
  _fit_edge (L189-240), _fit_edges_polyfit (L244-258), savgol_filter (L261-391)
- scipy/signal/_polyutils.py: polyval (L125-136), polyfit (L182-220), _lstsq (L79-105)
- scipy/signal/_signaltools.py: lfilter_zi (L4330-4462), sosfilt_zi (L4464-4539),
  filtfilt (L4721-4924, method="pad" only), _validate_pad (L4927-4960),
  _validate_x (L4963-4967), sosfiltfilt (L5091-5203)
- scipy/signal/_filter_design.py: _validate_sos (L824-838)
- scipy/signal/_arraytools.py: odd_ext, even_ext, const_ext (L57-212)
- scipy/_lib/_util.py: float_factorial (not in the sparse clone; read from the installed 1.18.1)

Only numpy, numty and the standard library are imported. Signals are
2-D, ``(n_channels, n_samples)``, and every routine works along the last
axis (scipy's default ``axis=-1``). The array-API ``xp.`` calls are numpy
calls. scipy routines the code needs are written here with numpy:

- ``scipy.linalg.lstsq(a, b, cond=eps)`` is ``np.linalg.lstsq(a, b, rcond=eps)``
  (both LAPACK gelsd with the same cutoff);
- ``scipy.ndimage.convolve1d`` along the last axis is a pad (ndimage's
  'mirror'/'nearest'/'wrap'/'constant' are numpy's 'reflect'/'edge'/'wrap'/
  'constant') followed by a sliding-window dot with the reversed weights;
- ``scipy.signal.lfilter`` / ``sosfilt`` with ``zi`` are a direct-form-II
  transposed loop over samples.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from typing import Literal, TypeVar

from numty import F64

# Sizes spelled as a single word need a declared TypeVar (multi-word specs
# mint their names).
window = TypeVar("window", bound=int)  # samples in a Savitzky-Golay window
n_coef = TypeVar("n_coef", bound=int)  # polyorder + 1 polynomial coefficients
n_b = TypeVar("n_b", bound=int)  # numerator taps
n_a = TypeVar("n_a", bound=int)  # denominator taps
n_zi = TypeVar("n_zi", bound=int)  # filter state: max(n_a, n_b) - 1

_PadMode = Literal["reflect", "edge", "wrap"]
_PAD_MODES: dict[str, _PadMode] = {"mirror": "reflect", "nearest": "edge", "wrap": "wrap"}


def float_factorial(n: int) -> float:
    """Factorial as a float, infinity when it is too large for a double."""
    return float(math.factorial(n)) if n < 171 else np.inf


# --------------------------------------------------------------------------
# Savitzky-Golay coefficients
# --------------------------------------------------------------------------


def savgol_coeffs(
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    pos: float | None = None,
    use: str = "conv",
) -> F64[window]:
    """Coefficients of a 1-D Savitzky-Golay FIR filter, one per window sample."""
    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length.")

    halflen, rem = divmod(window_length, 2)

    if pos is None:
        if rem == 0:
            pos = halflen - 0.5
        else:
            pos = halflen

    if not (0 <= pos < window_length):
        raise ValueError("pos must be nonnegative and less than window_length.")

    if use not in ["conv", "dot"]:
        raise ValueError("`use` must be 'conv' or 'dot'")

    if deriv > polyorder:
        return np.zeros(window_length, dtype=np.float64)

    # Design matrix A: the columns are powers of the integers from -pos to
    # window_length - pos - 1, the rows range over the powers 0..polyorder.
    x: F64[window] = np.arange(-pos, window_length - pos, dtype=np.float64)

    if use == "conv":
        # Reverse so that result can be used in a convolution.
        x = np.flip(x)

    order: F64["n_coef 1"] = np.reshape(
        np.arange(polyorder + 1, dtype=np.float64), (-1, 1)
    )
    A = x**order

    # y determines which order derivative is returned.
    y: F64[n_coef] = np.zeros(polyorder + 1, dtype=np.float64)
    y[deriv] = float_factorial(deriv) / (delta**deriv)

    # Least-squares solution of A @ c = y (scipy.linalg.lstsq, cond=eps).
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=np.finfo(np.float64).eps)
    return coeffs


# --------------------------------------------------------------------------
# Old-style polynomial helpers (scipy/signal/_polyutils.py), 2-D columns
# --------------------------------------------------------------------------


def _polyval(p: F64["n_coef n_cols"], x: F64["n_pts 1"]) -> F64["n_pts n_cols"]:
    """Evaluate the polynomial in each column of `p` at the points `x`."""
    # upstream starts from zeros_like(x), (n_pts, 1), which the first step
    # broadcasts to (n_pts, n_cols); start from that shape instead.
    y = np.zeros((x.shape[0], p.shape[1]), dtype=x.dtype)
    for j in range(p.shape[0]):
        y = y * x + p[j, ...]
    return y


def _polyfit(x: F64[window], y: F64["window n_cols"], deg: int) -> F64["n_coef n_cols"]:
    """Least-squares fit of a degree-`deg` polynomial to each column of `y`."""
    order = int(deg) + 1

    if deg < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")

    rcond = x.shape[0] * np.finfo(x.dtype).eps

    # Least-squares equation for powers of x: lhs = vander(x, order).
    powers: F64[n_coef] = np.flip(np.arange(order, dtype=x.dtype))
    lhs = x[:, None] ** powers[None, :]

    # Scale lhs to improve the condition number and solve.
    scale = np.sqrt(np.sum(lhs * lhs, axis=0))
    lhs /= scale

    c, _, rank, _ = np.linalg.lstsq(lhs, y, rcond=rcond)
    c = (c.T / scale).T  # broadcast scale coefficients

    if rank != order:
        warnings.warn("Polyfit may be poorly conditioned", np.exceptions.RankWarning, stacklevel=2)

    return c


def _polyder(p: F64["n_coef n_cols"], m: int) -> F64["int n_cols"]:
    """Differentiate `m` times the polynomial in each column of `p`.

    The result has n_coef rows for m == 0, one row when m >= n_coef and
    n_coef - m rows otherwise.
    """
    result: F64["int n_cols"]
    if m == 0:
        result = p
    else:
        n = p.shape[0]
        if n <= m:
            result = np.zeros_like(p[:1, ...])
        else:
            dp = np.asarray(p[:-m - 1, ...], copy=True)
            for k in range(m):
                rng = np.arange(n - k - 1, m - k - 1, -1, dtype=p.dtype)
                dp *= np.reshape(rng, (n - m,) + (1,) * (p.ndim - 1))
            result = dp
    return result


# --------------------------------------------------------------------------
# Savitzky-Golay filter
# --------------------------------------------------------------------------


def _convolve1d(
    x: F64["n_channels n_samples"],
    weights: F64[window],
    mode: str = "constant",
    cval: float = 0.0,
) -> F64["n_channels n_samples"]:
    """scipy.ndimage.convolve1d along the last axis (origin 0)."""
    n_w = weights.shape[0]
    right = n_w // 2
    left = n_w - 1 - right
    padded: F64["n_channels n_padded"]  # n_samples + n_w - 1
    if mode == "constant":
        padded = np.pad(x, ((0, 0), (left, right)), mode="constant", constant_values=cval)
    else:
        padded = np.pad(x, ((0, 0), (left, right)), mode=_PAD_MODES[mode])
    windows = sliding_window_view(padded, n_w, axis=1)
    return windows @ weights[::-1]


def _fit_edge(
    x: F64["n_channels n_samples"],
    window_start: int,
    window_stop: int,
    interp_start: int,
    interp_stop: int,
    polyorder: int,
    deriv: int,
    delta: float,
    y: F64["n_channels n_samples"],
) -> F64["n_channels n_samples"]:
    """Fit a polynomial to x[:, window_start:window_stop] and write its values
    on interp_start:interp_stop into y."""
    # Get the edge into a (window_length, n_channels) array.
    x_edge = x[:, window_start:window_stop]
    xx_edge: F64["n_win n_channels"] = np.swapaxes(x_edge, -1, 0)
    xx_edge = np.reshape(xx_edge, (xx_edge.shape[0], -1))

    # Fit the edges. poly_coeffs has shape (polyorder + 1, n_channels).
    t_win: F64["n_win"] = np.arange(0, window_stop - window_start, dtype=x.dtype)
    # (polyorder + 1, n_channels), then fewer rows after differentiation.
    poly_coeffs: F64["int n_channels"] = _polyfit(t_win, xx_edge, polyorder)

    if deriv > 0:
        poly_coeffs = _polyder(poly_coeffs, deriv)

    # Compute the interpolated values for the edge.
    i = np.arange(
        interp_start - window_start, interp_stop - window_start, dtype=poly_coeffs.dtype
    )
    values = _polyval(poly_coeffs, np.reshape(i, (-1, 1))) / (delta**deriv)

    # Reshape values to match y, then put them in the slice of y.
    values = np.reshape(values, (interp_stop - interp_start, y.shape[0]))
    values = np.swapaxes(values, 0, -1)
    y[:, interp_start:interp_stop] = values
    return y


def _fit_edges_polyfit(
    x: F64["n_channels n_samples"],
    window_length: int,
    polyorder: int,
    deriv: int,
    delta: float,
    y: F64["n_channels n_samples"],
) -> F64["n_channels n_samples"]:
    """Fill the halflen values at each end of y by polynomial interpolation of x."""
    halflen = window_length // 2
    y = _fit_edge(x, 0, window_length, 0, halflen, polyorder, deriv, delta, y)
    n = x.shape[-1]
    y = _fit_edge(x, n - window_length, n, n - halflen, n, polyorder, deriv, delta, y)
    return y


def savgol_filter(
    x: F64["n_channels n_samples"],
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    mode: str = "interp",
    cval: float = 0.0,
) -> F64["n_channels n_samples"]:
    """Apply a Savitzky-Golay filter to each row of `x`."""
    if mode not in ["mirror", "constant", "nearest", "interp", "wrap"]:
        raise ValueError(
            "mode must be 'mirror', 'constant', 'nearest' 'wrap' or 'interp'."
        )

    coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)

    if mode == "interp":
        if window_length > x.shape[-1]:
            raise ValueError(
                "If mode is 'interp', window_length must be less "
                "than or equal to the size of x."
            )
        # Do not pad: the window_length // 2 values at each end come from a
        # polynomial fitted to the last window_length samples.
        y = _convolve1d(x, coeffs, mode="constant")
        y = _fit_edges_polyfit(x, window_length, polyorder, deriv, delta, y)
    else:
        y = _convolve1d(x, coeffs, mode=mode, cval=cval)

    return y


# --------------------------------------------------------------------------
# Filter initial conditions
# --------------------------------------------------------------------------


def lfilter_zi(b: F64[n_b], a: F64[n_a]) -> F64[n_zi]:
    """Initial state of lfilter for the steady state of the step response."""
    if not (b.ndim == a.ndim == 1):
        raise ValueError(
            "Numerator `b` and Denominator `a` must be 1-D arrays, "
            + f"but {b.shape = }, {a.shape = }!"
        )

    if a[0] == 0:
        raise ValueError("First coefficient of parameter `a` must be non-zero!")

    if a[0] != 1:  # Normalize the coefficients so a[0] == 1:
        b, a = b / a[0], a / a[0]

    if (sum_a := np.sum(a)) == 0:
        raise ValueError(
            "Filter not stable due to sum(a) == 0, i.e., having a pole at z = 1!"
        )

    y_inf = np.sum(b) / sum_a  # y[k -> inf] for unit-step input

    # zi[k] = zi[k+1] + b - y_inf*a, allowing different lengths for a and b.
    n_a, n_b = a.shape[0], b.shape[0]
    n = max(n_a, n_b)
    b_pad = np.pad(b, (0, n - n_b))
    a_pad = np.pad(a, (0, n - n_a))
    return np.flip(np.cumsum(np.flip(b_pad - y_inf * a_pad)))[1:]


def sosfilt_zi(sos: F64["n_sections 6"]) -> F64["n_sections 2"]:
    """Initial state of sosfilt for the steady state of the step response."""
    if sos.ndim != 2 or sos.shape[1] != 6:
        raise ValueError("sos must be shape (n_sections, 6)")

    n_sections = sos.shape[0]
    zi = np.empty((n_sections, 2), dtype=sos.dtype)
    scale = 1.0
    for section in range(n_sections):
        b = sos[section, :3]
        a = sos[section, 3:]
        zi[section, ...] = scale * lfilter_zi(b, a)
        # b.sum()/a.sum() is H(1), the gain of this section at omega=0.
        scale *= np.sum(b) / np.sum(a)
    return zi


# --------------------------------------------------------------------------
# Edge extensions (scipy/signal/_arraytools.py), last axis
# --------------------------------------------------------------------------


def odd_ext(x: F64["n_channels n_samples"], n: int) -> F64["n_channels int"]:
    """Odd extension of `x` by `n` samples at each end of the last axis."""
    if n < 1:
        return x
    if n > x.shape[-1] - 1:
        raise ValueError(
            ("The extension length n (%d) is too big. "
             + "It must not exceed x.shape[axis]-1, which is %d.")
            % (n, x.shape[-1] - 1)
        )
    left_end = x[:, 0:1]
    left_ext = x[:, n:0:-1]
    right_end = x[:, -1:]
    right_ext = x[:, -2:-(n + 2):-1]
    ext = np.concatenate((2 * left_end - left_ext, x, 2 * right_end - right_ext), axis=-1)
    return ext


def even_ext(x: F64["n_channels n_samples"], n: int) -> F64["n_channels int"]:
    """Even extension of `x` by `n` samples at each end of the last axis."""
    if n < 1:
        return x
    if n > x.shape[-1] - 1:
        raise ValueError(
            ("The extension length n (%d) is too big. "
             + "It must not exceed x.shape[axis]-1, which is %d.")
            % (n, x.shape[-1] - 1)
        )
    left_ext = x[:, n:0:-1]
    right_ext = x[:, -2:-(n + 2):-1]
    ext = np.concatenate((left_ext, x, right_ext), axis=-1)
    return ext


def const_ext(x: F64["n_channels n_samples"], n: int) -> F64["n_channels int"]:
    """Constant extension of `x` by `n` samples at each end of the last axis."""
    if n < 1:
        return x
    left_end = x[:, 0:1]
    ones = np.ones((1, n), dtype=x.dtype)
    left_ext = ones * left_end
    right_end = x[:, -1:]
    right_ext = ones * right_end
    ext = np.concatenate((left_ext, x, right_ext), axis=-1)
    return ext


# --------------------------------------------------------------------------
# lfilter / sosfilt with initial conditions (numpy replacements)
# --------------------------------------------------------------------------


def _lfilter(
    b: F64[n_b],
    a: F64[n_a],
    x: F64["n_channels n_t"],
    zi: F64["n_channels n_zi"],
) -> tuple[F64["n_channels n_t"], F64["n_channels n_zi"]]:
    """scipy.signal.lfilter(b, a, x, axis=-1, zi=zi), direct form II transposed."""
    n = max(a.shape[0], b.shape[0])
    b_pad = np.pad(b, (0, n - b.shape[0])) / a[0]
    a_pad = np.pad(a, (0, n - a.shape[0])) / a[0]
    z = zi.copy()
    y = np.empty_like(x)
    for t in range(x.shape[1]):
        xt = x[:, t]
        if n == 1:
            y[:, t] = b_pad[0] * xt
            continue
        yt = b_pad[0] * xt + z[:, 0]
        z[:, :-1] = z[:, 1:] + np.outer(xt, b_pad[1:-1]) - np.outer(yt, a_pad[1:-1])
        z[:, -1] = b_pad[-1] * xt - a_pad[-1] * yt
        y[:, t] = yt
    return y, z


def _sosfilt(
    sos: F64["n_sections 6"],
    x: F64["n_channels n_t"],
    zi: F64["n_sections n_channels 2"],
) -> tuple[F64["n_channels n_t"], F64["n_sections n_channels 2"]]:
    """scipy.signal.sosfilt(sos, x, axis=-1, zi=zi): sections in cascade."""
    y = x
    zf = np.empty_like(zi)
    for s in range(sos.shape[0]):
        y, zf[s] = _lfilter(sos[s, :3], sos[s, 3:], y, zi[s])
    return y, zf


# --------------------------------------------------------------------------
# Forward-backward filtering: padding and zi shaping
# --------------------------------------------------------------------------


def _validate_pad(
    padtype: str | None,
    padlen: int | None,
    x: F64["n_channels n_samples"],
    ntaps: int,
) -> tuple[int, F64["n_channels int"]]:
    """Validate padding for filtfilt and build the extended signal."""
    if padtype not in ["even", "odd", "constant", None]:
        raise ValueError(
            f"Unknown value '{padtype}' given to padtype. "
            "padtype must be 'even', 'odd', 'constant', or None."
        )

    if padtype is None:
        padlen = 0

    if padlen is None:
        # Original padding; preserved for backwards compatibility.
        edge = ntaps * 3
    else:
        edge = padlen

    # x's last dimension must be bigger than edge.
    if x.shape[-1] <= edge:
        raise ValueError(
            "The length of the input vector x must be greater than padlen, "
            f"which is {edge}."
        )

    ext: F64["n_channels int"]
    if padtype is not None and edge > 0:
        # Make an extension of length `edge` at each end of the input array.
        if padtype == "even":
            ext = even_ext(x, edge)
        elif padtype == "odd":
            ext = odd_ext(x, edge)
        else:
            ext = const_ext(x, edge)
    else:
        ext = x
    return edge, ext


def _validate_x(x: F64["n_channels n_samples"]) -> F64["n_channels n_samples"]:
    if x.ndim == 0:
        raise ValueError("x must be at least 1-D")
    return x


def _validate_sos(sos: F64["n_sections 6"]) -> tuple[F64["n_sections 6"], int]:
    """Helper to validate a SOS input."""
    if sos.ndim != 2:
        raise ValueError("sos array must be 2D")
    n_sections, m = sos.shape
    if m != 6:
        raise ValueError("sos array must be shape (n_sections, 6)")
    if not np.all(sos[:, 3] == 1):
        raise ValueError("sos[:, 3] should be all ones")
    return sos, n_sections


def filtfilt(
    b: F64[n_b],
    a: F64[n_a],
    x: F64["n_channels n_samples"],
    padtype: str | None = "odd",
    padlen: int | None = None,
) -> F64["n_channels n_samples"]:
    """Forward-backward IIR filtering along the last axis (method='pad')."""
    edge, ext = _validate_pad(padtype, padlen, x, ntaps=max(len(a), len(b)))

    # Steady state of the filter's step response.
    zi = lfilter_zi(b, a)

    # Reshape zi and create x0 so that zi*x0 broadcasts to the 'zi' of lfilter.
    zi_row = np.reshape(zi, (1, zi.size))
    x0 = ext[:, :1]

    # Forward filter. (The final states are not used; upstream binds them
    # to `zf` twice.)
    (y, _) = _lfilter(b, a, ext, zi_row * x0)

    # Backward filter, y0 so that zi*y0 broadcasts.
    y0 = y[:, -1:]
    (y, _) = _lfilter(b, a, y[:, ::-1], zi_row * y0)

    # Reverse y.
    y = y[:, ::-1]

    if edge > 0:
        # Slice the actual signal from the extended signal.
        y = y[:, edge:-edge]

    # ext has n_samples + 2*edge samples, a size over the runtime int `edge`.
    return y  # type: ignore[return-value]  # NUMTY: n_samples + 2*edge has no spelling, so y[:, edge:-edge] is not provably n_samples


def sosfiltfilt(
    sos: F64["n_sections 6"],
    x: F64["n_channels n_samples"],
    padtype: str | None = "odd",
    padlen: int | None = None,
) -> F64["n_channels n_samples"]:
    """Forward-backward filtering with cascaded second-order sections."""
    sos, n_sections = _validate_sos(sos)
    x = _validate_x(x)

    ntaps = 2 * n_sections + 1
    ntaps -= min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum()))
    edge, ext = _validate_pad(padtype, padlen, x, ntaps=ntaps)

    # (n_sections, 2) --> (n_sections, 1, 2) so that zi * x_0 is
    # (n_sections, n_channels, 2).
    zi = sosfilt_zi(sos)
    zi_3d = np.reshape(zi, (zi.shape[0], 1, 2))
    x_0 = ext[:, :1]
    # (n_sections, 1, 2) * (n_channels, 1) is (n_sections, n_channels, 2).
    (y, _) = _sosfilt(sos, ext, zi=zi_3d * x_0)  # type: ignore[arg-type]  # NUMTY: literal 2 broadcast against ?U (x_0's size-1 axis) gives ?U, not 2
    y_0 = y[:, -1:]
    (y, _) = _sosfilt(sos, y[:, ::-1], zi=zi_3d * y_0)  # type: ignore[arg-type]  # NUMTY: literal 2 broadcast against ?U gives ?U, not 2
    y = y[:, ::-1]
    if edge > 0:
        y = y[:, edge:-edge]
    return y  # type: ignore[return-value]  # NUMTY: n_samples + 2*edge has no spelling, so y[:, edge:-edge] is not provably n_samples
