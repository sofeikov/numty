# Symbolic arithmetic

One signature, no annotations in the body. Every comment below is the
shape numty infers, as a formula over the names in the signature:

```python
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numty import F64, I64

def features(
    signal: F64["batch time"],
    image: F64["batch chan height width"],
    offsets: I64["batch+1"],
) -> None:
    np.diff(signal, axis=1)                     # (batch, time-1)
    np.fft.rfft(signal, axis=1)                 # (batch, time//2+1), complex128
    sliding_window_view(signal, 5, axis=1)      # (batch, time-4, 5)
    np.pad(signal, ((0, 0), (2, 2)))            # (batch, time+4)
    np.concatenate([signal, signal], axis=1)    # (batch, 2*time)
    image.reshape(image.shape[0], -1)           # (batch, chan*height*width)
    np.diff(offsets)                            # (batch,)
    signal[:, 1:] - signal[:, :-1]              # (batch, time-1)
    signal[:, 1:] - signal                      # off by one
```

```text
error: broadcast shapes disagree at trailing axis 0: time-1 vs time (shapes (batch, time-1) vs (batch, time))  [numty-broadcast]
```

`batch`, `time`, `chan`, `height` and `width` are never declared: the
strings mint them. The sizes are unknown until the function is called,
and the off-by-one on the last line is still an error, because
`time-1` cannot equal `time` whatever `time` is.

## What is tracked

Dims stay exact through arithmetic:
`repeat`/`tile` multiply (`N*2`), `pad` adds (`N+2`), `convolve` adds
lengths (`N+M-1`), `rfft` halves (`N//2+1`), `split` divides (`N//2`),
and reshape proves `N*2` vs `2*N` (dividing `-1` exactly); mismatches
error with readable formulas. Products of distinct symbolic dims track
as monomials (`H*D`, `C*H*W`): flattening `(B, C, H, W)` with `-1`
infers `(B, C*H*W)`, `(N, K)` reshaped to `(N, -1)` infers `(N, K)`,
and the round-trip back proves clean by commutativity. A reshape
target may spell its sizes as arithmetic over known ones:
`x.reshape(x.shape[0] * x.shape[1], -1)` on `(N, M, K)` is `(M*N, K)`,
and `2 * n`, `n + 1` and `n // 2` are read the same way.
`sliding_window_view` appends the window
after the shrunken dims (`N-w+1`, symbolic included — the conv-output
formula falls out for a literal window). A window that is a size,
`w.shape[0]` of `(W,)`, is `W` on the window axis and leaves the
shortened axis `int`: code pads the input to a size of its own name
first (`n_padded`, really `n + w - 1`), and `n_padded - W + 1` would not
be `n`. An `int` window keeps the rank with `int` sizes. Linear forms also spell in annotations: `I64["N+1"]`
denotes one more than `N`, so the offsets/lengths duality checks both
ways — `np.diff` of `I64["N+1"]` is `I64[N]`, and
`np.concatenate([[0], np.cumsum(lengths)])` of `I64[N]` is `I64["N+1"]`
(no `type: ignore`). Supported per dim: `N+1`, `N-1`, `2*N`, `N+M`,
plain constants (`"5"` is `Literal[5]`); anything non-linear
(`N*M`, `N//2`) fails loudly instead of going quietly `Any`.

## What that unlocks

Code that failed before, exact now:

```python
def f(u: F64[N], w: F64[N, K]) -> None:
    n: N = u.shape[0]
    # Was: "reshape changes total size: 1 -> 2". Now: exact F64[2, N].
    r: F64[2, N] = np.reshape(np.repeat(u, 2), (2, n))
    # -1 divides the canonical product (N*2 / 2 = N); the quotient
    # contracts exactly downstream. Was: bailed to Any.
    t = np.reshape(np.repeat(u, 2), (2, -1))
    m: F64[2, K] = np.dot(t, w)
    # rfft on symbolic input claimed (N,) — silently wrong.
    # Now: (N//2+1), and it participates in checks.
    s = np.fft.rfft(u)
```
