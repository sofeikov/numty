# Chains compose: reshape, project, slice, reduce, return

Single ops are table stakes — the point is a whole block staying
checked end to end. Ten tracked steps, one signature, no intermediate
annotations; every middle shape is inferred and re-checked:

```python
import numpy as np

from numty import F64, batch, seq

def autoencode(
    patches: F64[batch, seq, 16],
    w_enc: F64[16, 8],
    w_dec: F64[8, 16],
) -> F64[batch, seq, 16]:
    h = np.matmul(patches, w_enc)  # (B, S, 8)
    h = np.maximum(h, 0.0)  # relu
    z = h.reshape(h.shape[0], h.shape[1], 2, 4)  # (B, S, 2, 4)
    zt = z.transpose(0, 1, 3, 2)  # (B, S, 4, 2)
    f = zt.reshape(zt.shape[0], zt.shape[1], 8)  # (B, S, 8)
    y = np.matmul(f, w_dec)  # (B, S, 16)
    first = y[:, 0, :]  # (B, 16)
    mu = y.mean(axis=1)  # (B, 16)
    gate = first * mu  # (B, 16)
    out = gate[:, None, :] + patches  # (B, S, 16)
    return out
```

```text
Success: no issues found in 1 source file
(5, 7, 16)
```

Two things the chain exercises deliberately: reshapes spell their
kept dims through `.shape` (bare `TypeVar`s are not runtime values —
`h.reshape(batch, seq, 2, 4)` would crash at import, and the checker
rejects it too), and rebinding a new shape to an old name needs
`allow_redefinition = True` ([recommended](../get-started.md#recommended-mypy-settings)) — under default mypy
it is an `[assignment]` error, not a silent widening. This exact
function is `tests/fixtures/chained_block.py`: checked clean, runs to
`(5, 7, 16)`.
