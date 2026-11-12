# A whole attention block from one signature

```python
import numpy as np
from numty import F32

def attention(
    q: F32["batch heads seq dim"],
    k: F32["batch heads seq dim"],
    v: F32["batch heads seq dim"],
) -> F32["batch heads seq dim"]:
    scores = q @ k.swapaxes(-1, -2)            # (batch, heads, seq, seq)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ v
```

```text
Success: no issues found in 1 source file
(2, 4, 7, 16)
```

Four symbolic dims through a batched matmul, a transpose, two
keepdims reductions and two broadcasts, with the return checked
against the signature. Return the weights by mistake (drop the `@ v`)
and the two shapes are named side by side:

```text
error: Incompatible return value type (got "ndarray[tuple[batch, heads, seq, seq], dtype[floating[_32Bit]]]", expected "ndarray[tuple[batch, heads, seq, dim], dtype[floating[_32Bit]]]")  [return-value]
```
