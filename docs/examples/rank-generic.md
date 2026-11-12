# One signature, every rank

The headline: rank-generic helpers with no overloads, no per-rank copies.
mypy solves `*Ts` from each call site, and the body checks once,
rank-generically:

```python
from typing import Literal
from typing_extensions import TypeVarTuple, Unpack
from numty import FlexArray

Ts = TypeVarTuple("Ts")

def squared_norm(
    v: FlexArray[tuple[Unpack[Ts], Literal[2]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> FlexArray[tuple[Unpack[Ts]], np.dtype[np.float64]]:  # type: ignore[type-var]
    return v[..., 0] ** 2 + v[..., 1] ** 2

squared_norm(np.zeros((5, 7, 2)))  # (5, 7)
squared_norm(np.zeros((5, 2)))     # (5,)
squared_norm(np.zeros((5, 7, 3)))  # error: expected (..., 2), got (..., 3)
```

Two honest `type: ignore[type-var]` notes per signature (NumPy's stubs bind
the shape to `tuple[int, ...]`, which an unsolved `*Ts` cannot satisfy) buy
all of this: inside, the solver threads `*Ts` through ellipsis indexing
(`v[..., 0]` on `(*Ts, 2)` is `(*Ts,)`, and out-of-range indices still
error against the known suffix), same-`*Ts` broadcasts, unary ufuncs,
`astype`/`asarray`, `.shape`, `&`/`|` on bool pairs, `where`/`clip`,
comparisons, reductions along axes that resolve to a known dim
(`.all(axis=-1)` on `(*Ts, 2)` is `(*Ts,)`), and rank arithmetic over
the unknown middle — `matmul` on `(*Ts, N, M)` pairs (inner dims must
align, batches broadcast through the unpack) and leading `None`/slice
indexing without an ellipsis (`v[0]` on `(5, *Ts, 7)` is `(*Ts, 7)`).
`reshape` over `*Ts` returns the spelled target as-is (`(*Ts, 2, 3)`
to `(6,)`), and `transpose`/`.T` over `*Ts` bails out instead of
claiming an order no annotation can spell.
Anything that would need the unknown length or order — several array
indices or an integer beside one (numpy may move their block to the
front), an index reaching into `*Ts` itself, reducing over `*Ts` itself,
mixing different `Ts` declarations, explicit transpose axes, a `-1`
that would absorb `*Ts` — bails out instead of guessing. A lone integer
array or boolean mask on the known prefix gathers in place
(`frames[idx]` on `(total, *Ts)` is `(K, *Ts)`).
