# Real generic code, checked end to end

Symbolic dimensions ride TypeVars in the shape tuple — same name means same
size — and the solver propagates them through whole function bodies:

```python
from typing import TypeVar
N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)
C = TypeVar("C", bound=int)
A = np.ndarray[tuple[N, K, C], np.dtype[np.float64]]

def pack(values: A[N, K, C], present: np.ndarray[tuple[N, K], np.dtype[np.bool_]]) -> A[N, K, C]:
    order = np.argsort(~present, axis=1, kind="stable")
    return np.take_along_axis(values, order[..., None], axis=1)

pack(np.zeros((5, 10, 2)), np.zeros((5, 10)) > 0)  # (5, 10, 2), precise
```

Two arguments sharing one shape TypeVar are checked at the call site,
where the actual shapes are still visible: `f(x_5, y_6)` against `(A[N],
A[N])` errors with the two sizes named. Different TypeVars (`N` vs `M`)
stay silent by default — their equality is unprovable, and unknown is not
an error (the opt-in `numty-names` code reports them, see
[Error codes](../reference/error-codes.md)). The same holds for `matmul` inner dims between distinct free
symbolics (`(S,F)@(S,T)` keeps `(S,T)` "if legal", issue #127): a generic
function whose body needs `F == S` but declares them independent cannot
be proven wrong inside the body, so the check stays at the call site,
where concrete sizes are visible. Apply the alias explicitly at each use (`A[N, K, C]`, never bare
`A` — a bare generic alias silently means unknown sizes). The same holds
for a dataclass holding shapes: `list[Box]` with `Box(Generic[N])`
holding `F64[N, 3]` is an error (`Box used bare ... [numty-annotation]`,
issue #38) — write `list[Box[T_ctx]]` so the field checks end to end.
A generic `NamedTuple` of arrays is reported the same way, a bare
parameter (`def f(box: Box)`) included.

`int` in a size position of such a container means "not followed", as
it does in an array. At a call, a parameter `box: Box[int]` accepts a
`Box[N]` and a `Box[Literal[1]]`, although mypy itself treats a class as
invariant in its parameters. This holds for a `NamedTuple`, a dataclass
or any generic class of arrays, also as `Box[int] | None` and inside
`list[...]`. It holds at calls only: an assignment or a `return` of a
`Box[N]` where `Box[int]` is declared is still mypy's `[assignment]` or
`[return-value]` error. An array argument whose size is `int` does not
bind a name that a container argument binds too: `pair(flags: F64[int,
S], box: Box[S])` called with `np.zeros((n, n))` and a `Box[S]` takes
`S` from the container (`tests/fixtures/bad_container_int_sizes.py`).

A size that only the return of a property or of a method names (`def
rows(self) -> I64[E]` in a class that is not generic in `E`) is unknown
at each read: `c.rows` is `tuple[?U@line]`, and adding it to a caller's
`F64[E]` does not check as two arrays of one size
(`tests/fixtures/property_own_size.py`).
