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
are not a provable error inside the body, since their equality is
unprovable. A body that needs them equal is still reported, as a likely
mistake, by the `numty-names` code, which is on by default (with
`disable_error_code = numty-names` they stay silent, see
[Error codes](../reference/error-codes.md)). The same holds for `matmul` inner dims between distinct free
symbolics (`(S,F)@(S,T)` keeps `(S,T)` "if legal", issue #127): a generic
function whose body needs `F == S` but declares them independent cannot
be proven wrong inside the body, so `numty-names` reports the pair and
the provable check stays at the call site, where concrete sizes are
visible. Apply the alias explicitly at each use (`A[N, K, C]`, never bare
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
`[return-value]` error (see the [limitations](../internals/limitations.md)).
A class keeps the variance its type variables declare, so two arguments
that give one name two sizes (`both(a: Sys[N], b: Sys[N])` with a
`Sys[Literal[3]]` and a `Sys[Literal[4]]`, also through `*args`) are
mypy's `Cannot infer value of type parameter "N"`
(`tests/fixtures/bad_generic_class_sizes.py`). An array argument whose size is `int` does not
bind a name that a container argument binds too: `pair(flags: F64[int,
S], box: Box[S])` called with `np.zeros((n, n))` and a `Box[S]` takes
`S` from the container (`tests/fixtures/bad_container_int_sizes.py`).

A method that returns a formula over the class's names (`def
resid(self) -> F64[N, "J-1"]`) is in the receiver's terms, as an
attribute declared with it is: on an `MN[N, Literal[4]]` it is `(N, 3)`.
A formula is a type argument too: `def ar(y: F64[N]) -> Model[N,
"N-1"]` gives `Model[N, N-1]`, and `Model[5, 4]` for a `F64[5]`. As a
base class (`class AR(Model[N, "N-1"])`) the string is also read by
mypy as a runtime expression, which it reports as `Invalid type comment
or annotation [valid-type]`; with `# type: ignore[valid-type]` on that
line the base still carries the formula. A name of a class return that
no parameter binds (`-> Model[N, W]`) is a size the function chooses,
as on an array axis: the body returns a value whose size there is born
in it (`Model(y, y[y > 0])`), and a call gives `Model[N, ?W@line]`
(`tests/fixtures/generic_class_sizes_ok.py`). A formula type argument of
a parameter is held at a call to the size the other type arguments give
its names, as a formula axis of an array parameter is: `wide(m:
Model[N, "N+1"])` given a `Model[K, K]` is `call-site shape conflict for
'N': N (argument 'm', type argument 0) gives N+1, not N (argument 'm',
type argument 1)` (`tests/fixtures/bad_generic_class_sizes.py`).

A `Protocol` member that returns a formula over the protocol's names
(`def sigmas(self, x: F64[X]) -> F64["2*X+1", X]`) is not matched by a
class that declares it over its own names: mypy compares the two
methods itself and does not put the type arguments into a formula, so
`Merwe[Literal[4]]` against `Pts[X]` is an `[arg-type]` error. Declare
the protocol's formula position with `int`.

A size that only the return of a property or of a method names (`def
rows(self) -> I64[E]` in a class that is not generic in `E`) is unknown
at each read: `c.rows` is `tuple[?U@line]`, and adding it to a caller's
`F64[E]` does not check as two arrays of one size (it is a
`[numty-names]` report, `E and ?c.E are different names`)
(`tests/fixtures/property_own_size.py`).
