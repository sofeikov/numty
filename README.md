# numty — static shape checking for NumPy

Wrong-shaped arrays are the `null` of numerical code: everything type-checks,
everything imports, and it blows up three functions later — or worse, silently
computes garbage. [jaxtyping](https://docs.kidger.site/jaxtyping/) solved this
at runtime. numty explores how far you can get *statically*, before anything
runs, using only what type checkers already understand plus one small mypy
plugin.

## What it catches

You define the shapes you care about in one line — no pre-baked list from
the library:

```python
from typing import Literal
from numty import FloatArray

A2x3 = FloatArray[tuple[Literal[2], Literal[3]]]

def use_2x3(m: A2x3) -> None: ...

def bad_rank(v: Vector) -> None:
    use_2x3(v)  # error: 1-D is not 2-D
```

```
demo.py:21: error: Argument 1 to "use_2x3" has incompatible type
"ndarray[tuple[int], dtype[float64]]";
expected "ndarray[tuple[Literal[2], Literal[3]], dtype[float64]]"  [arg-type]
```

Mixing up sizes:

```python
def bad_size() -> None:
    use_2x3(np.zeros((3, 2)))  # error: 3x2 is not 2x3
```

Mismatched matmul, for any rank — with the output shape propagated:

```python
c = matmul(np.zeros((2, 3)), np.zeros((3, 5)))  # c is known to be 2x5
matmul(np.zeros((2, 3)), np.zeros((2, 3)))  # error: inner dimensions disagree: 3 vs 2
```

Broadcasting that keeps precision through whole expressions:

```python
c = matmul(np.zeros((2, 3)), np.zeros((3, 4))) + np.zeros((2, 4))  # still known 2x4
np.zeros((2, 3)) + np.zeros((2, 4))  # error: broadcast shapes disagree
```

Mixed dtypes promote like NumPy, verified against `np.result_type`
over the full dtype matrix (plus weak Python scalars):

```python
f32 + np.zeros((2, 3))   # float64, shape kept
f32 + 1.5                # float32: weak Python scalars convert
i32 + 2.5                # float64, shape kept
f32 * 1j                 # complex64
```

Reshape with size conservation and `-1` inference:

```python
np.zeros((2, 3)).reshape((3, -1))  # known to be 3x2
np.zeros((2, 3)).reshape((2, 4))   # error: reshape changes total size: 6 -> 8
np.zeros((2, 3)).reshape((4, -1))   # error: reshape cannot infer -1
```

Rank-changing ops that track the new shape:

```python
np.zeros((2, 3)).transpose()          # known to be 3x2
np.zeros((1, 2, 3)).squeeze(0)        # known to be 2x3
np.expand_dims(np.zeros((3,)), 0)     # known to be 1x3
np.zeros((2, 3)).sum(axis=1)          # known to be a vector of 2
np.zeros((2, 3)).transpose(0, 0)      # error: not a permutation
np.zeros((2, 3)).squeeze(0)          # error: axis 0 has size 2, not 1
```

No-axis reductions (`.sum()` with no axis) return a scalar, so the plugin
leaves them to numpy's own stubs; only float/complex dtypes keep their
dtype through a reduction, anything else bails out.

Comparisons broadcast shapes and return bool arrays; `argmax` drops the
axis to integer positions, `sort` keeps the shape:

```python
probs.argmax(axis=1) == labels        # bool vector, shape tracked
np.zeros((2, 3)) == np.zeros((3, 2))  # error: broadcast shapes disagree
```

`Generator` random methods solve from `size=` (`rng.normal(size=(n, d))`,
`rng.integers`, `choice`, `permutation`); a missing size means a scalar,
so the plugin bails to the stub.

Elementwise ops broadcast shapes and promote dtypes — including the bare
`np.*` ufunc forms, which the plugin reaches through the ufunc's own
generic type (the stubs declare them as variables, not functions):

```python
np.sqrt(a)                 # same shape; ints become float64
np.abs(c)                  # complex becomes real (float)
np.isfinite(a)             # bool, same shape
np.add(a, b)               # same as a + b, with promotion
np.where(cond, x, 0.0)     # broadcast of all three, promoted branches
np.equal(a, b)             # bool, broadcast
a - b; a / b; a**2         # operators too; int / int is float64
np.clip(a, 0.0, 1.0)       # same shape and dtype as a (method too)
np.broadcast_to(v, (2, 3)) # new shape, same dtype
np.asarray(a)              # identity on known arrays
np.asarray(a, dtype=np.float32)  # named dtype used as-is
np.array([105.0, 68.0])    # (2,) from the display
a.astype(np.float32)       # same shape, named dtype
len(a)                     # first dim: literal, or the TypeVar itself
len(Slots)                 # member count of an IntEnum (aliases count once)
a.shape                    # precise tuple, literals and TypeVars
np.matmul(a, b)            # raw np.matmul solves now, errors included
a & mask; a.any(axis=1)    # bool logic and reductions, same axis rules
np.zeros((2, 3)) / np.zeros((4,))  # error: true_divide shapes disagree
```

`einsum` parses the subscript string and solves label sizes, including
implicit mode, ellipsis, and diagonals:

```python
np.einsum("ij,jh->ih", a, b)  # (2, 4), like matmul
np.einsum("ij,jh", a, b)      # implicit: once-labels, alphabetical
np.einsum("...ij,j->...i", batched, v)
np.einsum("ij,jh->ih", a, c)  # error: label 'j' has conflicting sizes
```

`linalg` solvers check squareness and inner dims, and track dtypes
(`eig` of float is complex, `eigh` stays real, ints become float64):

```python
x = np.linalg.solve(a4, b)          # (4,) or (4, K), batch-broadcast
w, v = np.linalg.eigh(cov)          # (4,) float, (4, 4)
u, s, vh = np.linalg.svd(m)         # full_matrices=False shrinks U, Vh
sol, _, _, _ = np.linalg.lstsq(a, y)  # only the solution is provable;
                                      # residuals depend on runtime rank
np.linalg.norm(m, axis=1)           # (2,); no-axis norm is a scalar,
                                    # so the plugin leaves it to the stubs
np.linalg.solve(non_square, b)      # error: requires a square array
```

Symbolic dimensions work through TypeVars in the shape tuple — same
name means same size, and the solver propagates them through bodies:

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

Apply the alias explicitly at each use (`A[N, K, C]`, never bare
`A` — a bare generic alias silently means unknown sizes). Inside
bodies, equal names unify, ranks still check, and anything unprovable
bails: `N` vs `M` never errors (they could match at runtime), but
`a[0, 1, 2]` on a 2-d symbolic array does. Call-site conflicts also
stay silent — `f(x_5, y_6)` against `(A[N], A[N])` unifies through
`int`, exactly like the generic-`int`-parameter limitation below.

Subscripts that track shapes through indexing:

```python
a[0]          # first axis dropped
a[:, 1]       # mixed int and slice
a[1:3]        # slice bounds compute the length (via folding: H*2 works too)
a[::-1]       # reversed, same shape
a[None, :]    # new axis of size 1
a[..., 1]     # ellipsis fills the middle
a[5]          # error: index 5 is out of range for axis 0 with size 2
a[::0]        # error: slice step is zero
a[..., Slots.X]   # IntEnum members index by their value
a[:, COL]         # a Final slice(Slots.X, Slots.Y + 1) object, same thing
a[Slots.VX]       # error: index 2 is out of range for axis 0 with size 2
```

Slicing past the end clips like NumPy (`a[1:99]` on a size-2 axis is
size 1); boolean masks and unknown bounds bail out. A full slice on a
symbolic axis keeps the TypeVar (`a[:, :]` on `(N, K)` stays `(N, K)`);
a bounded slice on one (`a[:, :k]`) bails out. Indexing every axis
with integers yields a scalar, which the plugin leaves to the stubs.

Integer arrays index fancy-style: the index shape replaces the axis,
broadcast when there are several; `take_along_axis` broadcasts the two
shapes, with the index winning along the axis (a 1 there falls back to
the array dim):

```python
table[ids]                        # (5, 4): index shape + trailing dims
b[np.arange(5), actor]            # (5,): broadcast (5,) + (5,)
np.take_along_axis(a, idx, axis=1)  # broadcast shape, arr's dtype
b[np.zeros(3)]                    # error: integer (or boolean) type required
np.take_along_axis(a, idx2d, axis=1)  # error: same number of dimensions
```

Shape builders that compute the result shape:

```python
np.zeros_like(a)                          # same shape and dtype as a
np.zeros_like(a, shape=(3, 2))            # override wins, precisely
np.full((2, 3), 5.0)                      # known 2x3 float64
np.eye(3)                                 # known 3x3
np.arange(6)                              # known vector of 6
np.concatenate((a, b), axis=1)            # dims added on the axis
np.stack((a, b), axis=0)                  # new axis of stacked count
np.stack([blocks[c] for c in Kinds], -1)  # enum-driven stacks too
np.concatenate((a, b), axis=0)            # error: dimension 1 differs
```

Reshape conserves the total with symbolic sizes: int products must
match and the TypeVar multisets must agree (`(N, 10, 2)` to `(N, 20)`
is fine; `(N, 10, 2)` to `(N, 21)` errors; `(N, 10)` to `(M, 10)`
bails — unprovable, not wrong):

```python
a.reshape(a.shape[0], 20)  # (N, 20): TypeVar sizes compare by name
np.arange(len(a))          # (N,): arange of a symbolic count
```

Tuple displays (`(a, b)`) keep per-item precision; list displays
`[a, b]` work too — the plugin re-reads each item's type from its
expression, since lists arrive joined against the `ArrayLike` formal.
An explicit `dtype=` you name is used as-is when it resolves to a
concrete dtype (`np.float32`, `np.int64`, `"float64"`, ...); anything
exotic bails out instead of guessing.

## Quickstart

Named sizes work too — mark them `Final` and the solver folds names
and arithmetic (`H * 2`, `W + 1`, `M // 2`) just like literals, and
`len()` of an `IntEnum` class folds to its member count (aliases share
one value, so they count once). Plain variables and parameters stay
unknown: their values are gone from the types before the plugin sees
them, so they bail out instead of erroring. A generic `int` parameter
can never be verified against a shape — the contract says "every
int", and no annotation survives the join to `int`. But two arguments
sharing one shape TypeVar are checked at the call site, where the
actual shapes are still visible: `f(x_5, y_6)` against `(A[N], A[N])`
errors with the two sizes named. Different TypeVars (`N` vs `M`) stay
silent by design — their equality is unprovable, and unknown is not
an error.

```sh
uv venv
uv pip install mypy numpy pytest
```

```sh
PYTHONPATH=. uv run mypy demo.py   # expect 2 errors: rank, size
uvx ty check numty.py demo.py      # expect the same 2 errors
uv run pytest                      # full suite: runtime + mypy-driven (80 and growing)
```

## How it works

Three layers, each building on the previous one:

| Layer | Idea | Works under |
|---|---|---|
| Rank aliases | `Matrix = ndarray[tuple[int, int], ...]` — tuple length is the rank | mypy, pyright, ty |
| Your own sizes | `A2x3 = FloatArray[tuple[Literal[2], Literal[3]], ...]`, or `Array2D[Literal[2], Literal[3]]` | mypy, pyright, ty |
| Shape solver | [numty_plugin.py](/home/sofeikov/my/numty/numty_plugin.py) hooks constructors, `matmul`, arithmetic ops and ufuncs (with promotion), `where`/`clip`/`broadcast_to`, `reshape` (incl. symbolic conservation), rank-changing ops, shape builders, comparisons, bool logic (`&`/`\|`), `any`/`all`, the arg family, `einsum`, `linalg` solvers, fancy indexing, `take_along_axis`, symbolic TypeVar dims (same name unifies, ranks check, the rest bails), call-site TypeVar conflicts, enum-driven stacks, folds constant int expressions and `len()` (`Final` arithmetic, enum member counts), solves shapes itself | mypy only |

The first two layers are pure standard typing — no magic, portable everywhere.
The library ships only the generic bases (`FloatArray`, `Array1D`,
`Array2D`) plus rank aliases; exact sizes live in your code, one line each.
Plain constructors just work: `np.zeros((2, 3))` already has a precise type,
and the plugin keeps it precise even through assignment.

Layers 1–2 hit a wall at cross-argument checks: `ndarray`'s shape parameter is
effectively covariant, so a shared `TypeVar` silently joins conflicting
literals (3 vs 2 becomes `int`) instead of erroring. The plugin stops asking
the type system to solve shapes and does the solving itself, with readable
errors like `matmul batch dimensions disagree: (5,) vs (4,)` — including
a call-site check that catches the shared-`TypeVar` conflict the join
hides (`f(x_5, y_6)` against `(A[N], A[N])`).

The plugin is registered in [mypy.ini](/home/sofeikov/my/numty/mypy.ini) and
needs `PYTHONPATH=.` so mypy can import it (`MYPYPATH` does not work for
plugins). Without the plugin, checking degrades gracefully to the first two
layers.

Narrowing a dynamically-shaped value uses a witness instead of a per-shape
helper:

```python
use_2x3(narrow(m, np.zeros((2, 3))))  # asserts at runtime, precise statically
```

## Limitations, stated plainly

- The plugin is mypy-only and pinned to mypy internals that occasionally
  shift. `ty` and pyright users get layers 1–2, which still catch rank and
  fixed-size bugs.
- Without the plugin, `matmul` returns a loose type: usable but unchecked.
- Dynamic shapes (`np.zeros((n, 3))`) cannot be proven wrong, so they are
  left alone. Unknown is not an error.
- An explicit `dtype=` is honored when it resolves (`np.float32`,
  `"float64"`, bare `int`/`float`/`bool`, ...); exotic forms bail out.
  Same for unknown `keepdims`.
- On lines the solver rejects, mypy may add its own follow-on error next
  to ours (an overload complaint for tuple `reshape`, `Unsupported
  operand` for broadcast ops). Ours is the readable one.
- Test runs use a dedicated mypy cache dir cleared per suite run (see
  `test_static.py`): mypy does not reliably invalidate plugin results
  across plugin edits, so never trust the default cache while hacking
  on the plugin.
- The `@` operator is accepted but unchecked: operator dispatch consulted
  the method hook inconsistently in testing. Raw `np.matmul` solves
  through the ufunc hook instead, as does the `numty.matmul` wrapper.
  Same for reflected ops beyond the hooked pairs.
- Ufunc calls need mypy to follow imports (the default): under
  `--follow-imports=skip` the ufunc generic aliases stay unresolved and
  bare `np.sqrt`-style calls bail out. Operators, methods, and real
  `def` stubs are unaffected.
- An inline call to an overloaded function nested directly inside an
  overloaded numpy call dissolves to `Any` (mypy keeps the
  overload-inferred type over the hook result): hoist it to a local
  first (`kernel = gaussian_kernel(...); np.where(cond, kernel, 0.0)`).
- Union-typed array inputs (`(N, K, M) | (N, 1, M)`) bail out: the
  solver tracks one shape per value, so bodies over unions stay
  unchecked (call sites still check each concrete shape).
- Rank-polymorphic helpers (`(..., 2)` → `...`) need `@overload` per
  rank: mypy cannot express a `*Ts, 2` implementation whose body still
  checks, so the implementation carries an explicit narrow ignore while
  its body verifies at the max rank.
- Heterogeneous generator unpacking (`a, b = (f(p) for p in (p1, p2))`
  with different shapes) joins to `Any`: spell the lines out.
- Different TypeVars (`N` vs `M`) never conflict, even when provably
  different at a call site — only same-name TypeVars are unified.

## Layout

- [numty.py](/home/sofeikov/my/numty/numty.py) — rank aliases, generic shape
  bases, `narrow` witness helper, shape-preserving `ident`, loose `matmul`
  fallback.
- [numty_plugin.py](/home/sofeikov/my/numty/numty_plugin.py) — the solver.
- [demo.py](/home/sofeikov/my/numty/demo.py) — good and bad usages, clean under
  both checkers except the two intentional errors.
- [bench/](/home/sofeikov/my/numty/bench) — realistic benchmark (MLP training,
  math library, 17 of 18 seeded mistakes caught), with its own scorecard in
  [bench/README.md](/home/sofeikov/my/numty/bench/README.md).
- [tests/test_runtime.py](/home/sofeikov/my/numty/tests/test_runtime.py) —
  shapes, values, narrowing accept/reject.
- [tests/test_static.py](/home/sofeikov/my/numty/tests/test_static.py) — runs
  real mypy over [fixtures](/home/sofeikov/my/numty/tests/fixtures/static_ok.py):
  each bad one must fail with our message, and `reveal_type` pins must show
  precise shapes on the good paths.

## Roadmap

Value-dependent frontiers: NEP 50's value-based edge cases (out-of-bound
Python ints) and union dtypes currently bail out. The `@` operator stays
accepted-but-unchecked (mypy consults the method hook inconsistently).
`ty` and pyright get layers 1–2 — neither offers a plugin API to build
the solver on. Rank polymorphism wants `TypeVarTuple` (`tuple[*Ts, 2]`)
instead of per-rank overloads; a union-distributing solver would check
bodies over union-typed inputs instead of bailing.
