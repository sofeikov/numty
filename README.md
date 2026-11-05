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
np.asarray(a)              # identity on known arrays (explicit dtype= bails)
np.matmul(a, b)            # raw np.matmul solves now, errors included
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
```

Slicing past the end clips like NumPy (`a[1:99]` on a size-2 axis is
size 1); boolean masks, fancy indexing, and unknown bounds bail out.
Indexing every axis with integers yields a scalar, which the plugin
leaves to the stubs.

Shape builders that compute the result shape:

```python
np.zeros_like(a)                          # same shape and dtype as a
np.zeros_like(a, shape=(3, 2))            # override wins, precisely
np.full((2, 3), 5.0)                      # known 2x3 float64
np.eye(3)                                 # known 3x3
np.arange(6)                              # known vector of 6
np.concatenate((a, b), axis=1)            # dims added on the axis
np.stack((a, b), axis=0)                  # new axis of stacked count
np.concatenate((a, b), axis=0)            # error: dimension 1 differs
```

Tuple displays (`(a, b)`) keep per-item precision; list displays
`[a, b]` work too — the plugin re-reads each item's type from its
expression, since lists arrive joined against the `ArrayLike` formal.
Explicit `dtype=` always bails out — the plugin never guesses a dtype
you named.

## Quickstart

Named sizes work too — mark them `Final` and the solver folds names
and arithmetic (`H * 2`, `W + 1`, `M // 2`) just like literals.
Plain variables and parameters stay unknown: their values are gone
from the types before the plugin sees them, so they bail out instead
of erroring. A generic `int` parameter can never be verified against
a shape — the contract says "every int", and no annotation (not even
a `TypeVar` linking the argument to the shape) survives the join to
`int`.

```sh
uv venv
uv pip install mypy numpy pytest
```

```sh
PYTHONPATH=. uv run mypy demo.py   # expect 2 errors: rank, size
uvx ty check numty.py demo.py      # expect the same 2 errors
uv run pytest                      # full suite: runtime + mypy-driven (53 and growing)
```

## How it works

Three layers, each building on the previous one:

| Layer | Idea | Works under |
|---|---|---|
| Rank aliases | `Matrix = ndarray[tuple[int, int], ...]` — tuple length is the rank | mypy, pyright, ty |
| Your own sizes | `A2x3 = FloatArray[tuple[Literal[2], Literal[3]], ...]`, or `Array2D[Literal[2], Literal[3]]` | mypy, pyright, ty |
| Shape solver | [numty_plugin.py](/home/sofeikov/my/numty/numty_plugin.py) hooks constructors, `matmul`, arithmetic ops and ufuncs (with promotion), `where`/`clip`/`broadcast_to`, `reshape`, rank-changing ops, shape builders, comparisons, the arg family, `einsum`, and `linalg` solvers, folds constant int expressions, solves shapes itself | mypy only |

The first two layers are pure standard typing — no magic, portable everywhere.
The library ships only the generic bases (`FloatArray`, `Array1D`,
`Array2D`) plus rank aliases; exact sizes live in your code, one line each.
Plain constructors just work: `np.zeros((2, 3))` already has a precise type,
and the plugin keeps it precise even through assignment.

Layers 1–2 hit a wall at cross-argument checks: `ndarray`'s shape parameter is
effectively covariant, so a shared `TypeVar` silently joins conflicting
literals (3 vs 2 becomes `int`) instead of erroring. The plugin stops asking
the type system to solve shapes and does the solving itself, with readable
errors like `matmul batch dimensions disagree: (5,) vs (4,)`.

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
- Explicit `dtype=` always bails out — the plugin never guesses a dtype
  you named. Same for unknown `keepdims`.
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
Python ints), `dtype=` resolution, and union dtypes currently bail out.
The `@` operator stays accepted-but-unchecked (mypy consults the method
hook inconsistently). `ty` and pyright get layers 1–2 — neither offers
a plugin API to build the solver on.
