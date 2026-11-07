# numty — static shape checking for NumPy

Wrong-shaped arrays are the `null` of numerical code: everything type-checks,
everything imports, and it blows up three functions later — or worse, silently
computes garbage. [jaxtyping](https://docs.kidger.site/jaxtyping/) solved this
at runtime. numty catches it *before anything runs*, using only what type
checkers already understand plus one small mypy plugin.

## It works on real code: ridge regression from sklearn

This is scikit-learn 1.9.1's dense Cholesky solver
(`_ridge.py::_solve_cholesky`, plus dense centering from
`_base.py::_preprocess_data`), annotated jaxtyping-style and checked
clean — no `type: ignore`, two `TypeVar` lines in the whole file,
everything else minted implicitly from the strings:

```python
from typing import TypeVar

import numpy as np

from numty import F64

n_features = TypeVar("n_features", bound=int)
n_targets = TypeVar("n_targets", bound=int)

def solve_cholesky(
    X: F64["n_samples n_features"],
    y: F64["n_samples n_targets"],
    alpha: F64["n_targets"],
) -> F64["n_targets n_features"]:
    # w = inv(X'X + alpha*Id) X'y
    n_features = X.shape[1]
    n_targets = y.shape[1]

    A = np.matmul(X.T, X)     # (n_features, n_features)
    Xy = np.matmul(X.T, y)    # (n_features, n_targets)

    one_alpha = np.array_equal(alpha, len(alpha) * [alpha[0]])

    if one_alpha:
        A.flat[:: n_features + 1] += alpha[0]
        return np.linalg.solve(A, Xy).T
    coefs = np.empty([n_targets, n_features], dtype=X.dtype)
    for coef, target, current_alpha in zip(coefs, Xy.T, alpha):
        A.flat[:: n_features + 1] += current_alpha
        coef[:] = np.linalg.solve(A, target).ravel()
        A.flat[:: n_features + 1] -= current_alpha
    return coefs
```

The full file (`tests/fixtures/sklearn_ridge.py`) adds mean-centering,
intercept recovery, and a run that asserts `allclose` agreement with
`Ridge(alpha=1.0, solver="cholesky")`:

```text
Success: no issues found in 1 source file
(3, 8) (3,)
```

Adaptations, all documented in the file: dense path only,
`safe_sparse_dot` → `np.matmul` (plain `@` is unchecked by design —
only the named call solves), `scipy.linalg.solve` →
`numpy.linalg.solve`, unweighted means. Pass a
wrongly-sized target and the mistake sklearn raises `ValueError` for at
runtime is caught before anything runs:

```text
error: call-site shape conflict for 'n_samples': 50 (argument 'X', axis 0) vs 40 (argument 'y', axis 0)
```

## Chains compose: reshape, project, slice, reduce, return

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
rejects it too), and rebinding a new shape to an old name is an error,
not a silent widening. This exact function is
`tests/fixtures/chained_block.py`: checked clean, runs to `(5, 7, 16)`.

## One signature, every rank

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
align, batches broadcast through the unpack), `transpose`/`.T`
reversal (`(*Ts, 2)` reverses to `(2, *Ts)`), block-preserving
`reshape` (`(*Ts, 2, 3)` to `(*Ts, 6)`), and leading `None`/slice
indexing without an ellipsis (`v[0]` on `(5, *Ts, 7)` is `(*Ts, 7)`).
Anything that would need the unknown length or order — fancy indexing,
an index reaching into `*Ts` itself, reducing over `*Ts` itself,
mixing different `Ts` declarations, explicit transpose axes, a `-1`
that would absorb `*Ts` — bails out instead of guessing.

## Bugs, caught while you type

A weight matrix passed transposed — the classic silent killer:

```python
logits = np.matmul(batch, weights) + bias
# weights came in as (10, 28) instead of (28, 10)?
# error: matmul inner dimensions disagree: 28 vs 10
```

A bias that broadcasts today and breaks tomorrow:

```python
np.zeros((2, 4)) + np.zeros((3, 4))  # error: broadcast shapes disagree
```

A reshape that silently drops a column:

```python
np.zeros((2, 3)).reshape((2, 4))   # error: reshape changes total size: 6 -> 8
```

Per-channel weights that do not match the axis they average over —
a classic silent-wrong-answer in sensor and finance code:

```python
np.average(np.zeros((128, 6)), axis=1, weights=np.zeros(5))
# error: average weights length 5 does not match axis 1 size 6 (shape (128, 6))
```

A difference along an axis that is not there — and the length math,
checked instead of trusted:

```python
np.diff(np.zeros((4, 3)), axis=2)  # error: axis 2 is out of range for 2 dimensions (shape (4, 3))
np.diff(y, n=2, axis=0)             # clean: a (129, 4) input is (127, 4) out
```

A `dot` that silently misaligns past rank 2 — no batch broadcast to
hide behind, the contracted pair is checked directly:

```python
np.dot(np.zeros((5, 2, 3)), np.zeros((5, 4, 5)))
# error: dot contracted dimensions disagree: 3 vs 4
```

A `tensordot` whose axes don't line up:

```python
np.tensordot(np.zeros((2, 3)), np.zeros((4, 5)), axes=1)
# error: tensordot contracted dimensions disagree: 3 vs 4
np.tensordot(np.zeros((2, 3)), np.zeros((3, 2)), axes=([1], [0]))  # clean: (2, 2)
```

These are not cherry-picked: [bench/](bench) seeds 18
realistic mistakes into MLP-training and math-library code, and the checker
flags 17 of them before anything runs (the 18th is unprovable on purpose —
see limitations). Each error names the axes and the shapes, e.g.
`broadcast shapes disagree at trailing axis 1: 10 vs 2 (shapes (N, 10, 2)
vs (N, 2, 1))`.

## Familiar shapes: `F64[batch, features]`

If you like jaxtyping's look, write that — no converter, no rewrite.
The dtype rides the head name, the dims ride flat args, and everything
below keeps working (matmul, broadcast, reshape, the lot):

```python
from numty import F64, batch, features, hid, inp, matmul

def forward(
    x: F64[batch, inp], w: F64[inp, hid], b: F64[hid],
) -> F64[batch, hid]:
    return matmul(x, w) + b
```

Dims are TypeVars: the ready-made ones (`batch`, `channel`, `height`,
`width`, `depth`, `features`, `classes`, `seq`, `hidden`, `hid`, `inp`,
`out`, `head`) or your own (`N = TypeVar("N", bound=int)`). Same name
unifies — the same rule as above — so mis-wiring the bias is caught at
the call site:

```text
error: call-site shape conflict for 'hid': 8 (argument 'w', axis 1) vs 2 (argument 'b', axis 0)
```

Fixed sizes spell as integers (`F64[3, 4]`), rank polymorphism as
`Unpack[Ts]` with a declared TypeVarTuple (one `# type: ignore[type-var]`
per signature — numpy's own stub bound can't see through `*Ts`, same as
with `FlexArray`). Heads cover the compute workhorses and the common
storage dtypes, so a float16 or int8 block is one head per field rather
than a spelled-out `FlexArray[..., np.dtype[np.float16]]`:

| Head | dtype | Head | dtype |
| --- | --- | --- | --- |
| `F64` | `float64` | `I64` | `int64` |
| `F32` | `float32` | `I32` | `int32` |
| `F16` | `float16` | `I16` | `int16` |
| `Bool` | `bool` | `I8` | `int8` |
| | | `U8` | `uint8` (images) |

Storage blocks are usually built from runtime sizes. A builder keeps the
literal dims of its shape next to the dynamic ones, so a fixed channel
width can be pinned instead of spelled as one more `TypeVar`
(`tests/fixtures/bad_mixed_builder.py`):

```python
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from numty import F16

NB = TypeVar("NB", bound=int)


@dataclass(frozen=True)
class Positions(Generic[NB]):
    xy: F16[NB, 2]


def build(n: int) -> Positions[int]:
    xy = np.zeros((n, 3), dtype=np.float16)  # tuple[int, Literal[3]]
    return Positions(xy)
```

```text
error: Argument 1 to "Positions" has incompatible type "ndarray[tuple[int, Literal[3]], dtype[floating[_16Bit]]]"; expected "ndarray[tuple[int, Literal[2]], dtype[floating[_16Bit]]]"  [arg-type]
```

Prefer jaxtyping's string look? Write that instead — one spec per
parameter, shared names tie the signature together, same solver
underneath:

```python
from numty import F64, matmul

def matrix_multiply(x: F64["dim1 dim2"], y: F64["dim2 dim3"]) -> F64["dim1 dim3"]:
    return matmul(x, y)
```

No imports beyond the heads, no `TypeVar` boilerplate — `dim1`,
`dim2`, `dim3` are minted implicitly from the strings. (The function
above is `tests/fixtures/f64_str_ok.py` verbatim: it type-checks clean
and runs, printing `(5, 3)` first.)

Seed a bug — `mm(zeros((2, 4)), zeros((5, 3)))` — and the shared `dim2`
is caught at the call site:

```text
error: call-site shape conflict for 'dim2': 4 (argument 'x', axis 1) vs 5 (argument 'y', axis 0)
```

String words resolve exactly like bare dims, so the styles mix freely:
`F64["batch features"]` and `F64[batch, features]` (or even one of each
in a single signature) mean the same thing. Only space-separated names
in v1 — a stray `"batch 3"` fails loudly with a redirect, same as
every other misuse. At runtime a multi-word spec erases to the bare
head (`F64["batch features"] is F64`), so importing the module never
crashes on the annotation; specs that already subscribe fine
(`F64["hid"]`, `F64[3, 4]`) behave exactly as before.

One honest boundary: multi-word specs need no declarations at all —
`dim1`, `dim2`, `dim3` above are minted implicitly. A lone single-word
spec (`F64["hid"]`) is a forward reference, which mypy resolves before
any plugin runs, so it still needs a resolvable name: one of the
thirteen prelude dims or a one-line `hid = TypeVar("hid", bound=int)`.
Proven with a lobotomized-hook experiment (hook returning `Any` without
touching args still yields the `Name "hid" is not defined` error), so no
plugin-side fix exists on this mypy version.

Two rules, both honest. First, every bare name must resolve: mypy binds
function type variables by looking names up *before* any plugin runs,
so an undeclared bare dim is an error, not a silent hole — `"hid"` and
`hid` mean the same thing, and `batsh` fails the way any typo fails.
(Quoted multi-word specs are the exception: their words mint implicit
dims, per the boundary note above.)
Second, misuse fails loudly with a redirect, never a silent `Any`:
`F64[tuple[N, M]]` tells you to write `F64[N, M]`, `F64[()]` and `_`
tell you what's wrong. Quoted or bare is a style choice, not a semantic
one.

## Named sizes: one name, everywhere it matters

Magic numbers in two places drift apart. A `Literal` alias gives a
size one name that works in every annotation, and every message still
spells the resolved numbers:

```python
from typing import Literal
import numpy as np
from numty import F64

N = Literal[128]  # samples in
M = Literal[127]  # samples out: one difference (the checker verifies this)
C = Literal[6]    # channels

def profile(frames: F64[N, C], mix: F64[C]) -> F64[N]:
    return np.average(frames, axis=1, weights=mix)

def step_back(y: F64[N, C]) -> F64[M, C]:
    return np.diff(y, axis=0)   # checked: 128 - 1 = 127
```

Rename `M` to 126 and the return no longer lines up; pass five
weights where six channels are named and the call fails with the
numbers, not the names:

```text
error: Incompatible return value type (got "ndarray[tuple[Literal[127], Literal[6]], dtype[float64]]", expected "ndarray[tuple[Literal[126], Literal[6]], dtype[float64]]")
error: Argument 2 to "profile" has incompatible type "ndarray[tuple[Literal[5]], dtype[float64]]"; expected "ndarray[tuple[Literal[6]], dtype[float64]]"
```

The clean file is `tests/fixtures/named_dims.py` (checked clean,
runs to `(128,) (127, 6)`); the two errors are pinned by
`tests/fixtures/bad_diff_size.py` and
`tests/fixtures/bad_named_dims_weights.py`.

## Unknown dims: dynamic shapes stay checkable

Some sizes are unknowable statically — a mask's hit count, a
loop-built list's length, a `-1` over symbolic dims. Instead of
failing or dissolving to `Any`, the checker mints a fresh unknown
per unknowable axis (`?U@line`) and keeps checking everything around it:
ranks, known dims, and downstream ops.

```python
def masked_mean(y: F64[N, M], keep: Bool[N]) -> F64[M]:
    return np.mean(y[keep], axis=0)   # (?U, M) -> (M,) ✓

def dyn_concat_mean(blocks: list[F64[N, M]]) -> F64[M]:
    return np.mean(np.concatenate(blocks, axis=0), axis=0)
```

A rank clash on a *known* dim still fires with the unknown next to it,
and asserting a concrete size for dynamic data is correctly rejected:

```text
error: matmul inner dimensions disagree: 4 vs 5
error: Incompatible types in assignment (expression has type "ndarray[tuple[?U@13, M], dtype[float64]]", variable has type "ndarray[tuple[Literal[3], M], dtype[float64]]")
```

Each unknown is named after the line that minted it (`?U@13` is the
mask on line 13 of `tests/fixtures/bad_unknown_concrete.py`), so
unrelated unknowns print differently. In
`tests/fixtures/unknown_names_ok.py`, two masks on lines 17 and 18 give
an outer product of `tuple[?U@17, ?U@18]`, not a square, and padding a
mask taken on line 24 gives `?U@24+2`. A second unknown minted on the
same line gets a letter: `a, b = x[x > 0.0], x[x > 1.5]` on line 7
of a file gives `?U@7` and `?U@7b` (numpy: `(4,)` and `(3,)` for
`x = [1., 2., 2., 3.]`).

The boundary is principled: unknowns flow through bodies but cannot be
named in signatures — there is no spelling for "somebody else's count".
Reduce them, or leave the position to inference.

## Real generic code, checked end to end

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
stay silent by design — their equality is unprovable, and unknown is not
an error. Apply the alias explicitly at each use (`A[N, K, C]`, never bare
`A` — a bare generic alias silently means unknown sizes).

## Coming from jaxtyping? Convert once, check forever

Already have `Float[np.ndarray, "batch features"]` everywhere? For new
code, prefer the native spelling above — same feel, no middleman. For
existing files, a plugin cannot read jaxtyping's dim strings — they ride
inside `Annotated` metadata, which no mypy type-analyze hook ever sees
(probed every hook: only jaxtyping's internals fire, the string never
arrives). So the bridge is a one-time source rewrite instead of a hook:
[numty_convert.py](numty_convert.py) mechanically
translates the annotations, then gets out of the way:

```sh
python numty_convert.py mlp.py mlp_static.py   # warnings go to stderr, if any
```

```python
# before
def forward(x: Float[np.ndarray, "batch inp"], ...) -> Float[np.ndarray, "batch out"]: ...
# after (import of jaxtyping dropped — nothing references it anymore)
BATCH = TypeVar('BATCH', bound=int)
INP = TypeVar('INP', bound=int)
def forward(x: FlexArray[tuple[BATCH, INP], np.dtype[Any]], ...) -> FlexArray[tuple[BATCH, OUT], np.dtype[Any]]: ...
```

Names become shared TypeVars — the same same-name-unifies rule as above,
case-sensitive like jaxtyping (`B` and `b` stay two variables) —
fixed sizes become `Literal`s, `...`, `_` and unchecked `_name` become
fresh variables, `*name` becomes a named `*Ts` (with the
`# type: ignore[type-var]` such signatures need), exact dtypes (`Float64`,
`Int32`, …) keep their precision, and family heads (`Float`, `Int`, …)
keep the shape while the dtype stays `Any`. Only `def` parameter and
return annotations convert. The rest of the file is left byte for byte:
comments, pragmas and blank lines survive, and a file with nothing to
convert comes back unchanged. It catches real mistakes: mis-annotate the output
bias as `"hid"` instead of `"out"` and the converted file fails with

```text
error: call-site shape conflict for 'HID': 8 (argument 'w1', axis 1) vs 2 (argument 'b2', axis 0)
```

Anything the static solver cannot express is left verbatim with a
warning, never silently changed: symbolic dims (`"n+1"`), broadcastable
`#name` dims (size 1 *or* the name — no static spelling), two variadics,
stacked subscriptions, non-numpy bases, aliased heads (`Float as F`),
string annotations, jaxtyping subscripts outside a signature (module
variables, `isinstance`), and names your code already binds (a
module-level `BATCH = 5` or `from config import BATCH` would be rebound by
the emitted decl, so that annotation stays jaxtyping). A fully converted
file runs unchanged — the annotations evaluate — and no longer needs
jaxtyping installed. The round trip is pinned by
[tests/test_convert.py](tests/test_convert.py):
goldens for every dim kind plus an end-to-end MLP that checks clean and a
seeded-bug twin that must fail.

## Indexing that tracks shapes

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
size 1); a 1-D boolean mask turns its axis into a fresh unknown
(`a[np.ones(3, dtype=bool)]` on `(3, 4)` is `(?U0, 4)`, see unknown dims
above), and unknown bounds bail out. An n-D mask is still typed as if it
masked only the first axis (`a[a > 0]` on `(3, 4)` is `(?U0, 4)`, numpy returns
1-D; [#49](https://github.com/sofeikov/numty/issues/49)). A full slice on a
symbolic axis keeps the TypeVar (`a[:, :]` on `(N, K)` stays `(N, K)`);
a bounded slice on one (`a[:, :k]`) bails out. Indexing every axis
with integers yields a scalar, which the plugin leaves to the stubs.

Integer arrays index fancy-style: the index shape replaces the axis,
broadcast when there are several; `take_along_axis` broadcasts the two
shapes, with the index winning along the axis. Known bug: an index of
size 1 along the axis keeps the array's dim instead
(`take_along_axis(a, idx, axis=1)` with `a: (3, 4)`, `idx: (3, 1)` is
typed `(3, 4)`, numpy returns `(3, 1)`;
[#46](https://github.com/sofeikov/numty/issues/46)):

```python
table[ids]                        # (5, 4): index shape + trailing dims
b[np.arange(5), actor]            # (5,): broadcast (5,) + (5,)
np.take_along_axis(a, idx, axis=1)  # broadcast shape, arr's dtype
b[np.zeros(3)]                    # error: integer (or boolean) type required
np.take_along_axis(a, idx2d, axis=1)  # error: same number of dimensions
```

## Unions that check every member

A union-typed array (`(N, K, M) | (N, 1, M)`, or a loop variable over
mixed shapes) solves once per member and unions the results, so bodies
that used to dissolve to `Any` now check each shape:

```python
b = np.where(valid, distance, np.inf)  # union in, union of precise out
clean = [np.where(np.isfinite(v), v, np.nan) for v in (p1, p2)]
```

Members pair index-wise, which is exact for co-derived unions (the same
value mapped through ops — the overwhelmingly common source). Unrelated
unions of different member counts bail out instead of guessing, and
dunder ops with unions on both sides (`u + u`) still go through mypy's
native cartesian matching, so cross terms may dissolve.

Unpacking is the one place precision does not follow the members: a
generator, list comprehension, `tuple(...)` call, starred target, or
`map` assigns the union to every name (mypy semantics — `map` dissolves
further to `Any`). Only the tuple display unpacks per name:

```python
a, b = (clean(p1), clean(p2))   # (N, 2) and (N, K, 2): precise
a, b = (clean(v) for v in (p1, p2))  # both: the union (body still checked)
```

So when the shapes must reach the code below, spell the display; when
only the per-member checking matters, the generator form is fine.

## The rest of the catalog

**Construction.** `zeros_like` keeps shape and dtype, `full((2, 3), 5.0)`
is a known 2x3 float64, plus `eye(3)`, `arange(6)`, and `atleast_2d`
promoting `(N,)` to `(1, N)`.

**Joining.** `concatenate` adds dims on the axis, `stack` adds the
stacked axis, `vstack`/`hstack`/`dstack`/`column_stack` promote then
concatenate — including enum-driven stacks; mismatches error.

**Reshape.** Conserves the total with symbolic sizes and infers `-1`
(`(N, 10, 2)` to `(N, 20)` is fine, `(N, 10, 2)` to `(N, 21)` errors,
`(N, 10)` to `(M, 10)` bails). Regroups block-preserving runs over
`*Ts` (`(*Ts, 2, 3)` to `(*Ts, 6)` is fine; a `-1` that would absorb
`*Ts`, or a total no `*Ts` can absorb, bails or errors respectively).

**Rank changes.** `transpose`/`.T` (reversal included over `*Ts`),
`squeeze`, `expand_dims`; bad permutations, repeated axes and non-1
squeeze axes error.

**Reductions.** `sum(axis=1)` (plus `median` and every `nan*` twin:
`nansum`, `nanmean`, `nanargmin`, …), `average` (1-D weights must match
the averaged axis), `diff` (concrete lengths subtract, bad axes and
negative orders error), `cumsum`/`cumprod` (shape-preserving).

**Matmul family.** `matmul` solves inner dims and batch-broadcasts
through shared `*Ts` (a lone concrete 1-D/2-D operand contributes an
empty batch). The rest solves through the same inner-dim machinery:
`dot` (no batch broadcast past rank 2), `tensordot` (int or pair-of-lists
axes), `inner`/`outer`/`vdot` (scalars included), and `take` (the axis is
swapped for the index shape; literal indices past the axis error unless
`mode` clips or wraps). `matmul` on a scalar or 0-d operand errors, and
`linalg.norm` with an explicit `ord` needs one or two reduced axes.

**Indexing.** Consumes leading dims without an ellipsis (`v[0]`,
`v[None, ...]`). Comparisons broadcast and return bool arrays; `argmax`
drops the axis, `sort` keeps the shape, and a literal `partition` kth
past the axis errors.

**Shape math.** `flip`/`fliplr`/`flipud`/`roll` preserve (with rank floors
and axis checks; `flip` rejects a repeated axis, `roll` a shift that does
not match its axes), `rot90` swaps dims on odd turns (equal axes error),
`gradient` keeps the shape for one resolved axis, `convolve`/`correlate`
compute full/valid/same lengths (empty operands error). The fft family tracks complex dtypes — `rfft` halves (`N//2+1`),
`fftn`/`rfftn` apply `s` over `axes` with trailing alignment; bad axes and
non-positive sizes error.

**Data prep.** `tile` multiplies reps, `repeat` scales the axis
(flattening to one dim for `axis=None`; a count list of several entries
must have one per element and literal counts sum to the new size, other
array repeats give `?U`), `pad`
adds widths per axis, `split` returns a list of equal parts (`?U` for
uneven index splits); negative counts and uneven divisions error.

**Selection.** `unique` keeps the dtype with a `?U` length (rank kept for
`axis=`), `nonzero`/`where` with one argument return per-dim `?U` tuples
(`flatnonzero` flattens, `argwhere` pairs), `isin` keeps the input shape
with a bool dtype, `searchsorted` follows the values shape; 0-d nonzero
errors.

**Symbolic arithmetic.** Dims stay exact through arithmetic:
`repeat`/`tile` multiply (`N*2`), `pad` adds (`N+2`), `convolve` adds
lengths (`N+M-1`), `rfft` halves (`N//2+1`), `split` divides (`N//2`),
and reshape proves `N*2` vs `2*N` (dividing `-1` exactly); mismatches
error with readable formulas. `sliding_window_view` appends the window
after the shrunken dims (`N-w+1`, symbolic included — the conv-output
formula falls out).

What that unlocks — code that failed before, exact now:

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

**Elementwise, einsum, linalg, random.** Operators and bare `np.*`
ufuncs alike — broadcast shapes and promote dtypes. Add-style promotion
(`+`, `-`, `*`) matches NumPy (verified against `np.result_type` over
the full dtype matrix, plus weak Python scalars). True division mixing a
`float16`/`float32` with a bool or 8/16-bit int, and float ufuncs on
small ints, do not yet: they are typed `float64` where NumPy picks a smaller float
(`float32 / uint8` and `np.sqrt` of `int8` give `float32` and `float16`
in NumPy; [#61](https://github.com/sofeikov/numty/issues/61)).
`einsum` parses the subscript string (explicit, implicit,
ellipsis, diagonals) and `linalg` checks squareness and inner dims while
tracking dtypes (`eigh` stays real, `svd` shrinks with
`full_matrices=False`). `Generator` random methods solve from `size=`. Tuple and list displays keep per-item precision. An explicit
`dtype=` you name is used as-is when it resolves; anything exotic bails
out instead of guessing.

## Your own sizes, in one line

No pre-baked list from the library — exact sizes live in your code,
one line each, in standard typing that needs no plugin to write (what
each checker catches without the plugin is in "How it works" below):

```python
from typing import Literal
from numty import FloatArray

A2x3 = FloatArray[tuple[Literal[2], Literal[3]]]

def use_2x3(m: A2x3) -> None: ...

use_2x3(np.zeros((3, 2)))  # error: 3x2 is not 2x3
```

Rank-only aliases (`Vector`, `Matrix`) catch transposed/flattened
mixups even more cheaply. Named sizes work in expressions — mark them
`Final` and the solver folds names and arithmetic (`H * 2`, `W + 1`,
`M // 2`) just like literals, so `np.zeros((H * 2, 3))` is `(56, 3)`
for `H: Final = 28`, and `len()` of an `IntEnum` class folds to its
member count (aliases share one value, so they count once). A `Final`
name is not a dim in an annotation: `I64[H, 2]` fails with `I64 dim
"H" must be a TypeVar, an integer, Unpack[Ts], or Literal[...]` and
the parameter becomes `Any`. Name an annotation size with a `Literal`
alias instead (`H = Literal[28]`, see "Named sizes" above). Plain
variables and parameters stay unknown: their values are gone from the
types before the plugin sees them, so they bail out instead of
erroring. A generic `int` parameter can never be verified against a
shape — the contract says "every int", and no annotation survives
the join to `int`.

Narrowing a dynamically-shaped value uses a witness instead of a
per-shape helper:

```python
use_2x3(narrow(m, np.zeros((2, 3))))  # checked at runtime, precise statically
```

`narrow` returns `m` itself when its shape and dtype equal the witness's,
and otherwise raises `numty.NarrowError` (a `ValueError`, and also an
`AssertionError` for older callers), e.g. `narrow: shape (3, 2) does not
match witness (2, 3)`. The check is an explicit raise, not an `assert`,
so it still runs under `python -O`.

At runtime the heads are annotations, not array classes: calling
`F64()` raises `TypeError`. `isinstance(x, F64)` is true exactly when
`x` is an `ndarray` whose dtype equals native float64 (likewise every
head with its own dtype, e.g. `U8` and uint8; a byte-swapped `'>f8'`
array is not an `F64`), so
when mypy narrows a union of arrays by head, the runtime branch agrees.
Rank and dims are not checked at runtime. A subscripted head
(`isinstance(x, F64[3])`) raises `TypeError` like any subscripted
generic, except a multi-word string spec, which erases to the bare head
(`isinstance(x, F64["a b"])` is the dtype check alone); use `narrow` or
compare `x.shape` for dims. `issubclass(np.ndarray, F64)` stays false.

## Quickstart

```sh
uv venv
uv pip install mypy numpy pytest ty==0.0.97
```

```sh
PYTHONPATH=. uv run mypy demo.py      # expect 2 errors: rank, size
uvx ty@0.0.97 check numty.py demo.py  # the same 2 errors, none in numty.py
uv run pytest                         # full suite: runtime, mypy- and ty-driven
```

The ty run ends in `Found 2 diagnostics`, both `invalid-argument-type`
on `demo.py:30` and `demo.py:34`; `uvx pyright@1.1.408 --pythonpath
.venv/bin/python numty.py demo.py` reports the same two (without
`--pythonpath` it cannot find numpy). ty 0.0.97 is pinned in the dev group,
and `tests/test_ty.py` holds the ty line to exactly those two.

mypy skips the bodies of unannotated functions by default, so a script
or notebook cell written as `def main():` gets no shape checking at all.
Run with `--check-untyped-defs` (or set `check_untyped_defs = True` in
your `mypy.ini`) to have the solver check them too.

## Error codes

Every numty diagnostic carries one of six mypy error codes, so each kind
can be silenced or selected on its own:

| Code | Covers | Example message |
|---|---|---|
| `numty-shape` | dims, ranks or size arguments that do not fit the operation (matmul, reshape, concatenate, linalg, einsum, …) | `matmul inner dimensions disagree: 3 vs 2` |
| `numty-broadcast` | elementwise operands (arithmetic, comparisons, ufuncs, `where`, `broadcast_*`, index arrays) that do not broadcast | `broadcast shapes disagree at trailing axis 0: 3 vs 4` |
| `numty-callsite` | one shape variable bound to two sizes by the arguments of a call | `call-site shape conflict for 'batch': 4 (argument 'x', axis 0) vs 5 (argument 'y', axis 0)` |
| `numty-axis` | axis or index arguments out of range, duplicated or mismatched; zero slice steps; non-integer index arrays; einsum output labels | `axis 2 is out of range for 2 dimensions (shape (4, 3))` |
| `numty-bail` | correct code numty cannot check and says so loudly (stacks built dynamically) | `stack needs a sequence of known length (built dynamically; preallocate or ignore)` |
| `numty-annotation` | malformed `F64[...]`-style annotations | `F64 takes at least one dim` |

For example, `codes.py`:

```python
import numpy as np
from numty import F64, batch


def pair(x: F64[batch], y: F64[batch]) -> None: ...


def demo(blocks: list[F64[3]], bad: F64[()]) -> None:
    a, b = np.zeros((4, 3)), np.zeros((2, 5))
    np.matmul(a, b)                     # numty-shape
    pair(np.zeros(4), np.zeros(5))      # numty-callsite
    np.where(a > 0, a, np.zeros(4))     # numty-broadcast
    a.sum(axis=2)                       # numty-axis
    np.stack([b for b in blocks])       # numty-bail
```

```text
codes.py:8: error: F64 takes at least one dim  [numty-annotation]
codes.py:10: error: matmul inner dimensions disagree: 3 vs 2  [numty-shape]
codes.py:11: error: call-site shape conflict for 'batch': 4 (argument 'x', axis 0) vs 5 (argument 'y', axis 0)  [numty-callsite]
codes.py:12: error: where shapes disagree: broadcast shapes disagree at trailing axis 0: 3 vs 4 (shapes (4, 3) vs (4,))  [numty-broadcast]
codes.py:13: error: axis 2 is out of range for 2 dimensions (shape (4, 3))  [numty-axis]
codes.py:14: error: stack needs a sequence of known length (built dynamically; preallocate or ignore)  [numty-bail]
```

Code that stacks or concatenates lists built at runtime can turn the
bails off and keep every real shape error:

```ini
[mypy]
plugins = numty_plugin
disable_error_code = numty-bail
```

or per line, `np.stack([b for b in blocks])  # type: ignore[numty-bail]`, which still
reports a shape error on that line. All six are sub-codes of `misc`, so
an existing `# type: ignore[misc]` or `disable_error_code = misc` keeps
covering them. Under `--warn-unused-ignores` (part of `--strict`), mypy
then asks for the narrower code (`Unused "type: ignore" comment, use
narrower [numty-shape] instead of [misc] code`), so switch those
comments to the `numty-*` code when upgrading.

## How it works

Three layers, each building on the previous one:

| Layer | Idea | Works under |
|---|---|---|
| Rank aliases | `Matrix = ndarray[tuple[int, int], ...]` — tuple length is the rank | mypy, pyright, ty |
| Your own sizes | `A2x3 = FloatArray[tuple[Literal[2], Literal[3]], ...]`, or `Array2D[Literal[2], Literal[3]]` | mypy, pyright, ty |
| Shape solver | [numty_plugin.py](numty_plugin.py) hooks constructors, `matmul` (incl. `(*Ts, N, M)` pairs), arithmetic ops and ufuncs (with promotion), `where`/`clip`/`broadcast_to`, `reshape` (incl. symbolic conservation and `*Ts` block regroups), rank-changing ops (`transpose`/`.T` reversal incl. over `*Ts`), shape builders, comparisons, bool logic (`&`/`\|`), `any`/`all`, the arg family, `einsum`, `linalg` solvers, fancy indexing, `take_along_axis`, symbolic TypeVar dims (same name unifies, ranks check, the rest bails), call-site TypeVar conflicts, enum-driven stacks, folds constant int expressions and `len()` (`Final` arithmetic, enum member counts), solves shapes itself | mypy only |

The first two layers are pure standard typing — no magic, portable everywhere.
The library ships only the generic bases (`FloatArray`, `FlexArray`,
`Array1D`, `Array2D`) plus rank aliases; exact sizes live in your code,
one line each. Plain constructors just work: `np.zeros((2, 3))` already
has a precise type, and the plugin keeps it precise even through
assignment.

Layers 1–2 hit a wall at cross-argument checks: `ndarray`'s shape parameter is
effectively covariant, so a shared `TypeVar` silently joins conflicting
literals (3 vs 2 becomes `int`) instead of erroring. The plugin stops asking
the type system to solve shapes and does the solving itself, with readable
errors like `matmul batch dimensions disagree: (5,) vs (4,)` — including
a call-site check that catches the shared-`TypeVar` conflict the join
hides (`f(x_5, y_6)` against `(A[N], A[N])`).

The plugin is registered in [mypy.ini](mypy.ini) and
needs `PYTHONPATH=.` so mypy can import it (`MYPYPATH` does not work for
plugins). Without the plugin, only annotations spelled with the first
two layers keep any checking, and under plain mypy less than the table
suggests: rank aliases still catch rank mixups, but mypy widens a
constructor's sizes to `int` once the array is assigned to a name, so
exact-size aliases then reject correct code. `mypy demo.py` with no
plugin configured reports 4 errors, 3 of them on correct code, and
misses the 3x2 size bug. The `F64`-style heads (`F32`, `U8`, ...) need
the plugin: without it a correct call is rejected (`f(np.zeros((2, 3)))`
against `F64[batch, features]` fails with `expected "F64[Never,
Never]"` under plain mypy) and results become `F64[Any, Any]`
([#5](https://github.com/sofeikov/numty/issues/5)).

## Limitations, stated plainly

- The plugin is mypy-only and pinned to mypy internals that occasionally
  shift. `ty` and pyright users get layers 1–2, which still catch rank and
  fixed-size bugs. They do not get the `F64`-style heads: ty 0.0.97
  rejects correct calls into head-annotated functions too (a `(2, 3)`
  array passed to `F64[batch, features]` is an `invalid-argument-type`
  error; [#5](https://github.com/sofeikov/numty/issues/5)).
- **Adopting numty in a repo that gates on ty (or pyright) alone:** the
  heads give that gate no shape checking at all. Every call that passes
  an `ndarray` into an `F64[...]` signature is an error there, correct or
  not, so the heads
  cost the full annotation effort and catch nothing. Shapes that must
  hold under such a gate go in layers 1–2 or `FlexArray`, which are
  plain `ndarray` types to every checker:

  ```python
  # fixed rank: either spelling works under mypy+plugin, ty and pyright
  def f(x: FlexArray[tuple[N, T, C], np.dtype[np.float16]]) -> ...: ...
  def f(x: np.ndarray[tuple[N, T, C], np.dtype[np.float16]]) -> ...: ...
  ```

  ty checks rank, `Literal` sizes and dtype through these, and the
  rank-generic `squared_norm` above also works there: ty 0.0.97 reveals
  `(5, 7)` for a `(5, 7, 2)` input and rejects `(5, 7, 3)` with
  `invalid-argument-type` (the two `type: ignore[type-var]` notes are
  for mypy; ty does not need them). pyright 1.1.408 flags the same two
  bugs in [tests/fixtures/flex_portable.py](tests/fixtures/flex_portable.py),
  which the suite checks under mypy+plugin and ty. Neither does any
  solving: `matmul` inner dims, broadcasts and call-site `TypeVar`
  conflicts pass silently and stay mypy+plugin only. Making the heads
  themselves rank-check without the plugin is still open in
  [#5](https://github.com/sofeikov/numty/issues/5).
- Without the plugin, `matmul` returns a loose type: usable but unchecked.
- Dynamic shapes (`np.zeros((n, 3))`) cannot be proven wrong, so they are
  left alone. Unknown is not an error. A builder (`zeros`/`ones`/`empty`/
  `full`, `*_like(shape=...)`, `Generator` `size=`) keeps the literal dims
  of a tuple display next to plain `int` ones: `np.zeros((n, 3))` is
  `tuple[int, Literal[3]]`, also once assigned. The solver does not read
  `int` dims, so ops on such an array (or on one annotated that way) use
  numpy's stubs. Shape-keeping ones (arithmetic, `astype`, `copy`,
  `flip`, sorts along an axis, ...) keep the literal, and transposes move
  it with its axis. For the rest it widens to `int`, because the stubs
  would leave it on the wrong axis after `swapaxes`, `pad` or `diff`. Written
  inline as an argument whose expected literal differs, a builder falls
  through to numpy's catch-all overload and goes unchecked, as before.
  Assign it to a name first to have it checked.
- An explicit `dtype=` is honored when it resolves (`np.float32`,
  `"float64"`, bare `int`/`float`/`bool`, ...); exotic forms bail out.
  Same for unknown `keepdims`.
- On lines the solver rejects, mypy may add its own follow-on error next
  to ours (an overload complaint for tuple `reshape`, `Unsupported
  operand` for broadcast ops, `Invalid index type` for an out-of-range
  tuple index, a `DTypeT` type-var complaint for `fft` with a bad `n`).
  Ours is the readable one.
- Test runs use a dedicated mypy cache dir cleared per suite run (see
  `test_static.py`): mypy does not reliably invalidate plugin results
  across plugin edits, so never trust the default cache while hacking
  on the plugin.
- The `@` operator is accepted but unchecked: operator dispatch consulted
  the method hook inconsistently in testing. Raw `np.matmul` solves
  through the ufunc hook instead, as does the `numty.matmul` wrapper.
  Same for reflected ops beyond the hooked pairs.
- numty needs mypy to follow imports (the default). Under
  `--follow-imports=skip` mypy skips `numty` itself, so every
  `F64[...]`-annotated parameter and return becomes `Any` and nothing
  through them is checked, with no warning. The ufunc generic aliases
  also stay unresolved, so bare `np.sqrt`-style calls bail out;
  operators, methods, and real `def` stubs on plain numpy arrays are
  unaffected. If you need `skip`, keep following `numty` with a
  per-module override, which brings the heads back (ufunc calls still
  bail):

  ```ini
  [mypy-numty]
  follow_imports = normal
  ```
- Unannotated functions are not checked: mypy skips their bodies unless
  `check_untyped_defs` is on (see Quickstart).
- An inline call to an overloaded function nested directly inside an
  overloaded numpy call dissolves to `Any` (mypy keeps the
  overload-inferred type over the hook result): hoist it to a local
  first (`kernel = gaussian_kernel(...); np.where(cond, kernel, 0.0)`).
- Union-typed array inputs (`(N, K, M) | (N, 1, M)`) distribute per
  member: each member solves with the full solver and the results
  union. Members pair index-wise (exact for co-derived unions);
  unrelated unions of different member counts, and dunder ops with
  unions on both sides (mypy matches those cartesianly), bail out
  instead of guessing.
- Rank-polymorphic helpers (`(..., 2)` → `...`) spell the shape
  through `FlexArray` with one `*Ts` signature: each signature carries
  two `type: ignore[type-var]` notes (the `tuple[int, ...]` stub bound
  cannot see through an unsolved `*Ts`), while bodies and call sites
  check fully. A provable `*Ts` error may draw the stubs' own echo next
  to ours (an `[index]` complaint on unpack shapes); ours names the
  axis, the size, and the `(*Ts, ...)` shape.
- Rank arithmetic over `*Ts` keeps the unknown block ordered, never
  reordered: `transpose`/`.T` reversal moves the known runs around a
  stationary `*Ts` (no annotation can spell its reversal), explicit
  transpose axes bail, and a `reshape` `-1` bails because the inferred
  dim would absorb the `*Ts` product. A non-ellipsis index that reaches
  past the known prefix bails — rank-generic bodies spell the ellipsis
  (`a[:, ..., None, :]`); a `1` that aligns against `*Ts` from the
  wrong side bails for the same reason. `matmul` needs one shared
  `Ts` and two known trailing dims per unpack side; vectors over
  `*Ts`, mixed unpacks, and concrete batches against `*Ts` bail.
- Heterogeneous unpacking (`a, b = (f(p) for p in (p1, p2))` with
  different shapes) assigns the whole union to every name — genexp,
  list comprehension, `tuple(...)`, starred targets, and `map` (which
  dissolves to `Any`) all behave this way. The body still checks per
  member; per-name precision needs the tuple display.
- Different TypeVars (`N` vs `M`) never conflict, even when provably
  different at a call site — only same-name TypeVars are unified.

## Layout

- [numty.py](numty.py) — rank aliases, generic shape
  bases, `FlexArray` rank-polymorphic base, `narrow` witness helper,
  shape-preserving `ident`, loose `matmul` fallback, familiar-style heads
  (`F64` and friends) with ready-made dims (`batch`, `features`, …).
- [numty_plugin.py](numty_plugin.py) — the solver.
- [demo.py](demo.py) — good and bad usages, clean under
  both checkers except the two intentional errors.
- [bench/](bench) — realistic benchmark (MLP training,
  math library, 17 of 18 seeded mistakes caught), with its own scorecard in
  [bench/README.md](bench/README.md).
- [tests/test_runtime.py](tests/test_runtime.py) —
  shapes, values, narrowing accept/reject (also under `python -O`),
  `isinstance` against heads.
- [tests/test_static.py](tests/test_static.py) — runs
  real mypy over [fixtures](tests/fixtures/static_ok.py):
  each bad one must fail with our message, and `reveal_type` pins must show
  precise shapes on the good paths. The `f64_ok` + `bad_f64_*` fixtures do
  the same for the familiar-style heads (clean end-to-end MLP included).
- [tests/test_ty.py](tests/test_ty.py) — runs the pinned ty over
  `numty.py demo.py` (exactly the 2 demo errors) and over
  [flex_portable.py](tests/fixtures/flex_portable.py) (`FlexArray`
  keeps its rank checks without the plugin).
- [numty_convert.py](numty_convert.py) +
  [tests/test_convert.py](tests/test_convert.py) —
  the jaxtyping bridge: one-time annotation rewrite with goldens for every
  dim kind and an end-to-end MLP (clean + seeded-bug twins in
  [fixtures](tests/fixtures/jaxtyping_mlp.py)).

## Roadmap

Frontiers, ordered by value: per-name precision past lazy unpacks
(a tuple display is the only form mypy's binder distributes — genexp,
listcomp, `tuple(...)`, starred, and `map` all assign the union);
unpack-vs-concrete merges (provably unknowable today, but narrower bails
are possible); NEP 50's value-based edge cases (out-of-bound Python ints)
and union dtypes; the `@` operator (mypy consults the method hook
inconsistently); `ty` and pyright (neither offers a plugin API to build
the solver on).
