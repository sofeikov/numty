# How it works

Three layers, each building on the previous one:

| Layer | Idea | Works under |
|---|---|---|
| Rank aliases | `Matrix = ndarray[tuple[int, int], ...]` — tuple length is the rank | mypy, pyright, ty |
| Your own sizes | `A2x3 = FloatArray[tuple[Literal[2], Literal[3]], ...]`, or `Array2D[Literal[2], Literal[3]]` | mypy, pyright, ty |
| Shape solver | [numty_plugin](https://github.com/sofeikov/numty/tree/main/numty_plugin/) hooks constructors, `matmul` (incl. `(*Ts, N, M)` pairs), arithmetic ops and ufuncs (with promotion), `where`/`clip`/`broadcast_to`, `reshape` (incl. symbolic conservation and `*Ts` block regroups), rank-changing ops (`transpose`/`.T` reversal incl. over `*Ts`), shape builders, comparisons, bool logic (`&`/`\|`), `any`/`all`, the arg family, `einsum`, `linalg` solvers, fancy indexing, `take_along_axis`, symbolic TypeVar dims (same name unifies, ranks check, the rest bails), call-site TypeVar conflicts, enum-driven stacks, folds constant int expressions and `len()` (`Final` arithmetic, enum member counts), solves shapes itself | mypy only |

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

The plugin is registered in [mypy.ini](https://github.com/sofeikov/numty/tree/main/mypy.ini) and
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
