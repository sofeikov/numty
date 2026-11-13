# Limitations, stated plainly

numty prefers silence to a wrong claim. This page lists where it goes
quiet, where it is approximate, and what to write instead.

## Other type checkers

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
    rank-generic [`squared_norm`](../examples/rank-generic.md) also works there: ty 0.0.97 reveals
    `(5, 7)` for a `(5, 7, 2)` input and rejects `(5, 7, 3)` with
    `invalid-argument-type` (the two `type: ignore[type-var]` notes are
    for mypy; ty does not need them). pyright 1.1.408 flags the same two
    bugs in [tests/fixtures/flex_portable.py](https://github.com/sofeikov/numty/tree/main/tests/fixtures/flex_portable.py),
    which the suite checks under mypy+plugin and ty. Neither does any
    solving: `matmul` inner dims, broadcasts and call-site `TypeVar`
    conflicts pass silently and stay mypy+plugin only. Making the heads
    themselves rank-check without the plugin is still open in
    [#5](https://github.com/sofeikov/numty/issues/5).
- Without the plugin, `matmul` returns a loose type: usable but unchecked.

## mypy behaviour

- On lines the solver rejects, mypy may add its own follow-on error next
    to ours (an overload complaint for tuple `reshape`, `Unsupported
    operand` for broadcast ops, `Invalid index type` for an out-of-range
    tuple index, a `DTypeT` type-var complaint for `fft` with a bad `n`).
    Ours is the readable one.
- Test runs use a dedicated mypy cache dir cleared per suite run (see
    `test_static.py`): mypy does not reliably invalidate plugin results
    across plugin edits, so never trust the default cache while hacking
    on the plugin.
- The `@` operator solves through the same matmul hooks as raw
    `np.matmul` and the `numty.matmul` wrapper (`__matmul__`/`__rmatmul__`,
    with `__imatmul__` checked in place): wrong shapes fail with the same
    `[numty-shape]` error and correct shapes stay precise instead of
    dissolving to `Any`.
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
    `check_untyped_defs` is on (see [Get started](../get-started.md#recommended-mypy-settings)).
- An inline call to an overloaded function nested directly inside an
    overloaded numpy call dissolves to `Any` (mypy keeps the
    overload-inferred type over the hook result): hoist it to a local
    first (`kernel = gaussian_kernel(...); np.where(cond, kernel, 0.0)`).

## Symbolic sizes

- A shape error on symbolic dims is about sizes in general, not about
    every size. `signal[:, 1:] - signal` is reported as `time-1` vs
    `time`, and that is wrong code for any `time` above 2, but numpy
    accepts it when `time` is 0 (both sides empty), 1 (the empty side
    broadcasts) or 2 (a length-1 axis broadcasts). The same holds for
    `N+1` against `N`. The oracle counts an error as false only when
    numpy accepts the expression at every set of sizes it tries.
- A symbolic subtraction assumes the axis is long enough. `a[1:]` is
    `(N-1,)` and `np.diff(a)` is `(N-1,)`, but numpy stops at zero: on an
    empty axis both are `(0,)`, not `(-1,)`. The oracle counts a dim
    that comes out negative as a known gap.
- Sizes that differ only by a factor are not reported outside
    `reshape`: `2*M` and `M` are equal at `M == 0`, so `np.vdot` of
    `(2, M)` and `(M,)` is not an error. `reshape` reads totals up to a
    zero factor (`(2*N,)` to `(3, N)` is an error).
- At a call, a formula parameter is solved when the argument gives a
    whole linear answer: `"N+1"` given `M+3` binds `N` to `M+2`, and
    `"2*N"` given `2*M+4` binds it to `M+2`. `"2*N"` given `M` does
    not (M need not be even), and the return dims that need `N` are
    unknown. So is a result dim that would be spelled exactly like one
    of the callee's own formula parameters over the same module-level
    name: `def g(x: F64[N], y: F64["N+1"]) -> F64[N]` called with an
    `N+1`-sized `x` returns an unknown size, not `N+1`.

## Dynamic shapes and dtypes

- Dynamic shapes (`np.zeros((n, 3))`) cannot be proven wrong, so they are
    left alone. Unknown is not an error. A builder (`zeros`/`ones`/`empty`/
    `full`, `*_like(shape=...)`) keeps the literal dims of a tuple
    display next to plain `int` ones: `np.zeros((n, 3))` is
    `tuple[int, Literal[3]]`, also once assigned. A dynamic size in a
    directly returned value (or in an unannotated local it is returned
    through) is `Any` there instead, so it fits a declared named size
    (issues #269/#271); annotated locals, attribute or destructured
    targets, and `yield`ed values keep the precise type and still check. A `Generator` sampler's
    mixed `size=` keeps the stubs' `tuple[Any, ...]`: its answer is final
    (no overload takes the shape from the assignment), so a name first
    bound to `(n, 3, 4)` could not be rebound to `size=(n, d, 4)`. A
    `reshape` from an input with no literal axis (`x.ravel().reshape(n,
    d)`) makes its `int` sizes unknown for the same reason. Most of the
    solver does not read `int` dims. The exceptions: `reshape` keeps the literal sizes
    it is asked for whenever one of them is a plain `int` or the input's
    shape cannot be read (`x.reshape(-1, 3)`, `x.reshape(n, 3)` and
    `x.reshape(x.shape)` are `tuple[int, Literal[3]]`; a `-1` next to an
    `int` size is unknown, so `x.reshape(n, -1)` is `tuple[int, Any]`, and
    so is an `int` target size on an axis where the input had a literal:
    `x.reshape(-1, d)` is `tuple[Any, Any]`); `gradient` along one axis
    keeps the operand's type and still checks the axis;
    `np.dot`/`np.matmul` take an `int` dim as matching and keep the outer
    dims (`(n, 3)` by `(3, 3)` is `tuple[int, Literal[3]]`, while 3 against
    4 is an error as for literal shapes; an `int` dim landing on an axis
    where an operand has a literal turns unknown, so `(n, 3)` by `(3, k)`
    is `tuple[Any, Any]`); transposes move the literal with its axis;
    arithmetic, comparisons, `&`, `|` and `^` between two arrays keep the
    rank, make each `int` axis unknown and still check the named axes
    (`a[:, 1:] | a` with `a: Bool[int, N]` is `N-1 vs N`);
    `np.searchsorted(a, v)` and `a.searchsorted(v)` have the shape of `v`
    as it stands.
    Other ops on such an array (or on one annotated that way) use numpy's
    stubs. Shape-keeping ones (arithmetic, `astype`, `copy`, `flip`, sorts
    along an axis, ...) keep the literal. For the rest the stubs' answer turns
    unknown (`Any`) at its literal dims, because the stubs would leave
    them on the wrong axis after `swapaxes`, `pad` or `diff`, and at its
    `int` dims on axes where an operand had a literal (`np.resize(x,
    x.shape)` is `tuple[int, Any]`, not `tuple[int, int]`). Unknown still
    fits where the literal was expected, so `x = np.diff(x, axis=0)` keeps
    type checking, as does every rebinding in
    `tests/fixtures/mixed_rebind_ok.py`. Later stub ops on such a result
    may answer `ndarray[Any, Any]`, losing the dtype as well. Since a
    transpose moves the literal, so rebinding a name to its own transpose
    (`e = np.zeros((n, 3))` then `e = e.T`) is an `[assignment]` error
    under default mypy, as it already was for all-literal shapes. With
    the recommended `allow_redefinition = True` it checks clean (see
    `tests/fixtures/rebind_allow_ok.py`); otherwise bind the transpose
    to a new name.
    Written inline as an argument whose expected literal differs, a
    builder falls through to numpy's catch-all overload and goes
    unchecked, as before. Assign it to a name first to have it checked.
- An explicit `dtype=` is honored when it resolves (`np.float32`,
    `"float64"`, bare `int`/`float`/`bool`, ...); exotic forms bail out.
    Same for unknown `keepdims`.

## Rank polymorphism (`*Ts`)

- Rank-polymorphic helpers (`(..., 2)` → `...`) spell the shape
    through `FlexArray` with one `*Ts` signature: each signature carries
    two `type: ignore[type-var]` notes (the `tuple[int, ...]` stub bound
    cannot see through an unsolved `*Ts`), as does each builder fed a
    `*Ts` slice (`np.zeros((n, *x.shape[1:]))`), while bodies and call
    sites check fully. The familiar heads need none of these: they mark
    their `*Ts` int-only. mypy itself rejects `shape[1:]` on a bare
    `(*Ts,)` (ambiguous variadic slice); name the leading axis. A provable `*Ts` error may draw the stubs' own echo next
    to ours (an `[index]` complaint on unpack shapes); ours names the
    axis, the size, and the `(*Ts, ...)` shape.
- Rank arithmetic over `*Ts` never reorders the unknown block:
    `transpose`/`.T` bail out (no annotation can spell its reversal),
    explicit transpose axes bail, and a `reshape` `-1` bails because the
    inferred dim would absorb the `*Ts` product. A non-ellipsis index that reaches
    past the known prefix bails — rank-generic bodies spell the ellipsis
    (`a[:, ..., None, :]`); a `1` that aligns against `*Ts` from the
    wrong side bails for the same reason. `matmul` needs one shared
    `Ts` and two known trailing dims per unpack side; vectors over
    `*Ts`, mixed unpacks, and concrete batches against `*Ts` bail.

## Unions and unpacking

- Union-typed array inputs (`(N, K, M) | (N, 1, M)`) distribute per
    member: each member solves with the full solver and the results
    union. Members pair index-wise (exact for co-derived unions);
    unrelated unions of different member counts, and dunder ops with
    unions on both sides (mypy matches those cartesianly), bail out
    instead of guessing.
- Heterogeneous unpacking (`a, b = (f(p) for p in (p1, p2))` with
    different shapes) assigns the whole union to every name — genexp,
    list comprehension, `tuple(...)`, starred targets, and `map` (which
    dissolves to `Any`) all behave this way. The body still checks per
    member; per-name precision needs the tuple display (or
    `python -m numty_codmod --unpack-displays`, which unrolls mapped comprehensions
    over bare names to the display).

## Named sizes and index positions

- Different TypeVars (`N` vs `M`) never conflict by default, even when
    provably different at a call site — only same-name TypeVars are
    unified. The opt-in `numty-names` code reports them inside a generic
    body (see [Error codes](../reference/error-codes.md)).
- Index *positions* are tracked through `Index[P, N]`, which records
    that an int array's values index size P. A recorded size above the
    indexed axis errors as `[numty-axis]` when the overrun is certain
    (`arange(5)` on a size-3 axis, an `argsort` of `N+1` values on an
    axis of `N`). A possible overrun (`searchsorted` positions on `N+1`
    over `N`) and a different named size (positions on M over N) need the
    opt-in `numty-names` code (`enable_error_code = numty-names`) and are
    silent by default. Producers record the fact themselves when the values are
    certainly positions (`argmin`/`argmax` and the `nan` twins,
    `argsort`/`argpartition` on one known axis with a 1-d result,
    0-based `arange`, 1-d `flatnonzero`, per-member `nonzero`/`where`,
    `searchsorted`). Everything else stays silent: multi-dim results,
    records flowing through value-preserving ops (`reshape`, `astype`,
    arithmetic), cross-module uses, ambiguous records (two producers on
    one line with same-shaped results), unions that disagree, and scalar
    positions (a `-1`-for-none sentinel is not tracked).
- A size that is a sum of values is not followed. `np.repeat(x, counts)`
    with an array of counts has `sum(counts)` entries: a plain `int`.
    That the counts sum to a named size is the author's to state, with
    an annotated local or the declared return (`per_entry: F64[N] =
    np.repeat(peak, lengths)`); it is not checked. With it, "reduce per
    run, then spread back per entry" is checked: the line that forgot
    the `np.repeat` assigns a per-run size where `N` is declared
    ([#289](https://github.com/sofeikov/numty/issues/289),
    `tests/fixtures/repeat_counts.py`). A runtime check is
    [#17](https://github.com/sofeikov/numty/issues/17).
- Index positions are best-effort by design. A certain overrun errors
    by default (`arange(5)` on a size-3 axis); a possible one
    (`searchsorted` positions on `N+1` over `N`) and a
    different named size need the opt-in `numty-names` code and are
    silent by default, including at a same-module plain call (an actual
    recorded on N passed where the formal declares `Index` on P,
    `tests/fixtures/bad_index_callsite.py`). Records come from the
    `Index[P, N]` annotation (resolved off the enclosing function, so
    two parameters on one `def` line stay distinct) and from producers
    whose values are certainly positions. Everything else stays silent:
    multi-dim results, records flowing through value-preserving ops
    (`reshape`, `astype`, arithmetic) or a rebinding (`clip`, `- 1`
    clear the record), cross-module or method calls, keyword arguments,
    aliases (`R = Index[P, N]`), ambiguous records (two producers on
    one line with same-shaped results), unions that disagree, and scalar
    positions (a `-1`-for-none sentinel is not tracked)
    ([#285](https://github.com/sofeikov/numty/issues/285),
    [#286](https://github.com/sofeikov/numty/issues/286),
    [#287](https://github.com/sofeikov/numty/issues/287)).

## Annotations and integer bookkeeping

- Bare ints are rejected in alias definitions by mypy itself, before
    any plugin runs (`Mat34 = F64[3, 4]` errors exactly like
    `X = list[3]` does). Spell the sizes with `Literal`
    (`Mat34: TypeAlias = F64[Literal[3], Literal[4]]`, or the PEP 695
    `type` form); uses then check and infer exactly.
- Integer bookkeeping on a *symbolic* dim keeps the dim, like plain
    mypy for generic containers (`n = x.shape[0]` infers `N`, so
    `n //= 2` is an error). Concrete dims widen to `int` (keeping the
    value for direct uses like `reshape(x.shape[0], 3)`), and unknown
    (`?U`) dims are plain `int`. For a loop counter over a symbolic
    axis, write `int(...)` explicitly (`n = int(x.shape[0])`). In array
    arithmetic such a size is a Python `int` scalar: `X @ X.T / X.shape[0]`
    keeps `(N, N)` like `/ float(X.shape[0])` does.
