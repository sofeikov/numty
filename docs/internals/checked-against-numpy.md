# Checked against numpy

The fixtures pin the cases someone thought of. The differential oracle
covers the ones nobody did: it generates numpy expressions, asks
mypy+numty for the type of each, evaluates the same text with numpy,
and compares. A run fails when

- numty claims a shape or dtype numpy does not produce,
- numpy rejects an expression that numty gave a fully concrete type
  with no diagnostic (concrete mode, `CLAIM_ON_ERROR`),
- numpy rejects a symbolic expression under *every* small size
  assignment (1-3 for each name, including equal names) with no
  diagnostic (issue #183, `MISSED_ERROR`), or
- numty reports a shape error on an expression numpy accepts. For a
  symbolic expression that means accepts at every set of sizes tried:
  an error there is about sizes in general (see [Limitations](limitations.md#symbolic-sizes)).

Losing precision (an unknown dim, `Any`, a bail) is allowed only when
the rejection is unprovable: some size assignment would make the
expression work (`F64[N] + F64[M]` with `N == M`). An assignment-proof
rejection is a miss, not prudence. `SIZE_ASSIGNMENTS` keeps names
distinct (and never 1) by design, so it cannot make this distinction
on its own; the oracle probes 1-3 including equal names separately.
TypeError dtype loops, `cross` with a repeated name, `repeat`/`vdot`/
stacking with cross-position same-name conflicts, and helpers /
formulas / `*Ts` (which need formula/prefix reasoning) stay
unprovable for now as follow-up. Two modes share the machinery. Concrete mode builds
operands with literal shapes across twelve dtypes. Symbolic mode wraps
the expression in a function with `F64[N, M]`-style parameters, calls
it with fixed sizes, and substitutes them into the revealed dims, so
`N+1`, `N*M` and `N//2+1` are checked as arithmetic. A symbolic answer
is a claim about every size, so each case is evaluated under four
assignments from one mypy run: odd sizes, even sizes, and two with an
empty axis. Parameters may carry formulas themselves (`F64[N, "N+1"]`),
so rules also meet inputs that are already expressions. Parameters may
also carry a variadic prefix (`F64[Unpack[Ts], N, M]`), instantiated at
runtime with prefixes of rank 0, 1 and 2; every such case is evaluated
under each prefix, and a revealed `*Ts` is substituted with the prefix
before the shape is compared.

Symbolic mode also calls generic functions it writes itself, such as
`def g(x0: F64[N, "N+1"]) -> F64["2*N", M]`, with arguments in the
caller's own names. There is no numpy answer for those, so the oracle
carries its own reading of a signature: bind the names from the
argument shapes, then build what the return spells. numty's answer at
the call site has to match it. Half of these are declared on a class,
as an instance method, a static method, a class method or `__call__`,
since methods reach the plugin by a different path than functions.
Some helpers carry `*Ts` themselves (e.g. `def g(x0: F64[Unpack[Ts], M])
-> F64[Unpack[Ts], "M+1"]`); the reference binds the shared prefix from
the actual arrays and solves the suffix like before.

The suite runs a small fixed-seed sample of both modes plus every
expression the oracle has caught so far, pinned by hand. For a longer
hunt:

```bash
NUMTY_FUZZ_SEEDS=1-30 NUMTY_FUZZ_COUNT=700 uv run pytest tests/test_differential.py
```

```bash
PYTHONPATH=. python -m tests.differential --seeds 1-30 --count 700
```

Each disagreement prints the expression, numty's type and numpy's
result:

```text
CLAIM_ON_ERROR: np.searchsorted(np.ones((3, 3)), np.ones((6,)))
    numty: numpy.ndarray[tuple[Literal[6]], numpy.dtype[numpy.signedinteger[...]]]
    numpy: raises ValueError: object too deep for desired array
```

What it does not cover: wrong answers that come from numpy's own stubs
on calls numty does not hook are listed in `KNOWN_GAPS` in
`tests/differential.py` rather than failed (two entries today:
`cov`/`corrcoef` and `digitize` on complex input). In symbolic mode,
rejections that survive every small size assignment are split from the
unprovable bucket (`numpy raises, not flagged (unprovable)`) and fail
as `MISSED_ERROR`; the remaining unprovable (e.g. `N` vs `M`, reshape
with `-1` on a size that could divide) stays allowed by design. A symbolic dim that
comes out negative at the sizes used (`N-1` on an empty axis) is
counted as a known gap too; see [Limitations](limitations.md#symbolic-sizes).
