# Tracing, strict mode and internal errors

## Give-up trace and strict mode

When a hook cannot prove a call it keeps mypy's stub type and says
nothing, so a green run does not tell you which shapes were actually
checked. Two opt-in knobs make those give-ups visible:

```bash
rm -rf /tmp/numty-cache
NUMTY_TRACE=/tmp/numty.jsonl mypy --cache-dir /tmp/numty-cache repro.py
cat /tmp/numty.jsonl
```

writes one JSON object per hook call that needed attention:
`{file, line, col, op, outcome, reason}` with `outcome` one of
`solved`, `bail` (a hooked call kept the stubs' type), `unhooked`
(a `numpy.*` call with no hook where a known shape flowed in
unchecked, e.g. `np.kron`) or `crash` (the hook itself failed; see
below). Only modules that are actually re-checked
emit events, so use a fresh `--cache-dir` when collecting a trace.

```bash
NUMTY_STRICT=1 mypy repro.py
```

additionally reports every hooked bail as
`error: numty could not prove <op>: unsolved (result is unchecked)
[numty-unsolved]`. For example, `np.sqrt(8)` bails (its operand is not
an array), its `Any` then poisons `... / np.sqrt(8)` whose hook mypy
skips, so strict flags the `sqrt` birth site itself. Silence one line
with `# type: ignore[numty-unsolved]`.

Strict mode only adds these reports: every inferred type and every
other error is the same as without it. mypy matches overloads, splits
unions and retries operators by checking a call quietly and dropping
the attempt when it raised any error, so the report is recorded where
those checks do not see it (an attempt mypy has already dropped, such
as an overload whose arguments do not fit, reports nothing). A call
checked more than once is reported once.

Not reported, because nothing there has a shape to prove (they still
appear in the trace):

- a plain attribute read (`X.dtype`, `np.finfo(X.dtype).eps`, a
  dataclass field) and the `__next__` of a `for` loop over `range()`;
- a builtin (`len`, `min`, `max`, `divmod`) with no array operand;
- an in-place op (`a += b`, `a /= s`) whose shapes were compared, a
  reported `[numty-broadcast]` mismatch included.

Unhooked numpy functions (e.g. `np.linalg.eigvals`) are trace-only.

`NUMTY_STRICT` is part of mypy's cache key (the plugin reports it as
config data), so switching it on or off re-checks the modules instead
of replaying a cached result.

`python -m numty_report` measures the give-ups instead of erroring on
them: it runs mypy with tracing on a fresh cache and prints per-file
`solved` / `bailed` / `unhooked` / `crashed` counts plus the top bail
reasons (the same idea as mypy's `--any-exprs-report`):

```bash
PYTHONPATH=. python -m numty_report --top 5 repro.py
```

```text
repro.py:4: note: Revealed type is "numpy.ndarray[Any, Any]"
Success: no issues found in 1 source file
numty report: 23 traced call(s) in 1 file(s) (694 stub/site-packages event(s) hidden; --all-files to show)
file             solved  bailed  unhooked  crashed  total
repro.py             11       6         6        0     23
top bail reasons (top 5):
  6x typing.SupportsIndex.__index__: unsolved (default stub type kept)
top unhooked ops (top 5):
  6x numpy.linalg._linalg.eigvals
```

mypy's own output passes through and its exit code is kept, so the
report can replace a bare `mypy` invocation. Third-party stub and
site-packages events are hidden by default (`--all-files` shows
them); `--cache-dir` and `--trace-file` override the fresh temporary
ones. Bails stay silent by default — erroring on them punishes
correct code, which is what strict is for.

## When numty itself fails

A bug inside the plugin never aborts the mypy run, and never passes for
a clean result either. The exception is caught, the call keeps the
stubs' type, and the line gets

```text
error: numty internal error in numpy.ndarray.__add__: RuntimeError: ... (result is unchecked; please report this)  [numty-internal]
```

The same failure is written to `NUMTY_TRACE` with outcome `crash`.
mypy may add a follow-on error of its own on that line (for an
operator, `Unsupported operand types`), because any diagnostic raised
while it matches overloads ends the match. Turn the report off with
`disable_error_code = numty-internal`; the line then goes unchecked
silently, as it did before this code existed. A few internal helpers
have no source position to report against, and their failures are
trace-only.

For a traceback, set `NUMTY_DEBUG=1`: the exception then propagates
and mypy stops with `INTERNAL ERROR` (add `--show-traceback`). The test
suite runs this way (`tests/conftest.py`), so an exception the plugin
would have swallowed fails the tests.

## Wrong-rank arguments: the note under `Never`

An argument of the wrong rank is rejected by mypy itself, with its own
`arg-type` error. The parameter's dim TypeVars cannot be solved from a
shape of another length, so that error prints them as `Never`; numty
adds a note under it with the dims as declared. For `rank.py`:

```python
from typing import TypeVar

from numty import F32

N = TypeVar("N", bound=int)
A = TypeVar("A", bound=int)


def act(action: F32[A]) -> None: ...


def run(batch: F32[N, A]) -> None:
    act(batch)  # forgot the [i]
```

```text
rank.py:13: error: Argument 1 to "act" has incompatible type "ndarray[tuple[N, A], dtype[floating[_32Bit]]]"; expected "ndarray[tuple[Never], dtype[floating[_32Bit]]]"  [arg-type]
rank.py:13: note: "act" declares parameter "action" with shape (A,); the argument has shape (N, A) (rank 2, not 1)
```

mypy's error line still says `Never`; the names are in the note only.
An argument dim numty knows only as a plain `int` (say a size `n: int`)
prints as `int` in the note. The note infers an inline argument such as
`act(np.zeros((n, 3)))` on its own, so its shape can be more precise than
the one in mypy's error line above it.

The note carries the same `arg-type` code, so `# type: ignore[arg-type]`
or `disable_error_code = arg-type` silences both. It is added only to
explicit calls with plain arguments: not to `*args`/`**kwargs` actuals,
and not to operators, indexing or `in`, which mypy reports under
`operator`, `index` or `assignment` instead.
