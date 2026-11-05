# numty — static shape checking for NumPy

Wrong-shaped arrays are the `null` of numerical code: everything type-checks,
everything imports, and it blows up three functions later — or worse, silently
computes garbage. [jaxtyping](https://docs.kidger.site/jaxtyping/) solved this
at runtime. numty explores how far you can get *statically*, before anything
runs, using only what type checkers already understand plus one small mypy
plugin.

## What it catches

Mixing up rank:

```python
def use_2x3(m: Matrix2x3) -> None: ...

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
    use_2x3(zeros_3x2())  # error: 3x2 is not 2x3
```

Mismatched matmul, for any rank — with the output shape propagated:

```python
c = matmul(np.zeros((2, 3)), np.zeros((3, 5)))  # c is known to be 2x5
matmul(zeros_2x3(), zeros_2x3())  # error: inner dimensions disagree: 3 vs 2
```

## Quickstart

```sh
uv venv
uv pip install mypy numpy pytest
```

```sh
PYTHONPATH=. uv run mypy demo.py   # expect 2 errors: rank, size
uvx ty check numty.py demo.py      # expect the same 2 errors
uv run pytest                      # full suite: 8 runtime + 5 mypy-driven
```

## How it works

Three layers, each building on the previous one:

| Layer | Idea | Works under |
|---|---|---|
| Rank aliases | `Matrix = ndarray[tuple[int, int], ...]` — tuple length is the rank | mypy, pyright, ty |
| Fixed sizes | `Matrix2x3 = ndarray[tuple[Literal[2], Literal[3]], ...]` | mypy, pyright, ty |
| Shape solver | [numty_plugin.py](/home/sofeikov/my/numty/numty_plugin.py) hooks constructors and `matmul`/`@`, solves shapes itself | mypy only |

The first two layers are pure standard typing — no magic, portable everywhere.
They hit a wall at cross-argument checks: `ndarray`'s shape parameter is
effectively covariant, so a shared `TypeVar` silently joins conflicting
literals (3 vs 2 becomes `int`) instead of erroring. The plugin stops asking
the type system to solve shapes and does the solving itself, with readable
errors like `matmul batch dimensions disagree: (5,) vs (4,)`.

The plugin is registered in [mypy.ini](/home/sofeikov/my/numty/mypy.ini) and
needs `PYTHONPATH=.` so mypy can import it (`MYPYPATH` does not work for
plugins). Without the plugin, checking degrades gracefully to the first two
layers.

## Limitations, stated plainly

- The plugin is mypy-only and pinned to mypy internals that occasionally
  shift. `ty` and pyright users get layers 1–2, which still catch rank and
  fixed-size bugs.
- Without the plugin, `matmul` returns a loose type: usable but unchecked.
- Dynamic shapes (`np.zeros((n, 3))`) cannot be proven wrong, so they are
  left alone. Unknown is not an error.
- No broadcasting of batch dims yet, no `reshape` validation, no elementwise
  ops — see roadmap. Dtype is taken from the left operand.
- `np.matmul` itself is a gufunc object in the stubs, not a plain function,
  so it cannot be hooked; the plugin solves at our own `matmul` wrapper
  instead.

## Layout

- [numty.py](/home/sofeikov/my/numty/numty.py) — aliases, typed constructors,
  narrowing helper, shape-preserving `ident`, loose `matmul` fallback.
- [numty_plugin.py](/home/sofeikov/my/numty/numty_plugin.py) — the solver.
- [demo.py](/home/sofeikov/my/numty/demo.py) — good and bad usages, clean under
  both checkers except the two intentional errors.
- [tests/test_runtime.py](/home/sofeikov/my/numty/tests/test_runtime.py) —
  shapes, values, narrowing accept/reject.
- [tests/test_static.py](/home/sofeikov/my/numty/tests/test_static.py) — runs
  real mypy over [fixtures](/home/sofeikov/my/numty/tests/fixtures/static_ok.py)
  and asserts each bad one fails with exactly one error.

## Roadmap

Broadcast elementwise ops (`+`, `*`) so precision survives whole function
bodies, then `reshape` validation (size conservation, `-1` inference).
