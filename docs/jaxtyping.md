# Coming from jaxtyping? Convert once, check forever

Already have `Float[np.ndarray, "batch features"]` everywhere? For new
code, prefer the [native spelling](guide/annotations.md) — same feel, no middleman. For
existing files, a plugin cannot read jaxtyping's dim strings — they ride
inside `Annotated` metadata, which no mypy type-analyze hook ever sees
(probed every hook: only jaxtyping's internals fire, the string never
arrives). So the bridge is a one-time source rewrite instead of a hook:
[numty_convert](https://github.com/sofeikov/numty/tree/main/numty_convert/) mechanically
translates the annotations, then gets out of the way:

```sh
python -m numty_convert mlp.py mlp_static.py   # warnings go to stderr, if any
```

```python
# before
def forward(x: Float[np.ndarray, "batch inp"], ...) -> Float[np.ndarray, "batch out"]: ...
# after (import of jaxtyping dropped — nothing references it anymore)
BATCH = TypeVar('BATCH', bound=int)
INP = TypeVar('INP', bound=int)
def forward(x: FlexArray[tuple[BATCH, INP], np.dtype[Any]], ...) -> FlexArray[tuple[BATCH, OUT], np.dtype[Any]]: ...
```

Names become shared TypeVars — same name means same size,
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
[tests/test_convert.py](https://github.com/sofeikov/numty/tree/main/tests/test_convert.py):
goldens for every dim kind plus an end-to-end MLP that checks clean and a
seeded-bug twin that must fail.
