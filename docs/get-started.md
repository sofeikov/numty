# Get started

numty needs Python 3.13+, NumPy 2.5+ and mypy 2.4.

## Install

```sh
pip install "numty @ git+https://github.com/sofeikov/numty" mypy
```

!!! warning "Do not `pip install numty` from PyPI"

    The name `numty` on PyPI belongs to an unrelated project. Install from
    GitHub as shown above.

numty ships as regular packages (`numty` plus the plugin and three tools,
each with its own PEP 561 `py.typed` marker), so no `PYTHONPATH` or
`mypy_path` is needed.

## Turn the plugin on

Add this to your `mypy.ini`:

```ini
[mypy]
plugins = numty_plugin
allow_redefinition = True
```

Every numty check is on once the plugin is, `numty-names` included: two
different named sizes where one size is needed (`F64[N] + F64[M]` in a
function generic in both) are reported. Add
`disable_error_code = numty-names` to keep such pairs silent (see
[Error codes](reference/error-codes.md#turning-it-off)).

## Check something

Save this as `train.py`. It imports nothing from numty and has no
annotations:

```python title="train.py"
import numpy as np

images = np.zeros((64, 28, 28))
weights = np.zeros((10, 784))  # transposed: should be (784, 10)
bias = np.zeros(10)

flat = images.reshape(64, -1)
logits = flat @ weights + bias
```

```sh
mypy train.py
```

```text
train.py:8: error: matmul inner dimensions disagree: 784 vs 10  [numty-shape]
```

## Annotate a function

Shapes written in a signature are checked through the whole body and at
every call site:

```python
import numpy as np
from numty import F64, batch, hid, inp

def forward(
    x: F64[batch, inp], w: F64[inp, hid], b: F64[hid],
) -> F64[batch, hid]:
    return x @ w + b

forward(np.zeros((32, 8)), np.zeros((8, 16)), np.zeros(4))
```

```text
error: call-site shape conflict for 'hid': 16 (argument 'w', axis 1) vs 4 (argument 'b', axis 0)  [numty-callsite]
```

Continue with [Annotating shapes](guide/annotations.md).

## Recommended mypy settings

**`allow_redefinition = True`.** Ordinary NumPy code rebinds names to new
shapes (`x = x.sum(axis=1)`, `a = a.T`). Under default mypy each is an
`[assignment]` error. With the flag on, the solver stays precise through
the rebinding.

**`check_untyped_defs = True`.** mypy skips the bodies of unannotated
functions by default, so a script or notebook cell written as
`def main():` gets no shape checking at all. Turn this on (or pass
`--check-untyped-defs`) to have them checked too.

**Follow imports (the default).** Under `--follow-imports=skip` mypy skips
`numty` itself and every `F64[...]` annotation silently becomes `Any`. See
[Limitations](internals/limitations.md#mypy-behaviour) for the per-module
override.

## Using ruff

With ruff's `UP` rules on, two spellings are rewritten:

- `UP044` turns `F32[Unpack[Ts]]` into `F32[*Ts]`, which numty reads as the
  same type.
- `UP037` strips the quotes of a formula (`U8["N+1"]` becomes `U8[N+1]`) in
  a file with `from __future__ import annotations`. mypy does not pass an
  unquoted `N+1` on to a plugin, so the result is a `[numty-annotation]`
  error.

For the second, either write the formula as `U8[Literal["N+1"]]`, which
the rule leaves alone, or turn the rule off:

```toml
[tool.ruff.lint]
ignore = ["UP037"]
```

## Supported versions and startup warnings

numty hooks key on private NumPy module paths (for example
`numpy._core.einsumfunc.einsum`) and on mypy internals, so a version drift
could silently unhook everything. Two gates warn loudly instead:

- The plugin compares the running mypy against the tested range (2.4.x)
  and prints one `numty: warning:` to stderr outside it.
- On first use, the installed NumPy is compared against the minimum
  supported (2.5), and a canary resolves a short list of hooked names in
  the loaded stubs, warning once and naming any that is missing.

CI runs the suite against both the oldest supported NumPy and the latest
release, so a stub rename that unhooks everything fails there first.

## Other type checkers

The shape solver is a mypy plugin. pyright and ty have no plugin API, so
they get rank and fixed-size checks through plain `ndarray` types but not
the `F64`-style heads. See [How it works](internals/how-it-works.md) and
[Limitations](internals/limitations.md#other-type-checkers).
