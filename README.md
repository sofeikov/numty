# numty micro-MVP (options 1+2)

Static numpy shape checks with plain typing only. No plugins.

```sh
uv run mypy demo.py
```

Expect 4 errors: rank mismatch, size mismatch, stored-constructor
widening, matmul inner-dimension mismatch. `pytest` runs the full suite.
