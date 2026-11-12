# Narrowing and runtime behaviour

## `narrow`: checked at runtime, precise statically

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

## `isinstance` and the heads

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
