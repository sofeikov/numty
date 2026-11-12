# Bugs, caught while you type

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

These are not cherry-picked: [bench/](https://github.com/sofeikov/numty/tree/main/bench) seeds 18
realistic mistakes into MLP-training and math-library code, and the checker
flags 17 of them before anything runs (the 18th is unprovable on purpose — see [limitations](../internals/limitations.md)). Each error names the axes and the shapes, e.g.
`broadcast shapes disagree at trailing axis 1: 10 vs 2 (shapes (N, 10, 2)
vs (N, 2, 1))`.
