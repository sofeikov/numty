# Unions that check every member

A union-typed array (`(N, K, M) | (N, 1, M)`, or a loop variable over
mixed shapes) solves once per member and unions the results, so bodies
that used to dissolve to `Any` now check each shape:

```python
b = np.where(valid, distance, np.inf)  # union in, union of precise out
clean = [np.where(np.isfinite(v), v, np.nan) for v in (p1, p2)]
```

Members pair index-wise, which is exact for co-derived unions (the same
value mapped through ops — the overwhelmingly common source). Unrelated
unions of different member counts bail out instead of guessing, and
dunder ops with unions on both sides (`u + u`) still go through mypy's
native cartesian matching, so cross terms may dissolve.

## Unpacking

Unpacking is the one place precision does not follow the members: a
generator, list comprehension, `tuple(...)` call, starred target, or
`map` assigns the union to every name (mypy semantics — `map` dissolves
further to `Any`). Only the tuple display unpacks per name:

```python
a, b = (clean(p1), clean(p2))   # (N, 2) and (N, K, 2): precise
a, b = (clean(v) for v in (p1, p2))  # both: the union (body still checked)
```

So when the shapes must reach the code below, spell the display; when
only the per-member checking matters, the generator form is fine.
`python -m numty_codmod --unpack-displays` rewrites the displayable forms for
you, including mapped comprehensions over bare names
(`(f(v) for v in (p1, p2))` -> `(f(p1), f(p2))`).
