# Unions that check every member

A union-typed array (`(N, K, M) | (N, 1, M)`, or a loop variable over
mixed shapes) solves once per member and unions the results, so bodies
that used to dissolve to `Any` now check each shape:

```python
b = np.where(valid, distance, np.inf)  # union in, union of precise out
clean = [np.where(np.isfinite(v), v, np.nan) for v in (p1, p2)]
```

Members pair index-wise, which is exact for co-derived unions (the same
value mapped through ops — the overwhelmingly common source):
`np.append(1 - y, y, axis=1)` and `np.concatenate([x, x])` with `y`, `x`
unions pair the first member with the first, never across. Unrelated
unions of different member counts bail out instead of guessing. A
dunder op with one variable on both sides (`u + u`, `self.u * self.u`)
pairs index-wise too; other dunder ops with unions on both sides go
through mypy's native cartesian matching, so cross terms may dissolve.

`np.atleast_1d` / `atleast_2d` / `atleast_3d` lift each member
(`(S, I) | (S,)` is `(S, I) | (1, S)` for `atleast_2d`). `.shape` of a union keeps literal sizes
as literals, so `U.shape[0]` of `(N, N) | (0, 0)` is `N | Literal[0]`
and `np.zeros(U.shape[0])` is `(N,) | (0,)`. A union value written with
`out[i] = v` or `t += v` is checked per member: each array member must
fit the target, unless the target is itself a member of a union.

## Optional and scalar members

A union with `None` or with a Python or NumPy scalar distributes the
same way ([#331](https://github.com/sofeikov/numty/issues/331)):

```python
def loss(y: F64[N, K], sample_weight: F64[N] | None = None) -> F64[K]:
    return np.average(y, weights=sample_weight, axis=0)  # (K,), weights checked

def kernel(x: F64[N, M], length_scale: F64[M] | float) -> F64[N, M]:
    return x / length_scale  # both members give (N, M)

threshold = 0.5 if y.min() >= 0 else 0  # float | int
y > threshold  # (N,) bool
```

Each array member is checked as a plain array would be: `weights=` of
the wrong length errors although the `None` member would not. `None`
distributes where it has a meaning (`np.average(weights=...)` and the
arguments of a user function); in arithmetic and comparisons a scalar
member is an operand like any other number. A union with a scalar
distributes when it is the only union among the operands. So does one
with an `Any` member (`F64[5, 1] | Any`, from `np.dot(B, u)` with an
untyped `u`), on either side of an operator: the array member is
checked, and the `Any` member stays `Any` in the result.

## Calls of your functions

A union actual at a call of a function you wrote is checked per member,
as an operator is: `xl(t, p)` with `p: F64[3, 2] | F64[6, 2]` is the
call-site conflict of the `(6, 2)` member (`[numty-callsite]`, and the
`[numty-names]` report for names), and an `F64[N] | None` actual is
checked with its array member. Several union actuals pair index-wise.
The return is still mypy's single solve, not a union per member.

## Loops

Inside a `for` or `while` body the joins that grow a size
(`np.concatenate`, the stacks, `np.append`) do not distribute: mypy
checks the body a few times and stops, so the accumulator
`out = np.concatenate([out, row])` would end on the members of the last
pass. It ends on the stubs' answer instead.

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
