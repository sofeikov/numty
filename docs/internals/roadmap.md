# Roadmap

Frontiers, ordered by value: per-name precision past lazy unpacks
(a tuple display is the only form mypy's binder distributes — genexp,
listcomp, `tuple(...)`, starred, and `map` all assign the union);
unpack-vs-concrete merges (provably unknowable today, but narrower bails
are possible); NEP 50's value-based edge cases (out-of-bound Python ints)
and union dtypes; the `@` operator (mypy consults the method hook
inconsistently); `ty` and pyright (neither offers a plugin API to build
the solver on).
