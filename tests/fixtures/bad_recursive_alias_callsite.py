"""Issue #164: a recursive alias among the formals must not cost the call
its shape checks. Walking `Json` used to recurse until RecursionError,
which the call-site hook swallowed, silently skipping the conflict below.
"""
import numpy as np

from numty import F64, batch

type Json = dict[str, Json] | list[Json] | str | int | None


def fit(x: F64[batch], y: F64[batch], meta: Json) -> F64[batch]:
    return x + y


fit(np.zeros(4), np.zeros(5), {"run": [1, 2]})  # batch: 4 vs 5
