"""Call-site conflicts fire inside comprehensions/lambdas (issue #44)."""
import numpy as np
from numty import F64, batch, features, hid


def proj(x: F64[batch, features], w: F64[features, hid]) -> F64[batch, hid]:
    return np.matmul(x, w)


x = np.zeros((4, 3))
w = np.ones((2, 5))
a = proj(x, w)
b = [proj(x, w) for _ in range(2)]
d = {i: proj(x, w) for i in range(2)}
e = list(proj(x, w) for _ in range(2))
f = (lambda: proj(x, w))()
g = np.sum(proj(x, w))
h = proj(x, w) if x.size else None
